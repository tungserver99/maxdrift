#!/usr/bin/env python3
"""MaxDrift-GQ optimizer and GuidedQuant LNQ-cache entry point.

This file deliberately reuses GuidedQuant's LNQ cache format:

    <gq_lnq_checkpoint>/
      weights/l{layer}.pt              # module -> uint8 codes [out, 1, in]
      lut_<bits>/l{layer}.pt           # module -> fp16 codebooks [out, 1, 2^bits]

and GuidedQuant's Hessian cache:

    <gq_cache_dir>/hessians/.../l{layer}.pt
      module -> Hessian [num_groups, in, in] or [in, in, num_groups]

The optimizer freezes the LNQ codebooks and changes only integer codes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@dataclass
class SweepLog:
    sweep: int
    num_code_changes: int
    p_start: float
    p_end: float
    p_budget: float
    d_start: float
    d_end: float
    relative_gain: float
    runtime_sec: float


@dataclass
class ChannelResult:
    q: torch.Tensor
    codes: torch.Tensor
    p_start: float
    p_final: float
    epsilon: float
    d_start: float
    d_final: float
    tau_feas: float
    num_sweeps: int
    num_changed_codes: int
    sweep_logs: List[SweepLog] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "p_start": self.p_start,
            "p_final": self.p_final,
            "epsilon": self.epsilon,
            "d_start": self.d_start,
            "d_final": self.d_final,
            "tau_feas": self.tau_feas,
            "num_sweeps": self.num_sweeps,
            "num_changed_codes": self.num_changed_codes,
        }


def compute_objectives(e: torch.Tensor, hessian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return preservation cost e^T H e and drift e^T e."""
    r = hessian @ e
    return torch.dot(e, r), torch.dot(e, e)


def reconstruct_from_codes(codes: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Reconstruct a vector from integer codes and a one-dimensional codebook."""
    return codebook.to(codes.device)[codes.long()]


def recover_codes_from_codebook(q: torch.Tensor, codebook: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """Recover nearest code indices and require reconstruction within tolerance."""
    q_work = q.detach().to(torch.float64)
    cb = codebook.detach().to(torch.float64)
    distances = torch.abs(q_work.unsqueeze(-1) - cb.reshape(1, -1))
    codes = torch.argmin(distances, dim=-1).long()
    reconstructed = reconstruct_from_codes(codes, cb)
    max_error = torch.max(torch.abs(reconstructed - q_work)).item() if q_work.numel() else 0.0
    if max_error > atol:
        raise ValueError(f"Codebook cannot reconstruct q within tolerance: max_error={max_error:.3e}, atol={atol:.3e}")
    return codes


def fix_hessian_shape(hessian: torch.Tensor) -> torch.Tensor:
    """Match GuidedQuant's helper: return [num_groups, input_dim, input_dim]."""
    if hessian.ndim != 3:
        raise ValueError(f"Expected 3-D Hessian, got shape {tuple(hessian.shape)}")
    if hessian.shape[1] == hessian.shape[2]:
        return hessian
    if hessian.shape[0] == hessian.shape[1]:
        return hessian.permute(2, 0, 1)
    raise ValueError(f"Invalid Hessian shape: {tuple(hessian.shape)}")


def maxdrift_channel(
    w: torch.Tensor,
    q0: torch.Tensor,
    codes0: torch.Tensor,
    codebook: torch.Tensor,
    hessian: torch.Tensor,
    rho: float,
    max_sweeps: int = 10,
    tau_gain: float = 1e-12,
    tau_feas_rel: float = 1e-8,
    relative_gain_tol: float = 1e-6,
    numeric_floor: float = 1e-12,
    dtype: torch.dtype = torch.float64,
) -> ChannelResult:
    """Exact constrained cyclic coordinate ascent for one output channel."""
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if w.shape != q0.shape or w.ndim != 1:
        raise ValueError(f"w and q0 must be 1-D with same shape, got {tuple(w.shape)} and {tuple(q0.shape)}")
    if codes0.shape != w.shape:
        raise ValueError(f"codes0 shape {tuple(codes0.shape)} does not match weight shape {tuple(w.shape)}")
    if hessian.shape != (w.numel(), w.numel()):
        raise ValueError(f"hessian shape {tuple(hessian.shape)} does not match input dimension {w.numel()}")

    device = w.device
    w = w.detach().to(device=device, dtype=dtype).clone()
    q = q0.detach().to(device=device, dtype=dtype).clone()
    codebook = codebook.detach().to(device=device, dtype=dtype).reshape(-1)
    hessian = hessian.detach().to(device=device, dtype=dtype)
    codes = codes0.detach().to(device=device, dtype=torch.long).clone()

    if torch.any(codes < 0) or torch.any(codes >= codebook.numel()):
        raise ValueError("codes0 contains indices outside the codebook")

    reconstructed = reconstruct_from_codes(codes, codebook)
    if not torch.allclose(reconstructed, q, rtol=0.0, atol=1e-5):
        max_err = torch.max(torch.abs(reconstructed - q)).item()
        raise ValueError(f"codes0/codebook do not reproduce q0; max_error={max_err:.3e}")

    e = q - w
    r = hessian @ e
    p_tensor = torch.dot(e, r)
    d_tensor = torch.dot(e, e)
    p = float(p_tensor.item())
    d = float(d_tensor.item())
    p_start, d_start = p, d
    epsilon = max((1.0 + rho) * p_start, p_start + numeric_floor)
    tau_feas = max(1e-10, tau_feas_rel * max(abs(epsilon), 1.0))

    if p_start > epsilon + tau_feas:
        raise ValueError(f"Initial channel is infeasible: P0={p_start:.8e}, epsilon={epsilon:.8e}")

    sweep_logs: List[SweepLog] = []
    total_changed = 0

    for sweep in range(max_sweeps):
        sweep_start_time = time.time()
        changed = 0
        p_before = p
        d_before = d

        for i in range(w.numel()):
            current_code = int(codes[i].item())
            current_value = q[i]
            ei = e[i]
            ri = r[i]
            hii = hessian[i, i]

            best_code = current_code
            best_delta_d = torch.zeros((), dtype=dtype, device=device)
            best_delta_p = torch.zeros((), dtype=dtype, device=device)

            for k in range(codebook.numel()):
                candidate = codebook[k]
                delta = candidate - current_value
                delta_p = 2.0 * delta * ri + delta * delta * hii
                delta_d = 2.0 * delta * ei + delta * delta

                feasible = (p + float(delta_p.item())) <= (epsilon + tau_feas)
                if not feasible:
                    continue

                delta_d_float = float(delta_d.item())
                best_delta_d_float = float(best_delta_d.item())
                if delta_d_float > best_delta_d_float + tau_gain:
                    best_code = k
                    best_delta_d = delta_d
                    best_delta_p = delta_p
                elif abs(delta_d_float - best_delta_d_float) <= tau_gain:
                    if k == current_code:
                        best_code = current_code
                        best_delta_d = torch.zeros((), dtype=dtype, device=device)
                        best_delta_p = torch.zeros((), dtype=dtype, device=device)
                    elif best_code != current_code:
                        best_delta_p_float = float(best_delta_p.item())
                        delta_p_float = float(delta_p.item())
                        if delta_p_float < best_delta_p_float - tau_gain:
                            best_code = k
                            best_delta_d = delta_d
                            best_delta_p = delta_p

            if best_code == current_code or float(best_delta_d.item()) <= tau_gain:
                continue

            new_value = codebook[best_code]
            delta = new_value - current_value
            h_col = hessian[:, i]

            q[i] = new_value
            codes[i] = best_code
            e[i] = e[i] + delta
            r = r + delta * h_col
            p += float(best_delta_p.item())
            d += float(best_delta_d.item())
            changed += 1

            if p > epsilon + tau_feas:
                raise AssertionError(f"Accepted update violated budget: P={p:.8e}, epsilon={epsilon:.8e}")

        e_exact = q - w
        r_exact = hessian @ e_exact
        p_exact_tensor = torch.dot(e_exact, r_exact)
        d_exact_tensor = torch.dot(e_exact, e_exact)
        p_exact = float(p_exact_tensor.item())
        d_exact = float(d_exact_tensor.item())
        if p_exact > epsilon + tau_feas:
            raise AssertionError(f"Sweep {sweep} exact P violates budget: P={p_exact:.8e}, epsilon={epsilon:.8e}")
        if d_exact < d_before - tau_gain:
            raise AssertionError(f"Sweep {sweep} decreased drift: before={d_before:.8e}, after={d_exact:.8e}")

        p, d = p_exact, d_exact
        e, r = e_exact, r_exact
        total_changed += changed
        relative_gain = (d - d_before) / max(abs(d_before), 1e-12)

        sweep_logs.append(
            SweepLog(
                sweep=sweep,
                num_code_changes=changed,
                p_start=p_before,
                p_end=p,
                p_budget=epsilon,
                d_start=d_before,
                d_end=d,
                relative_gain=relative_gain,
                runtime_sec=time.time() - sweep_start_time,
            )
        )

        if changed == 0 or relative_gain < relative_gain_tol:
            break

    return ChannelResult(
        q=q.detach().cpu(),
        codes=codes.detach().cpu(),
        p_start=p_start,
        p_final=p,
        epsilon=epsilon,
        d_start=d_start,
        d_final=d,
        tau_feas=tau_feas,
        num_sweeps=len(sweep_logs),
        num_changed_codes=total_changed,
        sweep_logs=sweep_logs,
    )


@dataclass
class LayerSummary:
    layer_idx: int
    module_name: str
    bits: int
    rho: float
    num_channels: int
    num_codes: int
    num_sweeps_max: int
    num_changed_codes: int
    fraction_changed_codes: float
    preservation_cost_start: float
    preservation_cost_final: float
    preservation_budget_total: float
    weight_drift_start: float
    weight_drift_final: float
    normalized_weight_drift_start: float
    normalized_weight_drift_final: float
    original_weight_norm_sq: float
    max_budget_ratio: float
    mean_budget_ratio: float
    runtime_sec: float


def _to_numpy_dict(tensor_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in tensor_dict.items()}


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")
    shutil.copytree(src, dst)


def _group_index(row_idx: int, output_dim: int, num_groups: int) -> int:
    if output_dim % num_groups != 0:
        raise ValueError(f"output_dim={output_dim} is not divisible by num_groups={num_groups}")
    return row_idx // (output_dim // num_groups)


def maxdrift_module(
    weight: torch.Tensor,
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    hessians: torch.Tensor,
    bits: int,
    rho: float,
    max_sweeps: int,
    tau_gain: float,
    tau_feas_rel: float,
    relative_gain_tol: float,
    numeric_floor: float,
    dtype: torch.dtype,
    objective_device: torch.device,
) -> Tuple[torch.Tensor, LayerSummary, List[Dict[str, Any]]]:
    """Run MaxDrift for one GuidedQuant module tensor."""
    if codes.ndim != 3 or codebooks.ndim != 3:
        raise ValueError(f"Expected codes/codebooks [out, group_count, *], got {tuple(codes.shape)} and {tuple(codebooks.shape)}")
    if codes.shape[1] != 1 or codebooks.shape[1] != 1:
        raise NotImplementedError("GuidedQuant pack currently assumes group_count == 1; MaxDrift follows that format")
    if codebooks.shape[-1] != 2 ** bits:
        raise ValueError(f"Codebook size {codebooks.shape[-1]} does not match bits={bits}")
    if weight.shape != (codes.shape[0], codes.shape[2]):
        raise ValueError(f"Weight shape {tuple(weight.shape)} does not match code shape {tuple(codes.shape)}")

    hessians = fix_hessian_shape(hessians)
    output_dim, input_dim = weight.shape
    if hessians.shape[1:] != (input_dim, input_dim):
        raise ValueError(f"Hessian shape {tuple(hessians.shape)} does not match input_dim={input_dim}")

    start = time.time()
    new_codes = codes.clone().long()
    channel_logs: List[Dict[str, Any]] = []
    p_start_total = 0.0
    p_final_total = 0.0
    budget_total = 0.0
    d_start_total = 0.0
    d_final_total = 0.0
    changed_total = 0
    max_sweeps_seen = 0
    budget_ratios: List[float] = []

    if objective_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA MaxDrift objective device, but CUDA is not available")

    weight_f = weight.detach().to(device=objective_device, dtype=dtype)
    codes_f = codes.detach().to(device=objective_device, dtype=torch.long).squeeze(1)
    codebooks_f = codebooks.detach().to(device=objective_device, dtype=dtype).squeeze(1)
    hessians_f = hessians.detach().to(device=objective_device, dtype=dtype)

    if output_dim % hessians_f.shape[0] != 0:
        raise ValueError(f"output_dim={output_dim} is not divisible by num_groups={hessians_f.shape[0]}")
    group_size = output_dim // hessians_f.shape[0]

    result = maxdrift_groups(
        w=weight_f,
        codes0=codes_f,
        codebooks=codebooks_f,
        hessians=hessians_f,
        rho=rho,
        max_sweeps=max_sweeps,
        tau_gain=tau_gain,
        tau_feas_rel=tau_feas_rel,
        relative_gain_tol=relative_gain_tol,
        numeric_floor=numeric_floor,
    )
    new_codes[:, 0] = result["codes"].cpu()
    changed_total += int(result["num_changed_codes"])
    max_sweeps_seen = max(max_sweeps_seen, int(result["num_sweeps"]))
    p_start_total += float(result["p_start_sum"])
    p_final_total += float(result["p_final_sum"])
    budget_total += float(result["epsilon_sum"])
    d_start_total += float(result["d_start_sum"])
    d_final_total += float(result["d_final_sum"])
    budget_ratios.extend(result["budget_ratios"])
    channel_logs.extend(result["sweep_logs"])

    weight_norm = float(torch.linalg.vector_norm(weight_f).item())
    norm_d_start = math.sqrt(max(d_start_total, 0.0)) / max(weight_norm, 1e-12)
    norm_d_final = math.sqrt(max(d_final_total, 0.0)) / max(weight_norm, 1e-12)
    total_codes = int(codes.numel())

    summary = LayerSummary(
        layer_idx=-1,
        module_name="",
        bits=bits,
        rho=rho,
        num_channels=output_dim,
        num_codes=total_codes,
        num_sweeps_max=max_sweeps_seen,
        num_changed_codes=changed_total,
        fraction_changed_codes=changed_total / max(total_codes, 1),
        preservation_cost_start=p_start_total,
        preservation_cost_final=p_final_total,
        preservation_budget_total=budget_total,
        weight_drift_start=d_start_total,
        weight_drift_final=d_final_total,
        normalized_weight_drift_start=norm_d_start,
        normalized_weight_drift_final=norm_d_final,
        original_weight_norm_sq=weight_norm * weight_norm,
        max_budget_ratio=max(budget_ratios) if budget_ratios else 0.0,
        mean_budget_ratio=sum(budget_ratios) / len(budget_ratios) if budget_ratios else 0.0,
        runtime_sec=time.time() - start,
    )
    return new_codes.to(torch.uint8), summary, channel_logs


def maxdrift_group(
    w: torch.Tensor,
    codes0: torch.Tensor,
    codebooks: torch.Tensor,
    hessian: torch.Tensor,
    rho: float,
    max_sweeps: int,
    tau_gain: float,
    tau_feas_rel: float,
    relative_gain_tol: float,
    numeric_floor: float,
) -> Dict[str, Any]:
    """Vectorized MaxDrift over output channels that share one Hessian."""
    if w.ndim != 2 or codes0.shape != w.shape:
        raise ValueError(f"Expected w/codes [channels, input_dim], got {tuple(w.shape)} and {tuple(codes0.shape)}")
    if codebooks.ndim != 2 or codebooks.shape[0] != w.shape[0]:
        raise ValueError(f"Expected codebooks [channels, codebook_size], got {tuple(codebooks.shape)}")
    if hessian.shape != (w.shape[1], w.shape[1]):
        raise ValueError(f"Hessian shape {tuple(hessian.shape)} does not match input_dim={w.shape[1]}")

    device = w.device
    channels, input_dim = w.shape
    channel_idx = torch.arange(channels, device=device)
    codes = codes0.clone().long()
    q = codebooks[channel_idx[:, None], codes]
    e = q - w
    r = e @ hessian.T
    p = torch.sum(e * r, dim=1)
    d = torch.sum(e * e, dim=1)
    p_start = p.clone()
    d_start = d.clone()
    epsilon = torch.maximum((1.0 + rho) * p_start, p_start + numeric_floor)
    tau_feas = torch.maximum(
        torch.full_like(epsilon, 1e-10),
        tau_feas_rel * torch.maximum(torch.abs(epsilon), torch.ones_like(epsilon)),
    )
    if w.dtype == torch.float32:
        dtype_slack = 1e-5 * torch.maximum(torch.abs(epsilon), torch.ones_like(epsilon))
    else:
        dtype_slack = 32.0 * torch.finfo(w.dtype).eps * torch.maximum(torch.abs(epsilon), torch.ones_like(epsilon))
    exact_tau_feas = torch.maximum(tau_feas, dtype_slack)
    if torch.any(p_start > epsilon + tau_feas):
        raise ValueError("Initial group contains infeasible channels")

    total_changed = 0
    sweep_logs: List[Dict[str, Any]] = []
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=w.dtype)

    for sweep in range(max_sweeps):
        sweep_start = time.time()
        p_before = p.clone()
        d_before = d.clone()
        changed_this_sweep = 0

        for i in range(input_dim):
            current_codes = codes[:, i]
            current_values = q[:, i]
            delta = codebooks - current_values[:, None]
            delta_p = 2.0 * delta * r[:, i : i + 1] + delta * delta * hessian[i, i]
            delta_d = 2.0 * delta * e[:, i : i + 1] + delta * delta
            feasible = p[:, None] + delta_p <= epsilon[:, None] + tau_feas[:, None]
            masked_delta_d = torch.where(feasible, delta_d, neg_inf)
            best_delta_d, best_codes = torch.max(masked_delta_d, dim=1)
            accept = (best_delta_d > tau_gain) & (best_codes != current_codes)
            if not torch.any(accept):
                continue

            selected_delta = codebooks[channel_idx, best_codes] - current_values
            selected_delta = torch.where(accept, selected_delta, torch.zeros_like(selected_delta))
            selected_delta_p = delta_p[channel_idx, best_codes]
            selected_delta_d = delta_d[channel_idx, best_codes]

            codes[accept, i] = best_codes[accept]
            q[:, i] = q[:, i] + selected_delta
            e[:, i] = e[:, i] + selected_delta
            p = p + torch.where(accept, selected_delta_p, torch.zeros_like(p))
            d = d + torch.where(accept, selected_delta_d, torch.zeros_like(d))
            r = r + selected_delta[:, None] * hessian[:, i][None, :]
            changed_this_sweep += int(torch.count_nonzero(accept).item())

        r_exact = e @ hessian.T
        p_exact = torch.sum(e * r_exact, dim=1)
        d_exact = torch.sum(e * e, dim=1)
        if torch.any(p_exact > epsilon + exact_tau_feas):
            worst = torch.max(p_exact - epsilon - exact_tau_feas).item()
            raise AssertionError(f"Sweep {sweep} exact P violates budget; worst_margin={worst:.8e}")
        if torch.any(d_exact < d_before - tau_gain):
            raise AssertionError(f"Sweep {sweep} decreased drift for at least one channel")

        p = p_exact
        d = d_exact
        r = r_exact
        total_changed += changed_this_sweep
        relative_gain = torch.sum(d - d_before) / torch.clamp(torch.sum(torch.abs(d_before)), min=1e-12)
        sweep_logs.append(
            {
                "sweep": sweep,
                "num_code_changes": changed_this_sweep,
                "p_start": float(torch.sum(p_before).item()),
                "p_end": float(torch.sum(p).item()),
                "p_budget": float(torch.sum(epsilon).item()),
                "d_start": float(torch.sum(d_before).item()),
                "d_end": float(torch.sum(d).item()),
                "relative_gain": float(relative_gain.item()),
                "runtime_sec": time.time() - sweep_start,
            }
        )
        if changed_this_sweep == 0 or float(relative_gain.item()) < relative_gain_tol:
            break

    budget_ratios = (p / torch.clamp(epsilon, min=1e-30)).detach().cpu().tolist()
    return {
        "codes": codes.detach(),
        "p_start_sum": float(torch.sum(p_start).item()),
        "p_final_sum": float(torch.sum(p).item()),
        "epsilon_sum": float(torch.sum(epsilon).item()),
        "d_start_sum": float(torch.sum(d_start).item()),
        "d_final_sum": float(torch.sum(d).item()),
        "num_sweeps": len(sweep_logs),
        "num_changed_codes": total_changed,
        "budget_ratios": budget_ratios,
        "sweep_logs": sweep_logs,
    }


def maxdrift_groups(
    w: torch.Tensor,
    codes0: torch.Tensor,
    codebooks: torch.Tensor,
    hessians: torch.Tensor,
    rho: float,
    max_sweeps: int,
    tau_gain: float,
    tau_feas_rel: float,
    relative_gain_tol: float,
    numeric_floor: float,
) -> Dict[str, Any]:
    """Vectorized MaxDrift over all Hessian groups in one module."""
    if w.ndim != 2 or codes0.shape != w.shape:
        raise ValueError(f"Expected w/codes [out, input_dim], got {tuple(w.shape)} and {tuple(codes0.shape)}")
    if codebooks.ndim != 2 or codebooks.shape[0] != w.shape[0]:
        raise ValueError(f"Expected codebooks [out, codebook_size], got {tuple(codebooks.shape)}")
    if hessians.ndim != 3 or hessians.shape[1:] != (w.shape[1], w.shape[1]):
        raise ValueError(f"Hessian shape {tuple(hessians.shape)} does not match input_dim={w.shape[1]}")
    if w.shape[0] % hessians.shape[0] != 0:
        raise ValueError(f"output_dim={w.shape[0]} is not divisible by num_groups={hessians.shape[0]}")

    device = w.device
    num_groups, input_dim, _ = hessians.shape
    group_size = w.shape[0] // num_groups
    codebook_size = codebooks.shape[1]

    w_g = w.reshape(num_groups, group_size, input_dim)
    codes = codes0.reshape(num_groups, group_size, input_dim).clone().long()
    codebooks_g = codebooks.reshape(num_groups, group_size, codebook_size)

    q = torch.gather(codebooks_g, dim=2, index=codes)
    sorted_codebooks, sorted_to_orig_codes = torch.sort(codebooks_g, dim=2)
    e = q - w_g
    r = torch.bmm(e, hessians.transpose(1, 2))
    p = torch.sum(e * r, dim=2)
    d = torch.sum(e * e, dim=2)
    p_start = p.clone()
    d_start = d.clone()
    epsilon = torch.maximum((1.0 + rho) * p_start, p_start + numeric_floor)
    tau_feas = torch.maximum(
        torch.full_like(epsilon, 1e-10),
        tau_feas_rel * torch.maximum(torch.abs(epsilon), torch.ones_like(epsilon)),
    )
    if w.dtype == torch.float32:
        dtype_slack = 1e-5 * torch.maximum(torch.abs(epsilon), torch.ones_like(epsilon))
    else:
        dtype_slack = 32.0 * torch.finfo(w.dtype).eps * torch.maximum(torch.abs(epsilon), torch.ones_like(epsilon))
    exact_tau_feas = torch.maximum(tau_feas, dtype_slack)
    if torch.any(p_start > epsilon + exact_tau_feas):
        raise ValueError("Initial module contains infeasible channels")

    total_changed = 0
    sweep_logs: List[Dict[str, Any]] = []
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=w.dtype)

    for sweep in range(max_sweeps):
        sweep_start = time.time()
        p_before = p.clone()
        d_before = d.clone()
        changed_this_sweep = 0

        for i in range(input_dim):
            current_codes = codes[:, :, i]
            current_values = q[:, :, i]
            hii = hessians[:, i, i].reshape(num_groups, 1)
            ri = r[:, :, i]
            budget = epsilon + tau_feas - p

            if torch.any(hii <= torch.finfo(w.dtype).eps):
                delta = codebooks_g - current_values[:, :, None]
                delta_p_all = 2.0 * delta * ri[:, :, None] + delta * delta * hii[:, :, None]
                delta_d_all = 2.0 * delta * e[:, :, i : i + 1] + delta * delta
                feasible = p[:, :, None] + delta_p_all <= epsilon[:, :, None] + tau_feas[:, :, None]
                best_delta_d, best_codes = torch.max(torch.where(feasible, delta_d_all, neg_inf), dim=2)
            else:
                discriminant = torch.clamp(ri * ri + hii * torch.clamp(budget, min=0.0), min=0.0)
                sqrt_discriminant = torch.sqrt(discriminant)
                lower_value = current_values + (-ri - sqrt_discriminant) / hii
                upper_value = current_values + (-ri + sqrt_discriminant) / hii

                low_pos = torch.searchsorted(sorted_codebooks, lower_value.unsqueeze(-1), right=False).squeeze(-1)
                high_pos = torch.searchsorted(sorted_codebooks, upper_value.unsqueeze(-1), right=True).squeeze(-1) - 1
                interval_valid = low_pos <= high_pos
                low_pos = torch.clamp(low_pos, min=0, max=codebook_size - 1)
                high_pos = torch.clamp(high_pos, min=0, max=codebook_size - 1)

                low_value = torch.gather(sorted_codebooks, dim=2, index=low_pos.unsqueeze(-1)).squeeze(-1)
                high_value = torch.gather(sorted_codebooks, dim=2, index=high_pos.unsqueeze(-1)).squeeze(-1)
                low_code = torch.gather(sorted_to_orig_codes, dim=2, index=low_pos.unsqueeze(-1)).squeeze(-1)
                high_code = torch.gather(sorted_to_orig_codes, dim=2, index=high_pos.unsqueeze(-1)).squeeze(-1)

                low_delta = low_value - current_values
                high_delta = high_value - current_values
                low_delta_p = 2.0 * low_delta * ri + low_delta * low_delta * hii
                high_delta_p = 2.0 * high_delta * ri + high_delta * high_delta * hii
                low_delta_d = 2.0 * low_delta * e[:, :, i] + low_delta * low_delta
                high_delta_d = 2.0 * high_delta * e[:, :, i] + high_delta * high_delta
                low_feasible = interval_valid & (p + low_delta_p <= epsilon + tau_feas)
                high_feasible = interval_valid & (p + high_delta_p <= epsilon + tau_feas)
                low_score = torch.where(low_feasible, low_delta_d, neg_inf)
                high_score = torch.where(high_feasible, high_delta_d, neg_inf)

                choose_low = (low_score > high_score + tau_gain) | (
                    torch.abs(low_score - high_score) <= tau_gain
                ) & (low_code <= high_code)
                best_delta_d = torch.where(choose_low, low_score, high_score)
                best_codes = torch.where(choose_low, low_code, high_code)
                delta_p_all = None
                delta_d_all = None

            accept = (best_delta_d > tau_gain) & (best_codes != current_codes)
            if not torch.any(accept):
                continue

            selected_delta = torch.gather(codebooks_g, dim=2, index=best_codes.unsqueeze(-1)).squeeze(-1) - current_values
            selected_delta = torch.where(accept, selected_delta, torch.zeros_like(selected_delta))
            if "delta_p_all" in locals() and delta_p_all is not None:
                selected_delta_p = torch.gather(delta_p_all, dim=2, index=best_codes.unsqueeze(-1)).squeeze(-1)
                selected_delta_d = torch.gather(delta_d_all, dim=2, index=best_codes.unsqueeze(-1)).squeeze(-1)
            else:
                selected_delta_p = 2.0 * selected_delta * ri + selected_delta * selected_delta * hii
                selected_delta_d = 2.0 * selected_delta * e[:, :, i] + selected_delta * selected_delta

            codes[:, :, i] = torch.where(accept, best_codes, codes[:, :, i])
            q[:, :, i] = q[:, :, i] + selected_delta
            e[:, :, i] = e[:, :, i] + selected_delta
            p = p + torch.where(accept, selected_delta_p, torch.zeros_like(p))
            d = d + torch.where(accept, selected_delta_d, torch.zeros_like(d))
            r = r + selected_delta[:, :, None] * hessians[:, None, :, i]
            changed_this_sweep += int(torch.count_nonzero(accept).item())

        r_exact = torch.bmm(e, hessians.transpose(1, 2))
        p_exact = torch.sum(e * r_exact, dim=2)
        d_exact = torch.sum(e * e, dim=2)
        if torch.any(p_exact > epsilon + exact_tau_feas):
            worst = torch.max(p_exact - epsilon - exact_tau_feas).item()
            raise AssertionError(f"Sweep {sweep} exact P violates budget; worst_margin={worst:.8e}")
        if torch.any(d_exact < d_before - tau_gain):
            raise AssertionError(f"Sweep {sweep} decreased drift for at least one channel")

        p = p_exact
        d = d_exact
        r = r_exact
        total_changed += changed_this_sweep
        relative_gain = torch.sum(d - d_before) / torch.clamp(torch.sum(torch.abs(d_before)), min=1e-12)
        sweep_logs.append(
            {
                "sweep": sweep,
                "num_code_changes": changed_this_sweep,
                "p_start": float(torch.sum(p_before).item()),
                "p_end": float(torch.sum(p).item()),
                "p_budget": float(torch.sum(epsilon).item()),
                "d_start": float(torch.sum(d_before).item()),
                "d_end": float(torch.sum(d).item()),
                "relative_gain": float(relative_gain.item()),
                "runtime_sec": time.time() - sweep_start,
            }
        )
        if changed_this_sweep == 0 or float(relative_gain.item()) < relative_gain_tol:
            break

    return {
        "codes": codes.reshape_as(codes0).detach(),
        "p_start_sum": float(torch.sum(p_start).item()),
        "p_final_sum": float(torch.sum(p).item()),
        "epsilon_sum": float(torch.sum(epsilon).item()),
        "d_start_sum": float(torch.sum(d_start).item()),
        "d_final_sum": float(torch.sum(d).item()),
        "num_sweeps": len(sweep_logs),
        "num_changed_codes": total_changed,
        "budget_ratios": (p / torch.clamp(epsilon, min=1e-30)).detach().cpu().flatten().tolist(),
        "sweep_logs": sweep_logs,
    }


def _load_guidedquant_analyzer(guidedquant_dir: Path, model: str, yaml_path: Optional[str]):
    sys.path.insert(0, str(guidedquant_dir.resolve()))
    from any_precision.analyzer import get_analyzer

    return get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)


def run_maxdrift_cache(
    fp_model: str,
    gq_lnq_checkpoint: Path,
    hessians_dir: Path,
    output_dir: Path,
    bits: int,
    rho: float,
    guidedquant_dir: Path,
    yaml_path: Optional[str] = None,
    max_sweeps: int = 10,
    tau_gain: float = 1e-12,
    tau_feas_rel: float = 1e-8,
    relative_gain_tol: float = 1e-6,
    numeric_floor: float = 1e-12,
    dtype: torch.dtype = torch.float32,
    objective_device: Optional[torch.device] = None,
    sub_qlayer: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a MaxDrift LNQ cache from a GuidedQuant LNQ cache."""
    if not gq_lnq_checkpoint.exists():
        raise FileNotFoundError(f"Missing GQ-LNQ checkpoint/cache: {gq_lnq_checkpoint}")
    if not (gq_lnq_checkpoint / "weights").exists():
        raise FileNotFoundError(f"Missing GuidedQuant weights directory: {gq_lnq_checkpoint / 'weights'}")
    if not (gq_lnq_checkpoint / f"lut_{bits}").exists():
        raise FileNotFoundError(f"Missing GuidedQuant lut directory: {gq_lnq_checkpoint / f'lut_{bits}'}")
    if not hessians_dir.exists():
        raise FileNotFoundError(f"Missing GuidedQuant Hessian directory: {hessians_dir}")
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory exists; pass --overwrite to replace it: {output_dir}")

    _copy_tree(gq_lnq_checkpoint, output_dir)
    (output_dir / "maxdrift_logs").mkdir(parents=True, exist_ok=True)

    analyzer = _load_guidedquant_analyzer(guidedquant_dir, fp_model, yaml_path)
    module_names = list(analyzer.module_names)
    layer_start, layer_end = sub_qlayer if sub_qlayer else (0, analyzer.num_layers)
    layer_indices = list(range(layer_start, layer_end))

    layer_summaries: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    model_start = time.time()
    if objective_device is None:
        objective_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("MaxDrift objective device=%s dtype=%s", objective_device, dtype)

    for layer_idx in layer_indices:
        logging.info("MaxDrift layer %s", layer_idx)
        weight_path = output_dir / "weights" / f"l{layer_idx}.pt"
        lut_path = output_dir / f"lut_{bits}" / f"l{layer_idx}.pt"
        hessian_path = hessians_dir / f"l{layer_idx}.pt"
        if not weight_path.exists() or not lut_path.exists() or not hessian_path.exists():
            raise FileNotFoundError(f"Missing layer cache for l{layer_idx}: {weight_path}, {lut_path}, {hessian_path}")

        codes_dict = torch.load(weight_path, map_location="cpu")
        lut_dict = torch.load(lut_path, map_location="cpu")
        hessian_dict = torch.load(hessian_path, map_location="cpu")
        weight_dict = analyzer.get_layer_weights(layer_idx)

        new_codes_dict: Dict[str, torch.Tensor] = {}
        for module_name in module_names:
            module_start = time.time()
            logging.info("MaxDrift layer %s module %s", layer_idx, module_name)
            if module_name not in codes_dict or module_name not in lut_dict or module_name not in hessian_dict:
                raise KeyError(f"Layer {layer_idx} missing module {module_name} in codes/lut/hessian cache")
            weight = weight_dict[module_name].detach().cpu().float()
            codes = torch.as_tensor(codes_dict[module_name])
            codebooks = torch.as_tensor(lut_dict[module_name])
            hessians = torch.as_tensor(hessian_dict[module_name])
            new_codes, summary, channel_logs = maxdrift_module(
                weight=weight,
                codes=codes,
                codebooks=codebooks,
                hessians=hessians,
                bits=bits,
                rho=rho,
                max_sweeps=max_sweeps,
                tau_gain=tau_gain,
                tau_feas_rel=tau_feas_rel,
                relative_gain_tol=relative_gain_tol,
                numeric_floor=numeric_floor,
                dtype=dtype,
                objective_device=objective_device,
            )
            new_codes_dict[module_name] = new_codes
            summary.layer_idx = layer_idx
            summary.module_name = module_name
            layer_summary = asdict(summary)
            layer_summaries.append(layer_summary)
            for row in channel_logs:
                row.update({"layer_name": f"l{layer_idx}", "layer_idx": layer_idx, "module_name": module_name, "bits": bits, "rho": rho})
                sweep_rows.append(row)
            logging.info(
                "MaxDrift layer %s module %s done: changed=%s frac=%.6f drift %.6e -> %.6e in %.1fs",
                layer_idx,
                module_name,
                summary.num_changed_codes,
                summary.fraction_changed_codes,
                summary.weight_drift_start,
                summary.weight_drift_final,
                time.time() - module_start,
            )

        torch.save(new_codes_dict, weight_path)

    summary = {
        "method": "MaxDrift-GQ",
        "bits": bits,
        "rho": rho,
        "num_layers": len(layer_indices),
        "num_changed_codes": int(sum(row["num_changed_codes"] for row in layer_summaries)),
        "fraction_changed_codes": _weighted_fraction_changed(layer_summaries),
        "preservation_cost_start": float(sum(row["preservation_cost_start"] for row in layer_summaries)),
        "preservation_cost_final": float(sum(row["preservation_cost_final"] for row in layer_summaries)),
        "weight_drift_start": float(sum(row["weight_drift_start"] for row in layer_summaries)),
        "weight_drift_final": float(sum(row["weight_drift_final"] for row in layer_summaries)),
        "normalized_weight_drift_start": _model_normalized_drift(layer_summaries, "weight_drift_start"),
        "normalized_weight_drift_final": _model_normalized_drift(layer_summaries, "weight_drift_final"),
        "runtime_sec": time.time() - model_start,
        "layers": layer_summaries,
    }

    with open(output_dir / "maxdrift_logs" / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    _write_csv(output_dir / "maxdrift_logs" / "layers.csv", layer_summaries)
    _write_csv(output_dir / "maxdrift_logs" / "sweeps.csv", sweep_rows)
    logging.info("MaxDrift complete: %s", output_dir)
    return summary


def _weighted_fraction_changed(layer_summaries: Sequence[Mapping[str, Any]]) -> float:
    changed = sum(float(row["num_changed_codes"]) for row in layer_summaries)
    denom = sum(float(row.get("num_codes", 0.0)) for row in layer_summaries)
    return changed / max(denom, 1.0)


def _model_normalized_drift(layer_summaries: Sequence[Mapping[str, Any]], key: str) -> float:
    drift = sum(float(row[key]) for row in layer_summaries)
    weight_norm_sq = sum(float(row.get("original_weight_norm_sq", 0.0)) for row in layer_summaries)
    return math.sqrt(max(drift, 0.0)) / max(math.sqrt(max(weight_norm_sq, 0.0)), 1e-12)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dtype_from_name(name: str) -> torch.dtype:
    table = {"float64": torch.float64, "fp64": torch.float64, "float32": torch.float32, "fp32": torch.float32}
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}") from exc


def _device_from_name(name: str) -> torch.device:
    lowered = name.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MaxDrift-GQ from a GuidedQuant LNQ cache")
    parser.add_argument("--fp_model", required=True, help="Fingerprint model path/name; used as GuidedQuant teacher/reference W")
    parser.add_argument("--gq_lnq_checkpoint", required=True, type=Path, help="GuidedQuant layerwise_quantized cache")
    parser.add_argument("--gq_cache_dir", required=True, type=Path, help="GuidedQuant Hessian cache directory containing l*.pt")
    parser.add_argument("--guidedquant_dir", type=Path, default=Path("GuidedQuant"), help="Path to GuidedQuant source")
    parser.add_argument("--yaml_path", default=None, help="Optional GuidedQuant architecture YAML")
    parser.add_argument("--bits", required=True, type=int)
    parser.add_argument("--rho", required=True, type=float)
    parser.add_argument("--max_sweeps", type=int, default=10)
    parser.add_argument("--tau_gain", type=float, default=1e-12)
    parser.add_argument("--tau_feas_rel", type=float, default=1e-8)
    parser.add_argument("--relative_gain_tol", type=float, default=1e-6)
    parser.add_argument("--numeric_floor", type=float, default=1e-12)
    parser.add_argument("--objective_dtype", type=_dtype_from_name, default=torch.float32)
    parser.add_argument("--objective_device", type=_device_from_name, default=_device_from_name("auto"))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sub_qlayer", nargs=2, type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s | %(levelname)s] %(message)s")
    run_maxdrift_cache(
        fp_model=args.fp_model,
        gq_lnq_checkpoint=args.gq_lnq_checkpoint,
        hessians_dir=args.gq_cache_dir,
        output_dir=args.output_dir,
        bits=args.bits,
        rho=args.rho,
        guidedquant_dir=args.guidedquant_dir,
        yaml_path=args.yaml_path,
        max_sweeps=args.max_sweeps,
        tau_gain=args.tau_gain,
        tau_feas_rel=args.tau_feas_rel,
        relative_gain_tol=args.relative_gain_tol,
        numeric_floor=args.numeric_floor,
        dtype=args.objective_dtype,
        objective_device=args.objective_device,
        sub_qlayer=tuple(args.sub_qlayer) if args.sub_qlayer else None,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

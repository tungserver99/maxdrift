import math

import torch

from maxdrift_gq import (
    compute_objectives,
    maxdrift_channel,
    maxdrift_group,
    maxdrift_groups,
    recover_codes_from_codebook,
    reconstruct_from_codes,
)


def _psd(dim: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    return a.T @ a + 0.05 * torch.eye(dim, dtype=torch.float64)


def test_delta_formulas_match_bruteforce_objectives():
    gen = torch.Generator().manual_seed(11)
    dim = 9
    hessian = _psd(dim)
    w = torch.randn(dim, generator=gen, dtype=torch.float64)
    q = torch.randn(dim, generator=gen, dtype=torch.float64)
    e = q - w
    r = hessian @ e
    p, d = compute_objectives(e, hessian)

    i = 4
    candidate = q[i] + torch.tensor(0.375, dtype=torch.float64)
    delta = candidate - q[i]
    q_next = q.clone()
    q_next[i] = candidate
    e_next = q_next - w
    p_next, d_next = compute_objectives(e_next, hessian)

    delta_p = 2.0 * delta * r[i] + delta * delta * hessian[i, i]
    delta_d = 2.0 * delta * e[i] + delta * delta

    assert torch.allclose(p_next - p, delta_p, rtol=0.0, atol=1e-10)
    assert torch.allclose(d_next - d, delta_d, rtol=0.0, atol=1e-10)


def test_maxdrift_channel_preserves_budget_and_increases_drift():
    hessian = torch.eye(4, dtype=torch.float64)
    w = torch.zeros(4, dtype=torch.float64)
    codebook = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    codes0 = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    q0 = reconstruct_from_codes(codes0, codebook)

    result = maxdrift_channel(
        w=w,
        q0=q0,
        codes0=codes0,
        codebook=codebook,
        hessian=hessian,
        rho=0.0,
        numeric_floor=2.0,
        max_sweeps=10,
    )

    assert result.p_final <= result.epsilon + result.tau_feas
    assert result.d_final >= result.d_start
    assert result.num_changed_codes == 2
    assert math.isclose(result.d_final, 2.0, rel_tol=0.0, abs_tol=1e-12)


def test_current_code_is_feasible_so_no_forced_update():
    hessian = torch.eye(3, dtype=torch.float64)
    w = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    codebook = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    codes0 = torch.tensor([1, 1, 1], dtype=torch.long)
    q0 = reconstruct_from_codes(codes0, codebook)

    result = maxdrift_channel(
        w=w,
        q0=q0,
        codes0=codes0,
        codebook=codebook,
        hessian=hessian,
        rho=0.0,
        max_sweeps=4,
    )

    assert result.p_final <= result.epsilon + result.tau_feas
    assert result.num_sweeps >= 1


def test_recover_codes_reconstructs_quantized_vector():
    codebook = torch.tensor([-1.5, -0.25, 0.75, 2.0], dtype=torch.float64)
    codes = torch.tensor([0, 3, 2, 1, 1, 0], dtype=torch.long)
    q = reconstruct_from_codes(codes, codebook)

    recovered = recover_codes_from_codebook(q, codebook, atol=1e-12)

    assert torch.equal(recovered, codes)
    assert torch.allclose(reconstruct_from_codes(recovered, codebook), q)


def test_maxdrift_channel_is_deterministic():
    hessian = _psd(6)
    w = torch.tensor([0.2, -0.4, 0.1, 0.9, -0.7, 0.05], dtype=torch.float64)
    codebook = torch.tensor([-1.0, -0.2, 0.25, 0.9], dtype=torch.float64)
    codes0 = torch.tensor([1, 1, 2, 2, 1, 2], dtype=torch.long)
    q0 = reconstruct_from_codes(codes0, codebook)

    first = maxdrift_channel(w, q0, codes0, codebook, hessian, rho=0.2)
    second = maxdrift_channel(w, q0, codes0, codebook, hessian, rho=0.2)

    assert torch.equal(first.codes, second.codes)
    assert torch.allclose(first.q, second.q)
    assert first.to_summary() == second.to_summary()


def test_maxdrift_group_matches_channel_reference():
    hessian = _psd(5)
    w = torch.tensor(
        [
            [0.2, -0.4, 0.1, 0.9, -0.7],
            [-0.3, 0.8, -0.2, 0.4, 0.05],
            [0.6, -0.1, -0.8, 0.2, 0.7],
        ],
        dtype=torch.float64,
    )
    codebooks = torch.tensor(
        [
            [-1.0, -0.2, 0.25, 0.9],
            [-0.8, -0.1, 0.35, 1.1],
            [-1.2, 0.0, 0.5, 0.95],
        ],
        dtype=torch.float64,
    )
    codes0 = torch.tensor(
        [
            [1, 1, 2, 2, 1],
            [1, 2, 1, 2, 1],
            [2, 1, 1, 2, 2],
        ],
        dtype=torch.long,
    )

    group = maxdrift_group(
        w=w,
        codes0=codes0,
        codebooks=codebooks,
        hessian=hessian,
        rho=0.15,
        max_sweeps=5,
        tau_gain=1e-12,
        tau_feas_rel=1e-8,
        relative_gain_tol=1e-6,
        numeric_floor=1e-12,
    )

    reference_codes = []
    reference_changed = 0
    for row in range(w.shape[0]):
        q0 = reconstruct_from_codes(codes0[row], codebooks[row])
        result = maxdrift_channel(
            w=w[row],
            q0=q0,
            codes0=codes0[row],
            codebook=codebooks[row],
            hessian=hessian,
            rho=0.15,
            max_sweeps=5,
        )
        reference_codes.append(result.codes)
        reference_changed += result.num_changed_codes

    assert torch.equal(group["codes"].cpu(), torch.stack(reference_codes))
    assert group["num_changed_codes"] == reference_changed


def test_maxdrift_groups_matches_per_group_reference():
    hessians = torch.stack([_psd(5), _psd(5) * 1.3])
    w = torch.tensor(
        [
            [0.2, -0.4, 0.1, 0.9, -0.7],
            [-0.3, 0.8, -0.2, 0.4, 0.05],
            [0.6, -0.1, -0.8, 0.2, 0.7],
            [-0.5, 0.15, 0.25, -0.6, 0.3],
        ],
        dtype=torch.float64,
    )
    codebooks = torch.tensor(
        [
            [-1.0, -0.2, 0.25, 0.9],
            [-0.8, -0.1, 0.35, 1.1],
            [-1.2, 0.0, 0.5, 0.95],
            [-0.9, -0.05, 0.4, 1.0],
        ],
        dtype=torch.float64,
    )
    codes0 = torch.tensor(
        [
            [1, 1, 2, 2, 1],
            [1, 2, 1, 2, 1],
            [2, 1, 1, 2, 2],
            [1, 1, 2, 1, 2],
        ],
        dtype=torch.long,
    )

    batched = maxdrift_groups(
        w=w,
        codes0=codes0,
        codebooks=codebooks,
        hessians=hessians,
        rho=0.15,
        max_sweeps=5,
        tau_gain=1e-12,
        tau_feas_rel=1e-8,
        relative_gain_tol=1e-6,
        numeric_floor=1e-12,
    )

    per_group_codes = []
    changed = 0
    group_size = 2
    for group_idx in range(2):
        start = group_idx * group_size
        end = start + group_size
        group = maxdrift_group(
            w=w[start:end],
            codes0=codes0[start:end],
            codebooks=codebooks[start:end],
            hessian=hessians[group_idx],
            rho=0.15,
            max_sweeps=5,
            tau_gain=1e-12,
            tau_feas_rel=1e-8,
            relative_gain_tol=1e-6,
            numeric_floor=1e-12,
        )
        per_group_codes.append(group["codes"])
        changed += group["num_changed_codes"]

    assert torch.equal(batched["codes"].cpu(), torch.cat(per_group_codes, dim=0))
    assert batched["num_changed_codes"] == changed


def test_maxdrift_groups_handles_unsorted_codebooks_like_exhaustive_reference():
    hessians = torch.stack([_psd(4), _psd(4) * 1.1])
    w = torch.tensor(
        [
            [0.2, -0.4, 0.1, 0.9],
            [-0.3, 0.8, -0.2, 0.4],
            [0.6, -0.1, -0.8, 0.2],
            [-0.5, 0.15, 0.25, -0.6],
        ],
        dtype=torch.float64,
    )
    codebooks = torch.tensor(
        [
            [0.9, -1.0, 0.25, -0.2],
            [0.35, -0.8, 1.1, -0.1],
            [0.5, -1.2, 0.95, 0.0],
            [1.0, -0.9, 0.4, -0.05],
        ],
        dtype=torch.float64,
    )
    codes0 = torch.tensor(
        [
            [3, 3, 2, 0],
            [3, 0, 3, 0],
            [0, 3, 3, 0],
            [3, 3, 2, 3],
        ],
        dtype=torch.long,
    )

    batched = maxdrift_groups(
        w=w,
        codes0=codes0,
        codebooks=codebooks,
        hessians=hessians,
        rho=0.10,
        max_sweeps=4,
        tau_gain=1e-12,
        tau_feas_rel=1e-8,
        relative_gain_tol=1e-6,
        numeric_floor=1e-12,
    )

    per_group_codes = []
    group_size = 2
    for group_idx in range(2):
        start = group_idx * group_size
        end = start + group_size
        group = maxdrift_group(
            w=w[start:end],
            codes0=codes0[start:end],
            codebooks=codebooks[start:end],
            hessian=hessians[group_idx],
            rho=0.10,
            max_sweeps=4,
            tau_gain=1e-12,
            tau_feas_rel=1e-8,
            relative_gain_tol=1e-6,
            numeric_floor=1e-12,
        )
        per_group_codes.append(group["codes"])

    assert torch.equal(batched["codes"].cpu(), torch.cat(per_group_codes, dim=0))

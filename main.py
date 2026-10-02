#!/usr/bin/env python3
"""End-to-end MaxDrift-GQ runner.

The heavy GuidedQuant stages are called through the official GuidedQuant
entrypoints in ``GuidedQuant/``. This file only orchestrates the pipeline.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def _model_name(model: str) -> str:
    return model.rstrip("/\\").replace("\\", "/").split("/")[-1]


def _run(cmd: Sequence[str], cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"[run] {printable}")
    if dry_run:
        return
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=True)


def _pack_maxdrift_cache(
    *,
    guidedquant_dir: Path,
    fp_model: str,
    yaml_path: Optional[str],
    lut_path: Path,
    output_model_path: Path,
    bits: int,
    dry_run: bool,
) -> None:
    print(f"[pack] GuidedQuant pack {lut_path} -> {output_model_path}")
    if dry_run:
        return
    sys.path.insert(0, str(guidedquant_dir))
    from any_precision.analyzer import get_analyzer
    from any_precision.quantization.pack import pack

    analyzer = get_analyzer(fp_model, yaml_path=yaml_path, include_tokenizer=True)
    analyzer.drop_original_weights()
    pack(
        analyzer=analyzer,
        lut_path=str(lut_path),
        output_model_path=str(output_model_path),
        seed_precision=bits,
        parent_precision=bits,
    )


def guidedquant_paths(
    guidedquant_dir: Path,
    fp_model: str,
    bits: int,
    dataset: str,
    num_examples: int,
    seq_len: int,
    num_groups: int,
    num_iterations: int,
    cd_cycles: int,
    cache_dir: str,
) -> tuple[Path, Path]:
    name = _model_name(fp_model)
    cache_root = guidedquant_dir / cache_dir
    hessians_dir = cache_root / "hessians" / f"{name}-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}"
    lnq_dir = cache_root / "layerwise_quantized" / (
        f"{name}-w{bits}-{dataset}_s{num_examples}_blk{seq_len}"
        f"_g{num_groups}_iter{num_iterations}_cd{cd_cycles}"
    )
    return lnq_dir, hessians_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GuidedQuant LNQ -> MaxDrift-GQ -> optional PPL")
    parser.add_argument("--fp_model", required=True, help="Fingerprint model path/name")
    parser.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 8])
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--guidedquant_dir", type=Path, default=Path("GuidedQuant"))
    parser.add_argument("--yaml_path", default=None, help="Optional GuidedQuant architecture YAML")
    parser.add_argument("--cache_dir", default="cache", help="GuidedQuant cache dir, relative to GuidedQuant by default")
    parser.add_argument("--dataset", default="c4")
    parser.add_argument("--num_examples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--cd_cycles", type=int, default=4)
    parser.add_argument("--max_sweeps", type=int, default=10)
    parser.add_argument("--objective_dtype", default="fp32", choices=["fp32", "float32", "fp64", "float64"])
    parser.add_argument("--objective_device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_guidedquant", action="store_true", help="Use existing GQ-LNQ/Hessian caches")
    parser.add_argument("--gq_lnq_checkpoint", type=Path, default=None, help="Override GQ-LNQ layerwise cache")
    parser.add_argument("--gq_hessians_dir", type=Path, default=None, help="Override GuidedQuant Hessian cache")
    parser.add_argument("--pack", action="store_true", help="Pack MaxDrift cache with GuidedQuant pack after optimization")
    parser.add_argument("--packed_output_dir", type=Path, default=None, help="HF/AnyPrecision checkpoint directory produced by --pack")
    parser.add_argument("--run_ppl", action="store_true", help="Call eval_ppl.py after MaxDrift")
    parser.add_argument("--ppl_model_path", type=Path, default=None, help="Packed/fake-quant HF path to evaluate; defaults to output_dir")
    parser.add_argument("--ppl_datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    guidedquant_dir = args.guidedquant_dir.resolve()
    fp_model_path = Path(args.fp_model)
    fp_model_arg = str(fp_model_path.resolve()) if fp_model_path.exists() else args.fp_model

    derived_lnq, derived_hessians = guidedquant_paths(
        guidedquant_dir=guidedquant_dir,
        fp_model=args.fp_model,
        bits=args.bits,
        dataset=args.dataset,
        num_examples=args.num_examples,
        seq_len=args.seq_len,
        num_groups=args.num_groups,
        num_iterations=args.num_iterations,
        cd_cycles=args.cd_cycles,
        cache_dir=args.cache_dir,
    )
    gq_lnq_checkpoint = args.gq_lnq_checkpoint or derived_lnq
    gq_hessians_dir = args.gq_hessians_dir or derived_hessians
    output_dir = args.output_dir or (
        Path("outputs")
        / "maxdrift_gq"
        / _model_name(args.fp_model)
        / f"w{args.bits}_rho{args.rho:.2f}"
    )
    packed_output_dir = args.packed_output_dir or Path(f"{output_dir}_packed")

    if not args.skip_guidedquant:
        _run(
            [
                sys.executable,
                "quantize.py",
                fp_model_arg,
                "--seed_precision",
                str(args.bits),
                "--parent_precision",
                str(args.bits),
                *(["--yaml_path", args.yaml_path] if args.yaml_path else []),
                "--cache_dir",
                args.cache_dir,
                "--dataset",
                args.dataset,
                "--seq_len",
                str(args.seq_len),
                "--num_examples",
                str(args.num_examples),
                "--num_groups",
                str(args.num_groups),
                "--random_state",
                str(args.seed),
            ],
            cwd=guidedquant_dir,
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                "layerwise_nuq.py",
                fp_model_arg,
                "--seed_precision",
                str(args.bits),
                *(["--yaml_path", args.yaml_path] if args.yaml_path else []),
                "--cache_dir",
                args.cache_dir,
                "--dataset",
                args.dataset,
                "--seq_len",
                str(args.seq_len),
                "--num_examples",
                str(args.num_examples),
                "--num_groups",
                str(args.num_groups),
                "--num_iterations",
                str(args.num_iterations),
                "--cd_cycles",
                str(args.cd_cycles),
                "--random_state",
                str(args.seed),
            ],
            cwd=guidedquant_dir,
            dry_run=args.dry_run,
        )

    _run(
        [
            sys.executable,
            "maxdrift_gq.py",
            "--fp_model",
            fp_model_arg,
            "--gq_lnq_checkpoint",
            str(gq_lnq_checkpoint),
            "--gq_cache_dir",
            str(gq_hessians_dir),
            "--guidedquant_dir",
            str(guidedquant_dir),
            *(["--yaml_path", args.yaml_path] if args.yaml_path else []),
            "--bits",
            str(args.bits),
            "--rho",
            str(args.rho),
            "--max_sweeps",
            str(args.max_sweeps),
            "--objective_dtype",
            args.objective_dtype,
            "--objective_device",
            args.objective_device,
            "--output_dir",
            str(output_dir),
            *(["--overwrite"] if args.overwrite else []),
        ],
        dry_run=args.dry_run,
    )

    if args.pack:
        _pack_maxdrift_cache(
            guidedquant_dir=guidedquant_dir,
            fp_model=fp_model_arg,
            yaml_path=args.yaml_path,
            lut_path=output_dir,
            output_model_path=packed_output_dir,
            bits=args.bits,
            dry_run=args.dry_run,
        )

    if args.run_ppl:
        if args.ppl_model_path is None and not args.pack:
            raise SystemExit("--run_ppl needs --pack or an explicit --ppl_model_path; MaxDrift output_dir is an LNQ cache, not a HF checkpoint.")
        ppl_model_path = args.ppl_model_path or packed_output_dir
        _run(
            [
                sys.executable,
                "eval_ppl.py",
                "--model-path",
                str(ppl_model_path),
                "--datasets",
                *args.ppl_datasets,
                "--seqlen",
                str(args.seq_len),
            ],
            dry_run=args.dry_run,
        )

    manifest = {
        "fp_model": fp_model_arg,
        "bits": args.bits,
        "rho": args.rho,
        "yaml_path": args.yaml_path,
        "guidedquant_lnq_checkpoint": str(gq_lnq_checkpoint),
        "guidedquant_hessians_dir": str(gq_hessians_dir),
        "maxdrift_output_dir": str(output_dir),
        "packed_output_dir": str(packed_output_dir) if args.pack else None,
        "calibration": {
            "dataset": args.dataset,
            "num_examples": args.num_examples,
            "seq_len": args.seq_len,
            "seed": args.seed,
        },
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

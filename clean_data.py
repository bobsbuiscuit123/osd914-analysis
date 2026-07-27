from __future__ import annotations

"""Legacy clean-matrix CLI backed by the canonical reproduction code."""

import argparse
from pathlib import Path
from typing import Iterable, Optional

from reproduce import ROOT, project_path, rebuild_clean_matrices


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild clean liver and brain matrices from the raw gzipped inputs.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory that receives a clean/ subdirectory with rebuilt matrices.",
    )
    parser.add_argument(
        "--skip-root-copies",
        action="store_true",
        help="Do not write legacy liver_features_clean.csv and brain_targets_clean.csv copies in the repository root.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    liver_df, brain_df, metadata = rebuild_clean_matrices(output_dir.resolve())
    print(f"Clean liver matrix: {liver_df.shape[0]} samples x {liver_df.shape[1]} features")
    print(f"Clean brain matrix: {brain_df.shape[0]} samples x {brain_df.shape[1]} targets")
    print(f"Brain source: {metadata['brain_source']}")
    print(f"Wrote clean matrices under: {project_path(output_dir / 'clean')}")

    if not args.skip_root_copies:
        liver_path = ROOT / "liver_features_clean.csv"
        brain_path = ROOT / "brain_targets_clean.csv"
        liver_df.to_csv(liver_path)
        brain_df.to_csv(brain_path)
        print(f"Wrote legacy copy: {project_path(liver_path)}")
        print(f"Wrote legacy copy: {project_path(brain_path)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""hlaont: HLA typing from ONT long reads.

    hlaont run SAMPLE_ID READS.fastq --out-dir OUT/
    hlaont run SAMPLE_ID --bam ALIGNED.bam --out-dir OUT/
    hlaont run SAMPLE_ID --bam ALIGNED.cram --reference REF.fa --out-dir OUT/
    hlaont run SAMPLE_ID --bam PHASED.bam --phased --out-dir OUT/
    hlaont update-db [--hla-seq-dir HLA_SEQ_DIR]
    hlaont update-anchors

run: types every gene in resources/genes.yaml and writes
{sample}.calls.tsv, .consensus.fa, .consensus.qv.tsv, .variants.vcf and
.qc.json. With --bam, the header picks the MHC/ALT/decoy regions; those
reads plus unmapped reads are extracted and sieved exactly like FASTQ
input. --reference is needed only for CRAM. --phased (with --bam) types
each gene per HP haplotype when its reads are cleanly phased in one PS
block, falling back to unphased per gene (see qc.json "phasing").

update-db: rebuilds the template pool from IPD-IMGT/HLA plus
assembly-derived templates for HLA-N/P/T/U/W/Y/R (from the bundled
snapshot unless --hla-seq-dir is given).

update-anchors: rebuilds resources/anchors/anchors.fa.gz; needs cd-hit-est
on PATH.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Callable

import pysam

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "stages"))
sys.path.insert(0, str(REPO_ROOT / "input"))

from bam import extract_mini_bam, mini_bam_to_fastq  # noqa: E402
from output import (  # noqa: E402
    write_calls_tsv,
    write_consensus_fasta,
    write_qc_json,
    write_variants_vcf,
)
from chr17_control import compute_chr17_control_depth_from_bam  # noqa: E402
from pipeline import run_sample, run_sample_phased  # noqa: E402
from regions import RegionSet, derive_regions_from_header  # noqa: E402

def is_remote_bam(bam: str) -> bool:
    """True for an http(s):// --bam value (a URL kept as plain str, see the
    --bam argparse help above) rather than a local path."""
    return bam.startswith("http://") or bam.startswith("https://")


BUILD_PY = REPO_ROOT / "resources" / "templates" / "build.py"
ANCHORS_DIR = REPO_ROOT / "resources" / "anchors"
PREPARE_CANDIDATES_PY = ANCHORS_DIR / "prepare_candidates.py"
ANCHORS_BUILD_PY = ANCHORS_DIR / "build.py"
CDHIT_SCRIPT = ANCHORS_DIR / "build_staging" / "run_cdhit.sh"
CDHIT_CLUSTERED_DIR = ANCHORS_DIR / "build_staging" / "clustered"


def _extract_bam_to_fastq(
    sample_id: str, bam_path: Path | str, reference_path: Path | None, out_dir: Path,
    *, samtools_bin: str, n_threads: int, progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path, RegionSet]:
    """Stage 0 (DESIGN.md S3.1): BAM/CRAM -> mini-BAM -> FASTQ.
    `reference_path` is needed only for CRAM. Returns (fastq_path,
    mini_bam_path, region_set); the mini-BAM keeps HP/PS tags for --phased,
    and region_set locates the chr17 control region."""
    reference_filename = str(reference_path) if reference_path is not None else None
    with pysam.AlignmentFile(str(bam_path), reference_filename=reference_filename) as bf:
        region_set = derive_regions_from_header(bf.header)

    if progress:
        progress("Stage 0: extracting mini-BAM from BAM/CRAM")
    mini_bam_path = out_dir / f"{sample_id}.stage0.mini.bam"
    extract_mini_bam(
        bam_path, reference_path, region_set, mini_bam_path,
        samtools_bin=samtools_bin, n_threads=n_threads,
    )
    if progress:
        progress("Stage 0: converting mini-BAM to FASTQ")
    fastq_path = out_dir / f"{sample_id}.stage0.fastq"
    mini_bam_to_fastq(mini_bam_path, fastq_path, samtools_bin=samtools_bin, n_threads=n_threads)
    return fastq_path, mini_bam_path, region_set


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser("run", help="Type one sample's HLA genes from a FASTQ or BAM/CRAM")
    ap.add_argument("sample_id", help="Sample identifier, recorded in every output file")
    ap.add_argument(
        "fastq", nargs="?", type=Path, default=None,
        help="ONT R10 FASTQ (DESIGN.md S3.2 anchor-sieve input). Omit when using --bam.",
    )
    ap.add_argument("-o","--out-dir", type=Path, required=True, help="Directory for calls.tsv/consensus.fa/variants.vcf/qc.json")
    ap.add_argument("-t","--n-threads", type=int, default=4, help="minimap2/samtools resource knob, not a biological parameter")
    ap.add_argument("--minimap2-bin", default="minimap2")
    ap.add_argument("--dorado-version", default=None, help="Recorded in qc.json only -- not read from the FASTQ itself")
    ap.add_argument(
        "--bam", type=str, default=None,
        help="Aligned BAM/CRAM. Requires --reference for a .cram input; "
             "optional for a plain .bam. Indexed input required (.bai/.crai). May be an http(s):// "
             "URL instead of a local path."
             )
    ap.add_argument(
        "--reference", type=Path, default=None,
        help="Reference FASTA matching --bam/cram's alignment",
    )
    ap.add_argument("--samtools-bin", default="samtools", help="samtools binary")
    ap.add_argument(
        "--phased", action="store_true",
        help="Treat --bam's HP/PS tags as real read-backed phase and run the "
             "per-haplotype pipeline instead of the unphased. A gene whose reads don't cleanly split by HP under one PS block "
             "falls back to the unphased path for that gene only. "
    )
    ap.add_argument("--pool-fasta", type=Path, default=None, help="Override resources/templates/pool/ref.gene.fa.gz (versioned resource, DESIGN.md S2d)")
    ap.add_argument("--pool-meta", type=Path, default=None, help="Override resources/templates/pool/ref.gene.meta.tsv.gz")
    ap.add_argument("--models-path", type=Path, default=None, help="Override resources/templates/pool/ref.gene.models.jsonl.gz")
    ap.add_argument("--cds-catalogue-path", type=Path, default=None, help="Override resources/templates/pool/ref.gene.cds_catalogue.tsv.gz")
    ap.add_argument("--c4-pool-fasta", type=Path, default=None, help="Override resources/templates/assembly_bundle/c4.fa.gz (DESIGN.md S2f total-CN module)")
    ap.add_argument("--c4-pool-meta", type=Path, default=None, help="Override resources/templates/assembly_bundle/c4.meta.tsv.gz")
    ap.add_argument("--genes-yaml", type=Path, default=None, help="Override resources/genes.yaml")
    ap.add_argument("--anchors-path", type=Path, default=None, help="Override resources/anchors/anchors.fa.gz")
    ap.add_argument("--anchors-meta-path", type=Path, default=None, help="Override resources/anchors/anchors.meta.tsv.gz")
    ap.set_defaults(func=_run_command)


def _add_update_db_subparser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser(
        "update-db",
        help="Rebuild the template pool from IPD + real assembly-derived templates",
        description="Rebuilds resources/templates/pool/ from a pinned/fresh IPD-IMGT/HLA "
                     "tag, merged with real assembly-derived templates for HLA-N/P/T/U/W/Y/R "
                     "(resources/templates/assembly_sources.py).",
    )
    ap.add_argument(
        "--hla-seq-dir", type=Path, default=None,
        help="Real per-individual phased assemblies used to "
             "extract templates for genes.yaml's assembly_extraction genes (HLA-N/P/T/U/W/Y/R) "
             "via a fresh parse. "
    )
    ap.add_argument("--assembly-bundle-dir", type=Path, default=None,
                     help="Override resources/templates/assembly_bundle (used for HLA-N/P/T/U/W/Y/R "
                          "when --hla-seq-dir is not given)")
    ap.add_argument("--genes-yaml", type=Path, default=None, help="Override resources/genes.yaml")
    ap.add_argument("--raw-dir", type=Path, default=None, help="Override resources/templates/raw (IPD hla.dat cache)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Override resources/templates/pool")
    ap.add_argument("--ipd-tag", default=None, help="Pinned IMGTHLA git tag (default: build.py's own IPD_DEFAULT_TAG)")
    ap.add_argument("--ipd-dat", type=Path, default=None, help="Use a local hla.dat instead of downloading")
    ap.set_defaults(func=_update_db_command)


def _update_db_command(args: argparse.Namespace) -> int:
    if args.hla_seq_dir is not None and not args.hla_seq_dir.exists():
        sys.exit(f"[hlaont] ERROR: --hla-seq-dir {args.hla_seq_dir} does not exist")

    cmd = [sys.executable, str(BUILD_PY)]
    if args.hla_seq_dir is not None:
        cmd += ["--hla-seq-dir", str(args.hla_seq_dir)]
    if args.assembly_bundle_dir is not None:
        cmd += ["--assembly-bundle-dir", str(args.assembly_bundle_dir)]
    if args.genes_yaml is not None:
        cmd += ["--genes-yaml", str(args.genes_yaml)]
    if args.raw_dir is not None:
        cmd += ["--raw-dir", str(args.raw_dir)]
    if args.out_dir is not None:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.ipd_tag is not None:
        cmd += ["--ipd-tag", args.ipd_tag]
    if args.ipd_dat is not None:
        cmd += ["--ipd-dat", str(args.ipd_dat)]

    print(f"[hlaont] update-db: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)

    if args.hla_seq_dir is None:
        print(
            "\n[hlaont] update-db: --hla-seq-dir not given -- HLA-N/P/T/U/W/Y/R "
            "templates came from the checked-in assembly_bundle/ snapshot "
            "(hla_seq-2025-03-v1), not a fresh extraction. Pass --hla-seq-dir "
            "if you have a newer/larger hla_seq/ to extract from instead.\n",
            file=sys.stderr,
        )

    print(
        "\n[hlaont] update-db: pool rebuilt. This does NOT auto-run (human-gated):\n"
        "  1. Rebuild anchors so read recruitment matches the new pool:\n"
        "       hlaont update-anchors   # needs cd-hit-est installed\n"
        "     Confirm any gene that gained real templates this run drops out of\n"
        "     anchors_manifest.json's missing_target_anchors.\n"
        "  2. Re-check ipd_audit.py's counts against the new IPD tag for any\n"
        "     typing_module: null / tier-4 gene that may have gained real IPD\n"
        "     genomic records -- genes.yaml is not auto-edited by this command.\n",
        file=sys.stderr,
    )
    return 0


def _add_update_anchors_subparser(subparsers: argparse._SubParsersAction) -> None:
    ap = subparsers.add_parser(
        "update-anchors",
        help="Rebuild the anchor set (ported FuFiHLA + cd-hit-est-derived + IPD singletons)",
        description="Companion to update-db: reruns resources/anchors/prepare_candidates.py "
                     "to stage per-gene candidate FASTAs against the current template pool, "
                     "runs cd-hit-est on them, then resources/anchors/build.py to consolidate "
                     "the result into anchors.fa.gz. "
    )
    ap.set_defaults(func=_update_anchors_command)


def _run_cdhit_commands(script_path: Path) -> None:
    """Run the cd-hit-est commands in run_cdhit.sh, skipping `module load`
    lines."""
    CDHIT_CLUSTERED_DIR.mkdir(parents=True, exist_ok=True)
    for line in script_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("cd-hit-est"):
            continue
        print(f"[hlaont] update-anchors: {line}", file=sys.stderr)
        subprocess.run(shlex.split(line), check=True)


def _update_anchors_command(args: argparse.Namespace) -> int:
    if shutil.which("cd-hit-est") is None:
        print(
            "[hlaont] update-anchors: ERROR: cd-hit-est not found on PATH.\n"
            "  Install it (e.g. `conda install -c bioconda cd-hit`, a package\n"
            "  manager of your choice, or your cluster's module system,\n"
            "  make sure it's on PATH, then re-run:\n"
            "    hlaont update-anchors\n",
            file=sys.stderr,
        )
        return 1

    print(f"[hlaont] update-anchors: staging candidates ({PREPARE_CANDIDATES_PY})", file=sys.stderr)
    subprocess.run([sys.executable, str(PREPARE_CANDIDATES_PY)], check=True)

    print(f"[hlaont] update-anchors: running cd-hit-est ({CDHIT_SCRIPT})", file=sys.stderr)
    _run_cdhit_commands(CDHIT_SCRIPT)

    print(f"[hlaont] update-anchors: consolidating anchor set ({ANCHORS_BUILD_PY})", file=sys.stderr)
    subprocess.run([sys.executable, str(ANCHORS_BUILD_PY)], check=True)

    print("\n[hlaont] update-anchors: anchors.fa.gz / anchors.meta.tsv.gz / "
          "anchors_manifest.json rebuilt. Check anchors_manifest.json's "
          "missing_target_anchors for any v0 target gene still uncovered.", file=sys.stderr)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    if (args.fastq is None) == (args.bam is None):
        sys.exit("[hlaont] ERROR: exactly one of the FASTQ positional argument or --bam is required")
    # PurePosixPath: Path() would collapse the "//" of a remote URL.
    if args.bam is not None and args.reference is None and PurePosixPath(args.bam).suffix.lower() == ".cram":
        sys.exit("[hlaont] ERROR: --reference is required for a .cram --bam input (CRAM cannot decode without it)")
    if args.phased and args.bam is None:
        sys.exit("[hlaont] ERROR: --phased requires --bam (DESIGN.md S10 -- phase tags are BAM/CRAM-only, the FASTQ path has none)")

    if args.fastq is not None and not args.fastq.exists():
        sys.exit(f"[hlaont] ERROR: {args.fastq} does not exist")
    # Remote --bam: the header read below reports connection errors.
    if args.bam is not None and not is_remote_bam(args.bam) and not Path(args.bam).exists():
        sys.exit(f"[hlaont] ERROR: {args.bam} does not exist")
    if args.reference is not None and not args.reference.exists():
        sys.exit(f"[hlaont] ERROR: {args.reference} does not exist")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Elapsed time only; there is no runtime model for an ETA.
    t0 = time.monotonic()

    def progress(msg: str) -> None:
        print(f"[hlaont] {msg} ({time.monotonic() - t0:.0f}s elapsed)", file=sys.stderr)

    mini_bam_path: Path | None = None
    control_depth: float | None = None
    control_n_reads = 0
    if args.bam is not None:
        fastq_path, mini_bam_path, region_set = _extract_bam_to_fastq(
            args.sample_id, args.bam, args.reference, args.out_dir,
            samtools_bin=args.samtools_bin, n_threads=args.n_threads, progress=progress,
        )
        # Read chr17 control depth off the mini-BAM rather than realigning
        # the whole FASTQ (that pass made --bam runs ~3x slower). Builds
        # without a chr17_control region fall back to the realignment.
        chr17_region = next(
            (r for r in region_set.regions if r.kind == "chr17_control"), None
        )
        if chr17_region is not None and chr17_region.start is not None and chr17_region.end is not None:
            progress("Stage 1: measuring chr17 control depth (from mini-BAM, no realignment)")
            control_depth, control_n_reads = compute_chr17_control_depth_from_bam(
                mini_bam_path, chr17_region.contig, chr17_region.start, chr17_region.end,
                samtools_bin=args.samtools_bin,
            )
    else:
        fastq_path = args.fastq

    # Pass only given overrides so run_sample()'s defaults apply otherwise.
    overrides = {
        k: v for k, v in {
            "pool_fasta": args.pool_fasta,
            "pool_meta": args.pool_meta,
            "models_path": args.models_path,
            "cds_catalogue_path": args.cds_catalogue_path,
            "c4_pool_fasta": args.c4_pool_fasta,
            "c4_pool_meta": args.c4_pool_meta,
            "genes_yaml": args.genes_yaml,
            "anchors_path": args.anchors_path,
            "anchors_meta_path": args.anchors_meta_path,
            "chr17_control_fasta": args.chr17_control_fasta,
            "c4_flanked_fasta": args.c4_flanked_fasta,
        }.items() if v is not None
    }
    if control_depth is not None:
        overrides["control_depth"] = control_depth
        overrides["control_n_reads"] = control_n_reads

    if args.phased:
        assert mini_bam_path is not None  # guaranteed by the --phased/--bam check above
        result = run_sample_phased(
            args.sample_id, fastq_path, mini_bam_path,
            n_threads=args.n_threads, minimap2_bin=args.minimap2_bin, samtools_bin=args.samtools_bin,
            progress=progress, **overrides,
        )
    else:
        result = run_sample(
            args.sample_id, fastq_path,
            n_threads=args.n_threads, minimap2_bin=args.minimap2_bin, samtools_bin=args.samtools_bin,
            progress=progress, **overrides,
        )

    # Sample-prefixed outputs so one --out-dir can hold several samples.
    prefix = args.sample_id
    calls_kwargs = {"pool_meta": overrides["pool_meta"]} if "pool_meta" in overrides else {}
    write_calls_tsv(result, args.out_dir / f"{prefix}.calls.tsv", **calls_kwargs)
    write_consensus_fasta(
        result, args.out_dir / f"{prefix}.consensus.fa", args.out_dir / f"{prefix}.consensus.qv.tsv"
    )
    vcf_kwargs = {"pool_fasta": overrides["pool_fasta"]} if "pool_fasta" in overrides else {}
    write_variants_vcf(result, args.out_dir / f"{prefix}.variants.vcf", **vcf_kwargs)
    write_qc_json(
        result, args.out_dir / f"{prefix}.qc.json",
        dorado_version=args.dorado_version,
        reference_build_md5=args.reference_build_md5,
    )

    n_genes = len({c.gene for c in result.haplotype_calls})
    print(
        f"[hlaont] {args.sample_id}: {len(result.haplotype_calls)} haplotype "
        f"call(s) across {n_genes} gene(s), {len(result.failures)} gene "
        f"failure(s) (see qc.json). Output written to {args.out_dir} "
        f"({time.monotonic() - t0:.0f}s total)",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest="command", required=True)
    _add_run_subparser(subparsers)
    _add_update_db_subparser(subparsers)
    _add_update_anchors_subparser(subparsers)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

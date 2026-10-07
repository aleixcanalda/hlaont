"""Stage 0 BAM/CRAM -> mini-BAM (DESIGN.md S3.1).

Extracts the RegionSet's regions plus all unmapped reads (`samtools view`
region "*") into a mini-BAM, then converts it to FASTQ for the same anchor
sieve as FASTQ input. ONT reads have no mates to chase.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from regions import Region, RegionSet

# Exclude secondary/supplementary so each read appears once, full length.
_SAMTOOLS_FASTQ_EXCLUDE_FLAGS = 0x900


def _samtools_region(region: Region) -> str:
    """samtools region string (1-based inclusive) for a 0-based half-open
    Region; whole contig if start/end are None."""
    if region.start is None:
        return region.contig
    return f"{region.contig}:{region.start + 1}-{region.end}"


def extract_mini_bam(
    input_path: Path | str,
    reference_path: Path | None,
    region_set: RegionSet,
    output_path: Path,
    *,
    samtools_bin: str = "samtools",
    n_threads: int = 1,
) -> None:
    """Write and index the mini-BAM: region_set's regions plus all unmapped
    reads. `input_path` must be indexed and may be an http(s) URL.
    `reference_path` is needed only for CRAM."""
    region_args = [_samtools_region(r) for r in region_set.regions]
    reference_args = ["-T", str(reference_path)] if reference_path is not None else []
    subprocess.run(
        [
            samtools_bin, "view", "-b",
            "-@", str(n_threads),
            *reference_args,
            "-o", str(output_path),
            str(input_path),
            *region_args, "*",
        ],
        check=True,
    )
    subprocess.run(
        [samtools_bin, "index", "-@", str(n_threads), str(output_path)],
        check=True,
    )


def mini_bam_to_fastq(
    mini_bam_path: Path,
    fastq_path: Path,
    *,
    samtools_bin: str = "samtools",
    n_threads: int = 1,
) -> None:
    """Mini-BAM -> FASTQ for sieve_fastq (primary records only)."""
    with open(fastq_path, "w") as out:
        subprocess.run(
            [
                samtools_bin, "fastq",
                "-@", str(n_threads),
                "-F", str(_SAMTOOLS_FASTQ_EXCLUDE_FLAGS),
                str(mini_bam_path),
            ],
            stdout=out,
            check=True,
        )

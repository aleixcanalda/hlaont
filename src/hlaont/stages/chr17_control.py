"""chr17 control-region depth (DESIGN.md S1/S2f), shared by cn.py and
c4_chr17_cn.py.

Mean per-base `samtools depth` over the segdup-free chr17 interval
(chr17:72,950,001-77,450,000, CHM13v2.0), either by aligning reads to the
shipped reference (resources/controls/chr17_control.chm13v2.fa.gz) or read
off a sample's own BAM.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from align_depth import (
    align_to_sorted_bam,
    filter_bam_by_min_aligned_length,
    first_contig_name,
    n_distinct_reads_over_region,
    sum_depth_over_region,
)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_CHR17_CONTROL_FASTA = REPO_ROOT / "resources" / "controls" / "chr17_control.chm13v2.fa.gz"

# Length of the shipped control region; overridable for test fixtures.
CHR17_CONTROL_LENGTH = 4_500_000

# Minimum aligned length per record. Realigning against the isolated
# control read ~19% deeper than the sample's own BAM, likely from reads
# sharing only a short repeat with the region. On HG00253 2kb brought the
# two paths within 1.4%; 1.5kb keeps a bit more signal. Provisional.
_MIN_ALIGNED_BP = 1_500


def compute_chr17_control_depth(
    fastq_path: Path,
    *,
    chr17_control_fasta: Path = DEFAULT_CHR17_CONTROL_FASTA,
    control_length: int = CHR17_CONTROL_LENGTH,
    min_aligned_bp: int = _MIN_ALIGNED_BP,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
) -> tuple[float, int]:
    """Mean control depth from aligning `fastq_path` to the shipped control
    reference, after the `min_aligned_bp` filter (no coverage-fraction
    filter at this scale). Returns (depth, n_reads). depth 0.0 means no
    control reads (e.g. a --bam FASTQ without chr17); callers must check."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_bam = Path(tmp) / "chr17_control.raw.bam"
        align_to_sorted_bam(
            fastq_path, chr17_control_fasta, raw_bam,
            n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
        )
        bam_path = Path(tmp) / "chr17_control.filtered.bam"
        filter_bam_by_min_aligned_length(
            raw_bam, bam_path, min_aligned_bp=min_aligned_bp, samtools_bin=samtools_bin,
        )
        contig = first_contig_name(bam_path)
        total_depth = sum_depth_over_region(bam_path, contig, 0, control_length, samtools_bin=samtools_bin)
        n_reads = n_distinct_reads_over_region(bam_path, contig, 0, control_length)
    return total_depth / control_length, n_reads


def compute_chr17_control_depth_from_bam(
    bam_path: Path,
    contig: str,
    start: int,
    end: int,
    *,
    min_aligned_bp: int = _MIN_ALIGNED_BP,
    samtools_bin: str = "samtools",
) -> tuple[float, int]:
    """Mean control depth read directly off an already-aligned,
    coordinate-sorted, indexed BAM (cli.py's --bam mini-BAM), with the same
    `min_aligned_bp` filter. Avoids a 95-300s realignment. The filter scans
    the whole file."""
    with tempfile.TemporaryDirectory() as tmp:
        filtered_bam = Path(tmp) / "chr17_control_from_bam.filtered.bam"
        filter_bam_by_min_aligned_length(
            bam_path, filtered_bam, min_aligned_bp=min_aligned_bp, samtools_bin=samtools_bin,
        )
        control_length = end - start
        total_depth = sum_depth_over_region(filtered_bam, contig, start, end, samtools_bin=samtools_bin)
        n_reads = n_distinct_reads_over_region(filtered_bam, contig, start, end)
    return total_depth / control_length, n_reads

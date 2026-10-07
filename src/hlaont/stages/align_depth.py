"""samtools-depth coverage on minimap2 alignments (DESIGN.md S1/S2f).

Align with `minimap2 -a` piped into `samtools sort`, index, optionally
filter records, then sum `samtools depth` over a region.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import pysam


def write_sequences_fasta(seqs: Sequence[tuple[str, str]], fasta_path: Path) -> None:
    """Write (record_id, sequence) pairs as FASTA."""
    with open(fasta_path, "w") as fh:
        for record_id, seq in seqs:
            fh.write(f">{record_id}\n{seq}\n")


def align_to_sorted_bam(
    query_fasta: Path,
    ref_fasta: Path,
    out_bam: Path,
    *,
    n_best: int | None = None,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
) -> Path:
    """`minimap2 -a -x map-ont [-N n_best] ref query | samtools sort`, then
    index. Uses the standard ONT preset; FuFiHLA's scoring string gave no
    better depth here. Leave `n_best` None (minimap2's -N 5): -N 256 made
    the chr17 control alignment ~3x slower with no benefit."""
    minimap2_cmd = [minimap2_bin, "-t", str(n_threads), "-a", "-x", "map-ont"]
    if n_best is not None:
        minimap2_cmd += ["-N", str(n_best)]
    minimap2_cmd += [str(ref_fasta), str(query_fasta)]
    sort_cmd = [samtools_bin, "sort", "-@", str(n_threads), "-o", str(out_bam), "-"]
    mm2 = subprocess.Popen(minimap2_cmd, stdout=subprocess.PIPE)
    assert mm2.stdout is not None
    subprocess.run(sort_cmd, stdin=mm2.stdout, check=True)
    mm2.stdout.close()
    if mm2.wait() != 0:
        raise RuntimeError(f"minimap2 failed (exit {mm2.returncode}): {' '.join(minimap2_cmd)}")
    subprocess.run([samtools_bin, "index", str(out_bam)], check=True)
    return out_bam


def sort_and_index_bam(bam_path: Path, out_bam: Path, *, samtools_bin: str = "samtools") -> Path:
    """Re-sort and index a BAM (e.g. after a pysam rewrite)."""
    subprocess.run([samtools_bin, "sort", "-o", str(out_bam), str(bam_path)], check=True)
    subprocess.run([samtools_bin, "index", str(out_bam)], check=True)
    return out_bam


def filter_bam_by_target_coverage(
    bam_path: Path,
    out_bam: Path,
    *,
    min_target_coverage: float,
    samtools_bin: str = "samtools",
) -> Path:
    """Drop records covering < `min_target_coverage` of their contig.
    For gene-scale targets only, not the chr17 control."""
    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in:
        lengths = dict(zip(bam_in.references, bam_in.lengths))
        with pysam.AlignmentFile(str(out_bam), "wb", template=bam_in) as bam_out:
            for read in bam_in.fetch(until_eof=True):
                if read.is_unmapped or read.reference_name is None:
                    continue
                target_length = lengths[read.reference_name]
                if not target_length or read.reference_length is None:
                    continue
                if read.reference_length / target_length < min_target_coverage:
                    continue
                bam_out.write(read)
    subprocess.run([samtools_bin, "index", str(out_bam)], check=True)
    return out_bam


def filter_bam_by_min_aligned_length(
    bam_path: Path,
    out_bam: Path,
    *,
    min_aligned_bp: int,
    samtools_bin: str = "samtools",
) -> Path:
    """Drop records whose reference span is < `min_aligned_bp`: removes
    short repeat-driven matches on a multi-megabase target, where a
    coverage fraction is meaningless."""
    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in:
        with pysam.AlignmentFile(str(out_bam), "wb", template=bam_in) as bam_out:
            for read in bam_in.fetch(until_eof=True):
                if read.is_unmapped or read.reference_length is None:
                    continue
                if read.reference_length < min_aligned_bp:
                    continue
                bam_out.write(read)
    subprocess.run([samtools_bin, "index", str(out_bam)], check=True)
    return out_bam


def sum_depth_over_region(
    bam_path: Path, contig: str, start: int, end: int, *, samtools_bin: str = "samtools",
) -> int:
    """Sum of `samtools depth` over [start, end) (0-based half-open). Divide
    by (end - start) for mean depth; uncovered positions count as 0."""
    if end <= start:
        return 0
    region = f"{contig}:{start + 1}-{end}"
    proc = subprocess.run(
        [samtools_bin, "depth", "-r", region, str(bam_path)],
        capture_output=True, text=True, check=True,
    )
    return sum(int(line.split("\t")[2]) for line in proc.stdout.splitlines() if line)


def n_distinct_reads_over_region(bam_path: Path, contig: str, start: int, end: int) -> int:
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        return len({r.query_name for r in bam.fetch(contig, start, end)})


def first_contig_name(bam_path: Path) -> str:
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        if not bam.references:
            raise ValueError(f"{bam_path} has no reference contigs in its header")
        return bam.references[0]

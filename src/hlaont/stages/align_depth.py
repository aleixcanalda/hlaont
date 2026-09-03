"""Shared real-alignment, samtools-depth-based coverage measurement
(DESIGN.md S1/S2f).

Real per-base coverage (`samtools depth` on a real, sorted+indexed BAM)
rather than a PAF-derived aligned-bases approximation or a distinct-read-
count proxy -- confirmed 2026-08-31/09-01 (the development notes) as the
most reliable depth signal this project has tried. User-directed,
2026-09-01: this is now the shared measurement method for the chr17
control region (`stages/chr17_control.py`), C4's own per-form CN
(`stages/c4_chr17_cn.py`), and the classical/pseudogene HLA CN numerator
itself (`stages/cn.py`'s `stage0_depth_by_gene()`) -- one implementation,
not three independent minimap2/samtools-invocation copies.

Every caller: write query sequences to FASTA (if not already a file on
disk), align with `minimap2 -a` (real SAM -- `samtools depth` needs a real
BAM, not a PAF line), pipe straight into `samtools sort` (never materialise
the unsorted SAM), index, optionally drop alignment records that cover too
little of their own target contig (meaningful for anchor/gene-scale
targets, a few kb to ~20kb -- NOT meaningful for the whole-chromosome-scale
chr17 control region, where no single ONT read could ever clear a 10% bar
of a 4.5Mb target, so callers measuring that region skip the filter), then
sum real `samtools depth` output over the region of interest.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import pysam


def write_sequences_fasta(seqs: Sequence[tuple[str, str]], fasta_path: Path) -> None:
    """seqs: (record_id, sequence) pairs -- e.g. Stage 0's own CandidateRead
    read_id/sequence, written the same bare-FASTA way assign.py's own
    (private) _write_reads_fasta() already does for its own minimap2 call;
    duplicated rather than imported to keep this module decoupled from
    assign.py, same reasoning cn.py's own length-correction copy already
    documents for a different function."""
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
    """`minimap2 -a -x map-ont [-N n_best] ref query | samtools sort -o
    out_bam -`, then `samtools index`. Standard, un-tuned ONT preset --
    deliberately not this project's own FuFiHLA-ported scoring string
    (assign.py's `_MINIMAP2_SCORING_ARGS`); real-data finding
    (`stages/c4_chr17_cn.py`'s own docstring/the development notes): the
    harsh scoring string gave no better recruited depth against this kind
    of whole-region/whole-anchor-pool reference than the standard preset.
    Invariant 11 -- a fresh choice, not an unexamined port.

    **`n_best=None` (minimap2's own default `-N` of 5), reverted back from
    256 the same day it was raised.** A real bug on this project's very
    first real-sample validation run (2026-09-01) needed a higher `-N`:
    this function used to also be the classical/pseudogene-gene CN
    numerator's own alignment, a *single* pass against the *entire*
    ~187-contig anchors.fa.gz at once, where a gene with several real
    target-role anchors (HLA-A alone has 9) competing against every other
    gene's anchors in the same alignment could silently lose any read
    whose true best match didn't crack the global top-5. That caller was
    removed the same day (`stages/cn.py`'s `stage0_depth_by_gene()` now
    reconstructs depth from Stage 0's own sieve alignment instead of
    calling this function at all), so the elevated default no longer has
    a beneficiary -- and, confirmed directly on a second real-sample run,
    it actively hurt chr17-control's own alignment (a single, real,
    repetitive-content 4.5Mb genomic region, not a curated low-repeat
    anchor set): real time 94.995s -> 299.350s for the identical input,
    since a higher `-N` makes minimap2 chain more candidate secondary
    loci per read within that one contig's own repeat content. Neither
    chr17-control (1 contig) nor C4 (4 contigs, real-data-confirmed
    unaffected either way -- 4.394s vs 4.344s) needs more than minimap2's
    own default, so this reverts to not passing `-N` at all rather than
    silently keeping an elevated value with no remaining justification."""
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
    """Re-sort+index an existing BAM -- used after a pysam-level rewrite
    (e.g. `stages/cn.py`'s per-gene alignment dedup) that does not
    guarantee coordinate order, since `samtools depth -r` needs a real
    sorted+indexed BAM."""
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
    """Drop alignment records covering less than `min_target_coverage` of
    their own reference contig's full length -- mirrors Stage 0's own
    `min_anchor_coverage` (`input/fastq.py`'s `SieveParams`,
    `resources/models/stage0_fastq_sieve.yaml`), so a short/spurious
    partial match doesn't count toward depth here either. Module docstring:
    only meaningful for anchor/gene-scale targets, never applied to the
    chr17 control region."""
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
    """Drop alignment records whose own `reference_length` (CIGAR-derived
    span on the reference, matching `filter_bam_by_target_coverage()`'s own
    definition) is below an absolute floor -- 2026-09-01, user-directed,
    real chr17-control finding: `filter_bam_by_target_coverage()`'s
    coverage-*fraction* bar is meaningless at chromosome scale (module
    docstring), but a real read that only shares a short transposable
    element or other common repeat with the control region can still
    produce a short, clean (low-mismatch) partial alignment there even
    when it truly belongs elsewhere in the genome -- a coverage-fraction
    or mismatch-rate filter would not catch this (the matched span itself
    can look like a perfectly clean match), but an absolute length floor
    does, since a genuinely on-target ONT read aligns for its whole real
    length, not just a repeat element's span. Distinct from
    `filter_bam_by_target_coverage()` (a *fraction* of the target's own
    length, meaningful for anchor/gene-scale targets) -- this is an
    *absolute* base-pair floor, the right shape of bar for a
    multi-megabase target where no single read could ever clear a
    meaningful fraction of it. `min_aligned_bp` is caller-supplied, not a
    module constant -- provisional (invariant 11), not yet validated
    against real data at any specific value."""
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
    """Sum of real `samtools depth`'s 3rd column over [start, end)
    (0-based half-open, converted to samtools' 1-based inclusive region
    string). The caller divides by (end - start) for a mean depth over the
    FULL region including positions with zero reported coverage --
    `samtools depth` without `-a` only prints covered positions; summing
    those and dividing by the known full region length gives the identical
    mean without paying to materialise every zero-depth line."""
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

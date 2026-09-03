"""Shared chr17 control-region depth measurement (DESIGN.md S1/S2f).

Real per-base coverage (`samtools depth` on a real, sorted+indexed BAM,
`stages/align_depth.py`) against the shipped, reference-only chr17 control
region (`resources/controls/chr17_control.chm13v2.fa.gz`,
`resources/controls/build.py`) -- see that module's docstring for the full
locityper-derived rationale (a real, external, structurally-boring
background interval, chr17:72,950,001-77,450,000 CHM13v2.0, used in place
of an HLA-gene calibrator panel). Deliberately reference-shipped, not
BAM-derived at runtime: a sample's own reads get aligned against this
shipped reference with a plain minimap2 call, no BAM/external reference
needed, matching the FASTQ-only critical path.

Extracted 2026-09-01 from `stages/c4_chr17_cn.py` (the module that
introduced this control for C4's own total-CN readout) once
`stages/cn.py`'s classical/pseudogene CN also needed the identical
control-depth measurement, user-directed -- one shared implementation, not
two independent copies.

**Revised 2026-09-01, same session, user-directed: `samtools depth`, not a
PAF-derived aligned-bases sum.** The first version of this module
(extracted verbatim from the original, never-executed `c4_chr17_cn.py`
code) shelled out to `minimap2 -ax map-ont ... -c` and parsed PAF -- a real
bug, caught the moment this module was first actually run (test_cli.py's
own end-to-end CLI test): `-a` forces SAM output and minimap2 silently
ignores `-c` (documented as PAF-only) once `-a` is set, so every "PAF"
line was really a SAM line, and `best_hit_per_read()` crashed on the first
`@HD` header row. Rather than just dropping `-a` to get real PAF, the user
redirected to the more reliable method this project's own C4 investigation
already found (the development notes: "the single most reliable number
produced all session") -- real `samtools depth` over a real BAM -- and
extended it to be the shared method for chr17, C4, and the classical-gene
CN numerator alike (`stages/align_depth.py`, `stages/c4_chr17_cn.py`,
`stages/cn.py`).
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

# Real length of the shipped control region (resources/controls/build.py's
# own real output) -- not re-derived from the FASTA at runtime, same
# "fixed, versioned resource" convention as c4_chr17_cn.py's own
# _GENE_BODY_LENGTH (that module's docstring has the full rationale).
# Overridable so a test can point at a small synthetic control fasta
# without lying about a 4.5Mb region.
CHR17_CONTROL_LENGTH = 4_500_000

# 2026-09-01, user-directed, real HG00253 finding: realigning against the
# isolated chr17_control reference (compute_chr17_control_depth(), below)
# measured ~19% higher depth than reading the sample's own real
# whole-genome alignment directly (compute_chr17_control_depth_from_bam());
# the leading hypothesis is reads that only share a short transposable
# element or other common repeat with the control region getting force-
# mapped there in the isolated realignment, since there's no rest-of-genome
# for them to lose to. A coverage-*fraction* filter is meaningless at this
# scale (module docstring below) and a mismatch-rate filter would not catch
# this either (the repeat-driven match itself can be clean) -- an absolute
# aligned-length floor does, since a genuinely on-target ONT read aligns
# for its whole real length, not just a repeat element's span.
# `align_depth.filter_bam_by_min_aligned_length()`'s own docstring has the
# full mechanism. Provisional (invariant 11), revised same day: real
# HG00253 data swept 1/2/5kb on both the whole-FASTQ and from-BAM paths --
# whole-FASTQ vs from-BAM converged tightest at 2kb (30.92 vs 31.35, ~1.4%
# apart) while discarding much less real signal than 5kb did (5kb dropped
# ~33%/17% of the unfiltered whole-FASTQ/from-BAM depth respectively, 2kb
# only ~22%/6%) -- 5kb was over-correcting. User's own final choice, 1.5kb,
# splits the difference toward keeping more real signal than 2kb already
# does; not itself re-measured at this exact value, still provisional.
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
    """Real mean per-base coverage (`samtools depth`, summed over the
    control region and divided by its own known length) against the
    shipped chr17 control reference -- one whole-FASTQ alignment, sorted
    into a real BAM, independent of Stage 0's gene-anchor sieve (a read
    supporting the chr17 control has nothing to do with any HLA/C4 anchor
    and would never pass that sieve). No target-*coverage-fraction*
    pre-filter here (module docstring -- meaningless at chromosome scale).
    **`min_aligned_bp`, 2026-09-01, user-directed**: an absolute
    aligned-length floor IS applied (`filter_bam_by_min_aligned_length()`)
    -- a different, chromosome-scale-appropriate filter shape than the
    fraction-based one this paragraph rules out; see the module-level
    `_MIN_ALIGNED_BP` comment for the real finding that motivated it.
    Returns (depth, n_reads_aligned). `depth == 0.0` (no chr17 signal in
    this input -- e.g. a `--bam`-derived FASTQ that never pulled chr17
    reads, DESIGN.md S1's own documented gap) is a real, loud value
    callers must check for, not silently treated as a normal control
    depth."""
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
    """Real mean per-base coverage read *directly* off an already-aligned
    BAM's own chr17-control region -- no realignment against the shipped
    reference at all. User-directed 2026-09-01, real-data-motivated: the
    whole-FASTQ realignment `compute_chr17_control_depth()` does is
    necessary for raw-FASTQ input (no prior alignment exists to reuse),
    but for a `--bam`/`--cram` sample the mini-BAM extraction
    (`input/bam.py`'s `extract_mini_bam()`, `input/regions.py`'s own
    `chr17_control`-kind `Region`, added the same session) already pulled
    exactly this region using the *sample's own real alignment*, which is
    both more accurate (the sample's own real mapping, not a fresh,
    generically-tuned realignment) and dramatically cheaper -- a real
    HG00253 measurement found the realignment pass alone cost 95-300s
    (`stages/align_depth.py`'s own docstring has the `-N`-driven variance)
    against a whole-sample FASTQ of tens of thousands of reads, almost all
    of which have nothing to do with chr17 at all.

    **`min_aligned_bp`, same day, real HG00253 cross-check:** even though
    this path reads the sample's own real whole-genome alignment (already
    far less prone to the isolated-reference force-mapping
    `compute_chr17_control_depth()`'s own docstring names), a real
    aligner can still occasionally place a short repeat-driven partial
    match here over a marginally-worse true location elsewhere -- applying
    the identical absolute-length floor keeps both paths honest against
    the same failure mode rather than assuming the real alignment is
    immune to it. `bam_path` must already be coordinate-sorted and
    indexed (true of `extract_mini_bam()`'s own output); the length filter
    scans the whole file (`filter_bam_by_min_aligned_length()`), fine for
    the small, already-region-restricted mini-BAM this is normally called
    with, but a caller passing a full whole-genome BAM directly should
    expect a full-file scan, not a region-restricted one."""
    with tempfile.TemporaryDirectory() as tmp:
        filtered_bam = Path(tmp) / "chr17_control_from_bam.filtered.bam"
        filter_bam_by_min_aligned_length(
            bam_path, filtered_bam, min_aligned_bp=min_aligned_bp, samtools_bin=samtools_bin,
        )
        control_length = end - start
        total_depth = sum_depth_over_region(filtered_bam, contig, start, end, samtools_bin=samtools_bin)
        n_reads = n_distinct_reads_over_region(filtered_bam, contig, start, end)
    return total_depth / control_length, n_reads

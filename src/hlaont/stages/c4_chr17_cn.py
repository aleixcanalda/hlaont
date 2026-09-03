"""C4A/C4B copy number via a real external control region (DESIGN.md S2f),
added 2026-08-31/09-01 session, user-directed.

**Why this exists alongside `c4_cn.py`'s own, older total-CN mechanism,
not instead of it (yet):** `c4_cn.py`'s calibrator-panel-normalised total
undershot real truth badly in at least one real sample (HG00253: 2.00
against a true 4) -- a long, real-data-driven investigation this session
(the development notes' own session log has the full derivation, do not re-derive it
here) traced this to the calibrator panel itself being a noisy, HLA-gene-
based baseline, not a fundamentally different question. The user's own
fix: normalise against a real, external, structurally boring control
region instead -- locityper's own published segdup-free background
interval, chr17:72,950,001-77,450,000 (CHM13v2.0) -- using real per-base
`samtools depth` coverage (module docstring below) for both the control
and C4 itself.

**Deliberately reference-shipped, not BAM-derived at runtime.** Early
versions of this investigation pulled the control region's reads straight
from a sample's own BAM via `samtools depth` -- accurate, but BAM-only,
breaking the FASTQ-only critical path this project has prioritised since
DESIGN.md's own Status section. The fix, user-directed: `resources/controls/
build.py` extracts the real chr17 interval AND C4's own real genomic
sequence (both forms, both genes, each with real +-3kb flanking sequence)
*once*, from a real CHM13v2.0 reference, and ships them as ordinary
resource files (`chr17_control.chm13v2.fa.gz`, `c4_flanked_forms.chm13v2.
fa.gz`) -- exactly the same shape as `resources/anchors/anchors.fa.gz`.
A sample's own reads get aligned against these shipped references with a
plain minimap2 call, same as Stage 0 already does against the anchor set;
no BAM, no external reference, no per-sample coordinate lookup needed at
runtime. This is why this module takes a FASTQ path, not a `CandidateRead`
list or a `GeneAssignmentMatrix` -- it does its own alignment pass,
independent of Stage 0's gene-anchor sieve (a read that only overlaps the
chr17 control region would never pass that sieve at all, since it has
nothing to do with any HLA/C4 anchor).

**Four forms, one combined alignment, real HG00253 validation (this
session's own investigation, not re-derived here):** C4A and C4B share
most of their sequence with their own long/short form and with each
other, so aligning against a single combined reference (not four separate
per-form runs) is what lets minimap2's own primary-alignment resolution
correctly split a read across the four competing targets instead of
double-crediting it. Real HG00253 result (`+-3kb` flanks, the validated
choice -- `+-6kb` was tried and rejected, see `resources/controls/
build.py`'s own docstring), from the original PAF-based measurement this
module used through the first 2026-09-01 revision (module docstring
below has the real bug that motivated moving off it -- the *shape* of
this finding, not the exact numbers, is what still applies): C4AL depth
18.21 (truth: 2 real copies), C4BL 15.29 (truth: 1), C4A-short 5.04
(truth: 0), C4B-short 12.00 (truth: 1) against a control depth of 33.41 --
the *relative* pattern across forms was qualitatively right (C4A-long >>
C4A-short, C4B-long ~= C4B-short) even though the *total* still undershot
truth by ~25%, consistent with a real but not-yet-root-caused uniform
recruitment shortfall, not a form-specific error. Flagged, not hidden
(invariant 11): not yet re-validated against real data under the
samtools-depth revision below.

**Revised 2026-09-01, same session, user-directed: real `samtools depth`
on a real BAM, not a PAF-derived aligned-bases sum.** The original version
of this module (written, per the development notes' own handoff note, without ever
actually being run) shelled out to `minimap2 -ax map-ont ... -c` and
parsed PAF -- a real bug, caught the moment this module was first actually
executed (test_cli.py's own end-to-end CLI test, which exercises the
sibling `stages/chr17_control.py` module this one calls into): `-a` forces
SAM output and minimap2 silently ignores `-c` (documented as PAF-only)
once `-a` is set, so every "PAF" line was really a SAM line. Rather than
just fixing the flag to get real PAF back, the user redirected to the
more reliable method this project's own C4 investigation already found
(the development notes: "the single most reliable number produced all
session") -- real `samtools depth` over a real, sorted+indexed BAM
(`stages/align_depth.py`) -- and extended it to be the shared method for
chr17 (`stages/chr17_control.py`), C4 (this module), and the classical-
gene CN numerator (`stages/cn.py`) alike. `n_reads`/`depth` per form now
come from real `samtools depth`/distinct-read-name counts over the body
span of a real, sorted BAM (built once against the combined four-form
reference), not PAF-line overlap arithmetic -- same semantics, more
reliable measurement.

**Coverage pre-filter: none, by design (`min_target_coverage=0.0`).** A
version added the same revision this module first shipped with mirrored
Stage 0's own `min_anchor_coverage` (0.1) here -- reverted 2026-09-01,
same day, user-directed: this project's own real C4 investigation
(module docstring above) already found a coverage bar actively harmful
for C4 recruitment specifically, so re-applying one here contradicted a
lesson this module's own docstring already records. `0.0` makes
`filter_bam_by_target_coverage()`'s `< min_target_coverage` check never
fire, i.e. every mapped read is kept regardless of how little of its own
flanked-form contig it covers.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from align_depth import align_to_sorted_bam, filter_bam_by_target_coverage, n_distinct_reads_over_region, sum_depth_over_region
from chr17_control import CHR17_CONTROL_LENGTH, DEFAULT_CHR17_CONTROL_FASTA, compute_chr17_control_depth

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_C4_FLANKED_FASTA = REPO_ROOT / "resources" / "controls" / "c4_flanked_forms.chm13v2.fa.gz"

FLANK_BP = 3_000

# gene-body length per form (total sequence length minus the two 3kb
# flanks) -- must match resources/controls/build.py's own real output
# exactly; not re-derived from the FASTA at runtime since these are fixed,
# versioned resources, same convention as assign.py's own DEFAULT_POOL_*
# constants not being re-measured on every call.
_GENE_BODY_LENGTH = {
    "C4AL": 20_624,
    "C4BL": 20_624,
    "C4AS": 14_257,
    "C4BS": 14_258,
}

# 2026-09-01, user-directed: no coverage pre-filter for C4 at all (0.0 --
# `filter_bam_by_target_coverage()`'s own `< min_target_coverage` check
# never fires at 0.0, so every mapped read passes). Deliberately NOT
# mirroring Stage 0's own `min_anchor_coverage` (0.1) the way this used to
# -- last session's own real HG00253 investigation (module docstring
# above) already found a coverage bar actively harmful for C4 recruitment
# specifically (real data: lifting Stage 0's sieve bar entirely, before
# the flanking fix even existed, nearly doubled real recruited reads with
# no corresponding drop in signal quality) -- so this module keeps that
# lesson rather than re-applying a bar the C4 investigation itself showed
# doesn't belong here.
_DEFAULT_MIN_TARGET_COVERAGE = 0.0


@dataclass(frozen=True)
class C4FormCN:
    form: str  # "C4AL" | "C4AS" | "C4BL" | "C4BS"
    n_reads: int
    depth: float  # real samtools-depth mean coverage over the form's own gene body
    cn_estimate: float | None  # 2 * depth / control_depth; None if control_depth is 0


@dataclass(frozen=True)
class C4Chr17ControlResult:
    control_depth: float
    control_n_reads: int
    forms: dict[str, C4FormCN]  # keyed "C4AL"/"C4AS"/"C4BL"/"C4BS"
    total_cn_estimate: float | None  # sum of the four forms' own cn_estimate


def estimate_c4_cn_via_chr17_control(
    fastq_path: Path,
    *,
    chr17_control_fasta: Path = DEFAULT_CHR17_CONTROL_FASTA,
    c4_flanked_fasta: Path = DEFAULT_C4_FLANKED_FASTA,
    control_depth: float | None = None,
    control_n_reads: int = 0,
    min_target_coverage: float = _DEFAULT_MIN_TARGET_COVERAGE,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
) -> C4Chr17ControlResult:
    """Real samtools-depth-based C4 copy number (module docstring). Two
    independent alignment passes over the same `fastq_path` -- one against
    the shipped chr17 control region (`chr17_control.
    compute_chr17_control_depth()`), one against the shipped four-form C4
    reference -- neither reuses Stage 0's gene-anchor sieve (a read
    supporting the chr17 control has nothing to do with any HLA/C4 anchor
    and would never pass that sieve; C4 has no Stage 0 anchor at all, so
    there is no sieve output to reuse the way `stages/cn.py`'s classical-
    gene depth now does). `min_target_coverage` defaults to
    `_DEFAULT_MIN_TARGET_COVERAGE` (0.0, module docstring) -- pass an
    explicit float to override for a specific investigation.

    `control_depth`/`control_n_reads`, added 2026-09-01 once
    `stages/pipeline.py` started calling this alongside `stages/cn.py`'s
    own classical-gene CN in the same sample run: if `control_depth` is
    given (not `None`), it's used directly instead of this function
    computing its own separate chr17-control alignment pass -- avoids a
    redundant whole-FASTQ minimap2 run when the caller already has one.
    Standalone/test callers that don't pass it get this function's
    original, fully self-contained behaviour."""
    if control_depth is None:
        control_depth, control_n_reads = compute_chr17_control_depth(
            fastq_path, chr17_control_fasta=chr17_control_fasta,
            control_length=CHR17_CONTROL_LENGTH, n_threads=n_threads,
            minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
        )

    with tempfile.TemporaryDirectory() as tmp:
        raw_bam = Path(tmp) / "c4_raw.bam"
        align_to_sorted_bam(
            fastq_path, c4_flanked_fasta, raw_bam,
            n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
        )
        filtered_bam = Path(tmp) / "c4_filtered.bam"
        filter_bam_by_target_coverage(
            raw_bam, filtered_bam, min_target_coverage=min_target_coverage, samtools_bin=samtools_bin,
        )

        forms: dict[str, C4FormCN] = {}
        total_cn = 0.0
        for form, body_length in _GENE_BODY_LENGTH.items():
            contig = f"{form}_flanked_{FLANK_BP}bp"
            body_start, body_end = FLANK_BP, FLANK_BP + body_length
            total_depth = sum_depth_over_region(filtered_bam, contig, body_start, body_end, samtools_bin=samtools_bin)
            depth = total_depth / body_length
            n_reads = n_distinct_reads_over_region(filtered_bam, contig, body_start, body_end)
            cn = 2.0 * depth / control_depth if control_depth > 0 else None
            forms[form] = C4FormCN(form=form, n_reads=n_reads, depth=depth, cn_estimate=cn)
            if cn is not None:
                total_cn += cn

    return C4Chr17ControlResult(
        control_depth=control_depth,
        control_n_reads=control_n_reads,
        forms=forms,
        total_cn_estimate=total_cn if control_depth > 0 else None,
    )

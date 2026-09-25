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

**Revised 2026-09-25, user-directed: count-based total, joint integer
call.** Real benchmark finding (hlaont_bench/c4_cn_improve_20260925/, 26
runs, 13 samples x 15x/30x, Immuannot truth): the depth-based total above
over-reads by ~10% (est/truth 1.105), and its per-form rounding got all four
forms right in 14/26 runs. Now:
- *Total CN* = distinct reads overlapping the 5' window shared by all four
  forms (gene-body start -> HERV breakpoint), divided by the reads expected
  per copy from the chr17 control's own read count and mean read length
  (`control_n_reads`, `control_depth`). Unbiased on the benchmark (1.012) with
  no calibration constant. The 3' window was tighter but biased +21% for an
  unexplained reason, so it is not used.
- *Per-form `cn_estimate`* = that total times the form's share of the summed
  per-form depth, so depth only sets proportions, never the total. The old
  depth-based values stay on `depth_cn_estimate`/`depth_total_cn_estimate`.
- *HERV-spanning reads*: reads spanning the whole long-form HERV vs the
  short-form breakpoint, with read-length exposure, as long/short evidence.
- `call_c4_forms_jointly()`: integer AL/AS/BL/BS call (17/26 all-four-correct
  vs 14/26) from those signals plus Phase 2's A/B vote counts, exact over all
  non-negative integers (no copy-number cap), with a GQ.
Falls back to the depth-based numbers, and no joint call, when
`control_n_reads` is 0 (a caller that passed `control_depth` alone).
"""
from __future__ import annotations

import itertools
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pysam
import yaml

from align_depth import align_to_sorted_bam, filter_bam_by_target_coverage, n_distinct_reads_over_region, sum_depth_over_region
from chr17_control import CHR17_CONTROL_LENGTH, DEFAULT_CHR17_CONTROL_FASTA, compute_chr17_control_depth

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_C4_FLANKED_FASTA = REPO_ROOT / "resources" / "controls" / "c4_flanked_forms.chm13v2.fa.gz"
DEFAULT_C4_JOINT_CALL_PARAMS = REPO_ROOT / "resources" / "models" / "c4_joint_call.yaml"
FORMS = ("C4AL", "C4AS", "C4BL", "C4BS")

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
class C4JointCallParams:
    herv_start: int
    herv_end: int
    herv_anchor_bp: int
    prob_floor: float
    zero_cn_mean_frac: float
    form_proportion_sd: float
    gq_cap: float


def load_c4_joint_call_params(path: Path = DEFAULT_C4_JOINT_CALL_PARAMS) -> C4JointCallParams:
    doc = yaml.safe_load(path.read_text())
    jc = doc["joint_call"]
    return C4JointCallParams(
        herv_start=int(doc["herv_start"]), herv_end=int(doc["herv_end"]),
        herv_anchor_bp=int(doc["herv_anchor_bp"]),
        prob_floor=float(jc["prob_floor"]), zero_cn_mean_frac=float(jc["zero_cn_mean_frac"]),
        form_proportion_sd=float(jc["form_proportion_sd"]), gq_cap=float(jc["gq_cap"]),
    )


@dataclass(frozen=True)
class C4FormCN:
    form: str  # "C4AL" | "C4AS" | "C4BL" | "C4BS"
    n_reads: int
    depth: float  # real samtools-depth mean coverage over the form's own gene body
    # count-based total x this form's depth share (module docstring, 2026-09-25);
    # falls back to depth_cn_estimate when the count-based total is unavailable
    cn_estimate: float | None
    depth_cn_estimate: float | None = None  # 2 * depth / control_depth; None if control_depth is 0


@dataclass(frozen=True)
class C4Chr17ControlResult:
    control_depth: float
    control_n_reads: int
    forms: dict[str, C4FormCN]  # keyed "C4AL"/"C4AS"/"C4BL"/"C4BS"
    total_cn_estimate: float | None  # sum of the four forms' own cn_estimate
    depth_total_cn_estimate: float | None = None  # sum of depth_cn_estimate (pre-2026-09-25 total)
    n_reads_5p_window: int = 0  # distinct reads over the shared 5' window, all four forms
    reads_per_copy: float | None = None  # expected n_reads_5p_window per copy; None without control_n_reads
    herv_long_spanning: int = 0  # primary alignments spanning the whole HERV on a long form
    herv_short_spanning: int = 0  # primary alignments spanning the breakpoint on a short form
    herv_long_exposure: float = 0.0  # sum over primary reads of max(0, len - long window)
    herv_short_exposure: float = 0.0  # same, short window


@dataclass(frozen=True)
class C4JointCall:
    forms: dict[str, int]  # keyed "C4AL"/"C4AS"/"C4BL"/"C4BS"
    total: int
    gq: float  # phred posterior of the best call vs all others, flat prior, capped
    log_likelihood: float


def _contig(form: str) -> str:
    return f"{form}_flanked_{FLANK_BP}bp"


def _read_signals(bam_path: Path, params: C4JointCallParams) -> tuple[int, dict[str, float]]:
    """Distinct reads over the 5' window shared by all four forms (gene-body
    start -> HERV breakpoint, same coordinates on every contig), and HERV-
    spanning primary alignments with their read-length exposure: a read of
    length l can span a window w from max(0, l - w) start positions, so
    long-form spans are rarer and must be weighted, not counted raw."""
    w_long = params.herv_end - params.herv_start + 2 * params.herv_anchor_bp
    w_short = 2 * params.herv_anchor_bp
    left = params.herv_start - params.herv_anchor_bp
    names: set[str] = set()
    herv = {"long": 0, "short": 0, "long_exp": 0.0, "short_exp": 0.0}
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for form in FORMS:
            for r in bam.fetch(_contig(form)):
                s, e = r.reference_start, r.reference_end
                if s < params.herv_start and e > FLANK_BP:
                    names.add(r.query_name)
                if r.is_secondary or r.is_supplementary:
                    continue
                ql = r.infer_read_length() or 0
                herv["long_exp"] += max(0, ql - w_long)
                herv["short_exp"] += max(0, ql - w_short)
                if s <= left:
                    if form.endswith("L") and e >= params.herv_end + params.herv_anchor_bp:
                        herv["long"] += 1
                    elif form.endswith("S") and e >= params.herv_start + params.herv_anchor_bp:
                        herv["short"] += 1
    return len(names), herv


def estimate_c4_cn_via_chr17_control(
    fastq_path: Path,
    *,
    chr17_control_fasta: Path = DEFAULT_CHR17_CONTROL_FASTA,
    c4_flanked_fasta: Path = DEFAULT_C4_FLANKED_FASTA,
    control_depth: float | None = None,
    control_n_reads: int = 0,
    min_target_coverage: float = _DEFAULT_MIN_TARGET_COVERAGE,
    joint_call_params: C4JointCallParams | None = None,
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

        per_form: dict[str, tuple[int, float, float | None]] = {}
        for form, body_length in _GENE_BODY_LENGTH.items():
            contig = _contig(form)
            body_start, body_end = FLANK_BP, FLANK_BP + body_length
            total_depth = sum_depth_over_region(filtered_bam, contig, body_start, body_end, samtools_bin=samtools_bin)
            depth = total_depth / body_length
            n_reads = n_distinct_reads_over_region(filtered_bam, contig, body_start, body_end)
            cn = 2.0 * depth / control_depth if control_depth > 0 else None
            per_form[form] = (n_reads, depth, cn)
        params = joint_call_params or load_c4_joint_call_params()
        n_5p, herv = _read_signals(filtered_bam, params)

    depth_total = sum(cn for _, _, cn in per_form.values() if cn is not None) if control_depth > 0 else None
    # Count-based total (module docstring, 2026-09-25): reads expected per
    # copy over a window of length b = N_ctrl * (b + mean_len) / (L_ctrl +
    # mean_len) / 2, mean_len being the control's own mean read length.
    reads_per_copy = None
    if control_depth > 0 and control_n_reads > 0:
        mean_len = control_depth * CHR17_CONTROL_LENGTH / control_n_reads
        b = params.herv_start - FLANK_BP
        reads_per_copy = control_n_reads * (b + mean_len) / (CHR17_CONTROL_LENGTH + mean_len) / 2.0
    count_total = None if reads_per_copy is None else n_5p / reads_per_copy

    forms: dict[str, C4FormCN] = {}
    for form, (n_reads, depth, cn) in per_form.items():
        if count_total is not None and depth_total:
            est = count_total * cn / depth_total  # type: ignore[operator]
        elif count_total is not None:
            est = 0.0
        else:
            est = cn
        forms[form] = C4FormCN(form=form, n_reads=n_reads, depth=depth, cn_estimate=est, depth_cn_estimate=cn)

    return C4Chr17ControlResult(
        control_depth=control_depth,
        control_n_reads=control_n_reads,
        forms=forms,
        total_cn_estimate=count_total if count_total is not None else depth_total,
        depth_total_cn_estimate=depth_total,
        n_reads_5p_window=n_5p,
        reads_per_copy=reads_per_copy,
        herv_long_spanning=herv["long"],
        herv_short_spanning=herv["short"],
        herv_long_exposure=herv["long_exp"],
        herv_short_exposure=herv["short_exp"],
    )


def _norm_ll(x: float, mu: float, sd: float) -> float:
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)


def _binom_ll(k: int, n: int, p: float) -> float:
    return k * math.log(p) + (n - k) * math.log(1 - p)


def _pois_ll(k: int, mu: float) -> float:
    return k * math.log(mu) - mu - math.lgamma(k + 1)


def call_c4_forms_jointly(
    chr17: C4Chr17ControlResult,
    n_a: int,
    n_b: int,
    params: C4JointCallParams | None = None,
) -> C4JointCall | None:
    """Integer C4AL/AS/BL/BS maximising the joint likelihood of: the 5'
    window read count (Poisson, mean total x reads_per_copy); the Phase 2
    A/B exon-26 votes (binomial); the HERV-spanning long/short counts
    (binomial, exposure-weighted); and each form's depth share (normal
    around its called share of the total). None without a count-based
    total (`reads_per_copy` is None).

    **No copy-number cap, exact.** Totals are walked upward from 0, each
    split into every 4-form composition. Every non-Poisson term is bounded
    above by a constant (binomials by their value at p = k/n, normals by
    -log sd), and past its mode the Poisson term only falls, so once
    Poisson(T) + those bounds drops below the best call found no larger
    total can win. Total 0 is scored by the Poisson term (mean floored at
    `zero_cn_mean_frac` x reads_per_copy) plus the same upper bounds -- the
    most favourable reading of the undefined share terms, so a CN=0 call
    rests on the read count alone (invariant 4).

    Deterministic: compositions are visited in a fixed order and ties keep
    the first seen (invariant 9)."""
    if chr17.reads_per_copy is None:
        return None
    p = params or load_c4_joint_call_params()
    rpc, n5 = chr17.reads_per_copy, chr17.n_reads_5p_window
    nL, nS = chr17.herv_long_spanning, chr17.herv_short_spanning
    exp_tot = chr17.herv_long_exposure + chr17.herv_short_exposure
    kL = chr17.herv_long_exposure / exp_tot if exp_tot > 0 else 0.5
    depth_cn = {f: chr17.forms[f].depth_cn_estimate or 0.0 for f in FORMS}
    depth_sum = sum(depth_cn.values())
    share = {f: depth_cn[f] / depth_sum for f in FORMS} if depth_sum > 0 else None
    sd = p.form_proportion_sd

    def binom_max(k: int, n: int) -> float:
        if n == 0:
            return 0.0
        q = min(max(k / n, p.prob_floor), 1 - p.prob_floor)
        return _binom_ll(k, n, q)

    others_max = binom_max(n_a, n_a + n_b) + binom_max(nL, nL + nS)
    if share is not None:
        others_max += len(FORMS) * -math.log(sd)

    def clamp(q: float) -> float:
        return min(max(q, p.prob_floor), 1 - p.prob_floor)

    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    best_ll = _pois_ll(n5, p.zero_cn_mean_frac * rpc) + others_max
    best = (0, 0, 0, 0)
    scored.append((best_ll, best))
    mode = n5 / rpc
    t = 1
    while True:
        pois = _pois_ll(n5, t * rpc)
        if t > mode and pois + others_max < best_ll:
            break
        for al, as_, bl in itertools.product(range(t + 1), repeat=3):
            bs = t - al - as_ - bl
            if bs < 0:
                continue
            g = (al, as_, bl, bs)
            ll = pois
            if n_a + n_b:
                ll += _binom_ll(n_a, n_a + n_b, clamp((al + as_) / t))
            if nL + nS:
                nlong = al + bl
                ll += _binom_ll(nL, nL + nS, clamp(kL * nlong / (kL * nlong + (1 - kL) * (t - nlong))))
            if share is not None:
                ll += sum(_norm_ll(share[f], v / t, sd) for f, v in zip(FORMS, g))
            scored.append((ll, g))
            if ll > best_ll:
                best_ll, best = ll, g
        t += 1

    rest = sum(math.exp(ll - best_ll) for ll, g in scored if g != best)
    gq = p.gq_cap if rest <= 0 else min(p.gq_cap, -10.0 * math.log10(rest / (1.0 + rest)))
    return C4JointCall(forms=dict(zip(FORMS, best)), total=sum(best), gq=gq, log_likelihood=best_ll)

"""Stage 2 -- template set selection (DESIGN.md S2e).

Replaces FuFiHLA's lexicographic S(A,B) = (n reads, coverage, matched bp)
tuple with a proper read-level likelihood under a platform error model, then
selection over candidate template *sets* of size CN (DESIGN.md S2a: "a
template set of size CN, not a fixed pair" -- CN is Stage 1's, per gene's
segment, not assumed to be 2). The read x allele matrix this module scores
is stages/assign.py's GeneAssignmentMatrix; CN comes from stages/cn.py.

Candidate alleles are every allele in the gene's pool that has at least one
real (non-floor) read alignment -- not a fixed top-N. This is what
DESIGN.md S2e means by "raise the candidate cap well above FuFiHLA's
top-15/30": there is no cap at all, only the natural restriction that an
allele with zero read support cannot be part of any genotype the data
supports (invariant 11 -- FuFiHLA's cap is a Python performance workaround,
not a principle, and assign.py's matrix already makes it unnecessary).

**Two read-likelihood scorers, chosen by the caller, never branched on
inside the selection algorithm itself:**

- `context_log_likelihoods()` -- the ONT R10 shipped default (invariant 1).
  Uses stages/error_model.py's Badread-derived per-7-mer empirical error
  table (`resources/models/vendor/nanopore2023.gz`, real ONT R10.4.1
  alignments) via each candidate (read, allele) pair's actual CIGAR
  (stages/assign.py's `GeneAssignmentMatrix.spans`) -- homopolymer-context
  aware, not a flat rate. This is what replaced the original flat
  `edit_error_rate` placeholder once a real error model was available.
- `read_log_likelihoods()` -- a flat per-aligned-base binomial over
  aggregate edit distance (PAF's NM tag). Kept for the milestone-1 HiFi
  regression check (DESIGN.md S6), which runs on a platform Badread has no
  matching trained model for in this project's vendor set: that caller
  supplies its own literature/fitted rate as a plain `ErrorModel` argument,
  never a second persisted "shipped" model file (confirmed with the user
  2026-07-23).

Both scorers produce the same shape of output -- (n_reads, n_candidates)
log-likelihoods -- which `select_template_set()` consumes generically; the
selection/EM/LRT machinery below has no platform-specific branching at all.

Genotype likelihood: reads originate from one of the CN template copies
(unknown which), so P(reads | {a_1..a_k}) is a k-component equal-support
mixture with EM-estimated weights (allelic balance) -- vectorised over
reads via numpy, per invariant "vectorise, do not loop over reads". The
loop this module does have is over candidate *template sets*
(itertools.combinations), not reads; each set's mixture likelihood is one
vectorised computation across every read at once. Candidate-set counts are
bounded by real read support, typically small; profile before optimising
further (see development notes conventions) rather than pre-emptively flattening
the outer loop into a 3-D tensor.

Homozygous-vs-heterogeneous LRT: DESIGN.md S2e says replace FuFiHLA's hard
"4x coverage ratio -> force homozygous" rule with a likelihood-ratio test.
The single-allele (CN=1-style, i.e. every copy the same allele) hypothesis
is the boundary case of the k-component mixture where all components are
identical -- nested in the general model, so `2 * (best_k_mixture_logL -
best_single_allele_logL)` is a legitimate LRT statistic. This is reported
as a statistic, not converted into a p-value: the models compared are over
a discrete allele search space, not a continuous parameter, so a formal
chi-square asymptotic does not strictly hold. Do not oversell the number.
"""
from __future__ import annotations

import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pysam
import yaml

from assign import DEFAULT_POOL_FASTA, GeneAssignmentMatrix, realign_to_candidates

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_ONT_R10_MODEL = REPO_ROOT / "resources" / "models" / "read_error_ont_r10.yaml"

# Below this many candidate template sets, ProcessPoolExecutor's own
# spawn/pickle overhead costs more than the sequential loop it would
# replace -- an engineering knob for worker-pool overhead, not a fitted
# biological threshold (invariant 10 is about error rates/model constants,
# not this).
_MIN_COMBOS_FOR_PARALLEL = 200

# Candidate-narrowing prefilter (added 2026-07-27, real HG00097 ONT
# benchmark): a template-heavy gene like HLA-A/HLA-C (thousands of IPD
# genomic templates) can leave candidate_alleles() with a long tail of
# alleles that only one noisy ONT read weakly touched -- each one still
# costs a full row in context_log_likelihoods()'s per-cell Python loop
# (before any combination search even starts) and multiplies the
# itertools.combinations() search below. This threshold drops candidates
# whose *aggregate* mismatch rate (summed nm/aln_block_len across every
# read that has a real alignment to them -- already computed by
# assign.build_gene_matrices(), no extra alignment work) is more than this
# multiple of the best candidate's rate -- same "N x best" shape as
# input/fastq.py's mismatch_rate_multiplier decoy-rejection, chosen for
# consistency with that already-accepted provisional judgment call, not
# independently re-derived. Aggregating over many reads per candidate
# (rather than fastq.py's single-read hits) is inherently a lower-variance
# statistic, so this multiplier is not simply "the same risk again" -- but
# it is still unvalidated against real data (invariant 11), a data-adaptive
# margin rather than a fixed top-N/count cap (the FuFiHLA cap this module's
# docstring says was deliberately removed), and applies only to the ONT R10
# path (select_template_set_ont_r10), not the flat/HiFi milestone-1
# validation path (select_template_set_flat), which this prefilter does not
# touch.
_CANDIDATE_MISMATCH_RATE_MULTIPLIER = 10.0

# Floor under the "best candidate's rate" before the multiplier is applied.
# Without it, a best candidate with a literal 0.0 aggregate mismatch rate
# (a clean run of perfectly-matching reads -- not implausible) would collapse
# the threshold to 0.0 too, filtering out a genuinely real second haplotype
# allele purely because it carries real, biological, nonzero divergence from
# the best-matching one. Unlike fastq.py's per-read filter (fine to lose a
# marginal read), this filter must not be able to drop a true candidate the
# combination search still needs for cn>1. 0.01 (1%) is comfortably above
# ONT R10's per-base error rate floor on well-aligned reads and comfortably
# below a genuinely wrong/paralogous candidate's rate; provisional, same as
# the multiplier above.
_CANDIDATE_MISMATCH_RATE_FLOOR = 0.01

# Secondary, fixed-size cap applied after the multiplier/floor filter above:
# for a gene whose allele family is dense enough that many candidates
# cluster within the multiplier of the best (HLA-A/HLA-C's closely-related
# IPD catalogues), the multiplier alone has no bound on how many survive --
# it is a data-adaptive *quality* bar, not a worst-case *count* bound. This
# keeps only the `max_candidates` lowest-mismatch survivors of that filter
# (ties broken alphabetically, invariant 9). Deliberately not FuFiHLA's
# fixed top-15/30 cap in spirit -- it only ever trims what the multiplier
# already judged competitive, never substitutes for that judgement -- but
# it is still, like the two constants above, an unvalidated engineering
# threshold (invariant 11), chosen generously above what a real
# heterozygous (or higher-CN) call needs, not fitted.
_MAX_CANDIDATES_AFTER_PREFILTER = 50

# `_MAX_CANDIDATES_AFTER_PREFILTER`'s ranking key (real HG00097 ONT
# benchmark investigation, same session as the other 2026-07-27 constants):
# ranking survivors by raw aggregate mismatch rate alone, with no regard
# for how many reads produced that rate, let noise beat signal in
# template-heavy pools. In HLA-A's 5707-allele pool, dozens of alleles got
# exactly one stray read aligning over a short span with near-zero
# mismatch (observed: rate 0.0031, n=1) -- a small-sample fluke, not a
# better match -- and crowded the *true* allele (HLA-A*03:01:01:01: 27
# reads, rate 0.0120) out of the fixed-size-50 cap entirely.
#
# First attempted fix was a fixed floor (>=4 real-alignment reads before a
# candidate's rate is trusted): this helped HLA-A but not HLA-C or MICA --
# their pools are dense enough (6240, 659 alleles) that plenty of
# off-target candidates still pick up 4-11 real reads by chance proximity
# to the true allele, and an aggregate rate over that few reads is still
# noisy enough to out-rank a true 62-80-read signal by luck. A fixed
# absolute count can't distinguish "genuinely thin support" from "this
# gene's typical candidate only ever gets a few reads" across genes with
# very different pool densities and depths.
#
# Ranking is now (read depth descending, mismatch rate ascending, allele)
# -- depth-relative rather than a fixed count: among candidates that
# already cleared the multiplier/floor quality bar above ("a decent
# mismatch rate"), the ones real reads actually piled onto outrank ones
# that merely got lucky on a handful, with rate only breaking ties at
# equal depth. No new constant to (mis-)tune (invariant 11 is moot here --
# there is nothing left to fit).
#
# Depth-first ranking alone still let a *cluster* of read-indistinguishable
# names (module docstring on narrow_candidates_by_mismatch_rate()'s
# signature grouping, added same investigation) fill the whole cap by
# itself. Within one such signature, keep at most this many survivors
# rather than collapsing to a single representative: collapsing to 1 is
# unsound because a genuinely heterozygous pair's two *real* alleles can
# land in the identical signature too (equal read depth/mismatch load in
# each direction is symmetric, not a special case -- confirmed by a real
# end-to-end test regression during this investigation). Not a new guess:
# reuses stages/pipeline.py's own `max_cn` default, the already-established
# ceiling on how many genuinely distinct alleles a segment carries.
_MAX_CANDIDATES_PER_SIGNATURE = 2

# Pre-search read-count/ratio floor (added 2026-09-10, real H2009 CASTLE
# HLA-DQB1 finding, checked directly against real numbers before building
# this). select_template_set()'s own argmax-over-combinations search picks
# whichever candidate SET has the highest total *summed* log-likelihood --
# not the set with the most reads agreeing with it. Confirmed directly on
# real data: a combo containing a component backed by only 2 reads, each
# with a catastrophic ~1800-log-unit advantage over everything else in that
# gene's true allele family, out-scored the true heterozygous pair (whose
# 20 genuinely-discriminating reads each carried only a ~40-log-unit
# advantage) -- summed log-likelihood rewards a couple of extreme per-read
# gaps over a broader, subtler true signal. By the time stages/pipeline.py's
# own post-consensus min_reads_per_haplotype/min_haplotype_read_ratio floor
# would reject that 2-read winner, the true pair never got a turn: only the
# argmax winner is ever checked against it.
#
# A combo whose EM-implied hard read split (argmax responsibility per read
# from its own converged weights, the same per-read winner-takes-all
# reduction stages/consensus.py's real post-alignment partition uses
# downstream) fails either floor is disqualified from ever winning the
# argmax, so the search naturally falls through to the next-best combo.
#
# Disabled by default (0 reads / 0.0 ratio, i.e. no combo is ever
# disqualified) so every existing caller of select_template_set() keeps its
# exact prior behaviour unless it opts in.
# select_template_set_ont_r10() opts in with the ONT-path values below;
# select_template_set_flat() (the frozen DESIGN.md S6 HiFi milestone-1
# FuFiHLA-parity regression target) deliberately does not, matching how
# every other ONT-specific tuning constant in this module is scoped.
#
# `_ONT_MIN_HAPLOTYPE_READ_RATIO`: tried 0.1, then 0.2, both reverted --
# real 2026-09-10/11 investigation, both raised and root-caused by the user
# directly questioning the mechanism rather than more threshold sweeps.
#
# The 0.1 -> 0.2 revision (same day) was itself a real fix for three cross-
# sample regressions (H2009/MICA, HG03583/DPA1, HG03583/DPB1, ratio
# 0.130-0.154) without disturbing the six genuine heterozygous cases that
# motivated 0.1 (ratio >= 0.280) -- but a follow-up real-benchmark check
# found 0.2 itself broke three *different* real cases (HG01258/HLA-J,
# HG00253/HLA-N, HG00320/HLA-DRB1, ratio 0.184-0.302) that 0.1 had gotten
# right, a net wash (+3/-3) rather than a fix.
#
# Root cause, confirmed by direct per-combination diagnostics (not
# guessed): select_template_set()'s pre-search floor uses each combo's
# *EM-implied* hard read split (a cheap proxy computed before any real
# alignment) to decide eligibility -- but that proxy systematically
# UNDER-estimates how skewed a real, correct pair's split actually is once
# Stage 3 does real alignment. Real HG03583/DPA1 case: the true pair's
# EM-implied split looked close enough to disqualify at both 0.1 and 0.2,
# but its *real* post-consensus split (computed by
# stages/consensus.py's actual re-alignment) was 9:113 (ratio 0.08) --
# far more skewed than any ratio floor at this stage should ever pass, yet
# the pre-search floor's own under-estimate had already rejected it in
# favour of a worse-scoring, more EM-evenly-split WRONG pair. Any ratio
# value here trades one class of real case against another, because the
# proxy driving the decision is the wrong signal for a ratio floor to use
# pre-search -- the same lesson TAP1/DQB1's own reverted attempts already
# found for other thresholds in this module (module docstring above).
#
# Fix: the ratio component of the pre-search floor is retired (0.0, a
# no-op -- see _passes_read_count_floor). Only the absolute-count
# component (_ONT_MIN_READS_PER_HAPLOTYPE=3) still disqualifies a combo
# pre-search -- sufficient on its own for the original DQB1 bug (the
# spurious pair's real problem was an EM-implied count of ~2, well under
# 3, no ratio needed) without ever touching a combo whose *EM-implied*
# minority count already clears 3, which is exactly the case for every one
# of the six real regressions above. The ratio question -- is a
# genuinely-selected pair's split too skewed to represent two real
# alleles -- is left entirely to stages/pipeline.py's existing
# *post*-consensus min_haplotype_read_ratio check (still 0.1, unchanged),
# which acts on the real alignment-based count, not the EM proxy, and
# whose failure mode (fall back to the major allele of the *same*,
# already-correctly-selected pair) is much gentler than the pre-search
# floor's (jump to a different, unrelated combo -- see genotype.py's
# _score_combo argmax-over-eligible-combos mechanism).
#
# Real validation, all on the 2026-09-10/11 session's own already-
# extracted samples, floor=(abs3, ratio0.0), post-consensus ratio
# unchanged at 0.1: DQB1/H2009 recovers the true pair (LRT=730, real split
# 28:22); HG03583 DPA1/DPB1 both now correctly collapse to their true
# homozygous allele (real splits 9:113 and 117:3, both -- previously
# unfixed by either 0.1 or 0.2); HG01258/HLA-J, HG00253/HLA-N and
# HG00320/HLA-DRB1 all reach the search stage with their correct/expected
# pair undisturbed (verified via the real post-consensus counts each now
# produces against the unchanged 0.1 post-consensus ratio). Still
# provisional (invariant 11) -- not yet re-run through the full 6-sample
# benchmark end-to-end as of this constant's own change.
_ONT_MIN_READS_PER_HAPLOTYPE = 3
_ONT_MIN_HAPLOTYPE_READ_RATIO = 0.0

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import CandidateRead  # noqa: E402

from error_model import KmerErrorModel, load_kmer_error_model, read_allele_log_likelihood  # noqa: E402

_EM_MAX_ITER = 100
_EM_TOL = 1e-8


@dataclass(frozen=True)
class ErrorModel:
    """Flat per-base binomial model -- see module docstring for its scope
    (the HiFi validation path; ONT R10's shipped model is OntR10Config
    below, not this)."""

    platform: str
    edit_error_rate: float
    unaligned_edit_rate: float


def load_error_model(path: Path) -> ErrorModel:
    """Load a versioned flat-model file (invariant 10). Caller picks which
    file -- this function does not default to a platform based on any
    gene/sample property."""
    raw = yaml.safe_load(path.read_text())
    return ErrorModel(
        platform=raw["platform"],
        edit_error_rate=float(raw["edit_error_rate"]),
        unaligned_edit_rate=float(raw["unaligned_edit_rate"]),
    )


def read_log_likelihoods(
    matrix: GeneAssignmentMatrix,
    error_model: ErrorModel,
    columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    """log P(read | allele) for every (read, allele) cell under the flat
    model, shape (n_reads, n_alleles) -- or (n_reads, len(columns)) if
    `columns` restricts it to a specific allele subset (e.g. candidate_alleles()'s
    output), which avoids ever materialising the full-pool-width array.

    Aligned cells: per-aligned-base binomial over the reported edit
    distance. Unaligned cells (aln_block_len == 0, meaning minimap2
    reported no hit for that pair at all): scored as if the read's own
    best-aligned span against this gene were entirely edits, under the
    model's separate (much higher) `unaligned_edit_rate` -- a floor, not a
    zero, so no allele is silently excluded from the mixture and log
    arithmetic never sees log(0)."""
    eps = error_model.edit_error_rate
    aligned_ll = np.where(
        matrix.aln_block_len > 0,
        matrix.nm * np.log(eps) + (matrix.aln_block_len - matrix.nm) * np.log(1 - eps),
        0.0,
    )

    # Per-read effective span for the unaligned floor: the longest real
    # alignment that read achieved anywhere in this gene's pool. Every read
    # in the matrix has at least one real alignment (assign.py only creates
    # a row when minimap2 reported >=1 hit), so this is always > 0.
    row_span = matrix.aln_block_len.max(axis=1, keepdims=True)
    floor_eps = error_model.unaligned_edit_rate
    floor_ll = row_span * np.log(floor_eps)

    full = np.where(matrix.aln_block_len > 0, aligned_ll, floor_ll)
    if columns is None:
        return full
    idx = [matrix.allele_ids.index(a) for a in columns]
    return full[:, idx]


@dataclass(frozen=True)
class OntR10Config:
    context_model: KmerErrorModel
    unaligned_edit_rate: float


def load_ont_r10_config(path: Path = DEFAULT_ONT_R10_MODEL) -> OntR10Config:
    """Load the shipped ONT R10 configuration (invariant 10): the Badread-
    derived context model plus the flat floor rate applied only to (read,
    allele) cells with no real alignment at all (no CIGAR to take context
    from -- see context_log_likelihoods())."""
    raw = yaml.safe_load(path.read_text())
    context_model_path = (path.parent / raw["context_model"]).resolve()
    return OntR10Config(
        context_model=load_kmer_error_model(context_model_path),
        unaligned_edit_rate=float(raw["unaligned_edit_rate"]),
    )


def load_allele_sequences(pool_fasta: Path, allele_ids: Sequence[str]) -> dict[str, str]:
    """Stream pool_fasta once, keeping only the requested sequences.
    Candidates are typically a small fraction of a gene's full pool (tens
    vs thousands for HLA-A), so this avoids loading records that can never
    be scored. No FASTA index is built (pool_fasta isn't bgzipped); profile
    before adding one if this becomes a bottleneck across many genes in one
    run (the development notes' "profile before optimising")."""
    wanted = set(allele_ids)
    found: dict[str, str] = {}
    with pysam.FastxFile(str(pool_fasta)) as fh:
        for rec in fh:
            if rec.name in wanted:
                if rec.sequence is None:
                    raise ValueError(f"{pool_fasta}: record {rec.name!r} has no sequence")
                found[rec.name] = rec.sequence
                if len(found) == len(wanted):
                    break
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{pool_fasta}: missing sequence(s) for {sorted(missing)}")
    return found


def context_log_likelihoods(
    matrix: GeneAssignmentMatrix,
    read_sequences: dict[str, str],
    allele_sequences: dict[str, str],
    candidates: tuple[str, ...],
    config: OntR10Config,
) -> np.ndarray:
    """Context-aware log P(read | allele) for `candidates` only, shape
    (n_reads, len(candidates)) -- the ONT R10 shipped scorer (invariant 1),
    using stages/error_model.py's Badread-derived per-7-mer model against
    each (read, allele) pair's actual alignment
    (GeneAssignmentMatrix.spans, from stages/assign.py).

    Cells with no captured span (no real alignment for that pair) fall
    back to the same floor treatment as read_log_likelihoods(): the read's
    own best real span, scored at a flat `unaligned_edit_rate` per base --
    there is no alignment to take k-mer context from there, so the context
    model has nothing to apply.

    **Cells with a captured span are also charged for whatever the span
    doesn't cover relative to the best any candidate achieved for this read
    (added 2026-07-29, real HG00097 finding, revised same day).** minimap2
    is a local aligner: a candidate that only explains part of a read (e.g.
    a truncated/partial IPD "genomic" record that runs out before the read
    does) gets soft-clipped with no record of the missing part in its own
    aligned-region stats -- `read_allele_log_likelihood()` alone would
    silently omit that read length from the score entirely, which is not
    "no opinion", it's a free pass. Since every per-window term the context
    model returns is <= 0 (a log-probability), a candidate that has to
    explain *fewer* bases can only ever score higher than one explaining
    *more* of the same read, independent of match quality -- confirmed as
    the mechanism behind a real wrong call: a 3323bp candidate beat the
    3502bp truth allele the reads actually supported end-to-end, purely
    because the shorter one was never asked to account for the other
    179bp.

    **First attempt used the read's true full length (PAF's `qlen`) as the
    reference point and was wrong -- caught directly against real HG00097
    data, not by inspection.** This project's ONT reads include
    ultralongs (DESIGN.md S6); a read can be 100kb+ while only ~3.5kb of it
    is the gene at all, the rest genuinely-uninvolved flanking genomic
    sequence no candidate should ever be expected to explain. Charging
    every candidate for `qlen - covered` treated all of that flanking
    length as evidence against the candidate, a penalty of 50,000+ log
    units on some reads that drowned out the real, correct ~180bp signal
    the fix exists to surface. The reference point that is actually
    self-limiting to gene scale is `row_span` (below) -- the best aligned
    length *any* candidate in this matrix achieved for this read, already
    computed for the wholly-unaligned floor case -- since no HLA allele in
    the pool is longer than the gene itself, `row_span` can never run away
    to a read's full ultralong length the way `qlen` does. A candidate is
    now charged for covering less of the read than the best candidate
    managed (`row_span[i] - (read_end - read_start)`), not for failing to
    cover the read's entire, possibly-irrelevant length -- the same floor
    rate as the wholly-unaligned case, and the two are now the same
    formula in the limit (an unaligned cell is the `covered == 0` case of
    this one).

    **Attempted-and-reverted 2026-07-30: capping this penalty at the
    candidate's own unused reference length.** Real HG00097 finding:
    HLA-DRB1*15:01:01:03's own genomic record (11056bp) is shorter than
    HLA-DRB1*15:01:01:02's (11571bp), and *:03 aligned perfectly (nm=0)
    across its *entire* record while *:02 covered the full consensus with a
    real 2bp deletion -- the uncapped penalty charged *:03 for the ~515bp
    its own record structurally could not describe, making the 2bp-deletion
    candidate outscore the exact match. Capping `uncovered` at
    `len(allele) - (ref_end - ref_start)` (zero for any candidate that
    fully consumes its own reference) fixed that one case, but caused a
    large net regression on the same real sample: HLA-A, HLA-B, MICA and
    others flipped to substantially *shorter, wrong* candidates (e.g.
    HLA-A*03:01:01:59 at 2944bp beating the true HLA-A*03:01:01:01 at
    3502bp) that happened to align cleanly over their own short span --
    exactly the exploit the 2026-07-29 fix exists to close, just reopened
    under a different trigger. The cap could not distinguish "this
    candidate's record is genuinely, only slightly shorter due to real
    inter-submission UTR/flanking variance" (DRB1: 95.5% of row_span) from
    "this candidate is a substantially shorter, likely-wrong record that
    happens to match its own short span well" (the regressions: 71-91% of
    row_span) -- "fully consumes its own reference" is true in both shapes
    and cannot tell them apart. Reverted.

    **Attempted-and-reverted 2026-08-02: Stage-4-only forgiveness gated on
    core-length parity.** A narrower design than the above: only forgive a
    candidate's boundary overhang when its own annotated CDS+intron
    `core_length` (real IPD FT data, no pool-wide aggregation) is within a
    small tolerance of the best candidate's -- i.e. genuinely the same real
    core content, short only because of UTR submission completeness. This
    does fix HLA-DRB1 hap1 specifically (verified), but real HG00097
    benchmarking found it regressed several other genes (HLA-C, HLA-DMB,
    HLA-DOB, HLA-DQA2, HLA-DQB1 -- 2-field allele accuracy 26/40 -> 18/40):
    traced to the fact that "same core_length" is not a rare, precise
    signal isolating one true sibling pair the way it is for DRB1 -- for a
    common allele group like HLA-C*07:02, *dozens* of real IPD names share
    near-identical core content by construction (field 3/4 differences are
    non-coding by IPD's own nomenclature convention), so forgiving the
    whole family at once removes the very signal (whichever specific
    submission's full record, UTR included, best matches the sample) that
    was previously doing real disambiguation work within it, leaving the
    model to pick among dozens of near-tied candidates on scattered minor
    real variants alone. Reverted; HLA-DRB1 hap1 is open again. Any future
    attempt needs a mechanism that behaves differently for a small, forced
    2-candidate tie (DRB1's actual shape) than for a large, natural
    same-protein allele family (the norm for common HLA genes) -- treating
    every same-core-length candidate identically, as this attempt did,
    does not have that distinction available.

    This loops over (read, candidate) cells in Python -- CIGAR-based
    context scoring isn't a numpy-vectorisable operation the way the flat
    model's aggregate-count arithmetic is. Candidate counts are bounded by
    real read support (candidate_alleles()), typically tens of alleles;
    not expected to be a bottleneck at v0 (profile before optimising,
    development notes conventions).
    """
    col_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    row_span = matrix.aln_block_len.max(axis=1)
    floor_per_base = math.log(config.unaligned_edit_rate)

    n_reads = len(matrix.read_ids)
    ll = np.empty((n_reads, len(candidates)), dtype=np.float64)
    for i, read_id in enumerate(matrix.read_ids):
        floor_ll = float(row_span[i]) * floor_per_base
        read_seq = read_sequences[read_id]
        for k, allele in enumerate(candidates):
            span = matrix.spans.get((i, col_index[allele]))
            if span is None:
                ll[i, k] = floor_ll
                continue
            aligned_ll = read_allele_log_likelihood(
                config.context_model, allele_sequences[allele], read_seq,
                ref_start=span.ref_start, ref_end=span.ref_end,
                read_start=span.read_start, read_end=span.read_end,
                strand=span.strand, cigar=span.cigar,
            )
            uncovered = float(row_span[i]) - (span.read_end - span.read_start)
            ll[i, k] = aligned_ll + uncovered * floor_per_base
    return ll


def candidate_alleles(matrix: GeneAssignmentMatrix) -> tuple[str, ...]:
    """Every allele backed by at least one real read alignment -- the only
    restriction on the search space (see module docstring on the FuFiHLA
    top-15/30 cap this replaces). Sorted for determinism (invariant 9)."""
    has_support = (matrix.aln_block_len > 0).any(axis=0)
    return tuple(sorted(a for a, keep in zip(matrix.allele_ids, has_support) if keep))


def narrow_candidates_by_mismatch_rate(
    matrix: GeneAssignmentMatrix,
    candidates: tuple[str, ...],
    *,
    multiplier: float = _CANDIDATE_MISMATCH_RATE_MULTIPLIER,
    floor: float = _CANDIDATE_MISMATCH_RATE_FLOOR,
    max_candidates: int = _MAX_CANDIDATES_AFTER_PREFILTER,
) -> tuple[str, ...]:
    """Performance prefilter for template-heavy genes (see module docstring
    constants above) -- drops candidates whose aggregate mismatch rate
    exceeds `multiplier` x max(best candidate's rate, `floor`), using only
    the alignment stats assign.build_gene_matrices() already computed (no
    new alignment work). Vectorised (invariant: no per-read loop).

    Before the fixed-size cap is applied, survivors sharing a signature --
    (a) a real alignment from the identical set of reads, and (b) the same
    mismatch-rate bucket (rate rounded to `floor`'s own scale, reused
    rather than inventing a new precision constant) -- are capped at
    `_MAX_CANDIDATES_PER_SIGNATURE` (2) each, not collapsed to one. This
    exists because IPD's own nomenclature routinely assigns dozens to
    hundreds of distinct names to genomic sequences differing only outside
    every read's aligned span (real HLA-A: 127 names under "03:01:01", all
    hit by the identical 27 reads at near-identical rates) -- left
    uncapped, that one cluster can fill the entire cap by itself and
    silently exclude a genuinely different second-haplotype family from
    ever becoming a candidate at all (confirmed on real HG00097 data).

    2, not 1: collapsing a signature to a single representative is unsound
    -- a genuinely heterozygous pair's two *real*, distinct alleles can
    land in the identical signature too (a real regression caught by this
    module's own end-to-end test: two alleles differing by 5 real
    substitutions produced bit-identical aggregate rates by symmetry --
    equal read depth and equal mismatch load in each direction -- which is
    exactly what a real heterozygous pair typically looks like, not a
    special case to exclude). 2 is not a new invariant-11 guess; it
    reuses this codebase's own established ceiling (stages/pipeline.py's
    `max_cn` default) for how many genuinely distinct alleles a segment
    is expected to carry. Within a signature, the lowest-rate members
    survive (ties broken alphabetically, invariant 9); the rest are
    dropped, not renamed -- Stage 4 (stages/hla_typing.py) re-scores the
    final consensus against the *unrestricted* full pool independently of
    which specific name Stage 2 used to seed consensus-building, so no
    naming precision is lost by not carrying every synonym through this
    search.

    Ranking for the cap itself is then (read depth descending, rate
    ascending, allele) over the surviving representatives -- depth first,
    not rate first, because a handful of off-target candidates can always
    out-score a true, well-supported allele's aggregate rate by chance in
    a dense enough pool, at any fixed read-count floor (module docstring
    history). Rate only breaks ties at equal depth; ties within that still
    break alphabetically, same as select_template_set() itself uses.

    Every element of `candidates` is guaranteed >=1 real-alignment cell
    (candidate_alleles()'s own contract), so each column's aligned-cell
    count is always > 0 here -- no division-by-zero guard needed. Sorted
    input in, sorted (subset) output -- determinism (invariant 9)."""
    col_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    cols = [col_index[a] for a in candidates]
    aln = matrix.aln_block_len[:, cols]
    nm = matrix.nm[:, cols]
    aligned = aln > 0
    total_aln = np.where(aligned, aln, 0).sum(axis=0)
    total_nm = np.where(aligned, nm, 0).sum(axis=0)
    mismatch_rate = total_nm / total_aln
    n_reads_hit = aligned.sum(axis=0)

    threshold = multiplier * max(float(mismatch_rate.min()), floor)

    groups: dict[tuple[bytes, int], list[tuple[float, str, int]]] = {}
    for j, a in enumerate(candidates):
        rate = float(mismatch_rate[j])
        if rate > threshold:
            continue
        key = (aligned[:, j].tobytes(), round(rate / floor))
        groups.setdefault(key, []).append((rate, a, int(n_reads_hit[j])))

    representatives: list[tuple[float, str, int]] = []
    for members in groups.values():
        members.sort(key=lambda t: (t[0], t[1]))
        representatives.extend(members[:_MAX_CANDIDATES_PER_SIGNATURE])

    ranked = sorted(representatives, key=lambda t: (-t[2], t[0], t[1]))
    return tuple(sorted(a for _, a, _ in ranked[:max_candidates]))


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def _em_mixture_log_likelihood(ll_sub: np.ndarray) -> tuple[float, np.ndarray]:
    """EM for a k-component equal-read-pool mixture over one candidate
    template set. ll_sub is (n_reads, k) -- read log-likelihoods restricted
    to the k alleles in the candidate set. Returns (marginal log-likelihood
    at convergence, per-allele mixing weights).

    Vectorised over reads throughout (invariant: no per-read loop); the only
    Python-level loop is the fixed small EM iteration count.
    """
    n_reads, k = ll_sub.shape
    weights = np.full(k, 1.0 / k)
    prev_ll = -np.inf
    for _ in range(_EM_MAX_ITER):
        log_weighted = ll_sub + np.log(weights)[None, :]
        log_row_total = _logsumexp(log_weighted, axis=1)
        total_ll = float(log_row_total.sum())

        # E-step: per-read responsibilities.
        resp = np.exp(log_weighted - log_row_total[:, None])
        # M-step: mixing weights are the mean responsibility per component.
        weights = resp.mean(axis=0)
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()

        if total_ll - prev_ll < _EM_TOL:
            prev_ll = total_ll
            break
        prev_ll = total_ll

    # Final log-likelihood at the converged weights (not the pre-update one
    # from the last M-step iteration).
    log_weighted = ll_sub + np.log(weights)[None, :]
    final_ll = float(_logsumexp(log_weighted, axis=1).sum())
    return final_ll, weights


def _hard_read_counts(ll_sub: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-component read counts implied by one combo's own converged EM
    weights -- a hard argmax over each read's `ll_sub + log(weights)` row
    (the same winner-takes-all reduction stages/consensus.py's real
    post-alignment `assign_reads_to_haplotypes()` performs downstream, used
    here as a cheap proxy with no new alignment work: the EM run this needs
    already happened). Module docstring on `_ONT_MIN_READS_PER_HAPLOTYPE`."""
    log_weighted = ll_sub + np.log(weights)[None, :]
    hard = np.argmax(log_weighted, axis=1)
    return np.bincount(hard, minlength=ll_sub.shape[1])


def _passes_read_count_floor(
    counts: np.ndarray, min_reads_per_haplotype: int, min_haplotype_read_ratio: float,
) -> bool:
    """True iff every component's implied hard read count clears both the
    absolute floor and the min/max ratio floor -- module docstring on
    `_ONT_MIN_READS_PER_HAPLOTYPE`. `min_reads_per_haplotype=0` and
    `min_haplotype_read_ratio=0.0` (select_template_set()'s own defaults)
    make this always True, i.e. a no-op for any caller that doesn't opt in."""
    if int(counts.min()) < min_reads_per_haplotype:
        return False
    max_count = int(counts.max())
    if max_count > 0 and counts.min() / max_count < min_haplotype_read_ratio:
        return False
    return True


# Module-level, set once per worker process by _init_combo_worker (the
# ProcessPoolExecutor initializer) rather than re-pickled on every task --
# log_likelihoods is the one thing every combination scores against, so
# sending it once per worker instead of once per combo is what makes
# parallelising the outer loop worthwhile instead of IPC-bound.
_worker_log_likelihoods: np.ndarray | None = None
_worker_col_index: dict[str, int] | None = None
_worker_min_reads_per_haplotype: int = 0
_worker_min_haplotype_read_ratio: float = 0.0


def _init_combo_worker(
    log_likelihoods: np.ndarray, col_index: dict[str, int],
    min_reads_per_haplotype: int = 0, min_haplotype_read_ratio: float = 0.0,
) -> None:
    global _worker_log_likelihoods, _worker_col_index
    global _worker_min_reads_per_haplotype, _worker_min_haplotype_read_ratio
    _worker_log_likelihoods = log_likelihoods
    _worker_col_index = col_index
    _worker_min_reads_per_haplotype = min_reads_per_haplotype
    _worker_min_haplotype_read_ratio = min_haplotype_read_ratio


def _score_combo(combo: tuple[str, ...]) -> tuple[float, tuple[str, ...], np.ndarray, bool]:
    assert _worker_log_likelihoods is not None and _worker_col_index is not None
    cols = [_worker_col_index[a] for a in combo]
    ll_sub = _worker_log_likelihoods[:, cols]
    set_ll, set_weights = _em_mixture_log_likelihood(ll_sub)
    eligible = _passes_read_count_floor(
        _hard_read_counts(ll_sub, set_weights),
        _worker_min_reads_per_haplotype, _worker_min_haplotype_read_ratio,
    )
    return set_ll, combo, set_weights, eligible


@dataclass(frozen=True)
class TemplateCall:
    gene: str
    cn: int
    alleles: tuple[str, ...]
    weights: tuple[float, ...]
    log_likelihood: float
    n_candidates: int
    lrt_statistic: float
    single_allele_log_likelihood: float
    single_allele: str
    # The full narrowed candidate set select_template_set() searched over
    # (not just the `cn` winners in `alleles`) -- added so stages/hla_typing.py's
    # Stage 4 re-typing pass can densely realign the polished consensus
    # against every allele Stage 2 already knows has real read-level
    # support, not just whatever a fresh whole-pool minimap2 search happens
    # to report for the consensus itself. Real HG00097 TAP2 finding: a
    # genuinely shorter-but-correct candidate (*06, 972bp less UTR than the
    # template Stage 2 picked) never appears in that fresh search at all --
    # minimap2's own raw-score length bias favours a longer, mismatch-
    # bearing candidate over a shorter, mismatch-free one -- even though
    # Stage 2's own narrowed candidate set (this field) already had it.
    candidates: tuple[str, ...]


def _single_allele_call(
    gene: str, cn: int, candidates: tuple[str, ...],
    best_single_ll: float, best_single_allele: str,
) -> TemplateCall:
    """Shared shape for every "collapse to one distinct allele, `cn`
    preserved" return path in select_template_set() below -- the
    too-few-candidates fallback and the (new) all-combos-disqualified-by-
    the-read-count-floor fallback both produce exactly this."""
    return TemplateCall(
        gene=gene, cn=cn, alleles=(best_single_allele,),
        weights=(1.0,), log_likelihood=best_single_ll,
        n_candidates=len(candidates), lrt_statistic=0.0,
        single_allele_log_likelihood=best_single_ll,
        single_allele=best_single_allele,
        candidates=candidates,
    )


def select_template_set(
    gene: str,
    candidates: tuple[str, ...],
    log_likelihoods: np.ndarray,
    cn: int,
    *,
    n_threads: int = 1,
    min_reads_per_haplotype: int = 0,
    min_haplotype_read_ratio: float = 0.0,
) -> TemplateCall:
    """Select the best-scoring template set of size `cn` (DESIGN.md S2a)
    from `candidates`, given a precomputed (n_reads, len(candidates))
    log-likelihood matrix (from either read_log_likelihoods() restricted to
    `candidates`, or context_log_likelihoods() -- this function has no
    opinion on which scorer produced it). Also returns the LRT statistic
    against the single-allele (all-copies-identical) baseline that replaces
    FuFiHLA's 4x-ratio homozygosity rule.

    `min_reads_per_haplotype`/`min_haplotype_read_ratio` (module docstring
    on `_ONT_MIN_READS_PER_HAPLOTYPE`): disqualify a candidate combo from
    ever winning the argmax search below if its own EM-implied hard read
    split fails either floor. Disabled by default (0 / 0.0) -- every
    existing caller keeps its exact prior behaviour unless it opts in;
    select_template_set_ont_r10() does, select_template_set_flat() (the
    frozen DESIGN.md S6 HiFi milestone-1 path) does not.

    Determinism (invariant 9): candidates must already be sorted (both
    scorers above produce them via candidate_alleles(), which sorts); ties
    keep the first (lexicographically smallest) set found, never an
    arbitrary one from hash/dict iteration -- true regardless of n_threads,
    since ProcessPoolExecutor.map() (used below `_MIN_COMBOS_FOR_PARALLEL`
    combinations) returns results in call order, not completion order, so
    the same "first max wins" scan runs over the same sequence either way.

    n_threads>1 parallelises the itertools.combinations(candidates, cn)
    search below, not the single-allele baseline (already a vectorised
    numpy reduction, nothing to parallelise) -- each candidate *set*'s EM
    run is independent of every other set's, which is what makes this loop
    (not context_log_likelihoods()'s per-cell scoring, a separate,
    still-sequential cost) the one worth spreading across processes rather
    than threads (the EM/logsumexp work is real CPU, GIL-bound).

    `0 < len(candidates) < cn` (real HG-benchmark HLA-K/HLA-U finding,
    development notes 2026-09-10 session): a genuinely homozygous sample can have
    only one real candidate allele with read support at all -- there is
    nothing for `itertools.combinations(candidates, cn)` to search, since a
    combination can't repeat one element. This is not a failure requiring
    a raise; it is exactly the same "real cn copies, one distinct allele"
    shape `pipeline.py`'s `_collapse_to_single_allele()` already reports
    for the *post-search* case (LRT/read-ratio collapse) -- handled here
    the same way, `cn` preserved on the returned `TemplateCall`, only the
    distinct-allele count collapses. `lrt_statistic=0.0` (the same sentinel
    the `cn==1` baseline below already uses): with only one real candidate,
    there is no second allele to test heterozygosity against, so "no
    evidence for a second allele" is not a meaningful statistic to report
    as anything but zero. Only `len(candidates) == 0` (no read-supported
    candidate at all) still raises -- that is a real recruitment failure,
    not a homozygosity shape, and invariant "silent failure is worse than a
    crash" says it must stay loud.
    """
    if cn < 1:
        raise ValueError(f"cn must be >= 1, got {cn} for gene {gene!r}")
    if len(candidates) == 0:
        raise ValueError(
            f"{gene}: no candidate allele(s) with read support at all "
            f"(need >= cn={cn} to form a template set)"
        )
    if log_likelihoods.shape[1] != len(candidates):
        raise ValueError(
            f"{gene}: log_likelihoods has {log_likelihoods.shape[1]} columns, "
            f"expected {len(candidates)} (one per candidate)"
        )

    # Single-allele baseline (also cn=1's own answer, and the too-few-
    # candidates fallback's own answer): every candidate scored alone, no
    # mixture needed.
    single_scores = [
        (float(log_likelihoods[:, j].sum()), a) for j, a in enumerate(candidates)
    ]
    best_single_ll, best_single_allele = max(
        single_scores, key=lambda t: (t[0], t[1])
    )

    if cn == 1 or len(candidates) < cn:
        return _single_allele_call(gene, cn, candidates, best_single_ll, best_single_allele)

    col_index = {a: j for j, a in enumerate(candidates)}
    combos = list(itertools.combinations(candidates, cn))
    best_ll = -np.inf
    best_set: tuple[str, ...] | None = None
    best_weights: np.ndarray | None = None

    if n_threads <= 1 or len(combos) < _MIN_COMBOS_FOR_PARALLEL:
        for combo in combos:
            cols = [col_index[a] for a in combo]
            ll_sub = log_likelihoods[:, cols]
            set_ll, set_weights = _em_mixture_log_likelihood(ll_sub)
            if not _passes_read_count_floor(
                _hard_read_counts(ll_sub, set_weights),
                min_reads_per_haplotype, min_haplotype_read_ratio,
            ):
                continue
            if set_ll > best_ll:
                best_ll, best_set, best_weights = set_ll, combo, set_weights
    else:
        chunksize = max(1, len(combos) // (n_threads * 4))
        with ProcessPoolExecutor(
            max_workers=n_threads, initializer=_init_combo_worker,
            initargs=(log_likelihoods, col_index, min_reads_per_haplotype, min_haplotype_read_ratio),
        ) as pool:
            for set_ll, combo, set_weights, eligible in pool.map(_score_combo, combos, chunksize=chunksize):
                if eligible and set_ll > best_ll:
                    best_ll, best_set, best_weights = set_ll, combo, set_weights

    if best_set is None:
        # Every combo of size cn failed the read-count/ratio floor -- the
        # same "real cn copies, one distinct allele" shape as the
        # too-few-candidates branch above (module docstring): no evidence
        # for cn genuinely distinct alleles survives, so collapse rather
        # than crown a winner stages/pipeline.py's own downstream check
        # would immediately reject anyway.
        return _single_allele_call(gene, cn, candidates, best_single_ll, best_single_allele)

    assert best_weights is not None
    lrt = 2.0 * (best_ll - best_single_ll)
    return TemplateCall(
        gene=gene, cn=cn, alleles=best_set,
        weights=tuple(float(w) for w in best_weights),
        log_likelihood=best_ll, n_candidates=len(candidates),
        lrt_statistic=lrt, single_allele_log_likelihood=best_single_ll,
        single_allele=best_single_allele,
        candidates=candidates,
    )


def select_template_set_flat(
    matrix: GeneAssignmentMatrix,
    cn: int,
    error_model: ErrorModel,
    *,
    n_threads: int = 1,
) -> TemplateCall:
    """Convenience wrapper for the flat-model path (see module docstring --
    the HiFi validation caller's case)."""
    candidates = candidate_alleles(matrix)
    ll = read_log_likelihoods(matrix, error_model, columns=candidates)
    return select_template_set(matrix.gene, candidates, ll, cn, n_threads=n_threads)


def select_template_set_ont_r10(
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    cn: int,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    config: OntR10Config | None = None,
    n_threads: int = 1,
    minimap2_bin: str = "minimap2",
    min_reads_per_haplotype: int = _ONT_MIN_READS_PER_HAPLOTYPE,
    min_haplotype_read_ratio: float = _ONT_MIN_HAPLOTYPE_READ_RATIO,
) -> TemplateCall:
    """Stage 2 entry point for the shipped ONT R10 model (invariant 1) --
    wires candidate discovery, sequence loading, and the context-aware
    scorer together. `config` defaults to the versioned
    resources/models/read_error_ont_r10.yaml; pass an explicit one only for
    testing or a re-fitted replacement (invariant 10). n_threads only
    parallelises select_template_set()'s combination search below --
    context_log_likelihoods() above is still sequential (see its own
    docstring). candidate_alleles()'s output is narrowed by
    narrow_candidates_by_mismatch_rate() before the expensive per-cell
    context scoring runs at all (module docstring on
    _CANDIDATE_MISMATCH_RATE_MULTIPLIER) -- this is the main lever for
    template-heavy genes (HLA-A/HLA-C), since it cuts cost here too, not
    only in the combination search n_threads parallelises.

    **Dense realignment before scoring, added after a real HG00097
    finding.** `matrix` (stages/assign.py's build_gene_matrices() output)
    is built at Stage 1's low `n_best` (that function's own docstring) --
    each read has only *one* reported hit anywhere in the gene's full
    pool, which for a template-heavy gene often isn't against the true
    best-matching candidate specifically (confirmed on real data: HLA-A's
    true allele had a captured span for only 3 of 26 genuinely matching
    reads). An earlier version of this function scored the narrowed
    candidates on that sparse data to pick a winner, then only densely
    re-aligned *the winning two* to refit their weights/LRT -- which
    fixed the reported statistics but not the pick itself: the coarse
    scoring pass could still settle on a near-identical but not-quite-
    right specific name (confirmed directly: reads correctly partitioned
    to the true haplotype, then polished straight against the *true* IPD
    allele with stages/consensus.py's own build_haplotype_consensus(),
    produced a zero-variant exact match -- proving the reported
    "novel"/"extension" mismatch was an artifact of which candidate name
    this search picked, not a real difference in the sample).

    The fix removes the two-pass split entirely: `context_log_likelihoods()`
    costs the same O(reads x candidates) regardless of which alignment
    source feeds it, so there is no efficiency reason to search on sparse
    data at all once narrow_candidates_by_mismatch_rate() has already
    bounded the candidate count (<= ~50) -- `assign.realign_to_candidates()`
    densely re-aligns every read against every narrowed candidate up front
    (reads x <=50, cheaper than Stage 1's own full-pool sweep), and the
    single resulting search runs entirely on accurate data.

    `min_reads_per_haplotype`/`min_haplotype_read_ratio` default to
    stages/pipeline.py's own already-validated values (module docstring on
    `_ONT_MIN_READS_PER_HAPLOTYPE` in this module) and are forwarded to
    select_template_set() to disqualify a combo from the search itself
    rather than only catching it after the fact."""
    if config is None:
        config = load_ont_r10_config()
    candidates = narrow_candidates_by_mismatch_rate(matrix, candidate_alleles(matrix))
    read_id_set = set(matrix.read_ids)
    candidate_reads = [r for r in reads if r.read_id in read_id_set]
    allele_sequences = load_allele_sequences(pool_fasta, candidates)

    dense_matrix = realign_to_candidates(
        matrix.gene, candidate_reads, allele_sequences,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    dense_read_sequences = {
        r.read_id: r.sequence for r in candidate_reads if r.read_id in set(dense_matrix.read_ids)
    }
    ll = context_log_likelihoods(
        dense_matrix, dense_read_sequences, allele_sequences, candidates, config,
    )
    return select_template_set(
        matrix.gene, candidates, ll, cn, n_threads=n_threads,
        min_reads_per_haplotype=min_reads_per_haplotype,
        min_haplotype_read_ratio=min_haplotype_read_ratio,
    )

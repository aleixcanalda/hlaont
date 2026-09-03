"""Stage 4 -- typing with per-field posteriors (DESIGN.md S4).

Named `hla_typing.py`, not `typing.py`: the latter shadows the stdlib
`typing` module the moment this directory is on `sys.path` (as it is at
runtime for every stage module here), which mypy refuses to even check --
the development notes' layout table says "4 typing" but that is the stage's name, not
a mandate on the filename.

Replaces FuFiHLA's hard penalty (9999/exonic mismatch, 1/intronic, lowest
wins) with `P(consensus | reference allele)` under the same context-aware
error model stages/genotype.py already uses for template selection --
DESIGN.md is explicit that this stage scores *the consensus*
(stages/consensus.py's polished per-haplotype sequence), not the raw reads
again: the consensus is the best available estimate of the true haplotype
sequence, and typing asks "which known IPD allele does it look like", a
different question from Stage 2's "which template set best explains these
reads".

Implementation reuses machinery rather than re-aligning by hand: the
consensus sequence is wrapped as a single synthetic CandidateRead and run
through stages/assign.py's build_gene_matrices() against the *same* gene
pool Stage 2 used, which gives a one-row GeneAssignmentMatrix (CIGAR spans
included) for free. stages/genotype.py's candidate_alleles() and
context_log_likelihoods() then do the scoring -- there is no separate
alignment or likelihood code in this module.

**Marginalising onto each nomenclature field** (the headline feature,
DESIGN.md S4's worked example) sums the posterior mass of every candidate
allele sharing the called allele's first k fields, for k = 1..n_fields. An
allele with fewer than k fields present cannot share a length-k prefix by
construction and is correctly excluded from that field's mass -- absence of
a field is not evidence for or against a specific value at it, but it is
evidence against being counted as "agreeing" with one.

No haplotype-frequency/LD prior is applied (uniform prior over candidates)
-- DESIGN.md S6 flags such a prior as "dangerous -- it biases against
exactly the rare/novel haplotypes we are hunting" and off by default; this
module does not build the on-by-request path DESIGN.md optionally allows,
since there is no fitted haplotype-frequency resource to apply yet.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import AnchorHit, CandidateRead  # noqa: E402

from assign import (  # noqa: E402
    DEFAULT_POOL_FASTA,
    DEFAULT_POOL_META,
    AlignmentSpan,
    GeneAssignmentMatrix,
    build_gene_matrices,
    realign_to_candidates,
    subset_matrix_by_read_ids,
)
from consensus import HaplotypeConsensus  # noqa: E402
from genotype import (  # noqa: E402
    OntR10Config,
    candidate_alleles,
    context_log_likelihoods,
    load_allele_sequences,
    load_ont_r10_config,
)

DEFAULT_TYPING_PARAMS = REPO_ROOT / "resources" / "models" / "stage4_typing.yaml"

_MIN_P_WRONG = 1e-6
_MAX_GQ = 60.0

# Tolerance for treating two candidates' mismatch rates as "the same"
# before falling back to coverage length as a tiebreak (2026-08-02,
# user-proposed, real HG00097 finding -- see development notes). Checked
# directly against real consensus-vs-candidate alignments: HLA-DRB1,
# HLA-B, and HLA-C's true allele all beat their real close rival on
# mismatch rate alone (a genuine difference of >=0.017 percentage points
# in every case checked); HLA-A's true-vs-wrong pair is an exact 0%/0%
# tie, where preferring the longer (more fully explained) candidate is
# what recovers the correct answer. 0.0001 (0.01 percentage points) sits
# with real headroom under the smallest genuine difference observed and
# above floating-point noise -- provisional (invariant 11).
#
# **Two follow-up attempts tried and reverted the same day (2026-08-04,
# both user-proposed, real HG00097 TAP1 finding), both a net regression
# on the full 37-gene/68-allele real benchmark, neither kept:**
#
# 1. Widen the flat tolerance to 0.001 -- motivated by TAP1*01:01:01:06
#    (a real IPD record 572bp shorter than its true, heavily-voted
#    siblings *:02/*:05) winning called_allele on a mismatch-rate gap of
#    ~0.0008 against them, outside 0.0001 so treated as decisive instead
#    of falling back to coverage. Result: TAP1 itself did not improve
#    (both haplotypes collapsed to the same call, net score unchanged),
#    and eight previously-correct genes broke (HLA-C, HLA-DOB, HLA-DPA1,
#    HLA-DQB1, HLA-DRB1, HLA-DRB3, HLA-G, HLA-L -- 16 correct alleles
#    lost), zero newly fixed (47/68 -> 34/68). A flat delta cannot
#    distinguish "TAP1's gap, plausibly noise given its shallow depth"
#    from "HLA-C/HLA-DRB1's real gap, small in absolute terms but backed
#    by enough aligned length to be a genuine signal".
# 2. Replace the flat tolerance with a depth-aware two-proportion z-test
#    (`nm`/`aln_block_len` as a Bernoulli count per candidate, gap only
#    decisive if it clears z>=1.96) -- motivated by (1)'s own diagnosis:
#    make the tie threshold scale with aligned depth instead of using a
#    bigger flat number. Result was a *larger* regression (47/68 ->
#    26/68, 14 genes broken, zero fixed). Root cause, found while
#    investigating why: the two-proportion model is the wrong lens for
#    this comparison. A candidate's mismatch count against the consensus
#    is not a random draw from a shared noise rate -- it is the fixed,
#    deterministic number of real bases the (mostly-correct) consensus
#    differs from that *specific* candidate's IPD sequence. Two very
#    close alleles routinely differ by just 2-3 real, discriminating
#    bases over several kb -- a significance test almost always calls
#    that gap "not significant" even though it is exactly the correct
#    signal, so the z-test ended up deferring nearly everything to the
#    coverage tiebreak, which is nearly as broad as removing the rate
#    check entirely. No z-critical value fixes this; the modeling
#    assumption itself doesn't fit how these mismatch counts arise.
#
# Back to the flat 0.0001 tolerance below -- a depth/support-aware fix
# for TAP1's shape remains an open problem; it needs a mechanism that
# doesn't treat real, small, deterministic sequence differences as
# sampling noise.
_MISMATCH_RATE_TIE_TOLERANCE = 0.0001

# Minimum secondary-to-primary alignment score ratio for the dense
# realignment fallback in _type_haplotype_from_single_row_matrix() below
# (2026-09-02, real HG00097 TAP2 finding). The shared _MINIMAP2_SCORING_ARGS
# (stages/assign.py) uses -p0.95, which drops a candidate's alignment
# record entirely -- not just ranks it lower -- whenever its best possible
# score against a full-length consensus query falls below 95% of the
# consensus's own best hit purely from matched-length (a real, genuinely
# correct but ~1kb-shorter-UTR IPD record can never clear that bar against
# a longer, mismatch-bearing sibling). 0.5 is a deliberately permissive
# floor, not fitted: this value only controls whether an alignment record
# exists at all for the fallback's own tiny candidate pool -- the
# mismatch-rate-first tie-break just below still judges actual candidate
# quality, so a spurious short/poor alignment let through here loses on
# its own merits rather than winning by default. Provisional (invariant
# 11) -- only exercised when `missing` below is non-empty, so it cannot
# change behaviour for any gene that never hits this fallback at all.
_STAGE4_FALLBACK_MIN_SECONDARY_RATIO = 0.5

# Sentinel gene name for the synthetic single-sequence "read" build_gene_matrices
# aligns the consensus as -- never a real IPD allele id, so it can't collide
# with a pool column (assign.py raises loudly if it ever did).
_CONSENSUS_READ_ID = "consensus"


@dataclass(frozen=True)
class TypingParams:
    ambiguity_log_margin: float

    @classmethod
    def load(cls, path: Path = DEFAULT_TYPING_PARAMS) -> "TypingParams":
        raw = yaml.safe_load(path.read_text())
        return cls(ambiguity_log_margin=float(raw["ambiguity_log_margin"]))


@dataclass(frozen=True)
class TypingCall:
    gene: str
    consensus_template_allele: str  # Stage 3's own selected template, for provenance
    called_allele: str              # Stage 4's nearest-posterior allele for this consensus
    n_fields: int
    field_gq: tuple[float, ...]     # GQ1..GQ(n_fields), phred-scaled
    posterior: dict[str, float]     # candidate allele -> posterior mass
    ambiguity_set: tuple[str, ...]  # candidates within TypingParams.ambiguity_log_margin of the best
    called_allele_span: AlignmentSpan  # consensus-vs-called_allele CIGAR (stages/novelty.py's diff input)


def _fields_of(allele: str) -> tuple[str, ...]:
    if "*" not in allele:
        raise ValueError(f"{allele!r}: not a star-allele name, cannot split into fields")
    return tuple(allele.split("*", 1)[1].split(":"))


def _phred(p_wrong: float) -> float:
    return min(-10.0 * math.log10(max(p_wrong, _MIN_P_WRONG)), _MAX_GQ)


def type_haplotype(
    consensus: HaplotypeConsensus,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    config: OntR10Config | None = None,
    params: TypingParams | None = None,
    extra_candidates: tuple[str, ...] = (),
    reads: Sequence[CandidateRead] | None = None,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
) -> TypingCall:
    """Stage 4 entry point: type one Stage 3 HaplotypeConsensus against its
    gene's full candidate pool. Raises loudly (never a silent best-effort
    guess) if the consensus fails to align to any pool allele at all --
    that is a data/pipeline problem upstream, not a typing outcome.

    **`called_allele` selection (2026-08-02, user-proposed, real HG00097
    finding): mismatch rate first, coverage length only as a tiebreak --
    not `argmax(posterior)`.** `posterior`/`field_gq`/`ambiguity_set` still
    come from `context_log_likelihoods()`'s existing uncovered-length-
    penalised score (unchanged, and still what Stage 2's template
    *selection* uses -- this only changes which candidate Stage 4 reports
    as `called_allele`). That penalised score charges a candidate for every
    base of the consensus it doesn't cover, at a rate steep enough to make
    a short-but-clean-content candidate (e.g. HLA-DRB1*15:01:01:03, a real
    IPD submission with less flanking sequence than its full-length sibling
    *:02, but zero real mismatches where it does align) lose to a longer
    candidate with real mismatches, purely because of how much less it
    covers -- confirmed directly on real consensus-vs-candidate alignments.
    Mismatch rate (real edit count / real aligned length, from the same
    PAF-derived `matrix.nm`/`aln_block_len` this module already has) has no
    such length bias: checked against four real HG00097 cases (HLA-DRB1,
    HLA-A, HLA-B, HLA-C), the true allele has a strictly lower mismatch
    rate than its real close rival in three of them, and is in an exact
    0%/0% tie with a real, genuinely shorter (not lower-quality) rival in
    the fourth (HLA-A) -- where preferring the candidate that explains more
    of the consensus is the correct call. Ties within
    `_MISMATCH_RATE_TIE_TOLERANCE` fall back to coverage length (longer
    wins), never the reverse -- length is only ever a tiebreaker, never
    primary evidence, so a real quality difference always decides first.
    Provisional (invariant 11): two widenings of this tolerance were
    tried and reverted 2026-08-04 -- see the constant's own docstring.

    **A third attempt, 2026-08-05, user-proposed real-depth gate, also
    tried and reverted:** filter candidates to those whose real Stage-1
    per-read vote count (production's stage1_n_best=1 matrix, threaded in
    as an optional `stage1_matrix` parameter) was within 10% of the
    best-supported candidate's, before this rate/coverage rule ran.
    Motivated by TAP1 hap0's real shape (called *01:01:01:06 had only 5
    real Stage-1 votes against its true siblings' 47-59), but real-data
    validated as a net regression on the full 37-gene/68-allele benchmark
    (47/68 -> 46/68): TAP1 itself was unaffected (5/47 = 0.106, just
    *above* the 10% floor, so the thin candidate was never actually
    excluded), while HLA-DRB1 hap1 -- the 2026-08-02 fix's own motivating
    case, `*15:01:01:03`, a real IPD record with less UTR than its
    full-length sibling `*:02` -- regressed, because its own real Stage-1
    vote count falls *below* that same floor for the identical structural
    reason DRB1's whole saga documents: minimap2's own -N secondary-
    alignment reporting inherits the same raw-score length bias as its
    primary ranking (root cause 1, 2026-07-31 session log), so a shorter
    real IPD record recruits disproportionately fewer per-read votes
    regardless of whether it is the correct answer (DRB1) or not (TAP1).
    Real per-read depth is therefore not an independent signal from
    record length/completeness here -- it inherits the identical confound
    coverage-length tiebreaks already have, just measured one stage
    earlier. No tightening/loosening of the ratio is expected to separate
    the two cases cleanly, for the same reason no single flat threshold
    has across every attempt in this area to date.

    **2026-08-28 performance fix (lever 3):** the actual scoring logic
    (everything from candidate selection onward) has moved to
    `_type_haplotype_from_single_row_matrix()`, shared with
    `type_haplotypes_batch()` below -- this function is now just the
    single-consensus convenience wrapper: build a one-read matrix, then
    delegate. Behaviour and signature unchanged."""
    if config is None:
        config = load_ont_r10_config()
    if params is None:
        params = TypingParams.load()

    synthetic_read = CandidateRead(
        read_id=_CONSENSUS_READ_ID,
        sequence=consensus.sequence,
        quality=None,
        hits=(AnchorHit(gene=consensus.gene, allele="__consensus__", coverage=1.0, mismatch_rate=0.0),),
    )
    matrices = build_gene_matrices(
        [synthetic_read], pool_fasta=pool_fasta, pool_meta=pool_meta,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    matrix = matrices.get(consensus.gene)
    if matrix is None or not matrix.read_ids:
        raise ValueError(
            f"{consensus.gene}: consensus for template {consensus.template_allele!r} "
            f"did not align to any allele in {pool_fasta} -- cannot type"
        )
    return _type_haplotype_from_single_row_matrix(
        consensus, matrix, pool_fasta=pool_fasta, config=config, params=params,
        extra_candidates=extra_candidates, reads=reads,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )


def _break_tie_by_real_reads(
    gene: str,
    tied_candidates: tuple[str, ...],
    reads: Sequence[CandidateRead],
    assigned_read_ids: tuple[str, ...],
    *,
    pool_fasta: Path,
    config: OntR10Config,
    n_threads: int,
    minimap2_bin: str,
) -> str | None:
    """Scores `tied_candidates` (all sharing the identical consensus-level
    rate/coverage key -- caller's own tie detection) against this
    haplotype's *real* assigned reads (stages/consensus.py's
    `HaplotypeConsensus.assigned_read_ids`), summing each candidate's
    per-read context log-likelihood the same way stages/genotype.py's
    `select_template_set()` scores its own single-allele baseline --
    reusing the existing scorer, not a new one.

    Deliberately restricted to just `tied_candidates` (typically 2, always
    small): the whole point is to let real per-read vote margins (e.g. 19
    reads for G vs 10 for A at one discriminating position) decide between
    otherwise-identical-looking candidates, which the one collapsed
    consensus sequence cannot represent. No `min_secondary_ratio` override
    here (unlike hla_typing's own dense-realignment fallback above) --
    tied candidates share the same coverage by construction (that's what
    makes them tied), so there is no length disparity between them for
    minimap2's secondary-ratio cutoff to penalise; real individual reads
    are also short enough relative to a full allele that this was never
    the failure mode read-level scoring hits (see `realign_to_candidates()`
    docstring).

    Returns `None` (caller keeps its own default pick) if this haplotype's
    real reads cannot be resolved from `reads` or none of them survive
    realignment against `tied_candidates` -- never silently invents a
    preference from no evidence."""
    read_id_set = set(assigned_read_ids)
    own_reads = [r for r in reads if r.read_id in read_id_set]
    if not own_reads:
        return None
    allele_sequences = load_allele_sequences(pool_fasta, tied_candidates)
    matrix = realign_to_candidates(
        gene, own_reads, allele_sequences, n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    if not matrix.read_ids:
        return None
    read_sequences = {
        r.read_id: r.sequence for r in own_reads if r.read_id in set(matrix.read_ids)
    }
    ll = context_log_likelihoods(matrix, read_sequences, allele_sequences, tied_candidates, config)
    totals = ll.sum(axis=0)
    return tied_candidates[int(np.argmax(totals))]


def _type_haplotype_from_single_row_matrix(
    consensus: HaplotypeConsensus,
    matrix: GeneAssignmentMatrix,
    *,
    pool_fasta: Path,
    config: OntR10Config,
    params: TypingParams,
    extra_candidates: tuple[str, ...] = (),
    reads: Sequence[CandidateRead] | None = None,
    n_threads: int = 1,
    minimap2_bin: str = "minimap2",
) -> TypingCall:
    """The scoring body of `type_haplotype()`, factored out 2026-08-28
    (lever 3, the development notes' phased-pipeline performance investigation) so it
    can run against a matrix `type_haplotypes_batch()` built once for many
    consensuses, not just `type_haplotype()`'s own single-read matrix.
    `matrix` must have exactly one row (this consensus's own synthetic
    read) -- every index below is hardcoded to row 0, same as the
    original single-call code, which is exactly what guarantees
    `candidate_alleles(matrix)` reduces to *this consensus's own* real
    alignment support rather than a cross-row union (see
    `type_haplotypes_batch()`'s own docstring for why that distinction
    matters).

    **`extra_candidates` (2026-09-02, real HG00097 TAP2 finding):**
    `matrix` above comes from a *fresh* whole-pool alignment of the
    consensus as a single synthetic read (`type_haplotype()`/
    `type_haplotypes_batch()`'s own `build_gene_matrices()` call) --
    `candidate_alleles(matrix)` is therefore only ever whatever minimap2
    chose to report for that one query against the *entire* pool. Confirmed
    on real data that this can silently omit a real, correct candidate:
    TAP2*01:01:07:06 is a genuine IPD allele, 972bp shorter than
    TAP2*01:01:07:05 (missing 5'/3' UTR) -- when the consensus (built
    *05-shaped by Stage 2/3) is realigned against the whole pool here,
    minimap2 never reports *06 as a hit at all, because a full-length
    near-perfect match to the longer *05 outscores (on raw matched-length)
    any alignment to the shorter *06 on raw score, even though *06 is the
    true allele and has *fewer* real mismatches over the span it can cover.
    Stage 2's own narrowed candidate set (stages/genotype.py's
    `TemplateCall.candidates`, threaded in here by the caller) already knew
    about *06 from real per-read alignments against the actual reads, not
    the consensus -- so when any of `extra_candidates` is missing from the
    fresh `candidate_alleles(matrix)` result, this densely realigns the
    consensus against the *union* of what was found plus what Stage 2 knew
    about (`assign.realign_to_candidates()`, the identical mechanism
    `select_template_set_ont_r10()`/`build_gene_consensus()` already use
    for the same class of problem) -- guaranteeing every candidate gets a
    real captured span regardless of minimap2's own length bias. This only
    triggers the extra alignment work when something is actually missing;
    the common case (nothing missing) costs nothing extra.

    **QC-gated, not gene-gated (2026-09-02, user-directed):** the extra
    realignment only runs when the *naive* best candidate here is already
    an imperfect match (nm > 0 over its own aligned span) -- a real
    mismatch is the only situation this fix can ever change anything for.
    A candidate with zero real mismatches already reports `exact`
    (stages/novelty.py has nothing to improve on), so there is nothing to
    gain from checking whether a better one was hidden by minimap2's
    length bias, regardless of what `extra_candidates` contains. This
    replaces an earlier per-gene manifest-flag design (TAP1/TAP2/MICA/MICB/
    HLA-DRB3/4/5, reasoned from the development notes' own named false-novel-pattern
    genes) that would have hardcoded which genes benefit -- gating on the
    call's own mismatch rate instead means any gene with this exact
    catalogue-completeness shape is covered automatically, and a gene that
    always aligns cleanly never pays the extra cost, without maintaining a
    list at all."""
    read_id = matrix.read_ids[0]
    candidates = candidate_alleles(matrix)
    if not candidates:
        raise ValueError(f"{consensus.gene}: no candidate alleles with real alignment support")

    naive_col_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    naive_best_rate = min(
        int(matrix.nm[0, naive_col_index[a]]) / int(matrix.aln_block_len[0, naive_col_index[a]])
        for a in candidates
    )

    missing = tuple(a for a in extra_candidates if a not in candidates)
    if missing and naive_best_rate > 0.0:
        all_candidates = tuple(sorted(set(candidates) | set(extra_candidates)))
        allele_sequences_dense = load_allele_sequences(pool_fasta, all_candidates)
        synthetic_read = CandidateRead(
            read_id=read_id, sequence=consensus.sequence, quality=None,
            hits=(AnchorHit(gene=consensus.gene, allele="__consensus__", coverage=1.0, mismatch_rate=0.0),),
        )
        matrix = realign_to_candidates(
            consensus.gene, [synthetic_read], allele_sequences_dense,
            n_threads=n_threads, minimap2_bin=minimap2_bin,
            min_secondary_ratio=_STAGE4_FALLBACK_MIN_SECONDARY_RATIO,
        )
        read_id = matrix.read_ids[0]
        candidates = candidate_alleles(matrix)

    allele_sequences = load_allele_sequences(pool_fasta, candidates)
    read_sequences = {read_id: consensus.sequence}
    ll = context_log_likelihoods(
        matrix, read_sequences, allele_sequences, candidates, config,
    )
    log_probs = ll[0]

    m = float(log_probs.max())
    unnorm = np.exp(log_probs - m)
    posterior = unnorm / unnorm.sum()

    col_index = {a: i for i, a in enumerate(matrix.allele_ids)}

    def _rate_and_coverage(allele: str) -> tuple[float, int]:
        j = col_index[allele]
        nm = int(matrix.nm[0, j])
        aln = int(matrix.aln_block_len[0, j])
        span = matrix.spans[(0, j)]
        return nm / aln, span.read_end - span.read_start

    def _tiebreak_key(i: int) -> tuple[int, int]:
        rate, coverage = _rate_and_coverage(candidates[i])
        return round(rate / _MISMATCH_RATE_TIE_TOLERANCE), -coverage

    best_idx = min(range(len(candidates)), key=_tiebreak_key)

    # Read-level tie-break (2026-09-02, real HG00097 TAP2 finding): the
    # rate/coverage rule above genuinely ties when two candidates differ
    # from each other at a small number of real positions and the
    # consensus's own hard per-position vote happens to split evenly
    # between them (real case: TAP2*01:01:07:03 vs *06, both nm=1/aln=9635
    # against the consensus, at the two *different* positions where those
    # two candidates disagree). The consensus is one collapsed hard call
    # per position; it cannot represent "63% of reads say G here, 37% say
    # A" the way the raw reads can. Only engages for a genuine tie (>1
    # candidate sharing the exact winning key) and only when this
    # haplotype's own real assigned reads are available -- falls back to
    # the existing (deterministic, invariant 9) alphabetical-first choice
    # among `candidates` otherwise, unchanged from before this existed.
    tied_idx = [i for i in range(len(candidates)) if _tiebreak_key(i) == _tiebreak_key(best_idx)]
    if len(tied_idx) > 1 and reads is not None and consensus.assigned_read_ids:
        tied_candidates = tuple(candidates[i] for i in tied_idx)
        refined = _break_tie_by_real_reads(
            consensus.gene, tied_candidates, reads, consensus.assigned_read_ids,
            pool_fasta=pool_fasta, config=config, n_threads=n_threads, minimap2_bin=minimap2_bin,
        )
        if refined is not None:
            best_idx = candidates.index(refined)

    called_allele = candidates[best_idx]
    best_ll = float(log_probs[best_idx])

    allele_col = col_index[called_allele]
    called_allele_span = matrix.spans.get((0, allele_col))
    if called_allele_span is None:
        raise ValueError(
            f"{consensus.gene}: {called_allele!r} was a scored candidate but "
            f"assign.py captured no alignment span for it -- matrix construction bug"
        )

    candidate_fields = {c: _fields_of(c) for c in candidates}
    called_fields = candidate_fields[called_allele]
    n_fields = len(called_fields)
    field_gq = []
    for k in range(1, n_fields + 1):
        prefix = called_fields[:k]
        mass = sum(
            float(p) for c, p in zip(candidates, posterior)
            if candidate_fields[c][:k] == prefix
        )
        field_gq.append(_phred(1.0 - mass))

    ambiguity = tuple(sorted(
        c for c, lp in zip(candidates, log_probs)
        if best_ll - float(lp) <= params.ambiguity_log_margin
    ))

    return TypingCall(
        gene=consensus.gene,
        consensus_template_allele=consensus.template_allele,
        called_allele=called_allele,
        n_fields=n_fields,
        field_gq=tuple(field_gq),
        posterior={c: float(p) for c, p in zip(candidates, posterior)},
        ambiguity_set=ambiguity,
        called_allele_span=called_allele_span,
    )


def type_haplotypes_batch(
    consensuses: Sequence[tuple[str, HaplotypeConsensus]],
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    config: OntR10Config | None = None,
    params: TypingParams | None = None,
    extra_candidates_by_key: dict[str, tuple[str, ...]] | None = None,
    reads: Sequence[CandidateRead] | None = None,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
) -> tuple[dict[str, TypingCall], dict[str, str]]:
    """Stage 4 entry point for many haplotype consensuses at once. Added
    2026-08-28 (the development notes' phased-pipeline performance investigation,
    lever 3): `type_haplotype()` pays for one `build_gene_matrices()`
    minimap2 subprocess call per haplotype -- a real HG00097 sample profiled
    at up to 68 unphased + 59 phased = 127 individually-full-pool-reloaded
    calls. This wraps every consensus as its own uniquely-keyed synthetic
    read and aligns the whole batch in a single call, the identical pattern
    stages/assign.py's own Stage 1 already uses for the sample's real
    reads.

    **Each consensus is scored against a single-row *subset* of the shared
    batched matrix, not the shared multi-row matrix directly.**
    `candidate_alleles()`'s "any row" union (its own docstring) is correct
    when many real reads share one candidate search (Stage 2); it is wrong
    here, where every row is an *independent* haplotype call that must not
    see another haplotype's candidate neighbourhood -- two real, unrelated
    consensuses for the same gene batched together would otherwise inflate
    each other's candidate set with alleles neither ever actually aligned
    to, and `_type_haplotype_from_single_row_matrix()`'s hardcoded-row-0
    lookups (`matrix.spans[(0, j)]`, `matrix.nm[0, j]`) would raise/divide-
    by-zero on any such spurious candidate that happens to have zero real
    alignment in *this* row. `assign.subset_matrix_by_read_ids()` (lever 2,
    same session) collapses `candidate_alleles()`'s union back down to
    exactly this consensus's own real support and keeps every downstream
    single-row assumption in the shared scoring function correct --
    behaviour is byte-identical to calling `type_haplotype()` once per
    consensus, just without the redundant subprocess calls.

    **Per-consensus fault isolation, not a single all-or-nothing call.**
    `type_haplotype()`'s "did not align to any allele" / "no candidate
    alleles with real alignment support" failures are real, expected
    per-haplotype outcomes (module docstring's own known failure modes for
    template-thin genes) that must not take down every other haplotype's
    real result in the same batch -- mirrors the per-gene/per-haplotype
    fault isolation stages/pipeline.py's own module docstring already
    documents as a hard requirement. Returns `(typing_calls, failures)`,
    both keyed by the caller's own `consensuses` keys (e.g. an f"{gene}#
    {h_idx}" string, used only for this lookup and for
    build_gene_matrices()'s per-read gene routing -- never written to any
    output file); a key present in `consensuses` is in exactly one of the
    two returned dicts, never both, never neither.

    `extra_candidates_by_key` (2026-09-02, real HG00097 TAP2 finding, see
    `_type_haplotype_from_single_row_matrix()`'s own docstring): optional
    per-key fallback candidate set (e.g. stages/genotype.py's
    `TemplateCall.candidates`, Stage 2's own narrowed list for that gene)
    -- used only when the batched whole-pool alignment above fails to
    report one of them as a hit for that specific consensus. `None`
    (default) behaves exactly as before this parameter existed.

    `reads` (2026-09-02, same finding): the sample's Stage 0 sieved reads,
    passed through so `_type_haplotype_from_single_row_matrix()`'s
    read-level tie-break can look up each haplotype's own real assigned
    reads (`HaplotypeConsensus.assigned_read_ids`) when the consensus-level
    rate/coverage rule genuinely ties between candidates. `None` (default)
    disables the tie-break entirely (falls back to the prior deterministic
    alphabetical-first behaviour) -- never required."""
    if config is None:
        config = load_ont_r10_config()
    if params is None:
        params = TypingParams.load()
    if extra_candidates_by_key is None:
        extra_candidates_by_key = {}
    if not consensuses:
        return {}, {}

    synthetic_reads = [
        CandidateRead(
            read_id=key, sequence=hap.sequence, quality=None,
            hits=(AnchorHit(gene=hap.gene, allele="__consensus__", coverage=1.0, mismatch_rate=0.0),),
        )
        for key, hap in consensuses
    ]
    matrices = build_gene_matrices(
        synthetic_reads, pool_fasta=pool_fasta, pool_meta=pool_meta,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )

    typing_calls: dict[str, TypingCall] = {}
    failures: dict[str, str] = {}
    for key, hap in consensuses:
        matrix = matrices.get(hap.gene)
        if matrix is None or key not in matrix.read_ids:
            failures[key] = (
                f"{hap.gene}: consensus for template {hap.template_allele!r} "
                f"did not align to any allele in {pool_fasta} -- cannot type"
            )
            continue
        row_matrix = subset_matrix_by_read_ids(matrix, {key})
        assert row_matrix is not None  # key in matrix.read_ids guarantees >=1 surviving row
        try:
            typing_calls[key] = _type_haplotype_from_single_row_matrix(
                hap, row_matrix, pool_fasta=pool_fasta, config=config, params=params,
                extra_candidates=extra_candidates_by_key.get(key, ()), reads=reads,
                n_threads=n_threads, minimap2_bin=minimap2_bin,
            )
        except ValueError as e:
            failures[key] = str(e)
    return typing_calls, failures

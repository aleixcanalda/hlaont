"""Orchestration: Stage 0 -> assign -> cn -> genotype -> consensus ->
hla_typing -> novelty, over every gene in the manifest (development notes
invariant 5 -- never a hardcoded gene list; whatever resources/genes.yaml
lists is what gets attempted).

This is the "Next" item the development notes' own progress log names: individual
stage modules existed and were unit-tested, but nothing ran them together
over a real sample end to end. `run_sample()` is that wiring.

**CN estimate is the fan-out point.** Every non-excluded, non-structural
gene in the manifest is considered, not just ones stages/assign.py happened
to build a matrix for -- a gene with a positive CN estimate but *zero*
reads recruited to its own matrix is a distinct, explicitly reported
failure ("template-recruitment failure candidate", DESIGN.md S2c's HLA-Y
warning), not silently skipped the same way as a gene with no calibrator
signal at all. Both are recorded in `SampleResult.failures`, never a
silent omission from calls.tsv. CN is estimated per gene, not pooled across
resources/genes.yaml's `segment` field (stages/cn.py's own docstring has
the full rationale/revision history) -- each gene's estimate depends only
on its own read depth.

CN is rounded to the nearest integer to size Stage 2's template-set search
(`cn_round`, injectable, defaults to `_round_cn_estimate` -- see that
function's own docstring for the 2026-08-23 asymmetric-threshold revision)
-- DESIGN.md documents CN as an estimate with a confidence interval, not an
integer, and this rounding step is a provisional policy choice (invariant
11: not fitted or re-derived against real data yet), flagged here rather
than presented as principled.

**Revised 2026-07-27, real HG00097 ONT benchmark -- `max_cn` (default 2) and
LRT-based homozygous collapse.** Two real, distinct failure modes showed up
running against real data for the first time (the development notes' own "Next" item):
(1) a noisy point estimate rounding *up* past the true CN -- HLA-C's
depth-based estimate (2.83, CI 1.76-3.89) rounded to 3, HLA-DRB1's (3.67,
CI 2.57-4.77) rounded to 4 -- forcing Stage 2 to search for 3rd/4th
distinct alleles that do not exist, producing visibly spurious low-read,
high-variant-count "haplotypes" (DRB1's 3rd/4th calls: 2 and 10 reads, 589
and 157 variants against their own selected template -- noise, not calls).
`cn_int` is now clamped to `max_cn` after rounding -- not a claim that no
HLA gene here can ever legitimately exceed 2 copies (RCCX/C4 genuinely can,
DESIGN.md S2f, but C4A/C4B are already out of this function's scope, module
docstring in stages/cn.py), rather that real depth-estimate noise inflating
past 2 is far better evidenced right now than genuine >2-copy duplication
in this manifest, and the raw uncapped `cn_estimate`/CI remain untouched in
`SampleResult.gene_cn` for diagnosis -- only the search-sizing integer is
clamped. (2) `select_template_set()`'s `cn=2` search always returns two
*distinct* alleles (`itertools.combinations` cannot repeat one), even for a
genuinely homozygous sample -- stages/consensus.py's own docstring already
named this gap and exactly how to close it ("a manufactured single-allele
TemplateCall... belongs to the orchestration layer once it exists"): when
`TemplateCall.lrt_statistic` falls below `min_heterozygous_lrt` (default
10.0, chosen between the two reference points stages/test_genotype.py's own
existing tests already assert -- a real heterozygous pair scores >20, a
synthetic homozygous sample scores <5 -- not independently fitted, invariant
11), the call is collapsed to `call.single_allele` at weight 1.0 before
Stage 3 runs, so every read in the gene's matrix builds *one* higher-depth
consensus instead of being split across two haplotype clusters where one is
pure noise. `cn_int` itself (the true estimated copy number) is still
recorded on that single reported row -- collapsing removes a spurious
second *distinct-allele* call, not the fact that two copies are present.

**Same session: `genes.yaml`'s new `low_cn_floor` field (DRB3/4/5, user's
own hypothesis).** DRB1 sits in the same physical block and is essentially
always present at real depth (its own raw estimate was 3.67 the same
benchmark run); a thin raw estimate for its presence/absence paralogs here
is plausibly cross-mapped DRB1 read leakage, not genuine partial presence
-- `assign.build_gene_matrices()`'s new `max_read_mismatch_rate` gate
(its own docstring) is the read-level mitigation for that same leakage,
this is a CN-decision-level backstop on top of it. A gene with
`low_cn_floor` set is reported as CN=0 (through the same `cn_int < 1`
failure path every other absent gene uses, invariant 4 -- not a silent
drop) when its raw estimate falls below that floor, rather than rounding
normally to 1. Data-driven per invariant 5 -- the gene list lives in
`genes.yaml`, not a hardcoded set here.

**Revised 2026-07-30 -- `min_reads_per_haplotype`, real HG00097 finding
after the row_span scoring fix.** `min_heterozygous_lrt` alone turned out
not to be the right test: real HLA-B data (truth homozygous
`07:02:01:01`/`07:02:01:01`) produced `lrt_statistic=847` -- nowhere near
the 10.0 collapse threshold -- yet the hard read partition
(`stages/consensus.py`'s `assign_reads_to_haplotypes`, argmax
responsibility) put *zero* of 61 real reads on the second allele
(`HLA-B*40:01:02:40`). A diffuse, sub-single-read signal spread thinly
across many reads' marginal preferences is enough to lift a continuous EM
mixture weight above zero and inflate the LRT, without any individual read
ever actually crossing over -- the LRT measures whether the aggregate
likelihood improved at all, not whether a real second haplotype exists.
Left unchecked, `stages/consensus.py`'s own zero-observation fallback
(`_call_substitution_or_deletion`'s `if not observations: return
template_base, 0.0, 0`) then builds that phantom haplotype's "consensus"
as a verbatim copy of its seed template, QV=0 throughout -- which Stage 4
confidently re-types as an exact match to itself, a real-looking call with
zero read evidence behind it. `n_reads_assigned` (already computed by
`build_gene_consensus`, no extra alignment work) is the direct signal the
LRT is a proxy for: after building consensus, if any haplotype in a
`cn > 1` call has fewer than `min_reads_per_haplotype` reads actually
assigned to it, the call is collapsed to `call.single_allele` the same way
the LRT path already does (factored into `_collapse_to_single_allele`,
shared by both checks) and consensus is rebuilt once more against the
collapsed call. The LRT check stays as a cheap first pass (skips a wasted
consensus build for genuinely low-LRT genes); this is a second,
read-count-based check afterward for the cases the LRT does not catch.

`min_reads_per_haplotype` default is 3, not 2 -- revised same session,
against a second real case. An initial 2 fixed HLA-B (the case above) but
left a real MICA artifact through: its phantom second haplotype had
*exactly* 2 reads, `min_qv=0` (real coverage gaps) and 48 variants against
its own seed template -- unambiguous noise, just one read short of the
first guess's floor. Still not independently fitted (invariant 11) --
raised on confirmed evidence from a second real gene, not re-derived from
first principles.

**Same session, `min_haplotype_read_ratio` (default 0.1), user-requested.**
`min_reads_per_haplotype` is an absolute floor -- it does nothing for a
gene with genuinely high depth where the *minor* haplotype's read count
clears 3 in absolute terms but is still a small fraction of the major
haplotype's, e.g. 8 reads vs 400: not noise by the absolute floor, but
still evidence the second "distinct" allele is a trickle of cross-mapped
or misassigned reads riding on a real, dominant homozygous signal rather
than a genuine second haplotype. After the two existing checks, if a
`cn > 1` call's two `n_reads_assigned` counts have `min(counts) /
max(counts) < min_haplotype_read_ratio`, collapse the same way (shared
`_collapse_to_single_allele`) -- the higher-depth haplotype's allele is
`call.single_allele` already (it is independently the best-supported
single allele across every read in the matrix, not merely the survivor of
the pair), so collapsing reports that allele as homozygous, not an
arbitrary pick between the two. 0.1 is the user's own stated threshold,
not fitted against real data -- provisional, invariant 11, same as the
other two collapse constants above.

**Same session, `cn_model`-aware search sizing, user-requested.** Real
HG00097 data showed a class of false-homozygous call distinct from the
LRT/floor/ratio gap above: TAP1, TAP2, HLA-DQB2, HLA-DPA2, HLA-DPB1 and
HLA-DQA1/DQA2 all have raw `cn_estimate`s landing at 1.3-1.7 on modest
depth (~40-55 reads) -- Poisson counting noise on the depth-ratio
estimator (their 95% CIs mostly straddle 2.0; a length-bias hypothesis was
checked directly against real depth/consensus-length data and rejected,
r=0.067), not evidence of real hemizygosity. Plain `round()` sends these
to `cn_int=1`, and `select_template_set()`'s cn=1 path returns immediately
with a trivial single-allele call (`lrt_statistic` hardcoded to 0.0) --
the gene never even reaches a cn=2 combination search, so none of the
collapse checks above get a chance to run either way; a real second
haplotype is structurally unreachable regardless of how much read evidence
supports it. `genes.yaml`'s own `cn_policy` note already says `cn_model`
is meant as "an expected-pattern label for CI width... never a hardcoded
assumption" and that any deviation from a gene's typical pattern "requires
segment-level support or reference-space corroboration before being
reported" -- corroboration this FASTQ-path run cannot provide (module
docstring's own invariant-4 framing). Read the other way: for a
`cn_model: diploid_typical` gene (`stages/cn.py`'s `GeneManifestEntry`,
newly exposed this session), the raw estimate is now used *only* to gate
presence vs absence (rounds to 0 -> absent, same failure path as before);
once present, `cn_int` defaults to `max_cn` unconditionally rather than
the noisy round-and-clamp, so the gene always gets a real cn=2 search and
the LRT/`min_reads_per_haplotype`/`min_haplotype_read_ratio` checks above
-- already built for exactly this decision on the CN-variable genes -- are
what settle homozygous-vs-heterozygous, not a pre-Stage-2 rounding
artifact. `variable_0_2`/`variable` genes (DRB3/4/5, MICA, the class I
pseudogenes) are untouched -- a real CN=1 is a common, genuine state for
these (DRB3 present on only one real haplotype in this very sample) and
must stay distinguishable, so they keep the original round-and-clamp path.
**Known tradeoff, flagged not hidden:** a diploid_typical gene with a real
single-copy deletion now always searches and reports `cn=2` even when
truth is 1 -- but per the `cn_policy` note there was never a corroboration
path that would justify reporting anything other than the typical pattern
from a bare depth ratio anyway, so this is a more consistent application
of the existing honesty bar, not a new relaxation of it. New unit test in
`stages/test_pipeline.py`: a diploid_typical fixture with `cn_estimate`
~1.4 and real heterozygous 2-allele read support, confirming the pipeline
now searches cn=2 and recovers both alleles (the old round-and-clamp path
would have searched cn=1 and never looked).

**Per-gene/per-haplotype fault isolation.** A single template-thin
pseudogene failing Stage 2 (too few candidates for its estimated CN) must
not take down a 36-gene sample-wide run -- population scale means this
happens routinely, not exceptionally. Failures are caught narrowly
(`ValueError`, the only exception type any downstream stage module raises
deliberately) and recorded in `SampleResult.failures` with a reason string;
anything else propagates, since an unexpected exception type is a real bug,
not a data-quality event this layer is designed to absorb.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import (  # noqa: E402
    DEFAULT_ANCHORS,
    DEFAULT_ANCHORS_META,
    CandidateRead,
    SieveParams,
    SieveStats,
    sieve_fastq,
)
from phasing import ReadPhase, load_read_phase_tags  # noqa: E402

from assign import (  # noqa: E402
    DEFAULT_POOL_FASTA,
    DEFAULT_POOL_META,
    GeneAssignmentMatrix,
    build_gene_matrices,
    classical_matrices_for_reads,
    subset_matrix_by_read_ids,
)
from cn import (  # noqa: E402
    DEFAULT_GENES_YAML,
    GeneCN,
    GeneManifestEntry,
    estimate_cn,
    load_gene_manifest,
)
from chr17_control import (  # noqa: E402
    DEFAULT_CHR17_CONTROL_FASTA,
    compute_chr17_control_depth,
)
from c4_cn import (  # noqa: E402
    C4_GENES,
    C4_REPORTED_GENE,
    DEFAULT_C4_POOL_FASTA,
    DEFAULT_C4_POOL_META,
    calibrator_genes_for_c4,
    estimate_c4_cn,
)
from c4_chr17_cn import (  # noqa: E402
    DEFAULT_C4_FLANKED_FASTA,
    C4Chr17ControlResult,
    estimate_c4_cn_via_chr17_control,
)
from c4_form import (  # noqa: E402
    C4AbCall,
    C4FormLengthCall,
    C4PerGeneCN,
    estimate_c4_form,
    estimate_c4_per_gene_cn,
)
from consensus import (  # noqa: E402
    HaplotypeConsensus,
    build_gene_consensus,
    build_phased_pair_consensus,
)
from genotype import (  # noqa: E402
    OntR10Config,
    TemplateCall,
    load_ont_r10_config,
    select_template_set_ont_r10,
)
from hla_typing import TypingCall, TypingParams, type_haplotypes_batch  # noqa: E402
from novelty import (  # noqa: E402
    DEFAULT_CDS_CATALOGUE,
    DEFAULT_POOL_MODELS,
    NoveltyCall,
    classify,
)

@dataclass(frozen=True)
class HaplotypeCallResult:
    gene: str
    haplotype_index: int
    cn: int
    called_allele: str
    proposed_name: str
    taxonomy: str
    candidate_null: bool
    field_gq: tuple[float, ...]
    ambiguity_set: tuple[str, ...]
    n_reads_assigned: int
    consensus_length: int
    mean_qv: float
    min_qv: int
    n_variants: int


@dataclass(frozen=True)
class GeneFailure:
    gene: str
    reason: str


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    sieve_stats: SieveStats
    gene_cn: dict[str, GeneCN]
    haplotype_calls: tuple[HaplotypeCallResult, ...]
    consensus_records: tuple[HaplotypeConsensus, ...]
    novelty_calls: dict[tuple[str, int], NoveltyCall]
    failures: tuple[GeneFailure, ...]
    # DESIGN.md S2f Phase 2 (stages/c4_form.py) -- a factual read-count
    # tally, not a hard genotype call, so a real run with e.g. zero C4
    # reads at all still gets a real, empty-count call object here, never
    # None; defaulted only so this doesn't become a required positional
    # arg at every other SampleResult call site (there are none besides
    # run_sample() itself today, but frozen-dataclass field ordering
    # otherwise forces every field after it to also take a default).
    c4_ab_call: C4AbCall | None = None
    c4_form_call: C4FormLengthCall | None = None
    # User-requested, 2026-08-27: CN_A/CN_B (a proportional split of the
    # combined C4 CN above by the Phase 2 vote ratio -- see
    # estimate_c4_per_gene_cn()'s own docstring). Same defaulting rationale
    # as the two fields above.
    c4_per_gene_cn: C4PerGeneCN | None = None
    # DESIGN.md S10: one entry per resources/genes.yaml manifest gene, plus
    # one extra entry keyed C4_REPORTED_GENE ("C4") for the combined
    # C4A/C4B read pool's own routing decision (S10.6) -- only populated by
    # run_sample_phased(). None means this sample was run unphased
    # (run_sample()) and the distinction never arose.
    phasing_routing: "dict[str, GenePhaseRouting] | None" = None
    # DESIGN.md S10.6: per-haplotype C4A/C4B form discrimination (Phase 2
    # only -- vote tallies from stages/c4_form.py's classify_a_vs_b/
    # classify_long_vs_short, re-run on each haplotype's own C4-informative
    # reads). Keyed by real HP value (1/2); only present for haplotypes that
    # cleared min_reads_per_haplotype. None unless C4_REPORTED_GENE's own
    # GenePhaseRouting.routing == "phased".
    c4_phased_ab_calls: "dict[int, C4AbCall] | None" = None
    c4_phased_form_calls: "dict[int, C4FormLengthCall] | None" = None
    # DESIGN.md S10.5/S10.6, added 2026-08-28: per-haplotype total C4 CN
    # (Phase 1), keyed by real HP value -- only present for a haplotype
    # whose own C4-informative reads cleared the phase-routing floor *and*
    # whose per-haplotype calibrator panel had at least one usable gene
    # (see run_sample_phased()'s own docstring for the PS-matching
    # mechanism this requires; a haplotype can clear the first without the
    # second, in which case its GeneCN.cn_estimate is None -- present but
    # honestly uncertain, not a guess and not silently absent). None unless
    # C4_REPORTED_GENE's own GenePhaseRouting.routing == "phased".
    # gene_cn["C4"] remains the separate sample-wide aggregate regardless.
    c4_phased_cn: "dict[int, GeneCN] | None" = None
    c4_phased_per_gene_cn: "dict[int, C4PerGeneCN] | None" = None
    # 2026-09-01, user-directed: the chr17-control mechanism
    # (stages/c4_chr17_cn.py) -- real per-form (C4AL/C4AS/C4BL/C4BS)
    # samtools-depth CN, now production (see write_calls_tsv's own
    # docstring). Unphased only for now -- the phased C4 path above
    # (c4_phased_cn/c4_phased_per_gene_cn) still uses the older,
    # calibrator-panel mechanism; extending the chr17-control approach to
    # per-haplotype C4 is real, unstarted follow-up work, not silently
    # covered by this field.
    c4_chr17_control: "C4Chr17ControlResult | None" = None


@dataclass(frozen=True)
class GenePhaseRouting:
    """DESIGN.md S10.3: per-gene record of whether this gene's own
    recruited reads were split by real HP/PS phase tags ("phased") or fell
    back to the ordinary unphased combinatorial path ("unphased_fallback"),
    and why -- user-requested, so this is inspectable in qc.json rather than
    only re-derivable from re-reading the code.
    """

    gene: str
    routing: str  # "phased" | "unphased_fallback"
    reason: str
    ps_values_seen: tuple[int, ...]
    n_reads_hp1: int
    n_reads_hp2: int
    n_reads_untagged: int
    # Which HP values (1/2) actually produced a real per-haplotype call --
    # only meaningful when routing == "phased"; always () otherwise.
    haplotypes_called: tuple[int, ...]


def _round_cn_estimate(cn_estimate: float) -> int:
    """Default `cn_round` (2026-08-23, user-requested) -- replaces plain
    `round()`'s symmetric 0.5/1.5 thresholds with asymmetric ones (>0.25 ->
    1, >1.25 -> 2) biased toward calling presence, per invariant 4 ("a CN=0
    call is guilty until proven innocent"). Motivated by the real HG00097
    benchmark's HLA-U (cn_estimate 0.41, present on ~86.5% of real hla_seq/
    haplotypes per genes.yaml's own audit note) rounding to 0 and dropping
    out of the output entirely under plain round()'s >=0.5 bar -- the same
    failure shape invariant 4 already names for HLA-Y. Applies everywhere
    `cn_round` is called (both the `diploid_typical` presence check and the
    general CN-variable-gene sizing branch below); `low_cn_floor`'s own
    dedicated gate (DRB3/4/5) runs first and is unaffected, since it never
    reaches this function when it fires. Provisional (invariant 11) -- not
    fitted or re-derived against real data, flagged here rather than
    presented as principled, same as the `round()` default it replaces."""
    if cn_estimate > 1.25:
        return 2
    if cn_estimate > 0.25:
        return 1
    return 0


def _raw_cn_int_for_gene(
    entry: GeneManifestEntry, cn_estimate: float, cn_round: Callable[[float], int], max_cn: int,
) -> int:
    """The per-gene cn_int decision (low_cn_floor / diploid_typical / plain
    round-and-clamp), factored out of the Stage 2-5 loop so the same logic
    backs both the DRB3/4/5 mutual-exclusivity pre-pass below and the loop
    itself. `cn_estimate` is the gene's own already-non-None GeneCN.cn_estimate
    (a plain float parameter, not the GeneCN itself, so mypy can see it's
    non-None at every call site without re-deriving that here)."""
    if entry.low_cn_floor is not None and cn_estimate < entry.low_cn_floor:
        # genes.yaml's low_cn_floor (DRB3/4/5) -- below this, a thin
        # estimate is more plausibly cross-mapped read leakage from a
        # well-covered segment neighbour than genuine partial presence;
        # treat as absent rather than rounding normally to 1.
        return 0
    # `diploid_typical`'s own unconditional-max_cn forcing (present the
    # 2026-07-30 revision built it on real, then-current data: a raw
    # depth-ratio estimate landing at 1.4-1.6 and rounding down to 1 was
    # Poisson counting noise on a thin read count, not a real single-copy
    # deletion, because Stage 1's matrix-derived depth (assign.py's
    # best_gene_per_read()) was too noisy to trust a CN=1 reading on its
    # own) is removed as of 2026-08-31, user-directed: stages/cn.py's
    # depth source changed the same session (see that module's own
    # docstring) specifically because the old signal was shown, on real
    # data, to systematically *undercount* HLA-H/K/T/U/W -- the new one
    # does not carry that same undercounting bias for diploid_typical
    # genes (real HG00097/HG00320 checks: every diploid_typical gene's
    # raw estimate already lands close to 2 under the new signal without
    # forcing), so trusting a genuine low estimate here is no longer the
    # same gamble it was. `max_cn` stays a hard ceiling (invariant 11 --
    # still an unfitted noise guard against a stray high estimate forcing
    # a 3rd/4th-allele search), but there is deliberately no forced floor
    # any more: a gene supported by real, low depth now reports CN=1 (or
    # 0) honestly instead of always searching for 2, since invariant 4's
    # "CN=0 is guilty until proven innocent" cuts the other way once the
    # estimate itself is trustworthy -- a hidden real deletion is exactly
    # the case forcing max_cn would silently mask.
    return min(int(cn_round(cn_estimate)), max_cn)


_DRB_MUTUALLY_EXCLUSIVE_GENES = ("HLA-DRB3", "HLA-DRB4", "HLA-DRB5")
_DRB_AMBIGUOUS_CN_LOW = 1.2
_DRB_AMBIGUOUS_CN_HIGH = 2.0


def _resolve_drb345_mutual_exclusivity(
    cn_int_by_gene: dict[str, int], gene_cn: dict[str, GeneCN],
) -> dict[str, int]:
    """User-requested, 2026-08-26. DRB3/DRB4/DRB5 occupy the same DRB-frame
    haplotype slot -- a real haplotype carries at most one of the three
    (never two), so their `cn_int`s can sum to at most 2 across a diploid
    sample, never 3. The per-gene rounding above (`_round_cn_estimate`,
    `>1.25 -> 2`) has no visibility into the other two genes' own calls, so
    a genuinely borderline gene (raw estimate just past that cutoff, e.g.
    1.49) can independently round to 2 in the same run where a second gene
    in the group independently rounds to a confident 1 -- summing to a
    biologically impossible 3 (real HG00097 shape: DRB3 1.49 -> 2, DRB5
    1.02 -> 1). Reads that specific pattern -- one gene still in the raw
    (1.2, 2.0) ambiguous band that rounded up to 2, another already at a
    confident 1 -- as heterozygous across the two genes (one haplotype
    each) and pulls the ambiguous one down to 1 rather than 2. Does not
    touch a gene that already rounds to 0, or fire when the group's sum is
    already <= 2 (the ordinary, non-conflicting case)."""
    present = {g: cn_int_by_gene[g] for g in _DRB_MUTUALLY_EXCLUSIVE_GENES if g in cn_int_by_gene}
    if sum(present.values()) <= 2:
        return cn_int_by_gene
    out = dict(cn_int_by_gene)
    for gene in sorted(present):  # invariant 9: deterministic iteration order
        if out[gene] != 2:
            continue
        estimate = gene_cn[gene].cn_estimate
        if estimate is None or not (_DRB_AMBIGUOUS_CN_LOW < estimate < _DRB_AMBIGUOUS_CN_HIGH):
            continue
        has_confident_partner = any(other != gene and out[other] == 1 for other in present)
        if has_confident_partner and sum(out[g] for g in present) > 2:
            out[gene] = 1
    return out


def _collapse_to_single_allele(call: TemplateCall) -> TemplateCall:
    """Manufacture a single-allele TemplateCall from `call`'s own
    single-allele baseline (already computed by Stage 2's search regardless
    of which `cn` was searched) -- the orchestration-layer decision
    stages/consensus.py's own docstring names but leaves unmade. `cn` stays
    at the original estimated copy number; only the distinct-allele count
    collapses, not the copy number itself. Shared by both collapse checks
    in the per-gene loop below (module docstring)."""
    return TemplateCall(
        gene=call.gene, cn=call.cn, alleles=(call.single_allele,),
        weights=(1.0,), log_likelihood=call.single_allele_log_likelihood,
        n_candidates=call.n_candidates, lrt_statistic=call.lrt_statistic,
        single_allele_log_likelihood=call.single_allele_log_likelihood,
        single_allele=call.single_allele,
        candidates=call.candidates,
    )


def _build_haplotype_call_result(
    gene: str,
    h_idx: int,
    cn_int: int,
    hap: HaplotypeConsensus,
    typing_call: TypingCall,
    novelty_call: NoveltyCall,
) -> HaplotypeCallResult:
    """Assembles the final per-haplotype output row from Stage 3's
    consensus plus Stage 4/5's already-computed typing/novelty calls --
    pure, no alignment work. Factored out of the old
    `_type_and_classify_haplotype()` 2026-08-28 (lever 3, the development notes'
    phased-pipeline performance investigation) so both the classical and
    phased per-gene loops share it once Stage 4 typing runs batched
    (`hla_typing.type_haplotypes_batch()`) rather than once inline per
    haplotype."""
    mean_qv = (sum(hap.qv) / len(hap.qv)) if hap.qv else 0.0
    min_qv = min(hap.qv) if hap.qv else 0
    result = HaplotypeCallResult(
        gene=gene, haplotype_index=h_idx, cn=cn_int,
        called_allele=typing_call.called_allele,
        proposed_name=novelty_call.proposed_name,
        taxonomy=novelty_call.taxonomy,
        candidate_null=novelty_call.candidate_null,
        field_gq=typing_call.field_gq,
        ambiguity_set=typing_call.ambiguity_set,
        n_reads_assigned=hap.n_reads_assigned,
        consensus_length=len(hap.sequence),
        mean_qv=mean_qv, min_qv=min_qv,
        n_variants=len(novelty_call.variants),
    )
    return result


def _run_sample_core(
    sample_id: str,
    fastq_path: Path,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    models_path: Path = DEFAULT_POOL_MODELS,
    cds_catalogue_path: Path = DEFAULT_CDS_CATALOGUE,
    c4_pool_fasta: Path = DEFAULT_C4_POOL_FASTA,
    c4_pool_meta: Path = DEFAULT_C4_POOL_META,
    genes_yaml: Path = DEFAULT_GENES_YAML,
    anchors_path: Path = DEFAULT_ANCHORS,
    anchors_meta_path: Path = DEFAULT_ANCHORS_META,
    chr17_control_fasta: Path = DEFAULT_CHR17_CONTROL_FASTA,
    c4_flanked_fasta: Path = DEFAULT_C4_FLANKED_FASTA,
    compute_c4_chr17_control: bool = True,
    control_depth: float | None = None,
    control_n_reads: int = 0,
    sieve_params: SieveParams | None = None,
    ont_config: OntR10Config | None = None,
    typing_params: TypingParams | None = None,
    cn_round: Callable[[float], int] = _round_cn_estimate,
    max_cn: int = 2,
    min_heterozygous_lrt: float = 10.0,
    min_reads_per_haplotype: int = 3,
    min_haplotype_read_ratio: float = 0.1,
    max_read_mismatch_rate: float = 0.06,
    stage1_n_best: int = 1,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
    progress: Callable[[str], None] | None = None,
) -> tuple[SampleResult, list[CandidateRead], dict[str, GeneAssignmentMatrix], dict[str, GeneAssignmentMatrix]]:
    """`run_sample()`'s actual body, plus its own Stage 0 sieved reads and
    Stage 1 (classical + C4) matrices exposed to the caller. Factored out
    2026-08-28 (the development notes' phased-pipeline performance investigation,
    lever 1) so `run_sample_phased()` can reuse this work directly instead
    of re-sieving the FASTQ a second time (measured 89s on real HG00097
    data) and reuse the per-gene matrices (lever 2, see
    `assign.subset_matrix_by_read_ids()`) instead of rebuilding them per
    haplotype. `run_sample()` itself is now a thin wrapper that discards
    the two extra return values -- behaviour and public signature
    unchanged for every existing caller.

    progress, if given, is called with one short label per stage
    boundary -- interactive single-sample feedback (cli.py wires it to a
    stderr line with elapsed time). No-op by default so a cohort-scale
    caller running this per-sample across UKB doesn't get unwanted stdout
    noise (DESIGN.md's population-scale framing).

    `stage1_n_best` (default 1, real HG00097 ONT benchmark finding)
    overrides assign.build_gene_matrices()'s own `_BEST_N` (512) default
    for this Stage 1/2 per-read matrix specifically -- see that function's
    docstring for why 512 remains right for stages/hla_typing.py's
    separate Stage 4 re-typing call (a single synthetic "read" needing
    many reported candidates for ambiguity_set/field_gq) but wrong here: a
    real per-read matrix at 512 let one read report a real alignment
    against dozens to hundreds of alleles simultaneously, which is what
    let both a redundant-name cluster (HLA-A) and diffuse paralog crowding
    across a dense catalogue (HLA-DRB1) out-compete the true allele on
    read depth. Restricting each read to its single best-scoring hit fixed
    both, confirmed directly against real data, without requiring
    genotype.py's narrow_candidates_by_mismatch_rate() to compensate after
    the fact. Provisional (invariant 11) -- validated on one real sample.

    `control_depth`, if given, bypasses the real chr17-control alignment
    pass below and is used directly. Two legitimate kinds of caller do
    this: the test suite (dependency injection for tests that already know
    the value they want CN normalised against and would otherwise need a
    real, matching synthetic chr17 reference plus reads that hit it just to
    exercise unrelated pipeline behaviour), and, 2026-09-01, `cli.py`'s own
    `--bam` path, which can read this off the mini-BAM's own real alignment
    via `chr17_control.compute_chr17_control_depth_from_bam()` instead of
    paying for a second, redundant whole-FASTQ realignment -- a real,
    computed value, not a synthetic one, so this is a genuine production
    use of the override, not just a test seam. `control_n_reads` is the
    matching read-count companion value (reporting only, module docstring
    of `chr17_control.py`); it is ignored unless `control_depth` is also
    given. `None` (the default) means "compute it for real" via the
    whole-FASTQ pass -- callers with no cheaper real alignment to read it
    from must not fake one.
    """
    if ont_config is None:
        ont_config = load_ont_r10_config()
    if typing_params is None:
        typing_params = TypingParams.load()

    if progress:
        progress("Stage 0: sieving reads")
    stats = SieveStats()
    reads: list[CandidateRead] = list(
        sieve_fastq(
            fastq_path, anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
            params=sieve_params, stats=stats, n_threads=n_threads,
        )
    )
    if progress:
        progress("Stage 1: assigning reads to gene templates")
    matrices = build_gene_matrices(
        reads, pool_fasta=pool_fasta, pool_meta=pool_meta,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
        max_read_mismatch_rate=max_read_mismatch_rate,
        n_best=stage1_n_best,
    )
    if progress:
        progress("Stage 1: measuring chr17 control depth")
    # Real base-level coverage against the shipped chr17 control region
    # (stages/chr17_control.py) -- one whole-`fastq_path` minimap2 pass,
    # independent of Stage 0's sieve (a read supporting the control has
    # nothing to do with any HLA anchor and would never pass it). Not
    # `reads` (Stage 0's own sieved output) -- same reasoning c4_chr17_cn.py
    # already established for C4's own control-depth measurement.
    if control_depth is None:
        control_depth, control_n_reads = compute_chr17_control_depth(
            fastq_path, chr17_control_fasta=chr17_control_fasta,
            n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
        )
    if progress:
        progress("Stage 1: estimating copy number")
    # reads (Stage 0's sieved output), not matrices -- stages/cn.py's own
    # 2026-09-01 depth-source revision (see that module's docstring: real
    # samtools-depth over each gene's own anchors, normalised against
    # control_depth -- real chr17-control samtools depth -- rather than
    # the old per-class calibrator-panel median).
    gene_cn = estimate_cn(reads, control_depth, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta_path)
    manifest = load_gene_manifest(genes_yaml)

    if progress:
        progress("Stage 1 (C4): assigning reads to C4A/C4B templates")
    # DESIGN.md S2f: C4A/C4B have no star-allele pool, so this is a
    # completely separate build_gene_matrices() call against
    # resources/templates/assembly_bundle/c4.fa.gz (stages/c4_cn.py's own
    # docstring) -- same Stage 0-sieved `reads`, different reference set.
    # Reported as one combined "C4" entry in gene_cn (never routed through
    # `manifest`, which stages/cn.py's load_gene_manifest() already excludes
    # structural genes from -- so the Stage 2-5 typing loop below is
    # unaffected and never attempts to type it).
    c4_matrices = build_gene_matrices(
        reads, pool_fasta=c4_pool_fasta, pool_meta=c4_pool_meta,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
        max_read_mismatch_rate=max_read_mismatch_rate,
        n_best=stage1_n_best,
    )
    reads_by_id = {r.read_id: r.sequence for r in reads}
    gene_cn[C4_REPORTED_GENE] = estimate_c4_cn(
        c4_matrices, matrices, reads_by_id,
        genes_yaml=genes_yaml, c4_pool_meta=c4_pool_meta, classical_pool_meta=pool_meta,
    )

    if progress:
        progress("Stage 1 (C4): A-vs-B / long-vs-short form (Phase 2)")
    # DESIGN.md S2f Phase 2 (stages/c4_form.py) -- reuses c4_matrices
    # unchanged, no new alignment work. Never gates the Phase 1 total-CN
    # readout above or the Stage 2-5 typing loop below.
    c4_ab_call, c4_form_call = estimate_c4_form(
        c4_matrices, reads_by_id,
        meta_path=c4_pool_meta, pool_fasta=c4_pool_fasta,
    )
    # User-requested, 2026-08-27: CN_A/CN_B, a proportional split of the
    # Phase 1 total by the Phase 2 vote ratio -- see
    # estimate_c4_per_gene_cn()'s own docstring for why this isn't an
    # independent per-gene depth-ratio estimate.
    c4_per_gene_cn = estimate_c4_per_gene_cn(gene_cn[C4_REPORTED_GENE], c4_ab_call)

    if progress:
        progress("Stage 1 (C4): chr17-control per-form CN")
    # 2026-09-01, user-directed: the chr17-control mechanism
    # (stages/c4_chr17_cn.py), validated in last session's real HG00253
    # investigation as measurably closer to truth than the calibrator-
    # panel total above, is now production for C4 -- see write_calls_tsv's
    # own docstring for how its total supersedes gene_cn["C4"] in
    # calls.tsv while the older mechanism stays computed (still visible in
    # qc.json's own "gene_cn" block) as a live cross-check, not silently
    # dropped. Reuses this call's own control_depth/control_n_reads
    # (already computed above for the classical-gene normaliser) rather
    # than repeating a second, redundant whole-FASTQ chr17 alignment pass.
    # `compute_c4_chr17_control=False` (default True) skips this real
    # alignment pass entirely -- dependency injection for the test suite,
    # same reasoning as `control_depth`'s own override (docstring above):
    # most callers of this function are testing unrelated pipeline
    # behaviour and would otherwise pay for, and need a synthetic fixture
    # for, an alignment pass against the real production
    # c4_flanked_fasta/chr17_control_fasta resources.
    c4_chr17_control = estimate_c4_cn_via_chr17_control(
        fastq_path, chr17_control_fasta=chr17_control_fasta, c4_flanked_fasta=c4_flanked_fasta,
        control_depth=control_depth, control_n_reads=control_n_reads,
        n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
    ) if compute_c4_chr17_control else None

    if progress:
        progress(f"Stages 2-5: typing {len(manifest)} gene(s)")

    haplotype_calls: list[HaplotypeCallResult] = []
    consensus_records: list[HaplotypeConsensus] = []
    novelty_by_key: dict[tuple[str, int], NoveltyCall] = {}
    failures: list[GeneFailure] = []
    # Stage 2-3 results collected across every gene, Stage 4-5 typing run
    # once in a single batched call afterward (module docstring, lever 3).
    pending_haplotypes: list[tuple[str, int, int, HaplotypeConsensus, tuple[str, ...]]] = []

    manifest_genes = sorted(manifest)
    # First pass: every gene's cn_int under the ordinary per-gene rules
    # (low_cn_floor / diploid_typical / plain round-and-clamp), with no
    # cross-gene knowledge yet -- genes with no usable CN estimate are left
    # out here and handled as failures in the second pass below, same as
    # before this pre-pass was introduced.
    raw_cn_int: dict[str, int] = {}
    for gene in manifest_genes:
        entry = manifest[gene]
        this_gene_cn = gene_cn.get(gene)
        if this_gene_cn is None or this_gene_cn.cn_estimate is None:
            continue
        raw_cn_int[gene] = _raw_cn_int_for_gene(entry, this_gene_cn.cn_estimate, cn_round, max_cn)
    # DRB3/DRB4/DRB5 mutual exclusivity (module docstring on
    # _resolve_drb345_mutual_exclusivity) needs to see all three genes'
    # raw cn_int together before any of them commits to a Stage 2 search
    # size, which is why this can't be folded into the per-gene branch
    # above.
    cn_int_by_gene = _resolve_drb345_mutual_exclusivity(raw_cn_int, gene_cn)

    for gene in manifest_genes:
        if progress:
            progress(f"Stage 2-5: {gene}")

        entry = manifest[gene]
        this_gene_cn = gene_cn.get(gene)
        if this_gene_cn is None or this_gene_cn.cn_estimate is None:
            failures.append(GeneFailure(gene, "no CN estimate (no chr17-control signal this run)"))
            continue

        cn_int = cn_int_by_gene[gene]
        if cn_int < 1:
            failures.append(GeneFailure(
                gene,
                f"CN estimate {this_gene_cn.cn_estimate:.2f} rounds to {cn_int} -- "
                f"reported absent, not independently corroborated (invariant 4)",
            ))
            continue

        matrix = matrices.get(gene)
        if matrix is None:
            failures.append(GeneFailure(
                gene,
                f"CN estimate {this_gene_cn.cn_estimate:.2f} but zero reads "
                f"recruited to this gene's own matrix -- possible "
                f"template-recruitment failure, not a confirmed deletion "
                f"(DESIGN.md S2c HLA-Y warning)",
            ))
            continue

        if progress:
            progress(f"Stage 2-3: {gene} template selection + consensus (cn={cn_int})")
        try:
            call = select_template_set_ont_r10(
                matrix, reads, cn=cn_int, pool_fasta=pool_fasta,
                config=ont_config, n_threads=n_threads, minimap2_bin=minimap2_bin,
            )
            if call.cn > 1 and call.lrt_statistic < min_heterozygous_lrt:
                # Weak evidence for `cn` genuinely distinct alleles -- collapse
                # to the single best-supported one (module docstring; the
                # orchestration-layer decision stages/consensus.py's own
                # docstring names but leaves unmade).
                call = _collapse_to_single_allele(call)
            haps = build_gene_consensus(
                matrix, reads, call, pool_fasta=pool_fasta, config=ont_config,
                n_threads=n_threads, minimap2_bin=minimap2_bin,
            )
            if call.cn > 1:
                counts = [h.n_reads_assigned for h in haps]
                if (
                    min(counts) < min_reads_per_haplotype
                    or min(counts) / max(counts) < min_haplotype_read_ratio
                ):
                    # The LRT check above didn't catch it, but the real hard
                    # read partition shows a "distinct" allele with
                    # essentially no reads behind it (absolute floor, real
                    # HLA-B finding) or a small fraction of the major
                    # haplotype's depth (ratio floor, module docstring) --
                    # collapse to the dominant, single-best-supported allele.
                    call = _collapse_to_single_allele(call)
                    haps = build_gene_consensus(
                        matrix, reads, call, pool_fasta=pool_fasta, config=ont_config,
                        n_threads=n_threads, minimap2_bin=minimap2_bin,
                    )
        except ValueError as e:
            failures.append(GeneFailure(gene, f"template selection/consensus failed: {e}"))
            continue

        if len(haps) == 1 and call.cn > 1:
            # Collapsed to one distinct allele (above) but still `cn`
            # real copies -- report one row per real copy, not one row
            # that reads identically to a genuine cn=1 call and can be
            # misread as a deletion (HLA-B/HLA-DMA/HLA-S/MICA, real
            # HG00097 finding, all confirmed genuinely homozygous against
            # truth 2026-08-07). `hap` is the same object each time, not
            # re-derived -- deterministic by construction (invariant 9).
            haps = haps * call.cn

        for h_idx, hap in enumerate(haps):
            pending_haplotypes.append((gene, h_idx, cn_int, hap, call.candidates))

    # 2026-08-28 performance fix (lever 3, development notes session log): Stage 4-5
    # typing used to run inline, once per haplotype, inside the loop above
    # (up to 68 individually-full-pool-reloaded minimap2 calls on a real
    # sample) -- now a single hla_typing.type_haplotypes_batch() call scores
    # every gene's every haplotype at once. classify() (Stage 5) never
    # shells out to minimap2 at all (reuses type_haplotype's own captured
    # span), so it stays a per-haplotype Python call here, unaffected.
    #
    # `extra_candidates_by_key` (2026-09-02, real HG00097 TAP2 finding):
    # Stage 2's own narrowed candidate set for this gene (call.candidates)
    # is passed through so Stage 4 can fall back to a dense realignment
    # against it when a fresh whole-pool alignment of the consensus itself
    # silently drops a genuinely shorter-but-correct candidate (see
    # hla_typing.py's own docstring on this). Cheap in the common case --
    # only triggers the extra realignment when something is actually
    # missing from the naive search.
    if progress:
        progress(f"Stage 4-5: typing {len(pending_haplotypes)} haplotype call(s)")
    typing_calls, typing_failures = type_haplotypes_batch(
        [(f"{gene}#{h_idx}", hap) for gene, h_idx, _cn_int, hap, _cands in pending_haplotypes],
        pool_fasta=pool_fasta, pool_meta=pool_meta,
        config=ont_config, params=typing_params,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
        extra_candidates_by_key={
            f"{gene}#{h_idx}": cands for gene, h_idx, _cn_int, _hap, cands in pending_haplotypes
        },
        reads=reads,
    )
    for gene, h_idx, cn_int, hap, _cands in pending_haplotypes:
        key = f"{gene}#{h_idx}"
        if key in typing_failures:
            failures.append(GeneFailure(f"{gene}#{h_idx}", f"typing/novelty failed: {typing_failures[key]}"))
            continue
        typing_call = typing_calls[key]
        try:
            novelty_call = classify(
                typing_call, hap.sequence, pool_fasta=pool_fasta, pool_meta=pool_meta,
                models_path=models_path, cds_catalogue_path=cds_catalogue_path,
            )
        except ValueError as e:
            failures.append(GeneFailure(f"{gene}#{h_idx}", f"typing/novelty failed: {e}"))
            continue
        haplotype_calls.append(
            _build_haplotype_call_result(gene, h_idx, cn_int, hap, typing_call, novelty_call)
        )
        consensus_records.append(hap)
        novelty_by_key[(gene, h_idx)] = novelty_call

    sample_result = SampleResult(
        sample_id=sample_id,
        sieve_stats=stats,
        gene_cn=gene_cn,
        haplotype_calls=tuple(haplotype_calls),
        consensus_records=tuple(consensus_records),
        novelty_calls=novelty_by_key,
        failures=tuple(failures),
        c4_ab_call=c4_ab_call,
        c4_form_call=c4_form_call,
        c4_per_gene_cn=c4_per_gene_cn,
        c4_chr17_control=c4_chr17_control,
    )
    return sample_result, reads, matrices, c4_matrices


def run_sample(
    sample_id: str,
    fastq_path: Path,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    models_path: Path = DEFAULT_POOL_MODELS,
    cds_catalogue_path: Path = DEFAULT_CDS_CATALOGUE,
    c4_pool_fasta: Path = DEFAULT_C4_POOL_FASTA,
    c4_pool_meta: Path = DEFAULT_C4_POOL_META,
    genes_yaml: Path = DEFAULT_GENES_YAML,
    anchors_path: Path = DEFAULT_ANCHORS,
    anchors_meta_path: Path = DEFAULT_ANCHORS_META,
    chr17_control_fasta: Path = DEFAULT_CHR17_CONTROL_FASTA,
    c4_flanked_fasta: Path = DEFAULT_C4_FLANKED_FASTA,
    compute_c4_chr17_control: bool = True,
    control_depth: float | None = None,
    control_n_reads: int = 0,
    sieve_params: SieveParams | None = None,
    ont_config: OntR10Config | None = None,
    typing_params: TypingParams | None = None,
    cn_round: Callable[[float], int] = _round_cn_estimate,
    max_cn: int = 2,
    min_heterozygous_lrt: float = 10.0,
    min_reads_per_haplotype: int = 3,
    min_haplotype_read_ratio: float = 0.1,
    max_read_mismatch_rate: float = 0.06,
    stage1_n_best: int = 1,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
    progress: Callable[[str], None] | None = None,
) -> SampleResult:
    """Thin wrapper over `_run_sample_core()` (see that function's own
    docstring for the full behaviour) -- discards the sieved-reads/matrices
    it exposes for `run_sample_phased()`'s benefit. Signature and behaviour
    unchanged from before the 2026-08-28 refactor."""
    result, _reads, _matrices, _c4_matrices = _run_sample_core(
        sample_id, fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue_path, c4_pool_fasta=c4_pool_fasta,
        c4_pool_meta=c4_pool_meta, genes_yaml=genes_yaml,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        chr17_control_fasta=chr17_control_fasta, c4_flanked_fasta=c4_flanked_fasta, control_depth=control_depth,
        control_n_reads=control_n_reads,
        compute_c4_chr17_control=compute_c4_chr17_control,
        sieve_params=sieve_params, ont_config=ont_config, typing_params=typing_params,
        cn_round=cn_round, max_cn=max_cn, min_heterozygous_lrt=min_heterozygous_lrt,
        min_reads_per_haplotype=min_reads_per_haplotype,
        min_haplotype_read_ratio=min_haplotype_read_ratio,
        max_read_mismatch_rate=max_read_mismatch_rate, stage1_n_best=stage1_n_best,
        n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin, progress=progress,
    )
    return result


def _decide_phase_routing(
    gene: str,
    gene_reads: Sequence[CandidateRead],
    phase_of: Callable[[CandidateRead], ReadPhase],
    min_phased_tagged_fraction: float,
    *,
    no_reads_reason: str = "no reads recruited to this gene",
) -> GenePhaseRouting:
    """DESIGN.md S10.3: the phased-routing decision (single PS phase block,
    HP-tagged reads present, tagged fraction clears
    `min_phased_tagged_fraction`) factored out so it is made identically --
    and is unit-testable on its own -- for both the per-manifest-gene loop
    and the C4A/C4B pseudo-gene check in `run_sample_phased()`, which need
    the same rule applied to two different read pools. `haplotypes_called`
    is always returned `()`: it is only known once Stage 2-5 (or, for C4,
    the form classifiers) actually run, filled in by the caller afterward
    via `dataclasses.replace()`.
    """
    if not gene_reads:
        return GenePhaseRouting(
            gene=gene, routing="unphased_fallback", reason=no_reads_reason,
            ps_values_seen=(), n_reads_hp1=0, n_reads_hp2=0,
            n_reads_untagged=0, haplotypes_called=(),
        )

    n_hp1 = n_hp2 = n_untagged = 0
    ps_values: set[int] = set()
    for r in gene_reads:
        phase = phase_of(r)
        if phase.hp == 1:
            n_hp1 += 1
        elif phase.hp == 2:
            n_hp2 += 1
        else:
            n_untagged += 1
        if phase.ps is not None:
            ps_values.add(phase.ps)

    if n_hp1 == 0 and n_hp2 == 0:
        return GenePhaseRouting(
            gene=gene, routing="unphased_fallback",
            reason="no HP-tagged reads recruited to this gene "
                   "(phasing dropout, or input has no real phase tags)",
            ps_values_seen=tuple(sorted(ps_values)),
            n_reads_hp1=n_hp1, n_reads_hp2=n_hp2, n_reads_untagged=n_untagged,
            haplotypes_called=(),
        )
    if len(ps_values) > 1:
        return GenePhaseRouting(
            gene=gene, routing="unphased_fallback",
            reason=f"reads split across {len(ps_values)} distinct PS phase "
                   f"blocks -- HP labels are not comparable across them (S10.3)",
            ps_values_seen=tuple(sorted(ps_values)),
            n_reads_hp1=n_hp1, n_reads_hp2=n_hp2, n_reads_untagged=n_untagged,
            haplotypes_called=(),
        )
    tagged_fraction = (n_hp1 + n_hp2) / len(gene_reads)
    if tagged_fraction < min_phased_tagged_fraction:
        return GenePhaseRouting(
            gene=gene, routing="unphased_fallback",
            reason=f"only {tagged_fraction:.0%} of this gene's recruited reads "
                   f"carry a usable HP tag (< {min_phased_tagged_fraction:.0%} "
                   f"floor) -- insufficient phased coverage, not a PS conflict "
                   f"(real HG00097 HLA-C finding, module docstring)",
            ps_values_seen=tuple(sorted(ps_values)),
            n_reads_hp1=n_hp1, n_reads_hp2=n_hp2, n_reads_untagged=n_untagged,
            haplotypes_called=(),
        )
    return GenePhaseRouting(
        gene=gene, routing="phased",
        reason="single PS phase block, HP-tagged reads present",
        ps_values_seen=tuple(sorted(ps_values)),
        n_reads_hp1=n_hp1, n_reads_hp2=n_hp2, n_reads_untagged=n_untagged,
        haplotypes_called=(),
    )


def _c4_calibrator_reads_for_haplotype(
    hp_val: int,
    c4_ps_values: tuple[int, ...],
    calibrator_genes: list[str],
    routing: dict[str, "GenePhaseRouting"],
    reads_by_gene: dict[str, list[CandidateRead]],
    phase_of: Callable[[CandidateRead], ReadPhase],
) -> list[CandidateRead]:
    """DESIGN.md S10.5/S10.6: which calibrator genes' HP-split reads are
    safe to use for *this* haplotype's per-haplotype C4 CN panel. S10.3's
    stitching warning is the reason this isn't a trivial lookup: `HP=1` for
    one gene is only guaranteed to mean the same physical chromosome as
    `HP=1` for another gene if both sit in the same `PS` phase block. A
    calibrator gene only contributes here if it independently routed
    "phased" (its own single-PS-block/tagged-fraction check already
    passed) *and* its own singleton `ps_values_seen` is identical to C4's
    own -- real, checked evidence, not an assumption. `c4_ps_values` of
    length != 1 (no real PS block, or C4's own reads span none) means
    nothing can ever match, by construction -- the caller gets an empty
    list for every haplotype, which is the honest answer, not a special
    case handled here."""
    if len(c4_ps_values) != 1:
        return []
    reads: list[CandidateRead] = []
    seen_ids: set[str] = set()
    for g in calibrator_genes:
        g_routing = routing.get(g)
        if g_routing is None or g_routing.routing != "phased":
            continue
        if g_routing.ps_values_seen != c4_ps_values:
            continue
        for r in reads_by_gene.get(g, []):
            if phase_of(r).hp == hp_val and r.read_id not in seen_ids:
                seen_ids.add(r.read_id)
                reads.append(r)
    return reads


def run_sample_phased(
    sample_id: str,
    fastq_path: Path,
    mini_bam_path: Path,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    models_path: Path = DEFAULT_POOL_MODELS,
    cds_catalogue_path: Path = DEFAULT_CDS_CATALOGUE,
    c4_pool_fasta: Path = DEFAULT_C4_POOL_FASTA,
    c4_pool_meta: Path = DEFAULT_C4_POOL_META,
    genes_yaml: Path = DEFAULT_GENES_YAML,
    anchors_path: Path = DEFAULT_ANCHORS,
    anchors_meta_path: Path = DEFAULT_ANCHORS_META,
    chr17_control_fasta: Path = DEFAULT_CHR17_CONTROL_FASTA,
    c4_flanked_fasta: Path = DEFAULT_C4_FLANKED_FASTA,
    compute_c4_chr17_control: bool = True,
    control_depth: float | None = None,
    control_n_reads: int = 0,
    sieve_params: SieveParams | None = None,
    ont_config: OntR10Config | None = None,
    typing_params: TypingParams | None = None,
    cn_round: Callable[[float], int] = _round_cn_estimate,
    max_cn: int = 2,
    min_heterozygous_lrt: float = 10.0,
    min_reads_per_haplotype: int = 3,
    min_haplotype_read_ratio: float = 0.1,
    max_read_mismatch_rate: float = 0.06,
    stage1_n_best: int = 1,
    min_phased_tagged_fraction: float = 0.5,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
    progress: Callable[[str], None] | None = None,
) -> SampleResult:
    """DESIGN.md S10: phased-input mode. Stage 0 (extraction -> mini-BAM ->
    FASTQ) is unchanged and already ran by the time this is called --
    cli.py passes both the FASTQ and the mini-BAM it was converted from
    (the mini-BAM still carries the real `HP`/`PS` tags `mini_bam_to_fastq`
    drops); this function does not extract anything itself.

    **Reuses `run_sample()`'s full per-gene CN/collapse logic (via
    `_run_sample_core()`, its shared body), rather than reimplementing it.**
    That logic (LRT collapse, min_reads_per_haplotype,
    min_haplotype_read_ratio, cn_model-aware search sizing, DRB3/4/5 mutual
    exclusivity -- all tuned against real HG00097 data, see this module's
    own docstring) is exactly what a gene on the *unphased fallback* path
    (S10.3: no HP-tagged reads recruited to it, or its reads span more than
    one PS phase block) still needs. Rebuilding it a second time here would
    either duplicate it (drift risk) or subtly diverge from the validated
    version -- so every gene starts from that run's own output, and only
    genes that clear the S10.3 bar have their calls *replaced* with a fresh
    per-haplotype result. **2026-08-28 performance fix:** this used to call
    `run_sample()` itself and then separately re-sieve the FASTQ (measured
    89s of pure waste) to recover the read list `run_sample()` doesn't
    expose; it now calls `_run_sample_core()` directly, which returns the
    unphased `SampleResult` *and* the sieved reads and Stage 1 matrices in
    one pass -- see that function's own docstring.

    **Per-haplotype replacement (S10.4).** For a gene whose own recruited
    reads split cleanly by `HP` under a single shared `PS` value, each
    haplotype's read subset is run through Stage 1 (assign) -> Stage 2
    (genotype, `cn` forced to 1 -- a single-template search, not a
    combination search, since the pool has already been split by real
    phase) -> Stage 3 (consensus) -> Stage 4-5 (typing/novelty)
    independently, with `haplotype_index` set to the real `HP` value (1/2)
    instead of Stage 2's arbitrary search-order label. A haplotype with no
    reads at all for that gene (real hemizygosity, or a deletion on that
    specific chromosome) simply contributes no row -- recorded via
    `GenePhaseRouting.haplotypes_called`, not silently absent (S10.3/qc.json).

    **Every manifest gene gets a `GenePhaseRouting` entry, user-requested,**
    so which genes went through which pipeline is an explicit, inspectable
    QC field (stages/output.py's write_qc_json) rather than something only
    re-derivable from re-reading the code.

    **C4A/C4B phasing (S10.6), user-requested, added 2026-08-27 -- Phase 2
    only.** C4A/C4B have no per-haplotype template call to speak of (S2f:
    no star-allele nomenclature), so there is nothing analogous to S10.4's
    single-template replacement to do for them. What *is* built: the same
    routing decision (single PS block, HP-tagged reads present, tagged
    fraction >= `min_phased_tagged_fraction`) runs a second time for the
    combined C4A/C4B read pool (keyed `C4_REPORTED_GENE` in `routing`,
    right alongside the manifest genes), and for each haplotype that
    clears it, `stages/c4_form.py`'s existing `classify_a_vs_b`/
    `classify_long_vs_short` (via `estimate_c4_form`) run again on just
    that haplotype's own C4-informative reads -- unchanged classifier
    logic, a smaller/real-phase-restricted read pool, exactly DESIGN.md
    S10.6's own description. Results land in `c4_phased_ab_calls`/
    `c4_phased_form_calls` (keyed by real HP value), surfaced in
    `qc.json`'s `c4_form.per_haplotype` and as extra per-haplotype
    `C4A`/`C4B` rows in `calls.tsv` (`stages/output.py`) when present.
    **Phase 1 (total CN), added 2026-08-28.** Splitting `gene_cn["C4"]`'s
    depth-ratio formula per haplotype needs a per-haplotype calibrator
    panel -- the real blocker DESIGN.md S10.5/S10.6 both flag: a class I
    calibrator gene's own `HP=1` is not guaranteed to be the same physical
    chromosome as C4's own `HP=1` unless the two share a `PS` phase block
    (S10.3's own stitching warning). `_c4_calibrator_matrices_for_haplotype()`
    resolves this directly rather than assuming it: a calibrator gene
    (`c4_cn.calibrator_genes_for_c4()`, the identical class I panel
    `estimate_c4_cn()` already uses sample-wide) only contributes its
    HP-split reads to a given haplotype's panel if that gene *independently*
    routed "phased" (its own single-PS-block/tagged-fraction check already
    passed) **and** its own `ps_values_seen` is identical to C4's own --
    i.e. real, checked evidence the two loci are phased on the same
    chromosome-scale block, not an assumption. A calibrator gene that
    doesn't share C4's PS block is simply excluded from that sample's
    per-haplotype panel (graceful degradation, the same "opportunistic when
    real data supports it" contract `stages/c4_cn.py`'s own length-inflation
    correction already uses) -- `estimate_c4_cn()` itself is unchanged,
    called a second time per haplotype on these PS-matched, HP-restricted
    matrices; `estimate_c4_per_gene_cn()` (also unchanged) then splits that
    haplotype's own total by its own `C4AbCall` vote ratio for `cn_a`/
    `cn_b`. Results land in `c4_phased_cn`/`c4_phased_per_gene_cn` (keyed by
    real HP value), surfaced in `qc.json`'s `c4_form.per_haplotype` and as
    extra per-haplotype `"C4"` rows (and populated `cn` on the existing
    per-haplotype `"C4A"`/`"C4B"` rows) in `calls.tsv`. When no calibrator
    gene shares C4's PS block for a sample (small-n, or genuinely
    non-chromosome-scale phasing), `GeneCN.cn_estimate` comes back `None`
    for that haplotype -- present in the output, honestly uncertain, never
    a guess and never silently omitted (invariant 4's "loud, not silent"
    convention). `gene_cn["C4"]` remains the separate sample-wide aggregate
    regardless. **Documented limit, unchanged from DESIGN.md S10.6:** a
    haplotype's own per-haplotype tally
    can still show both A and B (or both long and short) votes if that
    haplotype itself carries an internal RCCX duplication -- HP phasing
    resolves which chromosome a read came from, not which of several
    tandem repeats on that chromosome.

    **Still not yet done, flagged not hidden:** `gene_cn` (Stage 1's
    depth-ratio estimate) is left exactly as `run_sample()` computed it for
    every gene, informational only for a "phased"-routed gene -- the real
    per-haplotype presence/absence is `haplotype_calls`/
    `GenePhaseRouting.haplotypes_called`, not a re-derived CN number.

    **`min_phased_tagged_fraction` (default 0.5), added 2026-08-27 after
    real HG00097 validation surfaced a real gap.** The routing check
    (S10.3) originally only asked *whether* HP-tagged reads existed and
    agreed on one PS value -- not whether *enough* of a gene's reads were
    tagged at all. Real data broke this: HLA-C had only 11 of 72 recruited
    reads carrying an HP tag (61 untagged), giving a 3-read/8-read
    haplotype split that mistyped one haplotype as novel where the
    unphased path (using all 72 reads) called it correctly. A gene now
    also needs `(n_hp1 + n_hp2) / len(gene_reads) >=
    min_phased_tagged_fraction` to route "phased" -- otherwise it falls
    back with a `reason` naming the shortfall, same as a PS conflict.
    Provisional (invariant 11): chosen as a reasonable majority-coverage
    bar, not fitted -- HLA-C's real 0.153 sits nowhere near a defensible
    threshold either way, so 0.5 was not chosen to thread a narrow needle.
    **Separately, a per-haplotype floor:** even within a gene that clears
    the fraction bar, a haplotype with a nonzero but very small tagged-read
    count (below `min_reads_per_haplotype`, reusing the same threshold the
    unphased path already applies post-hoc) does not get a Stage 2-5
    attempt at all -- recorded as a failure, not a low-confidence call built
    from too little evidence to mean anything.
    """
    if ont_config is None:
        ont_config = load_ont_r10_config()
    if typing_params is None:
        typing_params = TypingParams.load()

    # 2026-08-28 performance fix (levers 1-2, development notes session log): calls
    # _run_sample_core() once instead of run_sample() + a second, separate
    # sieve_fastq() pass -- measured 89s of pure waste on real HG00097 data
    # (the sieve is deterministic, invariant 9, so the second pass always
    # produced an identical `reads` list to the one run_sample() itself
    # already computed and discarded). `matrices`/`c4_matrices` (Stage 1's
    # own per-gene assignment matrices) are reused below via
    # `assign.subset_matrix_by_read_ids()`/`classical_matrices_for_reads()`
    # (lever 2) instead of re-running minimap2 per haplotype.
    unphased, reads, matrices, c4_matrices = _run_sample_core(
        sample_id, fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue_path, c4_pool_fasta=c4_pool_fasta,
        c4_pool_meta=c4_pool_meta, genes_yaml=genes_yaml,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        chr17_control_fasta=chr17_control_fasta, c4_flanked_fasta=c4_flanked_fasta, control_depth=control_depth,
        control_n_reads=control_n_reads,
        compute_c4_chr17_control=compute_c4_chr17_control,
        sieve_params=sieve_params, ont_config=ont_config, typing_params=typing_params,
        cn_round=cn_round, max_cn=max_cn, min_heterozygous_lrt=min_heterozygous_lrt,
        min_reads_per_haplotype=min_reads_per_haplotype,
        min_haplotype_read_ratio=min_haplotype_read_ratio,
        max_read_mismatch_rate=max_read_mismatch_rate, stage1_n_best=stage1_n_best,
        n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin, progress=progress,
    )

    phase_tags = load_read_phase_tags(mini_bam_path)
    manifest = load_gene_manifest(genes_yaml)

    reads_by_gene: dict[str, list[CandidateRead]] = {}
    for r in reads:
        for hit in r.hits:
            reads_by_gene.setdefault(hit.gene, []).append(r)

    def _phase_of(read: CandidateRead) -> ReadPhase:
        return phase_tags.get(read.read_id, ReadPhase(hp=None, ps=None))

    routing: dict[str, GenePhaseRouting] = {}
    phased_genes: set[str] = set()
    for gene in sorted(manifest):
        gene_reads = reads_by_gene.get(gene, [])
        routing[gene] = _decide_phase_routing(
            gene, gene_reads, _phase_of, min_phased_tagged_fraction,
        )
        if routing[gene].routing == "phased":
            phased_genes.add(gene)

    if progress:
        progress(
            f"Phased: {len(phased_genes)} gene(s) cleanly phased, "
            f"{len(manifest) - len(phased_genes)} on the unphased fallback path"
        )

    phased_haplotype_calls: list[HaplotypeCallResult] = []
    phased_consensus_records: list[HaplotypeConsensus] = []
    phased_novelty_by_key: dict[tuple[str, int], NoveltyCall] = {}
    phased_failures: list[GeneFailure] = []
    # Stage 2-3 results collected across every phased gene, Stage 4-5
    # typing run once in a single batched call afterward (2026-08-28
    # performance fix, lever 3 -- same pattern as _run_sample_core()'s own
    # per-gene loop above). `routing[gene].haplotypes_called` is only
    # finalised after that batch, since consensus succeeding here does not
    # yet mean a real, fully-typed call exists.
    pending_phased_haplotypes: list[tuple[str, int, HaplotypeConsensus]] = []

    for gene in sorted(phased_genes):
        if progress:
            progress(f"Phased Stage 2-3: {gene}")
        gene_reads = reads_by_gene[gene]
        hp_reads_by_val: dict[int, Sequence[CandidateRead]] = {}
        hp_matrix_by_val: dict[int, GeneAssignmentMatrix] = {}
        hp_calls: dict[int, TemplateCall] = {}
        for hp_val in (1, 2):
            hp_reads = [r for r in gene_reads if _phase_of(r).hp == hp_val]
            if not hp_reads:
                continue
            if len(hp_reads) < min_reads_per_haplotype:
                phased_failures.append(GeneFailure(
                    f"{gene}#hp{hp_val}",
                    f"only {len(hp_reads)} HP-tagged read(s) for this haplotype "
                    f"(< min_reads_per_haplotype={min_reads_per_haplotype}) -- too "
                    f"little evidence for a reliable per-haplotype call",
                ))
                continue
            # lever 2 (2026-08-28): row-subset Stage 1's own already-computed
            # matrix for this gene instead of calling build_gene_matrices()
            # again -- see assign.subset_matrix_by_read_ids()'s own
            # docstring for why this is exactly equivalent, not an
            # approximation.
            full_matrix = matrices.get(gene)
            matrix = (
                subset_matrix_by_read_ids(full_matrix, {r.read_id for r in hp_reads})
                if full_matrix is not None else None
            )
            if matrix is None:
                phased_failures.append(GeneFailure(
                    f"{gene}#hp{hp_val}",
                    "HP-tagged reads present but none survived the "
                    "per-gene read-quality gate (assign.build_gene_matrices)",
                ))
                continue
            try:
                call = select_template_set_ont_r10(
                    matrix, hp_reads, cn=1, pool_fasta=pool_fasta,
                    config=ont_config, n_threads=n_threads, minimap2_bin=minimap2_bin,
                )
            except ValueError as e:
                phased_failures.append(GeneFailure(
                    f"{gene}#hp{hp_val}", f"phased template selection failed: {e}",
                ))
                continue
            hp_reads_by_val[hp_val] = hp_reads
            hp_matrix_by_val[hp_val] = matrix
            hp_calls[hp_val] = call

        if not hp_calls:
            phased_failures.append(GeneFailure(
                gene,
                "routed to the phased path but neither haplotype produced "
                "a real call (see per-haplotype failures above)",
            ))
            routing[gene] = replace(routing[gene], haplotypes_called=())
            continue

        try:
            if len(hp_calls) == 2:
                # lever 4 (2026-08-28): one combined realign+consensus call
                # covering both haplotypes' own winning alleles, instead of
                # two separate build_gene_consensus() calls -- see
                # build_phased_pair_consensus()'s own docstring for why this
                # preserves the real, known HP partition rather than
                # re-deriving it via EM responsibility.
                haps_by_hp = build_phased_pair_consensus(
                    gene, hp_reads_by_val, hp_matrix_by_val,
                    {hp: call.alleles[0] for hp, call in hp_calls.items()},
                    pool_fasta=pool_fasta, n_threads=n_threads, minimap2_bin=minimap2_bin,
                )
            else:
                (only_hp, only_call), = hp_calls.items()
                single = build_gene_consensus(
                    hp_matrix_by_val[only_hp], hp_reads_by_val[only_hp], only_call,
                    pool_fasta=pool_fasta, config=ont_config,
                    n_threads=n_threads, minimap2_bin=minimap2_bin,
                )
                haps_by_hp = {only_hp: single[0]}
        except ValueError as e:
            for hp_val in hp_calls:
                phased_failures.append(GeneFailure(
                    f"{gene}#hp{hp_val}", f"phased consensus build failed: {e}",
                ))
            routing[gene] = replace(routing[gene], haplotypes_called=())
            continue

        for hp_val, hap in haps_by_hp.items():
            pending_phased_haplotypes.append((gene, hp_val, hap))

    # 2026-08-28 performance fix (lever 3): batched Stage 4-5 typing across
    # every phased haplotype call collected above, mirroring
    # _run_sample_core()'s own per-gene loop -- see that call site's own
    # comment and hla_typing.type_haplotypes_batch()'s docstring.
    if progress:
        progress(f"Phased Stage 4-5: typing {len(pending_phased_haplotypes)} haplotype call(s)")
    phased_typing_calls, phased_typing_failures = type_haplotypes_batch(
        [(f"{gene}#hp{hp_val}", hap) for gene, hp_val, hap in pending_phased_haplotypes],
        pool_fasta=pool_fasta, pool_meta=pool_meta,
        config=ont_config, params=typing_params,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    haplotypes_called_by_gene: dict[str, list[int]] = {}
    for gene, hp_val, hap in pending_phased_haplotypes:
        key = f"{gene}#hp{hp_val}"
        if key in phased_typing_failures:
            phased_failures.append(GeneFailure(key, f"phased typing/novelty failed: {phased_typing_failures[key]}"))
            continue
        typing_call = phased_typing_calls[key]
        try:
            novelty_call = classify(
                typing_call, hap.sequence, pool_fasta=pool_fasta, pool_meta=pool_meta,
                models_path=models_path, cds_catalogue_path=cds_catalogue_path,
            )
        except ValueError as e:
            phased_failures.append(GeneFailure(key, f"phased typing/novelty failed: {e}"))
            continue
        phased_haplotype_calls.append(
            _build_haplotype_call_result(gene, hp_val, 1, hap, typing_call, novelty_call)
        )
        phased_consensus_records.append(hap)
        phased_novelty_by_key[(gene, hp_val)] = novelty_call
        haplotypes_called_by_gene.setdefault(gene, []).append(hp_val)

    # Finalise routing for every gene that reached the typing attempt above
    # (genes that failed earlier already set their own routing/failure
    # directly, inside the loop) -- a gene with zero successfully-typed
    # haplotypes here still routed "phased" through Stage 2-3 but produced
    # no real call, the same "neither haplotype produced a real call"
    # outcome the pre-batching code detected inline per gene.
    for gene in sorted({g for g, _hp, _hap in pending_phased_haplotypes}):
        called = tuple(sorted(haplotypes_called_by_gene.get(gene, [])))
        routing[gene] = replace(routing[gene], haplotypes_called=called)
        if not called:
            phased_failures.append(GeneFailure(
                gene,
                "routed to the phased path but neither haplotype produced "
                "a real call (see per-haplotype failures above)",
            ))

    # DESIGN.md S10.6: per-haplotype C4A/C4B form discrimination (Phase 2
    # only). C4A/C4B are not in `manifest` (load_gene_manifest() excludes
    # structural genes), so this runs as one extra pseudo-gene routing
    # decision alongside the per-manifest-gene loop above, keyed
    # C4_REPORTED_GENE ("C4") -- same routing rules (single PS block,
    # tagged-fraction floor, per-haplotype read-count floor), reusing the
    # same `reads_by_gene`/`_phase_of` machinery.
    c4_reads_by_id_map: dict[str, CandidateRead] = {}
    for c4_gene in C4_GENES:
        for r in reads_by_gene.get(c4_gene, []):
            c4_reads_by_id_map[r.read_id] = r
    c4_reads = list(c4_reads_by_id_map.values())

    c4_phased_ab_calls: dict[int, C4AbCall] = {}
    c4_phased_form_calls: dict[int, C4FormLengthCall] = {}
    c4_phased_cn: dict[int, GeneCN] = {}
    c4_phased_per_gene_cn: dict[int, C4PerGeneCN] = {}

    c4_routing = _decide_phase_routing(
        C4_REPORTED_GENE, c4_reads, _phase_of, min_phased_tagged_fraction,
        no_reads_reason="no reads recruited to C4A/C4B",
    )
    routing[C4_REPORTED_GENE] = c4_routing

    if c4_routing.routing == "phased":
        c4_calibrator_genes = calibrator_genes_for_c4(genes_yaml)
        c4_haplotypes_called: list[int] = []
        for hp_val in (1, 2):
            hp_c4_reads = [r for r in c4_reads if _phase_of(r).hp == hp_val]
            if not hp_c4_reads:
                continue
            if len(hp_c4_reads) < min_reads_per_haplotype:
                phased_failures.append(GeneFailure(
                    f"{C4_REPORTED_GENE}#hp{hp_val}",
                    f"only {len(hp_c4_reads)} HP-tagged C4A/C4B read(s) for "
                    f"this haplotype (< min_reads_per_haplotype="
                    f"{min_reads_per_haplotype})",
                ))
                continue
            # 2026-08-28 performance fix (lever 2, extended to the C4 phased
            # block): reuse _run_sample_core()'s own sample-wide c4_matrices
            # (built once already) instead of a fresh build_gene_matrices()
            # call -- same equivalence argument as the manifest-gene loop
            # above, just via the multi-gene helper since C4_GENES has two
            # members.
            hp_c4_matrices = classical_matrices_for_reads(c4_matrices, hp_c4_reads, C4_GENES)
            hp_c4_reads_by_id = {r.read_id: r.sequence for r in hp_c4_reads}
            ab_call, form_call = estimate_c4_form(
                hp_c4_matrices, hp_c4_reads_by_id,
                meta_path=c4_pool_meta, pool_fasta=c4_pool_fasta,
            )
            c4_phased_ab_calls[hp_val] = ab_call
            c4_phased_form_calls[hp_val] = form_call

            # DESIGN.md S10.5/S10.6, added 2026-08-28: per-haplotype total
            # C4 CN (Phase 1) -- PS-matched calibrator panel, see
            # _c4_calibrator_reads_for_haplotype()'s own docstring.
            hp_calibrator_reads = _c4_calibrator_reads_for_haplotype(
                hp_val, c4_routing.ps_values_seen, c4_calibrator_genes,
                routing, reads_by_gene, _phase_of,
            )
            # Same reuse (lever 2): calibrator genes are always a subset of
            # `manifest`, so Stage 1's own `matrices` already has each one
            # fully aligned -- classical_matrices_for_reads() reconstructs
            # the per-gene subset build_gene_matrices() would have, without
            # a second minimap2 call.
            hp_classical_matrices = (
                classical_matrices_for_reads(matrices, hp_calibrator_reads, c4_calibrator_genes)
                if hp_calibrator_reads else {}
            )
            hp_reads_by_id = dict(hp_c4_reads_by_id)
            hp_reads_by_id.update({r.read_id: r.sequence for r in hp_calibrator_reads})
            hp_cn = estimate_c4_cn(
                hp_c4_matrices, hp_classical_matrices, hp_reads_by_id,
                genes_yaml=genes_yaml, c4_pool_meta=c4_pool_meta,
                classical_pool_meta=pool_meta,
            )
            c4_phased_cn[hp_val] = hp_cn
            c4_phased_per_gene_cn[hp_val] = estimate_c4_per_gene_cn(hp_cn, ab_call)

            c4_haplotypes_called.append(hp_val)

        if not c4_haplotypes_called:
            phased_failures.append(GeneFailure(
                C4_REPORTED_GENE,
                "routed to the phased path but neither haplotype produced "
                "a real C4A/C4B form call (see per-haplotype failures above)",
            ))
        routing[C4_REPORTED_GENE] = replace(c4_routing, haplotypes_called=tuple(c4_haplotypes_called))

    final_haplotype_calls = tuple(
        c for c in unphased.haplotype_calls if c.gene not in phased_genes
    ) + tuple(phased_haplotype_calls)
    final_consensus_records = tuple(
        hc for hc in unphased.consensus_records if hc.gene not in phased_genes
    ) + tuple(phased_consensus_records)
    final_novelty_by_key = {
        k: v for k, v in unphased.novelty_calls.items() if k[0] not in phased_genes
    }
    final_novelty_by_key.update(phased_novelty_by_key)
    final_failures = tuple(
        f for f in unphased.failures if f.gene.split("#", 1)[0] not in phased_genes
    ) + tuple(phased_failures)

    return replace(
        unphased,
        haplotype_calls=final_haplotype_calls,
        consensus_records=final_consensus_records,
        novelty_calls=final_novelty_by_key,
        failures=final_failures,
        phasing_routing=routing,
        c4_phased_ab_calls=c4_phased_ab_calls or None,
        c4_phased_form_calls=c4_phased_form_calls or None,
        c4_phased_cn=c4_phased_cn or None,
        c4_phased_per_gene_cn=c4_phased_per_gene_cn or None,
    )

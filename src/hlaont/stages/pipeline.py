"""Orchestration: Stage 0 -> assign -> cn -> genotype -> consensus ->
hla_typing -> novelty, over every manifest gene (invariant 5).

Every non-excluded, non-structural gene is attempted. A gene with a positive
CN but no recruited reads, no CN estimate, or CN rounding to 0 is recorded
in `SampleResult.failures`, never silently dropped.

CN sizing: `cn_round` (default `_round_cn_estimate`) then clamp to `max_cn`
(default 2); depth noise past 2 is far likelier than real duplication.
Raw `cn_estimate`/CI stay in `gene_cn`. Genes with `low_cn_floor`
(DRB3/4/5) below the floor are treated as absent.

Homozygous collapse: Stage 2 always returns `cn` distinct alleles, so a
call collapses to its single best allele (cn kept) when
- LRT < `min_heterozygous_lrt` (10.0), or, after consensus,
- any haplotype has < `min_reads_per_haplotype` (3) reads, or
- minor/major read ratio < `min_haplotype_read_ratio` (0.1).
A high LRT alone can come from diffuse sub-read signal with no read on the
second allele (HLA-B). All three thresholds are provisional (invariant 11).

Fault isolation: only `ValueError` (what stage modules raise deliberately)
is caught per gene/haplotype and recorded; anything else propagates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

from hlaont.input.fastq import (
    DEFAULT_ANCHORS,
    DEFAULT_ANCHORS_META,
    CandidateRead,
    SieveParams,
    SieveStats,
    sieve_fastq,
)
from hlaont.input.phasing import ReadPhase, load_read_phase_tags

from hlaont.stages.assign import (
    DEFAULT_POOL_FASTA,
    DEFAULT_POOL_META,
    GeneAssignmentMatrix,
    build_gene_matrices,
    classical_matrices_for_reads,
    gene_allele_pool,
    merge_matrices,
    realign_to_candidates,
    subset_matrix_by_read_ids,
)
from hlaont.stages.cn import (
    DEFAULT_GENES_YAML,
    GeneCN,
    GeneManifestEntry,
    estimate_cn,
    load_gene_manifest,
)
from hlaont.stages.chr17_control import (
    DEFAULT_CHR17_CONTROL_FASTA,
    compute_chr17_control_depth,
)
from hlaont.stages.c4_cn import (
    C4_GENES,
    C4_REPORTED_GENE,
    DEFAULT_C4_POOL_FASTA,
    DEFAULT_C4_POOL_META,
    calibrator_genes_for_c4,
    estimate_c4_cn,
)
from hlaont.stages.c4_chr17_cn import (
    DEFAULT_C4_FLANKED_FASTA,
    C4Chr17ControlResult,
    C4JointCall,
    call_c4_forms_jointly,
    estimate_c4_cn_via_chr17_control,
)
from hlaont.stages.c4_form import (
    C4AbCall,
    C4FormLengthCall,
    C4PerGeneCN,
    estimate_c4_form,
    estimate_c4_per_gene_cn,
)
from hlaont.stages.consensus import (
    HaplotypeConsensus,
    build_gene_consensus,
    build_phased_pair_consensus,
)
from hlaont.stages.genotype import (
    OntR10Config,
    TemplateCall,
    load_allele_sequences,
    load_ont_r10_config,
    select_template_set_ont_r10,
)
from hlaont.stages.hla_typing import TypingCall, TypingParams, type_haplotypes_batch
from hlaont.stages.novelty import (
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
    extension_allele: str | None
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
    # C4 Phase 2 read tallies (c4_form.py); never None in a real run.
    c4_ab_call: C4AbCall | None = None
    c4_form_call: C4FormLengthCall | None = None
    # CN_A/CN_B: total C4 CN split by the Phase 2 vote ratio.
    c4_per_gene_cn: C4PerGeneCN | None = None
    # One entry per manifest gene plus "C4"; None for unphased runs.
    phasing_routing: "dict[str, GenePhaseRouting] | None" = None
    # Per-haplotype C4 form tallies keyed by HP; None unless C4 routed phased.
    c4_phased_ab_calls: "dict[int, C4AbCall] | None" = None
    c4_phased_form_calls: "dict[int, C4FormLengthCall] | None" = None
    # Per-haplotype total C4 CN keyed by HP; cn_estimate is None when no
    # calibrator gene shares C4's PS block. None unless C4 routed phased.
    c4_phased_cn: "dict[int, GeneCN] | None" = None
    c4_phased_per_gene_cn: "dict[int, C4PerGeneCN] | None" = None
    # Per-form (C4AL/C4AS/C4BL/C4BS) chr17-control CN. Unphased only.
    c4_chr17_control: "C4Chr17ControlResult | None" = None
    # Integer per-form C4 call (c4_chr17_cn.call_c4_forms_jointly). Unphased
    # only; None without a count-based chr17-control total.
    c4_joint_call: "C4JointCall | None" = None


@dataclass(frozen=True)
class GenePhaseRouting:
    """Per-gene record of whether the gene was typed per HP haplotype
    ("phased") or via the unphased path ("unphased_fallback"), and why."""

    gene: str
    routing: str  # "phased" | "unphased_fallback"
    reason: str
    ps_values_seen: tuple[int, ...]
    n_reads_hp1: int
    n_reads_hp2: int
    n_reads_untagged: int
    # HP values (1/2) that produced a call; () unless routing == "phased".
    haplotypes_called: tuple[int, ...]


def _round_cn_estimate(cn_estimate: float) -> int:
    """Default `cn_round`: >1.25 -> 2, >0.25 -> 1, else 0. Biased toward
    presence (invariant 4); plain round() dropped HLA-U at 0.41.
    Provisional."""
    if cn_estimate > 1.25:
        return 2
    if cn_estimate > 0.25:
        return 1
    return 0


def _raw_cn_int_for_gene(
    entry: GeneManifestEntry, cn_estimate: float, cn_round: Callable[[float], int], max_cn: int,
) -> int:
    """Per-gene cn_int: low_cn_floor gate, then round and clamp to max_cn."""
    if entry.low_cn_floor is not None and cn_estimate < entry.low_cn_floor:
        # Below low_cn_floor (DRB3/4/5): likely paralog leakage; absent.
        return 0
    # No forced floor for diploid_typical genes: the current depth signal is
    # trustworthy enough that a low estimate may be a real deletion.
    return min(int(cn_round(cn_estimate)), max_cn)


_DRB_MUTUALLY_EXCLUSIVE_GENES = ("HLA-DRB3", "HLA-DRB4", "HLA-DRB5")
_DRB_AMBIGUOUS_CN_LOW = 1.2
_DRB_AMBIGUOUS_CN_HIGH = 2.0


def _resolve_drb345_mutual_exclusivity(
    cn_int_by_gene: dict[str, int], gene_cn: dict[str, GeneCN],
) -> dict[str, int]:
    """DRB3/4/5 share one haplotype slot, so their cn_ints sum to <= 2.
    If one gene rounded up to 2 from the ambiguous (1.2, 2.0) band while
    another sits at 1, pull the ambiguous one down to 1."""
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


_MIN_MISSING_READS_TO_RESCUE = 10
_MIN_MISSING_FRACTION_TO_RESCUE = 0.15


def rescue_stolen_reads(
    gene: str,
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    n_threads: int = 1,
    minimap2_bin: str = "minimap2",
    min_missing_reads: int = _MIN_MISSING_READS_TO_RESCUE,
    min_missing_fraction: float = _MIN_MISSING_FRACTION_TO_RESCUE,
) -> GeneAssignmentMatrix:
    """NOT CALLED: unwired after it failed to fix H2009 DQB1 and corrupted
    DRB4/MICB (rescued reads lack a mismatch-rate gate). Do not re-wire
    without addressing both.

    Recovers reads Stage 0 sieved to `gene` that Stage 1's n_best=1 matrix
    assigned elsewhere: if at least `min_missing_reads` and
    `min_missing_fraction` are missing, densely realign them against the
    gene's pool and merge into `matrix`."""
    sieved_ids = {r.read_id for r in reads if gene in {h.gene for h in r.hits}}
    missing_ids = sieved_ids - set(matrix.read_ids)
    if not missing_ids or len(sieved_ids) == 0:
        return matrix
    if (
        len(missing_ids) < min_missing_reads
        or len(missing_ids) / len(sieved_ids) < min_missing_fraction
    ):
        return matrix

    gene_alleles = gene_allele_pool(gene, pool_meta)
    if not gene_alleles:
        return matrix
    missing_reads = [r for r in reads if r.read_id in missing_ids]
    allele_sequences = load_allele_sequences(pool_fasta, gene_alleles)
    rescued = realign_to_candidates(
        gene, missing_reads, allele_sequences, n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    if not rescued.read_ids:
        return matrix
    return merge_matrices(matrix, rescued)


def _collapse_to_single_allele(call: TemplateCall) -> TemplateCall:
    """Single-allele TemplateCall from `call`'s baseline; `cn` is kept."""
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
    """Final per-haplotype row from Stage 3 consensus and Stage 4/5 calls."""
    mean_qv = (sum(hap.qv) / len(hap.qv)) if hap.qv else 0.0
    min_qv = min(hap.qv) if hap.qv else 0
    result = HaplotypeCallResult(
        gene=gene, haplotype_index=h_idx, cn=cn_int,
        called_allele=typing_call.called_allele,
        proposed_name=novelty_call.proposed_name,
        taxonomy=novelty_call.taxonomy,
        extension_allele=novelty_call.extension_allele,
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
    pre_search_min_reads_per_haplotype: int = 3,
    pre_search_min_haplotype_read_ratio: float = 0.0,
    max_read_mismatch_rate: float = 0.06,
    stage1_n_best: int = 1,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
    progress: Callable[[str], None] | None = None,
) -> tuple[SampleResult, list[CandidateRead], dict[str, GeneAssignmentMatrix], dict[str, GeneAssignmentMatrix]]:
    """Body of `run_sample()`; also returns Stage 0 reads and Stage 1
    matrices so `run_sample_phased()` can reuse them.

    progress: optional per-stage callback (cli.py prints elapsed time).

    stage1_n_best: hits reported per read in Stage 1 (default 1). Higher
    values let redundant-name clusters (HLA-A) and paralogue crowding
    (DRB1) outvote the true allele on depth. Provisional.

    control_depth / control_n_reads: precomputed chr17 control depth (tests;
    cli.py's --bam path reads it off the mini-BAM). None = compute it by
    aligning the whole FASTQ.

    pre_search_min_reads_per_haplotype / _ratio: Stage 2's pre-search combo
    floor (genotype.py). Separate from min_reads_per_haplotype /
    min_haplotype_read_ratio, the post-consensus collapse check, so each can
    be ablated alone. The pre-search ratio defaults to 0.0 (off).
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
    # chr17 control depth from the whole FASTQ: control reads never pass
    # the HLA anchor sieve.
    if control_depth is None:
        control_depth, control_n_reads = compute_chr17_control_depth(
            fastq_path, chr17_control_fasta=chr17_control_fasta,
            n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
        )
    if progress:
        progress("Stage 1: estimating copy number")
    gene_cn = estimate_cn(reads, control_depth, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta_path)
    manifest = load_gene_manifest(genes_yaml)

    if progress:
        progress("Stage 1 (C4): assigning reads to C4A/C4B templates")
    # C4A/C4B: separate alignment of the same reads against the C4 bundle;
    # reported as "C4" in gene_cn, never typed by Stages 2-5.
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
    # C4 Phase 2 (A/B, long/short) from c4_matrices; no new alignment.
    c4_ab_call, c4_form_call = estimate_c4_form(
        c4_matrices, reads_by_id,
        meta_path=c4_pool_meta, pool_fasta=c4_pool_fasta,
    )
    c4_per_gene_cn = estimate_c4_per_gene_cn(gene_cn[C4_REPORTED_GENE], c4_ab_call)

    if progress:
        progress("Stage 1 (C4): chr17-control per-form CN")
    # chr17-control per-form C4 CN (production C4 total in calls.tsv; the
    # calibrator-panel total above stays in qc.json as a cross-check).
    # compute_c4_chr17_control=False skips it for tests.
    c4_chr17_control = estimate_c4_cn_via_chr17_control(
        fastq_path, chr17_control_fasta=chr17_control_fasta, c4_flanked_fasta=c4_flanked_fasta,
        control_depth=control_depth, control_n_reads=control_n_reads,
        n_threads=n_threads, minimap2_bin=minimap2_bin, samtools_bin=samtools_bin,
    ) if compute_c4_chr17_control else None
    c4_joint_call = None if c4_chr17_control is None else call_c4_forms_jointly(
        c4_chr17_control,
        n_a=c4_ab_call.n_a if c4_ab_call is not None else 0,
        n_b=c4_ab_call.n_b if c4_ab_call is not None else 0,
    )

    if progress:
        progress(f"Stages 2-5: typing {len(manifest)} gene(s)")

    haplotype_calls: list[HaplotypeCallResult] = []
    consensus_records: list[HaplotypeConsensus] = []
    novelty_by_key: dict[tuple[str, int], NoveltyCall] = {}
    failures: list[GeneFailure] = []
    # Stage 2-3 per gene; Stage 4-5 typing runs batched afterward.
    pending_haplotypes: list[tuple[str, int, int, HaplotypeConsensus, tuple[str, ...]]] = []

    manifest_genes = sorted(manifest)
    # Per-gene cn_int; genes without a CN estimate fail in the loop below.
    raw_cn_int: dict[str, int] = {}
    for gene in manifest_genes:
        entry = manifest[gene]
        this_gene_cn = gene_cn.get(gene)
        if this_gene_cn is None or this_gene_cn.cn_estimate is None:
            continue
        raw_cn_int[gene] = _raw_cn_int_for_gene(entry, this_gene_cn.cn_estimate, cn_round, max_cn)
    # DRB3/4/5 exclusivity needs all three raw cn_ints at once.
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
                min_reads_per_haplotype=pre_search_min_reads_per_haplotype,
                min_haplotype_read_ratio=pre_search_min_haplotype_read_ratio,
            )
            if call.cn > 1 and call.lrt_statistic < min_heterozygous_lrt:
                # Weak LRT: collapse to the single best allele.
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
                    # Too few reads (absolute or ratio) on a haplotype:
                    # collapse to the dominant allele.
                    call = _collapse_to_single_allele(call)
                    haps = build_gene_consensus(
                        matrix, reads, call, pool_fasta=pool_fasta, config=ont_config,
                        n_threads=n_threads, minimap2_bin=minimap2_bin,
                    )
        except ValueError as e:
            failures.append(GeneFailure(gene, f"template selection/consensus failed: {e}"))
            continue

        if len(haps) == 1 and call.cn > 1:
            # Collapsed but still `cn` copies: one row per copy, so a
            # homozygote isn't misread as a deletion.
            haps = haps * call.cn

        for h_idx, hap in enumerate(haps):
            pending_haplotypes.append((gene, h_idx, cn_int, hap, call.candidates))

    # Batched Stage 4 typing for every haplotype. call.candidates is passed
    # so Stage 4 can densely realign when the whole-pool search drops a
    # shorter-but-correct candidate (TAP2).
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
                typing_call, hap.sequence, consensus_variants=hap.variants,
                pool_fasta=pool_fasta, pool_meta=pool_meta,
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
        c4_joint_call=c4_joint_call,
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
    pre_search_min_reads_per_haplotype: int = 3,
    pre_search_min_haplotype_read_ratio: float = 0.0,
    max_read_mismatch_rate: float = 0.06,
    stage1_n_best: int = 1,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
    progress: Callable[[str], None] | None = None,
) -> SampleResult:
    """`_run_sample_core()` without the extra return values."""
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
        pre_search_min_reads_per_haplotype=pre_search_min_reads_per_haplotype,
        pre_search_min_haplotype_read_ratio=pre_search_min_haplotype_read_ratio,
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
    """Phased-routing decision (DESIGN.md S10.3): reads present, HP-tagged
    reads in a single PS block, tagged fraction >= the floor. Shared by the
    gene loop and the C4 check. `haplotypes_called` is filled in later."""
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
    """Calibrator reads usable for one haplotype's C4 CN panel. HP=1 only
    means the same chromosome across loci within one PS block, so a
    calibrator gene contributes only if it routed "phased" with the same
    single PS value as C4."""
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
    pre_search_min_reads_per_haplotype: int = 3,
    pre_search_min_haplotype_read_ratio: float = 0.0,
    max_read_mismatch_rate: float = 0.06,
    stage1_n_best: int = 1,
    min_phased_tagged_fraction: float = 0.5,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    samtools_bin: str = "samtools",
    progress: Callable[[str], None] | None = None,
) -> SampleResult:
    """Phased-input mode (DESIGN.md S10). Takes the FASTQ and the mini-BAM
    (which keeps HP/PS tags).

    Runs the unphased pipeline first (`_run_sample_core()`), then for each
    gene that routes "phased" (`_decide_phase_routing`) replaces its calls
    with per-HP results: Stage 1 matrix subset by HP, Stage 2 with cn=1,
    Stage 3, Stage 4-5; `haplotype_index` is the HP value. Other genes keep
    the unphased calls. Every manifest gene gets a `GenePhaseRouting`.

    A gene routes "phased" only if >= `min_phased_tagged_fraction` (0.5) of
    its reads are HP-tagged (HLA-C had 11/72 tagged and mistyped). A
    haplotype with < `min_reads_per_haplotype` reads is a failure, not a
    call.

    C4 (S10.6): the same routing runs on the C4A/C4B read pool; per phased
    haplotype, form classifiers and total CN rerun on that haplotype's reads.
    The CN calibrator panel uses only genes phased in C4's PS block
    (`_c4_calibrator_reads_for_haplotype`); none -> cn_estimate None. A
    haplotype carrying an RCCX duplication can still show both A and B.

    `gene_cn` stays the unphased sample-wide estimate.
    """
    if ont_config is None:
        ont_config = load_ont_r10_config()
    if typing_params is None:
        typing_params = TypingParams.load()

    # Reuse the unphased run's sieved reads and Stage 1 matrices.
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
        pre_search_min_reads_per_haplotype=pre_search_min_reads_per_haplotype,
        pre_search_min_haplotype_read_ratio=pre_search_min_haplotype_read_ratio,
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
    # Stage 2-3 per phased gene; Stage 4-5 batched afterward, then routing
    # is finalised.
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
            # Row-subset Stage 1's matrix (equivalent to rebuilding it).
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
                # One realign+consensus for both haplotypes, keeping the
                # real HP partition.
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

    # Batched Stage 4-5 typing for phased haplotypes.
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
                typing_call, hap.sequence, consensus_variants=hap.variants,
                pool_fasta=pool_fasta, pool_meta=pool_meta,
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

    # Finalise routing for genes that reached typing.
    for gene in sorted({g for g, _hp, _hap in pending_phased_haplotypes}):
        called = tuple(sorted(haplotypes_called_by_gene.get(gene, [])))
        routing[gene] = replace(routing[gene], haplotypes_called=called)
        if not called:
            phased_failures.append(GeneFailure(
                gene,
                "routed to the phased path but neither haplotype produced "
                "a real call (see per-haplotype failures above)",
            ))

    # C4 (not in manifest) as an extra routed pseudo-gene, same rules.
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
            # Reuse sample-wide c4_matrices.
            hp_c4_matrices = classical_matrices_for_reads(c4_matrices, hp_c4_reads, C4_GENES)
            hp_c4_reads_by_id = {r.read_id: r.sequence for r in hp_c4_reads}
            ab_call, form_call = estimate_c4_form(
                hp_c4_matrices, hp_c4_reads_by_id,
                meta_path=c4_pool_meta, pool_fasta=c4_pool_fasta,
            )
            c4_phased_ab_calls[hp_val] = ab_call
            c4_phased_form_calls[hp_val] = form_call

            # Per-haplotype C4 CN with a PS-matched calibrator panel.
            hp_calibrator_reads = _c4_calibrator_reads_for_haplotype(
                hp_val, c4_routing.ps_values_seen, c4_calibrator_genes,
                routing, reads_by_gene, _phase_of,
            )
            # Calibrator genes are already in Stage 1's matrices.
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

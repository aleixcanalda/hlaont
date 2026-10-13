"""Unit tests for Stage 2 template set selection (DESIGN.md S2e)."""
from __future__ import annotations

import gzip
import math
import random

import numpy as np
import pytest
import yaml

from hlaont.stages.assign import GeneAssignmentMatrix, build_gene_matrices
from hlaont.stages.genotype import (
    _MIN_COMBOS_FOR_PARALLEL,
    ErrorModel,
    candidate_alleles,
    context_log_likelihoods,
    load_error_model,
    load_ont_r10_config,
    narrow_candidates_by_mismatch_rate,
    read_log_likelihoods,
    select_template_set,
    select_template_set_flat,
    select_template_set_ont_r10,
)

from hlaont.input.fastq import AnchorHit, CandidateRead

_MODEL = ErrorModel(platform="test", edit_error_rate=0.01, unaligned_edit_rate=0.75)


def _matrix(
    gene: str,
    allele_ids: tuple[str, ...],
    read_ids: tuple[str, ...],
    cells: dict[tuple[str, str], tuple[int, int, int]],
) -> GeneAssignmentMatrix:
    """GeneAssignmentMatrix from (matches, aln_block_len, nm) cells, without
    minimap2."""
    matches = np.zeros((len(read_ids), len(allele_ids)), dtype=np.int32)
    aln_len = np.zeros_like(matches)
    nm = np.zeros_like(matches)
    r_idx = {r: i for i, r in enumerate(read_ids)}
    a_idx = {a: i for i, a in enumerate(allele_ids)}
    for (read_id, allele), (m, al, n) in cells.items():
        i, j = r_idx[read_id], a_idx[allele]
        matches[i, j], aln_len[i, j], nm[i, j] = m, al, n
    return GeneAssignmentMatrix(
        gene=gene, read_ids=read_ids, allele_ids=allele_ids,
        matches=matches, aln_block_len=aln_len, nm=nm,
    )


def test_read_log_likelihoods_scores_aligned_and_unaligned_cells():
    m = _matrix(
        "G", ("A1", "A2"), ("r1",),
        {("r1", "A1"): (100, 100, 0)},  # A2 cell left at (0, 0, 0) -- unaligned
    )
    ll = read_log_likelihoods(m, _MODEL)

    expected_aligned = 100 * np.log(1 - _MODEL.edit_error_rate)
    assert ll[0, 0] == pytest.approx(expected_aligned)

    # Unaligned floor uses this read's own best real span (100, from A1) at
    # the model's unaligned_edit_rate, not zero and not -inf.
    expected_floor = 100 * np.log(_MODEL.unaligned_edit_rate)
    assert ll[0, 1] == pytest.approx(expected_floor)
    assert ll[0, 1] < ll[0, 0]


def test_read_log_likelihoods_columns_restricts_output():
    m = _matrix(
        "G", ("A1", "A2", "A3"), ("r1",),
        {("r1", "A1"): (100, 100, 0), ("r1", "A2"): (70, 100, 30)},
    )
    full = read_log_likelihoods(m, _MODEL)
    restricted = read_log_likelihoods(m, _MODEL, columns=("A2", "A1"))
    assert restricted.shape == (1, 2)
    assert restricted[0, 0] == full[0, 1]  # A2
    assert restricted[0, 1] == full[0, 0]  # A1


def test_candidate_alleles_excludes_unsupported_allele():
    m = _matrix(
        "G", ("A1", "A2", "A3"), ("r1",),
        {("r1", "A1"): (100, 100, 0), ("r1", "A2"): (70, 100, 30)},
        # A3 never aligned to anything -- must not appear as a candidate.
    )
    assert candidate_alleles(m) == ("A1", "A2")


def test_select_cn1_picks_best_single_allele():
    m = _matrix(
        "G", ("A1", "A2"), ("r1", "r2"),
        {
            ("r1", "A1"): (100, 100, 1), ("r1", "A2"): (70, 100, 30),
            ("r2", "A1"): (100, 100, 1), ("r2", "A2"): (70, 100, 30),
        },
    )
    call = select_template_set_flat(m, cn=1, error_model=_MODEL)
    assert call.alleles == ("A1",)
    assert call.single_allele == "A1"
    assert call.lrt_statistic == 0.0


def _diploid_matrix(hetero: bool) -> GeneAssignmentMatrix:
    # 20 reads total. A3 is a red herring with zero real support anywhere.
    read_ids = tuple(f"r{i}" for i in range(20))
    alleles = ("A1", "A2", "A3")
    cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    good, bad = (100, 100, 2), (70, 100, 180)
    for i, r in enumerate(read_ids):
        if hetero and i >= 10:
            cells[(r, "A1")] = bad
            cells[(r, "A2")] = good
        else:
            cells[(r, "A1")] = good
            cells[(r, "A2")] = bad
        # A3: no entry at all for any read -> no real alignment anywhere.
    return _matrix("G", alleles, read_ids, cells)


def test_select_cn2_recovers_true_heterozygous_pair():
    m = _diploid_matrix(hetero=True)
    call = select_template_set_flat(m, cn=2, error_model=_MODEL)

    assert call.alleles == ("A1", "A2")
    assert call.weights[0] == pytest.approx(0.5, abs=0.05)
    assert call.weights[1] == pytest.approx(0.5, abs=0.05)
    assert call.n_candidates == 2  # A3 excluded, never entered the search
    # Strong evidence for two distinct alleles.
    assert call.lrt_statistic > 20.0


def test_select_cn2_homozygous_sample_gives_small_lrt():
    m = _diploid_matrix(hetero=False)
    call = select_template_set_flat(m, cn=2, error_model=_MODEL)

    # cn=2 still returns two alleles, but the LRT vs single-allele is small.
    assert call.single_allele == "A1"
    assert call.lrt_statistic < 5.0


def test_select_cn2_falls_back_to_single_allele_if_only_one_candidate():
    # One supported candidate with cn=2 (HLA-K/U): single-allele call with
    # cn kept, not an error.
    m = _matrix("G", ("A1",), ("r1",), {("r1", "A1"): (100, 100, 0)})
    call = select_template_set_flat(m, cn=2, error_model=_MODEL)

    assert call.cn == 2
    assert call.alleles == ("A1",)
    assert call.weights == (1.0,)
    assert call.single_allele == "A1"
    assert call.lrt_statistic == 0.0
    assert call.n_candidates == 1


def test_select_template_set_raises_if_zero_candidates():
    # Distinct from the fallback above: zero read-supported candidates at
    # all is a real recruitment failure (nothing to type), not a
    # homozygosity shape, and must stay loud.
    with pytest.raises(ValueError, match="no candidate allele"):
        select_template_set("G", (), np.zeros((5, 0)), cn=2)


def test_select_template_set_rejects_mismatched_log_likelihood_shape():
    with pytest.raises(ValueError, match="log_likelihoods has"):
        select_template_set("G", ("A1", "A2"), np.zeros((5, 3)), cn=2)


def test_select_template_set_is_deterministic():
    m = _diploid_matrix(hetero=True)

    a = select_template_set_flat(m, cn=2, error_model=_MODEL)
    b = select_template_set_flat(m, cn=2, error_model=_MODEL)
    assert a == b


def _outlier_pair_log_likelihoods() -> tuple[tuple[str, ...], np.ndarray]:
    """20 reads x 3 candidates (H2009 DQB1 shape): A1/A2 are the true pair
    with small per-read differences; A3 has 2 reads with huge per-read
    gaps, so summed logL prefers (A1, A3)."""
    candidates = ("A1", "A2", "A3")
    ll = np.zeros((20, 3))
    ll[0:2] = [-50.0, -50.0, -1.0]  # 2 outlier reads: real signal is A3, not A1/A2
    for i in range(2, 7):
        ll[i] = [-1.0, -3.0, -1000.0]  # 5 reads, slightly favour A1
    for i in range(7, 12):
        ll[i] = [-3.0, -1.0, -1000.0]  # 5 reads, slightly favour A2
    ll[12:20] = [-1.0, -1.0, -1000.0]  # 8 neutral reads
    return candidates, ll


def test_select_template_set_without_floor_reproduces_outlier_pair_bug():
    # Without the floor the spurious pair wins (reproduces the bug).
    candidates, ll = _outlier_pair_log_likelihoods()
    call = select_template_set("G", candidates, ll, cn=2)
    assert call.alleles == ("A1", "A3")


def test_select_template_set_read_count_floor_rejects_outlier_pair():
    candidates, ll = _outlier_pair_log_likelihoods()
    call = select_template_set(
        "G", candidates, ll, cn=2,
        min_reads_per_haplotype=3, min_haplotype_read_ratio=0.0,
    )
    assert call.alleles == ("A1", "A2")


def test_select_template_set_read_count_ratio_floor_rejects_outlier_pair():
    # Isolates the ratio floor from the absolute floor: 3 outlier reads
    # (clears an absolute floor of 0, i.e. disabled here) but a 3-vs-17
    # split fails a 0.2 ratio.
    candidates = ("A1", "A2", "A3")
    ll = np.zeros((20, 3))
    ll[0:3] = [-50.0, -50.0, -1.0]
    for i in range(3, 7):
        ll[i] = [-1.0, -3.0, -1000.0]
    for i in range(7, 11):
        ll[i] = [-3.0, -1.0, -1000.0]
    ll[11:20] = [-1.0, -1.0, -1000.0]
    call = select_template_set(
        "G", candidates, ll, cn=2,
        min_reads_per_haplotype=0, min_haplotype_read_ratio=0.2,
    )
    assert call.alleles == ("A1", "A2")


def test_select_template_set_read_count_floor_collapses_when_every_combo_fails():
    # Every cn=2 combo fails the floor -> falls back to the single-allele
    # shape (same as the too-few-candidates case above), not a crash or a
    # disqualified-anyway pick.
    candidates = ("A1", "A2")
    ll = np.array([[-1.0, -50.0], [-1.0, -50.0], [-50.0, -1.0]])
    call = select_template_set(
        "G", candidates, ll, cn=2,
        min_reads_per_haplotype=5, min_haplotype_read_ratio=0.0,
    )
    assert call.cn == 2
    assert call.alleles == ("A1",)
    assert call.weights == (1.0,)
    assert call.lrt_statistic == 0.0


def test_select_template_set_read_count_floor_parallel_matches_sequential():
    # Parallel path (_score_combo) must apply the floor like the sequential one.
    read_ids = tuple(f"r{i}" for i in range(20))
    noise_alleles = tuple(f"N{i}" for i in range(25))
    alleles = ("A1", "A2") + noise_alleles
    cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    good, bad = (100, 100, 2), (70, 100, 180)
    for i, r in enumerate(read_ids):
        if i >= 10:
            cells[(r, "A1")] = bad
            cells[(r, "A2")] = good
        else:
            cells[(r, "A1")] = good
            cells[(r, "A2")] = bad
    for n_allele in noise_alleles:
        cells[("r0", n_allele)] = (10, 100, 90)
    m = _matrix("G", alleles, read_ids, cells)
    candidates = candidate_alleles(m)
    assert math.comb(len(candidates), 2) >= _MIN_COMBOS_FOR_PARALLEL

    ll = read_log_likelihoods(m, _MODEL, columns=candidates)
    sequential = select_template_set(
        "G", candidates, ll, cn=2, n_threads=1,
        min_reads_per_haplotype=3, min_haplotype_read_ratio=0.1,
    )
    parallel = select_template_set(
        "G", candidates, ll, cn=2, n_threads=2,
        min_reads_per_haplotype=3, min_haplotype_read_ratio=0.1,
    )
    assert parallel == sequential
    assert parallel.alleles == ("A1", "A2")


def test_narrow_candidates_keeps_close_drops_far():
    m = _matrix(
        "G", ("A1", "A2", "A3"), ("r0", "r1", "r2", "r3", "r4"),
        {
            **{(r, "A1"): (100, 100, 0) for r in ("r0", "r1", "r2", "r3", "r4")},
            **{(r, "A2"): (97, 100, 3) for r in ("r0", "r1", "r2", "r3", "r4")},  # rate 0.03
            ("r0", "A3"): (10, 100, 90),  # rate 0.9 -- one weak touch, real candidate_alleles() hit
        },
    )
    candidates = candidate_alleles(m)
    assert candidates == ("A1", "A2", "A3")

    kept = narrow_candidates_by_mismatch_rate(m, candidates)
    assert kept == ("A1", "A2")


def test_narrow_candidates_floor_prevents_zero_threshold_collapse():
    """With a 0.0 best rate, the floor keeps a real second allele."""
    m = _matrix(
        "G", ("A1", "A2"), ("r0", "r1"),
        {
            ("r0", "A1"): (100, 100, 0), ("r1", "A1"): (100, 100, 0),  # rate 0.0
            ("r0", "A2"): (95, 100, 5), ("r1", "A2"): (95, 100, 5),  # rate 0.05
        },
    )
    candidates = candidate_alleles(m)
    kept = narrow_candidates_by_mismatch_rate(m, candidates)
    assert kept == ("A1", "A2")


def test_narrow_candidates_caps_survivor_count():
    """All 5 candidates pass the multiplier/floor filter (rates 0.01-0.05,
    all within 10x of the best), but max_candidates=3 should keep only the
    3 lowest-mismatch ones."""
    read_ids = ("r0", "r1")
    alleles = tuple(f"A{i}" for i in range(5))
    rates = [0.01, 0.02, 0.03, 0.04, 0.05]  # A0 best .. A4 worst
    cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    for a, rate in zip(alleles, rates):
        nm = round(rate * 100)
        for r in read_ids:
            cells[(r, a)] = (100 - nm, 100, nm)
    m = _matrix("G", alleles, read_ids, cells)

    candidates = candidate_alleles(m)
    kept = narrow_candidates_by_mismatch_rate(m, candidates, max_candidates=3)
    assert kept == ("A0", "A1", "A2")


def test_narrow_candidates_prioritises_depth_over_single_read_noise():
    """A 27-read true allele outranks one-read flukes with lower rates
    (HLA-A). The 5 noise alleles share a signature, so 2 survive: 3 total."""
    read_ids = tuple(f"r{i}" for i in range(10))
    # TRUE: well-supported (10 reads), moderate rate (0.05).
    cells = {(r, "TRUE"): (95, 100, 5) for r in read_ids}
    # 5 one-read "noise" alleles, each with a much lower rate (0.01) purely
    # from a single fluke read -- would win a pure-rate ranking, and are
    # indistinguishable from each other (same read, same rate).
    noise_alleles = tuple(f"NOISE{i}" for i in range(5))
    for a in noise_alleles:
        cells[("r0", a)] = (99, 100, 1)
    alleles = ("TRUE",) + noise_alleles
    m = _matrix("G", alleles, read_ids, cells)

    candidates = candidate_alleles(m)
    kept = narrow_candidates_by_mismatch_rate(m, candidates, max_candidates=3)
    assert "TRUE" in kept
    assert len(kept) == 3


def test_narrow_candidates_depth_priority_holds_against_multi_read_noise():
    """Depth-first ranking also beats multi-read noise (HLA-C: 62 reads vs
    4-9). 3 total survive (per-signature cap)."""
    read_ids = tuple(f"r{i}" for i in range(30))
    # TRUE: deep support (30 reads), moderate rate.
    cells = {(r, "TRUE"): (95, 100, 5) for r in read_ids}  # rate 0.05
    # Off-target candidates with real but shallower support (8 reads each)
    # at a better rate than TRUE -- would win a pure-rate ranking, and are
    # indistinguishable from each other (same 8 reads, same rate).
    thin_alleles = tuple(f"THIN{i}" for i in range(5))
    for a in thin_alleles:
        for r in read_ids[:8]:
            cells[(r, a)] = (98, 100, 2)  # rate 0.02, 8 reads
    alleles = ("TRUE",) + thin_alleles
    m = _matrix("G", alleles, read_ids, cells)

    candidates = candidate_alleles(m)
    kept = narrow_candidates_by_mismatch_rate(m, candidates, max_candidates=3)
    assert "TRUE" in kept
    assert len(kept) == 3


def test_narrow_candidates_dedup_preserves_distinct_second_haplotype():
    """Read-indistinguishable names (HLA-A 03:01:01 cluster) take only 2 cap
    slots, leaving room for a distinct second haplotype."""
    cluster_reads = tuple(f"c{i}" for i in range(27))
    other_reads = tuple(f"o{i}" for i in range(14))
    cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    # 10 read-indistinguishable names for the first haplotype -- same
    # reads, same rate (0.012) to 3dp, i.e. the same floor-scaled bucket.
    cluster_alleles = tuple(f"HAP_A_{i}" for i in range(10))
    for a in cluster_alleles:
        for r in cluster_reads:
            cells[(r, a)] = (988, 1000, 12)  # rate 0.012
    # A real, distinct second haplotype -- its own reads, its own rate.
    for r in other_reads:
        cells[(r, "HAP_B")] = (985, 1000, 15)  # rate 0.015
    alleles = cluster_alleles + ("HAP_B",)
    m = _matrix("G", alleles, cluster_reads + other_reads, cells)

    candidates = candidate_alleles(m)
    # max_candidates smaller than the cluster size: without the
    # per-signature cap, the cluster alone would fill this and HAP_B would
    # never survive. 2 HAP_A survivors + HAP_B fits within max_candidates=3.
    kept = narrow_candidates_by_mismatch_rate(m, candidates, max_candidates=3)
    assert "HAP_B" in kept
    assert sum(a.startswith("HAP_A_") for a in kept) == 2
    assert len(kept) == 3


def test_narrow_candidates_is_deterministic():
    m = _matrix(
        "G", ("A1", "A2", "A3"), ("r0", "r1"),
        {
            ("r0", "A1"): (100, 100, 0), ("r1", "A1"): (100, 100, 0),
            ("r0", "A2"): (95, 100, 5), ("r1", "A2"): (95, 100, 5),
            ("r0", "A3"): (10, 100, 90),
        },
    )
    candidates = candidate_alleles(m)
    a = narrow_candidates_by_mismatch_rate(m, candidates)
    b = narrow_candidates_by_mismatch_rate(m, candidates)
    assert a == b == ("A1", "A2")


def test_select_template_set_parallel_matches_sequential():
    """n_threads > 1 gives the same result as sequential (map() keeps order).
    Padded past _MIN_COMBOS_FOR_PARALLEL to exercise the pool path."""
    read_ids = tuple(f"r{i}" for i in range(20))
    noise_alleles = tuple(f"N{i}" for i in range(25))
    alleles = ("A1", "A2") + noise_alleles
    cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    good, bad = (100, 100, 2), (70, 100, 180)
    for i, r in enumerate(read_ids):
        if i >= 10:
            cells[(r, "A1")] = bad
            cells[(r, "A2")] = good
        else:
            cells[(r, "A1")] = good
            cells[(r, "A2")] = bad
    for n_allele in noise_alleles:
        # One weak real alignment each -- enough for candidate_alleles() to
        # include it, not enough to compete with the true A1/A2 pair.
        cells[("r0", n_allele)] = (10, 100, 90)
    m = _matrix("G", alleles, read_ids, cells)

    n_candidates = len(candidate_alleles(m))
    assert math.comb(n_candidates, 2) >= _MIN_COMBOS_FOR_PARALLEL

    sequential = select_template_set_flat(m, cn=2, error_model=_MODEL, n_threads=1)
    parallel = select_template_set_flat(m, cn=2, error_model=_MODEL, n_threads=2)
    assert parallel == sequential
    assert parallel.alleles == ("A1", "A2")


def test_load_error_model(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({
        "platform": "ont_r10", "edit_error_rate": 0.02, "unaligned_edit_rate": 0.8,
    }))
    model = load_error_model(path)
    assert model == ErrorModel(platform="ont_r10", edit_error_rate=0.02, unaligned_edit_rate=0.8)


def test_load_ont_r10_config_loads_default():
    config = load_ont_r10_config()
    assert config.context_model.kmer_size == 7
    assert len(config.context_model.probs) == 16384
    assert 0 < config.unaligned_edit_rate < 1


# --- End-to-end: real minimap2 alignment -> spans -> context scoring ---

_RNG = random.Random(0)


def _random_seq(n: int) -> str:
    return "".join(_RNG.choice("ACGT") for _ in range(n))


def _mutate(seq: str, n_subs: int, rng: random.Random) -> str:
    bases = list(seq)
    positions = rng.sample(range(len(bases)), n_subs)
    for p in positions:
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = rng.choice(alts)
    return "".join(bases)


def _candidate_read(read_id: str, sequence: str, gene: str) -> CandidateRead:
    hits = (AnchorHit(gene=gene, allele=f"{gene}-anchor", coverage=1.0, mismatch_rate=0.0),)
    return CandidateRead(read_id=read_id, sequence=sequence, quality=None, hits=hits)


def test_select_template_set_ont_r10_end_to_end(tmp_path):
    allele_a = _random_seq(700)
    allele_b = _mutate(allele_a, 5, random.Random(1))

    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">GENE1*01:01\n{allele_a}\n")
        fh.write(f">GENE1*01:02\n{allele_b}\n")
    meta = tmp_path / "pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        fh.write("GENE1\tGENE1*01:01\n")
        fh.write("GENE1\tGENE1*01:02\n")

    reads = (
        [_candidate_read(f"a{i}", allele_a, "GENE1") for i in range(10)]
        + [_candidate_read(f"b{i}", allele_b, "GENE1") for i in range(10)]
    )

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    assert m.spans  # confirms assign.py actually captured alignment spans

    call = select_template_set_ont_r10(m, reads, cn=2, pool_fasta=fasta)
    assert set(call.alleles) == {"GENE1*01:01", "GENE1*01:02"}
    assert call.lrt_statistic > 0


def test_context_log_likelihoods_matches_select_template_set_ont_r10_inputs(tmp_path):
    """context_log_likelihoods() alone favours each read's true allele."""
    allele_a = _random_seq(700)
    allele_b = _mutate(allele_a, 5, random.Random(2))

    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">GENE1*01:01\n{allele_a}\n")
        fh.write(f">GENE1*01:02\n{allele_b}\n")
    meta = tmp_path / "pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        fh.write("GENE1\tGENE1*01:01\n")
        fh.write("GENE1\tGENE1*01:02\n")

    reads = [_candidate_read("a0", allele_a, "GENE1")]
    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    candidates = candidate_alleles(m)
    config = load_ont_r10_config()
    read_sequences = {"a0": allele_a}
    allele_sequences = {"GENE1*01:01": allele_a, "GENE1*01:02": allele_b}

    ll = context_log_likelihoods(m, read_sequences, allele_sequences, candidates, config)
    a1_col = candidates.index("GENE1*01:01")
    a2_col = candidates.index("GENE1*01:02")
    assert ll[0, a1_col] > ll[0, a2_col]


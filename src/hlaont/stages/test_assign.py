"""Unit tests for the shared reads x alleles assignment matrix (DESIGN.md S1/S2e)."""
from __future__ import annotations

import gzip
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from assign import (  # noqa: E402
    GeneAssignmentMatrix,
    build_gene_matrices,
    best_gene_per_read,
    gene_allele_pool,
    merge_matrices,
    subset_matrix_by_read_ids,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "input"))
from fastq import AnchorHit, CandidateRead  # noqa: E402

# Synthetic pool: two genes. GENE1 has two close alleles (a handful of
# substitutions apart); GENE2 is a *near*-duplicate of GENE1*01:01 (~5%
# substitutions) -- close enough that minimap2 will actually align a GENE1
# read to it, which is what makes the gene-restriction test meaningful:
# without the restriction-to-sieved-gene filter, that spurious cross-gene
# hit would otherwise land in the matrix.
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


GENE1_A = _random_seq(600)
GENE1_B = _mutate(GENE1_A, 3, random.Random(1))
GENE2_A = _mutate(GENE1_A, 30, random.Random(2))  # ~5% divergent, unrelated gene
GENE3_A = _random_seq(600)  # genuinely unrelated, for a clean-negative case


def _write_pool(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">GENE1*01:01\n{GENE1_A}\n")
        fh.write(f">GENE1*01:02\n{GENE1_B}\n")
        fh.write(f">GENE2*01:01\n{GENE2_A}\n")
        fh.write(f">GENE3*01:01\n{GENE3_A}\n")

    meta = tmp_path / "pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        fh.write("GENE1\tGENE1*01:01\n")
        fh.write("GENE1\tGENE1*01:02\n")
        fh.write("GENE2\tGENE2*01:01\n")
        fh.write("GENE3\tGENE3*01:01\n")

    return fasta, meta


def _read(read_id: str, sequence: str, genes: tuple[str, ...]) -> CandidateRead:
    hits = tuple(
        AnchorHit(gene=g, allele=f"{g}-anchor", coverage=1.0, mismatch_rate=0.0)
        for g in genes
    )
    return CandidateRead(read_id=read_id, sequence=sequence, quality=None, hits=hits)


def test_matrix_restricted_to_sieved_gene(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    # r1 is an exact copy of GENE1*01:01 and would also align to GENE2*01:01
    # (only ~5% diverged) -- but Stage 0 only ever sieved it to GENE1.
    reads = [_read("r1", GENE1_A, genes=("GENE1",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)

    assert set(matrices) == {"GENE1"}
    assert matrices["GENE1"].read_ids == ("r1",)
    assert matrices["GENE1"].allele_ids == ("GENE1*01:01", "GENE1*01:02")


def test_matrix_captures_alignment_span_for_context_scoring(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [_read("r1", GENE1_A, genes=("GENE1",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]

    i = m.read_ids.index("r1")
    j = m.allele_ids.index("GENE1*01:01")
    assert (i, j) in m.spans
    span = m.spans[(i, j)]
    assert span.strand in ("+", "-")
    assert span.cigar  # non-empty CIGAR string from minimap2's cg:Z: tag
    assert span.ref_end > span.ref_start
    assert span.read_end > span.read_start

    # The diverged allele was still aligned (just worse), so it also has a
    # captured span -- spans exist for every real alignment, not just the
    # best-matching one.
    j2 = m.allele_ids.index("GENE1*01:02")
    assert (i, j2) in m.spans


def test_matrix_has_full_gene_column_set_and_mismatch_rates(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [_read("r1", GENE1_A, genes=("GENE1",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    rates = m.mismatch_rate()[0]  # single read row

    a1_col = m.allele_ids.index("GENE1*01:01")
    a2_col = m.allele_ids.index("GENE1*01:02")
    assert rates[a1_col] == 0.0
    assert 0.0 < rates[a2_col] < 1.0


def test_matrix_correctly_built_when_properly_sieved_to_second_gene(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [_read("r2", GENE2_A, genes=("GENE2",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)

    assert set(matrices) == {"GENE2"}
    assert matrices["GENE2"].read_ids == ("r2",)
    assert matrices["GENE2"].mismatch_rate()[0][matrices["GENE2"].allele_ids.index("GENE2*01:01")] == 0.0


def test_no_matrix_when_read_matches_nothing_in_its_sieved_gene(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    # Sieved to GENE3, but the sequence is GENE1's -- no real GENE3 alignment
    # exists to restrict down to, so no matrix entry should appear at all.
    reads = [_read("r_orphan", GENE1_A, genes=("GENE3",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)

    assert matrices == {}


def test_best_gene_per_read_resolves_ambiguous_sieve_assignment(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    # r3 was sieved to both GENE1 and GENE2 (ambiguous at the anchor level),
    # but its actual sequence is an exact GENE1 copy against a ~5%-diverged
    # GENE2 -- a real, clear winner (far outside the margin-gated
    # resolution's 0.003 tie tolerance), so it must resolve to full 1.0
    # credit on GENE1 alone, not a split or default-to-iteration-order pick.
    reads = [_read("r3", GENE1_A, genes=("GENE1", "GENE2"))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    resolved = best_gene_per_read(matrices)

    assert resolved == {"r3": {"GENE1": 1.0}}


def test_best_gene_per_read_margin_gate_splits_near_tied_reads():
    # Hand-built matrices (no real alignment needed) isolate the
    # margin-gated resolution logic itself (assign.py's
    # _GENE_RESOLUTION_MISMATCH_MARGIN docstring). r_tied's own-pool rate
    # is 0.31% against GENE1 vs 0.12% against GENE2 -- the exact real
    # HG00097 HLA-H/HLA-T gap that motivated this mechanism -- both within
    # the 0.003 margin of each other, so its 1.0 depth credit must split
    # evenly rather than going entirely to GENE2 (the old hard-argmin
    # winner). r_clear's gap (0% vs 5%) is far outside the margin and must
    # still resolve to a single full-credit winner, unaffected.
    def _matrix(gene: str, nm_by_read: dict[str, int], aln_len: int = 10000) -> GeneAssignmentMatrix:
        read_ids = tuple(sorted(nm_by_read))
        nm = np.array([[nm_by_read[r]] for r in read_ids], dtype=np.int32)
        aln = np.full((len(read_ids), 1), aln_len, dtype=np.int32)
        return GeneAssignmentMatrix(
            gene=gene, read_ids=read_ids, allele_ids=(f"{gene}*01",),
            matches=aln - nm, aln_block_len=aln, nm=nm,
        )

    matrices = {
        "GENE1": _matrix("GENE1", {"r_tied": 31, "r_clear": 0}),
        "GENE2": _matrix("GENE2", {"r_tied": 12, "r_clear": 500}),
    }

    resolved = best_gene_per_read(matrices)

    assert resolved["r_tied"] == {"GENE1": 0.5, "GENE2": 0.5}
    assert resolved["r_clear"] == {"GENE1": 1.0}


def _write_solo_pool(tmp_path: Path) -> tuple[Path, Path, str]:
    """A single 3000bp allele, no near-identical second allele competing
    for minimap2's chaining -- deliberately not stages/test_assign.py's
    own 600bp GENE1_A/GENE1_B pair, whose near-duplicate alleles make
    minimap2's reported mismatch rate fragile/seed-dependent well below
    where a single, longer, uncontested reference behaves predictably
    (confirmed empirically while calibrating these tests: the shared pool
    stopped reporting hits at ~1.7% mismatch for some reads, while this
    pool reports cleanly and consistently up to ~8%). Needed for reliably
    testing max_read_mismatch_rate at specific rates, not for the
    gene-restriction/span/determinism tests above, which don't need it."""
    ref = _random_seq(3000)
    fasta = tmp_path / "solo_pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">SOLO*01:01\n{ref}\n")
    meta = tmp_path / "solo_pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        fh.write("SOLO\tSOLO*01:01\n")
    return fasta, meta, ref


def test_read_within_default_mismatch_rate_is_kept(tmp_path):
    fasta, meta, ref = _write_solo_pool(tmp_path)
    # 90/3000 = 3% mismatch -- comfortably under the 6% default.
    read = _mutate(ref, 90, random.Random(1))
    reads = [_read("r_ok", read, genes=("SOLO",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)

    assert "SOLO" in matrices
    assert "r_ok" in matrices["SOLO"].read_ids


def test_read_above_default_mismatch_rate_is_dropped(tmp_path):
    fasta, meta, ref = _write_solo_pool(tmp_path)
    # 210/3000 = 7% mismatch -- minimap2 still reports this hit under this
    # module's scoring string (confirmed empirically: ~8-9% is where it
    # stops reporting anything at all), but it's above the 6% default
    # gate, so the read must not survive into the matrix at all.
    read = _mutate(ref, 210, random.Random(2))
    reads = [_read("r_bad", read, genes=("SOLO",))]

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)

    assert matrices == {}


def test_max_read_mismatch_rate_is_configurable(tmp_path):
    fasta, meta, ref = _write_solo_pool(tmp_path)
    # 150/3000 = 5% mismatch -- kept under the 6% default, droppable with a
    # stricter caller-supplied threshold.
    read = _mutate(ref, 150, random.Random(3))
    reads = [_read("r_mid", read, genes=("SOLO",))]

    default_matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    assert "SOLO" in default_matrices

    strict_matrices = build_gene_matrices(
        reads, pool_fasta=fasta, pool_meta=meta, n_threads=2, max_read_mismatch_rate=0.03,
    )
    assert strict_matrices == {}


def test_assignment_is_deterministic_across_reruns(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [
        _read("r1", GENE1_A, genes=("GENE1",)),
        _read("r2", GENE2_A, genes=("GENE2",)),
        _read("r3", GENE1_A, genes=("GENE1", "GENE2")),
    ]

    def run():
        matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
        resolved = best_gene_per_read(matrices)
        return (
            {g: (m.read_ids, m.allele_ids, m.matches.tolist(), m.aln_block_len.tolist(), m.nm.tolist())
             for g, m in matrices.items()},
            resolved,
        )

    result_a = run()
    result_b = run()
    assert result_a == result_b


def test_gene_allele_pool_returns_sorted_full_pool(tmp_path):
    _fasta, meta = _write_pool(tmp_path)
    assert gene_allele_pool("GENE1", meta) == ("GENE1*01:01", "GENE1*01:02")
    assert gene_allele_pool("GENE3", meta) == ("GENE3*01:01",)
    assert gene_allele_pool("NO_SUCH_GENE", meta) == ()


def test_merge_matrices_reunites_disjoint_row_subsets(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [
        _read("r1", GENE1_A, genes=("GENE1",)),
        _read("r2", GENE1_B, genes=("GENE1",)),
        _read("r3", GENE1_A, genes=("GENE1",)),
    ]
    full = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)["GENE1"]

    part_a = subset_matrix_by_read_ids(full, {"r1", "r3"})
    part_b = subset_matrix_by_read_ids(full, {"r2"})
    merged = merge_matrices(part_a, part_b)

    assert merged.gene == full.gene
    assert merged.allele_ids == full.allele_ids
    assert merged.read_ids == full.read_ids  # sorted union == original row set
    for rid in full.read_ids:
        i_full = full.read_ids.index(rid)
        i_merged = merged.read_ids.index(rid)
        assert merged.matches[i_merged, :].tolist() == full.matches[i_full, :].tolist()
        assert merged.aln_block_len[i_merged, :].tolist() == full.aln_block_len[i_full, :].tolist()
        assert merged.nm[i_merged, :].tolist() == full.nm[i_full, :].tolist()
    # Spans survive the merge too, correctly reindexed.
    for (i_full, j), span in full.spans.items():
        rid = full.read_ids[i_full]
        i_merged = merged.read_ids.index(rid)
        assert merged.spans[(i_merged, j)] == span


def test_merge_matrices_rejects_gene_mismatch(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [_read("r1", GENE1_A, genes=("GENE1",))]
    a = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)["GENE1"]
    b = GeneAssignmentMatrix(
        gene="GENE_OTHER", read_ids=("r2",), allele_ids=a.allele_ids,
        matches=np.zeros((1, len(a.allele_ids)), dtype=np.int32),
        aln_block_len=np.zeros((1, len(a.allele_ids)), dtype=np.int32),
        nm=np.zeros((1, len(a.allele_ids)), dtype=np.int32),
    )
    try:
        merge_matrices(a, b)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "gene mismatch" in str(e)


def test_merge_matrices_rejects_overlapping_read_ids(tmp_path):
    fasta, meta = _write_pool(tmp_path)
    reads = [_read("r1", GENE1_A, genes=("GENE1",))]
    full = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)["GENE1"]
    try:
        merge_matrices(full, full)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "present in" in str(e)

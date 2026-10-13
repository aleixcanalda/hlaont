"""Unit tests for Stage 3 consensus (stages/consensus.py)."""
from __future__ import annotations

import gzip
import random

from hlaont.stages.assign import build_gene_matrices
from hlaont.stages.consensus import build_gene_consensus
from hlaont.input.fastq import AnchorHit, CandidateRead
from hlaont.stages.genotype import select_template_set_ont_r10


def _random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _substitute(seq: str, pos: int) -> str:
    alt = next(b for b in "ACGT" if b != seq[pos])
    return seq[:pos] + alt + seq[pos + 1:]


def _mutate(seq: str, n_subs: int, rng: random.Random) -> str:
    bases = list(seq)
    for p in rng.sample(range(len(bases)), n_subs):
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = rng.choice(alts)
    return "".join(bases)


def _candidate_read(read_id: str, sequence: str, gene: str) -> CandidateRead:
    hits = (AnchorHit(gene=gene, allele=f"{gene}-anchor", coverage=1.0, mismatch_rate=0.0),)
    return CandidateRead(read_id=read_id, sequence=sequence, quality=None, hits=hits)


def _build_pool(tmp_path, allele_a: str, allele_b: str):
    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">GENE1*01:01\n{allele_a}\n")
        fh.write(f">GENE1*01:02\n{allele_b}\n")
    meta = tmp_path / "pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        fh.write("GENE1\tGENE1*01:01\n")
        fh.write("GENE1\tGENE1*01:02\n")
    return fasta, meta


def test_consensus_recovers_a_real_haplotype_substitution(tmp_path):
    rng = random.Random(42)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(7))
    fasta, meta = _build_pool(tmp_path, allele_a, allele_b)

    # All haplotype-A reads carry a substitution at 200 absent from the
    # template; polishing should recover it.
    true_hap_a = _substitute(allele_a, 200)
    reads = (
        [_candidate_read(f"a{i}", true_hap_a, "GENE1") for i in range(12)]
        + [_candidate_read(f"b{i}", allele_b, "GENE1") for i in range(12)]
    )

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    call = select_template_set_ont_r10(m, reads, cn=2, pool_fasta=fasta)
    assert set(call.alleles) == {"GENE1*01:01", "GENE1*01:02"}

    results = build_gene_consensus(m, reads, call, pool_fasta=fasta)
    by_allele = {r.template_allele: r for r in results}

    hap_a = by_allele["GENE1*01:01"]
    assert hap_a.n_reads_assigned == 12
    assert hap_a.sequence[200] == true_hap_a[200]
    assert hap_a.sequence[200] != allele_a[200]
    subs = [v for v in hap_a.variants if v.ref and v.alt and len(v.alt) == 1]
    assert any(v.pos == 201 for v in subs)  # 1-based
    called = next(v for v in subs if v.pos == 201)
    assert called.alt == true_hap_a[200]
    assert called.qv > 10  # 12 agreeing reads should be confident
    assert called.dissent == 0  # every one of the 12 reads agreed

    hap_b = by_allele["GENE1*01:02"]
    assert hap_b.n_reads_assigned == 12
    assert hap_b.sequence == allele_b
    assert hap_b.variants == ()


def test_consensus_recovers_an_insertion(tmp_path):
    rng = random.Random(1)
    allele_a = _random_seq(500, rng)
    allele_b = _mutate(allele_a, 5, random.Random(9))
    fasta, meta = _build_pool(tmp_path, allele_a, allele_b)

    true_hap_a = allele_a[:300] + "TTT" + allele_a[300:]
    reads = (
        [_candidate_read(f"a{i}", true_hap_a, "GENE1") for i in range(12)]
        + [_candidate_read(f"b{i}", allele_b, "GENE1") for i in range(12)]
    )

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    call = select_template_set_ont_r10(m, reads, cn=2, pool_fasta=fasta)

    results = build_gene_consensus(m, reads, call, pool_fasta=fasta)
    by_allele = {r.template_allele: r for r in results}
    hap_a = by_allele["GENE1*01:01"]

    ins = [v for v in hap_a.variants if v.ref == ""]
    assert len(ins) == 1
    assert ins[0].alt == "TTT"
    assert len(hap_a.sequence) == len(allele_a) + 3


def test_consensus_variant_dissent_counts_real_minority_reads(tmp_path):
    """`dissent` counts minority reads: 9 call the substitution, 3 the
    template base."""
    rng = random.Random(5)
    allele_a = _random_seq(500, rng)
    allele_b = _mutate(allele_a, 5, random.Random(13))
    fasta, meta = _build_pool(tmp_path, allele_a, allele_b)

    true_hap_a = _substitute(allele_a, 150)
    reads = (
        [_candidate_read(f"a{i}", true_hap_a, "GENE1") for i in range(9)]
        + [_candidate_read(f"a_noisy{i}", allele_a, "GENE1") for i in range(3)]
        + [_candidate_read(f"b{i}", allele_b, "GENE1") for i in range(12)]
    )

    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    call = select_template_set_ont_r10(m, reads, cn=2, pool_fasta=fasta)

    results = build_gene_consensus(m, reads, call, pool_fasta=fasta)
    by_allele = {r.template_allele: r for r in results}
    hap_a = by_allele["GENE1*01:01"]

    subs = [v for v in hap_a.variants if v.ref and v.alt and len(v.alt) == 1 and v.pos == 151]
    assert len(subs) == 1
    called = subs[0]
    assert called.depth == 12
    assert called.dissent == 3


def test_build_gene_consensus_is_deterministic(tmp_path):
    rng = random.Random(3)
    allele_a = _random_seq(400, rng)
    allele_b = _mutate(allele_a, 5, random.Random(11))
    fasta, meta = _build_pool(tmp_path, allele_a, allele_b)

    reads = (
        [_candidate_read(f"a{i}", allele_a, "GENE1") for i in range(8)]
        + [_candidate_read(f"b{i}", allele_b, "GENE1") for i in range(8)]
    )
    matrices = build_gene_matrices(reads, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    m = matrices["GENE1"]
    call = select_template_set_ont_r10(m, reads, cn=2, pool_fasta=fasta)

    r1 = build_gene_consensus(m, reads, call, pool_fasta=fasta)
    r2 = build_gene_consensus(m, reads, call, pool_fasta=fasta)
    assert r1 == r2

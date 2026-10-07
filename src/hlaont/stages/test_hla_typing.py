"""Unit tests for Stage 4 typing (stages/hla_typing.py)."""
from __future__ import annotations

import gzip
import random

from consensus import HaplotypeConsensus
from hla_typing import TypingParams, type_haplotype


def _random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _substitute(seq: str, pos: int, rng: random.Random) -> str:
    alt = rng.choice([b for b in "ACGT" if b != seq[pos]])
    return seq[:pos] + alt + seq[pos + 1:]


def _build_pool(tmp_path, records: dict[str, str]):
    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        for allele, seq in records.items():
            fh.write(f">{allele}\n{seq}\n")
    meta = tmp_path / "pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        for allele in records:
            fh.write(f"GENE1\t{allele}\n")
    return fasta, meta


def test_exact_match_gets_high_confidence_at_every_field(tmp_path):
    rng = random.Random(5)
    base = _random_seq(600, rng)
    # A field-4-only variant sitting far from base, and an unrelated allele,
    # so the pool has real structure to marginalise over.
    field4_variant = _substitute(base, 550, rng)
    unrelated = _random_seq(600, random.Random(99))

    records = {
        "GENE1*01:01:01:01": base,
        "GENE1*01:01:01:02": field4_variant,
        "GENE1*02:01:01:01": unrelated,
    }
    fasta, meta = _build_pool(tmp_path, records)

    consensus = HaplotypeConsensus(
        gene="GENE1", template_allele="GENE1*01:01:01:01",
        n_reads_assigned=10, sequence=base, qv=tuple([40] * len(base)), variants=(),
    )
    call = type_haplotype(consensus, pool_fasta=fasta, pool_meta=meta, n_threads=2)

    assert call.called_allele == "GENE1*01:01:01:01"
    assert call.n_fields == 4
    # Field 1 should be very confident (only *02:... differs there).
    assert call.field_gq[0] > 20
    # Field 4 confidence should be lower than field 1/2/3 -- the field4
    # variant is a real, single-base-away rival at that field only.
    assert call.field_gq[3] <= call.field_gq[0]


def test_ambiguous_consensus_produces_a_multi_allele_ambiguity_set(tmp_path):
    rng = random.Random(21)
    base = _random_seq(500, rng)
    near_identical = _substitute(base, 10, rng)  # one base different from base
    unrelated = _random_seq(500, random.Random(3))

    records = {
        "GENE1*01:01": base,
        "GENE1*01:02": near_identical,
        "GENE1*02:01": unrelated,
    }
    fasta, meta = _build_pool(tmp_path, records)

    # One base in 500 costs ~4 nats, outside the default 3.0 margin; with a
    # permissive margin the near-identical allele is in the ambiguity set
    # and the distant one is not.
    consensus = HaplotypeConsensus(
        gene="GENE1", template_allele="GENE1*01:01",
        n_reads_assigned=10, sequence=base, qv=tuple([40] * len(base)), variants=(),
    )
    permissive = TypingParams(ambiguity_log_margin=100.0)
    call = type_haplotype(consensus, pool_fasta=fasta, pool_meta=meta, n_threads=2, params=permissive)

    assert call.called_allele == "GENE1*01:01"
    assert "GENE1*01:02" in call.ambiguity_set
    # The distant allele doesn't align at all, so it's never a candidate.
    assert "GENE1*02:01" not in call.posterior
    assert set(call.posterior) == {"GENE1*01:01", "GENE1*01:02"}
    assert abs(sum(call.posterior.values()) - 1.0) < 1e-9


def test_type_haplotype_is_deterministic(tmp_path):
    rng = random.Random(7)
    base = _random_seq(400, rng)
    variant = _substitute(base, 50, rng)
    records = {"GENE1*01:01": base, "GENE1*01:02": variant}
    fasta, meta = _build_pool(tmp_path, records)

    consensus = HaplotypeConsensus(
        gene="GENE1", template_allele="GENE1*01:01",
        n_reads_assigned=8, sequence=base, qv=tuple([30] * len(base)), variants=(),
    )
    c1 = type_haplotype(consensus, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    c2 = type_haplotype(consensus, pool_fasta=fasta, pool_meta=meta, n_threads=2)
    assert c1 == c2

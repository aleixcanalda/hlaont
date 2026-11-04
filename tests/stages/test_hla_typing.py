"""Unit tests for Stage 4 typing (stages/hla_typing.py)."""
from __future__ import annotations

import gzip
import random

from hlaont.stages.consensus import HaplotypeConsensus
from hlaont.stages.hla_typing import TypingParams, type_haplotype


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


def test_empty_dense_realignment_keeps_whole_pool_call(tmp_path, monkeypatch):
    # A consensus with no hit in the dense fallback (seen on an R9 sample)
    # used to raise IndexError; it must fall back to the whole-pool result.
    import numpy as np

    from hlaont.stages import hla_typing
    from hlaont.stages.assign import GeneAssignmentMatrix

    rng = random.Random(7)
    base = _random_seq(600, rng)
    records = {
        "GENE1*01:01:01:01": base,
        "GENE1*02:01:01:01": _random_seq(600, random.Random(11)),
    }
    fasta, meta = _build_pool(tmp_path, records)
    consensus = HaplotypeConsensus(
        gene="GENE1", template_allele="GENE1*01:01:01:01",
        n_reads_assigned=10, sequence=_substitute(base, 300, rng),
        qv=tuple([40] * len(base)), variants=(),
    )

    def empty_realign(gene, reads, allele_sequences, **kwargs):
        ids = tuple(sorted(allele_sequences))
        zeros = np.zeros((0, len(ids)), dtype=np.int32)
        return GeneAssignmentMatrix(gene, (), ids, zeros, zeros.copy(), zeros.copy())

    monkeypatch.setattr(hla_typing, "realign_to_candidates", empty_realign)
    call = type_haplotype(
        consensus, pool_fasta=fasta, pool_meta=meta, n_threads=2,
        extra_candidates=("GENE1*02:01:01:01",),
    )
    assert call.called_allele == "GENE1*01:01:01:01"


# CDS-first naming. A 1500bp gene: CDS exons 101-250 and 601-750 (a
# 100-codon ORF), intron 251-600. *01:02 records differ from *01:01 by one
# nonsynonymous CDS base (codon 34) and carry 6+ intronic differences, so
# the whole-gene fit prefers *01:01 even for a consensus with *01:02's CDS.
_EXONS = ((101, 250), (601, 750))
_INTRON_DIFFS = (300, 350, 400, 450, 500, 550, 260, 280)
_STOPS = {"TAA", "TAG", "TGA"}


def _cds_gene(rng: random.Random) -> str:
    codons = [c for c in (a + b + d for a in "ACGT" for b in "ACGT" for d in "ACGT") if c not in _STOPS]
    orf = ["ATG"] + [rng.choice(codons) for _ in range(98)] + ["TAA"]
    orf[33], orf[40] = "GCT", "GCT"
    cds = "".join(orf)
    seq = list(_random_seq(1500, rng))
    seq[100:250] = cds[:150]
    seq[600:750] = cds[150:]
    return "".join(seq)


def _set(seq: str, pos1: int, bases: str) -> str:
    return seq[:pos1 - 1] + bases + seq[pos1 - 1 + len(bases):]


def _with_intron_diffs(seq: str, n: int) -> str:
    for p in _INTRON_DIFFS[:n]:
        seq = _set(seq, p, "A" if seq[p - 1] != "A" else "C")
    return seq


def _cds_of(seq: str) -> str:
    return "".join(seq[s - 1:e] for s, e in _EXONS)


def _write_models_and_catalogue(tmp_path, records: dict[str, str], genomic: dict[str, bool] | None = None):
    import json

    models = tmp_path / "models.jsonl.gz"
    with gzip.open(models, "wt") as fh:
        for allele in records:
            fh.write(json.dumps({
                "gene": "GENE1", "allele": allele, "cds_exons": [list(e) for e in _EXONS],
                "introns": [[1, 251, 600]], "utrs": [[1, 100], [751, 1500]],
            }) + "\n")
    catalogue = tmp_path / "cds_catalogue.tsv.gz"
    with gzip.open(catalogue, "wt") as fh:
        fh.write("gene\tallele\tgenomic\tpartial\tcds_sequence\n")
        for allele, seq in records.items():
            g = (genomic or {}).get(allele, True)
            fh.write(f"GENE1\t{allele}\t{g}\tFalse\t{_cds_of(seq)}\n")
    return models, catalogue


def _cds_fixture(tmp_path):
    base = _cds_gene(random.Random(13))
    b_cds = _set(base, 200, "TGG")  # codon 34 GCT -> TGG (Ala -> Trp)
    records = {
        "GENE1*01:01:01:01": base,
        "GENE1*01:02:01:01": _with_intron_diffs(b_cds, 6),
        "GENE1*01:02:01:02": _with_intron_diffs(b_cds, 8),
    }
    return base, b_cds, records


def _type(consensus_seq, fasta, meta, models, catalogue):
    consensus = HaplotypeConsensus(
        gene="GENE1", template_allele="GENE1*01:01:01:01", n_reads_assigned=10,
        sequence=consensus_seq, qv=tuple([40] * len(consensus_seq)), variants=(),
    )
    return type_haplotype(
        consensus, pool_fasta=fasta, pool_meta=meta, n_threads=2,
        models_path=models, cds_catalogue_path=catalogue,
    )


def test_cds_first_naming_renames_and_picks_field4_by_whole_gene_fit(tmp_path):
    _base, b_cds, records = _cds_fixture(tmp_path)
    fasta, meta = _build_pool(tmp_path, records)
    models, catalogue = _write_models_and_catalogue(tmp_path, records)

    # Consensus = *01:02's CDS on *01:01's introns: 1 CDS mismatch (weight
    # 5) against *01:01 beats 6 intronic ones against *01:02:01:01.
    call = _type(b_cds, fasta, meta, models, catalogue)
    assert call.called_allele == "GENE1*01:02:01:01"

    # Without a catalogue the whole-gene call stands.
    call = _type(b_cds, fasta, meta, models, tmp_path / "missing.tsv.gz")
    assert call.called_allele == "GENE1*01:01:01:01"


def test_cds_first_naming_falls_back_to_protein_match(tmp_path):
    _base, b_cds, records = _cds_fixture(tmp_path)
    fasta, meta = _build_pool(tmp_path, records)
    models, catalogue = _write_models_and_catalogue(tmp_path, records)

    # Synonymous change at codon 41 (GCT -> GCC): no CDS matches, *01:02's
    # protein does.
    call = _type(_set(b_cds, 223, "C"), fasta, meta, models, catalogue)
    assert call.called_allele == "GENE1*01:02:01:01"


def test_cds_first_naming_ignores_exon_only_records(tmp_path):
    _base, b_cds, records = _cds_fixture(tmp_path)
    fasta, meta = _build_pool(tmp_path, records)
    models, catalogue = _write_models_and_catalogue(
        tmp_path, records, genomic={"GENE1*01:02:01:01": False, "GENE1*01:02:01:02": False},
    )
    call = _type(b_cds, fasta, meta, models, catalogue)
    assert call.called_allele == "GENE1*01:01:01:01"

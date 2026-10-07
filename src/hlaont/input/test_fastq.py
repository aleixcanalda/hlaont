"""Unit tests for the Stage 0 FASTQ anchor sieve (DESIGN.md S3.2)."""
from __future__ import annotations

import gzip
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fastq import SieveParams, SieveStats, sieve_fastq  # noqa: E402

# Small, well-separated synthetic anchor set: one target gene, one decoy
# gene, deliberately different sequences (not real HLA data -- the sieve's
# logic is what's under test, not minimap2's ability to align real alleles).
_RNG = random.Random(0)


def _random_seq(n: int) -> str:
    return "".join(_RNG.choice("ACGT") for _ in range(n))


TARGET_SEQ = _random_seq(1000)
DECOY_SEQ = _random_seq(1000)


def _write_anchors(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "anchors.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">TESTGENE-A*01:01\n{TARGET_SEQ}\n")
        fh.write(f">TESTGENE-B*01:01\n{DECOY_SEQ}\n")

    meta = tmp_path / "anchors.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\trole\tsource\tcluster_size\tlength\n")
        fh.write(f"TESTGENE-A\tTESTGENE-A*01:01\ttarget\ttest\t1\t{len(TARGET_SEQ)}\n")
        fh.write(f"TESTGENE-B\tTESTGENE-B*01:01\tdecoy\ttest\t1\t{len(DECOY_SEQ)}\n")

    return fasta, meta


def _write_fastq(tmp_path: Path, reads: dict[str, str]) -> Path:
    fq = tmp_path / "reads.fq"
    with open(fq, "w") as fh:
        for read_id, seq in reads.items():
            fh.write(f"@{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")
    return fq


def _params(tmp_path: Path, **overrides) -> SieveParams:
    d = dict(min_anchor_coverage=0.4, anchor_edge_distance_bp=200, mismatch_rate_multiplier=10.0)
    d.update(overrides)
    return SieveParams(**d)


def test_read_covering_target_is_kept(tmp_path):
    fasta, meta = _write_anchors(tmp_path)
    # 60% of the target anchor, well clear of both ends -- clears the 40%
    # coverage bar outright, no edge rescue needed.
    read_seq = TARGET_SEQ[200:800]
    fq = _write_fastq(tmp_path, {"r1": read_seq})

    stats = SieveStats()
    reads = list(sieve_fastq(fq, anchors_path=fasta, anchors_meta_path=meta,
                              params=_params(tmp_path), stats=stats))

    assert len(reads) == 1
    assert reads[0].read_id == "r1"
    assert reads[0].hits[0].gene == "TESTGENE-A"
    assert reads[0].hits[0].mismatch_rate == 0.0
    assert stats.n_kept == 1
    assert stats.kept_per_gene == {"TESTGENE-A": 1}


def test_low_coverage_not_near_edge_is_dropped(tmp_path):
    fasta, meta = _write_anchors(tmp_path)
    # Only 20% coverage, and the aligned region sits well inside the anchor
    # (not within 200bp of either end) -- fails both the coverage bar and
    # the edge-rescue clause.
    read_seq = TARGET_SEQ[400:600]
    fq = _write_fastq(tmp_path, {"r1": read_seq})

    stats = SieveStats()
    reads = list(sieve_fastq(fq, anchors_path=fasta, anchors_meta_path=meta,
                              params=_params(tmp_path), stats=stats))

    assert reads == []
    assert stats.n_dropped_no_qualifying_alignment == 1
    assert stats.n_kept == 0


def test_low_coverage_near_edge_is_rescued(tmp_path):
    fasta, meta = _write_anchors(tmp_path)
    # Covers 38% of the anchor but starts at 0 (edge rescue). Must be
    # >= ~350bp: -s350 hides shorter matches.
    read_seq = TARGET_SEQ[0:380]
    fq = _write_fastq(tmp_path, {"r1": read_seq})

    stats = SieveStats()
    reads = list(sieve_fastq(fq, anchors_path=fasta, anchors_meta_path=meta,
                              params=_params(tmp_path), stats=stats))

    assert len(reads) == 1
    assert reads[0].hits[0].gene == "TESTGENE-A"


def test_read_matching_only_decoy_is_dropped_entirely(tmp_path):
    fasta, meta = _write_anchors(tmp_path)
    read_seq = DECOY_SEQ[200:800]
    fq = _write_fastq(tmp_path, {"r1": read_seq})

    stats = SieveStats()
    reads = list(sieve_fastq(fq, anchors_path=fasta, anchors_meta_path=meta,
                              params=_params(tmp_path), stats=stats))

    assert reads == []
    assert stats.n_dropped_decoy_or_score_filter == 1
    assert stats.n_kept == 0
    assert stats.kept_per_gene == {}


def test_stats_accumulate_across_mixed_reads(tmp_path):
    fasta, meta = _write_anchors(tmp_path)
    fq = _write_fastq(tmp_path, {
        "keep1": TARGET_SEQ[200:800],
        "keep2": TARGET_SEQ[100:900],
        "drop_decoy": DECOY_SEQ[200:800],
        "drop_no_hit": "A" * 50,
    })

    stats = SieveStats()
    reads = list(sieve_fastq(fq, anchors_path=fasta, anchors_meta_path=meta,
                              params=_params(tmp_path), stats=stats))

    assert {r.read_id for r in reads} == {"keep1", "keep2"}
    assert stats.n_reads == 4
    assert stats.n_kept == 2
    assert stats.n_dropped_decoy_or_score_filter == 1
    assert stats.n_dropped_no_qualifying_alignment == 1
    assert stats.kept_per_gene == {"TESTGENE-A": 2}


def test_sieve_is_deterministic_across_reruns(tmp_path):
    fasta, meta = _write_anchors(tmp_path)
    fq = _write_fastq(tmp_path, {
        "r1": TARGET_SEQ[200:800],
        "r2": DECOY_SEQ[200:800],
        "r3": TARGET_SEQ[0:150],
    })

    def run():
        stats = SieveStats()
        reads = list(sieve_fastq(fq, anchors_path=fasta, anchors_meta_path=meta,
                                  params=_params(tmp_path), stats=stats))
        return [(r.read_id, r.hits) for r in reads], stats

    result_a, stats_a = run()
    result_b, stats_b = run()

    assert result_a == result_b
    assert stats_a == stats_b


def test_default_params_load_from_resources_models():
    params = SieveParams.load()
    # 0.4 -> 0.1, 2026-08-31/09-01 session -- see stage0_fastq_sieve.yaml's
    # own comment for the real-data rationale.
    assert params.min_anchor_coverage == 0.1
    assert params.anchor_edge_distance_bp == 200
    assert params.mismatch_rate_multiplier == 10.0

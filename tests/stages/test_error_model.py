"""Unit tests for the Badread-derived context error model (DESIGN.md S2e)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from hlaont.stages.error_model import (
    KmerErrorModel,
    build_aligned_strings,
    context_log_likelihood,
    load_kmer_error_model,
    log_prob,
    reverse_complement,
)


def _write_model(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "model.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_kmer_error_model_parses_probs_and_remainder(tmp_path):
    path = _write_model(tmp_path, [
        "AAAAAAA,0.900000;AAAAAAC,0.050000;AAAAAAG,0.030000",
        "CCCCCCC,0.800000;CCCCCCG,0.150000",
    ])
    model = load_kmer_error_model(path)

    assert model.kmer_size == 7
    assert model.probs["AAAAAAA"] == {"AAAAAAA": 0.9, "AAAAAAC": 0.05, "AAAAAAG": 0.03}
    assert model.remainder["AAAAAAA"] == pytest.approx(0.02)
    assert model.remainder["CCCCCCC"] == pytest.approx(0.05)
    assert model.default_remainder == pytest.approx((0.02 + 0.05) / 2)


def test_load_kmer_error_model_rejects_inconsistent_kmer_size(tmp_path):
    path = _write_model(tmp_path, ["AAAAAAA,0.9", "AAAAAA,0.9"])
    with pytest.raises(ValueError, match="inconsistent"):
        load_kmer_error_model(path)


def test_log_prob_self_listed_alt_remainder_and_missing_kmer(tmp_path):
    path = _write_model(tmp_path, [
        "AAAAAAA,0.900000;AAAAAAC,0.050000;AAAAAAG,0.030000",
        "CCCCCCC,0.800000;CCCCCCG,0.150000",
    ])
    model = load_kmer_error_model(path)

    assert log_prob(model, "AAAAAAA", "AAAAAAA") == pytest.approx(math.log(0.9))
    assert log_prob(model, "AAAAAAA", "AAAAAAC") == pytest.approx(math.log(0.05))
    # Unlisted alternative -> the kmer's remainder bucket.
    assert log_prob(model, "AAAAAAA", "TTTTTTT") == pytest.approx(math.log(0.02))
    # Reference kmer never seen in the table at all -> default_remainder.
    assert log_prob(model, "GGGGGGG", "AAAAAAA") == pytest.approx(
        math.log((0.02 + 0.05) / 2)
    )


def test_reverse_complement():
    assert reverse_complement("AACCGT") == "ACGGTT"
    assert reverse_complement("") == ""


def test_expand_cigar_perfect_match():
    aligned_ref, aligned_read = build_aligned_strings(
        "ACGTACG", "ACGTACG",
        ref_start=0, ref_end=7, read_start=0, read_end=7,
        strand="+", cigar="7M",
    )
    assert aligned_ref == "ACGTACG"
    assert aligned_read == "ACGTACG"


def test_expand_cigar_insertion():
    aligned_ref, aligned_read = build_aligned_strings(
        "ACGT", "ACGGT",
        ref_start=0, ref_end=4, read_start=0, read_end=5,
        strand="+", cigar="2M1I2M",
    )
    assert aligned_ref == "AC-GT"
    assert aligned_read == "ACGGT"


def test_expand_cigar_deletion():
    aligned_ref, aligned_read = build_aligned_strings(
        "ACGT", "AGT",
        ref_start=0, ref_end=4, read_start=0, read_end=3,
        strand="+", cigar="1M1D2M",
    )
    assert aligned_ref == "ACGT"
    assert aligned_read == "A-GT"


def test_expand_cigar_reverse_strand_revcomps_read_span():
    # revcomp("CGTACGT") == "ACGTACG" -- a reverse-strand perfect match.
    aligned_ref, aligned_read = build_aligned_strings(
        "ACGTACG", "CGTACGT",
        ref_start=0, ref_end=7, read_start=0, read_end=7,
        strand="-", cigar="7M",
    )
    assert aligned_ref == "ACGTACG"
    assert aligned_read == "ACGTACG"


def test_expand_cigar_rejects_unexpected_op():
    with pytest.raises(ValueError, match="unsupported"):
        build_aligned_strings(
            "ACGT", "ACGT", ref_start=0, ref_end=4, read_start=0, read_end=4,
            strand="+", cigar="4P",
        )


def _synthetic_model() -> KmerErrorModel:
    # Real trained alternatives always share their kmer's first and last
    # base (Badread's own training only records entries meeting that
    # anchor condition) -- "ACA" is a valid middle-base substitution of
    # "AAA", "AAC" (edits the last base) would not be a real entry.
    return KmerErrorModel(
        kmer_size=3,
        probs={"AAA": {"AAA": 0.7, "ACA": 0.2}, "CCC": {"CCC": 0.9}},
        remainder={"AAA": 0.1, "CCC": 0.1},
        default_remainder=0.1,
    )


def test_context_log_likelihood_disjoint_windows_perfect_match():
    model = _synthetic_model()
    ll = context_log_likelihood(model, "AAACCC", "AAACCC")
    assert ll == pytest.approx(math.log(0.7) + math.log(0.9))


def test_context_log_likelihood_uses_listed_alternative():
    model = _synthetic_model()
    ll = context_log_likelihood(model, "AAACCC", "ACACCC")
    assert ll == pytest.approx(math.log(0.2) + math.log(0.9))


def test_context_log_likelihood_anchor_failure_falls_back_to_default_remainder():
    model = _synthetic_model()
    # First window's own anchor bases (A...A) don't match the observed
    # (G...G) -- not a trained alternative, not attributable cleanly.
    ll = context_log_likelihood(model, "AAACCC", "GAGCCC")
    assert ll == pytest.approx(math.log(0.1) + math.log(0.9))


def test_context_log_likelihood_ignores_trailing_partial_window():
    model = _synthetic_model()
    # 4 ref bases with k=3: one full window ("AAA"), one trailing partial
    # ("C") that never reaches k reference bases -- contributes nothing.
    ll = context_log_likelihood(model, "AAAC", "AAAC")
    assert ll == pytest.approx(math.log(0.7))


def test_real_vendored_model_loads_and_scores_known_line():
    model = load_kmer_error_model()
    assert model.kmer_size == 7
    assert len(model.probs) == 16384
    # First line of resources/models/vendor/nanopore2023.gz: self-prob for
    # the homopolymer AAAAAAA.
    ll = context_log_likelihood(model, "AAAAAAA", "AAAAAAA")
    assert ll == pytest.approx(math.log(0.923511), abs=1e-4)

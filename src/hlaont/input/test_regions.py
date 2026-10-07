"""Unit tests for @SQ header region-set derivation (DESIGN.md S3.2)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from regions import derive_regions  # noqa: E402

CHR6_LEN = 170805979
CHM13_CHR6_LEN = 172126628


def test_chr_prefixed_header_finds_window_alts_and_decoys():
    sq = {
        "chr6": CHR6_LEN,
        "chr1": 248956422,
        "chr6_GL000250v2_alt": 4672374,
        "chr6_GL000253v2_alt": 4677643,
        "HLA-A*01:01:01:01": 3503,
        "HLA-DRB1*15:01:01:04": 11056,
        "HLA-DECOYISH": 500,  # no "*" -- must NOT match the decoy pattern
    }
    result = derive_regions(sq)
    assert result.build == "GRCh38"

    kinds = sorted((r.kind, r.contig) for r in result.regions)
    assert kinds == sorted([
        ("mhc_window", "chr6"),
        ("mhc_alt", "chr6_GL000250v2_alt"),
        ("mhc_alt", "chr6_GL000253v2_alt"),
        ("hla_decoy", "HLA-A*01:01:01:01"),
        ("hla_decoy", "HLA-DRB1*15:01:01:04"),
    ])

    window = next(r for r in result.regions if r.kind == "mhc_window")
    assert window.start == 28510119
    assert window.end == 33480577

    alt = next(r for r in result.regions if r.contig == "chr6_GL000250v2_alt")
    assert alt.label == "APD"
    assert alt.start is None and alt.end is None


def test_bare_numeric_naming_convention_is_resolved_via_length():
    sq = {"6": CHR6_LEN, "1": 248956422}
    result = derive_regions(sq)

    assert len(result.regions) == 1
    assert result.regions[0].contig == "6"
    assert result.regions[0].kind == "mhc_window"


def test_wrong_length_chr6_raises_loudly():
    sq = {"chr6": 999, "chr1": 248956422}
    try:
        derive_regions(sq)
    except ValueError as e:
        assert "unrecognized build" in str(e)
        assert "GRCh38" in str(e) and "CHM13v2.0" in str(e)
    else:
        raise AssertionError("expected ValueError for an unrecognized chr6")


def test_chm13_header_finds_window_and_decoys_but_no_alts():
    # CHM13v2.0 has no ALT contigs at all -- this is a real property of the
    # assembly (single-haplotype T2T), not a case this test is skipping.
    sq = {
        "chr6": CHM13_CHR6_LEN,
        "chr1": 248387328,
        "HLA-A*01:01:01:01": 3503,
    }
    result = derive_regions(sq)
    assert result.build == "CHM13v2.0"

    kinds = sorted((r.kind, r.contig) for r in result.regions)
    assert kinds == sorted([
        ("mhc_window", "chr6"),
        ("hla_decoy", "HLA-A*01:01:01:01"),
    ])

    window = next(r for r in result.regions if r.kind == "mhc_window")
    # 2026-08-31 correction: real HG03583 ONT data showed the previous
    # window (1,214,346-4,666,036) recruited essentially nothing -- see
    # mhc_window.chm13v2.bed's own header for the full root-cause story.
    assert window.start == 28385000
    assert window.end == 33500000


def test_chm13_accession_naming_convention_is_resolved():
    sq = {"CP068272.2": CHM13_CHR6_LEN}
    result = derive_regions(sq)
    assert result.build == "CHM13v2.0"
    assert result.regions[0].contig == "CP068272.2"


def test_grch38_header_with_chr17_includes_chr17_control_region():
    # chr17 control region (GRCh38 chr17:72,062,001-76,562,000).
    sq = {"chr6": CHR6_LEN, "chr17": 83257441}
    result = derive_regions(sq)
    assert result.build == "GRCh38"
    chr17_regions = [r for r in result.regions if r.kind == "chr17_control"]
    assert len(chr17_regions) == 1
    assert chr17_regions[0].contig == "chr17"
    assert chr17_regions[0].start == 72062000
    assert chr17_regions[0].end == 76562000


def test_chm13_header_with_chr17_includes_chr17_control_region():
    # User-supplied real coordinates, CHM13v2.0 chr17:72,950,001-77,450,000
    # (1-based) -- the identical interval resources/controls/build.py
    # extracted the shipped chr17_control.chm13v2.fa.gz resource from.
    sq = {"chr6": CHM13_CHR6_LEN, "chr17": 84276897}
    result = derive_regions(sq)
    assert result.build == "CHM13v2.0"
    chr17_regions = [r for r in result.regions if r.kind == "chr17_control"]
    assert len(chr17_regions) == 1
    assert chr17_regions[0].contig == "chr17"
    assert chr17_regions[0].start == 72950000
    assert chr17_regions[0].end == 77450000


def test_no_chr17_in_header_is_not_an_error_and_adds_no_region():
    # A header lacking chr17 entirely (e.g. an already-subset mini-BAM from
    # an earlier session) must not error or silently invent a region --
    # matches DESIGN.md S3.1's "if they exist" stance for ALTs/decoys.
    sq = {"chr6": CHR6_LEN}
    result = derive_regions(sq)
    assert not any(r.kind == "chr17_control" for r in result.regions)


def test_no_alts_or_decoys_is_not_an_error():
    sq = {"chr6": CHR6_LEN}
    result = derive_regions(sq)
    assert len(result.regions) == 1
    assert result.regions[0].kind == "mhc_window"


def test_derivation_is_deterministic_across_reruns():
    sq = {
        "chr6": CHR6_LEN,
        "chr6_GL000255v2_alt": 4606388,
        "HLA-DQB1*02:01:01": 500,
    }
    assert derive_regions(sq) == derive_regions(sq)


def test_default_resources_load_and_identify_real_grch38_naming():
    # Contig names/lengths from the real GRCh38_full_analysis_set_plus_decoy_hla.fa.fai.
    sq = {
        "chr6": CHR6_LEN,
        "chr1": 248956422,
        "chr6_GL000251v2_alt": 4795265,
        "chr6_GL000256v2_alt": 4929269,
        "HLA-A*01:01:01:01": 3503,
        "HLA-DRB1*16:02:01": 11005,
    }
    result = derive_regions(sq)
    kinds = {(r.kind, r.contig) for r in result.regions}
    assert ("mhc_window", "chr6") in kinds
    assert ("mhc_alt", "chr6_GL000251v2_alt") in kinds
    assert ("mhc_alt", "chr6_GL000256v2_alt") in kinds
    assert ("hla_decoy", "HLA-A*01:01:01:01") in kinds
    assert ("hla_decoy", "HLA-DRB1*16:02:01") in kinds

    cox = next(r for r in result.regions if r.contig == "chr6_GL000251v2_alt")
    assert cox.label == "COX"

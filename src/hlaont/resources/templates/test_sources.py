"""Unit tests for GeneModel parsing (sources.py's FT feature-table extension).

Synthetic hla.dat-shaped fixtures, not the real IPD file -- covers the
class-I-like multi-exon join(...) shape, a single-exon CDS (no join()),
and the partial/excluded-gene paths that must still work unchanged.
"""
from __future__ import annotations

from pathlib import Path

from sources import cds_exon_numbers, cds_nucleotide_sequence, parse_ipd_dat

_HEADER = """\
ID   HLA99999; SV 1; standard; DNA; HUM; {length} BP.
XX
DE   {allele}, Human MHC test sequence
XX
CC   IPD-IMGT/HLA Release Version 3.65.0
XX
"""

_MULTI_EXON_FT = """\
FT   source          1..1000
FT                   /organism="Homo sapiens"
FT   CDS             join(301..373,504..600,700..750)
FT                   /codon_start=1
FT                   /gene="HLA-TEST"
FT                   /translation="MTEST"
FT   UTR             1..300
FT   exon            301..373
FT                   /number="1"
FT   intron          374..503
FT                   /number="1"
FT   exon            504..600
FT                   /number="2"
FT   intron          601..699
FT                   /number="2"
FT   exon            700..750
FT                   /number="3"
FT   UTR             751..1000
"""

_SINGLE_EXON_FT = """\
FT   source          1..400
FT   CDS             50..349
FT                   /codon_start=1
FT                   /gene="HLA-TEST"
FT   UTR             1..49
FT   exon            50..349
FT                   /number="1"
FT   UTR             350..400
"""

# Real-shaped partial (exon-subset) submission -- HLA-C*07:457's own IPD
# record (HLA14025) is exactly this: join(<1..73,74..150) covering only
# exons 2-3, no exon 1/introns/UTRs at all, /codon_start offset mid-codon
# from the truncated 5' end, and the literal /partial qualifier that
# _handle_ft_line() keys off of.
_PARTIAL_FT = """\
FT   source          1..150
FT   CDS             join(<1..73,74..150)
FT                   /codon_start=3
FT                   /partial
FT                   /gene="HLA-TEST"
FT   exon            1..73
FT                   /number="2"
FT   exon            74..150
FT                   /number="3"
"""


def _write_record(tmp_path: Path, allele: str, ft: str, seq_len: int) -> Path:
    seq = "ACGT" * (seq_len // 4) + "A" * (seq_len % 4)
    body = (
        _HEADER.format(length=seq_len, allele=allele)
        + ft
        + "SQ   Sequence\n"
        + seq
        + "\n//\n"
    )
    path = tmp_path / "hla.dat"
    path.write_text(body)
    return path


def test_multi_exon_join_cds_parsed(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "HLA-TEST*01:01", _MULTI_EXON_FT, 1000)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=True))
    assert len(recs) == 1
    gm = recs[0].gene_model
    assert gm is not None
    assert gm.cds_exons == ((301, 373), (504, 600), (700, 750))
    assert gm.codon_start == 1
    assert gm.utrs == ((1, 300), (751, 1000))
    assert gm.exons == ((1, 301, 373), (2, 504, 600), (3, 700, 750))
    assert gm.introns == ((1, 374, 503), (2, 601, 699))


def test_single_exon_cds_no_join(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "HLA-TEST*01:02", _SINGLE_EXON_FT, 400)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=False))
    assert len(recs) == 1
    gm = recs[0].gene_model
    assert gm is not None
    assert gm.cds_exons == ((50, 349),)
    assert gm.exons == ((1, 50, 349),)
    assert gm.introns == ()
    assert gm.utrs == ((1, 49), (350, 400))


def test_cds_nucleotide_sequence_concatenates_exons_in_order(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "HLA-TEST*01:01", _MULTI_EXON_FT, 1000)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=True))
    rec = recs[0]
    gm = rec.gene_model
    assert gm is not None
    cds = cds_nucleotide_sequence(rec.sequence, gm)
    expected = rec.sequence[300:373] + rec.sequence[503:600] + rec.sequence[699:750]
    assert cds == expected
    assert len(cds) == (373 - 300) + (600 - 503) + (750 - 699)


def test_require_genomic_false_still_parses_gene_model(tmp_path: Path) -> None:
    # Single-exon record has no intron, so it fails the require_genomic=True
    # bar (DESIGN.md S2b) but must still parse cleanly for the anchor/decoy
    # path (require_genomic=False), gene model included.
    path = _write_record(tmp_path, "HLA-TEST*01:02", _SINGLE_EXON_FT, 400)
    assert list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=True)) == []
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=False))
    assert len(recs) == 1
    assert recs[0].gene_model is not None


def test_genomic_record_partial_flag_is_false(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "HLA-TEST*01:01", _MULTI_EXON_FT, 1000)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=True))
    assert recs[0].partial is False


def test_partial_record_dropped_by_default(tmp_path: Path) -> None:
    # drop_partial defaults True (real template/anchor pool use) -- a
    # genuinely truncated CDS (HLA-C*07:457's own shape) must never enter
    # either pool, regardless of require_genomic.
    path = _write_record(tmp_path, "HLA-TEST*07:457", _PARTIAL_FT, 150)
    assert list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=False)) == []


def test_partial_record_kept_with_drop_partial_false(tmp_path: Path) -> None:
    # Only resources/templates/build.py's CDS-catalogue pass sets this.
    path = _write_record(tmp_path, "HLA-TEST*07:457", _PARTIAL_FT, 150)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=False, drop_partial=False))
    assert len(recs) == 1
    rec = recs[0]
    assert rec.partial is True
    assert rec.gene_model is not None
    assert rec.gene_model.cds_exons == ((1, 73), (74, 150))
    assert rec.gene_model.exons == ((2, 1, 73), (3, 74, 150))


def test_cds_exon_numbers_resolves_from_numbered_exons(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "HLA-TEST*01:01", _MULTI_EXON_FT, 1000)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=True))
    gm = recs[0].gene_model
    assert gm is not None
    assert cds_exon_numbers(gm) == [1, 2, 3]


def test_cds_exon_numbers_partial_record(tmp_path: Path) -> None:
    path = _write_record(tmp_path, "HLA-TEST*07:457", _PARTIAL_FT, 150)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=False, drop_partial=False))
    gm = recs[0].gene_model
    assert gm is not None
    assert cds_exon_numbers(gm) == [2, 3]


def test_cds_exon_numbers_none_when_unresolvable(tmp_path: Path) -> None:
    # An exon feature present with no /number qualifier -- must not guess.
    ft = _SINGLE_EXON_FT.replace('                   /number="1"\n', "")
    path = _write_record(tmp_path, "HLA-TEST*01:02", ft, 400)
    recs = list(parse_ipd_dat(path, {"HLA-TEST"}, require_genomic=False))
    gm = recs[0].gene_model
    assert gm is not None
    assert cds_exon_numbers(gm) is None

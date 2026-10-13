"""Unit tests for Stage 0 mini-BAM extraction (DESIGN.md S3.1/S4 Stage 0)."""
from __future__ import annotations

from pathlib import Path

import pysam

from hlaont.input.bam import extract_mini_bam, mini_bam_to_fastq
from hlaont.input.regions import Region, RegionSet

_CONTIGS = {"chr6": 1000, "chr6_alt": 500, "HLA-DECOY": 300, "chr1": 1000}

_REGION_SET = RegionSet(
    build="TEST",
    regions=(
        Region(contig="chr6", start=50, end=200, kind="mhc_window", label="chr6"),
        Region(contig="chr6_alt", start=None, end=None, kind="mhc_alt", label="ALT"),
        Region(contig="HLA-DECOY", start=None, end=None, kind="hla_decoy", label="HLA-DECOY"),
    ),
)


def _write_reference(tmp_path: Path) -> Path:
    fasta = tmp_path / "ref.fa"
    with open(fasta, "w") as fh:
        for name, length in _CONTIGS.items():
            fh.write(f">{name}\n{'N' * length}\n")
    return fasta


def _make_read(
    name: str,
    *,
    contig: str | None = None,
    pos: int = 0,
    length: int = 40,
    unmapped: bool = False,
    supplementary: bool = False,
    header: pysam.AlignmentHeader,
) -> pysam.AlignedSegment:
    a = pysam.AlignedSegment(header=header)
    a.query_name = name
    a.query_sequence = "ACGT" * (length // 4)
    a.query_qualities = pysam.qualitystring_to_array("I" * length)
    if unmapped:
        a.flag = 4
        a.reference_id = -1
        a.reference_start = -1
    else:
        assert contig is not None
        a.flag = 0x800 if supplementary else 0
        a.reference_id = list(_CONTIGS).index(contig)
        a.reference_start = pos
        a.mapping_quality = 60
        a.cigartuples = [(0, length)]
    return a


def _write_source_bam(tmp_path: Path, reads: list[pysam.AlignedSegment]) -> Path:
    # Coordinate-sorted, unmapped reads last, matching real BAM/CRAM
    # ordering -- pysam.index requires it.
    ordered = sorted(
        reads,
        key=lambda r: (r.reference_id if r.reference_id >= 0 else len(_CONTIGS), r.reference_start),
    )
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    bam_path = tmp_path / "source.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bf:
        for r in ordered:
            bf.write(r)
    pysam.index(str(bam_path))
    return bam_path


def _read_names(bam_path: Path) -> set[str]:
    with pysam.AlignmentFile(str(bam_path), "rb") as bf:
        names = [r.query_name for r in bf]
    assert all(n is not None for n in names)
    return set(names)  # type: ignore[arg-type]


def test_extraction_keeps_window_alt_decoy_and_unmapped_reads(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    reads = [
        _make_read("r_in_window", contig="chr6", pos=100, header=header),
        _make_read("r_outside_window", contig="chr6", pos=500, header=header),
        _make_read("r_alt", contig="chr6_alt", pos=50, header=header),
        _make_read("r_decoy", contig="HLA-DECOY", pos=10, header=header),
        _make_read("r_unrelated", contig="chr1", pos=100, header=header),
        _make_read("r_unmapped", unmapped=True, header=header),
    ]
    source_bam = _write_source_bam(tmp_path, reads)
    reference = _write_reference(tmp_path)
    output_bam = tmp_path / "mini.bam"

    extract_mini_bam(source_bam, reference, _REGION_SET, output_bam)

    assert (tmp_path / "mini.bam.bai").exists()
    assert _read_names(output_bam) == {"r_in_window", "r_alt", "r_decoy", "r_unmapped"}


def test_extraction_works_without_a_reference(tmp_path):
    """2026-08-29: reference_path=None is valid for a plain BAM (its
    records already carry their own sequences, unlike CRAM) -- extraction
    must produce the identical read set as with a real reference."""
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    reads = [
        _make_read("r_in_window", contig="chr6", pos=100, header=header),
        _make_read("r_outside_window", contig="chr6", pos=500, header=header),
        _make_read("r_unmapped", unmapped=True, header=header),
    ]
    source_bam = _write_source_bam(tmp_path, reads)
    output_bam = tmp_path / "mini_no_ref.bam"

    extract_mini_bam(source_bam, None, _REGION_SET, output_bam)

    assert (tmp_path / "mini_no_ref.bam.bai").exists()
    assert _read_names(output_bam) == {"r_in_window", "r_unmapped"}


def test_extraction_is_deterministic_across_reruns(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    reads = [
        _make_read("r_in_window", contig="chr6", pos=100, header=header),
        _make_read("r_unmapped", unmapped=True, header=header),
    ]
    source_bam = _write_source_bam(tmp_path, reads)
    reference = _write_reference(tmp_path)

    out_a = tmp_path / "a.bam"
    out_b = tmp_path / "b.bam"
    extract_mini_bam(source_bam, reference, _REGION_SET, out_a)
    extract_mini_bam(source_bam, reference, _REGION_SET, out_b)

    assert _read_names(out_a) == _read_names(out_b) == {"r_in_window", "r_unmapped"}


def test_fastq_conversion_drops_supplementary_duplicate_record(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    reads = [
        _make_read("r_chimeric", contig="chr6", pos=100, length=40, header=header),
        _make_read(
            "r_chimeric", contig="chr6", pos=140, length=20,
            supplementary=True, header=header,
        ),
        _make_read("r_plain", contig="chr6", pos=100, length=40, header=header),
    ]
    mini_bam = _write_source_bam(tmp_path, reads)
    fastq_path = tmp_path / "out.fastq"

    mini_bam_to_fastq(mini_bam, fastq_path)

    with pysam.FastxFile(str(fastq_path)) as fq:
        entries = {entry.name: entry.sequence for entry in fq}

    assert set(entries) == {"r_chimeric", "r_plain"}
    assert len(entries["r_chimeric"]) == 40  # the primary record's full length, not the 20bp supplementary

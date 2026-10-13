"""Unit tests for HP/PS phase-tag extraction (DESIGN.md S10.2)."""
from __future__ import annotations

from pathlib import Path

import pysam

from hlaont.input.phasing import ReadPhase, load_read_phase_tags

_CONTIGS = {"chr6": 1000}


def _make_read(
    name: str, *, pos: int = 0, length: int = 40,
    hp: int | None = None, ps: int | None = None,
    supplementary: bool = False, header: pysam.AlignmentHeader,
) -> pysam.AlignedSegment:
    a = pysam.AlignedSegment(header=header)
    a.query_name = name
    a.query_sequence = "ACGT" * (length // 4)
    a.query_qualities = pysam.qualitystring_to_array("I" * length)
    a.flag = 0x800 if supplementary else 0
    a.reference_id = 0
    a.reference_start = pos
    a.mapping_quality = 60
    a.cigartuples = [(0, length)]
    tags = []
    if hp is not None:
        tags.append(("HP", hp))
    if ps is not None:
        tags.append(("PS", ps))
    if tags:
        a.set_tags(tags)
    return a


def _write_bam(tmp_path: Path, reads: list[pysam.AlignedSegment]) -> Path:
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    bam_path = tmp_path / "mini.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bf:
        for r in sorted(reads, key=lambda r: r.reference_start):
            bf.write(r)
    return bam_path


def test_reads_with_hp_and_ps_tags_are_captured(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    bam = _write_bam(tmp_path, [
        _make_read("r1", pos=10, hp=1, ps=100, header=header),
        _make_read("r2", pos=20, hp=2, ps=100, header=header),
    ])

    out = load_read_phase_tags(bam)

    assert out == {
        "r1": ReadPhase(hp=1, ps=100),
        "r2": ReadPhase(hp=2, ps=100),
    }


def test_untagged_read_is_recorded_not_omitted(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    bam = _write_bam(tmp_path, [_make_read("r_untagged", pos=10, header=header)])

    out = load_read_phase_tags(bam)

    assert out == {"r_untagged": ReadPhase(hp=None, ps=None)}


def test_supplementary_record_does_not_overwrite_primary(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    bam = _write_bam(tmp_path, [
        _make_read("r_chimeric", pos=10, hp=1, ps=100, header=header),
        _make_read("r_chimeric", pos=200, supplementary=True, header=header),
    ])

    out = load_read_phase_tags(bam)

    assert out == {"r_chimeric": ReadPhase(hp=1, ps=100)}


def test_deterministic_across_reruns(tmp_path):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": n, "LN": ln} for n, ln in _CONTIGS.items()]}
    )
    bam = _write_bam(tmp_path, [
        _make_read("r1", pos=10, hp=1, ps=100, header=header),
        _make_read("r2", pos=20, hp=2, ps=100, header=header),
        _make_read("r3", pos=30, header=header),
    ])

    assert load_read_phase_tags(bam) == load_read_phase_tags(bam)

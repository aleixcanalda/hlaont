"""HP/PS phase-tag extraction from a mini-BAM (DESIGN.md S10.2). Reads
existing tags only; routing is done in pipeline.run_sample_phased()."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pysam


@dataclass(frozen=True)
class ReadPhase:
    hp: int | None
    ps: int | None


def load_read_phase_tags(mini_bam_path: Path) -> dict[str, ReadPhase]:
    """read name -> ReadPhase for every primary record; untagged reads get
    ReadPhase(None, None). Secondary/supplementary records are skipped,
    matching mini_bam_to_fastq()."""
    out: dict[str, ReadPhase] = {}
    with pysam.AlignmentFile(str(mini_bam_path), "rb") as bf:
        for read in bf:
            if read.is_secondary or read.is_supplementary:
                continue
            name = read.query_name
            if name is None:
                continue
            hp = int(read.get_tag("HP")) if read.has_tag("HP") else None
            ps = int(read.get_tag("PS")) if read.has_tag("PS") else None
            out[name] = ReadPhase(hp=hp, ps=ps)
    return out

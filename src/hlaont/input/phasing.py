"""HP/PS phase-tag extraction from a mini-BAM (DESIGN.md S10.2).

Pure tag extraction, nothing else: this module does not decide which genes
get split by haplotype (that needs the anchor sieve's gene assignment too,
stages/pipeline.py's run_sample_phased()) and it does not itself do any
phasing -- it assumes standard SAM `HP:i`/`PS:i` tags already exist on the
input mini-BAM, however they were produced upstream (WhatsHap, longphase,
Dorado's own phasing, ...), per DESIGN.md S10.8.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pysam


@dataclass(frozen=True)
class ReadPhase:
    hp: int | None
    ps: int | None


def load_read_phase_tags(mini_bam_path: Path) -> dict[str, ReadPhase]:
    """One entry per distinct read name in the mini-BAM.

    A read with no `HP` tag still gets an entry (`ReadPhase(None, None)`) --
    absence is recorded, not omitted, so a caller can distinguish "this read
    was in the BAM but untagged" (phasing dropout) from "this read never
    made it into the BAM at all" (development notes: loud, not silent).

    Secondary/supplementary records are skipped: `input/bam.py`'s
    `mini_bam_to_fastq()` already excludes them (`-F 0x900`) so the one
    FASTQ record downstream code ever sees for a read is its primary
    alignment's -- reading the primary record's own tags here (not letting a
    later secondary/supplementary record for the same read name silently
    overwrite it) keeps this map consistent with that.
    """
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

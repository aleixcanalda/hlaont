"""Stage 0 FASTQ anchor sieve (DESIGN.md S3.2).

Aligns reads against the anchor set (resources/anchors/) and keeps reads
with a qualifying hit to a target gene; decoy anchors reject reads that
better match a paralogue. Logic ported from FuFiHLA's raw_reads_filter.py;
thresholds live in resources/models/stage0_fastq_sieve.yaml (invariant 11:
not re-derived for ONT). minimap2 uses FuFiHLA's scoring string verbatim
(milestone-1 parity):

    -e10000 -k19 -w19 -g10k -A1 -B4 -O6,26 -E2,1 -s350 -c --end-bonus=10

plus -N _BEST_N so every anchor hit, target or decoy, is reported.
"""
from __future__ import annotations

import csv
import gzip
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pysam
import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

DEFAULT_ANCHORS = REPO_ROOT / "resources" / "anchors" / "anchors.fa.gz"
DEFAULT_ANCHORS_META = REPO_ROOT / "resources" / "anchors" / "anchors.meta.tsv.gz"
DEFAULT_PARAMS = REPO_ROOT / "resources" / "models" / "stage0_fastq_sieve.yaml"

# >= the anchor count (~190), so every hit is reported.
_BEST_N = 256

# FuFiHLA step1 scoring string (-t/-N added separately).
_MINIMAP2_SCORING_ARGS = (
    "-e10000", "-k19", "-w19", "-g10k",
    "-A1", "-B4", "-O6,26", "-E2,1", "-s350",
    "-c", "--end-bonus=10",
)


@dataclass(frozen=True)
class SieveParams:
    min_anchor_coverage: float
    anchor_edge_distance_bp: int
    mismatch_rate_multiplier: float

    @classmethod
    def load(cls, path: Path = DEFAULT_PARAMS) -> "SieveParams":
        d = yaml.safe_load(path.read_text())
        return cls(
            min_anchor_coverage=float(d["min_anchor_coverage"]),
            anchor_edge_distance_bp=int(d["anchor_edge_distance_bp"]),
            mismatch_rate_multiplier=float(d["mismatch_rate_multiplier"]),
        )


@dataclass(frozen=True)
class AnchorHit:
    gene: str
    allele: str
    coverage: float
    mismatch_rate: float


@dataclass(frozen=True)
class CandidateRead:
    read_id: str
    sequence: str
    quality: str | None
    hits: tuple[AnchorHit, ...]  # qualifying target-gene hits only


@dataclass
class SieveStats:
    n_reads: int = 0
    n_kept: int = 0
    n_dropped_no_qualifying_alignment: int = 0
    n_dropped_decoy_or_score_filter: int = 0
    kept_per_gene: dict[str, int] = field(default_factory=dict)


def _load_allele_role(meta_path: Path) -> dict[str, tuple[str, str]]:
    """allele -> (gene, role), from anchors.meta.tsv.gz -- the single source
    of role assignment (resources/anchors/build.py), not recomputed here."""
    out: dict[str, tuple[str, str]] = {}
    with gzip.open(meta_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["allele"]] = (row["gene"], row["role"])
    return out


def _run_minimap2(
    anchors_path: Path, fastq_path: Path, n_threads: int
) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Run minimap2 (anchors as index, reads as query) and group hits by
    read: read_id -> [(anchor, anchor_len, a_start, a_end, nm)]."""
    proc = subprocess.run(
        [
            "minimap2",
            "-t", str(n_threads),
            *_MINIMAP2_SCORING_ARGS,
            "-N", str(_BEST_N),
            str(anchors_path),
            str(fastq_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    hits_by_read: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        read_id, ctg, ctg_len, r_st, r_en = (
            fields[0], fields[5], int(fields[6]), int(fields[7]), int(fields[8])
        )
        nm = next((int(f[5:]) for f in fields[12:] if f.startswith("NM:i:")), None)
        if nm is None:
            raise ValueError(f"minimap2 PAF line missing NM tag: {line!r}")
        hits_by_read.setdefault(read_id, []).append((ctg, ctg_len, r_st, r_en, nm))
    return hits_by_read


def sieve_fastq(
    fastq_path: Path,
    *,
    anchors_path: Path = DEFAULT_ANCHORS,
    anchors_meta_path: Path = DEFAULT_ANCHORS_META,
    params: SieveParams | None = None,
    stats: SieveStats | None = None,
    n_threads: int = 4,
) -> Iterator[CandidateRead]:
    """Yield CandidateReads passing the anchor sieve. `stats` is filled in
    as the iterator is consumed."""
    if params is None:
        params = SieveParams.load()
    if stats is None:
        stats = SieveStats()

    allele_role = _load_allele_role(anchors_meta_path)
    hits_by_read = _run_minimap2(anchors_path, fastq_path, n_threads)

    with pysam.FastxFile(str(fastq_path)) as fq:
        for entry in fq:
            stats.n_reads += 1
            if entry.name is None or entry.sequence is None:
                raise ValueError(f"malformed FASTQ record in {fastq_path}")
            read_id: str = entry.name
            sequence: str = entry.sequence

            # (gene, role, allele, mismatch_rate, coverage) for every hit
            # that survives the coverage-or-edge-proximity test.
            qualifying: list[tuple[str, str, str, float, float]] = []
            for ctg, ctg_len, r_st, r_en, nm in hits_by_read.get(read_id, []):
                match_len = r_en - r_st
                if match_len <= 0:
                    continue
                coverage = match_len / ctg_len
                # -s350 means hits < ~350bp are never reported, so this
                # rescue only applies to edge hits above that.
                near_edge = (
                    r_st <= params.anchor_edge_distance_bp
                    or r_en >= ctg_len - params.anchor_edge_distance_bp
                )
                if coverage < params.min_anchor_coverage and not near_edge:
                    continue
                if ctg not in allele_role:
                    raise ValueError(
                        f"anchor {ctg!r} (hit by read {read_id!r}) has no "
                        f"entry in {anchors_meta_path} -- anchors.fa.gz and "
                        f"anchors.meta.tsv.gz have drifted apart"
                    )
                gene, role = allele_role[ctg]
                mismatch_rate = nm / match_len
                qualifying.append((gene, role, ctg, mismatch_rate, coverage))

            if not qualifying:
                stats.n_dropped_no_qualifying_alignment += 1
                continue

            # Drop reads with no target hit within the multiplier of their
            # best (lowest) mismatch rate over all anchors, decoys included.
            best_score = min(score for _, _, _, score, _ in qualifying)
            threshold = params.mismatch_rate_multiplier * best_score
            target_hits = tuple(
                AnchorHit(gene=gene, allele=allele, coverage=coverage, mismatch_rate=score)
                for gene, role, allele, score, coverage in qualifying
                if role == "target" and score <= threshold
            )

            if not target_hits:
                stats.n_dropped_decoy_or_score_filter += 1
                continue

            stats.n_kept += 1
            for h in target_hits:
                stats.kept_per_gene[h.gene] = stats.kept_per_gene.get(h.gene, 0) + 1

            yield CandidateRead(
                read_id=read_id,
                sequence=sequence,
                quality=entry.quality,
                hits=target_hits,
            )

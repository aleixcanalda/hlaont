"""Stage 0 FASTQ anchor sieve (DESIGN.md S3.2).

Portability/validation path: HPRC/GIAB fixtures and any future amplicon data
where only FASTQ is available, not the primary UKB CRAM path (S3.1, not yet
built). Maps reads against the rebuilt anchor set (resources/anchors/) and
keeps only reads with a qualifying hit to a *target*-role gene, using decoy
anchors to reject reads that better match a paralog.

Logic ported from FuFiHLA's share/fufihla/src/raw_reads_filter.py (verified
against github.com/jingqing-hu/FuFiHLA -- both its HiFi and "nano" pipelines
call this same script unchanged). The three thresholds are shipped as a
versioned params file, not hardcoded (resources/models/stage0_fastq_sieve.yaml)
-- they are provisional ports, not re-derived for ONT R10 (invariant 11).

Alignment shells out to the minimap2 CLI (not mappy) using FuFiHLA's own
scoring string, ported verbatim from share/fufihla/scripts/FuFiHLA.sh and
FuFiHLA_nano.sh step1 -- identical in both, confirmed 2026-07-21 against
github.com/jingqing-hu/FuFiHLA, so there is no platform-specific tuning being
discarded by porting it as-is:

    -e10000 -k19 -w19 -g10k -A1 -B4 -O6,26 -E2,1 -s350 -c --end-bonus=10

mappy's map-ont preset is minimap2's own maintained default, not this string.
Milestone 1 is reproducing FuFiHLA's own numbers, so on the one axis that is
trivial to keep byte-identical -- alignment scoring -- there is no reason to
diverge from the tool being validated against. -N is raised from minimap2's
default (5) to _BEST_N below: that is not part of the ported scoring string,
it is a search-completeness knob the decoy-rejection logic (below) needs to
see every anchor a read hits, target or decoy, not an arbitrary top-N cap.
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

# Search-completeness knob, not a calibrated biological threshold: the anchor
# set is only ~170 sequences, so scoring every anchor per read is cheap and
# -N only needs to be >= that count to guarantee we see every hit a read
# makes (including decoys) rather than an arbitrary top-N by chaining score.
_BEST_N = 256

# FuFiHLA's step1 scoring string (FuFiHLA.sh / FuFiHLA_nano.sh), ported
# verbatim -- see module docstring. -t/-N are ours, not theirs: a resource
# knob and a search-completeness knob respectively, not part of the score.
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
    """Run minimap2 anchors-as-index/fastq-as-query, grouped by read id.

    Returns read_id -> list of (ctg, ctg_len, r_st, r_en, nm): ctg/ctg_len
    are the anchor name/length (PAF cols 6/7) and r_st/r_en are anchor-space
    coordinates (PAF cols 8/9), since anchors are the index/target and the
    FASTQ is the query -- matching FuFiHLA's own `minimap2 TEMPLATE READS`
    argument order in FuFiHLA.sh/FuFiHLA_nano.sh step1.

    Not a streaming call: minimap2 runs to completion before any hits are
    available. Acceptable here -- this path handles HPRC/amplicon-scale
    FASTQs (S3.2 docstring), never the 500K-sample UKB CRAM path.
    """
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
    """Stream CandidateReads from a FASTQ passing the anchor sieve (S3.2).

    stats, if given, is mutated in place as reads are consumed -- read it
    only after the iterator is exhausted. n_threads is a resource knob for
    the minimap2 subprocess (see _run_minimap2), independent of the ported
    scoring string.
    """
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
                # NB: -s350 in the ported scoring string means minimap2 never
                # reports a hit with match_len below ~350bp in the first
                # place (score is 1/matched base at -A1) -- this edge-rescue
                # clause is only reachable for matches that already clear
                # that floor, not for arbitrarily short edge overlaps.
                # Verified 2026-07-21: a 150bp perfect match against a
                # 1000bp anchor produces zero PAF lines under this string.
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

            # Best (lowest) mismatch rate across ALL anchors, target or decoy: a
            # read whose closest match is a paralog/decoy, with no target hit
            # within the multiplier of that, is dropped entirely (DESIGN S3.2).
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

"""`@SQ` header introspection -- MHC region-set derivation (DESIGN.md S3.1).

The tool derives its own extraction region set from a BAM/CRAM header
instead of asking the user which genome build/decoy set they have
(invariant 2). Given a header's contig names/lengths, this module identifies
which known primary-assembly build the header belongs to (by contig name AND
length together -- see DEFAULT_BUILDS) and returns the union of:

  - The MHC window on that build's primary assembly (resources/regions/
    mhc_window.<build>.bed), matched to whichever contig naming convention
    is actually present in the header.
  - Any of that build's known MHC ALT contigs present (resources/regions/
    alt_contigs.<build>.yaml). GRCh38 has 7; CHM13v2.0 has none at all (a
    real structural difference -- it is a single-haplotype T2T assembly,
    not a gap in this code).
  - Any HLA decoy contigs present, detected by name pattern (~525 in the
    1000G/UKB-style GRCh38 decoy reference, not enumerated -- see the
    GRCh38 YAML's header comment for why; the same pattern is checked
    regardless of which build matched, since decoy-contig presence depends
    on the exact reference FASTA used, not the primary-assembly build).
  - **Added 2026-09-01, user-directed:** the chr17 control region
    `stages/chr17_control.py`/`stages/c4_chr17_cn.py` normalise CN against
    (locityper's own published segdup-free background interval), one per
    build, user-supplied coordinates: CHM13v2.0 chr17:72,950,001-77,450,000
    (1-based -- the identical real interval `resources/controls/build.py`
    extracted the shipped `chr17_control.chm13v2.fa.gz` resource from) and
    GRCh38 chr17:72,062,001-76,562,000 (1-based). Both are alignment
    *targets*, not coordinate systems, for the mechanism that reads this
    region: `stages/chr17_control.py`'s own alignment pass takes real raw
    read sequence and realigns it against the shipped, CHM13-derived
    `chr17_control.chm13v2.fa.gz` reference regardless of which build the
    input BAM was originally aligned to -- so a GRCh38-aligned `--bam`
    sample only needs its own chr17 reads *extracted* using GRCh38's own
    real coordinates here; the control measurement itself is build-
    independent. Without this region, a `--bam`-derived FASTQ (whose
    mini-BAM extraction previously only pulled the MHC window +
    ALT/decoy/unmapped reads) never carried any chr17 signal at all,
    silently forcing every gene's CN to the loud-None "no control signal"
    path (`stages/cn.py`'s own invariant-4 guard) for every `--bam` user --
    a real, previously undiagnosed gap this closes. Matched by name only
    (not name+length like the MHC window's own primary contig, invariant
    2's usual convention) -- user-supplied coordinates, not independently
    re-derived against a real reference's own `.fai` this session.

Scope boundary: this derives *which regions to extract*, nothing more.
Turning a RegionSet into an actual mini-BAM (Stage 0), handling unmapped
reads whose mate maps into one of these regions, and the S3.3 minimizer
sweep over fully-unmapped reads are separate, not-yet-built pieces that
consume this module's output -- not built here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import pysam

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
_REGIONS_DIR = REPO_ROOT / "resources" / "regions"


@dataclass(frozen=True)
class BuildConfig:
    name: str
    mhc_window_path: Path
    alt_contigs_path: Path
    # (contig_name, start, end) 0-based half-open, chr17 control region
    # (module docstring) -- None if this build has no known interval yet.
    chr17_control_region: tuple[str, int, int] | None = None


# User-supplied real coordinates, 2026-09-01 (1-based -> converted to
# 0-based half-open here): CHM13v2.0 chr17:72,950,001-77,450,000, GRCh38
# chr17:72,062,001-76,562,000 -- module docstring has the full rationale.
_CHM13V2_CHR17_CONTROL = ("chr17", 72_950_000, 77_450_000)
_GRCH38_CHR17_CONTROL = ("chr17", 72_062_000, 76_562_000)

# Tried in order; the first build whose primary contig is identified in a
# given header wins (see derive_regions). Order has no effect on correctness
# since a header can only match one build's primary contig (name+length
# together), but is fixed for determinism (invariant 9).
DEFAULT_BUILDS: tuple[BuildConfig, ...] = (
    BuildConfig(
        name="GRCh38",
        mhc_window_path=_REGIONS_DIR / "mhc_window.grch38.bed",
        alt_contigs_path=_REGIONS_DIR / "alt_contigs.grch38.yaml",
        chr17_control_region=_GRCH38_CHR17_CONTROL,
    ),
    BuildConfig(
        name="CHM13v2.0",
        mhc_window_path=_REGIONS_DIR / "mhc_window.chm13v2.bed",
        alt_contigs_path=_REGIONS_DIR / "alt_contigs.chm13v2.yaml",
        chr17_control_region=_CHM13V2_CHR17_CONTROL,
    ),
)


@dataclass(frozen=True)
class Region:
    contig: str  # exact @SQ SN name as present in the given header
    start: int | None  # 0-based; None means whole contig
    end: int | None  # None means whole contig
    kind: str  # "mhc_window" | "mhc_alt" | "hla_decoy" | "chr17_control"
    label: str  # "chr6" / ALT haplotype name / the decoy contig's own name


@dataclass(frozen=True)
class RegionSet:
    build: str  # the DEFAULT_BUILDS entry that matched, e.g. "GRCh38"
    regions: tuple[Region, ...]


def _load_mhc_window(path: Path) -> tuple[str, int, int]:
    """Return (bed_chrom, start, end) from the single-interval BED, skipping
    `#`-comment lines."""
    lines = [
        line for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError(f"{path} must contain exactly one interval, found {len(lines)}")
    chrom, start, end = lines[0].split("\t")[:3]
    return chrom, int(start), int(end)


def _load_alt_contigs(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def derive_regions(
    sq_lengths: dict[str, int],
    *,
    builds: tuple[BuildConfig, ...] = DEFAULT_BUILDS,
) -> RegionSet:
    """Derive the MHC extraction region set from a header's {contig: length}.

    Raises ValueError if no known build's primary contig can be identified
    in sq_lengths at all (invariant 2: an unrecognized build is a loud
    error, never a silent empty region). It is NOT an error for a matched
    build to contribute zero ALT contigs or zero HLA decoys -- DESIGN.md
    S3.1: "if they exist" -- most references will lack one or both, and
    CHM13v2.0 never has ALT contigs at all.
    """
    tried: list[str] = []
    for build in builds:
        meta = _load_alt_contigs(build.alt_contigs_path)
        primary = meta["primary_contig"]
        primary_name = next(
            (alias for alias in primary["aliases"]
             if sq_lengths.get(alias) == primary["length"]),
            None,
        )
        if primary_name is None:
            tried.append(build.name)
            continue

        bed_chrom, bed_start, bed_end = _load_mhc_window(build.mhc_window_path)
        regions = [
            Region(
                contig=primary_name, start=bed_start, end=bed_end,
                kind="mhc_window", label=bed_chrom,
            )
        ]

        for alt in meta["alt_contigs"]:
            alias = next((a for a in alt["aliases"] if a in sq_lengths), None)
            if alias is not None:
                regions.append(
                    Region(
                        contig=alias, start=None, end=None,
                        kind="mhc_alt", label=alt["haplotype"],
                    )
                )

        decoy_pattern = re.compile(meta["hla_decoy_pattern"])
        for name in sq_lengths:
            if decoy_pattern.match(name):
                regions.append(
                    Region(contig=name, start=None, end=None, kind="hla_decoy", label=name)
                )

        if build.chr17_control_region is not None:
            chr17_contig, chr17_start, chr17_end = build.chr17_control_region
            if chr17_contig in sq_lengths:
                regions.append(
                    Region(
                        contig=chr17_contig, start=chr17_start, end=chr17_end,
                        kind="chr17_control", label=chr17_contig,
                    )
                )

        regions.sort(key=lambda r: (r.kind, r.contig))
        return RegionSet(build=build.name, regions=tuple(regions))

    raise ValueError(
        f"could not identify a known primary-assembly contig in this "
        f"header -- tried {tried}; unrecognized build"
    )


def derive_regions_from_header(header: "pysam.AlignmentHeader", **kwargs) -> RegionSet:
    """Convenience wrapper for a real pysam.AlignmentHeader."""
    return derive_regions(dict(zip(header.references, header.lengths)), **kwargs)

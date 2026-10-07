"""`@SQ` header introspection -> extraction region set (DESIGN.md S3.1).

Identifies the build by primary contig name and length (invariant 2: never
ask the user) and returns:
- the MHC window (resources/regions/mhc_window.<build>.bed), in the
  header's naming convention;
- the build's MHC ALT contigs present (GRCh38: 7; CHM13v2.0: none);
- HLA decoy contigs present, by name pattern;
- the chr17 control region (CHM13v2.0 chr17:72,950,001-77,450,000; GRCh38
  chr17:72,062,001-76,562,000), so --bam input carries control reads.
  Matched by name only.
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


# chr17 control, 0-based half-open.
_CHM13V2_CHR17_CONTROL = ("chr17", 72_950_000, 77_450_000)
_GRCH38_CHR17_CONTROL = ("chr17", 72_062_000, 76_562_000)

# Tried in order; a header matches at most one build (name + length).
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
    """Region set from a header's {contig: length}. Raises ValueError if no
    known build's primary contig is found (invariant 2). Zero ALTs or decoys
    is fine."""
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

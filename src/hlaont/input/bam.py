"""Stage 0 BAM/CRAM extraction -> mini-BAM (DESIGN.md S3.1/S4 Stage 0).

Primary UKB path: given a RegionSet (input/regions.py -- which coordinates
matter) and a source CRAM/BAM, pull every read in those regions plus every
unmapped read into a small per-sample mini-BAM, stored once as a shared
cohort resource (S4 Stage 0). Everything downstream reads the mini-BAM, not
the source CRAM.

No custom BAM-walking or minimizer-sweep rescue: these are ONT long reads,
not paired-end short reads, so there is no mate to chase for unmapped-read
recovery (S3.1, corrected 2026-07-22 -- the design previously described a
mate-based unmapped handling that does not apply to this data). And every
v0-manifest gene, pseudogenes included, sits inside the extended MHC window
S3.1 already extracts, so there is no known outside-the-window paralog case
needing a bespoke minimizer sweep either (S3.3, same correction). `samtools
view`'s "*" region -- reads with no RNAME -- gets every unmapped read in the
same query as the coordinate regions.

Fine-grained gene/paralog assignment is not done here: mini_bam_to_fastq
converts the extracted reads to FASTQ so they can run through the identical
anchor sieve (input/fastq.py's sieve_fastq) that raw S3.2 FASTQ input uses --
one sieve, two entry points, per S3.3's revision.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from regions import Region, RegionSet

# samtools fastq's default -F excludes secondary (0x100) and supplementary
# (0x800) records, so a read with a chimeric/split alignment contributes
# exactly one full-length sequence, not an extra partial-sequence record
# under the same read name.
_SAMTOOLS_FASTQ_EXCLUDE_FLAGS = 0x900


def _samtools_region(region: Region) -> str:
    """1-based inclusive samtools region string for a Region.

    Region.start/end are 0-based half-open (BED convention, see regions.py);
    None means whole contig, which samtools addresses with the bare contig
    name (no ":start-end" suffix).
    """
    if region.start is None:
        return region.contig
    return f"{region.contig}:{region.start + 1}-{region.end}"


def extract_mini_bam(
    input_path: Path,
    reference_path: Path | None,
    region_set: RegionSet,
    output_path: Path,
    *,
    samtools_bin: str = "samtools",
    n_threads: int = 1,
) -> None:
    """Write the Stage 0 mini-BAM: region_set's regions plus all unmapped
    reads, indexed and ready for repeated downstream reads.

    input_path must already be indexed (.bai/.crai) -- required for the
    region queries below; samtools raises loudly if it is not, which is the
    correct failure mode (a silently-empty extraction would be far worse).

    `reference_path` is `-T` for `samtools view`, needed to decode CRAM (its
    records are reference-compressed) but not a plain BAM (records already
    carry their own sequences) -- optional, 2026-08-29, real-data need: a
    real HG00320 whole-genome ONT BAM with no accompanying reference FASTA
    on this filesystem. Omitted from the command entirely when `None`
    rather than passed as some placeholder path, since samtools' CRAM-only
    use of `-T` means an irrelevant fasta there would be silently ignored
    for BAM input anyway -- omitting it is the honest version of the same
    behaviour, not a functional difference for this input type.

    n_threads is a resource knob (samtools' own "-@") for the view/index
    subprocesses, not a biological parameter -- caller-supplied, never
    hardcoded, per the same reasoning as fastq.py's minimap2 n_threads.
    """
    region_args = [_samtools_region(r) for r in region_set.regions]
    reference_args = ["-T", str(reference_path)] if reference_path is not None else []
    subprocess.run(
        [
            samtools_bin, "view", "-b",
            "-@", str(n_threads),
            *reference_args,
            "-o", str(output_path),
            str(input_path),
            *region_args, "*",
        ],
        check=True,
    )
    subprocess.run(
        [samtools_bin, "index", "-@", str(n_threads), str(output_path)],
        check=True,
    )


def mini_bam_to_fastq(
    mini_bam_path: Path,
    fastq_path: Path,
    *,
    samtools_bin: str = "samtools",
    n_threads: int = 1,
) -> None:
    """Convert a mini-BAM to the FASTQ that feeds sieve_fastq (S3.2/S3.3).

    n_threads is samtools' own "-@" resource knob, caller-supplied (see
    extract_mini_bam).
    """
    with open(fastq_path, "w") as out:
        subprocess.run(
            [
                samtools_bin, "fastq",
                "-@", str(n_threads),
                "-F", str(_SAMTOOLS_FASTQ_EXCLUDE_FLAGS),
                str(mini_bam_path),
            ],
            stdout=out,
            check=True,
        )

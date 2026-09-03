"""Builds the CHM13v2.0 control-region resources for C4A/C4B total copy
number (DESIGN.md S2f), 2026-08-31/09-01 session.

Two files, both shipped/versioned (never derived from a user's own BAM or
reference at runtime -- this is the whole point: a user's own sieved reads
get aligned against these, the same way Stage 0 already aligns reads
against resources/anchors/anchors.fa.gz, so the FASTQ-only path keeps
working with no BAM/reference dependency):

1. chr17_control.chm13v2.fa.gz -- chr17:72,950,001-77,450,000 (1-based),
   user-supplied, matching locityper's own published segdup-free
   background interval used for long-read CNV background normalisation.
   Real sequence, extracted from a real CHM13v2.0 reference (not
   synthesised), used as the depth-normalisation control for C4's own
   estimate: a base-level-coverage measurement here (real aligned bases /
   region length, not a distinct-read count) stands in for "what does 2
   true copies' worth of real depth look like in this sample", replacing
   the calibrator-panel ratio the rest of stages/cn.py still uses (this
   file is deliberately NOT wired into that module -- see the development notes' own
   session log for why extending this to classical genes is still an open
   question, not a decision).

2. c4_flanked_forms.chm13v2.fa.gz -- four sequences, one per (gene x form)
   combination: C4AL, C4AS, C4BL, C4BS. Each is the real long-form gene
   body (C4AL/C4BL, read directly off hla_seq/CHM13_0_HLA.gtf's own real
   `gene` feature coordinates -- NOT resources/anchors/anchors.fa.gz's
   cd-hit-derived long-form representatives, and NOT hla_genes.bed, which
   this same session found to be wrong/offset for the MHC window and was
   never independently re-verified per-gene) or the short-form cd-hit
   anchor (C4A*asm04c6691625 / C4B*asm00bff24228, from
   resources/anchors/anchors.fa.gz -- the short forms have no distinct
   CHM13 genomic locus of their own, since CHM13's own reference genotype
   happens to carry the long form at both C4A and C4B), each padded with
   +-3kb of REAL flanking sequence read directly from the same CHM13
   reference at that gene's own real locus (the short form's flanks are
   the *same* real flanking sequence as the long form at that locus,
   since flanking DNA sits outside the gene body and shouldn't depend on
   which form the gene itself takes).

   Real HG00253 validation (this session, scratch diagnostics, not
   checked in): a read genuinely extending past a bare gene-body-only
   anchor gets that extra length soft-clipped, contributing nothing to
   depth even though it's real on-target sequence -- +-3kb flanks
   recruited more reads (296->355) and raised gene-body depth
   substantially (e.g. C4B long-form anchor depth 17.3->23.2 in a
   preliminary two-target test). +-6kb was also tried and rejected: total
   hits exploded 355->10,073 while the gene-body-restricted depth barely
   moved, meaning 6kb of generic flanking sequence recruits a flood of
   off-target reads that happen to match the flanks, not the gene --
   +-3kb is the validated choice, do not widen it without re-testing.

CHM13-only for now (invariant 11: provisional). The GRCh38 equivalent of
either the chr17 control interval or C4's own real coordinates has not
been derived -- a GRCh38-aligned sample cannot use this mechanism yet,
same open gap this session's own development notes log names.

Usage: python build.py --chm13-reference /path/to/chm13v2.0.fa
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pysam

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DEFAULT_ANCHORS_FASTA = REPO_ROOT / "resources" / "anchors" / "anchors.fa.gz"

CHR17_CONTROL_REGION = ("chr17", 72_950_000, 77_450_000)  # 0-based half-open

# Real CHM13v2.0 coordinates, 0-based half-open, for C4AL/C4BL's own `gene`
# feature -- read directly off hla_seq/CHM13_0_HLA.gtf's real annotation
# (contig name literally `CHM13#0#chr6:28385000-33500000`, i.e. every
# coordinate in that file is relative to 28,385,000 and needs that added
# back to become a real absolute chr6 position -- confirmed directly this
# session after resources/regions/mhc_window.chm13v2.bed turned out to be
# wrong; do not source C4 coordinates from hla_genes.bed, which has no C4
# entry at all and was the file that misled the MHC-window derivation).
_MHC_WINDOW_ORIGIN = 28_385_000
C4AL_GENE = ("chr6", _MHC_WINDOW_ORIGIN + 3_450_262, _MHC_WINDOW_ORIGIN + 3_470_886)
C4BL_GENE = ("chr6", _MHC_WINDOW_ORIGIN + 3_483_000, _MHC_WINDOW_ORIGIN + 3_503_624)
FLANK_BP = 3_000

C4A_SHORT_ANCHOR_ALLELE = "C4A*asm04c6691625"
C4B_SHORT_ANCHOR_ALLELE = "C4B*asm00bff24228"


def _load_anchor_sequence(anchors_fasta: Path, allele: str) -> str:
    with gzip.open(anchors_fasta, "rt") as fh:
        want = False
        seq: list[str] = []
        for line in fh:
            if line.startswith(">"):
                if want:
                    break
                want = line[1:].split(" ", 1)[0] == allele
                continue
            if want:
                seq.append(line.strip())
    if not seq:
        raise ValueError(f"{allele!r} not found in {anchors_fasta}")
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chm13-reference", type=Path, required=True)
    ap.add_argument("--anchors-fasta", type=Path, default=DEFAULT_ANCHORS_FASTA)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    ref = pysam.FastaFile(str(args.chm13_reference))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    chrom, start, end = CHR17_CONTROL_REGION
    control_seq = ref.fetch(chrom, start, end)
    with gzip.open(args.out_dir / "chr17_control.chm13v2.fa.gz", "wt") as fh:
        fh.write(f">chr17_control_chm13v2_{start+1}-{end}\n{control_seq}\n")
    print(f"chr17_control.chm13v2.fa.gz: {len(control_seq)}bp from {chrom}:{start+1}-{end}")

    def fetch_flanked(gene: tuple[str, int, int], body: str) -> str:
        g_chrom, g_start, g_end = gene
        up = ref.fetch(g_chrom, g_start - FLANK_BP, g_start)
        down = ref.fetch(g_chrom, g_end, g_end + FLANK_BP)
        return up + body + down

    c4al_body = ref.fetch(*C4AL_GENE)
    c4bl_body = ref.fetch(*C4BL_GENE)
    c4as_body = _load_anchor_sequence(args.anchors_fasta, C4A_SHORT_ANCHOR_ALLELE)
    c4bs_body = _load_anchor_sequence(args.anchors_fasta, C4B_SHORT_ANCHOR_ALLELE)

    forms = {
        "C4AL": fetch_flanked(C4AL_GENE, c4al_body),
        "C4BL": fetch_flanked(C4BL_GENE, c4bl_body),
        "C4AS": fetch_flanked(C4AL_GENE, c4as_body),  # C4AL's own real flanks
        "C4BS": fetch_flanked(C4BL_GENE, c4bs_body),  # C4BL's own real flanks
    }
    with gzip.open(args.out_dir / "c4_flanked_forms.chm13v2.fa.gz", "wt") as fh:
        for name, seq in forms.items():
            fh.write(f">{name}_flanked_{FLANK_BP}bp\n{seq}\n")
    for name, seq in forms.items():
        print(f"{name}: {len(seq)}bp (incl. +-{FLANK_BP}bp real flanks)")


if __name__ == "__main__":
    main()

"""Build CHM13v2.0 control resources for C4 and classical-gene CN.

1. chr17_control.chm13v2.fa.gz: chr17:72,950,001-77,450,000 (1-based),
   locityper's segdup-free background interval.
2. c4_flanked_forms.chm13v2.fa.gz: C4AL, C4BL (CHM13 gene bodies from
   hla_seq/CHM13_0_HLA.gtf) and C4AS, C4BS (short-form cd-hit anchors; CHM13
   carries only long forms), each with +-3kb of CHM13 flanking sequence
   from its locus. Flanks recover reads that extend past the gene body
   (HG00253: 296 -> 355 reads); +-6kb flooded the flanks with off-target
   reads.

CHM13-only.

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

# C4AL/C4BL gene coordinates (0-based half-open). CHM13_0_HLA.gtf positions
# are relative to its contig CHM13#0#chr6:28385000-33500000, so add this.
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

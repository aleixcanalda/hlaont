#!/usr/bin/env python3
"""Extract C4A/C4B templates from a real hla_seq/ directory for the C4 copy
number module (DESIGN.md S2f) -- writes a small, standalone bundle
(assembly_bundle/c4.{fa,meta.tsv}.gz) that resources/anchors/prepare_candidates.py
reads for anchor candidates, and that stages/c4_cn.py reads as a Stage 1
alignment pool. This is a SEPARATE mechanism from build_assembly_bundle.py's
HLA-N/P/T/U/W/Y/R bundle and from resources/templates/build.py's IPD-driven
star-allele pool: DESIGN.md S2f is explicit that C4A/C4B have no star-allele
nomenclature and must not be routed through that pool at all (invariant 5 --
this is why the outputs land in separate files, not folded into ref.gene.*).

Real per-haplotype hla_seq/ GTFs (confirmed directly, not assumed) carry
gene_name rows for C4 already split into long/short form by Immuannot's own
annotation: C4AL, C4AS, C4BL, C4BS (long form's ~20kb genomic span includes
the ~6.4kb HERV-K intron-9 insertion DESIGN.md S2f names; short form's ~14kb
span doesn't). AssemblyGTFSource extracts these exactly like any other
assembly-derived gene -- full gene body, real exon/intron/UTR structure, real
per-haplotype dedup.

This script then collapses the four real forms down to two gene buckets,
C4A and C4B, matching resources/genes.yaml's two structural-gene entries
(DESIGN.md S2f: "Total C4 CN" is one segment-level readout, A-vs-B is a
discrimination call on top of it, not two independently-counted genes --
there is no manifest entry for "C4AL" etc. to preserve identity against).
The real long/short form is kept in each record's own `description` for
provenance/future use, not dropped -- but `gene`/`allele` are renamed
(C4AL/C4AS -> C4A*asm<hash>, C4BL/C4BS -> C4B*asm<hash>) so every consumer
that derives gene from allele.split("*")[0] (resources/anchors/build.py's
allele_and_gene()) agrees with this bundle's own `gene` column. This is a
deliberately coarser unit than the NPTUWYR bundle's per-form fidelity --
Stage 1's per-read matrix (stages/assign.py) only needs the two gene buckets
to recruit reads and count total depth (Phase 1 scope: total C4 CN via
depth ratio, see stages/c4_cn.py).

**2026-08-26 Phase 2 addition:** also writes `c4.models.jsonl.gz`
(build.py's own `write_gene_models_jsonl_gz`, same shape/keying as the
NPTUWYR bundle's `bundle.models.jsonl.gz`) -- each record's real
per-haplotype exon/intron structure (`AssemblyGTFSource` already produces
this; it just wasn't being written out before this addition).
`stages/c4_form.py` (the A-vs-B exon-26 / long-vs-short intron-9
discriminator DESIGN.md S2f names) needs real exon 26 and intron 9
boundaries in each individual real sequence's own local coordinate frame --
these differ slightly haplotype to haplotype, so a single fixed offset
would not transfer across records the way a shared reference coordinate
system might.

    python build_c4_bundle.py --hla-seq-dir /path/to/hla_seq
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from assembly_sources import HLA_SEQ_SNAPSHOT_VERSION, AssemblyGTFSource, DEFAULT_BUNDLE_DIR  # noqa: E402
from build import write_fasta_gz, write_gene_models_jsonl_gz, write_meta_tsv_gz  # noqa: E402

# Real gene_name values in hla_seq/'s own GTFs (confirmed directly against
# the real dataset, not guessed) -- long/short form x A/B.
_C4_FORM_GENES = {"C4AL": "C4A", "C4AS": "C4A", "C4BL": "C4B", "C4BS": "C4B"}
_FORM_LABEL = {"C4AL": "long", "C4AS": "short", "C4BL": "long", "C4BS": "short"}


def _collapse_to_c4a_c4b(records):
    """Rename each record's gene/allele from its real form (C4AL/C4AS/C4BL/
    C4BS) to the two genes.yaml buckets (C4A/C4B), keeping the real form as
    an explicit description note -- see module docstring for why."""
    out = []
    for r in records:
        bucket = _C4_FORM_GENES[r.gene]
        form = _FORM_LABEL[r.gene]
        assert r.allele.startswith(r.gene), f"unexpected allele shape: {r.allele!r}"
        new_allele = bucket + r.allele[len(r.gene):]
        out.append(dataclasses.replace(
            r, gene=bucket, allele=new_allele,
            description=f"form={form} ({r.gene}); {r.description}",
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hla-seq-dir", type=Path, required=True,
                     help="Real per-individual phased assemblies (each a "
                          "{sample}_{1,2}_HLA.gtf.gz + all_HLA.fasta.gz pair)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_BUNDLE_DIR,
                     help="Same directory as build_assembly_bundle.py's NPTUWYR "
                          "bundle, but distinct filenames (c4.*) -- kept separate "
                          "so build.py's BundledAssemblySource (fixed bundle.* "
                          "filenames) never picks these up.")
    args = ap.parse_args(argv)

    src = AssemblyGTFSource(genes=set(_C4_FORM_GENES), hla_seq_dir=args.hla_seq_dir)
    raw_records = list(src.records())
    if not raw_records:
        sys.exit(f"[build_c4_bundle] ERROR: zero records extracted from {args.hla_seq_dir} "
                  f"for {sorted(_C4_FORM_GENES)} — loud stop, not a silently empty bundle.")

    records = sorted(_collapse_to_c4a_c4b(raw_records), key=lambda r: (r.gene, r.allele, r.ipd_id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_fasta_gz(args.out_dir / "c4.fa.gz", records)
    write_meta_tsv_gz(args.out_dir / "c4.meta.tsv.gz", records)
    write_gene_models_jsonl_gz(args.out_dir / "c4.models.jsonl.gz", records)

    counts: dict[str, int] = {}
    for r in records:
        counts[r.gene] = counts.get(r.gene, 0) + 1
    forms: dict[str, int] = {}
    for raw in raw_records:
        forms[raw.gene] = forms.get(raw.gene, 0) + 1
    for gene in sorted(counts):
        print(f"[build_c4_bundle] {gene}: {counts[gene]} records", file=sys.stderr)

    manifest = {
        "hla_seq_snapshot_version": HLA_SEQ_SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_hla_seq_dir_name": args.hla_seq_dir.name,
        "records_per_gene": dict(sorted(counts.items())),
        "records_per_real_form": dict(sorted(forms.items())),
        "total_records": len(records),
        "note": "gene/allele collapsed C4AL/C4AS->C4A, C4BL/C4BS->C4B (see script "
                "docstring); real form kept in each record's own description only.",
    }
    (args.out_dir / "c4_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[build_c4_bundle] wrote {len(records)} records across {len(counts)} genes "
          f"to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

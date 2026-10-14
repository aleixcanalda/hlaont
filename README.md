# hlaont

Confidence-aware, full-field HLA typing from Oxford Nanopore (ONT) R10 long reads, designed for population-scale sequencing.

Hlaont does the following:

- Types **41 HLA/MIC/TAP genes**.
- Estimates **copy number per gene**; it can detect haploid instances.
- Reports **C4A/C4B total copy number and long/short-form composition** (the complement C4 segmental duplication).
- Flags **novel alleles** (`novel:intronic`, `novel:utr`, `novel:synonymous`, `novel:nonsynonymous`, `novel:null`, `extension`).
- Accepts **FASTQ or BAM/CRAM** input, and can use real read-backed phase (`HP`/`PS` tags) when available.

## Requirements

- Python ≥ 3.11
- [`minimap2`](https://github.com/lh3/minimap2) and [`samtools`](https://github.com/samtools/samtools) on `PATH` (external CLI tools, not pip-installable — install via conda/bioconda: `conda install -c bioconda minimap2 samtools`)

## Installation

```bash
# from source
git clone https://github.com/aleixcanalda/hlaont.git
cd hlaont
pip install -e .

# or, once published:
# pip install hlaont
# conda install -c bioconda hlaont
```

### First-time setup

The package ships the anchor set, models and a pre-extracted assembly bundle, but the template pool is built from IPD-IMGT/HLA once before `hlaont run` will work:

```bash
hlaont update-db --db-dir hlaont_db/   # downloads IPD-IMGT/HLA and builds the pool into hlaont_db/
hlaont run SAMPLE_ID reads.fastq --db-dir hlaont_db/ --out-dir OUT/
```

`--db-dir` can also be set once with `export HLAONT_DB=hlaont_db/`; without either, the pool lives inside the installed package.

`hlaont update-anchors --db-dir hlaont_db/` rebuilds the anchor set against a new pool (maintainers only; needs `cd-hit-est` on PATH).

## Usage

```bash
# FASTQ input
hlaont run SAMPLE_ID reads.fastq --out-dir OUT/

# BAM/CRAM input (region set auto-derived from the @SQ header — no build/reference flag needed)
hlaont run SAMPLE_ID --bam aligned.bam --out-dir OUT/
hlaont run SAMPLE_ID --bam aligned.cram --reference ref.fa --out-dir OUT/

# Use real read-backed phase (HP/PS tags), when the input BAM has them
hlaont run SAMPLE_ID --bam phased.bam --phased --out-dir OUT/
```

Each run writes, into `--out-dir`:

| file | contents |
|---|---|
| `{sample}.calls.tsv` | one row per haplotype call: gene, copy number, nearest allele, novelty tag, per-field GQ, read support, consensus QV |
| `{sample}.consensus.fa` + `.consensus.qv.tsv` | the polished per-haplotype consensus sequence and its per-base quality |
| `{sample}.variants.vcf` | variants relative to the nearest catalogued allele |
| `{sample}.qc.json` | per-gene copy-number estimates, C4 form breakdown, phasing routing (if `--phased`), and failure reasons for any gene that couldn't be called |

Run `hlaont run --help`, `hlaont update-db --help`, or `hlaont update-anchors --help` for the full flag list.

## Future additions
* IPD-KIR database for KIR allele typing

## License

hlaont's own code is MIT — see [LICENSE](LICENSE). The distributed package also contains Badread's error model (GPL-3.0-only), so the package as a whole is `MIT AND GPL-3.0-only`.

`resources/anchors/vendor/` ports a 136-allele anchor set from [FuFiHLA](https://github.com/jingqing-hu/FuFiHLA) (MIT); `resources/models/vendor/` ports an ONT R10.4.1 error model from [Badread](https://github.com/rrwick/Badread) (GPLv3). See each `NOTICE.md`/license file for full attribution. If you use hlaont, please also cite FuFiHLA (`10.1093/bioinformatics/btag231`).

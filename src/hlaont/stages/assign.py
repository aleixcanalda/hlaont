"""Reads x alleles assignment matrix shared by Stage 1 (CN) and Stage 2
(template selection) (DESIGN.md S1/S2e).

Stage 0's sieved reads are aligned against the full template pool (one
minimap2 index); each read's hits are kept only for the gene(s) its anchor
hits named (CandidateRead.hits), so the alignment cannot reopen paralogue
leakage the sieve rejected. Every allele of a gene is a column (no
candidate cap). Scoring parameters are FuFiHLA's, ported for parity.
"""
from __future__ import annotations

import csv
import gzip
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import CandidateRead  # noqa: E402

DEFAULT_POOL_FASTA = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.fa.gz"
DEFAULT_POOL_META = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.meta.tsv.gz"

# FuFiHLA's scoring string, as in input/fastq.py.
_MINIMAP2_SCORING_ARGS = (
    "-e10000", "-k19", "-w19", "-g10k",
    "-A1", "-B4", "-O6,26", "-E2,1", "-s350",
    "-c", "--end-bonus=10", "-p0.95"
)

# Default -N: a gene pool can hold thousands of alleles (HLA-A: 5707).
_BEST_N = 512

# Per-gene read gate: drop a read whose best cell in this gene's pool has a
# mismatch rate above this (cross-mapped/chimeric reads that passed the
# sieve). minimap2 with these scoring args reports nothing above ~8-9%, so
# 0.15 was a no-op. Too coarse to separate the closest paralogues (<10%
# divergent). Provisional (invariant 11).
_MAX_READ_MISMATCH_RATE = 0.06


@dataclass(frozen=True)
class AlignmentSpan:
    """Coordinates + CIGAR for one (read, allele) hit, PAF convention
    (0-based half-open; read coordinates on the forward strand)."""

    ref_start: int
    ref_end: int
    read_start: int
    read_end: int
    strand: str
    cigar: str


@dataclass(frozen=True)
class GeneAssignmentMatrix:
    """Reads x alleles matrix for one gene, dense over its full pool.
    matches/aln_block_len/nm are 0 where there is no alignment
    (aln_block_len == 0 is the sentinel). `spans` holds the best hit's
    coordinates/CIGAR per aligned (read_index, allele_index), for the
    context scorer."""

    gene: str
    read_ids: tuple[str, ...]
    allele_ids: tuple[str, ...]
    matches: np.ndarray
    aln_block_len: np.ndarray
    nm: np.ndarray
    spans: dict[tuple[int, int], AlignmentSpan] = field(default_factory=dict)

    def mismatch_rate(self) -> np.ndarray:
        """Per (read, allele) mismatch rate; 1.0 (worst) where there is no
        alignment, so unaligned pairs never win an argmin over this row."""
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = self.nm / self.aln_block_len
        return np.where(self.aln_block_len > 0, rate, 1.0)


def subset_matrix_by_read_ids(
    matrix: GeneAssignmentMatrix, read_ids: "set[str] | frozenset[str]",
) -> GeneAssignmentMatrix | None:
    """Row-subset a matrix to `read_ids` without realigning. Equivalent to
    rebuilding it (columns are the full pool; filters are per read). Returns
    None if no rows remain, like build_gene_matrices()."""
    keep = [i for i, rid in enumerate(matrix.read_ids) if rid in read_ids]
    if not keep:
        return None
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep)}
    spans = {
        (old_to_new[i], j): span
        for (i, j), span in matrix.spans.items()
        if i in old_to_new
    }
    return GeneAssignmentMatrix(
        gene=matrix.gene,
        read_ids=tuple(matrix.read_ids[i] for i in keep),
        allele_ids=matrix.allele_ids,
        matches=matrix.matches[keep, :],
        aln_block_len=matrix.aln_block_len[keep, :],
        nm=matrix.nm[keep, :],
        spans=spans,
    )


def classical_matrices_for_reads(
    matrices: dict[str, GeneAssignmentMatrix],
    reads: Sequence[CandidateRead],
    genes: "Sequence[str]",
) -> dict[str, GeneAssignmentMatrix]:
    """Multi-gene `subset_matrix_by_read_ids()`: per-gene matrices for a read
    pool spanning `genes`, using each read's own CandidateRead.hits."""
    genes_set = set(genes)
    read_ids_by_gene: dict[str, set[str]] = {}
    for r in reads:
        for hit in r.hits:
            if hit.gene in genes_set:
                read_ids_by_gene.setdefault(hit.gene, set()).add(r.read_id)
    out: dict[str, GeneAssignmentMatrix] = {}
    for gene, ids in read_ids_by_gene.items():
        full = matrices.get(gene)
        if full is None:
            continue
        sub = subset_matrix_by_read_ids(full, ids)
        if sub is not None:
            out[gene] = sub
    return out


def gene_allele_pool(gene: str, pool_meta: Path = DEFAULT_POOL_META) -> tuple[str, ...]:
    """One gene's sorted allele pool (the matrix columns); () if absent."""
    _, genes_to_alleles = _load_allele_gene_and_catalogue(pool_meta)
    return tuple(genes_to_alleles.get(gene, []))


def merge_matrices(a: GeneAssignmentMatrix, b: GeneAssignmentMatrix) -> GeneAssignmentMatrix:
    """Union two same-gene, same-column matrices over disjoint rows. Raises
    on gene/column mismatch or a shared read_id."""
    if a.gene != b.gene:
        raise ValueError(f"merge_matrices: gene mismatch ({a.gene!r} vs {b.gene!r})")
    if a.allele_ids != b.allele_ids:
        raise ValueError(
            f"{a.gene}: merge_matrices requires identical allele_ids "
            f"({len(a.allele_ids)} vs {len(b.allele_ids)} columns)"
        )
    overlap = set(a.read_ids) & set(b.read_ids)
    if overlap:
        raise ValueError(
            f"{a.gene}: merge_matrices got {len(overlap)} read_id(s) present in "
            f"both matrices -- caller must guarantee disjoint row sets"
        )
    read_ids = tuple(sorted(a.read_ids + b.read_ids))
    index_in_merged = {rid: i for i, rid in enumerate(read_ids)}
    matches = np.zeros((len(read_ids), len(a.allele_ids)), dtype=a.matches.dtype)
    aln_block_len = np.zeros_like(matches)
    nm = np.zeros_like(matches)
    spans: dict[tuple[int, int], AlignmentSpan] = {}
    for src in (a, b):
        for old_i, rid in enumerate(src.read_ids):
            new_i = index_in_merged[rid]
            matches[new_i, :] = src.matches[old_i, :]
            aln_block_len[new_i, :] = src.aln_block_len[old_i, :]
            nm[new_i, :] = src.nm[old_i, :]
        for (old_i, j), span in src.spans.items():
            spans[(index_in_merged[src.read_ids[old_i]], j)] = span
    return GeneAssignmentMatrix(
        gene=a.gene, read_ids=read_ids, allele_ids=a.allele_ids,
        matches=matches, aln_block_len=aln_block_len, nm=nm, spans=spans,
    )


def _load_allele_gene_and_catalogue(meta_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """allele -> gene, and gene -> sorted list of every allele in its pool
    (the latter defines each GeneAssignmentMatrix's full column set)."""
    allele_gene: dict[str, str] = {}
    genes_to_alleles: dict[str, list[str]] = {}
    with gzip.open(meta_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            allele_gene[row["allele"]] = row["gene"]
            genes_to_alleles.setdefault(row["gene"], []).append(row["allele"])
    for alleles in genes_to_alleles.values():
        alleles.sort()
    return allele_gene, genes_to_alleles


def _write_reads_fasta(reads: Sequence[CandidateRead], fasta_path: Path) -> None:
    with open(fasta_path, "w") as fh:
        for r in reads:
            fh.write(f">{r.read_id}\n{r.sequence}\n")


@dataclass(frozen=True)
class _RawHit:
    read_id: str
    allele: str
    matches: int
    aln_block_len: int
    nm: int
    span: AlignmentSpan


def _scoring_args_with_secondary_ratio(ratio: float) -> tuple[str, ...]:
    """`_MINIMAP2_SCORING_ARGS` with `-p0.95` replaced by `-p{ratio}`."""
    return tuple(a for a in _MINIMAP2_SCORING_ARGS if not a.startswith("-p")) + (f"-p{ratio}",)


def _run_minimap2(
    pool_fasta: Path, reads_fasta: Path, n_threads: int, minimap2_bin: str, n_best: int,
    *, scoring_args: tuple[str, ...] = _MINIMAP2_SCORING_ARGS,
) -> list[_RawHit]:
    """Run minimap2 with the pool as index and reads as query; one _RawHit
    per PAF line, including span and CIGAR."""
    cmd = [
        minimap2_bin,
        "-t", str(n_threads),
        *scoring_args,
        "-N", str(n_best),
    ]
    cmd += [str(pool_fasta), str(reads_fasta)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    hits: list[_RawHit] = []
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        read_id, allele = fields[0], fields[5]
        read_start, read_end = int(fields[2]), int(fields[3])
        strand = fields[4]
        ref_start, ref_end = int(fields[7]), int(fields[8])
        matches, aln_block_len = int(fields[9]), int(fields[10])
        nm = next((int(f[5:]) for f in fields[12:] if f.startswith("NM:i:")), None)
        if nm is None:
            raise ValueError(f"minimap2 PAF line missing NM tag: {line!r}")
        cigar = next((f[5:] for f in fields[12:] if f.startswith("cg:Z:")), None)
        if cigar is None:
            raise ValueError(f"minimap2 PAF line missing cg:Z: tag: {line!r}")
        hits.append(_RawHit(
            read_id=read_id, allele=allele, matches=matches,
            aln_block_len=aln_block_len, nm=nm,
            span=AlignmentSpan(
                ref_start=ref_start, ref_end=ref_end,
                read_start=read_start, read_end=read_end,
                strand=strand, cigar=cigar,
            ),
        ))
    return hits


def build_gene_matrices(
    reads: Sequence[CandidateRead],
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    max_read_mismatch_rate: float = _MAX_READ_MISMATCH_RATE,
    n_best: int = _BEST_N,
) -> dict[str, GeneAssignmentMatrix]:
    """One GeneAssignmentMatrix per gene in `reads`' Stage 0 hits.

    Hits are kept only for genes the read was sieved to, and a read is kept
    in a gene only if its best cell's mismatch rate <=
    `max_read_mismatch_rate`.

    `n_best` (minimap2 -N) defaults to 512, which Stage 4 needs (one
    consensus read must report many candidates for ambiguity_set/field_gq).
    pipeline.py's Stage 1 passes 1: with many hits per read, redundant names
    (HLA-A) and paralogue families (DRB1) outvote the true allele on depth.
    """
    if not reads:
        return {}

    allele_gene, genes_to_alleles = _load_allele_gene_and_catalogue(pool_meta)

    read_genes: dict[str, frozenset[str]] = {
        r.read_id: frozenset(h.gene for h in r.hits) for r in reads
    }

    with tempfile.TemporaryDirectory() as tmp:
        reads_fasta = Path(tmp) / "reads.fasta"
        _write_reads_fasta(reads, reads_fasta)
        raw_hits = _run_minimap2(pool_fasta, reads_fasta, n_threads, minimap2_bin, n_best)

    # Keep the best (most matches) of duplicate PAF lines per (read, allele)
    # so split alignments can't count twice.
    best_by_gene: dict[str, dict[tuple[str, str], _RawHit]] = {}
    for hit in raw_hits:
        if hit.allele not in allele_gene:
            raise ValueError(
                f"allele {hit.allele!r} (hit by read {hit.read_id!r}) has no "
                f"entry in {pool_meta} -- ref.gene.fa.gz and "
                f"ref.gene.meta.tsv.gz have drifted apart"
            )
        gene = allele_gene[hit.allele]
        if gene not in read_genes.get(hit.read_id, frozenset()):
            continue  # not one of this read's Stage 0 gene assignments
        cell = (hit.read_id, hit.allele)
        bucket = best_by_gene.setdefault(gene, {})
        prior = bucket.get(cell)
        if prior is None or hit.matches > prior.matches:
            bucket[cell] = hit

    # Apply the read gate; drop genes left with no reads (-> pipeline's
    # "zero reads recruited" failure).
    for gene in list(best_by_gene):
        cells = best_by_gene[gene]
        best_rate_per_read: dict[str, float] = {}
        for (read_id, _allele), hit in cells.items():
            rate = hit.nm / hit.aln_block_len
            if rate < best_rate_per_read.get(read_id, float("inf")):
                best_rate_per_read[read_id] = rate
        filtered = {
            (read_id, allele): hit for (read_id, allele), hit in cells.items()
            if best_rate_per_read[read_id] <= max_read_mismatch_rate
        }
        if filtered:
            best_by_gene[gene] = filtered
        else:
            del best_by_gene[gene]

    matrices: dict[str, GeneAssignmentMatrix] = {}
    for gene, cells in best_by_gene.items():
        read_ids = tuple(sorted({read_id for read_id, _ in cells}))
        allele_ids = tuple(genes_to_alleles[gene])  # already sorted, full pool
        read_index = {rid: i for i, rid in enumerate(read_ids)}
        allele_index = {a: i for i, a in enumerate(allele_ids)}

        matches_arr = np.zeros((len(read_ids), len(allele_ids)), dtype=np.int32)
        aln_len_arr = np.zeros_like(matches_arr)
        nm_arr = np.zeros_like(matches_arr)
        spans: dict[tuple[int, int], AlignmentSpan] = {}
        for (read_id, allele), hit in cells.items():
            i, j = read_index[read_id], allele_index[allele]
            matches_arr[i, j] = hit.matches
            aln_len_arr[i, j] = hit.aln_block_len
            nm_arr[i, j] = hit.nm
            spans[(i, j)] = hit.span

        matrices[gene] = GeneAssignmentMatrix(
            gene=gene,
            read_ids=read_ids,
            allele_ids=allele_ids,
            matches=matches_arr,
            aln_block_len=aln_len_arr,
            nm=nm_arr,
            spans=spans,
        )
    return matrices


def realign_to_candidates(
    gene: str,
    reads: Sequence[CandidateRead],
    allele_sequences: dict[str, str],
    *,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
    min_secondary_ratio: float | None = None,
) -> GeneAssignmentMatrix:
    """Densely realign `reads` against exactly `allele_sequences` (used as
    the whole pool, -N = candidate count), so every read gets a span against
    every final candidate. At n_best=1 most reads' single hit lands on a
    near-identical sibling (HLA-A: 3 of 26 true reads had a span). No read
    gate or gene restriction: `reads` are already known to belong to `gene`.

    `min_secondary_ratio` lowers minimap2's -p for Stage 4, where a full
    consensus query can drop a correct but shorter-UTR record entirely. None
    keeps the shared args."""
    if not reads:
        raise ValueError(f"{gene}: realign_to_candidates() called with no reads")
    if not allele_sequences:
        raise ValueError(f"{gene}: realign_to_candidates() called with no candidates")

    allele_ids = tuple(sorted(allele_sequences))
    scoring_args = (
        _MINIMAP2_SCORING_ARGS if min_secondary_ratio is None
        else _scoring_args_with_secondary_ratio(min_secondary_ratio)
    )
    with tempfile.TemporaryDirectory() as tmp:
        reads_fasta = Path(tmp) / "reads.fasta"
        candidates_fasta = Path(tmp) / "candidates.fasta"
        _write_reads_fasta(reads, reads_fasta)
        with open(candidates_fasta, "w") as fh:
            for a in allele_ids:
                fh.write(f">{a}\n{allele_sequences[a]}\n")
        raw_hits = _run_minimap2(
            candidates_fasta, reads_fasta, n_threads, minimap2_bin, len(allele_ids),
            scoring_args=scoring_args,
        )

    best: dict[tuple[str, str], _RawHit] = {}
    for hit in raw_hits:
        cell = (hit.read_id, hit.allele)
        prior = best.get(cell)
        if prior is None or hit.matches > prior.matches:
            best[cell] = hit

    # Only reads with a hit get a row.
    read_ids = tuple(sorted({read_id for read_id, _ in best}))
    read_index = {rid: i for i, rid in enumerate(read_ids)}
    allele_index = {a: i for i, a in enumerate(allele_ids)}

    matches_arr = np.zeros((len(read_ids), len(allele_ids)), dtype=np.int32)
    aln_len_arr = np.zeros_like(matches_arr)
    nm_arr = np.zeros_like(matches_arr)
    spans: dict[tuple[int, int], AlignmentSpan] = {}
    for (read_id, allele), hit in best.items():
        i, j = read_index[read_id], allele_index[allele]
        matches_arr[i, j] = hit.matches
        aln_len_arr[i, j] = hit.aln_block_len
        nm_arr[i, j] = hit.nm
        spans[(i, j)] = hit.span

    return GeneAssignmentMatrix(
        gene=gene,
        read_ids=read_ids,
        allele_ids=allele_ids,
        matches=matches_arr,
        aln_block_len=aln_len_arr,
        nm=nm_arr,
        spans=spans,
    )


# Reads whose best mismatch rate in several genes is within this margin of
# their global best split their depth credit evenly across those genes
# (sum stays 1.0). A hard argmin scattered HLA-H/K/T/U/W depth across the
# cluster on noise-level differences (0.31% vs 0.12%). Provisional.
_GENE_RESOLUTION_MISMATCH_MARGIN = 0.003


def best_gene_per_read(
    matrices: dict[str, GeneAssignmentMatrix],
    *,
    margin: float = _GENE_RESOLUTION_MISMATCH_MARGIN,
) -> dict[str, dict[str, float]]:
    """read_id -> {gene: weight}, weights summing to 1.0: genes within
    `margin` of the read's best mismatch rate share the credit evenly."""
    read_rates: dict[str, dict[str, float]] = {}
    for gene in sorted(matrices):
        m = matrices[gene]
        if m.read_ids == ():
            continue
        best_rate_per_read = m.mismatch_rate().min(axis=1)
        for read_id, rate in zip(m.read_ids, best_rate_per_read):
            read_rates.setdefault(read_id, {})[gene] = float(rate)

    resolved: dict[str, dict[str, float]] = {}
    for read_id, rates in read_rates.items():
        best_rate = min(rates.values())
        tied = sorted(g for g, r in rates.items() if r <= best_rate + margin)
        weight = 1.0 / len(tied)
        resolved[read_id] = {gene: weight for gene in tied}
    return resolved

"""Shared reads x alleles assignment matrix (DESIGN.md S1/S2e).

This is not a numbered stage on its own -- it is the machinery both Stage 1
(CN, S1) and Stage 2 (template selection, S2e) build on. DESIGN.md S1 is
explicit about this: "CN is not a separate stage -- it is a second readout
of the read x allele assignment matrix Stage 2 already builds." Building it
once here, shared, is what makes that true in code and not just in prose.

Inputs are Stage 0's sieved reads (input/fastq.py's CandidateRead stream --
already anchor-sieved and paralog/decoy-rejected, S3.2/S3.3), which come with
the gene(s) each read qualified for (CandidateRead.hits). This module does
not re-decide gene membership: it restricts each read's candidate alleles to
those genes, then aligns against the *full* per-gene IPD template pool
(resources/templates/pool/, S2d) rather than a filtered subset -- the v0
choice is one shared minimap2 index over the whole pool (rebuilding a
per-sample gene-subset index would cost more than it saves; the anchor-gene
restriction is applied to the alignment results afterward, not the index)
and every allele for the read's assigned gene(s), not a pre-capped candidate
list (S2e: FuFiHLA's top-15/top-30 cap is a performance workaround, not a
principle -- see invariant 11).

Alignment reuses the same minimap2 scoring string as the anchor sieve
(input/fastq.py) for the same reason it was ported there verbatim: parity
with FuFiHLA on the milestone-1 regression target, not a re-derived-for-ONT
constant.
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

# Identical to input/fastq.py's ported FuFiHLA scoring string -- see module
# docstring for why this is exempt from invariant 11 in the same way that
# module's copy is.
_MINIMAP2_SCORING_ARGS = (
    "-e10000", "-k19", "-w19", "-g10k",
    "-A1", "-B4", "-O6,26", "-E2,1", "-s350",
    "-c", "--end-bonus=10", "-p0.95"
)

# Search-completeness knob (same reasoning as input/fastq.py's _BEST_N): the
# per-gene pool can hold thousands of alleles (HLA-A: 5707), and every one of
# them is a legitimate matrix column, not a top-N-by-chaining-score cap.
_BEST_N = 512

# Read-quality gate (added 2026-07-27, real HG00097 ONT benchmark, invariant
# 11 -- provisional, not fitted): a read whose *best* cell within a given
# gene's own pool still has this poor a CIGAR-derived mismatch rate
# (nm/aln_block_len, from minimap2's own PAF NM tag -- no extra alignment
# work) is unlikely to genuinely originate from that gene. The Stage 0
# anchor sieve already rejects decoy/paralog hits at the anchor level
# (input/fastq.py), but that check is per-anchor, not per-full-length
# alignment against the real template pool this module builds -- a
# cross-mapped/misassigned or chimeric read can still pass the sieve and
# land in a gene's matrix here, inflating its depth (Stage 1 CN) and
# contaminating its candidate/consensus reads (Stage 2/3). This is a *per
# gene* gate, not global: a read failing it here can still be trusted for a
# different gene it aligns well to. 0.06 -- lowered from an initial 0.15
# after actually checking: this module's own _MINIMAP2_SCORING_ARGS
# (-A1 -B4 -s350, ported from FuFiHLA) already refuses to report *any*
# alignment above roughly 8-9% scattered mismatch at all (confirmed
# empirically), so 0.15 was silently a no-op -- every cell minimap2 ever
# reports already cleared it. 0.06 sits with real headroom under that
# ceiling (not right at the boundary, where the gate would only ever catch
# the rare read landing in a ~1% window) so it can actually bind on reads
# minimap2 still reports but that are still comparatively poor matches.
# Still not derived from real ONT R10 divergence data between this
# project's closest paralog pairs (HLA-A/HLA-H, DRB1/DRB3-5 -- DESIGN.md's
# own "makes or breaks the gene expansion" warning, and note those pairs
# are >90% identical, i.e. <10% divergent -- close enough to minimap2's own
# reporting ceiling that a mismatch-rate gate alone is unlikely to cleanly
# separate them; revisit once calib/ has real data to fit against).
_MAX_READ_MISMATCH_RATE = 0.06


@dataclass(frozen=True)
class AlignmentSpan:
    """Coordinates + CIGAR for one (read, allele) hit, PAF convention
    (0-based half-open; qstart/qend always on the read's *forward* strand
    regardless of `strand` -- see stages/error_model.py's
    build_aligned_strings for how that's resolved). Kept alongside the
    dense matrix (not folded into it) because a CIGAR string has no fixed
    width to sit in a numpy array column.
    """

    ref_start: int
    ref_end: int
    read_start: int
    read_end: int
    strand: str
    cigar: str


@dataclass(frozen=True)
class GeneAssignmentMatrix:
    """Reads x alleles alignment matrix for one gene, dense over the gene's
    full template pool -- every allele is a column, whether or not any read
    hit it (S2e: no candidate-cap pre-filtering).

    matches/aln_block_len/nm are 0 wherever minimap2 reported no alignment
    for that (read, allele) pair; aln_block_len == 0 is the sentinel for "no
    alignment", distinguishable from a real (impossible) zero-length one.

    `spans` carries the same "best duplicate" hit's coordinates/CIGAR as
    matches/aln_block_len/nm, keyed by (read_index, allele_index), but only
    for cells with a real alignment -- stages/genotype.py's context-aware
    scorer (stages/error_model.py) needs the actual alignment path, which a
    dense int array can't hold. Defaults to empty so callers that only need
    the aggregate counts (Stage 1 CN, stages/cn.py) are unaffected.
    """

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
    """Row-subset an already-built matrix to just `read_ids`, without
    re-running minimap2. Added 2026-08-28 (the development notes' phased-pipeline
    performance investigation, lever 2): `stages/pipeline.py`'s
    `run_sample_phased()` used to call `build_gene_matrices()` a second time
    per haplotype on the identical gene's already-aligned reads -- this
    reuses Stage 1's own full-sample matrix instead.

    Equivalent to calling `build_gene_matrices()` again on just `read_ids`,
    not merely a cheaper approximation of it: `allele_ids` is always the
    gene's *full* pool regardless of which reads are present (module
    docstring), and `max_read_mismatch_rate`/`n_best` are both applied
    per-read within a gene's own pool, independent of which other reads
    share that pool -- so which rows happen to be present changes nothing
    about how any individual row was computed. Returns `None` when no rows
    survive, matching `build_gene_matrices()`'s own behaviour of dropping a
    gene entirely rather than keeping an empty matrix (so callers can reuse
    the identical `matrix is None` -> GeneFailure path)."""
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
    """Multi-gene counterpart of `subset_matrix_by_read_ids()`: given a
    sample-wide `matrices` dict (Stage 1's own output) and a read pool that
    mixes reads sieved to several different `genes`, reconstruct the
    per-gene subset matrices `build_gene_matrices(reads, ...)` would have
    produced -- e.g. `stages/pipeline.py`'s per-haplotype C4 calibrator
    panel (S10.5/S10.6), which pools reads across several class I genes.
    Gene membership comes from each read's own `CandidateRead.hits`, the
    same source `build_gene_matrices()` itself uses, restricted to `genes`
    -- so a read hit against a gene outside that list is never included,
    matching what a real `build_gene_matrices(reads, ...)` call restricted
    to this read pool would have assigned it to anyway (that call sees the
    full pool regardless, but no other manifest gene has a matching
    calibrator/pool entry in this caller's `matrices` dict to land in)."""
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
    """One gene's full, sorted allele pool -- the same column set
    `build_gene_matrices()` gives that gene's matrix (module docstring),
    exposed standalone for callers that need it before/without a matrix
    (e.g. `stages/pipeline.py`'s `rescue_stolen_reads()`, which needs the
    full candidate set to densely realign a gene's own under-recruited
    reads against). Thin public wrapper over `_load_allele_gene_and_catalogue()`
    -- returns `()` for a gene with no pool entries, never a KeyError."""
    _, genes_to_alleles = _load_allele_gene_and_catalogue(pool_meta)
    return tuple(genes_to_alleles.get(gene, []))


def merge_matrices(a: GeneAssignmentMatrix, b: GeneAssignmentMatrix) -> GeneAssignmentMatrix:
    """Union two same-gene, same-column (`allele_ids`) matrices over
    disjoint rows -- e.g. `rescue_stolen_reads()` stitching a dense
    realignment of a gene's previously-unmatched reads back onto its
    original Stage 1 matrix. Raises loudly (never silently drops or
    double-counts a read) if the two don't actually share one gene/column
    set, or if a read_id appears in both -- both would mean a caller
    built `b` against the wrong candidate set or the wrong read pool,
    a real bug to surface, not paper over."""
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
    """`_MINIMAP2_SCORING_ARGS` with its `-p0.95` replaced by `-p{ratio}` --
    a standalone tuple, never mutating the shared constant, so every
    existing caller of `_run_minimap2()`/`realign_to_candidates()` that
    doesn't ask for this is byte-identical to before. See
    `realign_to_candidates()`'s `min_secondary_ratio` docstring for why
    this exists."""
    return tuple(a for a in _MINIMAP2_SCORING_ARGS if not a.startswith("-p")) + (f"-p{ratio}",)


def _run_minimap2(
    pool_fasta: Path, reads_fasta: Path, n_threads: int, minimap2_bin: str, n_best: int,
    *, scoring_args: tuple[str, ...] = _MINIMAP2_SCORING_ARGS,
) -> list[_RawHit]:
    """Run minimap2 pool-as-index/reads-as-query. Returns a flat list of
    one _RawHit per PAF line, including the alignment span/CIGAR
    (stages/error_model.py's context scorer needs the actual alignment
    path, not just the aggregate NM/aln_block_len counts).

    Pool as index/reads as query (not the reverse) matches input/fastq.py's
    own anchors-as-index/reads-as-query convention for the identical reason:
    the thing being scored per-row is "how well does this read match each
    candidate", so the read is the query in both places.

    `n_best` is minimap2's own `-N` (see build_gene_matrices()'s docstring
    for why this is caller-supplied rather than the fixed `_BEST_N`).
    `scoring_args` defaults to the shared, ported `_MINIMAP2_SCORING_ARGS`
    -- every existing caller gets identical behaviour; only
    `realign_to_candidates()`'s `min_secondary_ratio` override replaces it."""
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
    """Build one GeneAssignmentMatrix per gene present in `reads`' Stage 0
    hits. n_threads is minimap2's own resource knob (see input/fastq.py),
    caller-supplied.

    A read's candidate alleles are restricted to the gene(s) its Stage 0
    AnchorHits named (CandidateRead.hits) -- an allele hit against a gene the
    read was never sieved to is dropped, never a matrix entry. This is the
    paralog-leakage guard DESIGN.md S1 says makes or breaks the gene
    expansion: the sieve already rejected paralog/decoy hits at the anchor
    level, and this step must not silently reopen that by trusting whatever
    minimap2 happens to align best against the full pool.

    A second, narrower guard on top of that: within a gene's own pool, a
    read is only kept if its best cell's mismatch rate is <=
    max_read_mismatch_rate (module docstring on _MAX_READ_MISMATCH_RATE) --
    a read whose best-possible match to this specific gene is still poor is
    unlikely to really be this gene's read, whatever the sieve's anchor-level
    check made of it.

    `n_best` (minimap2's own `-N`) defaults to the module's `_BEST_N` (512)
    -- unchanged behaviour for any existing caller, notably
    stages/hla_typing.py's Stage 4 re-typing call, which wraps a whole
    consensus sequence as a single synthetic "read" and genuinely needs
    many reported candidates to populate `ambiguity_set`/`field_gq`
    (collapsing to one reported hit there would silently zero out both).
    stages/pipeline.py's Stage 1/2 call overrides this down to a small
    value instead (see its own docstring) -- a real per-read matrix has a
    completely different failure mode: at `_BEST_N=512`, one read can land
    a real, if slightly worse, alignment against dozens to hundreds of
    alleles at once (confirmed on real HG00097 data: HLA-A's true allele
    crowded out by redundant IPD names all hit by the identical reads;
    HLA-DRB1's true alleles losing a depth-based ranking to several
    off-target families that only accumulated real depth *because* every
    read voted for many candidates simultaneously, paralog-crowding
    diffused across its dense, closely-related catalogue). Restricting to
    each read's single best-scoring hit removes that fan-out at the
    source: real HG00097 data confirmed this recovers the true allele set
    for HLA-A, HLA-C, HLA-DRB1, MICA and MICB where `_BEST_N=512` did not,
    without needing genotype.py's narrow_candidates_by_mismatch_rate() to
    compensate after the fact. Provisional (invariant 11) -- validated on
    one real sample, not fitted.
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

    # (gene -> {(read_id, allele): best _RawHit}), keeping the best
    # (highest matches) of any duplicate PAF lines for the same (read,
    # allele) pair -- e.g. from a split/chimeric alignment -- so a read
    # cannot inflate its own score by aligning twice to one allele.
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

    # Drop reads whose best cell within this gene's own pool is still a
    # poor match (max_read_mismatch_rate) -- see the function docstring. A
    # gene left with zero surviving reads is dropped entirely (not kept as
    # an empty matrix) so it hits the same "zero reads recruited" failure
    # path as a gene that never had any hits at all (pipeline.py's
    # `matrix is None` branch, DESIGN.md S2c's HLA-Y warning).
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
    """Dense re-alignment of `reads` against exactly `allele_sequences` --
    the fix for a real gap `build_gene_matrices()`'s Stage 1/2 caller
    exposes when it lowers `n_best` (that function's own docstring): once
    Stage 2 has narrowed a template-heavy gene's field down to a handful
    of final candidates, most reads that genuinely belong to one of them
    can still be missing a captured span there, because their *one*
    reported hit (at low `n_best`) landed on a different, near-identical
    named allele instead -- confirmed on real HG00097 data (HLA-A: only 3
    of 26 true-allele reads had a captured span against the specific
    candidate name Stage 2 settled on).

    This re-runs the alignment with the candidate set *as the entire
    pool* -- cheap (reads x a handful of candidates, not reads x
    thousands) -- so every read gets a real, direct comparison against
    every final candidate specifically, not whatever a single broad sweep
    over the full pool happened to report. `-N` is sized to the candidate
    count itself (there is nothing else in the pool to rank against), so
    every real hit is captured. No `max_read_mismatch_rate` gate and no
    Stage 0 gene-hit restriction here (unlike build_gene_matrices) --
    `reads` is already the caller's known-relevant set for `gene`, and a
    poor-but-real alignment is still real information for the scorer, not
    something to filter out pre-emptively.

    Callers: stages/genotype.py's select_template_set_ont_r10() (to refine
    the winning set's reported likelihood/weights/LRT before Stage 3 ever
    runs) and stages/consensus.py's build_gene_consensus() (so the read
    partition and the per-base pileup both see the same accurate data).

    `min_secondary_ratio` (2026-09-02, real HG00097 TAP2 finding): `None`
    (default) keeps the shared, ported `_MINIMAP2_SCORING_ARGS` (`-p0.95`)
    unchanged -- every existing caller above is unaffected. That ratio is
    minimap2's own min-secondary-to-primary-*score* cutoff, and it applies
    per read regardless of candidate-pool size: a real bug found via a
    *query long enough to expose it* -- stages/hla_typing.py's Stage 4
    re-scores a full-length polished *consensus*, not a short raw read, and
    when one candidate (e.g. a real IPD allele missing ~1kb of UTR another
    submission has) can only ever cover a shorter fraction of that
    consensus, its best possible score against the ratio-0.95 threshold
    can fall below cutoff purely from matched-length, even with zero real
    mismatches over the span it does cover -- so minimap2 drops the
    alignment record entirely, not just ranks it lower. Individual reads
    rarely trigger this (short enough to usually sit fully inside whatever
    two closely-related candidates share), which is why this was never a
    visible problem for Stage 1/2/3's own read-level uses of this function.
    Passing a lower ratio here (hla_typing.py's own fallback path) trades
    permissiveness for completeness *at this one call site only* --
    downstream mismatch-rate-first scoring (stages/hla_typing.py's
    `_MISMATCH_RATE_TIE_TOLERANCE` rule) is what actually judges candidate
    quality, so a spurious short/poor alignment let through here still
    loses on its own merits rather than winning by default."""
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

    # Only reads with >=1 real hit against a candidate get a row -- same
    # "no information, not a zero" convention build_gene_matrices() uses,
    # not every read handed in (a read that fails to align to *any* final
    # candidate at all is rare here since `reads` is already known-relevant
    # to `gene`, but should be absent, not a spurious all-floor row).
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


# Margin-gated depth resolution (added 2026-08-31, user-directed real-data
# root cause). best_gene_per_read() used to be a hard argmin: whichever gene
# had the single lowest mismatch rate took the read's entire depth credit.
# For HLA-H/K/T/U/W -- the tight class I pseudogene paralog cluster
# DESIGN.md S1/assign.py's own module docstring already flag as this
# project's "makes or breaks the gene expansion" case -- a read genuinely
# from one of these loci routinely aligns near-identically to two or three
# of the others too, so the argmin was frequently resolved on a margin as
# small as one mismatched base over a few hundred: confirmed directly on
# real HG00097 data, one read's own-pool mismatch rate was 0.31% against
# HLA-H vs 0.12% against HLA-T -- both essentially perfect matches, not a
# real signal that the read belongs to one locus and not the other. A hard
# winner-take-all resolution still credited the whole read to a single gene
# regardless, scattering each cluster gene's true depth across its
# neighbours (real HG00097 counts: HLA-H had 33 reads with a genuine
# own-pool alignment but only 19 credited; HLA-K 45 -> 22; HLA-T 36 -> 25;
# HLA-U 37 -> 20) -- enough to push cn_estimate below the rounding
# threshold and hide a real second haplotype.
#
# Fix: a read's total depth credit (1.0) is split evenly across every gene
# whose own best-cell mismatch rate is within this absolute margin of the
# read's global-best rate, rather than concentrated on whichever gene
# happens to win the argmin. A read with one clear winner (no other gene
# within the margin) is unaffected -- full 1.0 credit, byte-identical to
# the pre-existing behaviour -- so this only changes genuinely near-tied
# reads. The 1.0 unit is *distributed*, never duplicated: a read's credit
# across every gene it resolves to still sums to exactly 1.0, keeping the
# "no double counting" property build_gene_matrices()'s own gene-
# restriction filter already establishes.
#
# 0.003 (0.3 percentage points) -- chosen with real headroom over the
# documented HLA-H/HLA-T case above (a 0.0019 gap), not fitted (invariant
# 11): comfortably wider than a single-base ONT-noise swing on a
# few-hundred-base alignment, comfortably narrower than the gap between a
# gene's own true allele and an unrelated gene outside this cluster (whose
# rates are not remotely this close in practice). Provisional, pending the
# real HG00097/HG00320 benchmark this change is being validated against --
# revisit if either regresses.
_GENE_RESOLUTION_MISMATCH_MARGIN = 0.003


def best_gene_per_read(
    matrices: dict[str, GeneAssignmentMatrix],
    *,
    margin: float = _GENE_RESOLUTION_MISMATCH_MARGIN,
) -> dict[str, dict[str, float]]:
    """Resolve each read's depth credit across the gene(s) its Stage 0 sieve
    named it for -- margin-gated, not a hard argmin (see
    _GENE_RESOLUTION_MISMATCH_MARGIN's docstring for the real-data
    motivation). Returns read_id -> {gene: weight}; the weights for one
    read's entry always sum to 1.0. Every gene within `margin` of the
    read's own global-minimum mismatch rate shares that 1.0 credit evenly
    -- ties are resolved by symmetry, not by an arbitrary/alphabetical
    pick, so determinism (invariant 9) holds without needing an explicit
    tie-break rule.

    This is the depth numerator for Stage 1 CN (stages/cn.py's
    depth_by_gene()): a read genuinely ambiguous between two paralogous
    genes' pools now contributes partial depth to each, instead of the
    whole read landing on whichever gene's rate happened to be
    infinitesimally lower.
    """
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

# Provenance

`nanopore2023.gz` is ported verbatim from Badread
(`github.com/rrwick/Badread`, `badread/error_models/nanopore2023.gz`),
GPLv3 licensed — see `BADREAD_LICENSE` in this directory for the full
notice. Fetched 2026-07-23.

Trained on ONT R10.4.1 reads from 2023 — Badread's own default
`--error_model`, and the one model in their shipped set that actually
matches this project's operating point (invariant 1). Not re-derived: this
is a real per-7-mer empirical error distribution fitted from real ONT R10
alignments, which is a far better starting point than a made-up flat
per-base rate (see `stages/error_model.py`'s docstring for how it's applied
to scoring rather than Badread's own simulation use).

File format (one line per possible 7-mer, 16384 total):
`kmer,P(correct);alt_kmer_1,P(alt_1);alt_kmer_2,P(alt_2);...` — the listed
alternatives are truncated to the top ones seen during training, so the
per-line probabilities sum to <= 1; the remainder is the probability mass
of an alternative not explicitly listed. Confirmed against
`github.com/rrwick/Badread/wiki/Error-models`.

Citation (regardless of what the licence compels):

> Wick, R.R. Badread: simulating error-prone long reads. J Open Source
> Softw 4, 1316 (2019). https://doi.org/10.21105/joss.01316

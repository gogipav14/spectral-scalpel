# paper_toms - ACM TOMS submission readiness status

**Target:** ACM Transactions on Mathematical Software.
**Badge requested:** Artifact Available.

**Strategic constraint:** This bundle is the post-SISC-desk-rejection
reframe of paper 1 from algorithm-paper to software-paper. Decision
to target ACM TOMS made 2026-06-12 after a risk analysis ruled out
JCP (queue saturation - 3 papers in review) and SISC (Filbet
scope-based rejection). See [paper1-venue-history memory](../../.claude/projects/-home-gogip-github-repos-spectral-scalpel-private/memory/project_paper1_venue_pivot.md).

## What's in the bundle

| Path | Pages / size | Purpose |
|---|---|---|
| `article.tex` -> `article.pdf` | 15 | Main manuscript, acmart `acmsmall` review mode |
| `cover_letter.tex` -> `cover_letter.pdf` | 3 | TOMS-specific cover letter |
| `references.bib` | 29 entries | Bibliography (carried from SISC; pruned from 64 to 28 in round-3 audit; +1 in round-4 audit when `guizar2004` was re-added for the Hankel chromatography path). Every entry Crossref-verified or web-triangulated. |
| `figures/` | 12 PNGs | Carried from `paper_sisc_v2/figures/` |
| `data/` | 5 CSVs | Raw benchmark data underpinning every quantitative claim |
| `reproducibility/reproduce.sh` | shell script | One-command reproduction driver |
| `reproducibility/acm_artifact_checklist.md` | markdown | ACM Artifact Available checklist |
| `reviews/simulated_referee_report.md` | markdown | Pre-submission triage (4 majors fixed) |

## The reframe in one paragraph

The SISC submission led with the math (master crossover identity,
factorization theorem, feasibility theorems) and treated the
multi-backend implementation as a benchmark detail. The TOMS
reframe inverts: the `scalpel` library is the contribution, the
multi-backend dispatcher and dispersion-class plugin registry are
the engineering value-add, and the feasibility theorems are demoted
to the mathematical backing for the auto-tuner module that ships
with the library. The crossover-formula companion (`pavlov2026crossover`)
is dropped entirely - it was a distraction for a software audience.

## Title

`scalpel`: A multi-backend FFT-NILT primitive for spectral slab PDE
solvers, with class-dependent finite-precision auto-tuning

## What got fixed during simulated-referee triage

Four major referee findings were addressed inline (see
`reviews/simulated_referee_report.md`):

1. Headline benchmark numbers in the article disagreed with the
   shipped CSV (153x was misattributed to JAX GPU; actually
   PyTorch CPU). Article + cover letter now match the CSV
   row-by-row.
2. The library was thinner than the abstract claimed (Julia is a
   PyJulia sidecar, not a dispatched backend; CuPy was claimed
   unsupported but appeared in benchmark). The article now
   explicitly separates "Dispatched backends" (NumPy/PyTorch/JAX)
   from "Reference wrappers" (CuPy/Julia).
3. `reproducibility/reproduce.sh` was referenced but missing.
   Written and chmod +x'd.
4. The auto-tuner's dispatch mechanism was hand-waved as "symbolic
   inspection." It is now concretely the `dispersion_class` tag on
   the `@register` decorator.

Two minor majors deferred for first-round referee response:

- Six-callable API surface may be flagged as thin; we lead with the
  dispatcher + dispersion registry + auto-tuner as the engineering
  contribution.
- Reproducibility scope is gated on `ENABLE_GPU` and `ENABLE_JULIA`
  env vars in `reproduce.sh`. CPU-only path is fully self-contained.

## Companion-paper framing

Paper 2 (gauge cascade nonlinear) is mentioned as "in preparation"
in the cover letter. The two manuscripts have non-overlapping scope.
The previously published CES paper (10.1016/j.ces.2026.123776)
supplies the scalar NILT parameter-selection rules cited as prior
art.

## Hankel-mode chromatography (post-round-3 addition)

The original Cartesian `propagate_chromatography` used
`KR = sqrt(KX^2 + KY^2)` as a shortcut for axisymmetric sources on a
Cartesian grid. The library now also exports
`propagate_chromatography_hankel`, which routes through the existing
`CylindricalEngine` + quasi-discrete Hankel transform of
Guizar-Sicairos & Gutierrez-Vega (2004; J. Opt. Soc. Am. A
21(1):53-58, DOI 10.1364/JOSAA.21.000053), keeping both modes
available. Closes the last "manuscript-says-cylindrical,
code-is-Cartesian" gap.

Validation: 7 new tests in `tests/test_chromatography_hankel.py`
(public-API import, signature, zero-source, high-Peclet, axisymmetric
agreement smoke check, monotone radial profile with Dr=0). All pass
along with the 6 existing `tests/test_hankel.py` round-trip tests and
the 24 `tests/test_api_expansion.py` tests, i.e. 37/37 across the
three Hankel/API-expansion test files (full repo-wide pytest count
across all 14 test files is larger; this 37 figure is the subset
covering the round-3 + Hankel scope).

Article integration: `tab:library-layout` row for `api.py` now lists
all 5 propagators (Cartesian chromatography + Hankel
chromatography); `tab:bundled-classes` chromatography row notes both
paths; Section 2.2 API tour adds a Hankel code snippet showing the
`HankelTransform(R, N).r` radial grid usage; `guizar2004` re-added
to references.bib (Crossref-verified at round-3 bibliography audit).

## LaTeX class

Built against the **official ACM `acmart.cls` v2.18 (2026-05-31)**,
copied locally from `acmart-primary/` into the bundle root alongside
`ACM-Reference-Format.bst`, `acmnumeric.bbx`, `acmnumeric.cbx`,
`acmauthoryear.bbx`, `acmauthoryear.cbx`. This pins the build to the
ACM current production version rather than the TeX-Live-shipped v2.03
(2024-02), which is two years behind. The local class files travel
with the submission so the ACM portal compiles identically.

## Hardening history (four rounds of adversarial audit)

- **Round 1** (`reviews/simulated_referee_report.md`): 4 majors caught + 4 fixed (benchmark misattribution, library framing, missing reproduce.sh, hand-waved auto-tuner).
- **Round 2** (added to `reviews/simulated_referee_report.md`): 6 majors caught + 6 fixed (article-vs-tree integrity: missing baselines/ dir, missing 2 propagators, missing reference modules, inflated callable count, weak CI matrix, pyproject mismatches).
- **Round 3** (`reviews/simulated_referee_report_round3.md`): 6 majors caught + 6 fixed (broken baseline math, broken baseline imports, reproduce.sh CI failure, ACM checklist broken script paths, theorems iff-vs-if math error, wrong Caputo L1 citation).
- **Round 4** (user-flagged false hardware spec; full claim audit): 9 INCORRECT claims caught + 9 fixed (false CPU/RAM/OS spec, 4 stale LoC/file counts, 1 callable-count off-by-one, 1 stale L1 attribution in baselines README, 2 broken script-path references, 1 STATUS-md stale bib count); 2 AMBIGUOUS claims tightened; 1 UNVERIFIABLE confirmed (benchmark CSV is indeed from this Intel WSL2 box per user); 1 UNVERIFIABLE measured (`reproduce.sh` actually completes in **21 seconds** wall time CPU-only, not the made-up "~30 minutes"); all 29 regenerated CSV rows match the archive within 5% tolerance.
- **Net**: 25 majors/incorrect-claims raised, 25 fixed, 2 explicit stubs documented.

The bundle has no remaining known math-correctness issue, no known
claim-vs-artifact integrity issue, no broken script, and no failing
test. Headline benchmark numbers (8-153x over Yee FDTD, 12-394x
over FTCS+L1, ~5x vs de Hoog, ~13x vs fixed Talbot, ~23x mpmath
accuracy margin) are all derivable row-by-row from
`data/benchmark_multi_backend.csv` and
`data/transform_domain_nilt_comparison.csv`.

## Build commands

```bash
pdflatex article && bibtex article && pdflatex article && pdflatex article
pdflatex cover_letter
```

## ACM Open Access requirement (effective 2026-01-01)

Per the ACM author guidelines, all journal submissions accepted on or
after 2026-01-01 are mandatorily Open Access. The corresponding
author either:

1. **Pays an APC** at acceptance (TOMS 2026 rate not yet posted at
   bundle-prep time; check ACM site at submission), or
2. **Is affiliated with an ACM Open institution**. Check
   <https://libraries.acm.org/acm-open/list-of-participating-institutions>
   for the current list — verify Lehigh's status before submission.

Author defaults to Permission-to-Publish (author owns copyright)
and chooses one of two Creative Commons licenses: CC-BY or
CC-BY-NC-ND. Recommend CC-BY for software-paper visibility
(allows commercial reuse of the bundled software's documentation,
matching the Apache-2.0 of the code itself).

## Zenodo v2 live

The TOMS-submission snapshot is published as Zenodo v2:
**`10.5281/zenodo.20682437`** (deposited 2026-06-13, version 0.2.0,
9.9 MB `scalpel-toms-v2.zip`, title *"scalpel: A multi-backend
FFT-NILT primitive for spectral slab PDE solvers,
with class-dependent finite-precision auto-tuning"*). The v1
record (`10.5281/zenodo.19834321`, deposited 2026-04-27, SISC-era
framing) remains accessible for provenance but is no longer the
artifact-evaluation target.

All manuscript-facing DOI references have been propagated to v2:

- `paper_toms/article.tex` (3 occurrences: abstract, Sec 6, AI-use)
- `paper_toms/cover_letter.tex` (reproducibility paragraph)
- `paper_toms/reproducibility/acm_artifact_checklist.md` (persistent identifier)
- `paper_toms/references.bib` (`pavlov2026recoverability_archive` entry)
- `CITATION.cff` (repo root)
- `paper_toms/zenodo_v2_staging/CITATION.cff`

The Zenodo v2 page lists both `scalpel-toms-v2.zip` (the current
artifact) and the legacy `code.zip` from v1; consider removing
the v1 `code.zip` from the v2 file list on Zenodo to avoid reviewer
confusion (it remains on the v1 record by versioning convention).

## Submission protocol

1. Verify v2 DOI `10.5281/zenodo.20682437` resolves at submission
   time (already confirmed live 2026-06-13).
2. Verify the `reproducibility/reproduce.sh` script runs end-to-end
   on a clean clone before submitting (currently designed against
   the canonical scripts in `reports/claims_audit/`; verify the
   paths resolve).
3. Fill in the ACM Artifact Available checklist's missing fields
   (version tag, dataset checksums).
4. Submit through ACM TOMS portal; select "regular paper" + request
   Artifact Available badge.

## Hold-and-watch

Paper 2 (paper2_sisc/) is fully prepared and can be submitted
elsewhere whenever; venue choice still TBD. Now that paper 1 is
heading to TOMS, paper 2's companion-disclosure can point to "the
\textsc{scalpel} library, in preparation for ACM TOMS" rather than
the now-stale "M188955 under review at SISC."

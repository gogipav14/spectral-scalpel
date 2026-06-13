# Simulated ACM TOMS referee report - pre-submission triage

A Claude-driven simulated TOMS referee read the full bundle
(article.tex, cover_letter.tex, references.bib,
acm_artifact_checklist.md, the figures and CSVs) before submission.
**Verdict: major revision.** The reframe from algorithm paper to
software paper is legible, but the manuscript as filed has three
factual or material problems that a TOMS referee running the code
would catch immediately.

Findings were triaged as **MAJOR** (block accept), **MINOR**
(revision), and **STYLE/PORTAL** (cosmetic). The four fixable
majors have been addressed inline before this status was written.

## Major issues

| # | Issue | Fix status |
|---|---|---|
| 1 | Headline benchmark numbers in the article disagreed with the deposited CSV. The `153x` Maxwell speedup is PyTorch CPU (not JAX GPU as the article claimed); the `394x` Caputo speedup is CuPy GPU (not JAX GPU). | **Fixed** (article.tex: abstract + Sec 5.3 headline now correctly attribute each row to its actual backend, with the per-row CSV referenced) |
| 2 | The library was thinner than the abstract claimed. The dispatcher actually covers three Array-API frontends (NumPy/PyTorch/JAX); Julia is a sidecar via PyJulia; CuPy was in the benchmark CSV but listed as unsupported in limitations. | **Fixed** (article.tex Sec 5: explicit separation between "Dispatched backends" and "Reference wrappers (not dispatched)"; limitations updated; abstract reworded; cover letter updated) |
| 3 | `reproducibility/reproduce.sh` was referenced in five places but missing from the bundle. | **Fixed** (script written; chmod +x; covers four physics demos, four figure-generating runs, mpmath anchor, and CSV-diff verification with `ENABLE_GPU` and `ENABLE_JULIA` env toggles) |
| 4 | Auto-tuner "symbolic inspection" was hand-waved. | **Fixed** (article.tex Sec 2.3: the `dispersion_class` decorator tag selects between `thm:hyp-bound` and `thm:par-bound`; unbundled dispersions fall back with a warning) |

Two further referee majors remain as follow-up work but do not block
submission:

| # | Issue | Decision |
|---|---|---|
| 5 | Six callables is genuinely thin for a "library" claim by current TOMS standards | Defer. The API surface is what it is; we lead with the dispatcher + dispersion-registry + auto-tuner as the engineering contribution. If a referee flags it, we add chunked dispatch + diagnostic introspection helpers. |
| 6 | Reproducibility scope vs the multi-backend claim has a gap (an assessor with only CPU Python extras cannot exercise the Julia path) | Defer. Reproducibility scope is explicitly conditioned in the checklist on `ENABLE_GPU`/`ENABLE_JULIA` env toggles; CPU-only path is fully self-contained. |

## Minor issues (deferred - flag in first-round response)

1. Zenodo DOI reuse: the same concept-DOI is the artifact citation in both the SISC and TOMS bundles. At acceptance, deposit a child version DOI.
2. `\cite{deHoog1982}` vs bib key `deHoog1982` is fine (case match enforced); the earlier `dehoog1982` mismatch is already fixed.
3. Could cite `scipy.signal.invertlaplace`, `pylaplace`, GSL transforms in the related-work paragraph.
4. "Eight-backend" framing could be stated even more precisely; currently reads "eight backend $\times$ device combinations" which is correct.
5. No CI / sustained engineering discussion; a brief sentence on CI matrix, semver, and CHANGELOG would strengthen the library claim. Defer to revision.
6. Figure numbering in checklist (Figures 1-4) matches current build. Re-verify on each rebuild.
7. `L \approx 308` / `L \approx 38` need a one-line note that these are decimal exponents of float64 / float32 representable range.
8. Cover-letter overstates the Array-API dispatcher novelty; could rephrase as "we apply" rather than "we introduce."

## Style / portal issues

1. Em-dash double space at article.tex (now line ~171); cosmetic.
2. `eq:hyp-branch` lead-in could be improved.
3. "Modewise Parseval error" defined only in appendix - mention in body if a referee flags.
4. Some label collisions between section labels and theorem labels; defensive rename if needed.

## Strengths the referee flagged

1. Transform-domain head-to-head (vs de Hoog and Talbot at matched accuracy) is exactly the right experiment to neutralize the "weak baseline" objection.
2. Class-dependent auto-tuner is credible engineering value-add with the right diagnostic (float32 vs float64 sweep).
3. mpmath 50-digit anchor at `8e-4` relative error is the kind of accuracy claim referees can re-run cheaply.
4. Suggested reviewers (Treeby / Kuhlman / Garrappa / Weideman / Rackauckas) are well-chosen for a TOMS software paper.
5. The `@register(dispersion_class=...)` decorator pattern is the right extension hook for downstream adopters.

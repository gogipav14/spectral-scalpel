# PNAS editorial / reviewer pass

Paper: *Conservation-law spectral factorization and a precision-limited
recoverability bound for linear PDE slabs* (Pavlov, submitted 2026-04).
Pass conducted 2026-04-24, post panel-(b) fractional-Caputo addition.
Current state: 6 pages, figures all render, bibliography complete.

## TL;DR editorial assessment

The paper now has two concrete empirical anchors that together defend the
central claim ("spectral factorization wins where time-stepping carries
structural cost"): the Maxwell wet-clay CFL-stiffness panel, and the new
fractional-Caputo panel where the L1 history convolution is the structural
cost the factorization absorbs. The recoverability bound (Eq. 3) remains
the single-strongest theoretical contribution — it is the claim most
likely to drive acceptance at PNAS because it is both novel and
operational (a reviewer can run the inequality and see it work).

**Most likely outcome at desk review:** goes out for review. Not desk
rejection material. The significance statement and abstract are now
concrete enough that the editor will send it.

**Most likely reviewer profile:** one theoretical numerical analyst
(focus on the bound), one computational-physics practitioner (focus on
the benchmarks), one application-domain expert (GPR, ultrasound, or
chromatography — most likely whoever the editor knows).

**Biggest vulnerabilities**, in descending order:
1. The FDTD baseline looks overdetermined — a reviewer will suggest
   comparing to k-Wave or an optimized column-parallel FDTD.
2. "Certified" is a strong word; expect pushback on whether the
   feasibility check really certifies numerical correctness.
3. Reference solutions for the fractional benchmark are self-consistency
   at two NILT resolutions, not an analytical reference. A reviewer may
   call this circular.

## Reviewer angle 1: Novelty

The factorization itself (Eq. 1) is not new. The paper acknowledges
this (`line 48: "not a new PDE method but a sharp instantiation of a
textbook factorization"`). That hedging is correct and should stay.

The **recoverability bound** (Eq. 3) is the defensible novelty. Writing
a branch-point-abscissa inequality and tying it to the floating-point
exponent range closes a gap in the NILT literature that has mostly
treated precision empirically. The Shannon-Hartley analogy is a rhetoric
device the paper explicitly flags as rhetoric (`line 50: "the analogy is
a convenience"`). Keep it that way — do not let the analogy become the
claim.

**Reviewer will ask:** "Is this bound truly new? Cite Abate-Whitt 2006,
Kuhlman 2013, Stehfest 1970, Weideman 2006 more prominently and
distinguish."

**Response:** the paper cites Abate 2006 and Weideman 2006, but it
could add one sentence distinguishing. Suggested addition after Eq. 3:

> "Prior work on NILT precision (Abate-Whitt 2006, Kuhlman 2013) focuses
> on the scalar-inversion error bounds for individual transforms; the
> bound here is a *transverse-wavenumber* ceiling derived from the
> branch-point structure of the modewise transfer function, and applies
> uniformly to any frequency-synthesis solver whose observation window
> multiplies by an $e^{a t_{\max}}$ rescaling factor."

**Word budget concern:** this adds ~45 words; may need to cut one
sentence elsewhere. Acceptable to skip for first submission and add in
revision.

## Reviewer angle 2: Baselines and benchmarks

### Strengths of current benchmarks

- Panel (a): across 8 backends the factorization wins $27\text{-}153\times$.
  The cross-backend spread is a strength, not a weakness: it shows the
  speedup is structural, not language- or compiler-specific.
- Panel (b): the fractional Caputo case directly answers "why not just
  use FFT?" The FTCS+L1 baseline cannot be replaced by a pure-Fourier
  closed form because $s^\alpha$ has no exponential time-propagator.
- Panel (b) result ($12\text{-}394\times$) spans two orders of magnitude
  across backends, reinforcing the structural-advantage story.

### Vulnerabilities

**Baseline critique for panel (a):** reviewers familiar with computational
electromagnetics will point out that a 3D Yee FDTD on a purely
homogeneous slab is not what anyone actually runs; optimized 1D column
solvers or analytical Green's functions are standard. The paper's
defense is in text (outer-loop embedding) — make sure that defense is
prominent. **Current placement is good** (opening of Performance section,
repeated in figure caption).

**Reviewer will still ask:** "Compare to k-Wave"
[treeby2010, already cited]. Either include k-Wave in the SI or state
explicitly in the Performance section that k-Wave is a superset
(pseudospectral *with* CFL-constrained time-stepping and thus subject
to numerical dispersion on the time grid), referring to the SI
discussion [line 139 refers to "SI Sec. 3"].

**Fractional reference:** the "60× smaller relative error" claim uses
scalpel @ N_NILT=4096 as a self-consistency reference. A reviewer will
flag this as circular (scalpel compared against scalpel).

**Mitigation options:**
- **Strong:** implement a Mittag-Leffler propagator on a periodic domain
  as an analytical reference. Well-studied [Garrappa 2015]. ~200 LOC.
  Do this in revision if asked.
- **Weak:** acknowledge the limitation in the benchmark caption with
  one sentence: "Reference is a high-precision (N_NILT=4096) NILT
  self-consistency comparison; an analytical Mittag-Leffler reference
  is deferred to the revised manuscript."
- **Practical:** for submission, rely on the fact that FTCS+L1 itself
  serves as an independent reference — if scalpel and FTCS+L1 agree,
  both are correct; if they disagree and FTCS+L1 has known error
  $\mathcal{O}(\Delta t^{2-\alpha})$, the discrepancy can be bounded.
  Document this argument in SI Sec. 3.

**Recommendation:** add two sentences to SI Sec. 3 mapping out the
error-argument chain for fractional, and note in main-text figure
caption that fractional accuracy is bounded by FTCS+L1 convergence.

## Reviewer angle 3: Certification language

The word "certified" appears repeatedly (`abstract, significance, line
48, line 161`). In formal-methods and program-verification contexts,
"certified" means machine-checked proof. Here it means "the feasibility
inequality flags infeasible modes before inversion."

**Reviewer push:** "Is the solver actually certified, or is it
diagnosed?"

**Mitigation:** consider substituting "certified" → "audited" or
"precision-classified" in one or two spots. Keeping the word in
"certified forward solve" as a defined term (preceded by scare quotes
or italicization on first use) is acceptable; then use the shorter
form throughout.

**Minimal edit:** first occurrence in Results section 1 (`line 48,
"every modewise NILT carries an a posteriori quality diagnostic tied
to its arithmetic precision"`) already avoids "certified" — good. The
abstract uses "certified" implicitly via the feasibility-and-Parseval
clause, which is fine because the claim is about mode flagging, not
proof. **No change strictly required**, but an editorial review-pass
would do one global substitution to soften.

## Reviewer angle 4: Presentation and readability

### Significance statement
Currently 113 words (within PNAS 120-word limit). Dense but readable for a
PNAS audience. **No change needed**.

### Abstract
Now 244 words after trim (within PNAS 250-word limit). Mentions both
benchmarks and the fractional extension. **No change needed**.

### Figure placement
Four `figure*` environments (full-width) plus one `figure` (column-width)
across 6 pages. Tight but valid. The benchmark figure as column-width
`figure` with vertical stack works at current layout; if the editor
requests all figures `figure*`, the draft will blow the page limit and
some body content must be cut — identify candidates now (see below).

### Jargon watch
- "NILT" is used without expansion after Sec. 1. Fine for PNAS Applied
  Physical Sciences; would not fly in a broader-readership section.
- "Bromwich contour" appears 5 times. Well-defined in the opening
  equation. Good.
- "L1 scheme" is defined in Methods [line 174 addition]. Good — readers
  unfamiliar with fractional-time discretization get the formula.

### Units
The abstract says "$t_{\text{end}} = 20\,$ms" but Methods has L1 params
at $D = 10^{-2}\,$m$^2$/s. Consistent. Physical parameters for wet clay
($\sigma = 0.1\,$S/m, $\epsilon_r = 10$) appear in benchmarks but not
Methods — **recommendation:** add one line to the NILT tuning subsection
with the canonical wet-clay constants.

## Concrete strengthening: prioritized

Ranked by (impact on acceptance) / (cost in words and time).

### Tier 1: must-do before submission
1. **Verify 6-page compile** after edits. Done.
2. **Add L1 scheme formula to Methods.** Done.
3. **Add Caputo row to Table 1.** Done.
4. **Update figure caption for two-panel story.** Done.

### Tier 2: do if word budget allows
5. **Distinguish recoverability bound from prior NILT precision work
   (Abate-Whitt, Kuhlman).** Add ~45-word sentence after Eq. 3.
6. **Add a mention of k-Wave head-to-head in SI Sec. 3.** Not in main
   text; just a forward pointer in Performance section.
7. **Mitigation sentence for fractional self-consistency reference.**
   One sentence in figure caption or SI.

### Tier 3: defer to revision
8. Implement Mittag-Leffler analytical reference for fractional.
9. Add a CN+L1 implicit baseline as a third bar per group in panel (b).
10. Dispersive-Maxwell (Cole-Cole) ADE-FDTD comparison — already
    documented as followup in
    `reports/claims_audit/option_b_cole_cole_maxwell_followup.md`.

## Anticipated reviewer pushback with rebuttal sketches

### "Your FDTD baseline is a straw man."
Rebuttal: the comparison is structural, not optimal-practice. FDTD is
the forward kernel invoked inside a heterogeneous outer loop, where
the slab's local homogeneity cannot be exploited globally. The paper
is explicit about this (Performance section, figure caption). The
fractional panel (b) is a separate regime where no closed-form
FFT-spectral alternative exists, reinforcing the non-straw-man framing.

### "Eq. 3 is known."
Rebuttal: the *scalar* NILT precision bound is known; the
*multi-dimensional* $k_\perp$-ceiling derived from the branch-point
abscissa of the modewise transfer function is not (cite Abate-Whitt
2006 and Kuhlman 2013 as the nearest prior; they bound scalar inversion
error but do not map to the transverse-wavenumber ceiling).

### "Why not use an exponential-integrator / Mittag-Leffler method?"
Rebuttal (main text): Mittag-Leffler propagators work for periodic
linear fractional diffusion but not for half-space / finite-slab /
multi-material geometries with non-trivial dispersion; the
factorization handles these uniformly. Add to SI Sec. 3.

### "Is the precision bound tight?"
Rebuttal: Fig. 4(a) shows the a-posteriori indicator collapses in the
reduced coordinate across three systems, and Fig. 4(b) shows the float32
vs float64 measured ratio matches Eq. 3's predicted ratio to three
significant figures. This is empirical tightness; theoretical tightness
would require a matching lower bound (reviewer asked, "if $k_\perp =
k_{\perp,\max} - \epsilon$, is reconstruction always feasible?") — this
is open for the paper as written.

### "What about multi-GPU scaling?"
Rebuttal: out of scope for the present paper (single-RTX-5060 consumer
baseline intentionally chosen as worst-case). Mention in SI or defer
to follow-up.

### "Would a reviewer in the chemistry audience (chromatography)
understand Eq. 3?"
Yes — the bound is stated as $k_{\perp, \max}(L, t_{\max})$ with clear
physical parameters ($L$ is the floating-point exponent range, $t_{\max}$
is the observation window). The Hankel-basis variant in cylindrical
chromatography is explicitly noted to have the same form with
$k_\perp$ replaced by the radial Bessel eigenvalue.

## Checklist for submission

- [x] Abstract < 250 words
- [x] Significance statement < 120 words
- [x] Main text 6 pages including figures and references
- [x] Bibliography entries complete for all `\cite{}` calls
- [x] Figures all render correctly at column or double-column width
- [x] Benchmark figure shows all backends (8)
- [x] Benchmark data reproducible via
      `reports/claims_audit/figure_benchmark_data.csv`
- [x] L1 scheme formula in Methods
- [x] Caputo row in Table 1
- [ ] SI Sec. 3 updated with k-Wave and fractional-reference discussion
      (tier-2, pre-submission if time)
- [ ] Data availability statement final-URL updated (currently Zenodo
      placeholder; replace with the actual DOI at submission time)
- [ ] Cover letter mentions the parallel Paper 2 submission status per
      `project_paper1_venue_pivot.md` memory

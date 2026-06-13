# Parallel submission: Paper 2 to NC while Paper 1 is in review at NCS

## Short answer

**Yes, you can submit Paper 2 to Nature Communications while Paper 1 is in review at Nature Computational Science.** But there are specific mechanics and disclosures required, and a few risks to manage.

## Nature Portfolio policy

The Nature Portfolio "duplicate and redundant submission" policy (applies to all Nature journals including NC and NCS) prohibits:
- Submitting the **same** manuscript to two journals simultaneously.
- Redundant publication of the same results.

It does **not** prohibit:
- Submitting **distinct** papers to different journals in parallel.
- Cross-citing papers under review at other journals (standard practice).
- Companion-paper structures where two papers build on each other.

Our situation: Paper 1 (linear PDE spectral factorization) and Paper 2 (nonlinear PDE gauge cascade) are **structurally distinct**:
- Paper 1's core object: the pencil $\gamma_z(s, k_\perp)$ for linear PDEs, established via transverse Fourier + Laplace factorization.
- Paper 2's core object: the gauge transform $\Phi$ and hypothesis (G), which unify nonlinear PDEs under a cascade that *uses* Paper 1's NILT machinery as a component.

Paper 2 cites Paper 1 but is self-contained; a reader with no prior exposure can understand Paper 2 on its own (Paper 1's relevant bits — the NILT feasibility bound — fit in a single paragraph in Paper 2's Methods).

**This is the standard companion-paper structure, permitted by Nature Portfolio policy.**

## Required disclosures

In the NC cover letter (and in the submission portal's "related work" field):

1. **Disclose Paper 1's NCS submission.** State: "An earlier companion paper (Pavlov, *Nature Computational Science*, in review) establishes the linear-regime spectral-scalpel framework that Paper 2 extends to nonlinear PDEs via gauge transforms." — we already have this in the current cover letter.

2. **Declare absence of redundancy.** Add a sentence: "The two papers have no textual or methodological overlap beyond a shared citation to the NILT feasibility bound; Paper 2's theorem, proof, and empirical demonstrations are independent contributions."

3. **Update the cover letter with concrete wording.** Sample paragraph (drop into the current cover letter):

> "**Relationship to work under review.** I have an earlier paper (title: *Conservation-law spectral factorization for PDE systems*) under review at *Nature Computational Science* that establishes the linear-regime framework this submission extends. The two papers are distinct contributions: Paper 1 treats linear PDEs, while Paper 2 addresses nonlinear PDEs via gauge transforms. There is no textual or empirical overlap. Paper 2 can be evaluated independently, and its results do not depend on Paper 1's acceptance. I can provide the companion paper to reviewers upon request."

## Risks and mitigations

### Risk 1: Overlapping reviewers

NC and NCS editors sometimes use the same reviewer for related submissions. If the same reviewer sees both papers, they may:
- Request consolidation (reject one, suggest merging).
- Review one more harshly because they're bored of the topic.
- Cross-contaminate their assessments.

**Mitigation:** in NC's referee-suggestion field, recommend referees whose expertise matches Paper 2's specific contributions (gauge theory, nonlinear PDE numerics) rather than the linear-PDE specialists who would have been suggested for Paper 1. Different referee pools reduce overlap.

### Risk 2: Editorial deferral

NC editor may say "this depends on the NCS paper — we'll wait for that decision before reviewing." This delays but doesn't damage the submission.

**Mitigation:** emphasize Paper 2's standalone validity in the cover letter. If the editor defers, convert into a pre-submission inquiry that declares both papers, lets the editor decide handling order.

### Risk 3: Inconsistent decisions

If Paper 1 is rejected and Paper 2 is accepted, Paper 2's citation to "Paper 1 under review at NCS" becomes awkward (Paper 1 would need to resubmit somewhere and the Paper 2 final version would update the citation). If Paper 1 is accepted and Paper 2 is rejected, no harm — Paper 2 can be improved and resubmitted to SISC or JCP.

**Mitigation:** Paper 2 should stand alone. The cover letter should say: "Paper 2 cites Paper 1 but does not depend on Paper 1's acceptance for its results." — we have this.

### Risk 4: Perceived self-inflation

Two simultaneous Nature-portfolio submissions from one author is an unusual move and could be perceived as attempt to maximize visibility. Some editors react negatively to this.

**Mitigation:** keep the cover letter matter-of-fact. No superlatives. Frame as a natural consequence of the two-paper structure. Single-author status is straightforward here (no co-author politics).

## Timing considerations

**If you submit Paper 2 today (2026-04-19):**
- NC first decision: typically 5–8 weeks.
- If invited for peer review: 12–16 weeks total.
- If rejected at triage: ~2 weeks.

**NCS Paper 1 status (submitted ~2026-04-01, so ~18 days in):**
- Past desk-reject threshold (~2 weeks for NCS).
- Likely in peer review or just about to enter it.
- First decision: typically 8–12 weeks from submission, so expect mid-to-late May.

**Scenarios:**

| Paper 1 decision | Paper 2 status | Action |
|---|---|---|
| Accept (likely with major revisions) | Paper 2 pre-review | Update Paper 2's citation, continue |
| Major revision | Paper 2 in review | Revise Paper 1, continue Paper 2 independently |
| Reject (desk or after review) | Paper 2 pre-review | Paper 2 still fine; citation becomes "in preparation" |
| Reject | Paper 2 accepted | Resubmit Paper 1 elsewhere; Paper 2 updates refs in proof |

All four scenarios are manageable. Parallel submission does not create a downside scenario that staggered submission avoids.

## Recommendation

**Submit Paper 2 to NC in parallel. Do it after two additional steps:**

1. **Confirm the slope-2 result with the sanity checks running now** (4 independent tests). If even one check fails, hold Paper 2 for investigation.

2. **Update cover letter** with the "Relationship to work under review" paragraph I drafted above.

After both are done, submit within 1–2 days. Parallel submission is the correct strategy given the independent contributions and the ~6-week timing window before Paper 1's first decision.

## One non-obvious benefit of parallel submission

If both papers are under review simultaneously, editors may coordinate. This can actually help:
- An editor at NC can ask the NCS editor about Paper 1's status.
- Positive signals at one journal can influence the other (if you gave NCS editor permission to share with NC editor).
- A companion-paper acceptance at both journals creates a stronger publication narrative than serial placement.

Not guaranteed, but realistic.

## What's in the current submission package

All three PDFs ready at `paper2/`:
- `gauge_cascade.pdf` (9 pp)
- `supplementary.pdf` (7 pp)
- `cover_letter.pdf` (2 pp) — will be updated with the "related work" paragraph above before submission.

Pending: slope-2 sanity checks (running; results in ~10 min).

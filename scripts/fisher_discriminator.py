"""
Fisher-information discriminator for the quotient representation.

Computes, for every labelled validation batch e (engine, laboratory, ...),

    paper score      Psi_e^paper = mean_i [ (K_i b_e)^2 - C_i ]
    corrected score  Psi_e^corr  = mean_i [ (K_i b_e)^2
                                            - K_i (Sigma_nu + sigma^2/n_e I) K_i'
                                            - C_i ]

with, at each validation observation i (local Jacobian J_i = dm/dq),

    a_i  = A' J_i
    I_P  = ||J_i - A a_i||^2 / sigma^2
    I_Y  = I_P + a_i' (Sigma_nu + sigma^2 I)^{-1} a_i
    C_i  = 1/I_P - 1/I_Y
    K_i  = (Sigma_nu + sigma^2 I)^{-1} a_i / I_Y
    b_e  = A' (mean residual of batch e - mean training centroid)

Why the correction
------------------
For the nominally efficient raw estimator, q_hat - q = K nu + xi with
Var(xi) = 1/I_Y - K Sigma_nu K'. Conditional on a batch with nuisance nu_e,

    R_Y(nu_e) - R_P = (K nu_e)^2 - K Sigma_nu K' - C,

and E[(K b_e)^2 | nu_e] = (K nu_e)^2 + (sigma^2 / n_e) ||K||^2.
Psi_e^corr is therefore unbiased for the batch's risk gap, and its average over
batches with nu_e ~ N(b, Sigma_nu) equals Psi(b) of Proposition 4.3.
Psi_e^paper (Section 5.5 / 5.10 of the draft) is biased upward by
K (Sigma_nu + sigma^2/n_e I) K'.

Uncertainty
-----------
`within` intervals resample observations inside each validation batch.
`batch` intervals resample validation batches (meaningless with <5 batches).
Neither propagates the uncertainty of the training calibration (A, Sigma_nu,
sigma^2); a full bootstrap that refits ResidualBatchPCA is the honest version.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class DiscriminatorResult:
    per_batch: dict
    macro: dict
    calibration: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"rank={self.calibration['rank']}  sigma2={self.calibration['sigma2']:.4g}  "
            f"Sigma_nu diag={np.round(np.diag(self.calibration['Sigma_nu']), 4).tolist()}",
            f"{'batch':<22}{'n':>7}{'G':>12}{'C':>12}{'bias':>12}"
            f"{'Psi_paper':>12}{'Psi_corr':>12}{'corr 95% CI':>28}",
        ]
        for b, r in self.per_batch.items():
            lines.append(
                f"{str(b):<22}{r['n']:>7d}{r['G']:>12.4g}{r['C']:>12.4g}{r['bias']:>12.4g}"
                f"{r['psi_paper']:>12.4g}{r['psi_corr']:>12.4g}"
                f"{'[%.4g, %.4g]' % (r['psi_corr_lo'], r['psi_corr_hi']):>28}"
            )
        m = self.macro
        lines.append(
            f"{'MACRO':<22}{'':>7}{m['G']:>12.4g}{m['C']:>12.4g}{m['bias']:>12.4g}"
            f"{m['psi_paper']:>12.4g}{m['psi_corr']:>12.4g}"
            f"{'[%.4g, %.4g]' % (m['psi_corr_lo'], m['psi_corr_hi']):>28}"
        )
        lines.append(
            f"macro paper-score CI: [{m['psi_paper_lo']:.4g}, {m['psi_paper_hi']:.4g}]   "
            f"decision(paper)={m['decision_paper']}   decision(corrected)={m['decision_corr']}"
        )
        return "\n".join(lines)


def calibration_from_rbpca(rbpca, batch_train, subtract_centroid_noise=True):
    """Sigma_nu and sigma^2 from the same residuals Residual Batch PCA used."""
    A = rbpca.A_hat_
    res = rbpca.residuals_
    batches = rbpca.batches_
    batch_train = np.asarray(batch_train)
    within = []
    for b, cen in zip(batches, rbpca.batch_means_raw_):
        rb = res[batch_train == b]
        if len(rb) > 1:
            within.append(((rb - cen) ** 2).sum() / ((len(rb) - 1) * res.shape[1]))
    sigma2 = float(np.mean(within))       # equal-batch average of within-batch variance
    z = rbpca.centered_batch_means_ @ A
    r = A.shape[1]
    S = np.atleast_2d(np.cov(z.T, ddof=1)).reshape(r, r)
    if subtract_centroid_noise:
        S = S - sigma2 * np.mean(1.0 / rbpca.batch_sizes_) * np.eye(r)
    w, V = np.linalg.eigh(S)
    S = (V * np.maximum(w, 1e-10)) @ V.T
    return dict(A=A, Sigma_nu=S, sigma2=sigma2, rank=r,
                centroid_mean=rbpca.batch_means_.mean(0))


def _local_terms(J, cal):
    A, S, s2 = cal["A"], cal["Sigma_nu"], cal["sigma2"]
    a = J @ A
    PJ = J - a @ A.T
    IP = (PJ ** 2).sum(1) / s2
    Sinv = np.linalg.inv(S + s2 * np.eye(cal["rank"]))
    IY = IP + np.einsum("ni,ij,nj->n", a, Sinv, a)
    K = (a @ Sinv) / IY[:, None]
    C = 1.0 / IP - 1.0 / IY
    return K, C


def _batch_score(K, C, r_mean, cal, n):
    b = cal["A"].T @ (r_mean - cal["centroid_mean"])
    K2 = (K ** 2).sum(1)
    Kb2 = (K @ b) ** 2
    M = cal["Sigma_nu"] + cal["sigma2"] / n * np.eye(cal["rank"])
    bias = np.einsum("ni,ij,nj->n", K, M, K)
    return dict(G=Kb2.mean(), C=C.mean(), bias=bias.mean(), K2=K2.mean(),
                psi_paper=(Kb2 - C).mean(), psi_corr=(Kb2 - bias - C).mean())


def fisher_discriminator(
    rbpca,
    X_forward_train: np.ndarray,
    batch_train: np.ndarray,
    design_at: Callable[[np.ndarray], np.ndarray],
    q_valid: np.ndarray,
    Y_valid: np.ndarray,
    batch_valid: np.ndarray,
    q_range: tuple[float, float],
    sample_weight_train: np.ndarray | None = None,
    Y_train: np.ndarray | None = None,
    h: float = None,
    n_boot: int = 2000,
    random_state: int = 0,
    subtract_centroid_noise: bool = True,
) -> DiscriminatorResult:
    """
    Parameters
    ----------
    rbpca : fitted ResidualBatchPCA.
    X_forward_train : forward design used to fit rbpca (rows of the training set).
    design_at : callable q -> forward design rows for the validation observations
        (same row order as q_valid), e.g. ``lambda q: design.transform(q, settings_valid)``.
    q_range : (min, max) of the training target, used to keep finite differences
        inside the fitted spline range (important for capped RUL).
    """
    rng = np.random.default_rng(random_state)
    cal = calibration_from_rbpca(rbpca, batch_train, subtract_centroid_noise)

    model = getattr(rbpca, "forward_model_", None)
    if model is None:  # cross-fitted calibration: one full-data forward fit for J and residuals
        if Y_train is None:
            raise ValueError("rbpca was cross-fitted; pass Y_train so the forward model can be refitted.")
        model = Ridge(alpha=rbpca.ridge_alpha, fit_intercept=False)
        model.fit(X_forward_train, rbpca.scaler_.transform(np.asarray(Y_train, float)),
                  sample_weight=sample_weight_train)
    B = model.coef_                                     # (m, p)

    q_valid = np.asarray(q_valid, float).ravel()
    lo_q, hi_q = q_range
    if h is None:
        h = 1e-3 * (hi_q - lo_q)
    # central differences evaluated inside the fitted target range; observations
    # outside it (or at a cap, e.g. RUL = 125) use the derivative at the boundary
    q_c = np.clip(q_valid, lo_q + h, hi_q - h)
    J = (design_at(q_c + h) - design_at(q_c - h)) @ B.T / (2 * h)

    Zv = rbpca.scaler_.transform(np.asarray(Y_valid, float))
    resid = Zv - design_at(q_valid) @ B.T
    K, C = _local_terms(J, cal)

    batch_valid = np.asarray(batch_valid)
    per_batch, boot_paper, boot_corr = {}, [], []
    for b in np.unique(batch_valid):
        idx = np.flatnonzero(batch_valid == b)
        n = len(idx)
        s = _batch_score(K[idx], C[idx], resid[idx].mean(0), cal, n)
        bs = np.empty((n_boot, 2))
        for t in range(n_boot):
            j = idx[rng.integers(0, n, n)]
            sb = _batch_score(K[j], C[j], resid[j].mean(0), cal, n)
            bs[t] = sb["psi_paper"], sb["psi_corr"]
        s.update(n=n,
                 psi_paper_lo=np.percentile(bs[:, 0], 2.5), psi_paper_hi=np.percentile(bs[:, 0], 97.5),
                 psi_corr_lo=np.percentile(bs[:, 1], 2.5), psi_corr_hi=np.percentile(bs[:, 1], 97.5))
        per_batch[b] = s
        boot_paper.append(bs[:, 0]); boot_corr.append(bs[:, 1])

    keys = ["G", "C", "bias", "psi_paper", "psi_corr", "K2"]
    macro = {k: float(np.mean([r[k] for r in per_batch.values()])) for k in keys}
    mp, mc = np.mean(boot_paper, 0), np.mean(boot_corr, 0)   # equal-batch macro, within-batch bootstrap
    macro.update(psi_paper_lo=np.percentile(mp, 2.5), psi_paper_hi=np.percentile(mp, 97.5),
                 psi_corr_lo=np.percentile(mc, 2.5), psi_corr_hi=np.percentile(mc, 97.5))
    vals = np.array([r["psi_corr"] for r in per_batch.values()])
    if len(vals) >= 5:
        bb = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(1)
        macro.update(psi_corr_batch_lo=np.percentile(bb, 2.5), psi_corr_batch_hi=np.percentile(bb, 97.5))

    def decide(lo, hi):
        return "deploy quotient" if lo > 0 else ("keep raw" if hi < 0 else "inconclusive")
    macro["decision_paper"] = decide(macro["psi_paper_lo"], macro["psi_paper_hi"])
    macro["decision_corr"] = decide(macro["psi_corr_lo"], macro["psi_corr_hi"])
    return DiscriminatorResult(per_batch=per_batch, macro=macro, calibration=cal)


def worst_case(result: DiscriminatorResult, rho: float) -> dict:
    """Eq. (15)-(16) for a scalar target: Psi_wc(rho) = rho^2 E||K||^2 - E C.

    rho is a bound on the deployment nuisance-mean shift in the estimated
    nuisance coordinates (standardized units). It must come from prior
    knowledge, not from the validation batches, or it adds nothing over Psi_corr.
    """
    K2, C = result.macro["K2"], result.macro["C"]
    return dict(psi_wc=rho ** 2 * K2 - C, rho_crit=float(np.sqrt(C / K2)))

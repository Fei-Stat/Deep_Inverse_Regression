from __future__ import annotations

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


class ResidualBatchPCA:
    """
    Residual Batch PCA for estimating a candidate nuisance subspace.

    This class is intentionally dataset-agnostic.

    Dataset-specific code must construct:
      * Y: high-dimensional measurements, shape (n_samples, n_features)
      * X_forward: design matrix for estimating E[Y | q, x]
      * batch: nuisance-group labels

    Examples
    --------
    FD004:
        Y = 63-dimensional windowed sensor representation
        X_forward = spline(RUL) + settings + regimes
                    + spline(RUL) x regimes
        batch = engine_id

    OSSL:
        Y = 426-dimensional MIR spectrum
        X_forward = cubic spline(organic carbon)
        batch = source/laboratory

    Residual Batch PCA only estimates a candidate nuisance subspace.
    Whether to remove it is a separate deployment decision.
    """

    def __init__(
        self,
        rank: int | None = None,
        variance_threshold: float = 0.90,
        ridge_alpha: float = 1.0,
        crossfit: str = "group_kfold",
        n_splits: int = 5,
    ):
        if rank is not None and int(rank) < 1:
            raise ValueError("rank must be None or a positive integer.")
        if not (0.0 < float(variance_threshold) <= 1.0):
            raise ValueError("variance_threshold must lie in (0, 1].")
        if float(ridge_alpha) < 0:
            raise ValueError("ridge_alpha must be nonnegative.")
        if crossfit not in {"group_kfold", "leave_one_batch_out", "none"}:
            raise ValueError(
                "crossfit must be 'group_kfold', 'leave_one_batch_out', or 'none'."
            )
        if int(n_splits) < 2:
            raise ValueError("n_splits must be at least 2.")

        self.rank = rank
        self.variance_threshold = float(variance_threshold)
        self.ridge_alpha = float(ridge_alpha)
        self.crossfit = crossfit
        self.n_splits = int(n_splits)

    @staticmethod
    def _validate_inputs(Y, X_forward, batch, sample_weight=None):
        Y = np.asarray(Y, dtype=float)
        X_forward = np.asarray(X_forward, dtype=float)
        batch = np.asarray(batch)

        if Y.ndim != 2:
            raise ValueError("Y must have shape (n_samples, n_features).")
        if X_forward.ndim != 2:
            raise ValueError(
                "X_forward must have shape (n_samples, n_forward_features)."
            )

        n = Y.shape[0]
        if X_forward.shape[0] != n or len(batch) != n:
            raise ValueError(
                "Y, X_forward, and batch must contain the same number of rows."
            )

        if not np.all(np.isfinite(Y)):
            raise ValueError("Y contains NaN or infinite values.")
        if not np.all(np.isfinite(X_forward)):
            raise ValueError("X_forward contains NaN or infinite values.")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(sample_weight) != n:
                raise ValueError(
                    "sample_weight must have one value per observation."
                )
            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight contains NaN or infinite values.")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be nonnegative.")
            if sample_weight.sum() <= 0:
                raise ValueError("sample_weight must have positive total weight.")

        return Y, X_forward, batch, sample_weight

    def _new_forward_model(self):
        # X_forward should contain an intercept column if one is desired.
        return Ridge(alpha=self.ridge_alpha, fit_intercept=False)

    @staticmethod
    def _fit_ridge(model, X, Z, sample_weight=None):
        if sample_weight is None:
            model.fit(X, Z)
        else:
            model.fit(X, Z, sample_weight=sample_weight)
        return model

    def _make_residuals(
        self,
        Z,
        X_forward,
        batch,
        sample_weight,
    ):
        residuals = np.empty_like(Z)
        unique_batches = np.unique(batch)
        n_batches = len(unique_batches)

        if self.crossfit == "none":
            model = self._new_forward_model()
            self._fit_ridge(
                model,
                X_forward,
                Z,
                sample_weight=sample_weight,
            )
            residuals[:] = Z - model.predict(X_forward)
            self.forward_model_ = model
            self.crossfit_splits_ = None
            return residuals

        if self.crossfit == "leave_one_batch_out":
            splits = []

            for held_out_batch in unique_batches:
                train_index = batch != held_out_batch
                held_out_index = batch == held_out_batch

                sw_train = (
                    None
                    if sample_weight is None
                    else sample_weight[train_index]
                )

                model = self._new_forward_model()
                self._fit_ridge(
                    model,
                    X_forward[train_index],
                    Z[train_index],
                    sample_weight=sw_train,
                )

                residuals[held_out_index] = (
                    Z[held_out_index]
                    - model.predict(X_forward[held_out_index])
                )

                splits.append(
                    (
                        np.flatnonzero(train_index),
                        np.flatnonzero(held_out_index),
                    )
                )

            self.forward_model_ = None
            self.crossfit_splits_ = splits
            return residuals

        # group_kfold
        if n_batches < 2:
            raise ValueError(
                "Grouped cross-fitting requires at least two unique groups."
            )

        n_splits = min(self.n_splits, n_batches)
        splitter = GroupKFold(n_splits=n_splits)
        splits = []

        for train_index, held_out_index in splitter.split(
            X_forward,
            groups=batch,
        ):
            sw_train = (
                None
                if sample_weight is None
                else sample_weight[train_index]
            )

            model = self._new_forward_model()
            self._fit_ridge(
                model,
                X_forward[train_index],
                Z[train_index],
                sample_weight=sw_train,
            )

            residuals[held_out_index] = (
                Z[held_out_index]
                - model.predict(X_forward[held_out_index])
            )

            splits.append((train_index.copy(), held_out_index.copy()))

        self.forward_model_ = None
        self.crossfit_splits_ = splits
        return residuals

    def fit(
        self,
        Y,
        X_forward,
        batch,
        sample_weight=None,
    ):
        """
        Fit Residual Batch PCA using training observations only.

        sample_weight is used only in the forward ridge regression.
        It is useful for OSSL equal-source weighting.
        """
        Y, X_forward, batch, sample_weight = self._validate_inputs(
            Y,
            X_forward,
            batch,
            sample_weight=sample_weight,
        )

        n, m = Y.shape

        # 1. Training-only standardization.
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(Y)

        self.n_features_in_ = m
        self.n_samples_fit_ = n

        # 2. Forward residuals.
        self.batches_ = np.unique(batch)
        K = len(self.batches_)

        if K < 3:
            raise ValueError(
                "At least three training nuisance groups are required."
            )

        residuals = self._make_residuals(
            Z=Z,
            X_forward=X_forward,
            batch=batch,
            sample_weight=sample_weight,
        )
        self.residuals_ = residuals

        # 3. One residual centroid per nuisance group.
        batch_means = np.vstack(
            [
                residuals[batch == b].mean(axis=0)
                for b in self.batches_
            ]
        )

        batch_mean_global = batch_means.mean(axis=0, keepdims=True)
        centered_batch_means = batch_means - batch_mean_global

        self.batch_means_ = batch_means
        self.batch_mean_global_ = batch_mean_global.reshape(-1)
        self.centered_batch_means_ = centered_batch_means

        # 4. Ordinary PCA via SVD.
        _, singular_values, Vt = np.linalg.svd(
            centered_batch_means,
            full_matrices=False,
        )

        eigenvalues = singular_values ** 2
        total = eigenvalues.sum()

        if total > 0:
            explained_ratio = eigenvalues / total
        else:
            explained_ratio = np.zeros_like(eigenvalues)

        cumulative_ratio = np.cumsum(explained_ratio)

        numerical_rank = np.linalg.matrix_rank(centered_batch_means)
        maximum_rank = min(K - 1, m, numerical_rank)

        if maximum_rank < 1:
            raise ValueError(
                "No nonzero residual between-batch direction was found."
            )

        # 5. Candidate nuisance rank.
        if self.rank is None:
            selected_rank = (
                np.searchsorted(
                    cumulative_ratio,
                    self.variance_threshold,
                    side="left",
                )
                + 1
            )
        else:
            selected_rank = int(self.rank)

        selected_rank = min(selected_rank, maximum_rank)

        self.selected_rank_ = selected_rank
        self.maximum_rank_ = maximum_rank
        self.singular_values_ = singular_values
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ratio_ = explained_ratio
        self.cumulative_explained_variance_ = cumulative_ratio

        # 6. Nuisance basis and quotient projector.
        self.A_hat_ = Vt[:selected_rank].T
        self.P_hat_ = (
            np.eye(m)
            - self.A_hat_ @ self.A_hat_.T
        )

        return self

    def transform_raw(self, Y):
        """Return measurements standardized by the fitted training scaler."""
        if not hasattr(self, "scaler_"):
            raise RuntimeError("Call fit() before transform_raw().")

        Y = np.asarray(Y, dtype=float)

        if Y.ndim != 2 or Y.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Y must have shape (n_samples, {self.n_features_in_})."
            )

        return self.scaler_.transform(Y)

    def transform(self, Y):
        """Return ambient-space quotient representation Z @ P_hat."""
        if not hasattr(self, "P_hat_"):
            raise RuntimeError("Call fit() before transform().")

        Z = self.transform_raw(Y)
        return Z @ self.P_hat_

    def transform_coordinates(self, Y):
        """Return nonredundant coordinates in the orthogonal complement."""
        if not hasattr(self, "A_hat_"):
            raise RuntimeError("Call fit() before transform_coordinates().")

        Z = self.transform_raw(Y)

        _, _, Vt = np.linalg.svd(
            self.A_hat_.T,
            full_matrices=True,
        )

        C = Vt[self.selected_rank_:]
        return Z @ C.T

    def diagnostics(self):
        if not hasattr(self, "A_hat_"):
            raise RuntimeError("Call fit() before diagnostics().")

        return {
            "n_samples_fit": int(self.n_samples_fit_),
            "n_batches": int(len(self.batches_)),
            "n_features": int(self.n_features_in_),
            "crossfit": self.crossfit,
            "selected_rank": int(self.selected_rank_),
            "maximum_rank": int(self.maximum_rank_),
            "singular_values": self.singular_values_.copy(),
            "eigenvalues": self.eigenvalues_.copy(),
            "explained_variance_ratio":
                self.explained_variance_ratio_.copy(),
            "cumulative_explained_variance":
                self.cumulative_explained_variance_.copy(),
            "orthogonality_error": float(
                np.linalg.norm(
                    self.A_hat_.T @ self.A_hat_
                    - np.eye(self.selected_rank_)
                )
            ),
            "projection_idempotence_error": float(
                np.linalg.norm(
                    self.P_hat_ @ self.P_hat_
                    - self.P_hat_
                )
            ),
            "annihilation_error": float(
                np.linalg.norm(
                    self.P_hat_ @ self.A_hat_
                )
            ),
        }

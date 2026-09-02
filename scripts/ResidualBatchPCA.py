from __future__ import annotations

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


class ResidualBatchPCA:
    """
    Dataset-agnostic Residual Batch PCA.

    Parameters
    ----------
    rank:
        Fixed candidate nuisance rank. If None, select the smallest rank
        reaching variance_threshold.

    variance_threshold:
        Cumulative between-batch residual variance used for automatic rank
        selection.

    ridge_alpha:
        L2 penalty for the multivariate forward ridge regression.

    crossfit:
        "group_kfold", "leave_one_batch_out", or "none".

    n_splits:
        Number of folds for grouped cross-fitting.

    centroid_shrinkage_kappa:
        Optional explicit shrinkage of each batch residual centroid toward
        the equal-batch global centroid:

            alpha_b = n_b / (n_b + kappa)
            r_tilde_b = r_global + alpha_b (r_bar_b - r_global)

        kappa=0 disables shrinkage exactly.

        IMPORTANT:
        the historical FD004 write-up mentions centroid shrinkage but the
        exact coefficient was not preserved. Therefore this implementation
        exposes the operation explicitly instead of silently inventing a
        coefficient.
    """

    def __init__(
        self,
        rank=None,
        variance_threshold=0.90,
        ridge_alpha=1.0,
        crossfit="group_kfold",
        n_splits=5,
        random_state=42,
        centroid_shrinkage_kappa=0.0,
    ):
        if rank is not None and int(rank) < 1:
            raise ValueError("rank must be None or a positive integer.")

        if not (0.0 < float(variance_threshold) <= 1.0):
            raise ValueError("variance_threshold must lie in (0, 1].")

        if float(ridge_alpha) < 0:
            raise ValueError("ridge_alpha must be nonnegative.")

        allowed = {"group_kfold", "leave_one_batch_out", "none"}
        if crossfit not in allowed:
            raise ValueError(
                f"crossfit must be one of {sorted(allowed)}, got {crossfit!r}."
            )

        if int(n_splits) < 2:
            raise ValueError("n_splits must be at least 2.")

        if float(centroid_shrinkage_kappa) < 0:
            raise ValueError("centroid_shrinkage_kappa must be nonnegative.")

        self.rank = rank
        self.variance_threshold = float(variance_threshold)
        self.ridge_alpha = float(ridge_alpha)
        self.crossfit = crossfit
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.centroid_shrinkage_kappa = float(centroid_shrinkage_kappa)

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
                raise ValueError(
                    "sample_weight contains NaN or infinite values."
                )

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be nonnegative.")

            if sample_weight.sum() <= 0:
                raise ValueError(
                    "sample_weight must have positive total weight."
                )

        return Y, X_forward, batch, sample_weight

    def _new_forward_model(self):
        return Ridge(
            alpha=self.ridge_alpha,
            fit_intercept=False,
        )

    @staticmethod
    def _fit_ridge(model, X, Z, sample_weight=None):
        if sample_weight is None:
            model.fit(X, Z)
        else:
            model.fit(
                X,
                Z,
                sample_weight=sample_weight,
            )

        return model

    def _cross_fitted_residuals(
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

            residuals[:] = (
                Z
                - model.predict(X_forward)
            )

            self.forward_model_ = model
            self.crossfit_splits_ = None

            return residuals

        if self.crossfit == "leave_one_batch_out":
            splits = []

            for held_out_batch in unique_batches:
                train_index = batch != held_out_batch
                held_out_index = batch == held_out_batch

                model = self._new_forward_model()

                sw_train = (
                    None
                    if sample_weight is None
                    else sample_weight[train_index]
                )

                self._fit_ridge(
                    model,
                    X_forward[train_index],
                    Z[train_index],
                    sample_weight=sw_train,
                )

                residuals[held_out_index] = (
                    Z[held_out_index]
                    - model.predict(
                        X_forward[held_out_index]
                    )
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

        if n_batches < 2:
            raise ValueError(
                "Grouped cross-fitting requires at least two groups."
            )

        n_splits = min(
            self.n_splits,
            n_batches,
        )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        splits = []

        for train_index, held_out_index in splitter.split(
            X_forward,
            groups=batch,
        ):
            model = self._new_forward_model()

            sw_train = (
                None
                if sample_weight is None
                else sample_weight[train_index]
            )

            self._fit_ridge(
                model,
                X_forward[train_index],
                Z[train_index],
                sample_weight=sw_train,
            )

            residuals[held_out_index] = (
                Z[held_out_index]
                - model.predict(
                    X_forward[held_out_index]
                )
            )

            splits.append(
                (
                    train_index.copy(),
                    held_out_index.copy(),
                )
            )

        self.forward_model_ = None
        self.crossfit_splits_ = splits

        if not np.all(np.isfinite(residuals)):
            raise RuntimeError(
                "Cross-fitting did not produce finite residuals for all rows."
            )

        return residuals

    def fit(
        self,
        Y,
        X_forward,
        batch,
        sample_weight=None,
    ):
        (
            Y,
            X_forward,
            batch,
            sample_weight,
        ) = self._validate_inputs(
            Y,
            X_forward,
            batch,
            sample_weight=sample_weight,
        )

        n, m = Y.shape

        # ---------------------------------------------------------
        # 1. Training-only standardization
        # ---------------------------------------------------------
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(Y)

        self.n_features_in_ = m
        self.n_samples_fit_ = n

        # ---------------------------------------------------------
        # 2. Forward residuals
        # ---------------------------------------------------------
        self.batches_ = np.unique(batch)
        K = len(self.batches_)

        if K < 3:
            raise ValueError(
                "At least three training batches are required."
            )

        residuals = self._cross_fitted_residuals(
            Z=Z,
            X_forward=X_forward,
            batch=batch,
            sample_weight=sample_weight,
        )

        self.residuals_ = residuals

        # ---------------------------------------------------------
        # 3. Batch residual centroids
        # ---------------------------------------------------------
        batch_means = []
        batch_sizes = []

        for b in self.batches_:
            rb = residuals[batch == b]

            batch_means.append(
                rb.mean(axis=0)
            )
            batch_sizes.append(
                len(rb)
            )

        batch_means = np.vstack(batch_means)
        batch_sizes = np.asarray(
            batch_sizes,
            dtype=float,
        )

        global_mean = batch_means.mean(
            axis=0,
            keepdims=True,
        )

        kappa = self.centroid_shrinkage_kappa

        if kappa > 0:
            factors = (
                batch_sizes
                / (batch_sizes + kappa)
            )

            batch_means_used = (
                global_mean
                + factors[:, None]
                * (batch_means - global_mean)
            )
        else:
            factors = np.ones_like(
                batch_sizes,
                dtype=float,
            )

            batch_means_used = (
                batch_means.copy()
            )

        centered_batch_means = (
            batch_means_used
            - batch_means_used.mean(
                axis=0,
                keepdims=True,
            )
        )

        self.batch_means_raw_ = batch_means
        self.batch_means_ = batch_means_used
        self.batch_sizes_ = batch_sizes.astype(int)
        self.shrinkage_factors_ = factors
        self.centered_batch_means_ = centered_batch_means

        # ---------------------------------------------------------
        # 4. PCA via SVD
        # ---------------------------------------------------------
        _, singular_values, Vt = np.linalg.svd(
            centered_batch_means,
            full_matrices=False,
        )

        eigenvalues = singular_values ** 2
        total = eigenvalues.sum()

        if total > 0:
            explained_ratio = (
                eigenvalues / total
            )
        else:
            explained_ratio = (
                np.zeros_like(eigenvalues)
            )

        cumulative_ratio = np.cumsum(
            explained_ratio
        )

        numerical_rank = np.linalg.matrix_rank(
            centered_batch_means
        )

        maximum_rank = min(
            K - 1,
            m,
            numerical_rank,
        )

        if maximum_rank < 1:
            raise ValueError(
                "No nonzero residual between-batch direction was found."
            )

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
            selected_rank = int(
                self.rank
            )

        selected_rank = min(
            selected_rank,
            maximum_rank,
        )

        self.selected_rank_ = selected_rank
        self.maximum_rank_ = maximum_rank
        self.singular_values_ = singular_values
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ratio_ = explained_ratio
        self.cumulative_explained_variance_ = cumulative_ratio

        # ---------------------------------------------------------
        # 5. Candidate nuisance basis and quotient projector
        # ---------------------------------------------------------
        self.A_hat_ = (
            Vt[:selected_rank].T
        )

        self.P_hat_ = (
            np.eye(m)
            - self.A_hat_
            @ self.A_hat_.T
        )

        return self

    def transform_raw(self, Y):
        if not hasattr(self, "scaler_"):
            raise RuntimeError(
                "Call fit() before transform_raw()."
            )

        Y = np.asarray(
            Y,
            dtype=float,
        )

        if (
            Y.ndim != 2
            or Y.shape[1]
            != self.n_features_in_
        ):
            raise ValueError(
                "Unexpected Y shape."
            )

        return self.scaler_.transform(Y)

    def transform(self, Y):
        if not hasattr(self, "P_hat_"):
            raise RuntimeError(
                "Call fit() before transform()."
            )

        Z = self.transform_raw(Y)
        return Z @ self.P_hat_

    def transform_coordinates(self, Y):
        if not hasattr(self, "A_hat_"):
            raise RuntimeError(
                "Call fit() before transform_coordinates()."
            )

        Z = self.transform_raw(Y)

        _, _, Vt = np.linalg.svd(
            self.A_hat_.T,
            full_matrices=True,
        )

        C = Vt[
            self.selected_rank_:
        ]

        return Z @ C.T

    def diagnostics(self):
        if not hasattr(self, "A_hat_"):
            raise RuntimeError(
                "Call fit() before diagnostics()."
            )

        return {
            "n_samples_fit":
                int(self.n_samples_fit_),

            "n_batches":
                int(len(self.batches_)),

            "n_features":
                int(self.n_features_in_),

            "crossfit":
                self.crossfit,

            "selected_rank":
                int(self.selected_rank_),

            "maximum_rank":
                int(self.maximum_rank_),

            "centroid_shrinkage_kappa":
                float(
                    self.centroid_shrinkage_kappa
                ),

            "shrinkage_factor_min":
                float(
                    self.shrinkage_factors_.min()
                ),

            "shrinkage_factor_max":
                float(
                    self.shrinkage_factors_.max()
                ),

            "singular_values":
                self.singular_values_.copy(),

            "explained_variance_ratio":
                self.explained_variance_ratio_.copy(),

            "cumulative_explained_variance":
                self.cumulative_explained_variance_.copy(),

            "orthogonality_error":
                float(
                    np.linalg.norm(
                        self.A_hat_.T
                        @ self.A_hat_
                        - np.eye(
                            self.selected_rank_
                        )
                    )
                ),

            "projection_idempotence_error":
                float(
                    np.linalg.norm(
                        self.P_hat_
                        @ self.P_hat_
                        - self.P_hat_
                    )
                ),

            "annihilation_error":
                float(
                    np.linalg.norm(
                        self.P_hat_
                        @ self.A_hat_
                    )
                ),
        }

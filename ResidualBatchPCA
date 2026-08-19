import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)
from sklearn.linear_model import Ridge


class ResidualBatchPCA:
    """
    Covariate-adjusted residual PCA for estimating a nuisance subspace.

    Observation model:
        y_i = f(gas_i, concentration_i) + A nu_batch(i) + epsilon_i

    Steps:
        1. Standardize Y using training data only.
        2. Construct gas-specific nonlinear concentration design.
        3. Generate residuals by leave-one-batch-out ridge regression.
        4. Average residuals within each batch.
        5. Apply PCA/SVD to centered batch residual means.
        6. Use the leading right singular vectors as A_hat.

    The quotient projection is applied in standardized feature space.
    """

    def __init__(
        self,
        rank=None,
        variance_threshold=0.90,
        ridge_alpha=1.0,
        n_knots=5,
        spline_degree=3,
    ):
        self.rank = rank
        self.variance_threshold = variance_threshold
        self.ridge_alpha = ridge_alpha
        self.n_knots = n_knots
        self.spline_degree = spline_degree

    def _fit_design(self, gas, concentration):
        gas = np.asarray(gas).reshape(-1, 1)
        concentration = np.asarray(concentration, dtype=float).reshape(-1, 1)

        if np.any(concentration <= 0):
            raise ValueError("Concentration must be positive for log transform.")

        log_concentration = np.log(concentration)

        # Baseline gas is absorbed by the intercept and main concentration curve.
        self.gas_encoder_ = OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )
        gas_design = self.gas_encoder_.fit_transform(gas)

        self.spline_transformer_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.spline_degree,
            include_bias=False,
        )
        concentration_design = self.spline_transformer_.fit_transform(
            log_concentration
        )

        return self._combine_design(gas_design, concentration_design)

    def _transform_design(self, gas, concentration):
        gas = np.asarray(gas).reshape(-1, 1)
        concentration = np.asarray(concentration, dtype=float).reshape(-1, 1)

        if np.any(concentration <= 0):
            raise ValueError("Concentration must be positive for log transform.")

        gas_design = self.gas_encoder_.transform(gas)
        concentration_design = self.spline_transformer_.transform(
            np.log(concentration)
        )

        return self._combine_design(gas_design, concentration_design)

    @staticmethod
    def _combine_design(gas_design, concentration_design):
        n = concentration_design.shape[0]

        # Gas-specific deviations from the baseline concentration curve.
        if gas_design.shape[1] > 0:
            interaction = (
                gas_design[:, :, None]
                * concentration_design[:, None, :]
            ).reshape(n, -1)
        else:
            interaction = np.empty((n, 0))

        return np.column_stack(
            [
                np.ones(n),
                gas_design,
                concentration_design,
                interaction,
            ]
        )

    def fit(self, Y, gas, concentration, batch):
        Y = np.asarray(Y, dtype=float)
        gas = np.asarray(gas)
        concentration = np.asarray(concentration, dtype=float)
        batch = np.asarray(batch)

        if Y.ndim != 2:
            raise ValueError("Y must have shape (n_samples, n_features).")

        n, m = Y.shape

        if not (
            len(gas) == n
            and len(concentration) == n
            and len(batch) == n
        ):
            raise ValueError("Y, gas, concentration and batch lengths differ.")

        # -------------------------------------------------------------
        # 1. Standardize measurements using training data only
        # -------------------------------------------------------------
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(Y)

        # -------------------------------------------------------------
        # 2. Target-signal design:
        #    gas + spline(log concentration) + interaction
        # -------------------------------------------------------------
        X = self._fit_design(gas, concentration)

        # -------------------------------------------------------------
        # 3. Leave-one-batch-out residuals
        # -------------------------------------------------------------
        self.batches_ = np.unique(batch)
        K = len(self.batches_)

        if K < 3:
            raise ValueError(
                "At least three training batches are recommended."
            )

        residuals = np.zeros_like(Z)

        for held_out_batch in self.batches_:
            train_index = batch != held_out_batch
            held_out_index = batch == held_out_batch

            model = Ridge(
                alpha=self.ridge_alpha,
                fit_intercept=False,
            )

            model.fit(
                X[train_index],
                Z[train_index],
            )

            fitted_held_out = model.predict(X[held_out_index])

            residuals[held_out_index] = (
                Z[held_out_index] - fitted_held_out
            )

        self.residuals_ = residuals

        # -------------------------------------------------------------
        # 4. Batch-level residual means
        # -------------------------------------------------------------
        batch_means = np.vstack(
            [
                residuals[batch == b].mean(axis=0)
                for b in self.batches_
            ]
        )

        # Equal weight for each batch.
        centered_batch_means = (
            batch_means
            - batch_means.mean(axis=0, keepdims=True)
        )

        self.batch_means_ = batch_means
        self.centered_batch_means_ = centered_batch_means

        # -------------------------------------------------------------
        # 5. PCA/SVD of adjusted batch effects
        #
        # centered_batch_means has shape K x m.
        # The right singular vectors lie in R^m and estimate col(A).
        # -------------------------------------------------------------
        U, singular_values, Vt = np.linalg.svd(
            centered_batch_means,
            full_matrices=False,
        )

        eigenvalues = singular_values ** 2

        if eigenvalues.sum() > 0:
            explained_ratio = eigenvalues / eigenvalues.sum()
        else:
            explained_ratio = np.zeros_like(eigenvalues)

        cumulative_ratio = np.cumsum(explained_ratio)

        # The centered K-batch matrix has rank at most K - 1.
        numerical_rank = np.linalg.matrix_rank(
            centered_batch_means
        )
        maximum_rank = min(K - 1, m, numerical_rank)

        if maximum_rank < 1:
            raise ValueError(
                "No nonzero residual batch-effect direction was found."
            )

        # -------------------------------------------------------------
        # 6. Select nuisance rank
        # -------------------------------------------------------------
        if self.rank is None:
            selected_rank = (
                np.searchsorted(
                    cumulative_ratio,
                    self.variance_threshold,
                )
                + 1
            )
        else:
            selected_rank = int(self.rank)

        selected_rank = min(selected_rank, maximum_rank)

        if selected_rank < 1:
            raise ValueError("Selected rank must be at least one.")

        self.selected_rank_ = selected_rank
        self.maximum_rank_ = maximum_rank
        self.singular_values_ = singular_values
        self.explained_variance_ratio_ = explained_ratio
        self.cumulative_explained_variance_ = cumulative_ratio

        # -------------------------------------------------------------
        # 7. Estimated nuisance basis and quotient projector
        # -------------------------------------------------------------
        self.A_hat_ = Vt[:selected_rank].T
        self.P_hat_ = (
            np.eye(m)
            - self.A_hat_ @ self.A_hat_.T
        )

        return self

    def transform(self, Y):
        """
        Apply the fitted quotient projection to new measurements.

        gas/concentration labels are not required at inference time.
        """
        Y = np.asarray(Y, dtype=float)

        Z = self.scaler_.transform(Y)

        return Z @ self.P_hat_

    def transform_coordinates(self, Y):
        """
        Return nonredundant quotient coordinates C y in R^(m-r),
        rather than the ambient representation P y in R^m.
        """
        Y = np.asarray(Y, dtype=float)
        Z = self.scaler_.transform(Y)

        # Complete SVD of A_hat^T:
        # the last m-r right singular vectors span col(A_hat)^perp.
        _, _, Vt = np.linalg.svd(
            self.A_hat_.T,
            full_matrices=True,
        )

        C = Vt[self.selected_rank_:]

        return Z @ C.T

    def diagnostics(self):
        return {
            "n_batches": len(self.batches_),
            "selected_rank": self.selected_rank_,
            "maximum_rank": self.maximum_rank_,
            "singular_values": self.singular_values_.copy(),
            "explained_variance_ratio":
                self.explained_variance_ratio_.copy(),
            "cumulative_explained_variance":
                self.cumulative_explained_variance_.copy(),
            "orthogonality_error": np.linalg.norm(
                self.A_hat_.T @ self.A_hat_
                - np.eye(self.selected_rank_)
            ),
            "projection_idempotence_error": np.linalg.norm(
                self.P_hat_ @ self.P_hat_
                - self.P_hat_
            ),
            "annihilation_error": np.linalg.norm(
                self.P_hat_ @ self.A_hat_
            ),
        }

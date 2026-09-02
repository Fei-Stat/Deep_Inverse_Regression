# Reproducibility notes

## FD004

The preprocessing code now guarantees:

- 20-cycle windows with stride 2.
- A final window ending at the final observed cycle for every engine.
- Left-padding with the first observation for official test engines whose
  entire observed trajectory is shorter than 20 cycles.
- 18,429 official-test windows on the standard NASA FD004 files.
- Engine-level train/validation splitting.

For exact reproduction of the historical 23,034 training windows and
5,417 validation windows, provide the original fixed 50 validation engine
IDs through:

    --valid-engine-ids-file path/to/fd004_valid_engines.txt
    --require-paper-split

The historical write-up also states that engine residual centroids were
shrunk toward the global mean, but the exact shrinkage coefficient was not
preserved. The implementation therefore exposes

    --centroid-shrinkage-kappa

explicitly. The default value 0 disables shrinkage rather than silently
inventing a coefficient.

## OSSL organic carbon

Validation model selection is source-balanced:

    macro-MSE =
        mean_s MSE(validation source s).

The same equal-source principle is used for training weights.

For TabM, both raw and quotient branches use the identical learning-rate
grid:

    1e-4, 3e-4, 1e-3, 3e-3, 1e-2

with 60 epochs per candidate in the formal run. Checkpoints and the final
learning rate are selected exclusively by validation macro-MSE.

XGBoost early stopping also uses validation macro-MSE through a custom
XGBoost metric.

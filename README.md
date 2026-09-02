# Deep Inverse Regression on Linear Quotient Spaces

## Pipeline 1: Estimate the nuisance subspace

`ResidualBatchPCA.py` estimates the
nuisance subspace using the training batches only. Validation and test
data must not be passed to `estimator.fit()`.

```python
# Y_train: (n_train, 128)
# gas_train: gas identity
# concentration_train: concentration
# batch_train: training-batch labels

estimator = ResidualBatchPCA(
    rank=None,
    variance_threshold=0.90,
    ridge_alpha=1.0,
    n_knots=5,
    spline_degree=3,
)

estimator.fit(
    Y=Y_train,
    gas=gas_train,
    concentration=concentration_train,
    batch=batch_train,
)

A_hat = estimator.A_hat_
P_hat = estimator.P_hat_

print(estimator.diagnostics())
```

Here, `A_hat` contains the estimated nuisance directions and

$$
P_{\mathrm{hat}}=I-A_{\mathrm{hat}}A_{\mathrm{hat}}^\top
$$

is the quotient projection. The 90% explained-variance rule provides a
candidate nuisance rank; the final deployment decision is made by the
proposed discriminator.

## Pipeline 2: Generate raw and quotient representations

Both branches use the same scaler fitted on the training data.

```python
# Raw standardized representations
Y_train_raw = estimator.scaler_.transform(Y_train)
Y_valid_raw = estimator.scaler_.transform(Y_valid)
Y_test_raw = estimator.scaler_.transform(Y_test)

# Quotient representations
Y_train_quotient = estimator.transform(Y_train)
Y_valid_quotient = estimator.transform(Y_valid)
Y_test_quotient = estimator.transform(Y_test)
```

Gas identity, concentration, and batch labels are required only when
estimating `A_hat`. They are not required by `estimator.transform()` at
inference time.

## Pipeline 3: Train the backbones

The raw and quotient models must use identical architectures, training
settings, validation procedures, and hyperparameter-search budgets.

```python
raw_model = build_backbone(model_config)
quotient_model = build_backbone(model_config)

raw_model.fit(
    Y_train_raw,
    q_train,
    validation_data=(Y_valid_raw, q_valid),
)

quotient_model.fit(
    Y_train_quotient,
    q_train,
    validation_data=(Y_valid_quotient, q_valid),
)
```

If the inverse model also uses known experimental information `x`, pass
the same `x` to both branches:

```python
raw_model.fit(
    [Y_train_raw, x_train],
    q_train,
    validation_data=([Y_valid_raw, x_valid], q_valid),
)

quotient_model.fit(
    [Y_train_quotient, x_train],
    q_train,
    validation_data=([Y_valid_quotient, x_valid], q_valid),
)
```

All preprocessing and model choices are fixed using the training and
validation sets. The isolated test batches are evaluated only once at
the end.

## The Illustrations of Two Datasets

### NASA C-MAPSS FD004
FD004 dataset essentially decribes the remaining useful life (RUL) of commercial turbofan engines developed by NASA. In particular, the included fields are given below:
| unit / engine_id | cycle | setting_1 | setting_2 | setting_3 | sensor_1 | sensor_2 | sensor_3 | sensor_4 | sensor_5 | sensor_6 | sensor_7 | sensor_8 | sensor_9 | sensor_10 | sensor_11 | sensor_12 | sensor_13 | sensor_14 | sensor_15 | sensor_16 | sensor_17 | sensor_18 | sensor_19 | sensor_20 | sensor_21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

where
1. engine_id: The identification of the studied engine;
2. cycle: The timestamp that the engine is run;
3. setting: The deterministic variables describing the running status;
4. sensor: Data collected by sensors to estimate the status.

Training_procedure
1. Split the training data FD004 into


### Open Soil Spectral Library
OSSL dataset is joint table of a variety of data collections to describe soil spectral records and related component contents:
| source / dataset.code_ascii_txt | sample_id / id.layer_uuid_txt | organic_carbon / oc_usda.c729_w.pct | mir_600 | mir_608 | mir_616 | ... | mir_3984 | mir_3992 | mir_4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |







## Inverse Task 1. UCI Gas Sensor Dataset

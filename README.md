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

## 流程
一、基本设定
数据集：UCI Gas Sensor Array Drift Dataset
来源：加州大学尔湾分校气体传感器阵列漂移数据集
13,910条记录、10个时间批次、6种气体、16个化学传感器。维度为6*16=96。

Backbone：
1. ResNet-Based MLP (可以查一下Kaiming He的ResNet)
2. FT-Transformer (可以查表格型Transformer)
3. TabM（来自俄罗斯Yandex公司的一篇论文）
4. XGBoost（轻量级统计算法）

随机种子：随机生成五个seed，比方说可以随机生成五个(0,1)的小数然后乘以10000向下取整。

数据划分：将数据分为十个批次，一到六折作为训练集，七到八为验证集，九到十为测试集

二、训练步骤
1. 导入数据集并完成划分
2. 调用ResidualBatchPCA在训练集拟合一个A矩阵（干扰参数）
3. 对训练集、验证集、测试集进行商化
4. 对于同一个backbone，分别在商化训练集和原始训练集训练出两个trained networks
5. 在验证集上选择参数，在测试集上进行测试，测试结果要写出五次训练的标准差
6. 画出相应的图表

4个backbone，训出8个模型，seed取五个，原始和商化训练两次，共40*2=80次训练。

三、任务划分

逸菲和维翰分别负责数据处理和模型评价。在程序和数据处理流程确定后，两人共同承担正式实验。

A：逸菲——数据处理与商化

1. 熟悉数据集的各个字段，按照批次或时间划分训练集、验证集和测试集，并保存固定的数据划分索引。

2. 保存一份未经修改的原始数据。仅使用训练集调用 `ResidualBatchPCA.fit()`，得到训练集标准化器、干扰子空间矩阵 (\hat A) 和商化投影矩阵 (\hat P)。

3. 使用同一个训练集标准化器生成三套数据的raw表示：

   * `Y_train_raw`
   * `Y_valid_raw`
   * `Y_test_raw`

4. 使用同一个 `ResidualBatchPCA`生成三套数据的quotient表示：

   * `Y_train_quotient`
   * `Y_valid_quotient`
   * `Y_test_quotient`

5. 保存标准化器、(\hat A)、(\hat P)、PCA解释率、候选rank及相关诊断结果，供后续实验统一调用。

验证集和测试集只能调用 `transform()`，不能参与 `ResidualBatchPCA.fit()`。

B：维翰——模型训练与结果评价

1. 实现统一的backbone训练与评价程序。对每一种backbone分别训练：

   * 使用标准化原始数据的raw模型；
   * 使用商化数据的quotient模型。

2. 保证raw和quotient模型采用相同的数据划分、网络结构、优化器、训练轮数、early stopping规则、随机种子和超参数搜索范围。

3. 使用验证集选择超参数和最佳checkpoint。完成选择后，锁定模型配置及参数，不再根据测试结果修改。

4. 将锁定后的raw和quotient模型分别应用到同一个测试集，计算预测指标、风险差、置信区间及判别器结论。

5. 汇总实验结果并制作论文所需的表格和图片。

正式实验分配

首先共同完成少量测试实验，确认数据维度、训练程序和输出结果均正确。

流程确认后，将80组实验平均分配。每一组实验必须包含同一配置下的raw和quotient两个模型，不能由一个人只跑raw、另一个人只跑quotient。

* 逸菲负责20组配对实验，共40次训练；
* 维翰负责20组配对实验，共40次训练。

每个实验都应记录实验编号、数据集、backbone、随机种子、超参数、模型checkpoint和最终评价指标。测试集仅在所有配置确定后使用。






## Inverse Task 1. UCI Gas Sensor Dataset

# Deep Inverse Regression on Linear Quotient Spaces

## The Illustrations of Two Datasets

### NASA C-MAPSS FD004
FD004 dataset essentially decribes the remaining useful life (RUL) of commercial turbofan engines developed by NASA. In particular, the included fields are given below:
| engine_id | cycle | operating settings | sensor features |
| --- | --- | --- | --- |
| `unit / engine_id` | `cycle` | `setting_1` ~ `setting_3` | `sensor_1` ~ `sensor_21` |

where
1. engine_id: The identification of the studied engine;
2. cycle: The timestamp that the engine is run;
3. setting: The deterministic variables describing the running status;
4. sensor: Data collected by sensors to estimate the status.

Training_procedure
1. Split the training data FD004 into


### Open Soil Spectral Library
OSSL dataset is joint table of a variety of data collections to describe soil spectral records and related component contents:
| source | sample_id | target | MIR features |
| --- | --- | --- | --- |
| `dataset.code_ascii_txt` | `id.layer_uuid_txt` | `oc_usda.c729_w.pct` | `scan_mir.600_abs` ~ `scan_mir.4000_abs` |

where
1. source: The unjoint dataset of origin;
2. sample_id: The identification of the studied sample;
3. target: The response variable, proportion of organic carbon;
4. MIR features: The predictors, MIR data.

### Pipeline 1: Extracting Nuisance Parameters

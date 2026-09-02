# Data setup (for students running this repo locally)

Raw data is **not** committed to this repository. Each user downloads the
data to their own machine and points the scripts at it with a local path.
This is intentional:

- **C-MAPSS FD004** is distributed by NASA's PCoE data repository without a
  stated redistribution license, so it should not be re-hosted here.
- **OSSL** is CC-BY-4.0 and could legally be re-hosted, but is kept out of
  version control anyway to avoid bloating the repo with large data files.

## 1. Environment

No `requirements.txt` is pinned yet. At minimum you need:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn xgboost torch tabm rtdl_revisiting_models
```

If you have a CUDA GPU, install the matching `torch` build from
https://pytorch.org/get-started/locally/ instead of the plain `pip install torch`
above.

## 2. C-MAPSS FD004

Source: NASA Prognostics Center of Excellence Data Set Repository
(https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
Find the "Turbofan Engine Degradation Simulation Data Set" (C-MAPSS) entry
and download it from there — no stable direct-file URL is documented here on
purpose, since NASA has changed hosting before.

Unzip it and place these three files together in one folder, keeping the
original NASA filenames:

```
data/CMAPSS/
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

Run a quick smoke test (1 backbone, 1 seed, ≤3 epochs — just to confirm the
environment and data path are correct):

```bash
python run_fd004.py --data-dir data/CMAPSS --smoke
```

Then the paper-reproduction run, using the validation-engine-ID file already
included in this repo:

```bash
python run_fd004.py \
  --data-dir data/CMAPSS \
  --valid-engine-ids-file fd004_valid_engines_COUNT_MATCH_ONLY_seed277.txt \
  --require-paper-split
```

Results are written to `results/fd004_results.csv` by default (`--output` to
change it).

## 3. OSSL (organic carbon)

Source: Open Soil Spectral Library, CC-BY-4.0
(https://zenodo.org/records/7599269). Direct download links (documented at
https://docs.soilspectroscopy.org/db-access.html):

```bash
mkdir -p data/OSSL
cd data/OSSL
curl -O https://storage.googleapis.com/soilspec4gg-public/ossl_all_L1_v1.2.csv.gz
cd -
```

(Or, if you'd rather download the two separate tables instead of the joined
one: `ossl_mir_L0_v1.2.csv.gz` and `ossl_soillab_L1_v1.2.csv.gz` from the same
base URL, then pass both via `--mir-l0` / `--soillab-l1` instead of `--all-l1`
below. `pandas.read_csv` reads `.csv.gz` directly — no need to decompress.)

Smoke test:

```bash
python run_ossl.py --all-l1 data/OSSL/ossl_all_L1_v1.2.csv.gz --smoke
```

Full run:

```bash
python run_ossl.py --all-l1 data/OSSL/ossl_all_L1_v1.2.csv.gz
```

Results are written to `results/ossl_results.csv` by default.

## Attribution reminder

If any results from the OSSL data are published or shared, cite the OSSL
dataset (DOI: 10.5281/zenodo.7599269) per its CC-BY-4.0 terms. For C-MAPSS,
NASA's repository asks that publications acknowledge both the repository and
the data donors.

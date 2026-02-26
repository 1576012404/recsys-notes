# DeepFM Reproduction (with FNN baseline)

Reproduction pipeline for CTR models on Criteo Kaggle dataset:

- LR
- FM
- FNN (FM-pretrained embedding initialization)
- DNN
- Wide&Deep
- DeepFM

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Data

Place Kaggle Criteo `train.txt` at:

`data/raw/train.txt`

Then preprocess:

```bash
python -m scripts.prepare_data --config configs/data.yaml
```

For `Criteo_x1` (`train.csv/valid.csv/test.csv`), use:

```bash
python -m scripts.prepare_data --config configs/data_criteo_x1.yaml
```

Outputs:

- `data/processed/criteo.npz`
- `data/processed/metadata.json`

## 3. Train

Train a single model:

```bash
python -m scripts.train --model deepfm --seed 2026
```

Train FM pretrain for FNN:

```bash
python -m scripts.train_fm_for_fnn_init --seed 2026
```

Train FNN from FM checkpoint (latest FM run by default):

```bash
python -m scripts.train_fnn --seed 2026
```

## 4. Evaluate and Compare

Evaluate one run:

```bash
python -m scripts.evaluate --run-dir results/runs/<run_id>
```

Generate comparison report:

```bash
python -m scripts.compare --input results/metrics.csv --output report/comparison.md
```

Run complete suite (all models, multiple seeds):

```bash
python -m scripts.run_all --seeds 2026 2027 2028
```

## 5. Results Schema

`results/metrics.csv` columns:

- `run_id`
- `model`
- `seed`
- `split`
- `auc`
- `logloss`
- `best_epoch`
- `config_hash`
- `timestamp`
- `init_from` (`none` or `fm_pretrain`)
- `pretrain_run_id`

## Requirements

The code is written in Python 3 and mainly depends on:

- numpy
- pandas
- scikit-learn
- matplotlib
- joblib

Install dependencies with:

~~~bash
pip install -r requirements.txt
~~~

## Example Usage

### Energies multi-basin workflow

~~~bash
python src/CN_train_energy_models.py --data_path path/to/your/data.xlsx --out_dir outputs
~~~

### Egypt Western Desert workflow

~~~bash
python src/egyptTrain.py --data_dir path/to/your/data --out_dir outputs
~~~

### US Bakken workflow

~~~bash
python src/BakkenTrain.py --data_path path/to/your/data.csv --out_dir outputs
~~~

## Quick Test

A lightweight quick-test guide is provided in:

- `examples/quick_test.md`

## Data Notes

The datasets used in this study are based on publicly available sources and/or literature-organized tabular datasets.

Users should prepare the corresponding input datasets locally in compatible CSV or Excel format before running the scripts.

## Reproducibility Note

This repository is intended to provide the core code and workflow used in the study.

Depending on the local environment and dataset preparation, users may need to adapt input file paths and file names.

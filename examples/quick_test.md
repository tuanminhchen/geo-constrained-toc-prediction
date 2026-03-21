# Quick Test

## Purpose

This quick test is intended to demonstrate the basic script usage and workflow structure in this repository.  
It is not intended to reproduce the full experiments reported in the manuscript.

## Basic Example Commands

### 1. Energies multi-basin script

~~~bash
python src/CN_train_energy_models.py --data_path path/to/your/data.xlsx --out_dir outputs
~~~

### 2. Egypt Western Desert script

~~~bash
python src/egyptTrain.py --data_dir path/to/your/data --out_dir outputs
~~~

### 3. Bakken script

~~~bash
python src/BakkenTrain.py --data_path path/to/your/data.csv --out_dir outputs
~~~

## Notes

Users may need to adapt file names, paths, and input table formats according to their local data organization.

This quick test is intended to provide a minimal example of the workflow rather than a full reproduction package for all experiments in the manuscript.

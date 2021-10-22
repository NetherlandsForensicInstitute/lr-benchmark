LRBenchmark
=====

Repository for benchmarking Likelihood Ratio systems.

Prerequisites
-----------
- This repository is developed for Python 3.8.

Dependencies
-----------
All dependencies can either be installed by running `pip install -r requirements.txt` or `pip install .`.
  
Add new dependencies to the setup.py and always update the requirements by running: 
`pip-compile --output-file=requirements.txt setup.py`.

Usage
-----------
Running the benchmark can be done as follows:
1. Specify the parameters for the benchmark in the `lrbenchmark.yaml`
2. Run `python run.py`

The parameters for the benchmark must be provided in the following structure: 
```
experiment:
  repeats: 10
  scorer:
    - 'name scorer 1'
    - 'name scorer 2'
  calibrator: 
    - 'name calibrator'
  dataset:
    - 'name dataset'
  preprocessor:
    - 'name preprocessor 1'
    - 'name preprocessor 2'
```
At least 1 setting needs to be provided for each parameter, but more settings per parameter can be provided. The pipeline will
create the cartesian product over all parameter settings (except `repeats`) and will execute the experiments accordingly.

All possible settings can be found in `params.py`. The parameters that need to be set are:
- `repeats`: Number of repeats for each experiment.
- `scorer`: Scoring models for generating scores.
- `calibrator`: Models for calibrating scores. 
- `dataset`: Datasets on which the experiments can be executed.
- `preprocessor`: Data preprocessing steps. You can use the value `'dummy'` if no preprocessing is needed.


Example: Benchmark feature rank based LR system
-----------
This repository supports several data transformations, such as the possibility to transform X features from values to ranks. 
To benchmark these models against models without any transformations on X features, the following experiments (among others) could be 
defined in `lrbenchmark.yaml`. 
```
experiment:
  repeats: 10
  scorer:
    - 'LR'
    - 'XGB'
  calibrator:
    - 'logit'
  dataset:
    drugs_xtc:
      n_splits: 2
    glass:
      n_splits: 2
  preprocessor:
    - 'dummy'
    - 'rank_transformer'
```
When executing `python run.py` an experiment for all possible combination of parameters will be executed. The results for each experiment (metrics + plots)
will be stored in separate folders within the `output` folder.

Datasets
----------
There are currently two datasets available for this project:
- drugs_xtc: available as a csv file in the resources folder
- glass: LA-ICPMS measurements of elemental concentration from floatglass. The data will be downloaded automatically from https://github.com/NetherlandsForensicInstitute/elemental_composition_glass when used in the pipeline for the first time. 

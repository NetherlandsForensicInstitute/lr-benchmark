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
2. Run `python run.py -d <dataset_config>`. The available dataset configuration files can be found in the `config/data` folder

The parameters for the benchmark must be provided in the following structure: 
```
experiment:
  repeats: 10
  scorer:
    - 'name scorer 1'
    - 'name scorer 2'
  calibrator: 
    - 'name calibrator'
  preprocessor:
    - 'name preprocessor 1'
    - 'name preprocessor 2'
  splitting_strategy:
    - train_size: <int for a specific number of sources in the training set, float for a fraction, None to be complementary to the test_size>
    - test_size: <int for a specific number of sources in the test set, float for a fraction, None to be complementary to the train_size>
  refnorm: <optional>
    refnorm_size: <int for a specific number of sources in the refnorm set, float for a fraction, None to use the Leave-One-Out method>    
    
    
```
At least 1 setting needs to be provided for each parameter, but more settings per parameter can be provided. The pipeline will
create the cartesian product over all parameter settings (except `repeats`) and will execute the experiments accordingly.

All possible settings can be found in `params.py`. The parameters that need to be set are:
- `repeats`: Number of repeats for each experiment.
- `scorer`: Scoring models for generating scores.
- `calibrator`: Models for calibrating scores. 
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
  preprocessor:
    - 'dummy'
    - 'rank_transformer'
```
When executing `python run.py -d glass` an experiment for all possible combination of parameters will be executed on the glass dataset. 
The results for each experiment (metrics + plots) will be stored in separate folders within the `output` folder.

Datasets
----------
There are currently a number of datasets implemented for this project:
- drugs_xtc: will be published on our github soon
- glass: LA-ICPMS measurements of elemental concentration from floatglass. The data will be downloaded automatically from https://github.com/NetherlandsForensicInstitute/elemental_composition_glass when used in the pipeline for the first time.
- asr: a sample dataset will be published on our github
- synthesized-normal: a dataset containing generated samples from normal distributions following the specifications in the `config/data/synthesized-normal.yaml` file.


Reference Normalization
----------
#Todo more explanation what refnorm is and why you use it.

If `refnorm` is specified in the `lrbenchmark.yaml` file, and the dataset is a `MeasurementPairsDataset`,
the scores of the measurement pairs will be transformed using reference normalization.
This reference normalization will be either performed using a separate refnorm dataset (when `refnorm.size` is defined). 
This dataset has a set of unique source ids that do not occur in the training or validation sets. The measurement pairs 
in the refnorm dataset have one measurement with a source from this refnorm specific source id set, and the other 
measurement comes from the training/validation set. 

An example: the dataset contains source ids `a`, `b` and `z`, the selected measurement pair has
a measurement 1 with source `a` and a measurement 2 with source `b`. The refnorm source ids are `c` and `d`. All 
measurement pairs in the refnorm set contain one measurement with source `c` or `d`, and one measurement with source 
`a`, `b`, or `z`. When selecting the refnorm measurement pairs for measurement 1, we take 
all the measurement pairs from the refnorm set that contain measurement 1 and a measurement with source `c` or `d`.

If the `refnorm.refnorm_size` is not defined, the normalization will be done with the Leave-One-Out method. This means
that for each measurement pair in the dataset, the rest of the dataset will be acting as refnorm set. 
For each measurement in the selected measurement pair, only the measurement pairs in the left-over dataset will be used 
where one of the measurements is equal to this measurement, and the other has a source_id that is different from the 
source ids in the selected measurement pair. 

An example: the selected measurement pair has a measurement 1 with source `a` and a measurement 2 with source `b`. When 
selecting the refnorm measurement pairs for measurement 1, we take all the measurement pairs from the dataset that 
contain measurement 1 and a measurement that does not come from source `a` or source `b`.

Once we have the appropriate refnorm pairs for each of the measurements in the selected pair, the score of the pair is 
transformed with each set of refnorm pairs by subtracting the mean of the scores of the refnorm pairs and 
dividing by the standard deviation. The subsequent two scores are averaged to come to the transformed score that will 
be used in the pipeline. 

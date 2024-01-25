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


Computing Case LRs
----------
To compute case LRs, provide a dataset with the ids of the sources of interest as `holdout_source_ids`. This will
train a model as specified in the yaml on the other sources, and apply it to the specified sources. All LRs for the
measurement pairs for these sources will be provided in a text file. Typically, you may want to run experimentation
and validation on a set that does not contain the case sources to select a set of parameters. You then run the pipeline
again after including the case-relevant source in the dataset, identifying them with the `holdout_source_ids` 
parameter. Examples are given in the `asr.yaml` and 'asr_case.yaml` files.


Reference Normalization
----------
Reference normalization is a procedure that helps combat condition mismatch. It attempts to measure the influence that 
the measurement conditions have on the comparison score. It then uses this measurement to compensate for unwanted 
influence of those measurement conditions. In the case of symmetrical normalization (S-norm), this is performed by 
comparing all the measurements in the reference normalization cohort with each of the two measurements that are to be 
compared. This produces two collections of scores, which, if the practitioner chose the reference normalization 
sources correctly, are all different source-scores. Of each of those sets of different source scores the average and 
standard deviation are calculated. The ‘unnormalized’ score that was obtained by comparing the to-be-compared 
measurement is then normalized with the first average and standard deviation. Then the same ‘unnormalized’ score is 
normalized again with the second average and standard deviation, again by subtracting the average and dividing over 
the standard deviation. This results in two intermediate normalized scores. These two scores are then averaged, 
resulting in the normalized score. 

This procedure is designed to discard effects on the score coming from measurement conditions that have an influence on 
both the actual comparison data and the reference normalization cohort data. Note that this procedure assumes the 
sources in the reference normalization cohort and the sources that are compared are different speakers. If, 
inadvertently, there is source overlap between the compared sources and the reference cohort sources, some of the 
comparisons are of same source comparisons instead of the expected different source comparisons. Both the average and 
the standard deviation of the sets of scores would become too high, resulting in a normalized score that is too low.

If `refnorm` is specified in the `lrbenchmark.yaml` file, the scores of the measurement pairs will be transformed using
reference normalization. This reference normalization will be either performed using a separate refnorm dataset (when 
`refnorm.size` is defined). This dataset has a set of unique sources that do not occur in the training or validation 
sets. An example: the dataset contains source ids `a`, `b` and `z`, the refnorm source ids are `c` and `d`, and the 
selected measurement pair has a measurement 1 with source `a` and a measurement 2 with source `b`. When 
performing reference normalization all measurements with source id `c` or `d` will be compared with both measurements in
the measurement pair. 

If the `refnorm.refnorm_size` is not defined, the normalization will be done with the Leave-One-Out method. This means
that for each measurement pair in the dataset, the rest of the dataset will be acting as refnorm set. 
For each measurement in the selected measurement pair, only the measurements in the left-over dataset will be used 
which have source ids that are not in the selected measurement pair. 

An example: the selected measurement pair has a measurement 1 with source `a` and a measurement 2 with source `b`. When 
selecting the refnorm measurement pairs for measurement 1, we take all the measurements from the dataset that do not 
come from source `a` or source `b`.

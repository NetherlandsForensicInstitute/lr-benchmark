LRBenchmark
=====

Repository for optimising and evaluating Likelihood Ratio (LR) systems. This allows benchmarking of LR systems on 
different datasets, investigating impact of different sampling schemes or techniques, and doing case-based validation
and computation of case LRs.

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
1. Specify the properties of the data to be used in `config/data/<datasetname>.yaml`
2. Specify the parameters for the experiments to be run in the `lrbenchmark.yaml`
3. Run `python run.py -d <dataset_name>`.

Dataset properties are given as follows:
```
dataset:
  name: <dataset_name>
  limit_n_measurements: <int that limits how many measurements to use (typically to reduce computation times during development)
  scores_path: <optionally provide the location of a file with pre-computed scores>
  meta_info_path: <optionally provide the location of a file with additional information on the measurements>
  filter_on_trace_reference_properties: <Boolean, indicating whether there are two types of measurements A and B, such
  that all pairs of measurements should be A-B. Typically a trace and a reference.>
  source_filter:
    <optional list of properties on which to filter sources. Typically used to select a relevant population.>
  trace_properties:
    <optional list of properties that defines `trace` measurements></list>
  reference_properties:
    <optional list of properties that defines `reference` measurements>
  holdout_source_ids: 
    <optional list of ids of source that should be set apart, to have LRs calculated at the end>
```

The parameters for the benchmark must be provided in the following structure: 
```
experiment:
  repeats: 10
  pairing:
    - name: cartesian
  scorer:
    - name: <first scorer>
      preprocessors:
        - name: <first preprocessor>
        - name: <second preprocessor>
    - name: <second scorer>
      preprocessors:
        - name: <first preprocessor>
  calibrator: 
    - name: 'calibrator 1'
  splitting_strategy:
    validation:
      split_type: [leave_one_out simple]
      train_size: <int for a specific number of sources in the training set, float for a fraction, None to be complementary to the test_size>
      validate_size: <int for a specific number of sources in the test set, float for a fraction, None to be complementary to the train_size>
    refnorm:
      split_type: [leave_one_out simple None]
      size: <int for a specific number of sources in the refnorm set, float for a fraction, None to use the Leave-One-Out method>    
        
    
```
At least 1 setting needs to be provided for each parameter, but more settings per parameter can be provided. The pipeline will
create the cartesian product over all parameter settings (except `repeats`) and will execute the experiments accordingly.

All possible settings can be found in `params.py`. The parameters that need to be set are:
- `repeats`: Number of repeats for each experiment.
- `scorer`: Scoring models for generating scores.
- `calibrator`: Models for calibrating scores. 
- `preprocessor`: Data preprocessing steps. You can use the value `'dummy'` if no preprocessing is needed.
- `pairing`: How to create pairs of measurements. `'Cartesian'` will create all possible pairs, `'balanced'` will 
   subsample so there are an equal number of same-source and different-souce pairs.
- `splitting_strategy`: How to split the data. The data are split into validation and training parts,
  e.g. by a `simple` split or using a `leave_one_out` scheme. Application of reference normalisation is optional, and
  possible through a `simple` split or using a `leave_one_out` scheme. Note that using `leave_one_out` twice will lead
  to long computation times.





Examples of typical use cases
-------------
### Benchmarking techniques: rank transformation
When researchers come up with a new generic technique that could be useful in constructing LR systems, 
it can be relevant to test a variety of models with or without this technique on a variety of datasets.
One such technique is the percentile rank transformation, which transforms the raw values of features of measurements
to their percentile rank in the distribution (cf Matzen et al. (2022) FSI). This could be achieved using the
following `lrbenchmark.yaml`. 
```
experiment:
  repeats: 10
  scorer:
    - 'rf'
    - 'xgb'
  calibrator:
    - 'logit'
  preprocessor:
    - 'dummy'
    - 'rank_transformer'
```
When executing `python run.py -d glass` both a random forest and extreme gradient boosting model will be trained and 
evaluated on the glass dataset, with or without using the rank transformation. 
The results for each experiment (metrics + plots) will be stored in separate folders within the `output` folder.

### Doing casework: Calibration, validation and case LR computation for ASR
A typical case may involve selecting relevant training/validation data matching the conditions of the case, performing
training/validation and, if results are good enough, applying the system to obtain the LRs for reporting.

Selecting the relevant data from a larger set is supported by providing the `source_filter` options in the dataset yaml,
and/or the `trace_properties` and `reference_properties`. The former filters source, the latter filters measurements. It
also allows for different filters on the two parts of the measurement pairs ('trace' and 'reference'). 

In the same datafile you can provide the information on the sources from the case, whose ids should be specified in 
`holdout_source_ids`. This will train and validate a model as specified in the yaml on the other sources, and apply it 
to the specified sources. All LRs for the measurement pairs for these sources will be provided in a text file. 

Below are two yaml files that together achieve this for ASR data

```
experiment:
  repeats: 1
  pairing:
    - name: cartesian
  scorer:
    - name: precalculated_asr
      scores_path: ${dataset.scores_path}
  calibrator:
    - name: elub_logit
  splitting_strategy:
    validation:
      split_type: leave_one_out # leave_one_out or simple.
      validate_size: 
    refnorm:
      split_type: simple 
      size: .2 
```


```
dataset:
  name: asr
  limit_n_measurements:
  scores_path: resources/asr/scorematrix.csv
  meta_info_path: resources/asr/recordings_anon.txt
  filter_on_trace_reference_properties: True
  holdout_source_ids:
    109051
    114844
  source_filter:
    sex: M
    beller_fluistert:
      - nee
      - kort
    languageID:
      - 7
      - 9
  trace_properties:
    auto: ja
    duration: 30
  reference_properties:
    auto: nee
    duration: 30
```

Datasets
----------
There are currently a number of datasets implemented for this project:
- drugs_xtc: measurements on xtc pills
- glass: LA-ICPMS measurements of elemental concentration from floatglass. The data will be downloaded automatically from https://github.com/NetherlandsForensicInstitute/elemental_composition_glass when used in the pipeline for the first time.
- asr (=automatic speaker recognition): a dataset of scores computed using the VOCALISE software on casework representative speaker recordings. Speaker ground truth
  as well as some speaker and recording metadata are provided.

Simulations
It is straightforward to simulate data for experimentation. Currently a very simple simulation 'synthesized-normal' of the two-level model
is provided, with sources and measurements drawn from normal distributions.



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


Data representation
-----------------
We model the data generically as a `Measurement` on a `Source`. The measurements typically have a `value` (numpy array).
Two measurements can be combined into a `MeasurementPair`, on which typical common-source LR systems operate. All of 
these data structures have an `extra` (Mapping) variable to flexibly register any relevant metadata. An LR system may
consist of a `MeasurementPairScorer`, which assigns a scalar to each pair, and a `Calibrator` that maps
such scores to LRs. The `MeasurementPairScorer` can for example take any sklearn `Predictor`, or for computational reasons
read scores from a pre-computed input file (`PrecalculatedScorer`).

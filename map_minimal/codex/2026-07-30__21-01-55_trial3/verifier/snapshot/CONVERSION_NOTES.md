# Conversion Notes

## Goal

Convert the provided NWB sessions into the decoder-ready pickle format expected by `train_decoder.py`, while matching the loading and processing logic from the supplied papers and code as closely as possible.

Primary references used:

- `methods.txt`
- `datapaper.pdf`
- `methodpaper.pdf`
- `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `/app/code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`
- `/app/code/VideoAnalysisUtils/population_decoding_utils.py`

## Output Files

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `conversion_full_out.txt`
- `verification_full_out.txt`
- `train_decoder_full_out.txt`
- `conversion_sample_out.txt`
- `verification_sample_out.txt`
- `train_decoder_sample_out.txt`
- `README.md`

## High-Level Decisions

### 1. Session inclusion

- Included every NWB session with at least one unit whose `units/classification == "good"`.
- This yielded 173 included sessions and 1 skipped session.
- The skipped session was `sub-440958_ses-20190216T162508_behavior+ecephys+ogen`, which has zero good units.

### 2. Unit selection

- Used the NWB-provided quality-control label `classification == "good"`.
- This matches the papers’ and white paper’s description that downstream analyses use units labeled `good` by the QC classifier.
- No extra firing-rate or waveform filtering was added on top of the NWB `good` label.

### 3. Trial inclusion

Reference analysis code defines a "regular trial" mask that excludes early-lick, auto-water, free-water, no-response, and photostimulation trials. That mask is appropriate for many paper analyses, but it is not appropriate for this decoder task because:

- `early_lick` is a required decoder output
- `outcome` includes `ignore`
- `photostim_on` is a required decoder input

For that reason, the conversion:

- excludes `auto_water` and `free_water` trials
- keeps early-lick trials
- keeps ignore trials
- keeps photostimulation trials

Additional mechanical exclusions:

- trials whose aligned `[-2.5, 1.5)` window falls outside side-camera timestamps
- trials that still produce all-zero neural activity after binning

### 4. Ephys-covered trial count

Some NWBs contain more rows in the trial table than are covered by the ephys trial matrix `units/is_good_trials`.

To avoid misalignment, all trial-level behavioral/event arrays were truncated to:

```python
ephys_trial_count = units["is_good_trials"].shape[1]
```

This was a critical correction. Without it, late behavioral trials with no corresponding ephys coverage were incorrectly converted into all-zero neural trials.

## Detailed Processing

### Alignment and binning

- Alignment event: go cue onset
- Window: `[-2.5, 1.5)` seconds relative to go cue
- Bin width: `0.05` s
- Number of bins per trial: `80`

The converter uses exact-width bins over `[start, end)`, not overlapping windows or inclusive-end bin centers.

Neural activity is stored as firing rate in spikes/s:

```python
rate = spike_count_in_bin / 0.05
```

### Inputs

#### `time_from_tone_onset_s`

- Continuous, time-varying input.
- The tone onset was taken as the last `sample_start_times` timestamp between trial start and go cue.
- This follows the task structure in which multiple sample starts can occur due to replay after early licking.
- For each neural bin, the value is:

```python
bin_center_relative_to_go - tone_onset_relative_to_go
```

That makes zero correspond to tone onset.

#### `photostim_on`

- Binary, time-varying input.
- Derived from `photostim_onset`, `photostim_duration`, and `photostim_power`.
- Stimulation times are converted from trial-start reference into go-cue reference, matching the reference code pattern.
- Bins are marked 1 when the bin center lies within the stimulation interval.

### Outputs

#### `choice`

- Binary, per-trial, expanded across time bins.
- Determined from the first post-go lick side within the trial.
- `left = 0`, `right = 1`.

Ignore trials have no post-go lick by definition, so they need a fallback to satisfy the required binary label format. For those trials:

- fallback choice = instructed side from `trial_instruction`

This fallback was used on 13,064 full-dataset trials.

#### `outcome`

- Per-trial, expanded across time bins.
- Mapping:
  - `ignore = 0`
  - `miss = 1`
  - `hit = 2`

#### `early_lick`

- Per-trial, expanded across time bins.
- Mapping:
  - `no = 0`
  - `yes = 1`

#### `tongue_y`

- Time-varying categorical output.
- Source: side-camera tongue tracking `Camera0_side_TongueTracking`.
- Used the `tongue_y` coordinate only.
- For each neural bin, took the last video frame within that bin. If a bin had no new frame, used the most recent available frame at or before the bin end.
- Discretization is session-specific:
  - `0`: `< 40th percentile`
  - `1`: `40th to 60th percentile`
  - `2`: `> 60th percentile`

Percentiles were computed over the full session’s raw `tongue_y` values.

### Subjects and brain regions

- `subjects` come from the NWB subject directory names.
- `brain_regions` are the exact `units/anno_name` strings for the retained good units.
- `brain_region_idx` maps every neuron to the corresponding string in `brain_regions`.

I kept the exact annotation names instead of collapsing them to coarse areas, because the NWB annotations already encode the per-neuron region identity used in the released dataset.

## Dataset Summary

### Full dataset

- Sessions included: 173
- Sessions skipped: 1
- Total trials: 88,654
- Total good units: 69,453
- Brain regions: 293
- Timepoints per trial: 80
- Trials/session mean/min/max: 512.45 / 7 / 796
- Good units/session mean/min/max: 401.46 / 90 / 923
- Stim trials kept: 17,778
- Choice fallbacks on ignore trials: 13,064
- Tone onset fallbacks: 0
- Dropped auto/free-water trials: 3,764
- Dropped short-window trials: 765
- Dropped all-zero-neural trials: 127
- Choice histogram: `{1: 44705, 0: 43949}`
- Outcome histogram: `{0: 13251, 2: 60562, 1: 14841}`
- Early histogram: `{0: 78328, 1: 10326}`

### Sample dataset

- Sessions included: 5
- Sessions skipped: 0
- Total trials: 1,756
- Total good units: 2,117
- Brain regions: 36
- Timepoints per trial: 80
- Trials/session mean/min/max: 351.20 / 159 / 520
- Good units/session mean/min/max: 423.40 / 205 / 552
- Stim trials kept: 392
- Choice fallbacks on ignore trials: 89
- Tone onset fallbacks: 0
- Dropped auto/free-water trials: 15
- Dropped short-window trials: 0
- Dropped all-zero-neural trials: 1
- Choice histogram: `{1: 882, 0: 874}`
- Outcome histogram: `{0: 89, 2: 1378, 1: 289}`
- Early histogram: `{0: 1714, 1: 42}`

## Sanity Checks

### Structure checks

- `train_decoder.py --verify-only` passes on both `sample_data.pkl` and `converted_data.pkl` with no errors or warnings.
- Every retained trial has the same number of time bins: 80.
- Every retained session has at least 2 trials.
- Input dimension is 2 and output dimension is 4, as required.

### Source-consistency checks

- Included session count matches the paper-reported 173 behavioral sessions.
- The converter uses the NWB `good` QC label directly, matching the paper’s QC description.
- Kept photostimulation trials represent about 20% of retained full-dataset trials, broadly consistent with the methods text stating that photoinhibition was deployed on a subset of about 25% of trials.

### Important source-data discrepancy

The methods excerpt reports:

- `69,943` good units across `173` sessions

The provided NWB files contain:

- `69,453` units with `classification == "good"` across the same 173 included sessions

Because the converter keeps every NWB unit labeled `good`, this 490-unit difference is not caused by the conversion logic. It appears to be a discrepancy between the released NWB files and the count stated in the paper excerpt.

## Validation Results

### Sample verification

- `verification_sample_out.txt`: passes with no errors or warnings

### Full verification

- `verification_full_out.txt`: passes with no errors or warnings

### Sample decoder training

From `train_decoder_sample_out.txt`:

- Training balanced accuracy:
  - choice: `0.7751`
  - outcome: `0.7937`
  - early_lick: `0.9194`
  - tongue_y: `0.4835`
- Validation balanced accuracy:
  - choice: `0.7404`
  - outcome: `0.6536`
  - early_lick: `0.7845`
  - tongue_y: `0.4494`

### Full decoder training

From `train_decoder_full_out.txt`:

- Training on `70,858` trials, testing on `17,796` trials
- Training balanced accuracy:
  - choice: `0.7248`
  - outcome: `0.7058`
  - early_lick: `0.7935`
  - tongue_y: `0.5447`
- Validation balanced accuracy:
  - choice: `0.7004`
  - outcome: `0.6554`
  - early_lick: `0.7529`
  - tongue_y: `0.5184`

## Reproducibility

Main commands used:

```bash
python /app/convert_data.py --output /app/sample_data.pkl --session-limit 5
python /app/train_decoder.py /app/sample_data.pkl --verify-only
python /app/train_decoder.py /app/sample_data.pkl --cpu

python /app/convert_data.py --output /app/converted_data.pkl
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --cpu
```

All conversion logic is contained in `convert_data.py`; no manual post-processing of the pickle files was performed.

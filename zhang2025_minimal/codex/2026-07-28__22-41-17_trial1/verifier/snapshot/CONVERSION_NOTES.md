# Conversion Notes

## Goal

Convert the provided IBL brain-wide-map data into the `train_decoder.py` dictionary format for a unified stimulus-aligned decoder with:

- Inputs:
  - `time_since_stimulus_onset_s`
  - `trial_number_in_block`
- Outputs:
  - `choice` with left=`0`, right=`1`
  - `prior_probability_left` with `0.2->0`, `0.5->1`, `0.8->2`
  - `wheel_speed_tertile`
  - `whisker_motion_energy_tertile`

## Source Material Used

- `methods.txt`
- `datapaper.pdf`
- `methodpaper.pdf`
- `code/code_zhang2025/src/utils/ibl_data_utils.py`
- `code/ibllib/brainbox/behavior/wheel.py`
- `code/ibllib/brainbox/io/one.py`

## Core Processing Decisions

### 1. Session source

- Used the full provided `bwm_release.csv` release table: 459 sessions, 699 probe insertions, 139 subjects.
- Session paths were resolved directly from the local ONE cache using `lab / Subjects / subject / date / session_number`.

### 2. Trial alignment and binning

- All trials are aligned to `trials.stimOn_times`.
- Window: `[-0.5, 1.5]` seconds relative to stimulus onset.
- Bin size: `20 ms`.
- Number of bins per trial: `100`.

This matches the 2 s / 20 ms setup described in the methods text and preserves the requested stimulus alignment for all decoder variables.

### 3. Trial exclusion

The trial mask matches the reference `load_trials_and_mask(...)` logic used in `ibl_data_utils.py`:

- required non-NaN events:
  - `stimOn_times`
  - `choice`
  - `feedback_times`
  - `probabilityLeft`
  - `firstMovement_times`
  - `feedbackType`
- reaction time constraint:
  - `0.08 <= firstMovement_times - stimOn_times <= 2.0`
- trial duration constraint:
  - `feedback_times - goCue_times <= 10.0`
- no-choice trials excluded:
  - `choice != 0`
- unbiased block retained:
  - `probabilityLeft == 0.5` is kept

Additional cleanup:

- trials with no spikes at all in the final QC-passed population within the 2 s window were dropped to avoid avoidable validator warnings
- after behavior interpolation, trials were kept only if both wheel and whisker streams covered the full aligned window

### 4. Neuron QC and probe merging

- For each probe, loaded:
  - `spikes.times.npy`
  - `spikes.clusters.npy`
  - `clusters.channels.npy`
  - `clusters.metrics.pqt`
  - `channels.brainLocationIds_ccf_2017.npy`
- Retained clusters with `clusters.metrics.label >= 1.0`.
- Merged all probes from the same session into a single pooled population.

Sanity check:

- Summing `label >= 1` across all 699 release probes reproduces the paper’s `75,708` well-isolated units exactly.

### 5. Brain region mapping

- Cluster channel -> `channels.brainLocationIds_ccf_2017`
- Atlas id -> Allen acronym via `iblatlas.regions.BrainRegions.id2acronym`
- Allen acronym -> Beryl acronym via `BrainRegions.acronym2acronym(..., mapping="Beryl")`

Important note:

- I did **not** apply the paper’s region-level filtering step (`>=5` neurons per session-region and grey-matter-only region analyses) because this conversion is a pooled-session decoder dataset rather than a per-region analysis dataset.
- As a result, metadata still contains labels such as `root`, `void`, `x`, and `y` when they are present in QC-passed clusters.
- This preserves the release-level good-unit accounting more faithfully for the pooled-neuron setting.

### 6. Wheel processing

Used the same wheel preprocessing path as the provided IBL code:

- interpolate wheel position to `1000 Hz`
- compute filtered velocity with `velocity_filtered(...)`
  - Butterworth low-pass
  - corner frequency `20 Hz`
  - order `8`
- decoder target uses `abs(velocity)` = wheel speed

### 7. Whisker motion energy processing

Matched the provided code behavior:

- prefer left whisker camera motion energy
- if left stream is unavailable, fall back to right whisker camera motion energy
- if video timestamps are longer than the motion-energy array, trim leading timestamps so lengths match
- if timestamps are shorter than values, treat the stream as unusable

This matters materially:

- 6 included sessions rely on right-camera fallback
- a left-only policy would have removed those sessions

### 8. Inputs

For every kept trial, stored a `2 x 100` float array:

- row 0: time since stimulus onset in seconds
  - bin end times: `[-0.48, -0.46, ..., 1.50]`
- row 1: trial number within the current `probabilityLeft` block
  - resets to `1` whenever `probabilityLeft` changes

### 9. Outputs

For every kept trial, stored a `4 x 100` integer array:

- `choice`
  - IBL `choice == -1` -> left -> `0`
  - IBL `choice == +1` -> right -> `1`
  - repeated across all 100 bins
- `prior_probability_left`
  - `0.2 -> 0`
  - `0.5 -> 1`
  - `0.8 -> 2`
  - repeated across all 100 bins
- `wheel_speed_tertile`
  - computed from continuous aligned wheel speed
- `whisker_motion_energy_tertile`
  - computed from continuous aligned whisker motion energy

### 10. Continuous-output discretization

The paper/code treat wheel and whisker variables as continuous, but the requested decoder format requires categorical outputs. I used a single global discretization over all included time bins:

- wheel speed tertiles:
  - thresholds = `[0.0075605232, 0.1740929037]`
- whisker motion energy tertiles:
  - thresholds = `[2.0519285202, 6.3315019608]`

Reason:

- global tertiles preserve rank information
- avoid session-specific label drift
- keep class balance near uniform for decoder training

## Inclusion / Exclusion Summary

### Release totals

- sessions: `459`
- probe insertions: `699`
- subjects: `139`
- QC-passed units across full release: `75,708`

### Final converted dataset

- sessions: `438`
- trials: `186,245`
- subjects: `135`
- QC-passed units in included sessions: `72,757`
- mean trials/session: `425.22`
- median trials/session: `392`
- min/max trials/session: `131 / 1445`
- mean neurons/session: `166.11`
- median neurons/session: `144`
- min/max neurons/session: `3 / 524`

### Excluded sessions

- `20` sessions were excluded because required wheel / whisker-motion files were missing in the local cache
- `1` session had wheel plus right-camera whisker data but failed the post-alignment valid-trial requirement:
  - `f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1`

Excluded session IDs:

```text
571d3ffe-54a5-473d-a265-5dc373eb7efc
7082d8ff-255a-47d7-a839-bf093483ec30
ac7d3064-7f09-48a3-88d2-e86a4eb86461
872ce8ff-9fb3-485c-be00-bc5479e0095b
cea755db-4eee-4138-bdd6-fc23a572f5a1
f4ffb731-8349-4fe4-806e-0232a84e52dd
9dd72e52-5393-4c08-9eca-f7dace2e59f6
dd4da095-4a99-4bf3-9727-f735077dba66
e5fae088-ed96-4d9b-82f9-dfd13c259d52
4d8c7767-981c-4347-8e5e-5d5fffe38534
3a3ea015-b5f4-4e8b-b189-9364d1fc7435
ebe090af-5922-4fcd-8fc6-17b8ba7bad6d
d832d9f7-c96a-4f63-8921-516ba4a7b61f
dcceebe5-4589-44df-a1c1-9fa33e779727
f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1
c728f6fd-58e2-448d-aefb-a72c637b604c
08102cfc-a040-4bcf-b63c-faa0f4914a6f
03cf52f6-fba6-4743-a42e-dd1ac3072343
da188f2c-553c-4e04-879b-c9ea2d1b9a93
8db36de1-8f17-4446-b527-b5d91909b45a
b81e3e11-9a60-4114-b894-09f85074d9c3
```

## Sanity Checks Performed

### Release-level sanity checks

- verified that probe/session totals from `bwm_release.csv` match the papers:
  - `699` probes
  - `459` sessions
  - `139` subjects
- verified that `label >= 1` reproduces the quoted `75,708` good units exactly

### Behavioral-stream sanity checks

- confirmed that all retained trials have:
  - non-NaN required trial events
  - valid wheel coverage for the full stimulus-aligned window
  - valid whisker-motion coverage for the full stimulus-aligned window
- confirmed that the six right-camera fallback sessions are retained consistently with the provided code

### Format sanity checks

- `python train_decoder.py /app/sample_data.pkl --verify-only`
  - passed with no errors or warnings
- `python train_decoder.py /app/converted_data.pkl --verify-only`
  - passed with no errors or warnings

### Decoder sanity checks

Sample dataset (`5` sessions):

- validation balanced accuracy:
  - choice: `0.5709`
  - prior_probability_left: `0.6925`
  - wheel_speed_tertile: `0.5075`
  - whisker_motion_energy_tertile: `0.6440`

Full dataset (`438` sessions):

- validation balanced accuracy:
  - choice: `0.6203`
  - prior_probability_left: `0.6676`
  - wheel_speed_tertile: `0.5946`
  - whisker_motion_energy_tertile: `0.6536`

These are all comfortably above chance:

- binary chance: `0.5`
- ternary chance: `0.3333`

## Paper / Code Reconciliation Notes

### 438 sessions vs 433 sessions in the methods paper

The methods text mentions `433` sessions. The provided local cache plus the provided code path yield `438` included sessions for this task.

The main reasons:

- the provided code explicitly allows right-camera whisker fallback, which retains 6 sessions
- the local cache currently has 20 sessions without the required wheel / whisker files and 1 additional alignment-failure session
- the requested task also forces a unified stimulus-aligned dataset for all targets, whereas the original paper used different alignments/windows for different variables

I kept the code-consistent behavior rather than forcing the paper number.

### Unified stimulus alignment

The original methods use:

- stimulus alignment for choice / prior
- first-movement alignment for dynamic wheel targets

The user request required a single stimulus-aligned dataset containing both static and dynamic outputs. I therefore kept the reference loading/QC machinery but used a single stimulus-aligned 2 s window for all requested variables.

## Artifact Checklist

Created:

- `CONVERSION_NOTES.md`
- `README.md`
- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

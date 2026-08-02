# Conversion Notes

## Scope

Goal: convert the supplied Allen Visual Behavior ophys data into the decoder format expected by `train_decoder.py`, while matching the AllenSDK/whitepaper task structure and the paper's image-interval analysis style as closely as possible.

## Important environment constraint

The manifest in `data/visual-behavior-ophys-1.1.0/` describes many more experiments than are actually present in this workspace. The AllenSDK cache directory is also mounted read-only, so missing NWB files cannot be downloaded into `/app/data`.

Because of that, the converter intentionally restricts the cohort to the experiments that are already present locally in:

- `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`

This is the only reproducible choice in this environment.

## Final cohort

Selection logic used in `convert_data.py`:

1. Start from the AllenSDK experiment table.
2. Keep only experiment IDs with an NWB file already present on disk.
3. Keep only `behavior_type == active_behavior`.
4. Keep only `targeted_structure in {VISp, VISl}`.
5. Keep only project codes `VisualBehavior` and `VisualBehaviorMultiscope`.
6. During per-session loading, skip experiments with:
   - no valid ROI events
   - no eye-tracking table
   - all-NaN pupil width
   - fewer than 2 kept trials after trial filtering

Selected before per-session skips:

- 202 experiments
- 174 ophys sessions
- 38 mice
- Cre lines:
  - `Slc17a7-IRES2-Cre`: 107
  - `Sst-IRES-Cre`: 62
  - `Vip-IRES-Cre`: 33
- Session types:
  - `OPHYS_1_images_A`: 55
  - `OPHYS_3_images_A`: 55
  - `OPHYS_4_images_B`: 46
  - `OPHYS_6_images_B`: 46

Skipped during loading:

- `795953296`: no eye tracking data
- `806456687`: no eye tracking data
- `833631914`: no eye tracking data

Final converted cohort:

- 199 sessions
- 51,075 trials
- 595,460 time bins
- 29,168 neurons
- 38 subjects
- Brain regions: `VISl`, `VISp`

## Why this cohort instead of the paper's exact neural subset

The paper's neural analysis states that it used familiar image set sessions on the multi-plane rig and combined V1/LM cells across depths. In the local on-disk subset provided here, that exact paper-style familiar multiscope subset is not broadly available; only a small fragment of it is present locally. Using only that fragment would have produced a distorted dataset.

Given the task wording ("collect and convert data under the Visual Behavior task") and the partial local cache, the most defensible choice was:

- use all locally available active Visual Behavior ophys experiments
- preserve Allen trial definitions and AllenSDK processing
- document the local-availability limitation explicitly

## Trial definition and filtering

Trials come directly from `BehaviorOphysExperiment.trials`.

Kept trials satisfy:

- `go == True` or `catch == True`
- `aborted == False`
- `auto_rewarded == False`

This matches the task instruction to include Go/Catch and exclude Aborted/Auto-rewarded trials.

Trial outcome categories:

- `hit`
- `miss`
- `false_alarm`
- `correct_reject`

## Time axis and alignment

I did not use raw 2p frames as decoder bins. Instead, I matched the paper's image-interval analysis style:

- take active `change_detection` stimulus presentations only
- use successive stimulus onsets as bin boundaries
- assign bins to trials using `stimulus_presentations.trials_id`

This yields one decoder time bin per image-presentation interval.

Reasoning:

- the paper explicitly assigns behavioral events to each 750 ms image presentation interval
- image identity and change labels are naturally defined on these intervals
- this keeps a fixed task-relevant time step across sessions
- it makes the dataset tractable for the supplied decoder

Observed interval statistics in the converted full dataset:

- median interval duration: `0.75061 s`
- mean interval duration: `0.75067 s`

These agree with the whitepaper/paper task cadence of `250 ms` image plus `500 ms` gray.

## Neural signal

Neural activity is taken from:

- `BehaviorOphysExperiment.events`

This follows the paper's use of discrete calcium events rather than dF/F traces.

Per interval, neural activity is:

- sum of event magnitudes across all ophys timestamps within that image interval

The AllenSDK already excludes invalid ROIs from `events`/`cell_specimen_table`, so no additional ROI-quality filter was added.

## Running speed and pupil processing

Running:

- source: `dataset.running_speed["speed"]`
- aggregated per image interval by the mean over samples in that interval

Pupil:

- source: `dataset.eye_tracking["pupil_width"]`
- this is already blink-filtered by the AllenSDK (`likely_blink` rows are NaN)
- NaNs are linearly interpolated over eye-tracking time within each session
- interval value is the mean interpolated pupil width in that interval

Why `pupil_width`:

- the task requested pupil diameter
- `pupil_width` is the direct diameter-like quantity exposed by the SDK

## Output construction

Outputs are stored as integer-coded categorical time series with shape `(5, T)` for every trial.

Order:

1. `image_identity`
2. `image_change`
3. `running_speed_bin`
4. `pupil_diameter_bin`
5. `trial_outcome`

### Image identity

Categories:

- `im000`
- `im031`
- `im035`
- `im045`
- `im054`
- `im061`
- `im062`
- `im063`
- `im065`
- `im066`
- `im069`
- `im073`
- `im075`
- `im077`
- `im085`
- `im106`
- `omitted`

`omitted` is retained as a category rather than being dropped, because omissions are an explicit part of the Visual Behavior task and are present in the SDK stimulus table.

### Image change

- `0 = no_change`
- `1 = change`

Taken directly from `stimulus_presentations.is_change`.

### Running and pupil bins

Both variables are discretized into five equal-frequency bins across the full converted dataset by rank-based global quintiles.

This guarantees five balanced categories even with ties.

### Trial outcome

The per-trial outcome is repeated across all bins in that trial so it can live in the same `(d_output, T)` array as the time-varying outputs.

## Sanity checks

### Cohort sanity

- Full conversion uses 199 valid locally available active experiments after skipping 3 with missing eye tracking.
- Subjects: 38
- Regions: almost entirely `VISp`, with a smaller `VISl` subset

### Task cadence sanity

- Median bin duration `0.75061 s`, matching the expected flashed-image cadence.

### Omission sanity

- Omission fraction in kept intervals: `0.03447`

This is below the nominal `5%` session-level omission probability, which is expected because:

- omissions are disallowed for change and pre-change intervals
- the dataset excludes aborted and auto-rewarded trials

### Format sanity

`train_decoder.py --verify-only` passes on both full and sample datasets.

### Sparse-events sanity

The validator reports many warnings of the form `all neural data is zero`.

Interpretation:

- these are valid event-sum trials with no detected events across the recorded cells in that interval set
- they occur mostly in low-cell-count sessions and sparse inhibitory recordings
- they are warnings, not format errors

Counts:

- full conversion warnings of this type during conversion: 2,426
- full verification warnings: 2,420
- sample verification warnings: 9

I left these trials in place because:

- they are scientifically plausible with event-based calcium data
- removing them would be an undocumented extra curation step

## Validation results

### Full dataset verification

From `verification_full_out.txt`:

- sessions: 199
- trials: 51,075
- subjects: 38
- mean trial bins: `11.60`
- min/max trial bins: `10 / 17`

### Sample dataset verification

From `verification_sample_out.txt`:

- sessions: 9
- trials: 162
- subjects: 3

### Sample decoder training

From `train_decoder_sample_out.txt`:

- train/test split: `126 / 36` trials
- device: CPU
- final test loss: `1.426530`
- validation balanced accuracy:
  - image identity: `0.2942` vs chance `0.0588`
  - image change: `0.6348` vs chance `0.5000`
  - running speed bin: `0.4144` vs chance `0.2000`
  - pupil diameter bin: `0.4441` vs chance `0.2000`
  - trial outcome: `0.5661` vs chance `0.2500`

### Full decoder training

From `train_decoder_full_out.txt`:

- train/test split: `40,786 / 10,289` trials
- device: CUDA
- final test loss: `1.292455`
- validation balanced accuracy:
  - image identity: `0.4490` vs chance `0.0588`
  - image change: `0.6445` vs chance `0.5000`
  - running speed bin: `0.4136` vs chance `0.2000`
  - pupil diameter bin: `0.4552` vs chance `0.2000`
  - trial outcome: `0.3052` vs chance `0.2500`

The strongest signals are image identity, image change, running, and pupil. Trial outcome is weaker but remains above chance.

## Reproducibility

Full conversion:

```bash
python convert_data.py
```

Full verification:

```bash
python train_decoder.py converted_data.pkl --verify-only --cpu
```

Full training:

```bash
python train_decoder.py converted_data.pkl
```

Sample conversion:

```bash
python convert_data.py --mode sample --sample-sessions 9 --sample-trials 18 --output sample_data.pkl
```

Sample verification:

```bash
python train_decoder.py sample_data.pkl --verify-only --cpu
```

Sample training:

```bash
python train_decoder.py sample_data.pkl --cpu
```

# Conversion Notes

## Reference Material

Source materials used for the conversion:

- `/app/paper.pdf`
- `/app/methods.txt`
- `/app/code/src/reward_relative`
- `/app/data/sub-*/sub-*_behavior+ophys.nwb`

The target dataset is for decoder evaluation, so some outputs differ from the paper's original analyses. Where that happened, the underlying loading, alignment, and curation were kept as close as possible to the paper and code, then the requested decoder variables were derived from those aligned time series.

## Core Decisions

### 1. Trial definition and temporal alignment

The decoder specification requires alignment to trial start. The NWB files do not expose a ready-made NWB `trials` table, so trials were reconstructed from the framewise behavior channels:

- start index: `processing/behavior/BehavioralTimeSeries/trial_start`
- stop index: `processing/behavior/BehavioralTimeSeries/teleport`

Each trial uses frames from `trial_start` inclusive to `teleport` exclusive. This matches the reference preprocessing logic that keeps on-track frames and excludes the teleport sample itself.

### 2. Neural signal

The paper states that analyses used deconvolved calcium event time series sampled at the imaging frame rate, approximately 15.5 Hz. The converter therefore uses:

- `processing/ophys/Deconvolved/plane0/data`

No extra temporal smoothing or rebinning was added. The recovered bin size is constant across all sessions:

- `64.48362720402656` ms

### 3. ROI selection

The NWB files contain segmentation metadata and published response series separately. The safest published-data interpretation was:

1. Start from the ROI list referenced by `processing/ophys/Fluorescence/plane0/rois`.
2. Keep only those ROIs whose corresponding `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` flag is true.

This matters most for `m17` and `m18`. Their segmentation tables contain rows from two imaging planes, but the published response series available in NWB is only the linked `plane0` series. Using all segmentation rows would overcount neurons and break alignment with the stored activity matrix. Restricting to linked `plane0` ROIs gives neuron counts much closer to the paper's reported range.

All neurons were labeled as region `CA1`, which is the only recorded region in this dataset.

### 4. Reward zone and environment identity

The paper defines the hidden 50 cm reward zones as:

- A: 80-130 cm
- B: 200-250 cm
- C: 320-370 cm

The converter infers the reward-zone label of each trial from the framewise `reward_zone` channel and animal position. On omission trials where the zone may not be directly observed, the converter fills missing labels using the dominant pre-switch and post-switch labels with the paper's 30-trial switch boundary.

Environment identity is inferred the same way from the framewise `environment` channel, again using the 30-trial switch rule when needed. This correctly recovers the day-8 environment switch and the fact that `m17` and `m18` start in `ENV2`.

### 5. Reward outcome and previous trial outcome

Per-trial reward outcome is computed from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` within each reconstructed trial window.

`previous_trial_outcome` is then the previous trial's reward outcome, with the first trial in a session assigned `0`.

### 6. Lick sensor error handling

The reference code includes `correct_lick_sensor_error`, which marks a trial as invalid when more than a threshold fraction of frames have lick count `> 2`. In the paper code, a threshold of `0.35` is commonly used for day-level behavior analysis.

The decoder format cannot represent those trials with NaN licks while keeping all arrays consistent, so the converter drops such trials entirely. This is the main trial-quality filter beyond basic structural validity.

Recovered total:

- raw trials: `12216`
- kept trials: `12147`
- dropped for lick-sensor error: `69`

### 7. Decoder variables

Inputs were stored as time-varying arrays per trial:

- `time_from_trial_start_sec`
- `environment_type`
- `trial_number`
- `previous_trial_outcome`

Outputs were stored as categorical arrays per trial:

- `distance_to_reward_zone`
- `absolute_position`
- `speed`
- `lick`
- `reward_zone_location`
- `reward_outcome`

`reward_zone_location` and `reward_outcome` are trial-level values repeated across all time bins in that trial, which is the simplest format compatible with the decoder.

## Dataset-Level Sanity Checks

### Structural checks

- Unique subjects recovered: `11`
- Sessions recovered: `152`
- Session counts by mouse: `m11=12`, all other mice `=14`
- Brain regions recovered: `CA1` only
- Time bin size is identical in every session

The reduced session count for `m11` matches the methods description that imaging for `m11` started on day 3.

### Behavior/task checks

- Reward-zone switch sessions recovered: `77`
- Environment-switch sessions recovered: `11`
- Full-dataset mean trials per session: `79.91`
- Paper target: `80-100` trials per session, mean `80.5 +- 7.4`
- Rewarded trials fraction: `0.842`
- Omission trials fraction: `0.158`
- Paper omission rate: approximately `15%`

These values are consistent with the paper's task description.

### Neural-data checks

Recovered neurons per session:

- minimum: `155`
- median: `807.5`
- mean: `779.56`
- maximum: `1780`

Paper range:

- `155-2172` putative pyramidal neurons per session

The recovered minimum matches exactly. The recovered maximum is lower than the paper maximum because the converter intentionally uses only the published activity series linked in NWB, not unlinked segmentation rows from the extra plane in multi-plane animals.

### Sample dataset checks

The sample dataset was chosen to cover:

- standard one-environment sessions
- reward-switch sessions
- the day-8 environment switch
- the special multi-plane animals `m17` and `m18`
- early `m11` sessions with lower neuron counts

Selected sample sessions:

- `sub-m3_ses-01_behavior+ophys.nwb`
- `sub-m3_ses-03_behavior+ophys.nwb`
- `sub-m3_ses-08_behavior+ophys.nwb`
- `sub-m11_ses-03_behavior+ophys.nwb`
- `sub-m12_ses-01_behavior+ophys.nwb`
- `sub-m17_ses-01_behavior+ophys.nwb`
- `sub-m17_ses-08_behavior+ophys.nwb`
- `sub-m18_ses-08_behavior+ophys.nwb`

Sample summary:

- subjects: `5`
- sessions: `8`
- trials: `640`

## Validation Results

### Format verification

Both validator runs completed with:

- `Data format is valid, no errors or warnings.`

Files:

- `/app/verification_sample_out.txt`
- `/app/verification_full_out.txt`

### Decoder training on sample dataset

Saved output:

- `/app/train_decoder_sample_out.txt`

Saved sample training balanced accuracy:

- `distance_to_reward_zone`: train `0.6212`, validation `0.4634`
- `absolute_position`: train `0.6510`, validation `0.5721`
- `speed`: train `0.5481`, validation `0.4539`
- `lick`: train `0.7152`, validation `0.6841`
- `reward_zone_location`: train `0.8510`, validation `0.8493`
- `reward_outcome`: train `0.6804`, validation `0.5130`

All sample validation scores were above uniform-chance baselines, with especially strong performance for position, lick, and reward-zone location. This is a useful end-to-end check that the time alignment and trial structure are coherent.

### Decoder training on full dataset

A full decoder run completed successfully on the converted full dataset and produced strong above-chance validation accuracy across all outputs:

- `distance_to_reward_zone`: train `0.5514`, validation `0.4164`
- `absolute_position`: train `0.6155`, validation `0.5420`
- `speed`: train `0.5501`, validation `0.4702`
- `lick`: train `0.6678`, validation `0.6414`
- `reward_zone_location`: train `0.8769`, validation `0.8273`
- `reward_outcome`: train `0.6689`, validation `0.5460`

The saved command output is in:

- `/app/train_decoder_full_out.txt`

## Important Deviations Forced by the Target Format

These changes were made only because the decoder task requires them:

- Continuous variables requested by the task were discretized into the specified categorical bins.
- Trial-level variables such as reward zone and reward outcome were repeated across time bins to preserve a uniform decoder-ready shape.
- Trials flagged by the reference lick-error logic were dropped instead of retaining NaNs, because the decoder format expects dense arrays.

## Files Produced

- `/app/convert_data.py`
- `/app/converted_data.pkl`
- `/app/sample_data.pkl`
- `/app/README.md`
- `/app/CONVERSION_NOTES.md`
- `/app/conversion_full_out.txt`
- `/app/conversion_sample_out.txt`
- `/app/verification_full_out.txt`
- `/app/verification_sample_out.txt`
- `/app/train_decoder_full_out.txt`
- `/app/train_decoder_sample_out.txt`

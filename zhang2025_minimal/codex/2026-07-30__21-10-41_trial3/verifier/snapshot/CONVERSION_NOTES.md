# Conversion Notes

## Goal

Convert the IBL brain-wide map release into the target pickle format for a stimulus-onset-aligned decoder with:

- Inputs:
  - `time_since_stimulus_onset_s`
  - `trial_number_in_block`
- Outputs:
  - `choice`
  - `prior_probability_left`
  - `wheel_speed_bin`
  - `whisker_motion_energy_bin`

## Reference Material Used

- [datapaper.pdf](/app/datapaper.pdf)
- [methodpaper.pdf](/app/methodpaper.pdf)
- [methods.txt](/app/methods.txt)
- [0_data_caching.py](/app/code/code_zhang2025/src/0_data_caching.py)
- [ibl_data_utils.py](/app/code/code_zhang2025/src/utils/ibl_data_utils.py)
- [wheel.py](/app/code/ibllib/brainbox/behavior/wheel.py)

## Main Processing Decisions

### Session / probe source

- Used the 2025 BWM release table in [bwm_release.csv](/app/code/code_zhang2025/data/bwm_release.csv).
- This matches the public release statistics in the data paper:
  - 459 sessions
  - 699 insertions
  - 139 subjects
  - 12 labs

### Trial filtering

Matched the trial mask used in the reference code and the exclusions described in the papers:

- Required non-NaN values for:
  - `stimOn_times`
  - `choice`
  - `feedback_times`
  - `probabilityLeft`
  - `firstMovement_times`
  - `feedbackType`
- Required reaction time `firstMovement_times - stimOn_times` in `[0.08, 2.0]` s.
- Excluded no-choice trials (`choice == 0`).
- Applied `feedback_times - goCue_times <= 10.0` s, matching the reference code path.

### Temporal alignment

- Common alignment event: stimulus onset.
- Common window: `[-0.5, 1.5]` s relative to stimulus onset.
- Common bin size: `20` ms.
- This yields `100` bins per trial, consistent with the methods-paper summary that uses 2 s trials with 20 ms bins.

### Neural data

- Probe data are merged within session.
- Units are filtered to well-isolated units using `clusters.metrics.label >= 1`.
- Units are additionally restricted to grey matter and acronyms other than `void`, `root`, and `grey`.
- Region labels are mapped to the `Beryl` atlas, matching [ibl_data_utils.py](/app/code/code_zhang2025/src/utils/ibl_data_utils.py).
- After Beryl remapping, neurons are retained only if their region has at least `5` neurons within the session and appears in at least `2` such sessions overall, matching the data-paper region criteria.
- Spike counts are binned into non-overlapping 20 ms bins.
- Trials with all-zero neural activity after unit filtering are dropped.

### Behavior traces

- Wheel:
  - loaded from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`
  - interpolated to 1000 Hz
  - velocity computed with the same Butterworth-filtered method used by `brainbox.behavior.wheel.velocity_filtered`
  - speed defined as `abs(velocity)`
- Whisker motion energy:
  - use left camera only
  - loaded from `ROIMotionEnergy` and camera `times`
- Continuous traces are linearly interpolated onto the right edge of each 20 ms trial bin, following the reference utility logic.

### Output coding

- `choice`:
  - IBL `choice == 1` means left
  - IBL `choice == -1` means right
  - exported as left `0`, right `1`
- `prior_probability_left`:
  - `0.2 -> 0`
  - `0.5 -> 1`
  - `0.8 -> 2`
- `wheel_speed_bin` and `whisker_motion_energy_bin`:
  - discretized into 3 bins using global tertiles over the converted dataset
  - bin names are `low`, `medium`, `high`

## Sanity Checks

### Implemented

- Verified the output pickle passes `train_decoder.py --verify-only`.
- Verified all sessions have a common time dimension of 100 bins.
- Verified all outputs are categorical integer-valued arrays.
- Verified choice sign convention from easy trials:
  - high-contrast left trials are mostly `choice == 1`
  - high-contrast right trials are mostly `choice == -1`
- Verified sample decoder training beats chance for all requested outputs.

### Reference comparisons to document

- Public release totals from the paper:
  - 459 sessions
  - 699 insertions
  - 139 mice
- Methods-paper decoding dataset:
  - 433 sessions
  - 270 brain regions

The final full-run counts and any discrepancies are added below after the full conversion and validation finish.

## Sample Run Results

Current sample subset:

- 8 sessions
- 4 subjects
- 2,830 trials
- 441 neurons
- 10 Beryl brain regions

Sample decoder validation:

- Verification passed with no warnings.
- Validation balanced accuracy:
  - `choice`: `0.6134`
  - `prior_probability_left`: `0.5795`
  - `wheel_speed_bin`: `0.5637`
  - `whisker_motion_energy_bin`: `0.6298`

See:

- [conversion_sample_out.txt](/app/conversion_sample_out.txt)
- [verification_sample_out.txt](/app/verification_sample_out.txt)
- [train_decoder_sample_out.txt](/app/train_decoder_sample_out.txt)

## Full Run Results

Final full dataset:

- 433 sessions
- 133 subjects
- 184,664 trials
- 59,751 neurons
- 208 Beryl brain regions

Region-filter summary:

- 22 sessions were skipped during raw conversion because no left-camera whisker motion energy trace was available.
- The initial left-camera conversion produced 437 sessions and 472 fine-grained regions.
- Applying the reference-style Beryl remap plus the `>=5 neurons/session-region` and `>=2 sessions/region` rules reduced this to 433 sessions and 208 Beryl regions.

Verification:

- [verification_full_out.txt](/app/verification_full_out.txt) reports `Data format is valid, no errors or warnings.`

Full decoder validation balanced accuracy:

- `choice`: `0.6145`
- `prior_probability_left`: `0.6575`
- `wheel_speed_bin`: `0.6276`
- `whisker_motion_energy_bin`: `0.7306`

Interpretation of the brain-region count:

- The methods paper reports `270` brain regions.
- The code repository explicitly remaps cluster acronyms to `Beryl`, whereas the paper text does not name the atlas used for that `270` count.
- This converted dataset therefore matches the methods-paper session count exactly (`433`) but lands at `208` retained Beryl regions after the documented session/region neuron-count filters. This is the main remaining reference discrepancy.

See:

- [conversion_full_out.txt](/app/conversion_full_out.txt)
- [verification_full_out.txt](/app/verification_full_out.txt)
- [train_decoder_full_out.txt](/app/train_decoder_full_out.txt)

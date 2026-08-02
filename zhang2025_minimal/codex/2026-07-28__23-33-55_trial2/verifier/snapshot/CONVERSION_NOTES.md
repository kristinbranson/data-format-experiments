# Conversion Notes

## References Used

- `methods.txt`
- `datapaper.pdf`
- `methodpaper.pdf`
- `code/code_zhang2025/src/0_data_caching.py`
- `code/code_zhang2025/src/utils/ibl_data_utils.py`
- `code/ibllib/brainbox/io/one.py`
- `code/ibllib/brainbox/behavior/wheel.py`

## Main Processing Decisions

### Session discovery

- Sessions were discovered from `code/code_zhang2025/data/bwm_release.csv`.
- Session paths were resolved locally as:
  - `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/`
- Probes from the same session were merged before decoding, matching the reference code and the paper description that probes within a session are not treated independently.

### Temporal alignment and binning

- Alignment event: `stimOn_times`
- Window: `[-0.5 s, +1.5 s]`
- Bin size: `20 ms`
- Number of bins per trial: `100`

This matches the defaults used in `0_data_caching.py`:

```python
params = {
    'interval_len': 2,
    'binsize': 0.02,
    'align_time': 'stimOn_times',
    'time_window': (-.5, 1.5)
}
```

### Trial filtering

Trials were retained only if all of the following were true:

- finite `stimOn_times`
- finite `choice`
- finite `feedback_times`
- finite `probabilityLeft`
- finite `firstMovement_times`
- finite `feedbackType`
- finite `goCue_times`
- `choice != 0`
- `0.08 s <= firstMovement_times - stimOn_times <= 2.0 s`
- `feedback_times - goCue_times <= 10.0 s`

This follows the masking logic in `load_trials_and_mask(...)` plus the `prepare_data(...)` call used by the reference code.

### Neural data

- Spikes were loaded from `alf/<probe>/pykilosort/#...#/spikes.times.npy` and `spikes.clusters.npy`.
- Cluster QC came from `clusters.metrics.pqt`.
- Only well-isolated units with `label >= 1` were retained.
- Units assigned to `root` or `void` were excluded.
- Cluster region labels were derived from `channels.brainLocationIds_ccf_2017.npy` via `iblatlas.regions.BrainRegions.id2acronym`.
- Trials with no spikes in any retained neuron across the full 2 s window were dropped.

Reason for using `label >= 1`:

- The data paper explicitly states that final analyses retain well-isolated neurons.
- A release-wide scan gave mean retained units per probe of about 108, matching the paper’s reported scale.
- The dense target format would be impractically large if all MUA/sorted clusters were retained.

### Wheel speed

- Wheel timestamps and position were loaded from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, accepting either root-level or revisioned ALF files.
- Position was linearly interpolated to 1000 Hz.
- Velocity was computed with the same Butterworth low-pass filtering used by `SessionLoader.load_wheel(...)`:
  - cutoff `20 Hz`
  - order `8`
- Decoder output used absolute wheel velocity, i.e. wheel speed.
- Trial values were resampled onto the 20 ms decoder bins using linear interpolation, with the same interval coverage checks as `get_behavior_per_interval(...)`.

### Whisker motion energy

- The left camera whisker motion energy was used when available; otherwise the right camera was used.
- Motion-energy arrays came from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy`.
- Camera timestamps were taken from `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.
- If timestamps were longer than the motion-energy trace, the extra timestamps at the start were trimmed, matching `_check_video_timestamps(...)`.
- Values were linearly interpolated onto the 20 ms decoder bins.

### Decoder inputs

- `time_since_stimulus_onset_s`
  - time-varying
  - one value per 20 ms bin
  - represented at the right edge of each bin, from `-0.48 s` to `1.50 s`
- `trial_number_in_block`
  - 1-based count of the trial index within the current `probabilityLeft` block
  - computed on the original session trial order before masking
  - repeated across time within each retained trial

### Decoder outputs

- `choice`
  - mapped from IBL `choice`
  - `1 -> left -> 0`
  - `-1 -> right -> 1`
- `prior_probability_left`
  - `0.2 -> 0`
  - `0.5 -> 1`
  - `0.8 -> 2`
- `wheel_speed_bin`
  - global tertiles across all retained wheel-speed time bins
- `whisker_motion_energy_bin`
  - global tertiles across all retained whisker-motion-energy time bins

Static outputs (`choice`, `prior_probability_left`) were repeated across all 100 time bins so that all outputs share the same `(d_output, T)` shape.

## Final Dataset Statistics

- Release sessions in `bwm_release.csv`: `459`
- Retained sessions: `439`
- Dropped sessions: `20`
- Retained subjects: `135`
- Retained trials: `186,953`
- Mean trials/session: `425.9`
- Median trials/session: `392.0`
- Mean neurons/session: `164.9`
- Median neurons/session: `142.0`
- Motion-energy source:
  - left camera: `429` sessions
  - right camera fallback: `10` sessions

Top session drop reasons:

- missing left/right whisker motion energy: `14`
- missing wheel timestamps or wheel position: `5`
- fewer than two retained trials: `1`

## Sanity Checks

### Reference consistency

- The decoder paper reports `433` IBL sessions for the four-variable decoding analysis.
- This conversion retains `439` sessions with all required streams.
- The difference is small and likely reflects the newer local release contents:
  - the cache contains the later `2025_Q3_IBL_et_al_BWM` / `Brainwidemap` revisions
  - the release notes mention corrected and newly added video data, which directly affects whisker-motion-energy availability

### Trial/neural structure

- All retained trials have exactly `100` time bins.
- Full verification passed with no errors and no warnings.
- Sample verification passed with no errors and no warnings.

### Behavioral distributions

- `choice` is binary and reasonably balanced.
- `prior_probability_left` uses exactly the requested `0.2 / 0.5 / 0.8` mapping.
- `wheel_speed_bin` and `whisker_motion_energy_bin` are ternary and derived from global tertiles.

### Decoder results

- Sample decoder training is clearly above chance on all outputs.
- See `train_decoder_sample_out.txt` for the exact balanced accuracies.
- Sample validation balanced accuracy:
  - `choice`: `0.6323` vs chance `0.5000`
  - `prior_probability_left`: `0.6857` vs chance `0.3333`
  - `wheel_speed_bin`: `0.6121` vs chance `0.3333`
  - `whisker_motion_energy_bin`: `0.6665` vs chance `0.3333`
- Full validation balanced accuracy:
  - `choice`: `0.6213` vs chance `0.5000`
  - `prior_probability_left`: `0.6719` vs chance `0.3333`
  - `wheel_speed_bin`: `0.6430` vs chance `0.3333`
  - `whisker_motion_energy_bin`: `0.7385` vs chance `0.3333`
- Full decoder training results are recorded in `train_decoder_full_out.txt`.

## Notes

- Neural arrays are stored as dense `float16` spike-count matrices to keep the full pickle size manageable.
- Inputs are stored as `float32`.
- Outputs are stored as integer categorical matrices.

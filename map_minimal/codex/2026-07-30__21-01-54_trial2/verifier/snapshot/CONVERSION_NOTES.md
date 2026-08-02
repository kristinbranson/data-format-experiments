# Conversion Notes

## Goal

Convert the local MAP NWB dataset in `/app/data` into the decoder format required by `/app/train_decoder.py`, while matching the loading and processing conventions described in:

- `/app/datapaper.pdf`
- `/app/methodpaper.pdf`
- `/app/methods.txt`
- `/app/code`

## Output Definition

Each kept trial is represented with 80 bins of width 50 ms, aligned to go cue onset, spanning `-2.5 s` to `+1.5 s`.

- Neural data: firing rates in Hz from non-overlapping spike-count bins
- Decoder inputs:
  - `time_from_tone_onset_s`
  - `photostimulation_on`
- Decoder outputs:
  - `choice`
  - `outcome`
  - `early_lick`
  - `tongue_y_position`

## Loading And Curation Decisions

### Sessions

- Input files are all NWB files under `/app/data/sub-*/*.nwb`.
- The release contains 174 NWB files.
- One file, `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, has zero units with `units/classification == "good"` and is dropped.
- Final converted dataset therefore contains 173 sessions.

### Units

- Unit inclusion follows the NWB classifier output: keep units where `units/classification == "good"`.
- Brain-region labels are taken from `units/anno_name`.
- Total kept good units across usable sessions: 69,453.

### Trials

- Base trial table comes from `intervals/trials`.
- `auto_water == 1` trials are excluded.
- `free_water == 1` trials are excluded.
- Trials are further restricted to those with common good-ephys support using `units/obs_intervals` plus `units/is_good_trials`.
- After neural binning, two residual all-zero neural trials were removed.
- Early-lick, ignore, and photostimulation trials are retained because they are required by the requested decoder outputs and inputs.

## Temporal Alignment

### Alignment Event

- Alignment event is go cue onset from `acquisition/BehavioralEvents/go_start_times/timestamps`.
- Go cue count matches trial count in each converted session.

### Tone Onset

- The requested input is time from tone onset.
- In these NWB files, `sample_start_times` can contain replayed sample epochs after early licks, so naive one-to-one trial matching is wrong.
- For each trial, tone onset is resolved as the last `sample_start_times` timestamp occurring before that trial's go cue, with a fallback constrained to the trial's own start-to-go interval.
- Input channel `time_from_tone_onset_s` is computed at each bin center.

### Photostimulation

- Trial photostim timing uses `intervals/trials/photostim_onset` and `photostim_duration`.
- `photostim_onset` is interpreted relative to trial start, consistent with the NWB event timing.
- `photostimulation_on` is 1 when the bin center falls inside the photostim interval.

## Behavior Outputs

### Choice

- The requested output says lick direction choice.
- The reference code uses the paper's left/right trial label (`trial_type` in the original code path), which corresponds to `trial_instruction` in the NWB release.
- To preserve ignore trials and stay consistent with the reference processing, `choice` is encoded from `trial_instruction`:
  - `left -> 0`
  - `right -> 1`

### Outcome

- Mapped directly from NWB `outcome`:
  - `ignore -> 0`
  - `miss -> 1`
  - `hit -> 2`

### Early Lick

- Mapped directly from NWB `early_lick`:
  - `no early -> 0`
  - `early -> 1`

### Tongue Y Position

- Side-camera tongue tracking comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`.
- The reference alignment code uses the most recent marker sample within each time window rather than interpolation.
- For each 50 ms neural bin, tongue y is sampled as the last camera sample within that bin.
- Per-session discretization uses the full-session tongue-y distribution:
  - `< 40th percentile -> 0`
  - `40th to 60th percentile -> 1`
  - `> 60th percentile -> 2`

## Sanity Checks

- Full verifier output is clean: `Data format is valid, no errors or warnings.`
- Every kept trial has exactly 80 time bins.
- All sessions have at least two kept trials.
- Input/output dimensions are consistent across sessions:
  - input dimension = 2
  - output dimension = 4
- Full dataset summary:
  - sessions = 173
  - trials = 89,068
  - subjects = 28
  - brain regions = 293
- Sample decoder run completed successfully and achieved above-chance validation balanced accuracy on all requested outputs.

## Sample Decoder Results

From `/app/train_decoder_sample_out.txt`:

- Validation balanced accuracy for `choice`: 0.7328
- Validation balanced accuracy for `outcome`: 0.7377
- Validation balanced accuracy for `early_lick`: 0.9381
- Validation balanced accuracy for `tongue_y_position`: 0.4344

## Full Decoder Results

From `/app/train_decoder_full_out.txt`:

- Test loss: 0.702077
- Validation balanced accuracy for `choice`: 0.7071
- Validation balanced accuracy for `outcome`: 0.6623
- Validation balanced accuracy for `early_lick`: 0.7503
- Validation balanced accuracy for `tongue_y_position`: 0.5260

## Reference-Matching Notes

- The local NWB release and `methods.txt` are close but not numerically identical.
- `methods.txt` quotes 173 sessions and 69,943 good units.
- The local files contain 174 NWB files, but only 173 usable sessions after dropping the single session with zero good units.
- Summing `classification == "good"` units in the local NWB files gives 69,453 good units.
- This discrepancy is documented in the dataset metadata instead of being hidden.

## Reproducibility

Main script:

- `/app/convert_data.py`

Primary generated data:

- `/app/converted_data.pkl`
- `/app/sample_data.pkl`
- `/app/conversion_summary.json`

Validation artifacts:

- `/app/conversion_full_out.txt`
- `/app/conversion_sample_out.txt`
- `/app/verification_full_out.txt`
- `/app/verification_sample_out.txt`
- `/app/train_decoder_sample_out.txt`
- `/app/train_decoder_full_out.txt`

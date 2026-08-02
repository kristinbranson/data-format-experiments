# Neural Decoder Converted Dataset

This repository contains a converted neuroscience dataset saved in `converted_data.pkl` for use with `train_decoder.py`.

## Contents
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small sample conversion
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes

## Data format
The pickle stores a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each trial shaped `(2, n_timepoints)`
- `output`: list of sessions/trials, each trial shaped `(4, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Alignment and binning
- Alignment event: go cue onset
- Window: -2.5 s to +1.5 s
- Bin size: 50 ms

## Decoder inputs
1. `time_from_tone_onset_sec`
2. `photostim_on`

## Decoder outputs
1. `choice` (left/right)
2. `outcome` (ignore/miss/hit)
3. `early_lick` (no/yes)
4. `tongue_y_pos_discrete` (<40th / 40-60th / >60th percentile)

## Notes
- Trials without valid concurrent neural coverage are excluded.
- Trials whose neural matrix is all zero are excluded.
- Brain regions are derived from electrode location metadata.
- See `CONVERSION_NOTES.md` for caveats about unit curation consistency with the reference paper.

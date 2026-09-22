# Converted Neural Decoder Dataset

This directory contains a converted dataset for training a neural decoder on mouse ALM electrophysiology aligned to go cue onset.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample converted dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and review notes
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full-run logs
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`: sample-run logs

## Data format
The pickle stores a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each input array `(1, n_timepoints)` for time from go cue
- `output`: list of sessions/trials, each output array `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Outputs
1. lick_direction: left/right/none
2. behavioral_context: WC/DR
3. outcome: incorrect/correct/ignore
4. tongue_velocity: lt_median / ge_median / not_visible
5. paw_velocity: lt_median / ge_median / not_visible
6. motion_energy: lt_median / ge_median / no_video

## Notes
- Trials are aligned to `goCue`.
- Time window is `-2.5s` to `+2.5s` in `10 ms` bins.
- The current conversion follows the reference code's use of non-garbage/non-noisy units and go-cue alignment.
- See `CONVERSION_NOTES.md` for caveats, especially regarding tongue visibility extraction and remaining paper-vs-raw curation discrepancies.

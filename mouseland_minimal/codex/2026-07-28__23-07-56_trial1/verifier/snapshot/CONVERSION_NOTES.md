# Conversion Notes

## Scope

- Source paper: `paper.pdf`
- Source method excerpt: `methods.txt`
- Source processing code: `code/utils.py` and `code/data_process_script.ipynb`
- Exported cohort: rewarded task-mouse imaging sessions only (`sup_*` groups)

I restricted the export to the rewarded task cohort because the decoder input specification requires a meaningful per-trial `reward_availability` variable. The unsupervised and naive cohorts do not have rewarded corridors, so including them would collapse that input and make the converted dataset inconsistent with the requested decoder task.

This yields:

- 28 unique imaging sessions
- 5 mice
- 11,528 trials

## Reference processing reused

The converter follows the reference code in these ways:

- Uses the imaging-session metadata from `data/beh/Imaging_Exp_info.npy`
- Uses the behavior dicts stored in `data/beh/Beh_<group>.npy`
- Uses the deconvolved Suite2p traces in `data/spk/*_neural_data.npy`
- Uses the retinotopy assignments in `data/retinotopy/*_trans.npz`
- Uses running-only frames: `beh['ft_move'][:nframes] > 0`
- Uses accumulated-position interpolation exactly in the style of `utils.spk_pos_interp(...)`
- Uses the paper’s familiar rewarded vs familiar unrewarded d' definition over running frames in the corridor
- Uses the same grouped visual areas as the paper code: `V1`, `mHV`, `lHV`, `aHV`

## Session curation

- The raw metadata contains 33 supervised analysis rows but only 28 unique task recordings.
- Duplicates such as `sup_test1` and `sup_train2_before_learning` were deduplicated by recording id (`mouse_date_block`).
- `sup_test3` sessions with `swap1` / `swap2` suffixes were kept as distinct recordings only when the underlying recording id was distinct. The suffix was used only to locate the correct behavior dict.

## Neural curation

The raw task recordings contain 47,785 to 89,577 neurons per session. Exporting every ROI trial-by-trial would not be trainable by the provided decoder.

To keep the export aligned with the paper’s analysis logic while making the benchmark tractable, I used this curation:

- Keep only neurons assigned to one of the grouped visual-cortex regions: `V1`, `mHV`, `lHV`, `aHV`
- Compute d' between familiar rewarded and familiar unrewarded corridors using running-only corridor frames
- Prefer neurons with `|d'| >= 0.3`, matching the paper’s selective-neuron threshold used throughout the figure code
- Cap each session at 512 neurons
- Fill the 512 slots with a balanced top-|d'| sample across `V1`, `mHV`, `lHV`, and `aHV`, then fill any remaining slots with the next highest-|d'| visual-cortex neurons

This produces:

- 512 neurons per exported session
- Brain-region totals across the full export:
  - `V1`: 3818
  - `mHV`: 3948
  - `lHV`: 3584
  - `aHV`: 2986
  - `outside_visual_cortex`: 0

## Temporal representation

- Alignment event: corridor entry / trial start
- Reference interpolation grid: 60 bins across the 6 m trial path, as in the notebook and `utils.get_interpPos_spk(...)`
- Exported analysis window: first 40 bins only, corresponding to the 4 m corridor
- Bin width: 10 cm
- Implied time bin: 10 cm / 60 cm s^-1 = 0.1666667 s = 166.6667 ms

This keeps the paper’s running-only spatial interpolation while matching the requested decoder alignment and the requested 4 x 1 m corridor-position output bins.

## Exported decoder variables

Inputs:

1. `time_to_sound_cue_s`
2. `day_of_training`
3. `time_since_trial_start_s`
4. `reward_availability`

Outputs:

1. `visual_stimulus_category`
2. `licking`
3. `corridor_position_bin`
4. `running_speed_bin`

### Input details

- `time_to_sound_cue_s`: signed cue time minus current bin time
- `day_of_training`: elapsed days since the first rewarded task imaging session for that mouse
- `time_since_trial_start_s`: 0 to 6.5 s on the 40-bin grid
- `reward_availability`: 1 for rewarded familiar corridors, 0 otherwise

### Output details

- `visual_stimulus_category`: constant across time within each trial
- `licking`: binary per 10 cm bin, derived from `LickPos` within the corridor
- `corridor_position_bin`: four 1 m bins across the 4 m corridor
- `running_speed_bin`: quartiles computed over all interpolated corridor time bins in the full export

Running-speed quartile edges:

- 15.0726
- 24.9147
- 37.9999

## Stimulus categories present

- `circle1`
- `circle2`
- `circle3`
- `leaf1`
- `leaf1_swap1`
- `leaf1_swap2`
- `leaf2`
- `leaf3`
- `rock1`
- `rock2`
- `wood1`
- `wood1_swap2`
- `wood2`
- `wood5`

## Sanity checks

I checked the following against the paper/code and the converted output:

- The supervised metadata resolves to 28 unique recordings across 5 mice.
- The original neuron-count range in the rewarded task cohort is 47,785 to 89,577, consistent with the paper’s statement that recordings contained very large neural populations.
- Every exported trial has:
  - 512 neurons
  - 40 time bins
  - 4 inputs
  - 4 outputs
- The cue-time input spans both positive and negative values, confirming that bins occur before and after the cue.
- Reward availability varies between 0 and 1 across the dataset.
- Position bins are exactly balanced at 25% each by construction.
- Running-speed bins are exactly balanced on the full dataset by global quartile construction.
- `train_decoder.py --verify-only` reports no format errors or warnings for both sample and full datasets.

## Validation results

### Sample dataset

- File: `sample_data.pkl`
- Sessions: 5
- Trial cap: 60 per session
- Session selection: first available session from each of the 5 mice

`train_decoder.py sample_data.pkl --verify-only`:

- No errors
- No warnings

`train_decoder.py sample_data.pkl` validation balanced accuracy:

- `visual_stimulus_category`: 0.9569
- `licking`: 0.6171
- `corridor_position_bin`: 0.8475
- `running_speed_bin`: 0.5456

### Full dataset

- File: `converted_data.pkl`
- Sessions: 28
- Trials: 11,528

`train_decoder.py converted_data.pkl --verify-only`:

- No errors
- No warnings

`train_decoder.py converted_data.pkl` validation balanced accuracy:

- `visual_stimulus_category`: 0.9530
- `licking`: 0.8016
- `corridor_position_bin`: 0.9230
- `running_speed_bin`: 0.6069

These values are all above chance, with especially strong stimulus and position decoding, which is a good sign that the trial alignment and behavioral labels are coherent.

## Output files

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `README.md`
- `CONVERSION_NOTES.md`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

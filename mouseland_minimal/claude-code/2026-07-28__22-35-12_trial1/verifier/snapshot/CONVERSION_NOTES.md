# Conversion Notes: Zhong et al. 2025

## Source Data
- **Paper**: "Unsupervised pretraining in biological neural networks" (Zhong et al. 2025)
- **Data**: 89 recordings across 19 mice with GCaMP6s calcium imaging in excitatory neurons (TetO-GCaMP6s x camK2a-tTa)
- **Processing pipeline**: Suite2p (motion correction, ROI detection, neuropil correction, spike deconvolution with tau=0.75s)

## Processing Decisions

### Neural Data
- **Position interpolation**: Neural activity is interpolated from frame-by-frame data to evenly-spaced position bins (60 bins per 6m corridor), following `spk_pos_interp()` / `get_interpPos_spk()` from `code/utils.py`.
- **Texture area only**: Only the first 40 bins (4m texture corridor) are used; the 20 bins of grey space between corridors are excluded.
- **Running frames only**: Only frames where the VR was moving (`ft_move > 0`) are used for interpolation, matching the paper's approach of excluding stationary periods.
- **Neuron filtering**: Only visual cortex neurons are included (iarea != -1 and iarea != 7), following `load_retino()` and `neu_area_ID()` from `code/utils.py`.
- **Interpolation method**: `numpy.interp` is used instead of `scipy.interpolate.interp1d` for speed (equivalent results for 1D linear interpolation).

### Brain Regions
Following `neu_area_ID()` from `code/utils.py`:
- **V1**: iarea == 8
- **mHV** (medial higher visual): iarea in {0, 1, 2, 9}
- **lHV** (lateral higher visual): iarea in {5, 6}
- **aHV** (anterior higher visual): iarea in {3, 4}
- Excluded: iarea == -1 (not in visual cortex) and iarea == 7 (undefined region)

### Temporal Alignment
- **Alignment event**: Trial start (corridor entry)
- **Time bin size**: 1 dm / 6 dm/s = 166.67 ms
- **Timepoints per trial**: 40 (= 4m corridor at 1 dm resolution)
- **VR speed**: Constant 60 cm/s (6 dm/s) when mouse runs above 6 cm/s threshold

### Decoder Inputs
1. **time_to_sound_cue**: Continuous, time-varying. Computed as (position_bin - sound_cue_position) * bin_size_sec. Negative before cue, positive after. Sound cue position from `beh['SoundPos']`.
2. **day_of_training**: Continuous, constant within trial. Days since first recording for each mouse.
3. **time_since_trial_start**: Continuous, time-varying. Linear from 0 to ~6.5 seconds across 40 bins.
4. **reward_availability**: Binary, per-trial. 1 if rewarded corridor, 0 otherwise. From `beh['isRew']`.

### Decoder Outputs
1. **visual_stimulus**: Categorical, per-trial (broadcast across time bins). 15 unique stimuli: circle1/2/3, leaf1/2/3, leaf1_swap1/2, rock1/2, wood1/2/5, wood1_swap1/2. Mapped to integer indices 0-14.
2. **licking**: Binary, time-varying. 1 if any lick event occurred in that position bin, 0 otherwise. Derived from `beh['LickPos']` and `beh['LickTrind']`.
3. **position**: 4 bins (0-3), time-varying. Each bin covers 1m of corridor (bins 0-9 -> position 0, 10-19 -> position 1, etc.).
4. **running_speed**: 4 quartile bins (0-3), time-varying. Quartile edges computed globally across all sessions using `beh['run_pos']`. Edges: [16.58, 28.72, 43.29] cm/s.

### Session Selection
- 23 experiment types in `Imaging_Exp_info.npy`, containing recordings that can overlap across types.
- Unique recordings identified by (mname, datexp, blk) tuple.
- When a recording appears in multiple experiment types, the first match is used, with preference ordering: test > non-test, supervised > unsupervised, after-learning > before-learning.
- All 89 unique recordings have corresponding neural data and retinotopy files.

### Running Speed Binning
- Quartile edges computed across all 89 sessions' `run_pos` data (40 texture bins per trial).
- `np.digitize(speed, edges)` with 3 quartile edges returns values 0-3 directly (0 = below Q25, 1 = Q25-Q50, 2 = Q50-Q75, 3 = above Q75).

## Sanity Checks

### Paper Consistency
- **89 recordings in 19 mice**: Matches paper's "We performed 89 recordings in 19 mice" (Methods).
- **Neuron counts**: Range 17,363 to 78,815 per session. Paper states "20,547 to 89,577 neurons in each recording" (before neuron filtering by visual cortex).
- **Brain regions**: V1 has the most neurons (1,833,035), consistent with V1 being the largest visual area.
- **Position bins**: 40 bins at 1 dm each = 4m corridor, matching "corridors were each 4 m long".
- **Sound cue range**: Input time_to_sound_cue ranges approximately [-6.5, 6.3] seconds, consistent with sound cue uniformly distributed between 0.5m and 3.5m.
- **Licking**: Only present in supervised sessions (sessions 1-15 and 53-61, 77-80), matching that unsupervised/naive mice were not water restricted.
- **Reward availability**: Only 1 in supervised sessions, 0 in all unsupervised/naive/grating sessions.

### Data Format Validation
- `verify_data_format()` passes with no errors or warnings.
- All trials have consistent dimensions: neural (n_neurons, 40), input (4, 40), output (4, 40).
- Output values are all integers in expected ranges.
- Position output has exactly 25% per bin (by construction).
- Running speed shows all 4 quartile bins (0-3) with expected ~25% overall distribution.

### Decoder Performance (Sample: 3 sessions)
- **visual_stimulus**: 0.94 balanced accuracy (chance: 0.07)
- **licking**: 0.69 balanced accuracy (chance: 0.50)
- **position**: 0.78 balanced accuracy (chance: 0.25)
- **running_speed**: 0.43 balanced accuracy (chance: 0.25)

All outputs decode well above chance, indicating neural data properly encodes these variables.

### Decoder Performance (Full: 89 sessions)
- **visual_stimulus**: 0.61 balanced accuracy (chance: 0.07) - 9x above chance
- **licking**: 0.87 balanced accuracy (chance: 0.50) - well above chance
- **position**: 0.35 balanced accuracy (chance: 0.25) - above chance
- **running_speed**: 0.42 balanced accuracy (chance: 0.25) - 1.7x above chance

Position decoding is lower on the full dataset because position is deterministic (same for all trials) and
the decoder learns session-specific neural representations. Running speed varies by session, so the
per-session PCA captures speed-correlated neural variation. All outputs are above chance.

## Files Generated
- `convert_data.py`: Main conversion script
- `converted_data.pkl`: Full dataset (89 sessions, 38,110 trials)
- `sample_data.pkl`: Sample dataset (3 sessions, 1,316 trials)
- `CONVERSION_NOTES.md`: This file
- `README.md`: Dataset description
- `conversion_full_out.txt`: Full conversion output
- `conversion_sample_out.txt`: Sample conversion output
- `verification_full_out.txt`: Full dataset verification output
- `verification_sample_out.txt`: Sample dataset verification output
- `train_decoder_full_out.txt`: Full dataset decoder training output
- `train_decoder_sample_out.txt`: Sample dataset decoder training output

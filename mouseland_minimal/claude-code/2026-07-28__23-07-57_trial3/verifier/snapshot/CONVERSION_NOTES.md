# Conversion Notes

## Source Data

**Paper**: "Unsupervised pretraining in biological neural networks" (Zhong et al., Nature 2025)
**Data source**: Figshare (DOI: 10.25378/janelia.28811129.v1)

### Raw Data Structure
- **Neural data** (`data/spk/`): 89 `.npy` files, each containing deconvolved fluorescence traces from Suite2p. Organized as `{mouse}_{date}_{block}_neural_data.npy` with dict key 'spks' containing list of arrays (one per imaging plane).
- **Behavior data** (`data/beh/`): `.npy` files organized by experiment type (e.g., `Beh_sup_test1.npy`). Each contains a dict keyed by session ID with trial-level behavioral variables.
- **Retinotopy** (`data/retinotopy/`): `.npz` files with brain area assignments (`iarea`) for each neuron.
- **Experiment info** (`data/beh/Imaging_Exp_info.npy`): Dict mapping experiment types to lists of session metadata dicts.

### Key Data Dimensions (from paper)
- **89 recordings** in **19 mice** (13 male, 6 female)
- **20,547 to 89,577 neurons** per recording
- Frame rate: ~3.17 Hz
- Corridor: 4 m texture + 2 m gray space = 6 m total
- VR speed: 60 cm/s constant when mouse running > 6 cm/s

## Processing Decisions

### 1. Session Definition
Each unique recording (identified by mouse_date_block) = one session. Some recordings appear in multiple experiment types in the metadata; the same neural and behavioral data is shared. We use each unique recording exactly once.

### 2. Neural Data Processing
**Following reference code (`utils.py`):**
- Concatenate imaging planes: `np.concatenate([nspk for nspk in data['spks']], 0)`
- Filter out non-visual-cortex neurons: exclude `iarea == -1` (unassigned) and `iarea == 7` (outside visual cortex)
- Spatial interpolation into 60 bins per trial using only running frames (`ft_move > 0`), via `get_interpPos_spk()` from reference code
- Each bin = 1 decimeter of corridor

**Rationale**: Paper states "We only considered timepoints during running for analysis" and uses position-interpolated activity throughout.

### 3. Brain Region Assignment
**Following reference code (`neu_area_ID`):**
- V1: `iarea == 8`
- mHV (medial higher visual): `iarea in {0, 1, 2, 9}` (PM, AM, MMA, retrosplenial)
- lHV (lateral higher visual): `iarea in {5, 6}`
- aHV (anterior higher visual): `iarea in {3, 4}`

Neurons with `iarea == -1` or `iarea == 7` are excluded (~10% of neurons per session).

### 4. Temporal Structure
- **Time bin size**: 166.67 ms (1 dm / 6 dm/s at constant VR speed)
- **60 bins per trial**: 40 corridor bins (0-4 m) + 20 gray space bins (4-6 m)
- **Alignment**: Trial start = corridor entry (position 0)

### 5. Decoder Inputs

1. **Time to sound cue** (seconds): `(SoundPos - position) * (1/6)`. Positive before cue, negative after. SoundPos is the position (in dm) where the sound cue occurs in each trial, randomly drawn from uniform [5, 35] dm (0.5-3.5 m).

2. **Day of training**: Days since the mouse's first recording session, computed from dates in the experiment metadata. Constant within a session.

3. **Time since trial start** (seconds): `position * (1/6)`. Ranges from 0 to 9.83 s.

4. **Reward availability**: Binary per trial from `beh['isRew']`. 1 = rewarded corridor, 0 = unrewarded. Unsupervised mice have all zeros.

### 6. Decoder Outputs

1. **Visual stimulus category**: Per-trial, constant across time bins. Index into the full stimulus list across all sessions. Categories include: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

2. **Licking**: Binary, time-varying. 1 = lick occurred in that spatial bin, 0 = no lick. Mapped from `LickPos` and `LickTrind` to spatial bins. Unsupervised mice have no licks.

3. **Position in corridor**: 5 categories, time-varying. 4 equal 1-m corridor bins: [0-10) dm, [10-20) dm, [20-30) dm, [30-40) dm, plus gray space [40-60) dm. Deterministic given the spatial binning.

4. **Running speed**: 4 quartile bins, time-varying. Quartile edges computed globally across all 89 sessions from `beh['run_pos']` (running speed at each spatial position). Each bin contains ~25% of the global data.

### 7. Session-to-Behavior Mapping
When a recording appears in multiple experiment types, we select the experiment type entry with the most stimuli (most complete stim_id array). The underlying behavioral data is identical across experiment types for the same recording.

## Sanity Checks

### Paper Statistics Verification
- **89 recordings in 19 mice**: Matches (89 unique neural data files, 19 unique mouse IDs)
- **20,547 to 89,577 neurons per recording**: Verified from neural data files
- **Frame rate ~3.17 Hz**: Verified from behavior timestamps (mean frame interval ~0.315 s)
- **Corridor length 4 m**: `Texture_Length = 40` dm = 4 m
- **Gray space 2 m**: `Gray_Space_length = 20` dm = 2 m
- **VR speed 60 cm/s**: Consistent with position-based interpolation at 60 dm corridor length

### Data Integrity Checks
- All neural data files loadable and concatenated correctly
- All retinotopy files matched to neural data (neuron counts agree)
- No NaN or Inf values in final neural arrays (cleaned during conversion)
- All trials have exactly 60 time bins
- All sessions have >= 2 trials
- Brain region indices valid (0-3 for all filtered neurons)
- Subject indices valid (0 to n_subjects-1)

## Known Limitations

1. **Large file size**: The full dataset is very large (~400+ GB) due to 50K+ neurons per session stored as float32 arrays
2. **Position-based binning**: Neural activity is interpolated into spatial bins rather than temporal bins. Each spatial bin corresponds to ~166.67 ms at constant VR speed, making this equivalent to temporal binning when the VR is moving
3. **Licking sparsity**: Unsupervised mice (no reward) have zero licking, making the licking output variable uninformative for those sessions
4. **Stimulus categories**: The full stimulus set (15 categories) includes stimuli that appear in only a subset of sessions. Some sessions have only 2-4 stimuli

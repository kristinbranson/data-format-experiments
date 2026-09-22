# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks dataset and code package in `/app`
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_exp_beh` | `/app/code/utils.py` | LOADING | Loads experiment behavior dictionaries from `Beh_<exp_type>.npy`. |
| `load_spk` | `/app/code/utils.py` | LOADING | Loads session neural activity from `<mouse>_<date>_<blk>_neural_data.npy` and concatenates entries in `['spks']` across imaging planes into one neuron-by-frame array. |
| `load_retino` | `/app/code/utils.py` | LOADING | Loads retinotopy transform and area assignments from `<mouse>_<date>_trans.npz`; maps numeric area ids into `V1`, `mHV`, `lHV`, `aHV`. |
| `neu_area_ID` | `/app/code/utils.py` | CURATION | Defines the mapping from raw `iarea` values to named brain-region masks. |
| `interp_value` | `/app/code/utils.py` | PROCESSING | Interpolates one vector from source indices to target indices. |
| `spk_pos_interp` | `/app/code/utils.py` | PROCESSING | Resamples each neuron's running-only activity by accumulated position to produce trial-by-position activity. |
| `get_interpPos_spk` | `/app/code/utils.py` | PROCESSING | Builds `(neurons, trials, 60)` interpolated corridor-position activity, using 60 bins for a 6 m corridor. |
| `Get_dprime_selective_neuron` | `/app/code/utils.py` | PROCESSING | Computes stimulus selectivity (`dprime`) using running frames inside corridor texture area only. |
| `Get_dprime_rewPred_neuron` | `/app/code/utils.py` | PROCESSING | Computes reward-prediction selectivity from interpolated corridor activity and cue-position splits. |
| `Get_coding_direction` | `/app/code/utils.py` | PROCESSING | Uses odd/even trial splits, d-prime selection, gray-space centering, and previous-trial appended bins to create coding-direction trajectories. |
| `Get_sort_spk` | `/app/code/utils.py` | PROCESSING | Uses odd/even trial splits and z-scored interpolated activity to sort selective neurons by peak corridor position. |
| `get_kfold_reward_response` | `/app/code/utils.py` | PROCESSING | Builds k-fold reward-response summaries aligned to cue and first lick. |
| `spk_2_firstLick` | `/app/code/utils.py` | PROCESSING | Aligns scalar neural activity to first lick over a fixed frame window. |
| `spk_2_cue` | `/app/code/utils.py` | PROCESSING | Aligns scalar neural activity to cue frame over a fixed frame window. |
| `data_process_script.ipynb` processing loop | `/app/code/data_process_script.ipynb` | PROCESSING | Main reference pipeline for precomputing interpolated spikes and downstream processed analysis files. |

### Notes
- The code base is mainly a figure-generation repository. The most relevant reference processing is in `/app/code/data_process_script.ipynb` plus `/app/code/utils.py`.
- Raw neural data are already stored as session arrays in `*_neural_data.npy`; the reference code does not compute `dF/F`. Instead it loads `['spks']` and concatenates them directly, so downstream processing operates on already-derived neural activity traces rather than raw fluorescence movies.
- The reference processing creates trial-by-position neural tensors by taking only frames with `ft_move > 0` and interpolating them onto 60 spatial bins across the corridor. This is the main canonical transformation used throughout the paper analyses.
- The corridor length is treated as 60 bins, corresponding to 6 meters with 1 decimeter bins. Texture corridor analyses usually focus on bins `0:40`; gray-space or inter-trial periods use later bins such as `40:60` or `42:52`.
- Behavioral and neural alignment in the reference code is mostly frame-based and/or position-based:
  - `ft_trInd` indexes frames by trial.
  - `ft_WallID` indexes frames by stimulus identity.
  - `ft_CorrSpc` and `ft_GraySpc` define corridor vs gray-space frames.
  - `SoundFr` and `LickFr` are used for cue- and lick-aligned analyses.
- Repeated curation logic in the paper code:
  - Use only running frames (`ft_move > 0`) for interpolated activity.
  - Often restrict raw-frame stimulus comparisons to corridor frames (`ft_CorrSpc`).
  - Retinotopy excludes neurons outside visual cortex in some spatial-map analyses (`iarea == -1` or `7` excluded).
  - There is no explicit electrophysiology-style unit-quality filtering in the reference code; these appear to be imaging-derived neural signals already curated upstream.
- Odd/even trial splitting is important in several analyses to avoid circularity when selecting neurons and evaluating coding structure. This may inform sanity checks even if the decoder dataset itself uses all trials.
- `README.md` explicitly says `data_process_script.ipynb` shows how intermediate results are processed and saved; that notebook is therefore the primary reference for reproducing author processing decisions.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Top-level organization under `/app/data`:
  - `beh/`: session-level behavioral dictionaries in `Beh_<exp_type>.npy`, plus `Imaging_Exp_info.npy`.
  - `spk/`: session-level neural activity files named `<mouse>_<date>_<blk>_neural_data.npy`.
  - `retinotopy/`: session-level retinotopy/area-label files named `<mouse>_<date>_trans.npz`, plus `areas.npz`.
- Behavioral files:
  - There are 23 imaging behavior files matching the 23 experiment types in `Imaging_Exp_info.npy`.
  - There are 3 additional behavior-only pretraining files under `beh/Unsupervised_pretraining_behavior/` plus one example behavior file; these do not have matching neural recordings and are not candidates for the decoder dataset.
  - Each imaging behavior file is a Python dict keyed by session identifier, usually `<mouse>_<date>_<blk>`, with 23 experiment-type-specific groupings. Some `test3` entries append `_swap1` or `_swap2` to the session key and reuse the same imaging session with alternate stimulus mappings.
- Native behavior session schema:
  - All 142 imaging behavior entries use the same 59 fields.
  - Trial-level arrays include `WallName`, `TrialStim`, `isRew`, `SoundPos`, `SoundFr`, `RewPos`, `RewTime`, `LickFr`, `LickPos`, `LickTrind`, `StartFr`, `EndFr`, `Trial_start_time`, `Trial_end_time`, `trInd`, and `run_pos`.
  - Frame-level arrays include `ft`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`, `ft_move`, `ft_isMoving`, `ft_trInd`, `ft_WallID`, `ft_CorrSpc`, `ft_GraySpc`, `AftCueFr`, and `BefCueFr`.
  - Constants are consistent across files: `Corridor_Length = 60`, `Texture_Length = 40`, `Gray_Space_length = 20`.
  - `run_pos` is already stored as `(n_trials, 60)` spatially binned position traces.
- Native neural session schema:
  - Each `*_neural_data.npy` file is a pickled dict with key `spks`.
  - `spks` is a list of 3 arrays (3 imaging planes for every session observed), each shaped `(n_neurons_in_plane, n_frames)`.
  - The reference code concatenates these 3 plane arrays along neuron axis to form one `(n_neurons_total, n_frames)` matrix per session.
- Retinotopy session schema:
  - Each `*_trans.npz` contains at least `xy_t` and `iarea`.
  - `iarea` has one label per neuron and matches the total concatenated neuron count for the session.
- Session/subject mapping:
  - `Imaging_Exp_info.npy` contains 23 experiment-type groups and 142 experiment entries that reference 89 unique underlying imaging recordings across 19 mice.
  - There are 99 unique imaging behavior session keys because some underlying recordings are represented more than once with `stimtype` variants such as `swap1` and `swap2` for test-3 analyses.
- Available task/stimulus labels:
  - Across imaging behavior files, observed wall names include `circle1`, `circle2`, `circle3`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`, `wood1`, `wood2`, `wood5`, `wood1_swap1`, `wood1_swap2`, `rock1`, and `rock2`.
  - `stim_id` values observed across imaging files are subsets of `{0,1,2,3,4,5,6}` depending on experiment type.
  - Reward modes present are `Passive` and `Active after cue`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 across 89 imaging sessions |
| Neurons / session | mean 52,708.25; range 20,547 to 89,577 |
| Subjects | 19 imaging mice |
| Sessions / subject | mean 4.68; range 1 to 8 |
| Trials (total) | 63,177 across imaging sessions referenced by `Imaging_Exp_info.npy` |
| Trials / session | mean 444.91; range 84 to 789 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not directly stated; 89 recordings total | "We performed 89 recordings in 19 mice..." |
| Neurons / session | 20,547 to 89,577 neurons/recording | "We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording." |
| Subjects | 19 imaging mice | "We performed 89 recordings in 19 mice..." |
| Sessions / subject | Mean 4.68 recordings/mouse implied by 89 recordings / 19 mice | "We performed 89 recordings in 19 mice..." |
| Trials (total) | Not directly stated in paper | No explicit trial-total count found in paper or methods excerpt |
| Trials / session | Not directly stated in paper | No explicit trial/session count found in paper or methods excerpt |
| Neural data time bin | Native imaging frames; analyses use deconvolved traces | "All our analyses were based on deconvolved fluorescence traces." |
| Behavior data time bin | Native VR/behavior timing; running speed interpolated to imaging frames when needed | "For Extended Data Fig. 8e, the running speed was interpolated to the timepoints of the imaging frames..." |
| Reward rate | Not directly stated; reward available only in rewarded corridor for task mice | "For task mice, the sound cue indicated the beginning of the reward zone in the rewarded corridor. The reward was delivered if a lick was detected after the sound cue in the rewarded corridor." |
| Cue position range | Uniform from 0.5 m to 3.5 m | "the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m" |
| Corridor geometry | 4 m texture region + 2 m grey space | "The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors." |
| Running threshold | 6 cm/s to move VR | "The mice moved forward in the virtual reality corridors by running faster than a threshold of 6 cm s−1" |
| Spatial interpolation bin size | 0.1 m positional step for some analyses | "the running speed for every position (0–6 m, with a 0.1-m step size) was also acquired through the same interpolation method." |
| Selectivity threshold | d′ >= 0.3 for selective neurons | "The criteria for selective neurons was d′ ≥ 0.3" |


### Processing Details
- Imaging data were processed with Suite2p and spike deconvolution; the paper explicitly states analyses were performed on deconvolved fluorescence traces.
- For selectivity analyses, the paper/code pair use original framewise deconvolved traces inside the 0–4 m texture region, not interpolated traces.
- Only timepoints during running are included in the main analyses, excluding stopped periods and reward-collection pauses.
- Position-based analyses interpolate neural activity by position across the full 0–6 m corridor, with 0.1 m spatial resolution in the paper/code reference.
- Cue timing is trial-variable and uniformly distributed between 0.5 m and 3.5 m from corridor entry.
- Reward-delivery location in rewarded task corridors is approximately uniformly distributed between 0.5 m and 3.5 m because mice lick soon after cue onset.
- Sequence similarity analyses use half the trials as train trials for neuron selection, then odd/even splits on held-out trials for preferred-position correlation.
- Coding-direction analyses use the top 5% selective neurons per class from train trials, interpolate neural activity by position, subtract grey-space baseline, and normalize by response variability.
- Reward-prediction analyses split leaf1 trials into early-cue versus late-cue groups and use 10-fold cross-validation for held-out trial population activity.

### Curation Steps

**Neuron curation rules**:
- No raw-fluorescence-to-`dF/F` computation is described; the paper starts from Suite2p-processed, deconvolved traces.
- Main analysis inclusion is activity during running only.
- Stimulus-selective neurons are often defined by `d′ >= 0.3` or `d′ <= -0.3`.
- Coding-direction analyses further restrict to the top 5% positively and negatively selective neurons.
- Density-map analyses exclude neurons outside visual cortex when applying retinotopy masks.

**Trial curation rules**:
- Only running timepoints are included in framewise selectivity analyses.
- Sequence/coding analyses reserve held-out trials after train/test splits.
- Reward-prediction analyses may exclude trials or mice lacking appropriate lick timing:
  - first-lick alignment kept only rewarded trials with first lick after 2 m;
  - one mouse was excluded when no such trials existed;
  - one mouse was excluded for leaf2 reward-prediction analysis when it had only one leaf2 no-lick trial.

### Decoders Trained
| Decoded variable | Accuracy |
| No explicit supervised decoder accuracy reported in the paper | N/A |
| Closest reported neural readouts | Sequence similarity correlation `r` and coding-direction similarity index `SI`, not decoder accuracy |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session counting | `Imaging_Exp_info.npy` is iterated as experiment entries; `stimtype` can create repeated views of one recording | 142 experiment entries, 99 unique behavior keys, but only 89 unique `mname/date/blk` recordings with matching neural files | "We performed 89 recordings in 19 mice" | Treat the 89 unique underlying recordings as dataset sessions. `stimtype`-specific duplicate views are analysis-specific reinterpretations of the same recording, not distinct neural recording sessions. |
| Corridor length units | Code interpolates to `n_bins=60` and uses bins `0:40` for texture area | `Corridor_Length=60`, `Texture_Length=40`, `Gray_Space_length=20` in all behavior files | Corridors are "4 m long, with 2 m of grey space between corridors" and methods mention 0.1 m interpolation steps | Data use decimeter units. Therefore 60 bins = 6 m full corridor, 40 bins = 4 m texture region, 20 bins = 2 m grey space. This is fully consistent after unit conversion. |
| Neural representation | `load_spk` directly loads `['spks']`; no `dF/F` computation in reference code | Neural files contain already-derived `spks` arrays split across 3 planes | "All our analyses were based on deconvolved fluorescence traces." | Use the stored `spks` matrices directly as neural activity. No additional fluorescence preprocessing or `dF/F` calculation is warranted. |
| Frame-count alignment | Reference code repeatedly slices behavior frame arrays with `[:nfr]` after loading spikes | Spot checks across 10 evenly spaced recordings showed `len(ft)` exceeds spike frame count by small offsets (`+1` to `+3` frames in tested sessions) | Paper does not mention this implementation detail | Follow reference behavior exactly: neural frame count is authoritative for framewise alignment, and behavior frame arrays should be trimmed to `nfr` before framewise indexing. |
| Cue-position range | Some code uses `SoundPos`; reward-prediction code often uses `SoundDelPos % 60` | `SoundPos` is typically near 4-36 bins (0.4-3.6 m) with a few wider edge cases; delayed cue/reward proxy values can extend later | Cue described as uniformly sampled from 0.5-3.5 m | Treat paper values as nominal design parameters and raw data fields as realized trial values with discretization/implementation edge effects. For decoder construction, rely on stored trial/frame variables instead of idealized nominal ranges. |
| Running-only analysis | Code consistently intersects corridor masks with `ft_move > 0` | Behavior files include `ft_move`, `ft_isMoving`, `ft_CorrSpc`, `ft_GraySpc` for all sessions | "We only considered timepoints during running for analysis" | Restrict framewise neural/behavior alignment to running frames whenever reproducing paper-style analyses; for trial tensors, preserve reference logic by deriving position-binned neural activity from running frames only. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk[*]['spks']` from `*_neural_data.npy` | `neural` | Concatenate 3 imaging planes along neuron axis to get `(n_neurons, n_frames)`; compute paper-style corridor selectivity; keep up to 64 top `|d′|` corridor-responsive neurons from each of `V1`, `mHV`, `lHV`, `aHV`; then segment into per-trial corridor-frame matrices | `load_spk`, `dprime`, `Get_coding_direction` | This is the main decoder-specific curation step to keep the exported dataset tractable while staying close to the paper’s selective-neuron logic. |
| `ft_trInd`, `ft_CorrSpc`, neural frame count `nfr` | `neural` trial segmentation | For each trial, keep frames with matching trial index and corridor mask after trimming behavior arrays to `nfr` | Reference code uses `ft_trInd`, `ft_CorrSpc`, and `[:nfr]` slicing throughout | Trial axis is corridor-entry aligned and excludes 2 m grey-space period so that position bins cleanly cover 0-4 m. |
| `SoundTime`, `Trial_start_time`, `ft` | `input[0]` = `time_to_sound_cue_s` | For each trial frame, compute `(SoundTime[trial] - ft_trial_frame_time) * 86400` seconds | Cue-alignment logic from `SoundFr`/`SoundTime` usage in `spk_2_cue` and methods text | Continuous, time-varying; negative after cue onset. |
| session date derived from `datexp` within subject chronology | `input[1]` = `day_of_training` | Continuous scalar per trial, repeated across frames: calendar days since first recording date for that subject | No exact reference helper; derived from recording chronology in `Imaging_Exp_info.npy` | Best available continuous training-day proxy in imaging data. |
| `Trial_start_time`, `ft` | `input[2]` = `time_since_trial_start_s` | `(ft_trial_frame_time - Trial_start_time[trial]) * 86400` seconds | Trial-start timing fields from behavior dictionary | Continuous, time-varying, aligned to corridor entry. |
| `isRew` | `input[3]` = `reward_available` | Boolean per trial cast to {0,1}, repeated across frames | Used directly throughout behavior code | For unsupervised sessions this remains 0 because no reward is available. |
| `WallName` | `output[0]` = `visual_stimulus_category` | Map raw wall names to coarse category families: `circle* -> circle`, `leaf* -> leaf`, `rock* -> rock`, `wood* -> wood`; repeat across frames | Raw trial identity is more stable than experiment-specific `stim_id` | Avoids duplicate-view ambiguity from `stim_id` remappings in repeated experiment labels. |
| `LickFr`, `LickTrind` | `output[1]` = `licking` | Convert lick events to binary per-frame vector within each trial after frame trimming; value 1 if one or more licks assigned to frame | `spk_2_firstLick`, `spk_2_cue` cast lick frame indices to integers | Time-varying binary output. |
| `ft_Pos` within corridor frames | `output[2]` = `position_bin` | Discretize position into 4 fixed 1 m bins over the 0-4 m texture corridor: `[0,10), [10,20), [20,30), [30,40+]` in data units | Paper methods define 0-4 m texture region; code frequently uses `:40` bins | Time-varying categorical output. |
| `ft_RunSpeed` within corridor frames | `output[3]` = `running_speed_bin` | After all trial tensors are built, assign bins 0-3 by global rank quartiles across all included corridor-frame speeds so each bin has 25% occupancy up to rounding | Running-speed analyses in methods; same framewise variable source | Time-varying categorical output; rank-based assignment is needed because many frames have exactly zero speed, so threshold-only quartiles do not satisfy the decoder requirement. |
| `iarea` from `*_trans.npz` | `brain_region_idx` | Map raw area ids via reference grouping: `8 -> V1`, `{0,1,2,9} -> mHV`, `{5,6} -> lHV`, `{3,4} -> aHV`; retain only selected neurons from these 4 grouped regions | `neu_area_ID`, `load_retino` | Exclude unassigned neurons from the exported decoder dataset because the curation step selects balanced visual-cortex neurons only. |
| mouse id from session key / `mname` | `subjects`, `subject_idx` | Unique subjects list plus per-session index | `Imaging_Exp_info.npy` | Subject is the mouse name. |

### Key Decisions
1. **Use 89 unique recordings as sessions**: The paper reports 89 recordings in 19 mice, while code/data expose 142 experiment entries and 99 behavior keys because some recordings are reused under multiple experiment labels or swap interpretations. For the decoder dataset, the unique underlying recording is the biologically meaningful session.
2. **Use raw `WallName` rather than `stim_id`/`TrialStim` for stimulus output labels**: Duplicate views of the same recording often differ only in `stim_id` remapping, while `WallName` stays fixed. This avoids double-counting and keeps labels tied to actual presented stimuli.
3. **Collapse stimulus exemplars into coarse visual categories**: The output requested is stimulus category ("circle, leaf, etc."). Mapping exemplar/swap names to `circle`, `leaf`, `rock`, and `wood` better matches the task specification and paper framing.
4. **Represent trials using corridor frames only**: The paper’s main sensory analyses focus on the 0-4 m texture corridor, and the requested 4 spatial bins are explicitly 1 m each. Excluding the 2 m grey period makes the position output well defined and avoids mixing inter-trial activity into trial content.
5. **Keep native framewise trial sequences instead of position-interpolated neural tensors**: The decoder task requires time-to-cue and time-since-start signals. Native imaging frames preserve real trial timing better than the paper’s position-interpolated tensors, while still respecting the same session loading and trial alignment structure.
6. **Trim behavior frame arrays to neural frame count**: Spot checks showed behavior frame arrays can be slightly longer than neural recordings; reference code resolves this by slicing behavior arrays to `[:nfr]`.
7. **Select a balanced subset of paper-style selective neurons per session**: Exporting all recorded neurons would create an impractically large dense dataset. To stay close to the paper, neuron selection will follow the paper’s main stimulus-selectivity logic: compute `d′` between the primary trained stimulus pair (`stim_id` 2 versus 0) on corridor running frames, keep corridor-responsive neurons with `|d′| >= 0.3`, and cap the export to up to 64 top-`|d′|` neurons per grouped visual region (`V1`, `mHV`, `lHV`, `aHV`).
8. **Restrict exported neurons to the four grouped visual regions**: These are the paper’s primary anatomical groupings and avoid unlabeled/out-of-map units that the reference analyses often exclude.
9. **Use global running-speed quartiles computed from included trial frames**: This follows the decoder specification exactly and avoids session-specific binning that would confound cross-session decoding. Because many corridor frames have tied zero speed, the final implementation assigns quartiles by global rank after conversion rather than by simple thresholding.
10. **Use calendar day since first recording as the continuous training-day variable**: The imaging data do not expose an exact per-trial "training day" number, so subject-specific elapsed recording day is the most defensible continuous proxy.

### Planned Sanity Checks
- [x] Check that one saved neural trial equals the raw concatenated spike matrix restricted to the same trimmed frame mask (`np.allclose` on a selected neuron/frame block).
- [x] Check that one saved input `time_to_sound_cue_s` vector equals raw `SoundTime - ft` on a selected trial (`np.allclose`).
- [x] Check that one saved licking / stimulus / position output block matches direct reconstruction from raw behavior on a selected trial (`np.allclose`).
- [x] Check that no trials were silently lost during canonical-session conversion (`raw_trials_total == converted_trials_total`, skipped trials total = 0).
- [x] Check that running-speed quartiles are globally balanced after conversion (`speed_bin_fractions == [0.25, 0.25, 0.25, 0.25]` up to rounding).

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `/app/convert_data.py` with:
  - canonical session selection over repeated experiment-label views;
  - per-subject session-day derivation from recording dates;
  - frame-interval summary before conversion and exact global rank-based running-speed quartile assignment after conversion;
  - per-session loading of spikes/retinotopy/behavior;
  - paper-style neuron selection using corridor-running `d′` between `stim_id` 2 and 0, capped to 64 neurons per grouped visual region;
  - corridor-only trial segmentation using `ft_trInd` and `ft_CorrSpc`;
  - construction of trial-aligned neural/input/output tensors;
  - optional processing plots for up to 2 sessions;
  - metadata and timing summaries.

Code inefficiencies identified:
- Raw dense spike files are very large; naive export of all neurons would be impractical in both time and output size.
- Full-session duplicate behavior views would duplicate work and inflate the dataset if not collapsed first.
- Exact speed-quartile balancing cannot be achieved by simple percentile thresholds because of the large mass of zero-speed frames.

Code speedups added:
- Process sessions one at a time to avoid holding multiple raw recordings in memory.
- Use a lightweight pre-pass only for frame-interval estimates; defer speed-bin assignment to a single vectorized post-pass over converted trial speeds.
- Cap exported neurons per region to bound memory and output size.
- Store neural trial arrays as `float16` and inputs as `float32`.
- An attempted optimization that pre-concatenated all selected traces per session caused a performance regression on sample data and was reverted.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 512 selected neurons |
| Neurons / session | 256, 256 |
| Subjects | 2 (`TX108`, `TX109`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 580 |
| Trials / session | 210, 370 |
| `time_to_sound_cue_s` range | [-282.2, 199.8] |
| `day_of_training` range | [0.0, 0.0] |
| `time_since_trial_start_s` range | [0.0, 284.8] |
| `reward_available` range | [0.0, 1.0] |
| `visual_stimulus_category` distribution | [0.231, 0.203, 0.293, 0.274] |
| `licking` distribution | [0.781, 0.219] |
| `position_bin` distribution | [0.226, 0.243, 0.227, 0.304] |
| `running_speed_bin` distribution | [0.250, 0.250, 0.250, 0.250] |

### Processing Plots Review
- `processing_TX108_2023_03_13_1.png` and `processing_TX109_2023_04_18_1.png` were generated.
- Neural trials, timing inputs, and categorical outputs were visually populated with no empty arrays or obvious session-level misalignment.
- Sample sessions intentionally include rewarded task data so reward availability and licking are non-degenerate during validation.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Plane-wise neuron selection without building one giant concatenated session matrix | Lower peak memory; avoids an extra full-session copy |
| Canonical-view deduplication before conversion | Prevents repeated work on the same recording |
| Capped export of 64 selective neurons per region | Keeps output size and downstream training tractable |
| Post-pass vectorized speed-quartile assignment | Exact quartile balancing without per-trial recomputation |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (`--sample`, 2 task sessions) | 14.92 s/session observed | 29.84 s for 2 sessions |
| Weighted full-dataset estimate using raw neuron×frame proxy | ~15.4 s/session effective | ~22.8 min for 89 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `visual_stimulus_category` | 0.9656 | 0.9781 |
| `licking` | 0.8437 | 0.8275 |
| `position_bin` | 0.8573 | 0.7212 |
| `running_speed_bin` | 0.6127 | 0.5861 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 687.19 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 20,547-89,577 per recording; total not explicitly stated | `load_spk` uses all neurons in session files | 4,691,034 raw neurons | 22,163 selected neurons | Curated subset, intentional |
| Mean neurons/session | 20,547-89,577 raw/session | all loaded before curation | 52,708.25 raw/session | 249.02 selected/session | Curated subset, intentional |
| Subjects | 19 | 19 unique mice in `Imaging_Exp_info.npy` | 19 | 19 | Yes |
| Sessions | 89 recordings | 89 unique `mname/date/blk` recordings after deduplication | 89 | 89 | Yes |
| Trials (total) | not explicitly stated | canonical behavior trial counts preserved | 38,110 | 38,110 | Yes |
| Trials/session (mean) | not explicitly stated | canonical behavior trial counts preserved | 428.20 | 428.20 | Yes |
| `day_of_training` range | training spans multiple days in paper | derived from recording chronology | 0.0 to 92.0 days from session dates | [0.0, 92.0] | Yes |
| `reward_available` range | binary rewarded vs unrewarded corridors | `isRew` is binary in raw behavior | {0,1} | [0.0, 1.0] | Yes |
| `visual_stimulus_category` distribution | not explicitly stated | driven by `WallName` identities | [0.320, 0.484, 0.081, 0.116] after canonical-session selection | [0.320, 0.484, 0.081, 0.116] | Yes |
| `running_speed_bin` distribution | decoder spec requires equal-occupancy quartiles | implemented by post-pass global rank assignment | [0.250, 0.250, 0.250, 0.250] | [0.250, 0.250, 0.250, 0.250] | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Verification log review: `verification_full_out.txt` reported no errors and no warnings.
2. Raw-data sanity check: `/app/cache/step10_sanity_check.py` reconstructed session `TX108_2023_03_13_1`, trial 10 directly from raw behavior/spike/retinotopy files and matched converted neural, input, and output tensors exactly with `np.allclose == True` and max absolute difference 0.0 for all three categories.
3. Reference code comparison:
   - data loading matches `load_spk`, `load_exp_beh`, and `load_retino` by using the same raw files and the same retinotopy region grouping;
   - neuron filtering matches the paper’s corridor-running selectivity logic (`ft_move > 0`, `ft_CorrSpc`, `d′ >= 0.3`) but caps each grouped region at 64 neurons for tractable decoder training;
   - temporal alignment matches the reference use of `ft_trInd` plus `[:nfr]` trimming, but keeps native imaging frames instead of 0.1 m position interpolation because the decoder task requires continuous time-to-cue and time-since-start signals;
   - input construction uses the same raw timing variables the paper uses for cue/trial alignment (`SoundTime`, `Trial_start_time`, `ft`);
   - output construction uses raw `WallName`, `LickFr`, and `ft_Pos`, with the decoder-required running-speed discretization applied after conversion by global rank quartiles.
4. Key statistics comparison: `/app/cache/step10_stats_check.py` confirmed 19 subjects, 89 sessions, 38,110 trials, 0 skipped trials, and exact 25/25/25/25 global speed-bin fractions. Raw neuron totals matched the dataset exploration count (4,691,034), while converted neurons were the intended curated subset (22,163 total; 249.02/session mean).
5. Edge-case review: the longest retained trial (`TX88_2022_07_19_1`, trial 391) has 5,607 corridor frames spanning 1,764.99 s with median running speed 0.0 and position concentrated in the 1-2 m bin. Raw behavior reproduces the same prolonged stall, so this is a genuine behavioral edge case rather than an indexing bug.

### Issues Found and Resolved
- Running-speed quartile imbalance under threshold-based quantiles: many corridor frames had exactly zero speed, so edge-threshold digitization produced `[0.302, 0.198, 0.250, 0.250]` on the full dataset. Resolved by assigning `running_speed_bin` via global rank quartiles across all converted corridor frames, yielding exact `[0.250, 0.250, 0.250, 0.250]`.
- Attempted session-level neural-trace pre-concatenation caused a runtime regression during sample conversion. Reverted to the earlier plane-wise slicing implementation because it was stable and preserved correctness.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `visual_stimulus_category` | 0.9541 | 0.9217 | 3.69x chance; consistent with strong visual identity coding. |
| `licking` | 0.9381 | 0.9100 | 1.82x chance; very small train-validation gap. |
| `position_bin` | 0.7993 | 0.7292 | 2.92x chance; consistent with position-sequence structure in the paper. |
| `running_speed_bin` | 0.6846 | 0.6595 | 2.64x chance; speed discretization is informative and balanced. |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `visual_stimulus_category` | 0.9217 validation balanced accuracy | No direct decoder metric reported; paper shows strong stimulus-sequence structure, so comfortably above-chance decoding is expected. |
| `licking` | 0.9100 validation balanced accuracy | No direct decoder metric reported; cue/reward/lick aligned analyses imply strong lick-related information. |
| `position_bin` | 0.7292 validation balanced accuracy | No direct decoder metric reported; position-sequence similarity and coding-direction analyses imply substantial positional information. |
| `running_speed_bin` | 0.6595 validation balanced accuracy | No direct decoder metric reported; methods explicitly analyze running-speed relationships, so above-chance decoding is expected. |

- All four outputs are above chance by more than 1.5x:
  - `visual_stimulus_category`: 3.69x chance
  - `licking`: 1.82x chance
  - `position_bin`: 2.92x chance
  - `running_speed_bin`: 2.64x chance
- Train/validation ratios are all close to 1.0 (`1.04`, `1.03`, `1.10`, `1.04` respectively), so there is no sign of severe overfitting or leakage.
- The paper does not report a supervised decoder benchmark for these exact variables, so there is no direct numerical target to exceed. The strongest available paper expectation is qualitative: visual category, position sequence, and reward/lick-related variables should all be decodable above chance if alignment is correct. The observed results meet that standard.

### Issues Found and Resolved
- No additional decoder-performance issues required fixes after the Step 10 corrections.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

# Dataset Conversion Notes

## Overview
- **Dataset**: "Unsupervised pretraining in biological neural networks" (paper.pdf) - virtual corridor visual task, 2p imaging
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3 with numpy 2.4.4, torch 2.6.0+cu124 (CUDA available, NVIDIA L4 23 GB), 128 CPUs, 1 TB RAM.

Directory contents of `/app`:
- `paper.pdf`, `methods.txt` - reference text
- `code/` - reference code (`utils.py`, `data_process_script.ipynb`, `Figures.ipynb`, `fig1-5.py`, `S6.py`, `README.md`)
- `data/` - `spk/` (89 session files, 405 GB), `beh/` (23 `Beh_<exp_type>.npy` + `Imaging_Exp_info.npy`, 6.6 GB), `retinotopy/` (89 `*_trans.npz`, 170 MB), `process_data/` (empty)
- `train_decoder.py`, `decoder.py` - decoder used for validation
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Produced by this work: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`, `README.md`, the conversion/verification/training logs, `processing_<session>.png`, and `cache/`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk(db, root)` | utils.py:338 | LOADING | Loads `<mname>_<datexp>_<blk>_neural_data.npy` (dict with key `spks`, a list of per-imaging-plane arrays) and concatenates planes along axis 0 -> (n_neurons, n_frames) deconvolved traces (float32). |
| `load_retino(db, root)` | utils.py:330 | LOADING | Loads `<mname>_<datexp>_trans.npz` giving per-neuron `iarea` (area id) and `xy_t` (retinotopically transformed cortical position). |
| `neu_area_ID(iarea)` | utils.py:312 | CURATION | Maps area ids to 4 visual areas: V1=8; mHV=0,1,2,9; lHV=5,6; aHV=3,4. ids -1 and 7 are NOT part of any area (excluded from analyses, see `Get_density_map`: `idx_neu = (arid!=-1) & (arid!=7)  # exclude neurons from outside of visual cortex`). |
| `load_exp_beh(root, exp_type)` | utils.py:326 | LOADING | Loads `beh/Beh_<exp_type>.npy` -> dict keyed by `<mname>_<datexp>_<blk>[_<stimtype>]`. |
| `get_interpPos_spk` / `spk_pos_interp` | utils.py:120/105 | PROCESSING | Interpolates deconvolved activity onto 60 position bins per trial (1 dm each, corridor cycle = 6 m: bins 0-39 = 4 m texture, bins 40-59 = 2 m grey), using ONLY frames where the VR was moving (`ft_move>0`). |
| `Get_dprime_selective_neuron` | utils.py:418 | PROCESSING/CURATION | Defines the canonical valid-frame mask: `VRmove = beh['ft_move'][:nfr]>0`, `isCorridor = beh['ft_CorrSpc'][:nfr]`, `fr_valid = VRmove & isCorridor` ("only use activity inside the texture area plus mouse is running"). Note the truncation of behaviour arrays to the number of neural frames `nfr`. |
| `Get_coding_direction` | utils.py:503 | PROCESSING | Same valid-frame logic; normalizes activity by grey-space baseline and per-corridor std. |
| `get_kfold_reward_response` | utils.py:814 | PROCESSING | z-scores the raw frame-wise activity (`stats.zscore(spk, axis=1)`) and the interpolated activity before analysis. |
| `spk_2_cue`, `spk_2_firstLick` | utils.py:935/913 | PROCESSING | Align frame-wise activity and lick histograms to `SoundFr` / first `LickFr` using integer neural-frame indices. |
| `dprime(x1,x2)` | utils.py:370 | PROCESSING | Selectivity index used throughout, threshold 0.3. |
| `data_process_script.ipynb` | code/ | LOADING | Top-level script: loads `beh/Imaging_Exp_info.npy` (23 experiment types -> list of session dicts), then for each session loads beh + spk and computes interpolated activity / d-primes. |

### Notes
- Calcium frame rate is stated in the notebook: **fs = 3.17 Hz** (measured median dt in the data = 0.3149 s).
- Deconvolved traces are already provided (Suite2p, tau=0.75 s); **no dF/F computation is needed** and no further spike deconvolution.
- **No per-neuron quality filtering** is performed in the reference beyond Suite2p cell classification; the only neuron curation is anatomical (exclude `iarea==-1` and `iarea==7`, i.e. neurons outside the visual cortex).
- Behaviour arrays are always truncated to the number of neural frames (`[:nfr]`), because behaviour can have 1 extra frame.
- Full beh variable dictionary is given in `data_process_script.ipynb` (cell 4) and is reproduced in Step 2.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/spk/<mname>_<datexp>_<blk>_neural_data.npy` (89 files, 405 GB total): pickled dict with a single key `spks` = list of per-plane float32 arrays (n_neurons_plane, n_frames). Concatenated they give (n_neurons, n_frames) deconvolved activity. Example: TX83_2022_08_31_1 -> 3 planes -> (50689, 33597), 6.8 GB, ~7 s to load.
- `/app/data/retinotopy/<mname>_<datexp>_trans.npz` (89 files): `iarea` (n_neurons,), `xy_t` (n_neurons,2), `xpos`,`ypos`,`A`,`dx`,`dy`. Neuron order matches the concatenated spk matrix (verified: 50689 == 50689).
- `/app/data/beh/Beh_<exp_type>.npy` (23 files): dict keyed by `<mname>_<datexp>_<blk>` or `<mname>_<datexp>_<blk>_<stimtype>` (swap1/swap2). 99 distinct keys covering **89 unique sessions** (10 keys are duplicate copies of the same session with a different `stim_id` labelling for the two leaf1-swap stimuli).
- `/app/data/beh/Imaging_Exp_info.npy`: dict of 23 experiment types -> list of session dicts (`mname`,`datexp`,`blk`,`exptype`,`rewType`,`stim_id`,`stimtype`,`sess#`,`days`,...). 142 entries -> **89 unique sessions, 19 mice**.
- `/app/data/process_data/`: empty (intermediate results not shipped).

### Key behaviour variables (per session; ft_* are per neural frame, ~3.17 Hz)
`ntrials`, `Trial_start_time`/`StartFr` (corridor entry), `GrayFr` (grey-space entry), `EndFr` (corridor exit), `SoundTime`/`SoundFr`/`SoundPos` (cue), `RewTime`/`RewardFr`/`RewPos`, `isRew` (rewarded corridor), `WallName`/`UniqWalls`/`stim_id` (stimulus identity), `LickTime`/`LickFr`/`LickPos`/`LickTrind`, `ft` (frame times, MATLAB datenum days), `ft_trInd`, `ft_Pos` (position in the 6 m VR cycle, in **decimeters**), `ft_PosCum`, `ft_move` (VR displacement per frame), `ft_CorrSpc`/`ft_GraySpc` (texture corridor / grey space), `ft_RunSpeed` (cm/s), `ft_WallID`, `Corridor_Length`=60 dm, `Texture_Length`=40 dm, `Gray_Space_length`=20 dm.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 (all recorded); 4,105,393 inside visual cortex (iarea not in {-1,7}) |
| Neurons / session | mean 52,708; min **20,547**; max **89,577** |
| Subjects | 19 (DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2) |
| Sessions / subject | 1-8 (89 sessions / 19 mice = 4.7 mean) |
| Trials (total) | 39,600 over the 89 unique sessions (44,060 counting the 10 duplicated swap keys) |
| Trials / session | mean ~445, min 84, max 789 |
| Frames / session | 17,198 - 33,598 (dt = 0.3144-0.3154 s) |
| Sessions with licking recorded | 26 unique sessions (task/`sup` mice, water restricted); unsupervised/naive mice have empty `LickFr` (no water, no licking) |

### Stimulus naming
Raw `WallName` values across the dataset: circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, rock1, rock2, wood1, wood2, wood5, wood1_swap1, wood1_swap2.
`beh['stim_id']` (parallel to `UniqWalls`) maps them onto the paper's 7 canonical stimuli:
`0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2` (rock->circle-role, wood/brick->leaf-role), i.e. exactly the pooling the paper describes ("we denote the stimuli as leaf and circle, even though other visual stimuli were also used in some mice (rock and bricks)").

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons / recording | 20,547 - 89,577 | "activity traces from 20,547 to 89,577 neurons in each recording" |
| Recordings (sessions) | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 mice (13 male, 6 female) | "89 recordings in 19 mice bred to express GCaMP6s" |
| Neural data time bin | 3.17 Hz -> 315 ms | notebook: "Calcium signal recording frame rate: fs = 3.17Hz" |
| Corridor geometry | 4 m textured corridor + 2 m grey | "virtual reality corridors were each 4 m long, with 2 m of grey space between corridors" |
| VR speed | 60 cm/s constant when running > 6 cm/s | "virtual corridors always moved at a constant speed (60 cm s-1)" |
| Sound cue position | uniform 0.5-3.5 m | "the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m" |
| Reward | only in rewarded corridor, after cue | "the sound cue indicated the beginning of the reward zone in the rewarded corridor" |
| Neural signal | deconvolved traces (Suite2p, tau=0.75 s) | "All our analyses were based on deconvolved fluorescence traces" |
| Running filter | only running timepoints analysed | "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards" |
| Analysis window | 0-4 m texture area | "we only selected data points inside the 0-4-m region of the corridors where the textures were shown. We excluded the data points in which the animal was not running" |
| Expected trial duration | ~6.7 s (4 m at 60 cm/s) -> ~21 frames | derived; data: median 21-24 frames/trial |

### Processing Details
- Temporal alignment for frame-wise analyses is by **neural frame index** (`StartFr`, `SoundFr`, `LickFr`, `EndFr` are frame indices into the ft/neural time base, so behaviour and neural data share one clock).
- Valid data = `ft_move>0` (VR moving == mouse running above 6 cm/s threshold) AND `ft_CorrSpc` (inside the textured corridor) - reference `fr_valid` in `Get_dprime_selective_neuron`.
- Behaviour arrays truncated to number of neural frames.
- Position analyses interpolate onto 1 dm bins (60 per 6 m cycle).

### Curation Steps
**Neuron curation rules**: Suite2p cell detection/classification already applied in the released `spks`. The only explicit filtering in the reference code is anatomical: exclude neurons with `iarea == -1` or `iarea == 7` (outside visual cortex); the rest are assigned to V1 / mHV / lHV / aHV.

**Trial curation rules**: No explicit trial rejection in the reference code; trials are implicitly restricted to frames in the textured corridor while running. Trials without any running frames contribute no data.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| (none) | The paper contains **no decoding analyses** (grep for "decod" in the extracted PDF text returns nothing), so no decoder accuracy benchmark exists. Expectations must come from the structure of the task itself. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | 142 exp_info entries over 23 exp types | 89 unique (mname,datexp,blk); 89 spk files; 89 retinotopy files | "89 recordings" | Sessions appear in several experiment types (before/after learning, test1/2/3). Use the **89 unique sessions**; for each session merge the `stim_id` labelling across all beh keys that contain it. |
| Beh keys | keys can carry a `_swap1`/`_swap2` suffix | 99 beh keys; the 10 extra are duplicates of 5 sessions with identical behaviour but complementary `stim_id` (swap1 labelled in one, swap2 in the other) | "The two leaf1-swap stimuli ... were treated as two different data points" | Deduplicate to one entry per session; take the union of the two `stim_id` labellings so both swap stimuli are labelled. |
| Number of frames | `spk[:, :nfr]`, `beh[...][:nfr]` | beh ft can have 1 more frame than spk | - | Truncate everything to `n = min(n_neural_frames, len(ft))`, as the reference does. |
| Licking | `LickFr` used for task mice | empty for unsupervised/naive mice | unsupervised mice were not water restricted and received no reward | Licking output = 0 for those sessions (mice genuinely do not lick without water); documented as a limitation. |
| Position units | `Corridor_Length=60`, texture 0-40 | `ft_Pos` in [0,60) | 4 m corridor + 2 m grey | `ft_Pos` is in decimeters; texture area = 0-40 dm = 0-4 m. |
| Running filter | `ft_move>0` | 40-78% of corridor frames are running frames | "only considered timepoints during running" | Use `ft_move>0 & ft_CorrSpc`. |

All three sources are consistent after these resolutions. Independent confirmation: the per-session neuron counts computed from the retinotopy files (min 20,547, max 89,577) match the paper's reported range **exactly**.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial definition
- **Sessions**: the 89 unique `(mname, datexp, blk)` recordings. Beh keys with `_swap1`/`_swap2` suffixes are duplicates of one recording; they are merged (union of their `stim_id` labellings).
- **Trial**: one corridor traversal. Aligned to **corridor entry** (`StartFr` / `Trial_start_time`). Frames kept per trial: `ft_trInd == trial` AND `ft_CorrSpc` (inside the 4 m textured corridor) AND `ft_move > 0` (VR moving, i.e. the mouse is running above the 6 cm/s threshold). This is exactly the reference `fr_valid = VRmove & isCorridor` in `utils.Get_dprime_selective_neuron`, and matches "We only considered timepoints during running for analysis".
- Behaviour arrays truncated to the number of neural frames (as in the reference).
- Trials with < 5 valid frames, with an unlabeled stimulus, or with a NaN cue time are dropped (counts reported by the conversion).

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/<sess>_neural_data.npy` -> `spks` (list of planes) | `neural` | concatenate planes, keep valid frames, keep visual-cortex neurons, subsample, z-score per neuron | `utils.load_spk`; z-scoring as in `utils.get_kfold_reward_response` | deconvolved traces (Suite2p, tau 0.75 s); float32 (n_neurons, T) per trial |
| `retinotopy/<sess>_trans.npz` -> `iarea` | `brain_region_idx`, neuron curation | `utils.neu_area_ID` mapping; drop `iarea in {-1,7}` | `utils.load_retino`, `Get_density_map` | regions V1 / mHV / lHV / aHV |
| `beh['SoundTime']`, `beh['ft']` | `input[0]` time_to_sound_cue | `(SoundTime[trial] - ft) * 86400` seconds, positive **before** the cue, negative after | `utils.spk_2_cue` uses `SoundFr` for the same alignment | continuous, time-varying |
| `datexp` per session | `input[1]` day_of_training | days elapsed since that mouse's first recording | exp_info | continuous, per-trial (broadcast over time) |
| `beh['ft']`, `beh['Trial_start_time']` | `input[2]` time_since_trial_start | `(ft - Trial_start_time[trial]) * 86400` seconds | - | continuous, time-varying |
| `beh['isRew']` | `input[3]` reward_availability | 1 if the trial's corridor is the rewarded one, else 0 | `utils.get_cat_id`, `Get_dprime_rewPred_neuron` | per-trial (broadcast) |
| `beh['WallName']` + `beh['stim_id']`/`UniqWalls` | `output[0]` stimulus category | map wall name -> canonical id 0-6 | `stim = ['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']` in `data_process_script.ipynb` | role-based canonical labels used throughout the reference; pools rock->circle-role and wood/brick->leaf-role exactly as the paper does |
| `beh['LickFr']` | `output[1]` licking | binary per frame: 1 if >=1 lick is assigned to that neural frame | `utils.spk_2_firstLick`, `spk_2_cue` (lick histograms on frame indices) | time-varying |
| `beh['ft_Pos']` | `output[2]` position bin | `floor(ft_Pos/10)` clipped to 0..3 (ft_Pos is in decimeters; 4 bins x 1 m over the 4 m texture area) | position binning as in `get_interpPos_spk` (1 dm bins) | time-varying |
| `beh['ft_RunSpeed']` | `output[3]` speed bin | global quartiles over all included timepoints -> 4 bins, each 25% of the data | `utils.load_exp_beh` / Running speed methods | time-varying |

### Key Decisions
1. **Only running frames inside the textured corridor are used** - matches the reference `fr_valid` mask and the paper ("We only considered timepoints during running"). It also makes the 4 x 1 m position bins well defined (positions 0-4 m only).
2. **Neurons subsampled to 1,000 per session (seeded, stratified by visual area)**: the full dataset is 405 GB (20k-90k neurons x ~25k frames per session) and cannot be stored in a pickle or loaded by the decoder. 1,000 neurons is far more than the 100 PCs the decoder uses, so decoding is not limited by this. Neurons outside the visual cortex (`iarea in {-1,7}`) are excluded first, following the reference.
3. **Per-neuron z-scoring over the included frames**: the reference z-scores deconvolved traces before population analyses (`get_kfold_reward_response`); it prevents a few very active neurons from dominating the decoder's PCA initialisation. Zero-variance neurons are excluded before sampling.
4. **Stimulus labels are the reference's role-based canonical ids** (`beh['stim_id']`), i.e. the paper's leaf/circle naming, merged over all experiment types in which a session appears so that the swap stimuli are labelled.
5. **All 89 sessions are kept**, including unsupervised/naive mice that never lick (they are not water restricted); licking is genuinely 0 for those sessions.
6. **Speed quartiles are computed globally** over all included timepoints of the converted dataset so that each bin holds 25% of the data, as required by the Decoder Task.
7. **Time bin** = the imaging frame (3.17 Hz, ~315 ms). No re-binning: the reference analyses all frame-wise quantities on this native grid.

### Planned Sanity Checks
- [ ] 89 sessions, 19 mice, per-session neuron counts in [20,547 , 89,577] (paper).
- [ ] Mean trial duration ~6.7 s (4 m at 60 cm/s) => ~21 frames/trial.
- [ ] Position bins: bin 0 dominates early in the trial, bin 3 at the end; position increases monotonically inside a trial.
- [ ] Cue position (derived from time_to_cue and speed) uniform between 0.5 and 3.5 m (paper).
- [ ] Reward availability only non-zero in task (sup) sessions; ~30-50% of trials there.
- [ ] Licking only in task sessions; ~7-15% of included frames.
- [ ] Speed bins each ~25% of timepoints.
- [ ] Spot-check raw values: neural z-score recomputed from the raw .npy, lick/position/speed values, against the converted arrays (`np.allclose`).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (580 lines). Structure:
- `load_exp_info()` - 89 unique sessions from `beh/Imaging_Exp_info.npy`.
- `load_behaviour()` - single pass over the 23 `Beh_*.npy` files; keeps one behaviour dict per session and merges the wall-name -> canonical `stim_id` mapping over all experiment types (handles the `_swap1`/`_swap2` duplicate keys).
- `select_neurons()` - reimplements `utils.neu_area_ID`; drops `iarea in {-1,7}`, then samples up to 1,000 neurons proportionally across V1/mHV/lHV/aHV with a per-session seeded RNG.
- `load_spk_rows()` - equivalent to `utils.load_spk(...)[rows]`: concatenation offsets over the per-plane arrays, but only the sampled rows are copied out, so the full (up to 90k x 33k) matrix is never materialised.
- `process_session()` - truncates behaviour to `nfr` (as the reference does), builds the reference valid-frame mask `fr_valid = (ft_move>0) & ft_CorrSpc`, z-scores each neuron over the included frames, bins licks onto neural frames, and splits the valid frames into trials by `ft_trInd`, aligned to corridor entry.
- `assemble()` - builds inputs/outputs, global running-speed quartiles, day-of-training (days since that mouse's first session), metadata and `session_info`.
- `show_processing()` - 6-panel figure per session covering frame curation, trial segmentation of the z-scored activity, inputs (with the cue-onset frame marked), position + its discretisation, speed + its quartile discretisation + licking, and the session speed histogram with the global quartile edges.
- `sanity_checks()` - prints dataset size, trial durations, input ranges, output distributions and task-level checks.

Code inefficiencies identified:
- Naively loading each session (`utils.load_spk`) would read 405 GB and hold up to 12 GB per session.

Code speedups added:
- Only the sampled neuron rows are copied out of each plane, and each plane is released immediately after use.
- Sessions are processed in a `multiprocessing` pool (fork, so the behaviour dict is shared copy-on-write).
- Trial segmentation is vectorised (sort by `ft_trInd`, `np.split` at the boundaries) instead of looping over trials.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sessions TX108_2023_03_25_1 - task/water-restricted mouse with licking and reward - and TX83_2022_08_31_1 - unsupervised mouse).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons used / session | 1000, 1000 (of 79,417 and 50,689 recorded) |
| Subjects | 2 |
| Trials (total) | 825 (423 + 402; **no trial lost** - 0 dropped for short/unlabeled/no-cue) |
| Timepoints / trial | mean 24.2, median 24, min 15, max 40 -> 7.6 s mean |
| time_to_sound_cue | [-183.8, 203.1] s |
| day_of_training | [0, 0] (both sessions are the first of their mouse in the sample) |
| time_since_trial_start | [0.0, 204.5] s |
| reward_availability | [0, 1], 13.2% of trials |
| stimulus_category | circle1 0.306, circle2 0.188, leaf1 0.300, leaf2 0.206 |
| licking | no_lick 0.947, lick 0.053 (0.101 in the task session, 0 in the unsupervised session) |
| position_bin | 0.249, 0.249, 0.248, 0.254 |
| speed_bin | 0.250, 0.250, 0.250, 0.250 |

### Processing Plots Review
`processing_TX108_2023_03_25_1.png`, `processing_TX83_2022_08_31_1.png`: position ramps 0->40 dm within every trial and the excluded (non-running / grey-space) frames are exactly the paused and inter-corridor periods; the z-scored raster shows clear trial structure; `time_to_sound_cue` decreases linearly and crosses zero exactly at the marked cue-onset frame; the position bins switch at 10/20/30 dm; the speed bins follow the global quartile edges; licking is confined to the task session. No anomalies.

### Extra sanity check on the sound cue (raw data)
`beh['SoundPos']` for TX108_2023_03_25_1 spans 0.4-3.6 m (1st pct 0.44 m, median 2.0 m, 99th pct 3.58 m), and `ft_Pos` at `SoundFr` gives the same - matching the paper's "uniform distribution between positions 0.5 m and 3.5 m".
Note: the time-based estimate of cue position printed by `sanity_checks` has a long tail because mice pause inside the corridor (23.6% of trials have >15 s of wall-clock duration) while the paused frames are excluded by the reference running filter; the position-based check above is the correct one.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| load only the sampled neuron rows | avoids materialising ~12 GB/session |
| multiprocessing pool (6-8 workers) | ~6x |
| vectorised trial segmentation | seconds per session |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| behaviour load (once) | - | 2 s |
| spk load + processing | 2-9 s (6.8-10 GB files) | 89 sessions / 6 workers ~ 5-15 min (I/O bound) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None ("Data format is valid, no errors or warnings.")
- Warnings: None

### Decoder Results (Sample, 2 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| stimulus_category | 0.9999 | 0.8984 | 0.1429 |
| licking | 0.9545 | 0.8590 | 0.5000 |
| position_bin | 0.9892 | 0.8299 | 0.2500 |
| speed_bin | 0.6983 | 0.5093 | 0.2500 |

Loss decreased monotonically (1.16 -> 0.31 over 200 epochs). All four outputs are well above chance; speed is the hardest, as expected for a variable that is only weakly represented in visual cortex and that the VR itself clamps (the corridor moves at a fixed 60 cm/s whenever the mouse runs above threshold).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --nworkers 8` (76 s wall-clock: 2 s behaviour load, 65 s for all 89 sessions, 5 s pickling).

### Output Files
- `converted_data.pkl`: 3.31 GB
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Converted dataset
| Statistic | Value |
|-----------|-------|
| Sessions | 89 |
| Subjects | 19 (1-8 sessions each) |
| Trials | 37,801 (mean 424.7/session, min 84, max 722) |
| Timepoints | 815,506 (mean 21.6/trial, median 21, min 11, max 178) |
| Mean trial duration | 6.79 s (median 6.61 s) |
| Time bin | 314.85 ms (3.176 Hz) |
| Neurons used / session | 1,000 (of 20,547-89,577 recorded) |
| Neurons by area | V1 39,140; mHV 25,298; lHV 10,314; aHV 14,248 |
| Cohorts | 51 unsupervised, 28 task (sup), 10 naive sessions |

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Recordings (sessions) | 89 | 89 unique (mname,datexp,blk) in exp_info | 89 spk files, 89 retinotopy files | 89 | YES |
| Subjects | 19 | 19 | 19 | 19 | YES |
| Neurons per recording | 20,547-89,577 | all Suite2p ROIs | min 20,547, max 89,577 | same (recorded); 1,000 sampled per session for the pickle | YES |
| Neurons outside visual cortex | excluded from analyses | `(arid!=-1)&(arid!=7)` | 585,641 such neurons (12.5%) | excluded | YES |
| Trials (total) | n/a | n/a | 38,110 corridor traversals | 37,801 (99.2%; 309 `circle3` trials in 4 sessions have no canonical `stim_id` and are dropped) | YES |
| Frame rate | "fs = 3.17 Hz" (notebook) | - | median dt 0.3144-0.3154 s | 314.85 ms | YES |
| Trial duration | 4 m at 60 cm/s -> 6.67 s | - | ~21 frames/trial | 6.79 s mean | YES |
| Corridor geometry | 4 m texture + 2 m grey | `Corridor_Length=60`, `Texture_Length=40` (dm) | positions 0-40 dm in corridor | position bins cover 0-4 m | YES |
| Cue position | uniform 0.5-3.5 m | `SoundPos`, `SoundFr` | `SoundPos` spans 0.4-3.6 m | cue-onset frame falls inside the trial in 95% of trials | YES |
| Valid frames | "only timepoints during running" | `fr_valid = VRmove & isCorridor` | 40-78% of corridor frames | identical mask | YES |
| Licking | only water-restricted task mice | `LickFr` | 26 sessions have licks | 11.7% of task-session timepoints, 0 elsewhere | YES |
| Rewarded corridors | ~half of trials in training sessions | `isRew` | 0.26-0.51 in task sessions | 37.6% of task trials (11.5% of all trials, because 61/89 sessions have no reward) | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt`: **"Data format is valid, no errors or warnings."** - no errors and no warnings to address.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
The script re-loads the **raw** files (`spk/*_neural_data.npy` concatenated exactly like `utils.load_spk`, `beh/Beh_*.npy`, `retinotopy/*_trans.npz`) with no use of `convert_data.py`, and compares against `converted_data.pkl` for three sessions spanning the cohorts (TX108_2023_03_25_1 task/rock-wood, TX83_2022_08_31_1 unsupervised, VR2_2021_04_11_1 task/leaf-circle):

| Check | Result |
|---|---|
| recorded neuron count (raw spk == retinotopy == metadata) | PASS (79,417 / 50,689 / 76,108) |
| number of trials rebuilt independently == converted | PASS (423, 402, 719) |
| every converted neuron matched to a unique raw row (1000 unique rows) | PASS |
| `np.allclose(converted neural, z-scored raw traces)` on 5 trials per session (incl. trial 5, neuron 3, t=10) | PASS (atol 1e-4) |
| `brain_region_idx` == area of the matched raw `iarea` (and none in {-1,7}) | PASS |
| `np.allclose` on all 5 inputs for 5 trials per session | PASS |
| output stimulus/licking/position-bin/speed-bin identical for 5 trials per session | PASS |
| position increases monotonically inside a trial and spans ~0.1-3.9 m | PASS |

### Check 3: Reference code comparison
| Step | Reference | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `utils.load_spk`: `np.concatenate([nspk for nspk in np.load(path).item()['spks']],0)` | `load_spk_rows`: identical concatenation order (plane offsets), but only the sampled rows are copied | YES (subset of the same matrix) |
| | `utils.load_retino`: `iarea`, `xy_t` from `<mname>_<datexp>_trans.npz` | same file, same `iarea` | YES |
| | `utils.load_exp_beh`: `beh/Beh_<exp_type>.npy` keyed by `mname_datexp_blk[_stimtype]` | same, with the `_swap1/_swap2` duplicates merged | YES |
| (b) neuron/trial filtering | `neu_area_ID` (V1=8, mHV=0,1,2,9, lHV=5,6, aHV=3,4); `Get_density_map` excludes `iarea` -1 and 7 | identical mapping and exclusion, then a seeded proportional subsample of 1,000 neurons | YES + subsampling (needed: 405 GB raw) |
| | reference applies no per-trial rejection | drops only trials with <5 valid frames (none occurred), unlabeled stimulus (309 `circle3` trials) or non-finite cue/start time (none) | YES |
| (c) temporal alignment | frame-index alignment (`StartFr`, `SoundFr`, `LickFr` index the neural frames); behaviour truncated with `[:nfr]` | same clock; trials aligned to corridor entry; behaviour truncated to `nfr` | YES |
| (d) binning | native imaging frames (3.17 Hz); position analyses use 1 dm bins | native imaging frames kept; position discretised into 4 x 1 m bins as the decoder task requires | YES |
| (e) input construction | `spk_2_cue` aligns to `SoundFr`; `Get_dprime_rewPred_neuron` uses `isRew`/cue position | time to cue from `SoundTime`, cue-onset frame, `isRew`, day from session dates | YES |
| (f) output construction | `WallName`/`stim_id` for stimulus identity; `LickFr` histograms for licking; `ft_Pos`/`ft_RunSpeed` for position and speed | identical source variables, discretised as the decoder task requires | YES |

Differences and their justification:
1. **Neuron subsampling (1,000/session)** - required: the full data is 405 GB and the decoder projects to 100 PCs anyway.
2. **Per-neuron z-scoring** - the reference z-scores deconvolved traces for population analyses (`get_kfold_reward_response`); needed here so that sessions/neurons are on a comparable scale for the shared decoder.
3. **Discretisation of position/speed into 4 bins** - required by the Decoder Task specification.
4. **Dropping 309 `circle3` trials** - `circle3` has no canonical `stim_id` anywhere in the dataset, so no stimulus label can be assigned.

### Check 4: Key statistics comparison
See the table in Step 9 - all statistics available in the paper (89 recordings, 19 mice, 20,547-89,577 neurons per recording, 4 m + 2 m corridor geometry, 3.17 Hz, cue at 0.5-3.5 m, running-only analysis) match the converted dataset.

### Check 5: Edge cases
- Behaviour arrays can be 1 frame longer than the neural data -> everything truncated to `nfr` (as the reference does). Verified: e.g. TX83_2022_08_31_1 has 33,598 behaviour frames vs 33,597 neural frames.
- `ft_trInd` contains NaN outside the behaviour recording -> excluded by `np.isfinite`.
- Trials can be re-entered after a pause; all valid frames of a trial index are grouped (sorted by frame), so no trial is split or duplicated.
- `LickFr` is empty (dtype uint8) for non-water-restricted mice -> handled with `np.atleast_1d` and a finite check; lick frames outside `[0,nfr)` are dropped.
- `stim_id` can be NaN for a stimulus in one experiment type but defined in another -> merged across experiment types; the remaining unlabeled stimulus (`circle3`) is dropped.
- Neurons with zero variance over the included frames are removed before z-scoring (none occurred in practice; the check prevents division by zero).
- Sessions with <2 usable trials would be dropped (none occurred).


### Check 6: Dataset-wide structural checks (`/app/cache/check_plots_numeric.py`)
- `sound_cue_onset` is exactly the first frame with `time_to_sound_cue <= 0` in **all 37,801 trials** (0 inconsistencies); the cue falls inside 37,785 trials (99.96%) - the remainder are trials in which the mouse left the corridor before the cue.
- `time_since_trial_start` is strictly increasing within every trial (0 violations) - no frame ordering errors.
- The position bin is 0 at the first frame and 3 at the last frame of **every** trial (means 0.00 and 3.00), confirming the alignment to corridor entry and the correctness of the 4 x 1 m discretisation.
- Per-cohort behaviour (`/app/cache/stats_cohort.py`): 51 unsupervised, 10 naive, 28 task sessions; licking occurs on 11.7% of task-session timepoints and never in the unsupervised/naive sessions (these mice were not water restricted); 37.6% of task trials are in the rewarded corridor.

### Issues found and resolved
- *Cue-position sanity check initially looked wrong* (99th pct 36 m): it was computed from elapsed **time** x VR speed, which is invalid for trials in which the mouse pauses (the paused frames are excluded by the running filter but wall-clock time keeps advancing). Re-checked against the raw `SoundPos`/`ft_Pos` at `SoundFr`: 0.4-3.6 m, matching the paper. The conversion itself was correct.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, NVIDIA L4).

### Training Progress
- Loss decreasing: **Yes** (1.29 at epoch 1 -> 0.267 at epoch 200; test loss 0.695).

### Decoder Results (Full, 89 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|--------|-------|
| stimulus_category | 0.9715 | 0.8274 | 0.1429 | 5.8x chance |
| licking | 0.9691 | 0.8115 | 0.5000 | 1.6x chance; only the 28 task sessions contain licks |
| position_bin | 0.9817 | 0.8434 | 0.2500 | 3.4x chance |
| speed_bin | 0.7791 | 0.5787 | 0.2500 | 2.3x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Validation Acc | Chance | Ratio |
|---|---|---|---|
| stimulus_category | 0.8274 | 0.1429 | 5.8x |
| licking | 0.8115 | 0.5000 | 1.6x |
| position_bin | 0.8434 | 0.2500 | 3.4x |
| speed_bin | 0.5787 | 0.2500 | 2.3x |

No output is at or below chance. Licking is at 1.6x chance, which is the ceiling for a binary variable (0.81 balanced accuracy on a variable that is 96% zeros overall and only ever non-zero in 28 of 89 sessions). Speed is the lowest multiple of chance; this is expected rather than a bug:
- the virtual reality clamps the *visual* speed at 60 cm/s whenever the mouse runs above the 6 cm/s threshold, so the running speed above threshold has little correlate in the visual input;
- the quartile edges are global (12.4, 25.3, 40.8 cm/s), so a large part of the speed label reflects between-session/between-mouse differences in running vigour rather than within-trial dynamics. Per-session distributions therefore vary strongly, which is the intended consequence of the "each bin = 25% of the data" instruction.

### Check 2: Comparison to the paper
The paper reports **no decoding analyses at all** (the extracted PDF text contains no occurrence of "decod"). The closest quantitative benchmarks are the selectivity analyses (e.g. fractions of stimulus-selective neurons at d' >= 0.3), which are not decoder accuracies. There is therefore no published accuracy to compare against; instead the results were checked against the structure of the task: stimulus identity is a per-trial constant driven by the visual input to visual cortex (highest accuracy, 5.8x chance), position in a fixed-speed corridor is strongly encoded (3.4x chance), licking is only present in a third of the sessions, and running speed is largely decoupled from the visual stream by the VR speed clamp.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| stimulus_category | 0.9715 | 0.8274 | 1.17 |
| licking | 0.9691 | 0.8115 | 1.19 |
| position_bin | 0.9817 | 0.8434 | 1.16 |
| speed_bin | 0.7791 | 0.5787 | 1.35 |

All ratios are below 1.5, so there is no sign of severe overfitting or leakage. The train/validation split is over trials within each session (performed by `train_validate_decoder`), and every output is derived only from the frames of its own trial, so no information leaks across the split.

### Additional experiment: are the heavy tails in the time inputs harmful?
27% of trials contain a pause inside the corridor (the paused frames are excluded by the reference running filter, so the wall-clock `time_since_trial_start` can reach 1,765 s in 1% of trials). Clipping both time inputs to +/-30 s and re-training on the sample changed validation accuracy only marginally (stimulus 0.898 -> 0.906, licking 0.859 -> 0.874, position 0.830 -> 0.833, speed 0.509 -> 0.510). The unclipped, physically correct elapsed times were therefore kept, and the binary `sound_cue_onset` input provides the decoder with a scale-free version of the cue timing.

### Issues found and resolved
- No further issues: all format checks pass, all independent raw-data sanity checks pass, and every accuracy is above chance with a small train/validation gap.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format specification, key statistics, reproduction commands)
- [x] `cache/` folder created with `README_CACHE.md` documenting every exploration/validation script
- [x] All files organized:
  - conversion: `convert_data.py`, `converted_data.pkl` (3.31 GB), `sample_data.pkl`
  - logs: `conversion_sample_out.txt`, `conversion_full_out.txt`, `verification_sample_out.txt`, `verification_full_out.txt`, `train_decoder_sample_out.txt`, `train_decoder_full_out.txt`
  - figures: `processing_TX108_2023_03_25_1.png`, `processing_TX83_2022_08_31_1.png` (conversion steps), `sample_trials.png`, `predictions.png` (decoder)
  - documentation: `CONVERSION_NOTES.md`, `README.md`, `cache/README_CACHE.md`

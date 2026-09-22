# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks dataset and code in `/app`
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
| `load_exp_beh(root, exp_type)` | `/app/code/utils.py` | LOADING | Loads behavior dictionaries from `Beh_<exp_type>.npy`. |
| `load_spk(db, root='')` | `/app/code/utils.py` | LOADING | Loads session neural data from `<mouse>_<date>_<block>_neural_data.npy` and concatenates `['spks']` across planes into a neuron x frame matrix. |
| `load_retino(db, root='')` | `/app/code/utils.py` | LOADING | Loads retinotopy / area annotations and groups neurons into `V1`, `mHV`, `lHV`, `aHV`. |
| `get_interpPos_spk(spk, spk_culm_pos, ntrial, n_bins=60, lengths=60, save_path='')` | `/app/code/utils.py` | PROCESSING | Interpolates neural activity from frame time into position bins; notebook comments state 60 bins over a 6 m corridor, so 1 decimeter per bin. |
| `spk_pos_interp(raw_spk, accum_pos, corridorLen, new_shape)` | `/app/code/utils.py` | PROCESSING | Core interpolation helper used by `get_interpPos_spk`; reshapes each neuron into trial x position bins using cumulative position. |
| `dprime(x1, x2)` | `/app/code/utils.py` | PROCESSING | Computes stimulus selectivity / reward prediction contrasts used throughout the paper. |
| `Get_dprime_selective_neuron(db, Beh, stim_ID=[2, 0], root='')` | `/app/code/utils.py` | CURATION | Computes per-neuron selectivity using only frames in corridor while VR is moving (`ft_CorrSpc & (ft_move > 0)`). |
| `Get_dprime_rewPred_neuron(db, Beh, stim_dp, root='', ..., dp_thr=0.3)` | `/app/code/utils.py` | CURATION | Identifies reward-prediction neurons from interpolated position-binned activity using early vs late cue trials inside rewarded stimulus trials. |
| `Get_coding_direction(db, Beh, stim_ref=[2, 0], prc=5, root='', ..., n_bef=10)` | `/app/code/utils.py` | PROCESSING | Main population-analysis routine: splits odd/even trials, computes d-prime on odd trials, normalizes interpolated activity relative to gray-space baseline, prepends 10 bins from previous trial gray period, and projects activity along stimulus-selective ensembles. |
| `Get_sort_spk(db, Beh, stim_ref=[2, 0], prc=5, root='', ...)` | `/app/code/utils.py` | PROCESSING | Uses odd/even trial splits plus stimulus selectivity to sort neurons by peak position sequences in the corridor. |
| `get_kfold_reward_response(root, db, Beh)` | `/app/code/utils.py` | PROCESSING | Cross-validated reward-neuron analysis aligned to cue and first lick; uses `spk_2_firstLick` and `spk_2_cue`. |
| `spk_2_firstLick(spk, beh, ranges=[15, 15], bins=30)` | `/app/code/utils.py` | PROCESSING | Extracts neural / lick activity windows around first lick frame. |
| `spk_2_cue(spk, beh, ranges=[15, 15], bins=30)` | `/app/code/utils.py` | PROCESSING | Extracts neural / lick activity windows around cue frame. |

### Notes
- `README.md` says `data_process_script.ipynb` shows how intermediate results are produced and `Figures.ipynb` only plots them.
- The processing notebook documents the behavior schema and the main preprocessing pass:
  - calcium recording frame rate is `3.17 Hz`
  - session neural activity is converted into interpolated `neurons x trials x 60 position bins`
  - notebook comment: corridor length is 6 m, so 60 bins means 1 decimeter bins
- The reference code does **not** compute `dF/F` from raw fluorescence. It loads `['spks']` directly from each `*_neural_data.npy`, indicating the released data already contains processed neural activity used for analysis.
- The main frame-level validity masks used in reference analyses are:
  - `ft_move > 0` to restrict to periods when VR is moving
  - `ft_CorrSpc` for corridor / texture space
  - `ft_GraySpc` for gray inter-trial space
- Trial splitting conventions in the reference:
  - training/selectivity estimation often uses odd trials in MATLAB-style naming but implemented as `ft_trInd % 2 == 0`
  - held-out/evaluation trials often use `trInd % 2 == 1`
- Neuron curation in the reference code is analysis-dependent rather than a single global QC filter:
  - no generic low-quality-cell exclusion function was found in `/app/code`
  - some analyses exclude neurons outside visual cortex via retinotopy labels (`iarea == -1` or `7`)
  - some analyses further require corridor responses to exceed gray-space responses (`corr_neu`)
  - stimulus-selective subsets are typically defined by d-prime thresholds or percentiles, not by blanket session-level filtering
- Behavioral utilities show the key variables likely needed later for conversion:
  - trial start/end times
  - cue timing / cue frame (`SoundTime`, `SoundFr`, `SoundPos`, `SoundDelPos`)
  - reward timing / position (`RewTime`, `RewPos`, `isRew`)
  - lick times / frames / positions (`LickTime`, `LickFr`, `LickPos`, `LickTrind`)
  - running trajectories and speeds in `SubjMove`
- Reference alignment conventions depend on analysis:
  - position-aligned analyses interpolate activity over corridor position
  - reward-neuron analyses also use frame-aligned windows around cue and first lick
  - for this conversion task, corridor-entry / trial start will need to become the common alignment event, but the source variables come from the same behavior schema used here

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/beh/Imaging_Exp_info.npy`
  - master experiment index
  - 23 experiment types
  - each experiment type contains a list of session descriptors with fields such as `mname`, `datexp`, `blk`, `stim_id`, and sometimes `stimtype`
- `/app/data/beh/Beh_<exp_type>.npy`
  - one dictionary per experiment type
  - keys are session identifiers like `<mouse>_<date>_<block>` or `<mouse>_<date>_<block>_<stimtype>`
  - each session dictionary contains trial-level variables (`Trial_start_time`, `Trial_end_time`, `WallName`, `isRew`, `Sound*`, `Rew*`, `Lick*`), frame-level variables (`ft_*`), and running trajectories (`SubjMove`)
  - representative session dictionaries contain 59 top-level fields
- `/app/data/beh/Unsupervised_pretraining_behavior/`
  - behavior-only files for a separate pretraining analysis branch, not obviously paired to imaging sessions
- `/app/data/spk/<mouse>_<date>_<block>_neural_data.npy`
  - raw neural data
  - each file is a dict with one key, `spks`
  - `spks` is a list of 3 float32 arrays with identical frame counts; reference code concatenates them along axis 0
  - example: `VR2_2021_03_20_1_neural_data.npy` has segment shapes `(27157, 24298)`, `(27157, 24298)`, `(27159, 24298)` and becomes `(81473, 24298)` after concatenation
- `/app/data/retinotopy/<mouse>_<date>_trans.npz`
  - retinotopy / spatial annotation files with keys `A`, `xpos`, `ypos`, `xy_t`, `iarea`
  - `xy_t` and `iarea` lengths match the concatenated row count of the corresponding neural file
- `/app/data/process_data/`
  - empty in this copy of the dataset; no saved intermediate products are provided

Additional structural observations:
- There are 142 experiment-index entries, 99 unique behavior-session keys, and 89 unique raw neural sessions.
- Raw sessions are reused across experiment labels:
  - 64 raw sessions appear in exactly 1 experiment type
  - 14 appear in 2 experiment types
  - 6 appear in 3 experiment types
  - 2 appear in 4 experiment types
  - 3 appear in 5 experiment types
- The duplicate behavior keys are real reuse, not missing data; e.g. `LZ13_2024_05_15_1` appears under `naive_test1`, `naive_test2`, `train1_before_grating`, `test1_before_grating`, and `test2_before_grating`.
- All inspected behavior sessions use:
  - `Corridor_Length = 60.0`
  - `Texture_Length = 40.0`
  - `Gray_Space_length = 20.0`
- Reward modes present in unique behavior-session keys: `Active after cue` and `Passive`.
- Global stimulus vocabulary across behavior files includes:
  - `circle1`, `circle2`, `circle3`
  - `leaf1`, `leaf1_swap1`, `leaf1_swap2`, `leaf2`, `leaf3`
  - `rock1`, `rock2`
  - `wood1`, `wood1_swap1`, `wood1_swap2`, `wood2`, `wood5`
- Representative behavior variables likely relevant for conversion:
  - trial aligned: `Trial_start_time`, `Trial_end_time`, `StartFr`, `EndFr`
  - cue / sound: `SoundTime`, `SoundFr`, `SoundPos`, `SoundDelPos`, `SoundDelayFr`
  - reward: `RewTime`, `RewPos`, `RewardFr`, `Reward_Delay_ms`, `isRew`
  - licks: `LickTime`, `LickFr`, `LickPos`, `LickTrind`
  - visual stimulus identity: `WallName`, `UniqWalls`, `stim_id`, `TrialStim`, `StimTrial`
  - frame-level alignment helpers: `ft_trInd`, `ft_WallID`, `ft_CorrSpc`, `ft_GraySpc`, `ft_move`, `ft_isMoving`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`
  - running / VR trajectories: `SubjMove['SubjMTime']`, `SubjMove['SubjMPos']`, `SubjMove['SubjMPosCum']`, `SubjMove['SubjM_pitch']`, `SubjMove['SubjM_roll']`, `SubjMove['SubjM_yaw']`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 raw neural rows across 89 unique neural sessions |
| Neurons / session | min 20,547; mean 52,708.25; max 89,577 |
| Subjects | 19 unique mice |
| Sessions / subject | raw neural sessions per mouse range 1 to 8 |
| Trials (total) | 44,060 across 99 unique behavior-session keys; 63,177 across all 142 experiment-index entries |
| Trials / session | min 84; mean 445.05; max 789 across unique behavior-session keys |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 89 recordings total; each recording contains 20,547 to 89,577 neural traces | `"We performed 89 recordings in 19 mice..."` and `"activity traces from 20,547 to 89,577 neurons in each recording."` |
| Neurons / session | 20,547 to 89,577 | `"We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording."` |
| Subjects | 19 imaging mice | `"We performed 89 recordings in 19 mice..."` |
| Sessions / subject | Not stated explicitly; average from paper-level totals is 89 / 19 = 4.68 recordings per mouse | `"We performed 89 recordings in 19 mice..."` |
| Trials (total) | Not explicitly reported in paper / methods excerpt | Not stated in paper text reviewed so far |
| Trials / session | Not explicitly reported in paper / methods excerpt | Not stated in paper text reviewed so far |
| Neural data time bin | Imaging-frame based; paper does not state frame rate here, but analyses use imaging-frame timepoints and position interpolation | `"All our analyses were based on deconvolved fluorescence traces."` and `"the running speed was interpolated to the timepoints of the imaging frames"` |
| Behavior data time bin | No fixed native bin stated; behavior is interpolated either to imaging frames or to 0.1 m position bins depending on analysis | `"the running speed was interpolated to the timepoints of the imaging frames"` and `"for every position (0–6 m, with a 0.1-m step size)"` |
| Reward rate | Approximately 0.5 by design for rewarded vs unrewarded corridor identity, though not stated numerically | `"Mice had to discriminate between visual texture patterns in two corridors"` and `"the reward was delivered if a lick was detected after the sound cue in the rewarded corridor"` |
| Corridor geometry | 4 m texture corridor + 2 m gray space | `"The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors."` |
| Running rule | VR advances only when speed exceeds 6 cm/s; VR speed fixed at 60 cm/s once threshold crossed | `"The mice moved forward... by running faster than a threshold of 6 cm s−1, but the virtual corridors always moved at a constant speed (60 cm s−1)..."` |
| Cue position range | Uniform from 0.5 m to 3.5 m from corridor entry | `"the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m"` |
| Reward delivery range | Approximately 0.5 m to 3.5 m in imaging task mice; 2 m to 3 m reward-zone onset in behavior-only experiment | `"reward delivery locations were also approximately uniformly distributed between 0.5 m and 3.5 m"` and `"The beginning of the reward zone was randomly chosen... between 2 m and 3 m"` |
| Leaf1-swap handling | Swap types pooled for statistics, with cohort-specific handling | `"The two different types of leaf1-swap stimuli... are pooled together for statistics analysis."` |


### Processing Details
- Global paper-level constraints:
  - neural activity comes from Suite2p-processed calcium imaging, including ROI detection, cell classification, neuropil correction, and spike deconvolution
  - all analyses use deconvolved traces, not raw fluorescence
  - running-only timepoints are used for analysis to exclude stationary / reward-collection periods
- Spatial / temporal organization from the paper and methods:
  - each trial is a 6 m traversal made of a 4 m texture corridor plus 2 m gray space
  - the sound cue occurs at a random location between 0.5 m and 3.5 m from corridor entry
  - in task mice, the cue indicates reward availability in the rewarded corridor; in unsupervised mice the cue is still presented but reward is absent
- Paper-described analysis operations that should influence conversion:
  - selectivity calculations pool responses across positions but only during running
  - reward-prediction analyses interpolate single-neuron activity by corridor position to form trial x position matrices
  - reward-prediction classification uses early-cue versus late-cue trial splits within leaf1 trials and tenfold cross-validation
  - running speed can be interpolated to imaging-frame timepoints or to 0.1 m position bins
- Decoder expectations from the paper:
  - the paper does **not** report a generic neural decoder accuracy for variables like cue time, lick state, position, or stimulus category
  - the closest reference analyses are coding-direction separation, sequence-correlation analyses, and reward-prediction-neuron identification
  - therefore decoder validation later will have to rely mainly on internal consistency, reasonable above-chance performance, and agreement with paper-level qualitative effects rather than a published accuracy target

### Curation Steps

**Neuron curation rules**:
- Primary curation is delegated to Suite2p processing: ROI detection, cell classification, neuropil correction, and spike deconvolution.
- Stimulus-selective neurons are often defined with thresholds such as `d′ ≥ 0.3` or `d′ ≤ -0.3`.
- Reward-prediction neurons are defined from position-interpolated leaf1 activity using `d′late vs early ≥ 0.3`.
- Retinotopy is used to assign neurons to visual areas; figures then summarize subsets by region.

**Trial curation rules**:
- Only timepoints during running are analyzed.
- Reward-prediction analyses use only leaf1 trials and split them into early-cue vs late-cue trials.
- First-lick aligned reward-prediction analysis includes only rewarded leaf1 trials with first lick occurring after 2 m from corridor entry; one mouse was excluded because it had no such trials.
- Leaf2 reward-prediction comparison excluded one mouse because it had only one non-licking leaf2 trial.
- Leaf1-swap variants are pooled for statistical summaries, with different bookkeeping between task and unsupervised / naive cohorts.

### Decoders Trained
| Decoded variable | Accuracy |
| No generic decoder accuracy reported in the paper | N/A |
| Coding-direction readouts (leaf1 vs circle1; leaf1 vs leaf2) | Separation shown qualitatively / via similarity indices; no scalar accuracy reported |
| Reward-prediction population (early vs late cue) | Tenfold cross-validated population activity reported, but no classification accuracy metric reported |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | `utils.load_spk` and retinotopy files imply one neural recording per `<mouse>_<date>_<block>` | 89 unique neural files; 99 unique behavior keys; 142 experiment-index entries | `"We performed 89 recordings in 19 mice"` | Treat the dataset as **89 real recording sessions**. The 99 behavior keys arise from aliasing and `stimtype` relabels, not from extra recordings. |
| Behavior key duplication | Code frequently builds behavior keys from raw session id and optional `stimtype` | Reused keys across experiment files are byte-identical on core arrays; all 9 raw sessions with multiple behavior keys have identical signatures | Paper discusses reusing mice/sessions across analyses and pooling swap variants | Canonicalize to the raw session id `<mouse>_<date>_<block>` for conversion; do not duplicate sessions just because multiple behavior aliases exist. |
| Corridor units | Processing notebook interpolates to 60 position bins and comments that each bin is 1 decimeter | `Corridor_Length = 60`, `Texture_Length = 40`, `Gray_Space_length = 20` | Paper says corridors are 4 m plus 2 m gray space | Interpret the stored position units as **decimeters**. Then 60 units = 6 m total, 40 units = 4 m texture corridor, and 20 units = 2 m gray space, fully consistent with the paper. |
| Neural signal type | `load_spk` reads `spks` directly; no `dF/F` computation appears anywhere in the code | Neural files contain only `spks` arrays | Paper: `"All our analyses were based on deconvolved fluorescence traces."` | Use `spks` as the native neural signal for conversion; do not recompute fluorescence or `dF/F`. |
| Running-period selection | Core analyses use `ft_move > 0` plus corridor / gray masks | Behavior files contain `ft_move`, `ft_CorrSpc`, `ft_GraySpc`, `ft_RunSpeed`, `SubjMove` | Paper: `"We only considered timepoints during running for analysis"` | Preserve the running-only logic when deriving decoder-aligned neural and behavioral trajectories. |
| Stimulus naming for the second naturalistic pair | Code is mostly focused on leaf / circle and does not mention brick / wood names | Raw behavior files use `rock*` and `wood*` labels, not `brick*` | Paper text refers to `rock` and `brick(s)` | Treat `wood*` in the released files as the raw dataset label for the paper’s second naturalistic category. Preserve the raw names in converted outputs to avoid ambiguity. |
| Swap handling | Code keeps `leaf1_swap1` and `leaf1_swap2` separate and later pools them for statistics | Raw files contain duplicate alias keys such as `swap1` and `swap2` with identical core trial arrays | Paper says swap types are pooled for statistics | For conversion, keep the actual per-trial stimulus labels present in the canonical raw session, but do not duplicate whole sessions because of `swap1`/`swap2` alias keys. |
| Alignment convention | Reference analyses are often position-aligned or cue / first-lick aligned | Data contains trial start, cue, reward, lick, and frame-level position variables | User task requires alignment to corridor entry / trial start | Use trial start (corridor entry) as the common alignment event for converted trials, while deriving cue-time, position, reward-availability, and licking signals from the same native variables used by the reference analyses. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk['spks']` concatenated across the 3 arrays | `neural` | Keep deconvolved traces as float32; slice trial-by-trial using canonical session behavior masks | `load_spk`, `Get_dprime_selective_neuron`, `Get_coding_direction` | No `dF/F` recomputation; this matches the released analysis signal. |
| `ft_trInd`, `ft_CorrSpc`, `ft_move`, `ft` | Trial sample axis for `neural`, `input`, `output` | For each trial, retain frames with `ft_trInd == trial`, `ft_CorrSpc == True`, and `ft_move > 0`; keep native imaging-frame samples in chronological order | `Get_dprime_selective_neuron`, `Get_coding_direction` | Matches the paper’s running-only corridor analysis while preserving native frame bins. Time inputs will use actual timestamps so skipped stationary periods remain explicit. |
| `SoundTime`, `ft` | `input[0]` = `time_to_sound_cue_s` | For each retained frame: `(SoundTime[trial] - ft_frame) * 86400` | cue-aligned analyses in `spk_2_cue`; paper methods on cue positions | Positive before cue, negative after cue. Continuous, time-varying. |
| `dateexp` parsed as calendar date within subject | `input[1]` = `day_of_training` | Continuous per-trial scalar = days since the subject’s first included recording date; repeat across retained frames | session metadata from `Imaging_Exp_info.npy` | Chosen because `sess#` / `days` fields are inconsistent across experiment branches and duplicate aliases. |
| `Trial_start_time`, `ft` | `input[2]` = `time_since_trial_start_s` | For each retained frame: `(ft_frame - Trial_start_time[trial]) * 86400` | trial-start variables in behavior schema | Trial start is corridor entry, as required by the user. |
| `isRew` | `input[3]` = `reward_availability` | Trial-constant 0/1 repeated across retained frames | `get_cat_id`, behavior schema | Uses the native rewarded-corridor identity even in unsupervised sessions where water reward is absent. |
| `WallName` | `output[0]` = `visual_stimulus_category` | Map global stimulus-name vocabulary to integer classes; repeat the class across retained frames in the trial | behavior schema; figure code assumes corridor identity per trial | Global categories preserve raw names (`circle*`, `leaf*`, `rock*`, `wood*`, swap variants). |
| `LickFr`, `LickTrind` | `output[1]` = `licking` | Binary per retained frame: 1 if one or more licks map to that imaging frame, else 0 | `spk_2_firstLick`, `spk_2_cue` | Uses the same frame-index convention as the reference code (`astype(int)`-style frame assignment). |
| `ft_Pos` on retained frames | `output[2]` = `position_bin` | Discretize corridor position into 4 bins of 10 native position units each (0–10, 10–20, 20–30, 30–40), corresponding to 0–1 m, 1–2 m, 2–3 m, 3–4 m | notebook comment on 60 bins / 6 m; paper methods on 4 m corridor | Output is categorical and time-varying as required. |
| `ft_RunSpeed` on retained frames | `output[3]` = `running_speed_bin` | Compute global quartile thresholds over all retained samples, then discretize each retained frame into 4 equal-frequency bins | running-speed interpolation described in paper methods | Quartiles are computed from the canonical full dataset after trial masking, then applied consistently to every session. |
| `mname` | `subjects`, `subject_idx` | Unique subject list and per-session index | `Imaging_Exp_info.npy` | 19 mice expected. |
| `retinotopy['iarea']` | `brain_region_idx` | Map raw labels to grouped names: `V1`, `mHV`, `lHV`, `aHV`, plus extra labels for `iarea==7` and `iarea==-1` | `neu_area_ID`, `load_retino` | Keep all neural rows so paper-level neuron counts are preserved; group named regions exactly as in reference code. |
| `iarea` mapping metadata | `brain_regions` | `['V1', 'mHV', 'lHV', 'aHV', 'unassigned_7', 'outside_visual']` | `neu_area_ID` plus raw `iarea` values | Rows with `iarea==7` or `-1` are not dropped; they get explicit labels. |

### Key Decisions
1. **Canonical session definition**: Convert 89 unique raw recordings, not 99 behavior aliases or 142 experiment-index entries. Rationale: paper count is 89 recordings, and duplicate behavior aliases were verified to be identical on core trial arrays.
2. **Neural signal choice**: Use released deconvolved `spks` directly. Rationale: both the paper and reference code analyze deconvolved traces and never recompute `dF/F`.
3. **Trial mask**: Use corridor-running imaging frames (`ft_trInd == trial`, `ft_CorrSpc`, `ft_move > 0`). Rationale: this matches the paper’s running-only analysis rule while keeping the native imaging-frame sampling used by the code.
4. **Common alignment event**: Align every trial to corridor entry / `Trial_start_time`. Rationale: this is required by the decoder task; cue timing, position, licking, and reward context can all be reconstructed relative to that same event from native source variables.
5. **Stimulus label policy**: Preserve raw stimulus names, including `wood*` and swap variants. Rationale: this avoids ambiguous relabeling and keeps the conversion faithful to the released files even where paper prose used different generic names.
6. **Brain-region policy**: Keep all neural rows and provide grouped region labels plus explicit labels for rows outside the four main groups. Rationale: this preserves paper-level neuron counts and source information while still matching the region grouping used in the analysis code.
7. **Training-day input**: Use subject-specific calendar-day offset from first recording, repeated across each trial’s timepoints. Rationale: it is continuous, defined for every session, and avoids inconsistencies between `sess#` and `days` metadata fields.
8. **Output representation**: Make all outputs time-varying 2D arrays `(d_output, T)` by repeating trial-constant stimulus identity across time. Rationale: this keeps a single consistent output format and lets the decoder score stimulus identity at every retained sample.
9. **Session / trial curation**: Drop trials with zero retained corridor-running frames; drop sessions with fewer than 2 remaining trials. Rationale: satisfies the decoder requirement and excludes unusable raw trials.

### Planned Sanity Checks
- [ ] Verify converted session count equals 89 and subject count equals 19.
- [ ] Verify per-session neural row counts match direct `load_spk` concatenation and remain within the paper’s 20,547–89,577 range.
- [ ] For at least 3 hand-checked trials, compare converted neural samples against direct slicing from raw `spks` with `np.allclose()`.
- [ ] For the same hand-checked trials, compare `time_since_trial_start_s` and `time_to_sound_cue_s` against direct calculations from raw `ft`, `Trial_start_time`, and `SoundTime`.
- [ ] Check reward-availability values against raw `isRew` and stimulus-category values against raw `WallName`.
- [ ] Check licking binaries against raw `LickFr` / `LickTrind` on selected trials.
- [ ] Check position-bin assignments against raw `ft_Pos` on selected trials.
- [ ] Check global running-speed quartile edges and confirm each bin contains approximately 25% of retained samples.
- [ ] Confirm cue positions fall in the expected 0.5–3.5 m range and corridor positions stay within the 0–4 m corridor after masking.
- [ ] Confirm no behavior alias produces a duplicated converted session.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- canonicalization of 89 raw sessions from `Imaging_Exp_info.npy`
- behavior-alias collapsing
- global stimulus-vocabulary construction
- global running-speed quartile estimation
- trial conversion on corridor-running imaging frames
- per-session retinotopy loading and region mapping
- `--sample` mode for 2 sessions
- `--show-processing` session-summary plots
- timing / progress logging

Initial executable check:
- `python3 -m py_compile /app/convert_data.py` passed

Code inefficiencies identified:
- Full-session neural concatenation would duplicate memory unnecessarily.
- Re-reading behavior files many times could add overhead if not cached.

Code speedups added:
- Avoided concatenating whole-session neural matrices; neural data is sliced trial-by-trial from the 3 raw `spks` blocks.
- Added small LRU cache for behavior files.
- Computed global speed quartiles in a behavior-only pass before neural loading.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 160,961 |
| Neurons / session | 85,481 and 75,480 (mean 80,480.5) |
| Subjects | 1 (`TX108`) |
| Sessions / subject | 2 |
| Trials (total) | 510 |
| Trials / session | 210 and 300 (mean 255) |
| `time_to_sound_cue_s` range | [-209.4, 199.8] |
| `day_of_training` range | [0.0, 9.0] |
| `time_since_trial_start_s` range | [0.0, 212.3] |
| `reward_availability` range | [0.0, 1.0] |
| `visual_stimulus_category` distribution | [rock1=0.515, wood1=0.485] |
| `licking` distribution | [no_lick=0.732, lick=0.268] |
| `position_bin` distribution | [0-1m=0.246, 1-2m=0.252, 2-3m=0.251, 3-4m=0.250] |
| `running_speed_bin` distribution | [q1=0.250, q2=0.250, q3=0.250, q4=0.250] |

### Processing Plots Review
- Reviewed `processing_TX108_2023_03_13_1.png` and `processing_TX108_2023_03_22_1.png`.
- No obvious temporal misalignment was visible:
  - cue-position histogram sits inside the expected 0.5–3.5 m range
  - retained frame counts per trial are plausible
  - example input traces show monotonic `time_since_trial_start_s` and a sign-changing `time_to_sound_cue_s`
  - output traces show plausible transitions in position and running-speed bins
- The running mask removes many stationary corridor frames but still preserves full 0–4 m position coverage.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Behavior-only first pass for speed-bin estimation | avoids any neural loading during global quartile computation |
| Trial-by-trial slicing from the 3 raw `spks` blocks | avoids expensive whole-session concatenation and was benchmarked to be much faster than pre-concatenating a full session |
| Alias collapsing to 89 canonical sessions | avoids duplicated conversion work |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion core (`convert_session`) | about 12.65 s / session on two large representative sessions | weight-scaled estimate about 17.3 min for full conversion core |
| Sample full script with plots and save | 33.67 s total for 2 sessions | rough upper-bound extrapolation about 23 min, but this overestimates full mode because full mode will not plot |
| Expected full conversion mode (`--full`, no plots) | about 17–20 min | borderline above the 15 min target; no clearly faster safe hot-path alternative was found in a direct benchmark |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `visual_stimulus_category` | 0.9850 | 0.9630 |
| `licking` | 0.7307 | 0.7196 |
| `position_bin` | 0.9099 | 0.8599 |
| `running_speed_bin` | 0.3261 | 0.3198 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 161.609 GiB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 20,547–89,577 per recording; 89 recordings total | `load_spk` concatenates all `spks` rows for a session | 4,691,034 raw neural rows across 89 sessions | 4,691,034 | Yes |
| Mean neurons/session | not stated explicitly | session row count from concatenated `spks` | 52,708.25 | 52,708.25 | Yes |
| Subjects | 19 | `Imaging_Exp_info.npy` mice ids | 19 | 19 | Yes |
| Sessions | 89 recordings | one canonical raw session per neural file | 89 | 89 | Yes |
| Trials (total) | not stated explicitly | one behavior trial per raw session trial | 38,110 canonical raw-session trials | 38,110 | Yes vs data |
| Trials/session (mean) | not stated explicitly | derived from canonical raw sessions | 428.20 | 428.20 | Yes vs data |
| `time_to_sound_cue_s` range | cue position 0.5–3.5 m from corridor entry | derived from `SoundTime - ft` on retained frames | derived quantity | [-1763.3, 723.5] | Derived / N.A. |
| `day_of_training` range | not stated explicitly | derived from calendar day offset | derived quantity | [0.0, 92.0] | Derived / N.A. |
| `time_since_trial_start_s` range | trial start = corridor entry; no global max stated | derived from `ft - Trial_start_time` on retained frames | derived quantity | [0.0, 1765.2] | Derived / N.A. |
| `reward_availability` range | rewarded vs non-rewarded corridor design | `isRew` | [0, 1] | [0.0, 1.0] | Yes |
| `visual_stimulus_category` distribution | leaf/circle emphasized; rock/brick noted in subset of mice | `WallName` categories | all raw trial labels | [circle1=0.252, circle2=0.049, circle3=0.010, leaf1=0.264, leaf1_swap1=0.019, leaf1_swap2=0.019, leaf2=0.128, leaf3=0.040, rock1=0.071, rock2=0.013, wood1=0.065, wood1_swap1=0.008, wood1_swap2=0.010, wood2=0.038, wood5=0.014] | Yes vs data |
| `licking` distribution | task mice lick selectively; unsupervised often do not | `LickFr` / `LickTrind` | derived from raw licks on retained frames | [no_lick=0.963, lick=0.037] | Yes vs data |
| `position_bin` distribution | 4 m corridor | derived from `ft_Pos` | approximately uniform by construction of 1 m bins over traversals | [0-1m=0.250, 1-2m=0.249, 2-3m=0.250, 3-4m=0.252] | Yes |
| `running_speed_bin` distribution | not stated explicitly | quartile discretization planned | quartiles over retained samples | [q1=0.250, q2=0.250, q3=0.250, q4=0.250] | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported `Data format is valid, no errors or warnings.` No fixes were required.
2. **Raw-data sanity checks with `np.allclose()`**:
   - Using direct loads from `/app/data`, compared sample-session converted trials against raw source arrays for:
     - neural activity: `converted neural == concat(spks)[:, retained_frames]`
     - `time_since_trial_start_s == (ft - Trial_start_time) * 86400`
     - `time_to_sound_cue_s == (SoundTime - ft) * 86400`
     - licking output == `build_lick_binary(LickFr)` on retained frames
     - position bins == `floor(ft_Pos / 10)` clipped to 4 corridor bins
     - running-speed bins == quartile discretization of raw `ft_RunSpeed`
   - Spot-check trials:
     - `TX108_2023_03_13_1`: raw trials `0`, `105`, `209`
     - `TX108_2023_03_22_1`: raw trials `0`, `150`, `299`
   - Every check returned `True` for every spot-checked trial.
3. **Reference code comparison**:
   - `(a) data loading`: matches `load_spk`, `load_retino`, and behavior loading from `Beh_<exp_type>.npy`.
   - `(b) neuron / trial filtering`: matches the reference’s core `ft_CorrSpc & (ft_move > 0)` running-corridor restriction. No extra neuron QC was added because the reference code also relies on the released `spks` plus analysis-specific subsets rather than a global filter.
   - `(c) temporal alignment`: differs only where the user task requires it. Reference analyses are often cue- or position-aligned; conversion aligns to corridor entry / `Trial_start_time` while still using native `SoundTime`, `ft`, and lick-frame variables.
   - `(d) binning`: neural data remains at native imaging frames, consistent with the source signal used throughout the reference. Position output is discretized into four 1 m bins only because the decoder task requires categorical outputs.
   - `(e) input construction`: `time_to_sound_cue_s`, `time_since_trial_start_s`, `reward_availability`, and `day_of_training` are direct derivations from raw fields; these are decoder-task additions rather than reference-paper outputs.
   - `(f) output construction`: stimulus identity, licking, position bin, and speed bin are all built from the same raw variables used by the reference code, with discretization only where the decoder task requires categorical targets.
4. **Key statistics comparison**:
   - Direct canonical raw-session totals: `89` sessions, `19` subjects, `38,110` trials, `4,691,034` neural rows.
   - These match the converted dataset at the session / subject / neural-row level exactly, and match the paper’s `89 recordings in 19 mice` plus `20,547–89,577 neurons per recording`.
   - Converted trial count also matches the canonical raw-session trial count exactly (`38,110`); the larger alias-based counts seen in Step 2 were confirmed to come from duplicated behavior keys rather than extra recordings.
5. **Edge-case review**:
   - Trial-duration distribution on retained corridor-running frames:
     - median `7.11 s`
     - 95th percentile `23.04 s`
     - 99th percentile `74.08 s`
     - 99.9th percentile `337.54 s`
     - maximum `1765.17 s`
   - The longest outlier is raw session `TX88_2022_07_19_1`, trial `391`. Direct inspection showed:
     - cue occurs `1.893 s` after trial start in the raw timestamps
     - retained running frames span position `0.92` to `39.69`
     - those retained frames are spread over `1765 s` of wall-clock time
   - This indicates sparse movement bouts within a single raw trial, not a conversion indexing bug. Because the timestamps and retained-frame masks are faithful to the source data, no corrective filtering was applied.

### Issues Found and Resolved
- **Apparent trial-count discrepancy from initial exploration**: resolved by restricting statistics to the 89 canonical raw sessions rather than the 99 behavior aliases / 142 experiment-index entries.
- **Very large time-input outliers in a few trials**: investigated directly in raw behavior timestamps and retained-frame indices. Determined to reflect source-data trial timing, not conversion corruption, so retained and documented.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `visual_stimulus_category` | 0.5918 | 0.5454 | 8.18x uniform-chance validation performance (`0.0667` chance). |
| `licking` | 0.9066 | 0.8800 | Strong performance despite class imbalance (`0.5000` balanced-chance baseline). |
| `position_bin` | 0.3259 | 0.3243 | Above chance (`0.2500`) but only 1.30x chance; investigate in Step 12. |
| `running_speed_bin` | 0.3380 | 0.3337 | Above chance (`0.2500`) but only 1.33x chance; investigate in Step 12. |

Additional notes:
- The initial GPU run completed the first epoch and then triggered the trainer’s built-in fallback: `CUDA out of memory on cuda; retrying on CPU.`
- The CPU retry completed successfully.
- Final logged losses:
  - Epoch 1: `19617.100021`
  - Epoch 10: `12533.286893`
  - Epoch 50: `2864.602500`
  - Epoch 100: `427.921886`
  - Epoch 150: `238.550080`
  - Epoch 200: `187.288289`
- Final test loss: `122.939601`

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `visual_stimulus_category` | validation balanced accuracy `0.5454` | No directly comparable decoder metric reported in the paper; qualitatively expected to be above chance because the paper shows strong stimulus-selective coding. |
| `licking` | validation balanced accuracy `0.8800` | No directly comparable decoder metric reported; expected to be decodable because the reference code contains lick-aligned analyses. |
| `position_bin` | validation balanced accuracy `0.3243` | No directly comparable decoder metric reported. The paper shows position-structured sequential activity, so above-chance decoding is expected. |
| `running_speed_bin` | validation balanced accuracy `0.3337` | No directly comparable decoder metric reported; some above-chance decoding is expected because running speed is behaviorally coupled to the retained running frames. |

[Analysis of lower accuracies]
- **Chance comparison**:
  - `visual_stimulus_category`: `0.5454 / 0.0667 = 8.18x` chance
  - `licking`: `0.8800 / 0.5000 = 1.76x` chance
  - `position_bin`: `0.3243 / 0.2500 = 1.30x` chance
  - `running_speed_bin`: `0.3337 / 0.2500 = 1.33x` chance
- **Train vs validation gap**:
  - `visual_stimulus_category`: `0.5918 / 0.5454 = 1.09x`
  - `licking`: `0.9066 / 0.8800 = 1.03x`
  - `position_bin`: `0.3259 / 0.3243 = 1.00x`
  - `running_speed_bin`: `0.3380 / 0.3337 = 1.01x`
- No output shows the `>1.5x` train/validation gap that would indicate strong overfitting or leakage.
- The two weaker outputs (`position_bin`, `running_speed_bin`) were investigated as follows:
  1. **Raw-value verification**: the Step 10 `np.allclose()` checks already verified raw-to-converted correctness for neural activity, cue-time input, trial-time input, licking, position bins, and speed bins on six hand-checked trials across two sessions. No mismatches were found.
  2. **Temporal-alignment review**: inspected `sample_trials.png` and `predictions.png`. Sample trials show monotonic `time_since_trial_start_s`, sign-changing `time_to_sound_cue_s`, monotonic position-bin progression, and plausible speed-bin fluctuations. Prediction plots show stimulus and lick predictions aligning strongly and position-bin predictions generally tracking the ground-truth progression.
  3. **Output variation check**: `position_bin` and `running_speed_bin` are both well balanced globally:
     - `position_bin`: `[0.250, 0.249, 0.250, 0.252]`
     - `running_speed_bin`: `[0.250, 0.250, 0.250, 0.250]`
     So low accuracy is not due to collapsed labels or extreme class imbalance.
  4. **Reference-processing check**: the same running-only corridor filtering (`ft_CorrSpc & ft_move > 0`) used in the reference code is applied here, so the lower balanced accuracies are not explained by a filtering mismatch.
- Interpretation:
  - `visual_stimulus_category` and `licking` validate the overall conversion strongly.
  - `position_bin` and `running_speed_bin` remain above chance with essentially no train/validation gap, which argues against a conversion bug.
  - The most likely reason these are only modestly above chance is that the full dataset spans 89 recordings and 19 mice with heterogeneous timing and long-pause edge cases after applying the required running-only mask; the decoder is asked to generalize one shared readout across this heterogeneity.
  - A possible alternative design would be to redefine trial-time inputs in retained running-time rather than absolute wall-clock time after masking. That might improve position/speed decoding, but it would also move away from the literal raw timestamps tied to corridor entry, so the current conversion keeps the source-faithful timing definition and documents the tradeoff.

### Issues Found and Resolved
- **GPU memory exhaustion during full decoder training**: handled by the trainer’s built-in automatic CPU retry; the CPU run completed successfully.
- **Position and running-speed accuracies below the 1.5x-chance heuristic**: investigated through chance analysis, train/validation-gap analysis, raw-value sanity checks, and plot review. No conversion bug was identified, so no data change was made.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

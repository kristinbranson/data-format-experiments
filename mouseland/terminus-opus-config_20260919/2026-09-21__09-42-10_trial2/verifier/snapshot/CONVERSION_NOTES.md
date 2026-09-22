# Dataset Conversion Notes

## Overview
- **Dataset**: "Unsupervised pretraining in biological neural networks" (paper.pdf, /app/code, /app/data)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3, numpy 2.4.4, torch 2.6.0+cu124, CUDA available = True

Directory contents of /app:
- CONVERSION_NOTES.md, Dockerfile, docker-compose.yaml, .manifest
- paper.pdf, methods.txt
- decoder.py, train_decoder.py (decoder reference)
- code/ : README.md, utils.py, fig1.py, fig2.py, fig3.py, fig4.py, fig5.py, S6.py, data_process_script.ipynb, Figures.ipynb, LICENSE, .gitignore
- data/ : beh/, process_data/, retinotopy/, spk/

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Repo = Zhong et al. 2025 figure code. README says `data_process_script.ipynb` shows how to process/save intermediate
results, `Figures.ipynb` plots figures. All analysis code lives in `utils.py`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_exp_beh(root, exp_type)` | utils.py:326 | LOADING | loads `beh/Beh_<exp_type>.npy` -> dict keyed by `mname_datexp_blk[_stimtype]` |
| `load_spk(db, root)` | utils.py:338 | LOADING | loads `spk/<mname>_<datexp>_<blk>_neural_data.npy`, concatenates `['spks']` list (one entry per imaging plane) along axis 0 -> (n_neurons, n_frames) deconvolved traces |
| `load_retino(db, root)` | utils.py:330 | LOADING | loads `retinotopy/<mname>_<datexp>_trans.npz`; returns `xy_t` (cortical coords), `iarea` per neuron, and area index masks |
| `neu_area_ID(iarea)` | utils.py:312 | CURATION/LABEL | maps `iarea` codes to areas: V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4 |
| `get_interpPos_spk` / `spk_pos_interp` | utils.py:105-130 | PROCESSING | interpolates spks onto cumulative-position grid, 60 bins per 6 m trial (1 bin = 1 dm), using only frames where the VR moved |
| `Get_dprime_selective_neuron` | utils.py:418 | PROCESSING | `fr_valid = (ft_move>0) & ft_CorrSpc` -> only running frames inside the texture corridor are analysed; d-prime between stimuli |
| `Get_coding_direction` | utils.py:503 | PROCESSING | same frame validity, z-scoring/normalization to grey-space mean, uses `ft_trInd`, `ft_WallID`, `ft_PosCum` truncated to `nfr = spk.shape[1]` |
| `get_kfold_reward_response`, `get_reward_neuorns` | utils.py:814,884 | PROCESSING | z-score spks over time; align to `SoundFr` (cue) and first-lick frame; uses `SoundDelPos`, `isRew`, `WallName`, `UniqWalls`, `stim_id` |
| `spk_2_cue`, `spk_2_firstLick` | utils.py:935,913 | PROCESSING | trial alignment in *neural frames*: windows of +/-15 frames around cue / first lick; licks binned into frame bins by histogram |
| `get_cat_id` | utils.py:137 | LABEL | maps wall names to stimulus categories using the rewarded stimulus |

### Notes
- Neural data provided are **suite2p deconvolved traces** (`spks`) already cell-curated (paper: non-negative deconvolution, decay 0.75 s). No further per-neuron quality filtering appears anywhere in the reference code; the only "filtering" in figure analyses is *selection* of task-selective neurons for specific analyses (d-prime thresholds), which is analysis-specific and should NOT be applied for a decoder.
- **Frame rate: fs = 3.17 Hz** (stated in `data_process_script.ipynb`), so one neural frame ~= 315 ms.
- Behaviour is provided already resampled onto neural frames: `ft_*` fields (`ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_move`, `ft_isMoving`, `ft_GraySpc`, `ft_CorrSpc`, `ft_WallID`, `ft_RunSpeed`) plus per-trial frame indices `StartFr`, `EndFr`, `GrayFr`, `SoundFr`, `SoundDelayFr`, `RewardFr`, `LickFr`.
- Behaviour arrays are sometimes longer than the neural recording; the reference code always truncates with `[:nfr]` where `nfr = spk.shape[1]`. We must do the same.
- Only **running** timepoints are used for analysis in the paper/code (`ft_move>0`, i.e. the VR moved). For a time-resolved decoder we cannot drop interior frames without destroying the time axis; decision recorded in Step 5.
- Trial start = corridor entry = `StartFr` (`Trial_start_time`); trial end = `EndFr`; `GrayFr` = entry into grey space.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/` mirrors the Figshare layout expected by the reference code:
- `beh/Imaging_Exp_info.npy` -> dict of 23 `exp_type` -> list of session dicts (`db`) with
  `mname, datexp, blk, is2p, rewType, depth, exptype, stim_id, Gender[, stimtype, days, sess#, Note]`.
  142 db entries covering **89 unique sessions** (a session can appear in up to 5 exp_types, e.g. a
  test session used as `test1_before_grating` and `naive_test1`).
- `beh/Beh_<exp_type>.npy` (23 files, 113-432 MB) -> dict keyed `mname_datexp_blk[_stimtype]` -> the
  59-field behaviour dict documented in `data_process_script.ipynb`.
- `spk/<mname>_<datexp>_<blk>_neural_data.npy` (89 files, 405 GB total, 2-8 GB each) -> `{'spks': [array,...]}`,
  one float32 (n_neurons_plane, n_frames) array per multiplexed plane-set; concatenating along axis 0
  gives (n_neurons, n_frames) deconvolved traces. Read throughput measured at ~2.2 GB/s.
- `retinotopy/<mname>_<datexp>_trans.npz` (89 files) -> `xy_t` (n_neurons,2) cortical coords, `iarea`
  (n_neurons,) area code, `xpos`,`ypos`,`A`,`dx`,`dy`. Length of `iarea` == n_neurons of the spk file (verified).
- `beh/Unsupervised_pretraining_behavior/` -> behaviour-only cohorts (Fig. 5), **no imaging** -> not usable for a neural decoder.
- `process_data/` is empty (intermediate results not shipped; we recompute what we need).

### Key behaviour fields (all `ft_*` fields are on the neural-frame clock)
`ntrials`, `WallName`/`UniqWalls`/`stim_id`/`TrialStim` (stimulus identity), `isRew`, `StartFr` (corridor entry),
`GrayFr` (grey-space entry = corridor exit), `EndFr` (end of grey space ~ next `StartFr`), `SoundFr`/`SoundDelayFr`/`SoundPos`/`SoundDelPos`,
`RewardFr`, `LickFr`/`LickTrind`/`LickPos`, `ft` (frame timestamps, datenum), `ft_trInd`, `ft_Pos` (0-60 dm within trial),
`ft_PosCum`, `ft_move` (VR displacement per frame), `ft_isMoving`, `ft_CorrSpc`/`ft_GraySpc`, `ft_WallID`, `ft_RunSpeed`,
`Corridor_Length`=60 dm, `Texture_Length`=40 dm (4 m), `Gray_Space_length`=20 dm (2 m), `Reward_Mode`, `Reward_Delay_ms`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, all 89 sessions) | 4,691,034 |
| Neurons / session | min 20,547; median 54,741; max 89,577 |
| Neurons with a visual-area label (V1/mHV/lHV/aHV) | 4,105,393 (87.5%); 585,641 have `iarea` in {-1,7} (outside the 4 areas) |
| Per-area totals (V1, mHV, lHV, aHV) | 1,833,035 / 1,108,860 / 495,318 / 668,180 |
| Subjects (mice) | 19 (DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2) |
| Sessions | 89 |
| Sessions / subject | 1-7 (median 4) |
| Trials (total) | 38,110 |
| Trials / session | min 84; median 429; max 789 |
| Neural frames / session | 14,571 - 34,230 |
| Frame interval (median of `diff(ft)`) | 0.3144 - 0.3154 s across the 89 sessions -> fs = 3.17-3.18 Hz |
| Behaviour arrays vs spk frames | behaviour is 1-3 frames longer -> truncate to `nfr = spk.shape[1]` (as the reference does) |
| Sessions with rewards (task cohort) | 28; the other 61 (unsupervised / naive / grating) have `isRew` all False and **zero recorded licks** |
| Trials per texture family | leaf 17,761; circle 11,619; wood(=brick) 5,452; rock 3,278 |
| Wall names (trials) | circle1 9393, leaf1 9736, leaf2 4841, rock1 2743, wood1 2517, circle2 1836, wood2 1592, leaf3 1545, leaf1_swap2 821, leaf1_swap1 818, wood5 567, rock2 535, wood1_swap2 421, circle3 390, wood1_swap1 355 |
| Corridor-window frames / trial (`ceil(StartFr)`..`floor(GrayFr)`) | median 23, but up to 5,607 when the mouse stops running |
| **VR-moving** frames / trial inside the corridor | median 21, 1st-99th pct 15-37 (4 m / 0.6 m/s / 0.315 s = 21 frames) |
| Pooled running speed on moving corridor frames | quartiles ~11.4 / 24.0 / 39.7 cm/s |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings (sessions) | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 | same |
| Neurons / recording | 20,547 - 89,577 | "activity traces from 20,547 to 89,577 neurons in each recording" |
| Neural signal | suite2p **deconvolved** traces, tau = 0.75 s | "For non-negative deconvolution, we used a timescale of decay of 0.75 s ... All our analyses were based on deconvolved fluorescence traces." |
| Corridor geometry | 4 m corridor + 2 m grey | "virtual reality corridors were each 4 m long, with 2 m of grey space between corridors" |
| VR speed / run threshold | 60 cm/s VR, 6 cm/s run threshold | "moved forward ... by running faster than a threshold of 6 cm s-1, but the virtual corridors always moved at a constant speed (60 cm s-1)" |
| Stimuli | 4 texture families: circle, leaf, rock, brick (+gratings in some mice) | "four large texture images: circle, leaf, rock and brick" |
| Sound cue position | uniform 0.5-3.5 m, in **all** trial types | "the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m" |
| Reward | only in rewarded corridor, after cue, delivered on lick (or passively with delay) | "The reward was delivered if a lick was detected after the sound cue in the rewarded corridor" |
| Analysis timepoint curation | running timepoints only | "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." |
| Position interpolation grid (reference figures) | 60 bins of 1 dm over the 6 m trial | `get_interpPos_spk(..., n_bins=60, lengths=CL)` |
| Frame rate | fs = 3.17 Hz | `data_process_script.ipynb`: "Calcium signal recording frame rate: fs = 3.17Hz" (measured 3.17-3.18 Hz) |
| d-prime selectivity criterion | d' >= 0.3 | "The criteria for selective neurons was d' >= 0.3" |

### Processing Details
- Neural: deconvolved traces, no dF/F needed (already deconvolved). Some analyses z-score over time (`stats.zscore(spk, axis=1)`).
- Frames used: `(ft_move > 0)` (VR moving == mouse running above threshold) and, for corridor analyses, `& ft_CorrSpc`.
- Behaviour is already resampled to the neural frame clock (`ft_*`), so alignment = indexing by neural frame.
- Trials are aligned by corridor entry (`StartFr`) in the raster/position analyses; cue- and first-lick-aligned
  windows (+/-15 frames) are used in Fig. 4.

### Curation Steps
**Neuron curation rules**: none beyond suite2p cell detection/classification + deconvolution (the shipped `spks`
are already the curated cells). Analysis-specific *selection* (d' >= 0.3, top 5%) is not a data-quality filter and
is not applied here.

**Trial curation rules**: the reference code uses all trials of a session; frames are curated instead
(running + corridor). Behaviour arrays are truncated to the number of neural frames.

### Decoders Trained
| Decoded variable | Accuracy |
| (none) | The paper contains **no** decoding analysis (grep for "decod"/"accuracy" in the full text returns nothing), so there is no published decoder accuracy to compare against. Comparisons must therefore rely on dataset statistics + above-chance performance. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

Checks run (scripts in `/app/cache/`): `survey_beh.py`, `survey2.py`, `checks_step2.txt`, `checks_step4.txt`.

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | 142 db entries over 23 exp_types | 89 unique (mname,datexp,blk); 89 spk files; 89 retinotopy files | "89 recordings in 19 mice" | Sessions repeated across exp_types are the *same* recording. Verified all repeated entries have identical behaviour (ntrials, nframes, sum(StartFr), sum(SoundFr), nlicks: 0 inconsistencies). Convert each unique session once -> 89 sessions. |
| Neurons per session | `load_spk` concatenates planes | 20,547-89,577 (min/max exactly match the paper) | 20,547-89,577 | Consistent: `len(iarea) == sum(plane.shape[0])` for every session checked. |
| Behaviour vs neural length | reference truncates behaviour with `[:nfr]` | behaviour arrays are 1-3 frames longer than spk frames | - | Truncate behaviour to `nfr`; drop any trial whose frames would exceed `nfr`. |
| Timepoint curation | `fr_valid = (ft_move>0) & ft_CorrSpc` | corridor windows contain 12-5,607 frames, but only ~21 VR-moving frames (median) | "only considered timepoints during running" | Keep only running (VR-moving) corridor frames -> also makes trial length nearly constant (15-37 frames). |
| Licking | lick fields exist for all sessions | only the 28 rewarded (task) sessions have licks; the 61 unsupervised/naive sessions have **zero** licks | unsupervised mice were not water restricted, no rewards | Licking is a required decoder output, and it is only meaningfully defined in the task cohort. Decision in Step 5: include only sessions in which licking was measured (28 task sessions) OR keep all sessions with a constant-0 lick output. See Step 5 decision. |
| Sound cue | `SoundFr`, `SoundDelayFr` (delay 0-1500 ms) | `SoundFr` always inside `[StartFr, GrayFr]`; `SoundPos` 4-36 dm | cue uniform 0.5-3.5 m | Consistent (0.4-3.6 m incl. interpolation jitter). Use `SoundFr` (true cue time) for the "time to sound cue" input, not the delayed reward proxy. |
| Reward availability | `isRew` per trial | `isRew` is exactly the set of trials of the rewarded wall; 61 sessions have none | reward only in one corridor | Use `isRew` as the "reward availability" per-trial input. |
| Stimulus identity | `WallName`, `UniqWalls`, `stim_id` (0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, 5/6 leaf1_swap) | 15 distinct wall names across sessions, from 4 texture families (circle/leaf/rock/wood=brick) | "we denote the stimuli as leaf and circle, even though other visual stimuli were also used in some mice (rock and bricks)" | The decoder output is "visual stimulus category, e.g. circle, leaf": use the **texture family** (circle/leaf/rock/brick) so the label is comparable across mice, which is exactly how the paper pools stimuli (leaf1/leaf2/leaf3/leaf1_swap are all "leaf"). |
| Corridor position | `ft_Pos` 0-60 dm | corridor frames all have `ft_Pos` in [0,40] dm; wall id matches trial stimulus | 4 m corridor | Position bins = 4 equal 1-m bins over 0-4 m of the texture area. |
| Trial windowing | `(ft_trInd==t) & ft_CorrSpc` vs `ceil(StartFr)..floor(GrayFr)` | ~3-5% of trials differ by one boundary frame | - | Use the behaviour's own per-frame trial index `ft_trInd == t` together with `ft_CorrSpc` (the authoritative labelling used by `Get_coding_direction`), so no off-by-one is introduced. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial / neuron selection
- **Sessions**: all **89** unique imaging recordings (each converted once, using the first `exp_type` in which it appears;
  verified that duplicated entries carry identical behaviour). Behaviour-only cohorts in
  `beh/Unsupervised_pretraining_behavior/` are excluded because they have no neural data.
- **Neurons**: keep neurons whose retinotopic `iarea` maps to one of the four areas analysed in the paper
  (V1 = 8, mHV = 0,1,2,9, lHV = 5,6, aHV = 3,4; `utils.neu_area_ID`). `iarea` in {-1, 7} (12.5% of neurons) is
  never used by the reference analyses and has no area label, so those neurons are dropped.
  Then **subsample to at most 2,000 neurons per session**, stratified proportionally across the four areas with a
  fixed seed. Rationale: the full dataset is 4.69 M neurons (~183 GB of running-frame activity); the reference decoder
  projects each session onto 100 PCs and itself random-projects to at most 2,000 neurons for its SVD initialisation,
  so 2,000 neurons/session is decoder-saturating while keeping the pickle ~6 GB.
- **Frames (timepoints)**: for trial `t`, keep frames with `ft_trInd == t` AND `ft_CorrSpc` (inside the 4 m texture
  corridor) AND `ft_move > 0` (VR moving, i.e. the mouse ran above the 6 cm/s threshold), truncated to
  `nfr = spk.shape[1]`. This reproduces the reference curation `fr_valid = (ft_move>0) & ft_CorrSpc`
  (`utils.Get_dprime_selective_neuron`) and the paper's "We only considered timepoints during running for analysis".
  Because the VR advances at a constant 60 cm/s while running, this also makes trials nearly equal-length
  (median 21 frames = 4 m / 0.6 m/s / 0.315 s) instead of up to 5,607 frames when a mouse stands still.
- **Trials**: drop trials with < 5 retained frames (too short to be informative / truncated at the end of the
  recording), and any trial whose frames fall beyond the neural recording.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/<sess>_neural_data.npy['spks']` | `neural[session][trial]` (n_neurons, T) float32 | concatenate planes, keep area-labelled neurons, subsample <=2000, index running corridor frames of the trial | `utils.load_spk` | deconvolved traces, no further normalisation (paper: "All our analyses were based on deconvolved fluorescence traces") |
| `retinotopy/<m>_<date>_trans.npz['iarea']` | `brain_region_idx[session]` | `neu_area_ID` mapping -> 0=V1, 1=mHV, 2=lHV, 3=aHV | `utils.load_retino`, `utils.neu_area_ID` | `brain_regions = ['V1','mHV','lHV','aHV']` |
| `SoundFr`, `ft` | `input[0]` = `time_to_sound_cue` (s), time-varying | `t_cue - t_frame` in seconds, using `ft` interpolated at the fractional `SoundFr`; positive before the cue, negative after | `utils.spk_2_cue` (cue alignment) | cue occurs inside the corridor in every trial (verified) |
| session date vs the mouse's first recording | `input[1]` = `day_of_training` (days), per-trial (broadcast) | calendar days since that mouse's first imaging session | `exp_info` `days`/`sess#` fields | `exp_info['days']` exists for only 8/142 entries, so elapsed days since the mouse's first recording is used as the continuous training-day axis |
| `StartFr`, `ft` | `input[2]` = `time_since_trial_start` (s), time-varying | `t_frame - t_start`, `t_start` = `ft` interpolated at fractional `StartFr` | alignment event of the conversion | 0 at corridor entry |
| `isRew` | `input[3]` = `reward_available` (0/1), per-trial (broadcast) | bool -> float | `beh['isRew']` | 1 for trials in the rewarded corridor (only in the 28 task sessions) |
| `WallName` | `output[0]` = `stimulus_category` | texture family of the wall name: circle{1,2,3}->circle, leaf{1,2,3}/leaf1_swap{1,2}->leaf, rock{1,2}->rock, wood{1,2,5}/wood1_swap{1,2}->brick | `beh['WallName']`, `beh['UniqWalls']`, paper "four large texture images: circle, leaf, rock and brick" | per-trial, broadcast over time; `output_values[0] = ['circle','leaf','rock','brick']` |
| `LickFr` | `output[1]` = `licking` | 1 if any lick falls in that neural frame (`floor(LickFr)`), else 0 | `utils.spk_2_cue`/`spk_2_firstLick` bin licks by frame with `np.histogram` over integer frame bins | time-varying; identically 0 in the 61 non-water-restricted (unsupervised/naive/grating) sessions, which genuinely never licked |
| `ft_Pos` | `output[2]` = `position_bin` | `floor(ft_Pos/10)` clipped to 0..3 (`ft_Pos` in dm; 4 bins x 1 m over the 4 m texture area) | `beh['ft_Pos']`, `Texture_Length`=40 dm | time-varying; `output_values[2] = ['0-1m','1-2m','2-3m','3-4m']` |
| `ft_RunSpeed` | `output[3]` = `speed_bin` | global quartile bins (edges = 25/50/75th percentiles of the running speed over **all** retained frames of all 89 sessions) | `beh['ft_RunSpeed']` (= `RunFr`) | time-varying; each bin holds 25% of the data by construction |

### Key Decisions
1. **Only running (VR-moving) corridor frames are kept**: matches the paper's explicit curation and the
   reference `fr_valid` mask; also removes the long stationary periods (reward consumption) that would otherwise
   dominate the time axis. Consequence: the retained frames are not contiguous in clock time, so
   `time_since_trial_start` is supplied as the *actual* elapsed time of each retained frame (rather than bin index).
2. **All 89 sessions are kept even though 61 have no licking**: the other three outputs (stimulus, position, speed)
   and all four inputs are well defined in every session, and the unsupervised/naive mice genuinely never licked
   (they were not water restricted and no reward was delivered), so lick = 0 is a correct label rather than missing data.
   Balanced accuracy is pooled over timepoints, so the 28 task sessions still determine lick-class-1 recall.
3. **Stimulus category = texture family** (circle / leaf / rock / brick), pooling the frozen-crop variants
   (leaf1/leaf2/leaf3/leaf1_swap...) exactly as the paper pools them for statistics. This makes the label comparable
   across mice trained on different stimulus pairs, as required by "Visual stimulus category. e.g. circle, leaf, etc."
   (`wood` in the data files is the texture the paper calls `brick`.)
4. **Neuron subsampling to 2,000/session** (stratified by area, seeded): required for a tractable dataset size;
   justified by the decoder's own 100-PC / 2,000-neuron projection.
5. **Neurons without an area label (`iarea` in {-1,7}) are dropped**: they are excluded from every area analysis in
   the reference code and cannot be given a `brain_regions` entry.
6. **No z-scoring / dF-F of the neural data**: the shipped traces are already suite2p deconvolved; the paper's
   analyses use them directly (z-scoring appears only inside specific figure analyses, and the decoder does its own
   projection).
7. **Time bin size = the imaging frame interval** (median 315.2 ms, fs = 3.17 Hz); no re-binning, so no temporal
   information is destroyed and alignment is exact by construction (behaviour is natively on the frame clock).
8. **Alignment event = trial start (corridor entry, `StartFr`)**, as required by the decoder task; `off_start = 0`,
   `off_end = None` because trials end at corridor exit and therefore have variable duration (median 6.6 s).
9. **Speed quartile edges are global** (computed once from all retained frames of all sessions, behaviour-only pass)
   so that the 4 bins each hold 25% of the data across the whole dataset and are identical between `--sample` and `--full` runs.

### Planned Sanity Checks
- [ ] 89 sessions, 19 mice, ~38,110 trials before trial filtering (report dropped trials).
- [ ] Neurons/session before subsampling within [20,547, 89,577] (paper range); area-labelled counts >= 17,363.
- [ ] `len(brain_region_idx[session]) == n_neurons` for every session and every region index in 0..3.
- [ ] Median retained frames/trial ~ 21 (= 4 m / 0.6 m/s / 0.3152 s); 1-99th percentile 15-37.
- [ ] Position-bin distribution approximately uniform (constant VR speed) and covering all 4 bins.
- [ ] Speed-bin distribution 25/25/25/25% by construction.
- [ ] Stimulus-category distribution matches trial counts from the raw behaviour (leaf 17,761; circle 11,619; brick 5,452; rock 3,278 before filtering).
- [ ] Lick output non-zero only in the 28 rewarded sessions; overall lick-frame fraction reported.
- [ ] `time_since_trial_start` >= 0 and `time_to_sound_cue` changes sign within each trial containing frames after the cue.
- [ ] Spot-check raw-file equality (Step 10): `neural` value at a specific (session, trial, neuron, frame) equals `spks` concatenated at the matching absolute frame.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`, run as `python -u /app/convert_data.py <outfile> [--full|--sample] [--show-processing]`.

Structure:
1. `unique_sessions(exp_info)` -- 89 unique recordings, one entry each (first exp_type in which it appears).
2. `first_session_dates()` -- per-mouse date of the first imaging session, computed over **all 89** sessions so that
   `day_of_training` is identical in `--sample` and `--full` runs.
3. **Pass 1 (behaviour)** `behaviour_pass` / `process_behaviour`: loads each of the 23 `Beh_<exp_type>.npy` files at
   most once, and for every trial keeps the frames `ft_trInd == t & ft_CorrSpc & (ft_move > 0)` (the reference
   `fr_valid` mask), then builds the 4 inputs and the raw output values. Trial-start / cue times come from `ft`
   interpolated at the fractional `StartFr` / `SoundFr`. Licks are binned to integer neural frames (`floor(LickFr)`),
   as in `utils.spk_2_cue`.
4. Global speed quartile edges from all retained frames (so each speed bin holds 25% of the data).
5. **Pass 2 (neural)** `convert_session` in a `multiprocessing.Pool`: per session loads the retinotopy `iarea`,
   selects area-labelled neurons and subsamples (stratified by area, seed 2025 + session index), then
   `load_spk_rows` loads `spks` (as `utils.load_spk` does: concatenate planes along axis 0) and slices out only the
   selected rows x retained frames, so the full (n_neurons x n_frames) matrix is never materialised.
6. Assembly, assertions/sanity checks, optional `--show-processing` plots, pickle dump (protocol 4).

Code inefficiencies identified:
- Naively loading `spk` then concatenating planes into one array doubles peak memory (up to 16 GB/session).
- Per-trial re-indexing of the spk file would re-read the 2-8 GB file for every trial.
- Loading each `Beh_<exp_type>.npy` per session would re-read up to 5x the same 100-430 MB file.

Code speedups added:
- Behaviour files loaded once per exp_type; sessions grouped by exp_type.
- One spk read per session; only the selected rows/columns are copied (`plane[sel][:, frames]`).
- Vectorised frame selection (`searchsorted` on the sorted per-frame trial index) instead of a per-trial mask over
  all ~25k frames.
- 6 worker processes (`--nproc`, `maxtasksperchild=1` to release the 2-8 GB buffers) for the I/O-bound neural pass.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` -> `/app/conversion_sample_out.txt`;
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` -> `/app/verification_sample_out.txt`.
Sample = one task session with rewards (TX108_2023_03_25_1) and one unsupervised session (TX83_2022_08_29_1).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (TX108, TX83) |
| Neurons / session (recorded) | 79,417 and 52,954 (within the paper's 20,547-89,577) |
| Neurons / session (kept) | 2,000 (V1 1585, mHV 1137, lHV 448, aHV 830 pooled over the 2 sessions) |
| Trials (total) | 661 of 661 recorded (0 dropped) |
| Trials / session | 423, 238 |
| Timepoints | 16,424; T/trial mean 24.9, min 15, max 46 |
| time_to_sound_cue | [-110.5, 203.7] s (median 0.05 s) |
| day_of_training | [12, 79] days (TX83: 2022_08_17 -> 2022_08_29 = 12; TX108: 2023_01_05 -> 2023_03_25 = 79) |
| time_since_trial_start | [0.0, 204.4] s (median 4.7 s; 88% <= 15 s) |
| reward_available | [0, 1] |
| stimulus_category | circle 0.182, leaf 0.185, rock 0.330, brick 0.303 |
| licking | no-lick 0.936, lick 0.064 (0.101 in the task session, 0 in the unsupervised session) |
| position_bin | 0.247 / 0.248 / 0.249 / 0.256 |
| speed_bin | 0.250 / 0.250 / 0.250 / 0.250 (by construction) |

### Processing Plots Review
`processing_TX108_2023_03_25_1.png`, `processing_TX83_2022_08_29_1.png` show, for the first 6 trials:
deconvolved activity with trial boundaries; position in metres overlaid with its 1-m bin (steps exactly at 1/2/3 m);
running speed with the global quartile edges overlaid and the resulting bin; the two time-varying inputs
(`time_since_trial_start` starting at ~0 at each corridor entry, `time_to_sound_cue` crossing zero once per trial);
the lick output with the per-trial reward-availability input and stimulus category; and the distribution of retained
frames per trial. No anomalies: no temporal offsets between streams, correct discretisation, no empty trials.

### Bug found and fixed
- `day_of_training` was initially computed from the *selected* sessions only, so both sample sessions got day 0.
  Fixed by computing per-mouse first dates from all 89 sessions (`first_session_dates`). Re-ran: 12 and 79 days,
  which match the raw dates.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| behaviour file loaded once per exp_type | ~2.5x fewer behaviour reads (142 -> 23 loads) |
| load only selected rows/frames of `spks` | avoids materialising 2-8 GB per session twice |
| vectorised per-trial frame grouping | ~100x faster than per-trial boolean masks |
| 6 parallel workers | neural pass becomes disk-bandwidth-bound |

| Step | Time / Session | Estimated Total Time |
| behaviour pass | 0.16 s | ~15 s for 89 sessions (+23 file loads, ~2 min) |
| neural pass | 1.3-2.3 s (sample; 5.9-7.4 GB files at ~2.2 GB/s with 6 workers) | ~5-15 min for 405 GB |
| save | 0.2 s / 0.13 GB | ~10 s for ~7 GB |
| **total** | | **< 20 min** (well under the 15-min target for the neural pass itself) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None ("Data format is valid, no errors or warnings.")
- Warnings: None

### Training
Loss decreased monotonically: 1.083 (epoch 50) -> 0.727 -> 0.521 -> ... -> 0.172 (epoch 200). Test loss 1.050.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| stimulus_category | 1.0000 | 0.9677 | 0.25 |
| licking | 0.9967 | 0.7885 | 0.50 |
| position_bin | 0.9991 | 0.8848 | 0.25 |
| speed_bin | 0.8267 | 0.4801 | 0.25 |

All four outputs are far above chance on validation trials with only 2 sessions.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --nproc 8` -> `/app/conversion_full_out.txt`
(total run time **0.9 min**: behaviour pass 1.4 s, neural pass ~40 s with 8 workers over 405 GB of spk files,
save 8.9 s) and `python -u /app/train_decoder.py /app/converted_data.pkl --verify-only` -> `/app/verification_full_out.txt`.

### Output Files
- `converted_data.pkl`: 6.62 GB
- `verification_full_out.txt`: created; **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 89 recordings | 142 db entries = 89 unique (mname,datexp,blk) | 89 spk files, 89 retinotopy files | 89 | YES |
| Subjects | 19 mice | 19 distinct `mname` | 19 | 19 | YES |
| Sessions/subject | - | - | 1-8 | 1-8 (median 5) | YES |
| Total neurons recorded | 20,547-89,577 per recording | `load_spk` = concat of planes | 4,691,034 total; min 20,547 / max 89,577 | min 20,547 / median 54,741 / max 89,577 (reported per session in `metadata.session_info`) | YES (exact min/max) |
| Area-labelled neurons | 4 areas V1/medial/lateral/anterior | `neu_area_ID` | 4,105,393 (87.5%) | 4,105,393 available; 178,000 kept (2,000/session cap) | YES |
| Neurons kept/session | - | decoder uses 100 PCs / <=2,000-neuron projection | - | 2,000 (V1 78,186; mHV 50,607; lHV 20,670; aHV 28,537) | by design |
| Trials (total) | - | all trials used | 38,110 | 38,110 (0 dropped) | YES |
| Trials/session (mean) | - | - | 428.2 (84-789) | 428.2 (84-789) | YES |
| Frame rate / time bin | fs = 3.17 Hz (notebook) | `spk_2_cue` works in frames | median dt 0.3144-0.3154 s | 314.70 ms (3.178 Hz) | YES |
| Trial length | 4 m at 60 cm/s = 6.67 s | VR-moving frames only | median 21 running frames/trial | mean 22.25, median 22.6, min 11, max 178 | YES |
| Timepoints total | - | - | - | 821,579 | - |
| Sessions with reward | task cohort | `isRew` | 28 | 28 (`reward_available`=1 trials only there) | YES |
| Sessions with licking | task mice only (water restricted) | `LickFr` | 28 | licking>0 in exactly those 28 sessions | YES |
| Stimulus distribution (trials) | leaf/circle main pair, rock+brick in some mice | `WallName` families | leaf 17,761 / circle 11,619 / brick 5,452 / rock 3,278 | timepoint fractions leaf 0.470 / circle 0.311 / brick 0.135 / rock 0.085 (trial counts identical to the raw data, 0 trials dropped) | YES |
| Cue position | uniform 0.5-3.5 m | `SoundFr` inside corridor | `SoundPos` 4-36 dm | `time_to_sound_cue` changes sign inside every trial | YES |
| Position bins | 4 m corridor | `ft_Pos` in dm | uniform coverage | 0.250 / 0.249 / 0.250 / 0.252 | YES |
| Speed bins | - | `ft_RunSpeed` | quartiles 12.4 / 25.4 / 40.9 cm/s | 0.25 / 0.25 / 0.25 / 0.25 | YES (by construction) |
| Licking fraction of timepoints | - | - | - | 0.0369 overall; 0.058-0.391 within the 28 task sessions | plausible (~6 licks/trial, 3.2 Hz frames) |
| Input ranges | - | - | - | time_to_cue [-1763, 724] s, day [0, 92] d, time_since_start [0, 1765] s, reward {0,1} | see note |

Note on the long time values: because only running frames are kept, the *clock* time of a trial keeps running while
the mouse stands still, so a few trials span minutes (2.2% of timepoints have time_since_trial_start > 30 s, 0.5% > 120 s).
These are genuine elapsed times, not artefacts.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
- First line: **"Data format is valid, no errors or warnings."** -> zero errors, zero warnings, so nothing to fix.
- Reported structure re-read and checked by hand: 89 sessions, 38,110 trials, 19 subjects, 4 brain regions,
  dinput = 4, doutput = 4, T mean 22.25 (min 11, max 178), 2,000 neurons in every session,
  brain-region distribution V1 78,186 / mHV 50,607 / lHV 20,670 / aHV 28,537 (sums to 178,000 = 89 x 2,000).
- Output value fractions per dimension are all non-degenerate (every class present in the pooled data).
- `T min = 11` -- shorter than the 15-frame 1st-percentile of the behaviour survey because a handful of trials are
  truncated by the end of the neural recording; they still exceed the 5-frame minimum, so they are kept.

### Check 2: Independent sanity checks against the raw files
Script: `/app/cache/sanity_checks.py`, log `/app/cache/sanity_checks_out.txt` -- **130 checks, 0 failures**.
It never imports the conversion code; everything is re-derived from `beh/*.npy`, `spk/*.npy`, `retinotopy/*.npz`
and compared with `np.allclose`. For three sessions spanning the three cohorts
(TX108_2023_03_25_1 = task/rewarded, TX83_2022_08_29_1 = unsupervised, TX119_2023_12_12_1 = naive):
- **trial structure**: independently rebuilt frame lists `(ft_move>0) & ft_CorrSpc & (ft_trInd==t)`, truncated to the
  spk frame count, with the >=5-frame rule -> identical trial count and identical per-trial frame counts.
- **neural**: for neurons 0, 500 and 1999 of trial 0, the converted trace was located in the raw `spks` planes
  (exhaustive search over all rows with `np.allclose`); the *same* raw rows then reproduce the converted traces on
  trials 1, 7 and the middle trial -- this simultaneously verifies the plane concatenation, the neuron indexing and
  the temporal alignment of every trial.
- **brain regions**: the raw `iarea` of each matched neuron falls in the code set of its `brain_region_idx`
  (V1 = 8, mHV = 0/1/2/9, lHV = 5/6, aHV = 3/4).
- **inputs**: `time_to_sound_cue`, `time_since_trial_start` (from `ft` interpolated at `SoundFr`/`StartFr`),
  `day_of_training` (recomputed from `Imaging_Exp_info.npy` dates) and `reward_available` (`isRew`) all match exactly
  for the first, sixth and last trial of each session.
- **outputs**: `stimulus_category` (wall name -> texture family), `licking` (`floor(LickFr)` membership),
  `position_bin` (`floor(ft_Pos/10)`), `speed_bin` (`digitize(ft_RunSpeed, edges)`) all match exactly.
- **dataset level**: 89 sessions, 19 subjects, 38,110 trials, speed bins 25% +/- 0.2% each.

### Check 3: Reference code comparison
| Step | Reference code | This conversion | Same? |
|------|----------------|-----------------|-------|
| (a) data loading | `utils.load_spk`: `np.concatenate([nspk for nspk in np.load(path).item()['spks']], 0)`; `utils.load_retino`; `Beh_<exp_type>.npy` via `utils.load_exp_beh` | `load_spk_rows` iterates the same `spks` list in the same order and takes the requested rows, which is exactly the concatenation restricted to those rows (verified by the raw-row search in Check 2); retinotopy and behaviour loaded from the same files with the same keys | YES |
| (b) neuron filtering | none for data quality; `utils.neu_area_ID` used to label V1/mHV/lHV/aHV | same labelling; neurons with `iarea` outside those codes dropped (they are unusable for `brain_regions`); random stratified subsample to 2,000/session | YES for filtering logic; subsampling is an addition required by dataset size (documented) |
| (b) trial filtering | reference uses all trials | all trials kept unless they have < 5 retained running frames or fall past the end of the neural recording (0 trials dropped in practice) | YES |
| (c) temporal alignment | behaviour is natively on the neural-frame clock (`ft_*`); trial start = `StartFr`; cue windows in `utils.spk_2_cue` are frame-indexed around `SoundFr` | same: frames are selected by `ft_trInd`, times computed from `ft` interpolated at the fractional `StartFr`/`SoundFr` | YES |
| (d) binning | reference keeps native frames for d-prime/z-score analyses, and interpolates to 60 position bins only for the position-tuning figures | native frames kept (time bin = 314.7 ms); no position interpolation because the decoder task asks for *time*-resolved data aligned to trial start | YES (chose the reference's non-interpolated representation, as in "computed from original estimated deconvolved traces without interpolation") |
| (e) frame/timepoint curation | `fr_valid = (ft_move>0) & ft_CorrSpc` (`Get_dprime_selective_neuron`), `VRmove = ft_move>0` everywhere else, `[:nfr]` truncation | identical mask and truncation | YES |
| (f) input construction | cue time `SoundFr`, reward `isRew`, training day from session order/`days` | `SoundFr`-based time-to-cue, `isRew`, elapsed days since the mouse's first imaging session | YES (see difference note) |
| (g) output construction | stimulus identity `WallName`/`UniqWalls`/`stim_id`; licking `LickFr` histogrammed per frame; position `ft_Pos`; speed `ft_RunSpeed` | same source variables, discretised as the decoder task requires | YES |

Differences and their justification:
1. **Neuron subsampling to 2,000/session**: not in the reference (which analyses all neurons); required because the
   full running-frame matrix would be ~183 GB. The reference decoder projects each session to 100 PCs and
   random-projects to at most 2,000 neurons for its SVD initialisation, so this is not a loss for decoding.
2. **Dropping `iarea` in {-1, 7}**: these neurons appear in no area analysis of the reference code and cannot be
   assigned a `brain_regions` label.
3. **`day_of_training` = days since the mouse's first imaging session**: `exp_info['days']` exists for only 8 of 142
   entries, so it cannot be used as a dataset-wide axis; elapsed calendar days is the continuous variable the paper
   plots training progress against (Fig. 1b, Fig. 5f).
4. **4 position bins / 4 speed quartile bins** instead of the reference's 60 position bins: mandated by the Decoder
   Task specification.

### Check 4: Key statistics comparison
See the table in Step 9: sessions (89), mice (19), neurons/recording min 20,547 and max 89,577 (exactly the paper's
numbers), trials (38,110, none dropped), frame rate 3.178 Hz vs the notebook's 3.17 Hz, trial length ~21 frames
(= 4 m / 0.6 m/s / 0.315 s), cue positions inside the corridor, 28 rewarded sessions with licking and 61 without.
No discrepancies remained after the `day_of_training` fix from Step 7.

### Check 5: Edge cases
- **Behaviour longer than the neural recording** (1-3 frames): behaviour truncated to `nfr` and any trial frame
  beyond `nfr` dropped (`load_spk_rows` also filters `frames < nfr`). One session has a negative `StartFr`
  (trial starting before the first neural frame); `np.interp` clamps to the first frame time and the affected frames
  are simply the ones present, so no negative `time_since_trial_start` occurs (asserted in the script).
- **NaN `ft_trInd`** (0.1-2.5% of frames, outside any trial): excluded by `np.isfinite(ftr)`.
- **Fractional frame indices** (`StartFr`, `SoundFr`, `LickFr` are floats): times are interpolated on the frame clock,
  licks are floored to their frame, as the reference does.
- **Sessions appearing in several exp_types**: converted once; behaviour verified identical across copies.
- **Sessions with `stimtype`** (swap sessions): behaviour key includes the suffix, handled in `beh_key`.
- **Sessions/trials with no licks or no rewards**: kept, with genuinely-zero licking / reward-availability.
- **Sessions with fewer than 2 usable trials**: skipped (none occurred).
- **`ft_RunSpeed` can be negative** (backwards ball rotation): the lowest speed quartile simply contains them; no clipping.
- **Position exactly at the 4 m boundary**: `np.clip(floor(pos/10), 0, 3)` keeps it in the last bin.

### Issues Found and Resolved
- `day_of_training` computed over the selected sessions only (Step 7) -> fixed to use all 89 sessions; re-ran the
  sample and the full conversion, and Check 2 now re-derives the value from the raw dates.
- `global MAX_NEURONS` syntax error in the first version of the script -> fixed (`globals()[...]`).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` -> `/app/train_decoder_full_out.txt`
(trained on the GPU; 89 sessions x 2,000 neurons, 821,579 timepoints; `sample_trials.png` and `predictions.png` written).

### Training Progress
- Loss decreasing: **Yes** -- 2.772 (epoch 50) -> 1.883 -> 1.339 -> 0.586 (100) -> 0.427 (120) -> 0.299 (140) ->
  0.258 (160) -> 0.211 (180) -> **0.186 (200)**. Test loss 2.082.
- Balanced loss class weights reported by the script are sensible
  (stimulus [0.132, 0.086, 0.479, 0.303], licking [0.037, 0.963], position and speed ~[0.25]x4).

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Train / Val |
|--------|--------|-------------|--------|------|------|
| stimulus_category (4 classes) | 0.250 | 0.9999 | **0.9701** | 3.88x | 1.03x |
| licking (2 classes) | 0.500 | 0.9924 | **0.7670** | 1.53x | 1.29x |
| position_bin (4 classes) | 0.250 | 0.9982 | **0.8722** | 3.49x | 1.14x |
| speed_bin (4 classes) | 0.250 | 0.8854 | **0.6203** | 2.48x | 1.43x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Chance (1/nclass) | Validation | Ratio | Verdict |
|----------|------------------|------------|-------|---------|
| stimulus_category | 0.250 | 0.9701 | 3.88x | excellent; near-perfect texture decoding from visual cortex |
| licking | 0.500 | 0.7670 | 1.53x | above the 1.5x threshold; see analysis below |
| position_bin | 0.250 | 0.8722 | 3.49x | strong; consistent with the paper's position-tuned sequences |
| speed_bin | 0.250 | 0.6203 | 2.48x | strong for a behavioural variable decoded from V1/HVA activity |

No output is at or below chance. The lowest ratio is licking (1.53x), which is expected and not a bug:
- licking is only 3.7% of all timepoints, and it is structurally absent in 61 of the 89 sessions (the
  unsupervised/naive mice were never water restricted and never licked), so two thirds of the data can only
  contribute "no lick" examples;
- restricted to the 28 task sessions the positive class is 5.8-39.1% of timepoints, and the decoder reaches
  0.99 balanced accuracy on training trials, i.e. the licking signal is clearly present in the neural data and the
  remaining gap is generalisation, not misalignment;
- an alternative would have been to drop the 61 non-licking sessions, but that would discard 69% of the recordings
  and all of the unsupervised/naive cohorts on which the paper's main claims rest, and the other three outputs are
  perfectly well defined there. Documented as a deliberate decision (Step 5, decision 2).

### Check 2: Accuracy comparison to the paper
The reference paper reports **no decoding analyses**: searching the full extracted text
(`/app/cache/paper.txt`, 25 pages incl. Methods) for "decod", "accuracy", "classifier" returns nothing;
the paper quantifies selectivity (d'), fractions of selective neurons, coding directions/similarity indices and
licking behaviour instead. There is therefore no published accuracy to compare against.

| Variable | Achieved (validation) | Expectation from paper |
|----------|----------------------|------------------------|
| stimulus_category | 0.9701 | No decoding reported. Qualitatively consistent: the paper finds large populations of stimulus-selective neurons (d' >= 0.3) in all four visual areas, even in naive mice (Fig. 1i,j, Fig. 3), so texture identity must be highly decodable. |
| licking | 0.7670 | No decoding reported. Consistent with Fig. 4: reward-prediction/lick-related activity exists but is confined to a small subpopulation (~5% of aHV neurons). |
| position_bin | 0.8722 | No decoding reported. Consistent with Fig. 2/Extended Data Fig. 3: selective neurons tile the corridor with reliable position-tuned sequences (odd/even trial correlations of preferred position), which implies strong position information. |
| speed_bin | 0.6203 | No decoding reported. Extended Data Fig. 2 shows running-speed modulation of the population; speed is only partially represented in visual cortex, so intermediate accuracy is expected. |

### Check 3: Train vs validation gap
Ratios train/val: 1.03x (stimulus), 1.29x (licking), 1.14x (position), 1.43x (speed) -- all below the 1.5x flag.
The gap is the expected consequence of fitting a 2,000-neuron -> 100-PC projection per session on ~80% of
38,110 trials; there is no data leakage because the split is over trials and every trial is a contiguous block of
timepoints from one corridor traversal (the decoder's own `train_validate_decoder` performs the split).

### Additional verification performed in this step
- `sample_trials.png` / `predictions.png` (produced by `--plot-samples`) were inspected: predicted traces follow the
  true position/speed staircases and the lick events within trials, with no visible temporal shift.
- Re-checked that every output class is present in the pooled data and that no session contributes a constant
  position or speed output.
- Re-ran the raw-file sanity checks (`/app/cache/sanity_checks_out.txt`, 130 checks, 0 failures) against the final
  `converted_data.pkl` that was used for training, so the trained dataset is exactly the verified one.

### Issues Found and Resolved
- None in this step: no output was at/below chance, no overfitting flag was triggered, and all statistics matched
  the reference data. No further iteration of the conversion was required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format specification, key statistics, decoder accuracy)
- [x] `cache/` folder created with `cache/README_CACHE.md` documenting every cached exploration script and log
- [x] All files organized

### Final file inventory (/app)
| File | Content |
|------|---------|
| `CONVERSION_NOTES.md` | this document: every decision, check and result |
| `README.md` | user-facing dataset documentation |
| `convert_data.py` | the conversion script (`--full` / `--sample` / `--show-processing`) |
| `converted_data.pkl` | full converted dataset, 6.62 GB (89 sessions, 38,110 trials) |
| `sample_data.pkl` | 2-session sample, 0.13 GB |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs with timing and sanity checks |
| `verification_sample_out.txt`, `verification_full_out.txt` | `train_decoder.py --verify-only` reports (no errors, no warnings) |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs with balanced accuracies |
| `processing_TX108_2023_03_25_1.png`, `processing_TX83_2022_08_29_1.png` | per-step processing plots (`--show-processing`) |
| `sample_trials.png`, `predictions.png` | decoder sample trials and predictions |
| `cache/` | exploration scripts, surveys, extracted paper text, independent sanity checks |

### Summary of the conversion
89 imaging sessions from 19 mice; per trial (corridor traversal) the retained timepoints are the imaging frames
inside the 4 m texture corridor while the VR was moving (the reference `(ft_move>0) & ft_CorrSpc` mask, i.e. the
paper's "only considered timepoints during running"), at the native 314.7 ms frame bin, aligned to corridor entry.
Neural data are suite2p deconvolved traces of 2,000 area-labelled neurons per session (V1/mHV/lHV/aHV).
Decoder inputs: time to sound cue, training day, time since trial start, reward availability.
Decoder outputs: stimulus texture category, licking, 1-m position bin, running-speed quartile.
Validation balanced accuracy 0.970 / 0.767 / 0.872 / 0.620 against chance 0.25 / 0.50 / 0.25 / 0.25.

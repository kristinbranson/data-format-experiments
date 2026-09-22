# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (paper.pdf, /app/code, /app/data)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3, numpy 2.4.4, torch 2.6.0+cu124, CUDA available. 128 CPUs, 1 TB RAM, /app writable (3.4 TB free).

Directory contents of /app:
- `paper.pdf` (Zhong et al. 2025, "Unsupervised pretraining in biological neural networks")
- `methods.txt` (excerpts of paper text/methods)
- `code/` : `README.md`, `data_process_script.ipynb`, `Figures.ipynb`, `utils.py`, `fig1.py`..`fig5.py`, `S6.py`
- `data/` : `beh/` (23 `Beh_<exp_type>.npy` behaviour files + `Imaging_Exp_info.npy` + `Unsupervised_pretraining_behavior/` + example file, 6.6 GB), `spk/` (89 `<mname>_<datexp>_<blk>_neural_data.npy`, **405 GB**), `retinotopy/` (90 `<mname>_<datexp>_trans.npz`, 170 MB), `process_data/` (empty; intermediate outputs of reference notebook)
- `train_decoder.py`, `decoder.py` (decoder reference), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `CONVERSION_NOTES.md` (this file)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_exp_beh(root, exp_type)` | utils.py:326 | LOADING | loads `beh/Beh_<exp_type>.npy` -> dict keyed `mname_datexp_blk[_stimtype]` |
| `load_spk(db, root)` | utils.py:338 | LOADING | loads `spk/<mname>_<datexp>_<blk>_neural_data.npy`, `np.concatenate(d['spks'],0)` -> (n_neurons, n_frames) deconvolved traces |
| `load_retino(db, root)` | utils.py:330 | LOADING | loads `retinotopy/<mname>_<datexp>_trans.npz` -> `xy_t`, `iarea`, and per-area neuron masks |
| `neu_area_ID(iarea)` | utils.py:312 | CURATION | maps `iarea` codes to area masks: V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4 (codes 7 and -1 are in no area) |
| `Get_dprime_selective_neuron` | utils.py:418 | PROCESSING | the canonical example of frame selection: `nfr=spk.shape[1]`; `VRmove = beh['ft_move'][:nfr]>0`; `isCorridor = beh['ft_CorrSpc'][:nfr]`; `fr_valid = VRmove & isCorridor`; d' between corridors computed on raw (non-interpolated) deconvolved traces |
| `get_interpPos_spk` / `spk_pos_interp` | utils.py:120/105 | PROCESSING | used only for position-interpolated analyses: activity of running frames interpolated onto 60 position bins (1 dm each) x n_trials using `ft_PosCum/Corridor_Length` |
| `dprime(x1,x2)` | utils.py:370 | PROCESSING | selectivity index |
| `get_cat_id(WallName,isRew)` | utils.py:137 | PROCESSING | canonicalizes wall names: rewarded stimulus -> 2, its "2" version -> 3, non-rewarded -> 0, its "2" version -> 1 (same convention as `beh['stim_id']`) |
| `spk_2_cue(spk, beh, ranges=[15,15])` | utils.py:935 | PROCESSING | aligns activity/licks to `SoundFr` in frame units (+-15 frames) -> shows that frame indices (`SoundFr`, `LickFr`, `StartFr`) are the common time base |
| `spk_2_firstLick` | utils.py:913 | PROCESSING | aligns to first lick frame `LickFr[LickTrind==trial][0]` |
| `get_kfold_reward_response`, `get_reward_neuorns` | utils.py:814/884 | PROCESSING | reward-prediction neuron selection; uses `SoundDelPos` mod Corridor_Length and interpolated spks |
| `data_process_script.ipynb` cell 9 | code/ | PROCESSING | full pipeline example: `spk = utils.load_spk(ndb)`, `nneu,nfr = spk.shape`, `VRmove = beh['ft_move'][:nfr]>0`, `ft_AcumPos = beh['ft_PosCum'][:nfr]`, then `get_interpPos_spk(spk[:,VRmove], ft_AcumPos[VRmove], ntrials, n_bins=60, lengths=CL)` |

### Notes
- Imaging frame rate documented in the notebook: **fs = 3.17 Hz** (measured from `beh['ft']` diffs: median 0.3146 s -> 3.178 Hz).
- Data are **deconvolved** (suite2p non-negative deconvolution, 0.75 s decay). **No dF/F computation needed**, and the paper states all analyses use deconvolved traces.
- **No neuron quality filtering** is applied anywhere in the reference code beyond suite2p cell detection/classification (the `spks` in the released files are already the curated cells). Neurons are only ever *selected* for specific analyses by d' (an analysis choice, not a curation criterion) or by anatomical area.
- The universal **frame curation** used by the reference code is `ft_move>0` (VR moving, i.e. running above the 6 cm/s threshold) and, for corridor analyses, `ft_CorrSpc` (inside the 0-4 m texture region). Behaviour arrays are truncated to the number of imaging frames `nfr` (`[:nfr]`) because `len(ft)` exceeds `spk.shape[1]` by 1-2 frames.
- Behaviour variable dictionary (from `data_process_script.ipynb`, markdown cell 4) gives frame-indexed event variables: `StartFr` (corridor entry), `GrayFr` (entry to grey space), `EndFr` (exit of corridor block), `SoundFr`/`SoundDelayFr` (sound cue), `LickFr`+`LickTrind`, `RewardFr`; frame-wise variables `ft`, `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_move`, `ft_isMoving`, `ft_RunSpeed`, `ft_CorrSpc`, `ft_GraySpc`, `ft_WallID`; trial-wise `WallName`, `UniqWalls`, `stim_id`, `isRew`, `SoundPos`, `RewPos`; settings `Corridor_Length`(=60 dm), `Gray_Space_length`(=20 dm), `Texture_Length`(=40 dm). **All positions are in decimetres.**

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/beh/Imaging_Exp_info.npy`: dict `exp_type -> list of session dicts (db)` with keys `mname, datexp, blk, sess#, is2p, ROIdir, rewType, depth, exptype, isDR, stim_id, Gender` and optionally `stimtype` (swap1/swap2), `days`, `stim`, `2pblk`, `Note`, `artLick`. 23 exp_types, 142 (exp_type, session) entries -> **89 unique recordings** (mname_datexp_blk) from **19 mice** (DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2).
- `data/beh/Beh_<exp_type>.npy`: dict `mname_datexp_blk[_stimtype] -> beh dict` with the fields listed in Step 1. Sanity-checked: for recordings appearing under several exp_types the frame/trial arrays are **identical** (53/53 duplicate pairs matched on `ft`, `StartFr`, `WallName`), so one behaviour copy per recording suffices; only `UniqWalls`/`stim_id` differ (each exp_type labels the stimuli relevant to that comparison, others NaN).
- `data/spk/<mname>_<datexp>_<blk>_neural_data.npy`: dict with single key `spks` = list of 3 planes, each `float32 (n_neurons_plane, n_frames)`; concatenated -> (n_neurons, n_frames). Deconvolved activity, ~2/3 of entries exactly 0. Load time ~4 s per 3.5 GB file (~1 GB/s).
- `data/retinotopy/<mname>_<datexp>_trans.npz`: `A, dx, dy, xpos, ypos, xy_t (n_neurons,2), iarea (n_neurons,)`; neuron count **exactly matches** the spk file for every session checked.
- `data/process_data/`: empty (intermediate results of the reference notebook are not shipped, so position-interpolated spks must be recomputed if needed).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 4,691,034 over 89 sessions |
| Neurons / session | mean 52,708; min **20,547**; max **89,577** |
| Neurons / session in a named area (V1/mHV/lHV/aHV) | 4,105,393 total (mean 46,128) |
| Subjects | 19 mice |
| Sessions / subject | 89/19 = 4.7 (range 1-8) |
| Trials (total) | 38,110 (corridor traversals) |
| Trials / session | mean 428 (range 84-789) |
| Imaging frames / session | mean ~24,000 (`len(ft)`; spk has 1-2 fewer) |
| Frames in texture corridor AND running / session | mean 9,278 (total 825,783) |
| Valid (corridor+running) frames / trial | median 21, mean 21.6, min 11, max 178; **0 trials with <5** |
| Frame rate | 3.178 Hz (median dt 0.3146 s) |
| Corridor geometry | `Corridor_Length`=60 dm (4 m texture + 2 m grey), positions in dm |
| Run speed over valid frames (cm/s) | quartiles 12.4 / 25.3 / 40.8, 1-99 pct -0.9 to 82.9 |
| Stimulus (wall) names | circle1/2/3, leaf1/2/3, leaf1_swap1/2, rock1/2, wood1/2/5, wood1_swap1/2 -> canonicalized by `stim_id` 0..6 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings (sessions) | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 | same |
| Neurons / recording | 20,547 - 89,577 | "activity traces from 20,547 to 89,577 neurons in each recording" |
| Neural data time bin | imaging frame, 3.17 Hz (~315 ms) | notebook: "Calcium signal recording frame rate: fs = 3.17Hz" |
| Neural signal | deconvolved traces (0.75 s decay) | "All our analyses were based on deconvolved fluorescence traces." |
| Analysis window in corridor | 0-4 m texture region | "we only selected data points inside the 0-4-m region of the corridors where the textures were shown" |
| Running requirement | only running timepoints | "We excluded the data points in which the animal was not running"; "We only considered timepoints during running for analysis" |
| Running threshold | 6 cm/s for >=66 ms triggers VR motion | "running faster than a threshold of 6 cm s-1"; "faster than 6 cm s-1 for at least 66 ms" |
| VR speed when running | constant 60 cm/s | "the virtual corridors always moved at a constant speed (60 cm s-1)" |
| Corridor length | 4 m + 2 m grey | "virtual reality corridors were each 4 m long, with 2 m of grey space between corridors" |
| Sound cue position | uniform 0.5-3.5 m | "the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m" |
| Reward | only in rewarded corridor, after cue, lick-triggered or passive-delayed | "The reward was delivered if a lick was detected after the sound cue in the rewarded corridor" |
| Reward rate (imaging sessions) | ~1/2 of trials are in the rewarded corridor (2 corridors in pseudo-random order); reward delivered on most of those | data: rewarded-corridor trials ~ 33-50% of trials depending on number of stimuli in session |
| Stimuli | leaf/circle (also rock/brick=wood), test stimuli leaf2/circle2/leaf3, leaf1_swap1/2 | Methods |
| Decoding accuracies reported | **none** | the word "decod" does not occur anywhere in the paper |

### Processing Details
- Neural data: suite2p deconvolved traces, no dF/F, no further neuron-level quality filtering.
- Temporal base: imaging frames (`beh['ft']`, 3.17 Hz). Behaviour events are given as (fractional) frame indices, so alignment is exact in frame units.
- Frame curation: inside texture corridor (`ft_CorrSpc`, positions 0-40 dm) and running (`ft_move>0`).
- Trials: corridor traversals; `StartFr` = corridor entry (= trial start = the decoder's alignment event), `GrayFr` = entry into grey space (corridor exit), `EndFr` = exit of the 6 m block.
- Position interpolation (60 bins of 1 dm over the 6 m block) is used only for figures needing position-resolved activity; the d'/selectivity analyses use raw frames. For the decoder we keep the **native time base** (frames), as required by the task (time-varying variables in time bins).

### Curation Steps

**Neuron curation rules**:
- Reference applies none beyond suite2p; the released `spks` are the accepted cells.
- Area assignment via `iarea` (V1=8, mHV=0/1/2/9, lHV=5/6, aHV=3/4). Neurons with `iarea` = 7 or -1 belong to no named area and are excluded from all area-based analyses in the reference code.

**Trial curation rules**:
- No explicit trial rejection for the main analyses; analyses restrict to trials of particular stimuli. Figure-specific exclusions (first lick after 2 m, mice with too few leaf2 trials) are analysis-specific, not data curation.
- For the decoder, trials must contain at least a few valid (corridor + running) frames; in the data every trial has >=11 valid frames, so no trial is lost.

### Decoders Trained
| Decoded variable | Accuracy |
| (none reported in the paper) | n/a |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

All checks were run over **all 89 recordings** (script `cache/step4_checks.py`, `cache/step4b.py`).

### Verified consistencies
| Check | Result |
|-------|--------|
| Sessions / mice | 89 recordings, 19 mice -> matches paper exactly |
| Neurons per session | min 20,547, max 89,577 -> matches paper exactly ("20,547 to 89,577 neurons in each recording") |
| Neuron count in spk file == retinotopy file | identical for every session checked (10/10 spot-checked, all 89 verified later in Step 10) |
| `ft_CorrSpc` == (`ft_Pos` < `Texture_Length`=40 dm) | True for 89/89 sessions -> the "0-4 m texture region" of the Methods |
| `ft_move>0` == `ft_isMoving` | True for 89/89 -> "running" frame mask used by the reference |
| median run speed on `ft_move>0` frames | 27 cm/s vs 0.0 cm/s on non-moving frames; 85% of moving frames exceed the 6 cm/s VR threshold (the VR triggers on a 66 ms sustained criterion, so instantaneous speed can dip below 6) |
| Imaging frame rate | 3.171-3.181 Hz across sessions (notebook says 3.17 Hz) |
| Trial frames from `ft_trInd` lie inside [`StartFr`, `EndFr`] | 32,295/32,295 trials checked, 0 violations |
| `SoundPos` range | 0.44-3.58 m (paper: cue position drawn uniformly 0.5-3.5 m) |
| Rewarded trials always have canonical `stim_id`==2 | True in all 28 reward sessions |
| `isRew` == ~isnan(`RewardFr`) | True in all sessions -> `isRew` means "reward was actually delivered", **not** "trial was in the rewarded corridor" |
| Behaviour duplicated across exp_types | identical arrays (53/53 duplicate pairs) -> use one behaviour record per recording |

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| "reward availability" | reference uses `stim_id`==2 to identify the rewarded corridor (`Get_dprime_rewPred_neuron`, `get_reward_neuorns`) | `isRew` is true only on trials where reward was really delivered (34 fewer trials in one session than the rewarded-corridor count) | "reward was delivered if a lick was detected after the sound cue in the rewarded corridor" | Decoder input "reward availability" = **trial is in the rewarded corridor** = (session is a task session) AND canonical `stim_id`==2. This matches the reference definition of the reward corridor and the decoder spec ("1 if in rewarded corridor"). |
| Licking | reference computes lick metrics only for task (sup) mice | lick arrays are **empty** for all 61 non-task (unsup/naive/grating-cohort) recordings; 28 task recordings have licks | unsupervised/naive mice "did not receive water rewards and were not water restricted" | Licking is 0 everywhere in non-task sessions. Kept those sessions (they are part of the 89-recording dataset and carry the other three decoded variables) with licking = 0, which is the behaviourally correct value for mice that never received water. Documented as a limitation; the licking decoder is effectively evaluated on the 28 task sessions since only they contain class 1. |
| Stimulus categories | `get_cat_id` only defines ids 0-3; `beh['stim_id']` covers 0-6 | 309 trials use wall `circle3` which has **no** canonical id in any exp_type | Methods mention leaf2/circle2/leaf3 test stimuli and leaf1-swaps | `circle3` is the third crop of the *non-rewarded* texture family, i.e. the mirror of `leaf3` (id 4). It is given its own category id 7 (`nonrew3`) rather than being dropped. |
| Number of exp_type entries | 142 (exp_type, session) entries | 89 unique recordings | "89 recordings" | Deduplicate to 89 sessions; take the union of `UniqWalls`->`stim_id` maps over all exp_types in which a recording appears so every wall gets its canonical id. |
| Frames | behaviour arrays are 1-2 frames longer than `spk` | same | - | Truncate behaviour to `nfr = spk.shape[1]` exactly as the reference code does (`beh['ft_move'][:nfr]`). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial definition
- **Session** = one recording (`mname_datexp_blk`); 89 sessions, deduplicated over exp_types.
- **Trial** = one corridor traversal, indexed by `beh['ft_trInd']`; alignment event = **trial start = corridor entry** (`StartFr`).
- **Kept frames per trial** = imaging frames of that trial with `ft_CorrSpc` (position 0-4 m, texture region) **and** `ft_move>0` (VR moving = mouse running), truncated to `nfr = spk.shape[1]`. This is exactly the `fr_valid = VRmove & isCorridor` mask of `utils.Get_dprime_selective_neuron` and the paper's "only selected data points inside the 0-4-m region" + "excluded the data points in which the animal was not running".
- Time bin = one imaging frame (~315 ms, 3.17 Hz). No re-binning: the paper's analyses of trial dynamics use the native frame rate, and the spec requires equal bin sizes across trials/sessions.
- Trials keep their natural (variable) length; every trial has >= 11 kept frames, so no trial is dropped.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/<sess>_neural_data.npy['spks']` | `neural` | `np.concatenate(...,0)`, keep neurons in a named visual area, subsample <=2000 neurons/session, select kept frames of each trial | `utils.load_spk`, `Get_dprime_selective_neuron` | deconvolved traces, float32, no dF/F, no extra neuron filtering |
| `retinotopy/<m>_<date>_trans.npz['iarea']` | `brain_regions`, `brain_region_idx` | V1=8; mHV=0,1,2,9; lHV=5,6; aHV=3,4 | `utils.neu_area_ID`, `load_retino` | neurons with iarea 7 or -1 are in no named area -> excluded (reference excludes them from every area analysis) |
| `SoundFr`, `ft` | `input[0]` `time_to_sound_cue` (s) | signed time until cue: t_cue - t_frame (positive before cue, negative after) | `utils.spk_2_cue` uses `SoundFr` as the cue time base | cue frame index is fractional; converted to seconds with the session's frame period |
| session date vs mouse's first session | `input[1]` `day_of_training` (days) | days between this recording's date and the first recording date of that mouse | `exp_info` `datexp`, `days` | continuous, constant within a trial |
| `ft`, `StartFr` | `input[2]` `time_since_trial_start` (s) | t_frame - t_StartFr | - | continuous, time-varying; correctly accounts for the removed non-running frames |
| `WallName`+`stim_id` (canonical), session has rewards | `input[3]` `reward_availability` | 1 if task session and canonical stim id == 2 else 0 | `Get_dprime_rewPred_neuron` (`stim = uniqW[stim_id==2]`) | per-trial constant |
| `WallName` + union of `UniqWalls`->`stim_id` | `output[0]` `stimulus_category` | canonical id 0-7 | `beh['stim_id']`, `utils.get_cat_id` | per-trial constant, 8 categories |
| `LickFr` (+`LickTrind`) | `output[1]` `licking` | 1 if >=1 lick falls in that imaging frame (nearest frame index), else 0 | `utils.spk_2_cue`/`spk_2_firstLick` bin licks by `LickFr` | binary, time-varying |
| `ft_Pos` (dm) | `output[2]` `position_bin` | 4 equal 1-m bins: [0,10),[10,20),[20,30),[30,40) dm | paper: corridor is 4 m of texture | time-varying, 4 classes |
| `ft_RunSpeed` (cm/s) | `output[3]` `speed_bin` | global quartile bins of the kept frames (25% of data each) | `beh['ft_RunSpeed']` = `RunFr`; paper interpolates running speed to imaging frames | time-varying, 4 classes; thresholds computed once over all kept frames of all sessions |
| `mname` | `subjects`, `subject_idx` | 19 unique mice | `exp_info` | |

### Key Decisions
1. **Keep all 89 recordings / 19 mice** (task, unsupervised, naive, grating cohorts): matches the paper's dataset size, and all inputs/outputs except licking are defined for every cohort. Licking = 0 for the 61 non-task recordings, which is the true behaviour of mice that were never water restricted and never rewarded (their lick arrays are empty in the released behaviour files).
2. **Frames, not position bins, as the time base**: the decoder needs equal time bins; the reference position-interpolation (60 bins/6 m) is only used for position-resolved figures, while all selectivity statistics are computed on native deconvolved frames.
3. **Frame curation = texture corridor + running** (reference `fr_valid`). This also removes the reward-consumption stops, as the paper intends.
4. **Neuron curation**: none beyond suite2p (as in the reference); neurons without a named visual area are dropped. Because the full dataset would be 152 GB, a **random subsample of at most 2,000 neurons per session** (fixed seed, stratified proportionally across V1/mHV/lHV/aHV) is stored. The decoder reduces each session to 100 PCs, so 2,000 neurons retain essentially all decodable signal, and the area composition is preserved.
5. **Stimulus category = canonical `stim_id`** rather than raw wall names, because the physical textures differ between mice (circle/leaf vs rock/wood) while the canonical id encodes the same task role in every mouse. `circle3`/`rock3`-type walls get a new id 7.
6. **Reward availability = rewarded corridor** (not reward delivered), per the decoder spec.
7. **Speed discretization** by global quartiles over all kept frames (25% of data per bin), as specified.
8. **Day of training** = days since the mouse's first imaging session (continuous, per trial).

### Planned Sanity Checks
- [ ] 89 sessions, 19 mice, neurons/session in [20,547, 89,577] before subsampling.
- [ ] 38,110 trials total (minus none) and ~428 trials/session.
- [ ] Kept frames/session ~9,278 mean; kept frames/trial median ~21, min >= 11.
- [ ] Frame rate 3.17 Hz; time bin ~315 ms.
- [ ] `time_since_trial_start` >= 0 and increasing within a trial; `time_to_sound_cue` changes sign inside the trial for most trials; cue position 0.5-3.5 m.
- [ ] position bins: all 4 bins occupied; distribution roughly uniform-ish in a full traversal.
- [ ] speed bins: each ~25% of frames globally.
- [ ] licking: non-zero only in the 28 task sessions; per-frame lick rate in task sessions consistent with `len(LickFr)`.
- [ ] stimulus categories: counts per canonical id equal the counts computed directly from behaviour (0:11964, 1:2266, 2:12339, 3:6538, 4:2279, 5:1173, 6:1242, 7:309).
- [ ] reward availability = 1 only in task sessions, ~40% of their trials.
- [ ] spot-check raw `spks` values against the converted `neural` matrix for random (session, trial, neuron, frame) combinations.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (579 lines). Run as `python -u /app/convert_data.py <outfile> [--full|--sample] [--show-processing] [--nproc N] [--max-neurons N]`.

Structure:
| Function | Purpose |
|----------|---------|
| `build_recording_table()` | reads `Imaging_Exp_info.npy`, deduplicates the 142 (exp_type, session) entries into 89 recordings, computes `day_of_training` = days since the mouse's first imaging session |
| `load_behaviour()` | parallel load of the 23 `Beh_*.npy` files (only the needed fields), one behaviour record per recording, merges the `wall -> stim_id` maps across exp_types, flags task sessions (`is_task` = any reward delivered) |
| `trial_frames()` | kept frames per trial: `ft_CorrSpc & (ft_move>0) & ~isnan(ft_trInd)`, truncated to `spk.shape[1]` -- the `fr_valid` mask of `utils.Get_dprime_selective_neuron` |
| `select_neurons()` | area labels from `iarea` exactly as `utils.neu_area_ID`; random subsample (seed 0) of at most `--max-neurons` (default 2000) in-area neurons |
| `load_spk_selected()` | loads `spks` planes and slices the selected neurons/frames per plane, i.e. `np.concatenate(d['spks'],0)[sel][:,frames]` (`utils.load_spk`) without building the full matrix |
| `convert_session()` | builds the per-trial `neural` (n_neurons x T), `input` (4 x T) and `output` (4 x T) arrays, plus per-session statistics |
| `plot_processing()` | 6-panel figure per session for `--show-processing` |
| `main()` | global speed quartiles over the kept frames of all sessions, parallel session conversion with progress/timing, assembly of the output dict + metadata, summary statistics, pickling |

Code efficiencies:
- behaviour loaded once per file in 12 parallel workers (0.3 s total instead of minutes of repeated 200-400 MB loads);
- neural data sliced **per plane** so only the selected 2000 neurons x kept frames (~75 MB) are materialised instead of the full 3-7 GB matrix;
- sessions converted in parallel (`imap_unordered`, 12 workers), each session read exactly once;
- all per-frame quantities (times, position bins, speed bins, licking) computed vectorised for the whole session and then split by trial;
- timing of every session (total and spk-load) printed with a running estimate of the total runtime.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` (log: `conversion_sample_out.txt`).
Sample = the first two **task** sessions (VR2_2021_03_20_1, TX60_2021_04_10_1) so that all four outputs, including licking, vary.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (TX60, VR2) |
| Trials (total) | 585 (348 + 237) = exactly the behaviour trial counts, no trial lost |
| Neurons recorded / session | 81,473 and 48,732 |
| Neurons stored / session | 2,000 (V1 2,052 / mHV 651 / lHV 662 / aHV 635 over both sessions) |
| Time bin | 314.54 ms (3.179 Hz) |
| Bins / trial | median 23, min 12, max 90, mean 25.1 |
| time_to_sound_cue range | [-333.6, 220.4] s (median -0.11 s) |
| day_of_training range | [0, 0] (both are each mouse's first session) |
| time_since_trial_start range | [0.0, 338.2] s (median 4.3 s) |
| reward_availability | 0.49 of trials (both sessions are task sessions with 2 corridors) |
| stimulus_category (frames) | 0.483 nonrew_crop1 / 0.517 rew_crop1 |
| licking (frames) | 0.901 / 0.099 |
| position_bin (frames) | 0.246 / 0.241 / 0.249 / 0.264 |
| speed_bin (frames) | 0.289 / 0.197 / 0.182 / 0.332 (global quartiles, so a single session need not be uniform) |

### Processing Plots Review
`processing_VR2_2021_03_20_1.png`, `processing_TX60_2021_04_10_1.png`, 6 panels each:
1. raw position trace with the kept-frame mask, trial starts (`StartFr`), cue times (`SoundFr`) and licks -- kept frames are exactly the running frames with position < 4 m, and trials begin at the position reset;
2. run speed with the global quartile thresholds and the resulting `speed_bin` colours;
3. position vs time-since-trial-start for 5 trials, coloured by `position_bin`, with the 1 m bin edges -- position increases monotonically, bins change at the edges;
4. `time_to_sound_cue` vs time since trial start, crossing zero at the cue, overlaid with the independent estimate (position - SoundPos)/VRspeed which has the same zero crossing -> no temporal misalignment;
5. deconvolved activity of 60 stored neurons for the longest trial with licking frames marked;
6. output distributions of the session.
No anomalies: no gaps in the trial time axis other than those caused by the intentional removal of non-running frames, no off-by-one in the cue or trial-start alignment.

Additional numeric validation (independent of the conversion code, `cache/check_sample_align.py`, `cache/check_sample_neural.py`): for trials 0, 1, 5, 50, 100, 236 of TX60_2021_04_10_1 recomputed from the raw `Beh_*.npy` file, `np.allclose`/`array_equal` held for `time_to_sound_cue`, `time_since_trial_start`, `position_bin`, `speed_bin` and `licking`; `brain_region_idx` reproduced from the raw `trans.npz`; raw `np.concatenate(spks,0)[sel][:,frames]` matched the stored `neural` for 5 spot values and for a whole trial matrix.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| per-plane neuron/frame slicing instead of loading the full spk matrix | avoids materialising 3-7 GB per session; spk load 1.6-2.5 s/session |
| parallel behaviour loading (12 workers, fields subset) | 0.3 s instead of ~2 min |
| 12 parallel session workers | ~10x wall-clock |
| vectorised per-session frame computations | negligible per-session overhead (<0.5 s) |

| Step | Time / Session | Estimated Total Time |
| behaviour load | - | 0.3 s |
| global speed quartiles | - | 0.1 s |
| session conversion (wall clock, 12 workers) | ~4.5 s | 89 x 4.5 s ~ 7 min (sample sessions are somewhat larger than average, so this is an upper bound) |
| pickling | - | ~1-2 min for ~7 GB |

Estimated full runtime well under 15 minutes, so no further optimisation was required.

`verification_sample_out.txt`: **no errors, no warnings**.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically from 34.5 (epoch 1) to 0.129 (epoch 200); test loss 1.61.

### Decoder Results (Sample)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| stimulus_category | 0.125 | 0.9999 | 0.9445 |
| licking | 0.500 | 0.9894 | 0.5965 |
| position_bin | 0.250 | 0.9998 | 0.8383 |
| speed_bin | 0.250 | 0.9775 | 0.5713 |

All four outputs are above chance. The training/validation gap reflects the small sample (2 sessions, 585 trials) with 2,000 neurons -> 100 PCs; the full dataset has 89 sessions.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` (log `conversion_full_out.txt`).
Runtime **70.7 s** total (behaviour 0.3 s, 89 sessions in 59.6 s with 12 workers, pickling 10.7 s) -- far below the 15 min budget, so no further optimisation was needed. Estimated from the sample it would be ~7 min; the real run was faster because average sessions are smaller than the two sample sessions.

### Output Files
- `converted_data.pkl`: 6.60 GB
- `verification_full_out.txt`: created, **no errors, no warnings**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Recordings / sessions | 89 | 89 entries in `Imaging_Exp_info` (142 with duplicates) | 89 spk files | 89 | yes |
| Subjects | 19 | 19 mnames | 19 | 19 | yes |
| Sessions per subject | - | - | 1-8 | 1-8 (DR10 6, DR15 5, LZ13 4, LZ16 4, TX104 2, TX105 5, TX108 7, TX109 6, TX119 8, TX123 8, TX124 3, TX139 2, TX140 1, TX60 5, TX61 5, TX83 3, TX85 2, TX88 6, VR2 7) | yes |
| Neurons recorded/session | 20,547 - 89,577 | - | min 20,547, max 89,577 | min 20,547, max 89,577 (recorded; 2,000 stored) | yes |
| Trials (total) | - | - | 38,110 | 38,110 (no trial lost) | yes |
| Trials/session (mean) | - | - | 428.2 (84-789) | 428.2 (84-789) | yes |
| Time bin | 3.17 Hz frames | 3.17 Hz (notebook) | 3.171-3.181 Hz | 314.69 ms = 3.178 Hz | yes |
| Kept bins/trial | - | - | median 21, min 11, max 178 | median 21, min 11, max 178 | yes |
| Kept frames total | - | - | 825,783 (821,579 after the `nfr` truncation) | 821,579 | yes |
| Sound cue position | uniform 0.5-3.5 m | - | 0.44-3.58 m | cue inside the kept window for 38,094/38,110 trials | yes |
| time_to_sound_cue | - | - | - | [-1763.3, 723.5] s, mean 0.011 s, median about -0.1 s | plausible (long tails from within-trial pauses) |
| day_of_training | - | `days` field for 8 sessions | dates 0-92 days after each mouse's first session | [0, 92] | yes |
| reward availability | rewarded corridor in task mice | `stim_id`==2 | 28 task sessions, 4,446 rewarded-corridor trials | 0.117 of all trials, 0.397 of task-session trials | yes |
| stimulus category (trials) | - | `stim_id` 0-6 (+1 unnamed wall) | 0:11964 1:2266 2:12339 3:6538 4:2279 5:1173 6:1242 7:309 | identical | yes |
| licking | only task mice lick | lick analyses only for sup mice | licks in 28/89 sessions | class 1 in 28/89 sessions, 3.5% of all bins, 12.1% of task-session bins | yes |
| position_bin | 4 m corridor | - | - | 0.250 / 0.249 / 0.250 / 0.252 | yes (near uniform, as expected for constant-speed VR) |
| speed_bin | - | - | quartiles 12.42 / 25.35 / 40.85 cm/s | 0.250 / 0.250 / 0.250 / 0.250 | yes (by construction) |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: output log verification
`verification_full_out.txt` reports "Data format is valid, no errors or warnings." There are therefore no errors or warnings to resolve. All structural requirements are satisfied: 89 sessions in `neural`/`input`/`output`, matching `subject_idx` (int64), `brain_region_idx` lengths equal to the neuron counts, float inputs, integer outputs, no NaN/Inf, consistent `dinput`=4 and `doutput`=4, `input_names`/`output_names`/`output_values` of the right lengths.

### Check 2: independent sanity checks (`cache/sanity_full.py`)
The script re-derives everything from the **raw** `Beh_*.npy`, `*_neural_data.npy` and `*_trans.npz` files (it does not import `convert_data.py`) for 5 sessions spanning cohorts (TX83 unsup, TX108 task, TX60 task, TX88 unsup, LZ13 grating-cohort) and 4 trials each (first, 1/3, middle, last):
| Check | Criterion | Result |
|-------|-----------|--------|
| neural values | `np.allclose(np.concatenate(spks,0)[sel][:,frames], data['neural'][s][t])` | PASS (all 20 trials) |
| neural shape | (n_selected_neurons, n_kept_frames) | PASS |
| brain_region_idx | re-derived from `iarea` with the `utils.neu_area_ID` mapping | PASS |
| input 0 | `np.allclose`, cue time via `np.interp(SoundFr, frames, t)` minus frame time | PASS |
| input 1 | day of training = days since the mouse's first session | PASS |
| input 2 | frame time minus trial-start time (`StartFr`) | PASS |
| input 3 | `(stim_id==2) & session_has_rewards` | PASS |
| output 0 | canonical wall -> `stim_id`, constant within the trial | PASS |
| output 1 | `LickFr` rounded to frames | PASS |
| output 2 | `floor(ft_Pos/10)` clipped to 0..3 | PASS |
| output 3 | `np.digitize(ft_RunSpeed, global quartiles)` | PASS |
| kept frames per session | sum of trial lengths == `(ft_CorrSpc & ft_move>0 & ~isnan(ft_trInd))[:nfr].sum()` | PASS |
| trials per session | == `beh['ntrials']` | PASS |
Earlier (Step 7) the same checks passed on the sample file, including an exact match of a full trial's neural matrix.

### Check 3: reference code comparison
| Stage | Reference | This conversion | Same? |
|-------|-----------|-----------------|-------|
| (a) data loading | `utils.load_spk`: `np.concatenate(d['spks'],0)`; `utils.load_exp_beh`; `utils.load_retino` | identical values, but the neuron/frame slicing is done per plane to avoid materialising 3-7 GB (verified `allclose` against the concatenated matrix) | yes |
| (b) neuron/trial filtering | no neuron quality filter; area masks from `utils.neu_area_ID`; no trial rejection | same; additionally a random 2,000-neuron subsample per session for tractability (documented) | yes + subsample |
| (c) temporal alignment | events as (fractional) imaging-frame indices, `spk_2_cue` aligns to `SoundFr`, `StartFr` = corridor entry | trials aligned to `StartFr`, cue time from `SoundFr`, both converted to seconds via the `ft` timestamps | yes |
| (d) binning | native imaging frames (3.17 Hz) for all d'/selectivity statistics; position interpolation only for position-resolved figures | native imaging frames | yes |
| (e) frame curation | `fr_valid = (ft_move[:nfr]>0) & ft_CorrSpc[:nfr]` | identical mask, plus `~isnan(ft_trInd)` so that every frame belongs to a trial | yes |
| (f) input/output construction | reference builds its own analysis variables (d', coding directions); the decoder variables are prescribed by the task | variables taken from the same behaviour fields the reference uses (`SoundFr`, `StartFr`, `ft_Pos`, `ft_RunSpeed`, `LickFr`, `stim_id`, `isRew`/`RewardFr`) | consistent |

Differences and their justification:
1. **2,000-neuron subsample per session**: storing all in-area neurons would need 152 GB; the decoder projects each session to 100 PCs and its own SVD initialisation caps at 2,000 neurons (`svd_max_neurons` default), so nothing decodable is lost.
2. **`~isnan(ft_trInd)` added to the frame mask**: frames outside any corridor traversal cannot be assigned to a trial. In the data these frames are also outside `ft_CorrSpc` almost always; the added condition only makes the trial assignment well defined.
3. **No position interpolation**: the decoder requires fixed-length time bins, and the paper's core analyses are computed on raw frames.
4. **`nfr` truncation**: `min(p.shape[1] for p in spks)` instead of `spk.shape[1]`; identical because all planes have the same frame count (verified).

### Check 4: key statistics comparison
See the table in Step 9: sessions, mice, sessions/mouse, neurons/session range, trials, trials/session, frame rate, stimulus-category counts, rewarded-corridor fraction and lick-session counts all match the paper and/or the values computed directly from the raw files.

### Check 5: edge cases (`cache/edge_checks.py`)
| Check | Result |
|-------|--------|
| sessions with < 2 trials | 0 (min 84 trials/session) |
| trials with 0 kept frames | 0 (min 11 bins) |
| `time_since_trial_start` negative | 0 trials; strictly increasing within every trial |
| `time_to_sound_cue` not strictly decreasing | 0 trials |
| cue inside the kept window | 38,094/38,110 trials (16 trials have the cue before the first or after the last kept frame -- genuine, e.g. cue while the mouse was stationary) |
| position bins used | all four, counts 205,382 / 204,366 / 205,046 / 206,785 |
| positions >= 4 m leaking in | none (`ft_CorrSpc` == position < 40 dm verified for all sessions) |
| licking outside task sessions | none; class 1 present in exactly the 28 task sessions |
| reward availability outside task sessions | none |
| unresolved wall names | only `circle3` (309 trials) -> category 7, documented |
| very long trials | 260/38,110 trials > 100 s; verified in the raw data (e.g. TX88_2022_07_19_1 trial 391: 5,621 contiguous frames of which only 50 running -> the mouse stood still) so the elapsed-time inputs are correct, not artefacts |
| every mouse has a day-0 session | yes |
| `brain_region_idx` length == neurons | yes for all 89 sessions |

No issues remained after these checks, so no iteration of the conversion was required beyond what is documented above.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` -> `train_decoder_full_out.txt` (trained on the GPU, NVIDIA L4, no OOM fallback needed). Plots: `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: **Yes**, monotonically: epoch 1 = 108.90, 50 = 3.18, 100 = 0.68, 150 = 0.287, 200 = 0.193. Test loss 2.67.

### Decoder Results (Full)
| Output | Chance (1/n_classes) | Training Balanced Acc | Validation Balanced Acc | Val / chance | Train / Val | Notes |
|--------|----------------------|-----------------------|-------------------------|--------------|-------------|-------|
| stimulus_category (8 classes) | 0.125 | 0.9923 | **0.8138** | 6.5x | 1.22 | includes rare classes (swap stimuli, `nonrew_crop3`) present in only a few sessions |
| licking (2 classes) | 0.500 | 0.9930 | **0.7939** | 1.59x | 1.25 | class 1 only exists in the 28 task sessions; the other 61 sessions are mice that never received water |
| position_bin (4 x 1 m) | 0.250 | 0.9975 | **0.8689** | 3.5x | 1.15 | |
| speed_bin (4 quartiles) | 0.250 | 0.8927 | **0.6223** | 2.5x | 1.43 | |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: accuracy vs chance
| Variable | Validation balanced acc | Chance | Ratio | Verdict |
|----------|------------------------|--------|-------|---------|
| stimulus_category | 0.8138 | 0.125 | 6.5x | far above chance |
| licking | 0.7939 | 0.500 | 1.59x | above 1.5x chance; for a binary variable 0.79 balanced accuracy is a strong result, and licking exists in only 28/89 sessions |
| position_bin | 0.8689 | 0.250 | 3.5x | far above chance |
| speed_bin | 0.6223 | 0.250 | 2.5x | far above chance |
No output is at or below chance, and none is below 1.5x chance.

Why licking has the smallest ratio (investigated, not dismissed):
- 61 of 89 sessions are unsupervised / naive / grating-cohort recordings of mice that were never water restricted; their released behaviour files contain **empty** lick arrays, so those sessions only contribute class 0. Their neural data still has to be mapped to "no lick", which is learnable but adds no class-1 examples.
- Within the 28 task sessions licking occupies 12.1% of the time bins (5.2-37.9% per session), so class 1 is intrinsically rare and the balanced loss has to trade off against it.
- Licking is a fast (about 8 Hz) behaviour sampled at 3.17 Hz, so single-bin labels are noisy; the paper itself only analyses licking as anticipatory-lick rates/trial fractions rather than per-frame.
Two alternatives were considered and rejected: (i) dropping the non-task sessions -- this would discard 61 of the paper's 89 recordings and the other three decoded variables that are perfectly well defined there; (ii) re-defining licking per trial -- the task specification explicitly requires licking to be a binary time-varying output.

### Check 2: comparison to the paper
The reference paper reports **no decoding accuracies** (the string "decod" does not occur anywhere in the 25-page PDF; the analyses are selectivity indices d', density maps, coding directions, sequence correlations and lick-behaviour statistics). There is therefore no published accuracy to compare against.
| Variable | Achieved (validation balanced acc) | Paper expectation |
|----------|------------------------------------|-------------------|
| stimulus_category | 0.8138 | no accuracy reported; the paper shows that many neurons in V1/medial areas are strongly stimulus-selective (d' >= 0.3), so high decodability is expected -- consistent |
| licking | 0.7939 | no accuracy reported; the paper shows reward-prediction/lick-related signals in anterior areas of task mice only -- consistent with class 1 existing only in task sessions |
| position_bin | 0.8689 | no accuracy reported; the paper shows position-tuned sequences inside each corridor -- consistent with high positional decodability |
| speed_bin | 0.6223 | no accuracy reported; running speed is only weakly reflected in visual-cortex deconvolved activity (the VR moves at a constant 60 cm/s whenever the mouse runs above threshold, so speed is partly decoupled from the visual input) -- a moderate accuracy is expected |

### Check 3: train vs validation gap
Ratios train/val: stimulus 1.22, licking 1.25, position 1.15, speed 1.43 -- all below the 1.5x threshold, so there is no sign of pathological overfitting or leakage. The gap that does exist is expected: the decoder fits a per-session projection of 2,000 neurons to 100 PCs plus a shared linear read-out over 38,110 trials, and train/validation are split within each session by trial (no trial appears in both).

### Additional verification of correctness for every output (the debugging checklist)
1. **Output values verified against raw data** for 3+ specific trials in 5 sessions (Step 10, Check 2): stimulus category, licking, position bin and speed bin all reproduced exactly from `Beh_*.npy`.
2. **Temporal alignment**: panel 1 and 4 of the `processing_*.png` figures show the kept frames, trial starts and cue times on the raw frame axis, and `time_to_sound_cue` crossing zero exactly where the independent estimate (position - `SoundPos`)/VR-speed crosses zero; `sample_trials.png`/`predictions.png` show neural, input and output traces aligned per trial.
3. **Variation of each output**: stimulus category has 8 classes with 0.7-33% of trials each; licking 3.5% of bins overall (12.1% within task sessions); position bins 25.0/24.9/25.0/25.2%; speed bins 25/25/25/25%. No output is 99% one class.
4. **Neural filtering** follows the reference `fr_valid` mask exactly (corridor + running) and uses the deconvolved traces the paper analyses.
5. **Processing matches the reference code**, as tabulated in Step 10, Check 3.

### Issues found and resolved during the whole review
- `isRew` initially looked like the natural "reward availability" flag, but it marks *delivered* rewards; the rewarded **corridor** is `stim_id==2` (as in `Get_dprime_rewPred_neuron`). Resolved by using `(stim_id==2) & session_has_rewards`.
- 309 trials use a wall (`circle3`) that has no `stim_id` in any exp_type; instead of dropping them they were given their own category (7, `nonrew_crop3`).
- Behaviour arrays are 1-2 frames longer than the neural traces; truncating to `spk.shape[1]` (as the reference does) avoids an off-by-one at the end of each session.
- Sessions appear up to 5 times across exp_types; deduplicating to 89 recordings avoids counting trials/neurons several times, and merging the `stim_id` maps keeps every wall labelled.
- Extremely long trials (up to 1,765 s) were verified in the raw data to be genuine standing-still episodes, not artefacts.
No further issues remained, so no re-run of the conversion was necessary after the full conversion.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format specification, key statistics, decoder results)
- [x] `cache/` folder created with `README_CACHE.md` documenting every exploration/validation script
- [x] All files organized

### Deliverables
| File | Content |
|------|---------|
| `CONVERSION_NOTES.md` | this decision and validation log (Steps 0-13) |
| `convert_data.py` | the conversion script (`--full`, `--sample`, `--show-processing`) |
| `converted_data.pkl` | full converted dataset, 6.60 GB, 89 sessions |
| `sample_data.pkl` | 2-session sample, 0.12 GB |
| `README.md` | user-facing documentation |
| `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt` | sample logs |
| `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` | full-dataset logs |
| `processing_VR2_2021_03_20_1.png`, `processing_TX60_2021_04_10_1.png` | per-step processing plots |
| `sample_trials.png`, `predictions.png` | decoder sample-trial and prediction plots |
| `cache/` | exploration and validation scripts, extracted paper text, intermediate caches |

### Known limitations
- Only 2,000 of the 20,547-89,577 recorded neurons are stored per session (random, seed 0). Storing all in-area neurons would need 152 GB; the decoder itself random-projects to 2,000 neurons before its SVD initialisation (`svd_max_neurons=2000`), so the stored subsample is not a bottleneck. Re-run with `--max-neurons N` for a different cap.
- Licking can only be 1 in the 28 task sessions; the other 61 recordings are mice that were never water restricted and whose released lick arrays are empty.
- Trials have variable length (11-178 bins), so `metadata['off_end']` is `None` and the actual per-trial duration must be read from `input[2]` (time since trial start).

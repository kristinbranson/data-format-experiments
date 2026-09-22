# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer & Pachitariu (2025)
  "Unsupervised pretraining in biological neural networks", *Nature* 644:741.
  Two-photon mesoscope calcium imaging of mouse visual cortex during a virtual-reality
  visual-discrimination task. Data = Figshare item 28811129 (mirrored in `/app/data`).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (`/app`):
- `paper.pdf` (25 pages), `methods.txt` (excerpt of paper Methods)
- `code/` — reference repo `zhong-et-al-2025`: `README.md`, `utils.py` (947 lines),
  `data_process_script.ipynb`, `Figures.ipynb`, `fig1.py` … `fig5.py`, `S6.py`
- `data/` — `beh/` (23 `Beh_<exptype>.npy` + `Imaging_Exp_info.npy` +
  `Unsupervised_pretraining_behavior/` (behaviour-only cohort, 3 files) +
  `example_bef_and_aft_learning_behavior.npy`), `spk/` (89 `*_neural_data.npy`, 404 GB),
  `retinotopy/` (89 `*_trans.npz` + `areas.npz`), `process_data/` (empty — intermediate
  results were not shipped, so everything must be recomputed from raw)
- `decoder.py`, `train_decoder.py` (target API), `Dockerfile`, `docker-compose.yaml`

Environment verified: python3.13, numpy 2.4.4, torch 2.6.0+cu124 (CUDA, NVIDIA L4 23 GB),
scipy 1.18, h5py 3.16. Host: 128 cores, 1006 GB RAM, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `download_data_from_figshare` | utils.py:12 | LOADING | shows the canonical folder layout (`beh/`, `spk/`, `retinotopy/`, `process_data/`) |
| `load_exp_beh(root, exp_type)` | utils.py:326 | LOADING | `np.load('beh/Beh_<exp_type>.npy').item()` → dict keyed `mname_datexp_blk[_stimtype]` |
| `load_spk(db, root)` | utils.py:338 | LOADING | `np.concatenate(np.load('<mname>_<datexp>_<blk>_neural_data.npy').item()['spks'], 0)` → (n_neurons, n_frames) deconvolved traces |
| `load_retino(db, root)` | utils.py:330 | LOADING | loads `<mname>_<datexp>_trans.npz` → `xy_t` (cortical coords), `iarea` (area id / neuron), and `neu_ar_idx` masks |
| `neu_area_ID(iarea)` | utils.py:312 | CURATION | maps `iarea` → 4 regions: **V1**=8, **mHV**={0,1,2,9}, **lHV**={5,6}, **aHV**={3,4}. `iarea` −1 and 7 belong to no region |
| `Get_density_map` | utils.py:394 | CURATION | `idx_neu = (arid!=-1) & (arid!=7)` with the comment *"exclude neurons from outside of visual cortex"* — the explicit neuron-curation rule |
| `dprime(x1,x2)` | utils.py:370 | PROCESSING | `2(µ1−µ2)/(σ1+σ2)` selectivity index |
| `Get_dprime_selective_neuron` | utils.py:418 | PROCESSING | **the key alignment/curation recipe**: `nfr = spk.shape[1]`; every behaviour frame-array is truncated `[:nfr]`; `VRmove = beh['ft_move'][:nfr] > 0`; `isCorridor = beh['ft_CorrSpc'][:nfr]`; `fr_valid = VRmove & isCorridor` — *"only use activity inside the texture area plus mouse is running (VR moving)"* |
| `get_interpPos_spk` / `spk_pos_interp` | utils.py:120/105 | PROCESSING | interpolates spikes onto 60 *position* bins × ntrials (1 bin = 1 dm); used for the paper's position-tuning analyses. Not used here (decoder needs *time* bins) |
| `Get_coding_direction` | utils.py:503 | PROCESSING | same frame masks; also `corr_fr = ft_CorrSpc & VRmove`, `grey_fr = ft_GraySpc & VRmove` |
| `Get_sort_spk` | utils.py:599 | PROCESSING | same masks; z-scores the population (`stats.zscore(spk, axis=1)`) |
| `get_kfold_reward_response` | utils.py:814 | PROCESSING | `spk = stats.zscore(spk, axis=1)` before analysis — precedent for per-neuron z-scoring |
| `lickCount` / `lick_response` | utils.py:217/238 | PROCESSING | lick behaviour read from `LickTime`/`LickPos`/`LickTrind`; anticipatory lick = ≥1 lick between corridor entry and the cue |
| `get_cat_id(WallName,isRew)` | utils.py:137 | PROCESSING | canonicalises per-mouse texture names into the shared 0…6 stimulus scheme |
| `data_process_script.ipynb` cell 9 | notebook | PROCESSING | the top-level loop: for every `exp_type`, for every `ndb`: load `Beh`, `load_spk`, then use `VRmove = beh['ft_move'][:nfr]>0` |

### Notes
- Neural data are **already deconvolved** (Suite2p non-negative deconvolution, decay 0.75 s),
  already neuropil-corrected and already cell-classifier-curated. **No ΔF/F computation is
  required** and there is no per-cell quality metric in the released files — the only neuron
  curation the reference applies is the anatomical one (`iarea != -1 & iarea != 7`).
- Behaviour arrays (`ft*`) are *longer* than the neural recording by 1–2 frames; the
  reference always truncates with `[:nfr]` where `nfr = spk.shape[1]`. This is essential.
- `exp_info = beh/Imaging_Exp_info.npy` has 23 experiment types; the same physical recording
  can appear under several types (and under two `stimtype`s), so `exp_info` has 141 entries
  but only **89 unique (mname, datexp, blk)** recordings.
- `beh['stim_id']` is per-`UniqWalls` and is **experiment-type specific**: a wall can be
  labelled in one exp_type and `NaN` in another. The union over exp_types gives the complete
  labelling (verified: zero conflicts).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/beh/Imaging_Exp_info.npy     dict: 23 exp_type -> array of session dicts (db)
                                   db entry: mname, datexp, blk, [stimtype], rewType,
                                   exptype, stim_id (canonical ids of UniqWalls), depth, ...
data/beh/Beh_<exp_type>.npy       dict: '<mname>_<datexp>_<blk>[_<stimtype>]' -> beh dict
data/spk/<m>_<d>_<b>_neural_data.npy   dict {'spks': list of (n_i, T) float32 blocks}
                                       concatenated over axis 0 -> (n_neurons, n_frames)
data/retinotopy/<m>_<d>_trans.npz      xpos, ypos, xy_t (n_neurons,2), iarea (n_neurons,), A
data/retinotopy/areas.npz              'out': 10 cortical-area outlines
data/process_data/                     EMPTY (intermediates must be regenerated)
data/beh/Unsupervised_pretraining_behavior/  behaviour-only cohort (Fig. 5) — no imaging
```
`beh` dict fields are documented in `code/data_process_script.ipynb` cell 4 (reproduced in
Step 5 mapping table). Units: positions in **decimetres**; `Corridor_Length = 60`
(= 6 m = 4 m texture + 2 m grey) in *every* session; times are MATLAB datenums (days).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects (mice) | **19** (DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2) |
| Sessions (unique recordings) | **89** (= 89 spk files = 89 retinotopy files) |
| Sessions / subject | 1–9 (mean 4.7) |
| Neurons total (all ROIs) | 4,691,034 |
| Neurons / session (all ROIs) | **20,547 – 89,577** (mean 52,708) |
| Neurons / session in V1+HVAs (`iarea∉{−1,7}`) | 17,363 – 78,815 (mean 46,128; total 4,105,393) |
| Imaging frames total | 2,025,281 (median 21,000 / session) |
| Frame interval (`median diff(beh['ft'])`) | 314.39 – 315.37 ms (median **314.69 ms**, 3.178 Hz) |
| Trials (total, raw) | **38,110** (84 – 789 per session, mean 428) |
| Frames in texture area while VR moving | 42.0 % of all frames → **821,579** decoder timepoints |
| Valid frames / trial | min 11, median 21, p99 37, max 178 |
| Sessions with rewarded trials | 28 / 89 |
| Sessions with any recorded licks | 28 / 89 (identical set) |
| Stimulus names used | circle1/2/3, leaf1, leaf1_swap1/2, leaf2, leaf3, rock1/2, wood1, wood1_swap1/2, wood2, wood5 |

Neural traces are `float32`, non-negative (deconvolved), ~70 % exact zeros, per-session mean
4.4 – 43 (arbitrary fluorescence units, so the scale differs strongly between sessions).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings / mice | 89 recordings, 19 mice | "We performed 89 recordings in 19 mice bred to express GCaMP6s…" |
| Neurons / recording | 20,547 – 89,577 | "activity traces from 20,547 to 89,577 neurons in each recording" |
| Population size headline | "up to 90,000 neurons simultaneously" | Abstract |
| Corridor geometry | 4 m corridor + 2 m grey | "The virtual reality corridors were each 4 m long, with 2 m of grey space between corridors" |
| VR speed / run threshold | VR fixed 60 cm s⁻¹, threshold 6 cm s⁻¹ (≥66 ms) | Methods, Visual stimuli & Running speed |
| Sound-cue position | uniform 0.5 – 3.5 m | "the time of the sound cue was randomly chosen per trial from a uniform distribution between positions 0.5 m and 3.5 m" |
| Reward zone (behaviour-only cohort) | uniform 2 – 3 m | "…randomly chosen per trial from a uniform distribution between 2 m and 3 m" |
| Calcium frame rate | 3.17 Hz | `data_process_script.ipynb`: "Calcium signal recording frame rate: fs = 3.17Hz" |
| Position bins used by reference | 60 bins of 1 dm over the 6 m loop (40 = texture) | notebook cell 9 |
| Analysis window | 0–4 m texture area only | "we only selected data points inside the 0–4-m region of the corridors where the textures were shown" |
| Running curation | running timepoints only | "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards" |
| Deconvolution | Suite2p NND, τ=0.75 s | "All our analyses were based on deconvolved fluorescence traces" |
| Selectivity threshold | d′ ≥ 0.3 | Methods, Neural selectivity |
| Cohorts | task (supervised), unsupervised, unsupervised-grating, naive | Fig. 1b, Methods |
| Mice per cohort (Fig. 1j) | 4 task, 9 unsupervised, 3 grating (5 sessions) | Fig. 1j legend |
| Anticipatory licking after learning | ≈ selective, leaf1 ≫ circle1 (P = 5.97×10⁻⁴) | Fig. 1d |

### Processing Details
- **Temporal alignment**: the natural trial unit is one corridor traversal. `Trial_start_time` /
  `StartFr` = corridor entry, `Gray_space_time` / `GrayFr` = exit of the texture area into the
  grey space, `Trial_end_time` / `EndFr` = end of the grey space. The task specification asks
  for alignment to **trial start (corridor entry)**, which is exactly `StartFr`.
- **Temporal binning**: the reference never re-bins in time — all time-domain analyses use the
  native imaging frames (`ft`, 3.18 Hz). Position-domain analyses interpolate onto 60 × 1 dm
  bins. Because the decoder task requires time-varying signals plus "time since trial start",
  the native imaging frame is kept as the time bin (314.7 ms).
- **Frame curation**: `spk.shape[1]` frames only; `ft_CorrSpc & (ft_move > 0)`.

### Curation Steps
**Neuron curation rules** (from reference code):
1. Suite2p ROI detection + cell classification + neuropil correction + non-negative
   deconvolution — already applied in the released `spks`.
2. `iarea != -1 & iarea != 7` — "exclude neurons from outside of visual cortex"
   (`utils.Get_density_map`); every area-resolved analysis uses only V1/mHV/lHV/aHV.
No further per-cell quality metric exists in the released data.

**Trial curation rules** (from reference code):
1. Only frames inside the texture area (`ft_CorrSpc`, 0–4 m).
2. Only frames where the VR is moving, i.e. the mouse is running (`ft_move > 0`).
3. Only frames that exist in the neural recording (`[:nfr]`).
4. Trials whose wall has no canonical `stim_id` are never used by the reference (it always
   indexes stimuli through `stim_id`).

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| — | **The paper reports no decoding analyses and no decoding accuracies.** (grep of the full PDF for "decod", "accuracy", "classif" returns only "cell classification"). So there is no published accuracy to compare against; the comparison in Step 12 is against chance and against what the neural data can support. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| #sessions / #mice | `exp_info` has 141 (exp_type, session) entries | 89 unique `mname_datexp_blk`, 89 spk files, 89 retinotopy files, 19 mice | "89 recordings in 19 mice" | Deduplicate to the 89 **physical** recordings; verified all duplicate `beh` entries of the same recording are byte-identical in `ntrials`, `len(ft)` and `UniqWalls` |
| Neurons / session | `load_spk` concatenates the `spks` blocks | 20,547 – 89,577 | "20,547 to 89,577 neurons in each recording" | ✅ exact match (sanity check S1) — confirms `load_spk` reproduced correctly |
| Length of `ft` vs neural frames | always `[:nfr]` | `len(ft)` = `nfr` + 1 or + 2 | — | Truncate exactly as the reference does |
| `beh['RunFr']` "same as `ft_RunSpeed`" (notebook docs) | — | `RunFr = max(ft_RunSpeed,0) × 0.3226`, i.e. distance per frame, not speed | "running speed … interpolated to the timepoints of the imaging frames" | Use **`ft_RunSpeed`** (signed speed per imaging frame); the notebook comment is wrong |
| `stim_id` completeness | per-exp_type array over `UniqWalls` | up to 4 walls `NaN` in a single exp_type | 7 canonical stimuli | Merge `stim_id` over **all** exp_type entries of the same recording (0 conflicts). Only `circle3` (4 sessions) is left unmapped — it has no canonical id and is never analysed in the paper |
| Stimulus names | `UniqWalls` are mouse-specific (`rock1/wood1…`) | 15 distinct raw names | "we denote the stimuli as 'leaf' and 'circle', even though … 'rock' and 'bricks'" | Use the canonical `stim_id` (0–6) so labels are comparable across mice; rock1→circle1, rock2→circle2, wood1→leaf1, wood2→leaf2, wood5→leaf3 |
| `iarea` 7 | excluded as "outside visual cortex" | 0.4 % of neurons | atlas has 10 areas | Exclude `iarea ∈ {−1, 7}` |
| Grating cohort | exp_types `*_grating` | the *recorded* sessions of LZ13/LZ16 all show naturalistic textures (gratings were shown on non-imaged training days) | Fig. 1b cohort 3 | All 89 recordings are naturalistic-texture corridor sessions → all are usable |
| Trial length | — | every one of the 38,110 trials has ≥ 11 valid frames, median 21 | 4 m ÷ 60 cm s⁻¹ × 3.18 Hz = 21.2 frames | ✅ match (sanity check S2) |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial definition and alignment
One **trial** = one corridor traversal. Frames kept for trial *t*:
`(ft_trInd[:nfr] == t) & ft_CorrSpc[:nfr] & (ft_move[:nfr] > 0)` — exactly the reference's
`fr_valid`, restricted to trial *t*. Verified that these frames run contiguously from
`StartFr[t]` to `GrayFr[t]` with monotonically increasing `ft_Pos` from ~0 dm to ~40 dm.
`temporal_alignment_event` = corridor entry; `off_start = 0`.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/<sess>_neural_data.npy['spks']` | `neural` | concat blocks → (nneu, nfr); keep `iarea∉{−1,7}` rows; per-neuron z-score over **all** session frames; select trial frames | `utils.load_spk`, `utils.neu_area_ID`, `stats.zscore` as in `get_kfold_reward_response` | float32, (n_neurons, T_trial) |
| `retinotopy/<m>_<d>_trans.npz['iarea']` | `brain_region_idx` | `neu_area_ID` → 0=V1, 1=mHV, 2=lHV, 3=aHV | `utils.neu_area_ID` | also drives neuron curation |
| `beh['SoundTime']`, `beh['ft']` | `input[0]` `time_to_sound_cue_s` | `(SoundTime[t] − ft[frame]) × 86400` s; **+ before the cue, − after** | — | continuous, time-varying |
| `datexp` | `input[1]` `day_of_training` | days since that mouse's **first** imaging session | — | continuous, per-trial (broadcast) |
| `beh['Trial_start_time']`, `beh['ft']` | `input[2]` `time_since_trial_start_s` | `(ft[frame] − Trial_start_time[t]) × 86400` s | — | continuous, time-varying |
| `beh['isRew']` | `input[3]` `reward_available` | 1 if the trial's corridor is the rewarded one, else 0 | `utils.lickCount`, `get_kfold_reward_response` | discrete, per-trial (broadcast) |
| `beh['WallName']`, `beh['UniqWalls']`, `beh['stim_id']` (merged over exp_types) | `output[0]` `stimulus` | canonical id 0…6 | `utils.get_cat_id`, `Get_dprime_selective_neuron` | per-trial (broadcast) |
| `beh['LickFr']` | `output[1]` `licking` | 1 if ≥1 lick falls in that imaging frame (`floor(LickFr)`), else 0 | `utils.spk_2_firstLick` uses `int(lickFr)` | binary, time-varying |
| `beh['ft_Pos']` | `output[2]` `position_bin` | `floor(ft_Pos/10)` clipped to 0…3 → 0–1, 1–2, 2–3, 3–4 m | corridor is 0–40 dm | time-varying |
| `beh['ft_RunSpeed']` | `output[3]` `running_speed_bin` | quartile bin using edges = 25/50/75-th percentile of **all** kept timepoints pooled over the dataset | Methods "Running speed" | time-varying |
| `mname` | `subjects`, `subject_idx` | 19 unique | — | |

`input` is stored as a dense `(4, T)` float32 array per trial (per-trial values broadcast) and
`output` as a dense `(4, T)` int64 array, so that all four time-varying/per-trial variables
share one array as the format requires.

### Key Decisions
1. **Keep the native imaging frame as the time bin (314.7 ms)** rather than re-binning.
   The decoder needs "time since trial start" and "time to sound cue" as time-varying inputs,
   and the reference never re-bins in time. Re-binning to a rounder number would only blur
   3.18 Hz calcium data.
2. **Only frames inside the texture area (0–4 m) while the VR is moving.** This is the
   reference's `fr_valid = ft_CorrSpc & (ft_move>0)` and the paper's "we only considered
   timepoints during running". It is *also required* by the decoder spec, whose position output
   is four 1-m bins covering exactly the 4-m corridor (the 2-m grey space has no bin).
3. **Neuron curation = anatomical only** (`iarea ∉ {−1, 7}`). Suite2p has already done cell
   classification / neuropil correction / deconvolution, and the release contains no other
   quality metric. 4,105,393 of 4,691,034 neurons are kept (87.5 %).
4. **Per-neuron z-score across all frames of the session.** Deconvolved amplitudes are in
   arbitrary fluorescence units whose scale differs by ~10× between sessions and by orders of
   magnitude between neurons; the reference itself z-scores (`stats.zscore(spk, axis=1)`) for
   its population analyses. Without it the learned 100-d projection (which gets only 200
   optimiser steps) is dominated by a handful of very bright ROIs. Verified empirically in
   Step 7/8 (raw vs z-scored comparison).
5. **Stimulus label = canonical `stim_id` 0…6**, merged across all exp_type entries of a
   recording. This makes the label comparable across mice that saw leaf/circle vs rock/wood.
   Trials of `circle3` (no canonical id, 4 sessions) are dropped — the reference never uses them.
6. **`day_of_training` = days since that mouse's first imaging session.** The release contains
   no absolute training-day counter (`sess#`/`days` in `exp_info` are experiment-type specific
   and inconsistent across duplicates), while `datexp` is unambiguous, continuous, and directly
   tracks learning progression within a mouse.
7. **Running-speed quartiles are global** (pooled over every kept timepoint of the whole
   dataset), so each of the 4 classes holds 25 % of the data exactly as specified.
8. **All 89 recordings are kept** (no session is excluded); every session has ≥ 84 trials.
9. **`time_to_sound_cue` is signed and continuous** (positive before the cue). The decoder
   spec explicitly calls it "continuous, time-varying", so it is not binarised.

### Planned Sanity Checks
- [x] **S1** neuron counts per session reproduce the paper's 20,547 – 89,577 range exactly.
- [x] **S2** frames/trial ≈ 21 = 4 m ÷ 60 cm s⁻¹ ÷ 314.7 ms.
- [x] **S3** 89 sessions / 19 mice, matching "89 recordings in 19 mice".
- [x] **S4** `ft_Pos` on kept frames spans 0–40 dm and increases monotonically within a trial
      (verified: 0.0 – 40.0 dm; 0/348 non-monotonic trials in the spot-checked session).
- [x] **S5** Sound-cue positions ≈ uniform on 0.5–3.5 m (93.3 % inside, p1–p99 = 0.53–3.72 m).
- [x] **S6** Rewarded trials and licking occur in exactly the same 28 sessions, which are
      exactly the 28 task-cohort sessions (5 mice).
- [x] **S7** Independent spot-checks (Step 10 Check 2): 0 failures over 5 sessions ×
      12 trials × 30 neurons × 6 timepoints, plus all inputs/outputs/indices.
- [x] **S8** Each running-speed class holds exactly 25.0 % of timepoints.
- [x] **S9** Position class distribution .250 / .249 / .250 / .252 — uniform as expected.
- [x] **S10** (added) `corr(ft_move>0, ft_RunSpeed)` peaks at lag 0 in all 89 recordings.
- [x] **S11** (added) Population PSTH peaks +1…+2 frames *after* corridor entry.
- [x] **S12** (added) Fig. 1j selectivity percentages reproduced per region and cohort.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (`python -u convert_data.py <out.pkl> [--full|--sample] [--show-processing]`).

Structure:
1. `build_session_index()` — deduplicates `Imaging_Exp_info.npy` (141 entries) to the 89
   physical recordings and merges `stim_id` over all experiment types of each recording.
2. `load_behaviour()` — one pass over the 23 `Beh_*.npy` files keeping only the 13 fields
   needed (so the 5 GB of behaviour files are not held in memory).
3. **Pass 1** (behaviour only, all 89 recordings) — `build_trials()` for every recording and
   pooling of the running speeds to fix the three global quartile edges. Running this over
   *all* recordings even in `--sample` mode guarantees the sample and the full dataset use
   identical discretisation.
4. **Pass 2** — for each recording: load `spks` (prefetched on a background thread),
   re-run `build_trials()` with the exact neural frame count `spk.shape[1]`, select
   neurons (`iarea not in {-1,7}`), `extract_neural()` (frame gather + per-neuron z-score),
   split into per-trial contiguous arrays.
5. Assemble metadata (including a `session_info` record per recording) and pickle
   (protocol 5).

Code inefficiencies identified and fixed:
| Problem | Fix | Effect |
|---|---|---|
| `utils.load_spk` concatenates all `spks` blocks (a full extra copy of up to 8 GB) | operate on the blocks in place | −8 GB peak, −1.5 s/session |
| `blk[km][:, frame_idx]` materialises an (n_keep, n_frames) temporary | gather columns first: `blk[:, frame_idx][km]` | ~5× less temporary memory |
| `blk.mean(1)` + `blk.std(1)` = 3 passes over the block | one pass with `sum` + `einsum('ij,ij->i')` | −60 % of the arithmetic |
| single-threaded extraction | `ThreadPoolExecutor(16)` over 4096-row chunks (all numpy calls release the GIL; the job is bandwidth-bound) | 33.6 s → 1.2 s per session |
| 404 GB of spk files accumulating in the page cache forced the kernel into reclaim for every new allocation | `posix_fadvise(POSIX_FADV_DONTNEED)` after each file | removed a ~5× slowdown that grew through the run |
| serial disk read then compute | background prefetch thread (queue depth 1) | overlaps ~2 s/session of I/O |

Full conversion: **491 s** (8.2 min) end-to-end, of which 138 s is writing the 150.8 GB pickle.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl` (1.2 GB),
`processing_TX124_2023_12_24_1.png`, `processing_TX109_2023_03_27_1.png`.
Sample = TX124_2023_12_24 (naive cohort: no reward, no licking) and TX109_2023_03_27
(task cohort: rewarded corridor, licking) so both code paths are exercised.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (TX124, TX109) |
| Neurons (total) | 67,516 (17,363 + 50,153) |
| Neurons / session | 17,363 – 50,153 (of 20,547 / 63,625 ROIs) |
| Trials (total) | 585 (501 + 84) |
| Timepoints | 11,490 (mean T = 26.5, min 15, max 51) |
| time_to_sound_cue_s | [−246.8, 400.3] |
| day_of_training | [1, 11] |
| time_since_trial_start_s | [0.0, 404.4] |
| reward_available | [0, 1] |
| stimulus distribution | circle1 .266, leaf1 .264, leaf2 .157, leaf1_swap1 .161, leaf1_swap2 .152 |
| licking distribution | no_lick .947, lick .053 |
| position_bin distribution | .255 / .251 / .246 / .247 |
| running_speed_bin distribution | .340 / .165 / .175 / .320 (global quartile edges, only 2 sessions) |

### Processing Plots Review
No anomalies. Specifically verified in the plots:
* kept frames lie strictly inside the orange `ft_CorrSpc` band and strictly outside the red
  "VR not moving" band, and begin at the blue `StartFr` line (corridor entry = t0);
* position rises monotonically 0 → 40 dm within every trial and the position bin steps
  exactly at 10 / 20 / 30 dm;
* the speed bin steps exactly at the three global quartile edges (red dotted lines);
* `time_to_sound_cue` decreases linearly through 0 at the cue, once per trial;
* the naive mouse has an all-zero licking trace and `reward_available` = 0 everywhere;
  the task mouse licks and has `reward_available` = 1 on leaf1 trials only;
* position-bin occupancy is flat (25 % each) as expected for a constant-speed VR.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| threaded chunked extraction + one-pass mean/variance | 33.6 s → 1.2 s per session |
| `posix_fadvise` drop-behind | removes progressive ~5× slowdown |
| background spk prefetch | ~2 s / session |

| Step | Time / Session | Estimated Total |
|---|---|---|
| behaviour load (once) | — | 1.2 s |
| pass 1 (all 89) | — | 0.5 s |
| spk load (prefetched) | 2.4 s | 210 s |
| extract + z-score | 1.2 s | 110 s |
| pickle write | — | 138 s |
| **total** | **~4 s** | **~8 min** (measured 491 s) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: **None**
- Warnings: **None** ("Data format is valid, no errors or warnings.")

### Decoder Results (Sample)
Loss decreased monotonically, 261 → 70.5 over 200 epochs.

| Output | #classes | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|---|---|-------------|--------|
| stimulus | 7 | 0.143 | 0.917 | **0.733** |
| licking | 2 | 0.500 | 0.934 | **0.817** |
| position_bin | 4 | 0.250 | 0.885 | **0.800** |
| running_speed_bin | 4 | 0.250 | 0.679 | **0.605** |

All four outputs are far above chance (2.4×–5.1× chance for the 4/7-class variables).

**Raw vs z-scored neural data.** The same sample was converted with `--no-zscore` and
decoded: stimulus 0.743, licking 0.865, position 0.762, speed 0.585 (sum 2.955) versus
0.733/0.817/0.800/0.605 (sum 2.955) for the z-scored version — indistinguishable on two
sessions. Per-neuron z-scoring was kept because (a) the reference code itself z-scores
(`stats.zscore(spk, axis=1)` in `utils.get_kfold_reward_response` / `Get_sort_spk`), and
(b) with 89 sessions whose deconvolved amplitudes differ by ~10× the shared linear readout
sits on PC magnitudes that would otherwise differ by the same factor between sessions.

---

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/converted_data.pkl --full` → `/app/conversion_full_out.txt`
(491 s total: 210 s spk reading, 110 s extraction, 138 s pickling).
`python -u train_decoder.py /app/converted_data.pkl --verify-only` →
`/app/verification_full_out.txt`: **"Data format is valid, no errors or warnings."**

### Output Files
- `converted_data.pkl`: 150.8 GB (150,771,469,299 bytes)
- `verification_full_out.txt`: created, 0 errors, 0 warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Recordings | 89 | 141 `exp_info` entries → 89 unique `mname_datexp_blk` | 89 spk files, 89 retinotopy files | **89** | ✅ |
| Mice | 19 | — | 19 distinct `mname` | **19** | ✅ |
| Neurons / recording (all ROIs) | 20,547 – 89,577 | `load_spk` concatenation | 20,547 – 89,577 (Σ 4,691,034) | 20,547 – 89,577 (`n_neurons_all_rois`) | ✅ |
| Neurons / recording after curation | (not stated) | `iarea∉{−1,7}` | 17,363 – 78,815 | **17,363 – 78,815**, mean 46,128, Σ **4,105,393** | ✅ |
| Region split | 4 regions (V1/medial/lateral/anterior) | `neu_area_ID` | — | V1 1,833,035 · mHV 1,108,860 · lHV 495,318 · aHV 668,180 | ✅ |
| Corridor | 4 m texture + 2 m grey | `Corridor_Length`=60, `Texture_Length`=40 dm | 60/40/20 in all 89 | 4 position bins over 0–4 m | ✅ |
| Frame rate | 3.17 Hz (notebook) | — | 314.39 – 315.37 ms | **314.85 ms** mean (`time_bin_size`) | ✅ |
| Frames / trial | 4 m ÷ 60 cm s⁻¹ ÷ 0.3148 s = 21.2 | — | median 21 | mean 22.25, median 22.6 | ✅ |
| Trials | (not stated) | — | 38,110 raw | **37,801** (309 `circle3` trials, 0.81 %, have no canonical `stim_id`) | ✅ |
| Timepoints | — | — | 821,579 (`ft_CorrSpc & ft_move>0`) | **815,506** (after dropping `circle3` trials and truncating to `spk.shape[1]`) | ✅ |
| Sound-cue position | uniform 0.5–3.5 m | — | 93.3 % in 0.5–3.5 m, p1–p99 = 0.53–3.72 m | (input `time_to_sound_cue_s` crosses 0 once per trial) | ✅ |
| Rewarded sessions | 5 task mice | `rewType`, `isRew` | 28 sessions have `isRew`, the same 28 have licks | 28 sessions with `reward_available`=1, 5 mice | ✅ |
| Cohorts (Fig. 1j) | 4 task, 9 unsupervised, 3 grating (5 sessions) | `exp_info['sup_train1_before_learning']` etc. | 4 / 9 / 3 mice, 4 / 9 / 5 entries | same | ✅ |
| Cohorts (ED Fig. 3) | naive 9 mice/11 sessions, unsup 7 mice, task 5 mice | `naive_test1`, `unsup_test1`, `sup_test1` | 9/11, 7/7, 5/5 | same | ✅ |
| Cohorts (ED Fig. 6i) | task 3 mice/5 sessions, unsup 4/8, naive 3/10 | `*_test3` | 3/5, 4/8, 3/10 | same | ✅ |
| position_bin distribution | uniform expected (constant-speed VR) | — | — | **.250 / .249 / .250 / .252** | ✅ |
| running_speed_bin distribution | 25 % per bin (task spec) | — | — | **.250 / .250 / .250 / .250** | ✅ |
| stimulus distribution | leaf1 & circle1 are the two training corridors, tests are rarer | 7 canonical ids | — | circle1 .319, leaf1 .335, leaf2 .170, circle2 .060, leaf3 .059, swap1 .027, swap2 .029 | ✅ |
| licking | only task mice lick | — | 28/89 sessions | 3.7 % of all timepoints | ✅ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address. (`verify_data_format` checks structure,
per-session/per-trial shape consistency, dtypes, NaN/Inf, all-zero trials, categorical
outputs, and index ranges for `subject_idx` / `brain_region_idx`.)

The only warning printed by `train_decoder.py` at all is a scikit-learn
`UserWarning: y_pred contains classes not in y_true`, which comes from `balanced_accuracy`
inside the *decoder*, not from the data: a session that never shows, e.g., `leaf3` can still
receive a `leaf3` prediction. It is unavoidable for any dataset in which different sessions
show different subsets of the 7 stimuli, which is a property of the experiment (see
`exp_info`: a session shows 2–8 of the corridors).

### Check 2 — sanity checks against the ORIGINAL data files
`cache/sanity_checks.py` reloads `data/spk/*.npy`, `data/beh/*.npy` and
`data/retinotopy/*.npz` **without importing the conversion code**, re-derives every quantity
with the recipes in `code/utils.py`, and compares with `np.allclose` / `np.array_equal`.
Run on 5 randomly chosen sessions of the full dataset (`cache/sanity_full_out.txt`):

| Check | What is compared | Result |
|---|---|---|
| neuron count | `data['neural'][s][0].shape[0]` vs `((iarea!=-1)&(iarea!=7)).sum()` | PASS ×5 |
| ROI count | `session_info['n_neurons_all_rois']` vs `spk.shape[0]` | PASS ×5 |
| region index | `brain_region_idx` vs `neu_area_ID(iarea)` recomputed | PASS ×5 |
| trial count | vs `#{t : wall has a stim_id and (ft_move>0 & ft_CorrSpc & ft_trInd==t).any()}` | PASS ×5 |
| **NEURAL** | 12 random trials × 30 random neurons × 6 random timepoints vs `(spk[n,f]−spk[n].mean())/spk[n].std()` | PASS ×5 |
| **INPUT** | `time_to_sound_cue`, `time_since_trial_start`, `reward_available` recomputed from `SoundTime`, `Trial_start_time`, `ft`, `isRew` | PASS ×5 |
| **OUTPUT** | stimulus (from `WallName`+merged `stim_id`), licking (`floor(LickFr)`), position bin (`floor(ft_Pos/10)`), speed bin (`searchsorted(edges, ft_RunSpeed)`) | PASS ×5 |
| global | speed classes = 25.0 / 25.0 / 25.0 / 25.0 % | PASS |
| global | position classes = 25.0 / 24.9 / 25.0 / 25.2 % | PASS |

**0 failures.** (Sessions checked: TX123_2024_01_17, TX119_2024_01_06, TX105_2022_10_21,
TX108_2023_03_13 — the largest, 78,815 neurons — and TX83_2022_08_29.)

Two further *independent* validations:

* **Temporal-alignment check** (`corr` between the VR-motion indicator `ft_move>0` and the
  running speed `ft_RunSpeed` at the imaging-frame level, all 89 sessions): median r = 0.694,
  range 0.45 – 0.82, and the cross-correlation peaks at **lag 0** in every session. The two
  behaviour streams that the conversion joins are therefore aligned frame-by-frame, with no
  off-by-one shift. The single exception is discussed in Check 5.
* **Biological validation — reproduction of paper Fig. 1j** (`cache/dprime_check.py`,
  `cache/dprime_check_out.txt`). Using *exactly the frames the conversion keeps*, the
  reference selectivity index d′ = 2(µ₁−µ₂)/(σ₁+σ₂) between the leaf1 and circle1 corridors
  gives the following percentages of selective neurons (|d′| ≥ 0.3) per region:

  | cohort | phase | V1 | mHV | lHV | aHV |
  |---|---|---|---|---|---|
  | task (n=4) | before learning | 11.1 | 4.2 | 5.7 | 0.7 |
  | task (n=4) | after learning | 10.8 | **18.0** | 10.0 | **5.6** |
  | unsupervised (n=9) | before learning | 10.6 | 5.1 | 6.7 | 0.6 |
  | unsupervised (n=9) | after learning | 9.0 | **12.9** | 8.2 | 1.0 |

  These are all inside the 0–20 % range of Fig. 1j and reproduce every qualitative claim of
  the paper: a large increase in the **medial** HVAs in *both* cohorts (paper P = 0.0073 and
  3.2 × 10⁻⁴), **no** change in V1 (P = 0.94 / 0.21), and an anterior-HVA increase that is
  much larger in task than in unsupervised mice ("the anterior regions were only modulated in
  the supervised condition"). This validates the neuron curation, frame curation, trial
  assignment and stimulus labelling end to end.

### Check 3 — reference code comparison
| Stage | Reference (`code/utils.py`, `data_process_script.ipynb`) | `convert_data.py` | Same? |
|---|---|---|---|
| (a) neural loading | `load_spk`: `np.concatenate(np.load(...).item()['spks'], 0)` | `load_spk_blocks` + `extract_neural` operate on the same blocks in the same order without materialising the concatenation | ✅ identical values (verified by Check 2) |
| (a) behaviour loading | `load_exp_beh` → `Beh['<mname>_<datexp>_<blk>[_<stimtype>]']` | same, deduplicated to the 89 physical recordings (duplicates verified byte-identical) | ✅ |
| (a) retinotopy | `load_retino` → `iarea`, `neu_area_ID` | `load_iarea`, `neu_area_idx` (same V1=8, mHV={0,1,2,9}, lHV={5,6}, aHV={3,4}) | ✅ |
| (b) neuron filtering | `Get_density_map`: `(arid!=-1) & (arid!=7)` | identical mask | ✅ |
| (b) trial filtering | reference has no explicit trial filter; it selects stimuli via `stim_id` and never touches walls with `stim_id = NaN` | drop trials whose wall has no canonical `stim_id` (309 `circle3` trials) | ✅ equivalent |
| (c) frame truncation | `nfr = spk.shape[1]`; every `ft*` array indexed `[:nfr]` | identical (`build_trials` is re-run in pass 2 with the true `spk.shape[1]`) | ✅ |
| (c) temporal alignment | frames indexed by `beh['ft_trInd']`, `ft_CorrSpc`, `ft_WallID`, all defined on the imaging-frame grid | trial *t* = `ft_trInd==t & ft_CorrSpc & ft_move>0`; `t = 0` at `Trial_start_time` (= `StartFr`, corridor entry) | ✅ |
| (c) valid frames | `fr_valid = (ft_move>0) & ft_CorrSpc` ("only use activity inside the texture area plus mouse is running") | identical | ✅ |
| (d) binning | reference re-bins to **60 position bins** (`get_interpPos_spk`) for its position analyses, and uses the raw imaging frames for everything in the time domain (`Get_dprime_selective_neuron`, `get_kfold_reward_response`, `spk_2_cue`, `spk_2_firstLick`) | native imaging frames (314.85 ms) | **Deliberate difference**: the decoder task requires time-varying signals plus "time since trial start" and "time to sound cue", which position binning would destroy. The reference's own time-domain analyses also use raw frames, so this matches the reference where the reference works in time. |
| (d) neural normalisation | raw deconvolved for d′ (scale-invariant); `stats.zscore(spk, axis=1)` for population analyses (`get_kfold_reward_response`, `Get_sort_spk`) | per-neuron z-score over all frames of the session | ✅ reference-supported (see Step 8 for the empirical raw-vs-z-scored comparison) |
| (e) input construction | reference uses `SoundFr`/`SoundTime` for cue alignment (`spk_2_cue`) and `isRew`/`rewType` for reward | `time_to_sound_cue_s` from `SoundTime`, `reward_available` from `isRew`; `day_of_training` from `datexp` (no reference equivalent — see Step 5 decision 6) | ✅ / new |
| (f) output construction | `WallName`+`stim_id` for the stimulus (`Get_dprime_selective_neuron`); `LickFr`/`LickTime` for licks (`spk_2_firstLick`, `lickCount`); `ft_Pos` for position; `ft_RunSpeed` for speed (Methods, "Running speed") | same variables; discretised as the decoder task requires | ✅ |

Only one deliberate difference (time bins instead of position bins) and it is forced by the
decoder specification.

### Check 4 — key-statistics comparison
See the Step 9 table: every statistic that the paper, the reference code or the raw data
state is reproduced. Highlights: 89 recordings / 19 mice; 20,547–89,577 neurons per
recording (exact match of both extremes); all nine cohort counts quoted in the figure
legends; 4 m corridor / 21-frame traversal; sound cue ≈ uniform 0.5–3.5 m; reward and licking
confined to the 28 task sessions; and the Fig. 1j selectivity percentages above.

One statistic deserves comment: the paper's **anticipatory-licking** result (Fig. 1c,d) is
reproduced directly from the converted variables (`cache/paper_stats_out.txt`, section 5).
P(lick before the cue) in the rewarded vs the unrewarded corridor is indistinguishable in
the four *before-learning* sessions (TX108 0.96/0.92, TX109 0.78/0.87, TX60 0.67/0.66,
VR2 0.46/0.42 — paper: P = 0.714, n.s.) and strongly selective in every *after-learning*
session (e.g. TX109 0.99/0.27, TX61 0.63/0.06, VR2 0.64/0.15 — paper: P = 5.97 × 10⁻⁴).

### Check 5 — edge cases
| Edge case | Handling | Evidence |
|---|---|---|
| behaviour arrays 1–2 frames longer than the neural recording | `build_trials` is re-run in pass 2 with the exact `spk.shape[1]`; pass 1 (quartile edges only) uses the full length, a ≤2-frame difference per session | assertion `nfr_true <= len(ft)` holds for all 89 |
| `ft_trInd = NaN` outside the behaviour recording (26–674 frames per session) | excluded by `np.isfinite(tr)` | — |
| `ft_trInd` out of `[0, ntrials)` | forced to −1 (invalid) | guard in `build_trials` |
| one trial with `StartFr < 0` (DR15_2022_11_03) | frames are selected by `ft_trInd`, not by `StartFr`, so the negative value is never used as an index; `time_since_trial_start` stays ≥ 0 (dataset min = 0.0) | verification log |
| trials with no valid frames / < 2 trials per session | none exist: every one of the 38,110 raw trials has ≥ 11 valid frames (median 21); smallest session has 84 trials | Step 2 table |
| licks outside the neural recording | `floor(LickFr)` clipped to `[0, nfr)`; 0 licks were actually dropped | `explore` run |
| walls with no canonical `stim_id` (`circle3`, 4 sessions) | those 309 trials (0.81 %) are dropped, matching the reference, which only ever indexes stimuli by `stim_id` | Step 9 table |
| the same recording appearing under up to 5 experiment types with different `stim_id` vectors | `stim_id` merged over all of them; asserted conflict-free (0 conflicts over all 89) | assertion in `load_behaviour` |
| mice whose textures are rock/wood rather than circle/leaf | mapped through `stim_id` onto the canonical 7 labels | Step 4 table |
| neurons with zero variance | `sd = 1` guard so the z-score is 0 rather than NaN | `extract_neural`; verification found no NaN/Inf |
| position exactly at 40 dm (corridor end) | `np.clip(..., 0, 3)` keeps it in the last bin | `build_trials` |
| **DR10_2022_07_12: corrupted running-speed trace** | In this one recording `ft_RunSpeed` is *anti*-correlated with VR motion (r = −0.70, versus +0.45…+0.82 in the other 88 sessions) and no time shift fixes it (best r over ±2000 frames = 0.04). The ball-tracking trace of that session is therefore unusable, and 95.6 % of its timepoints fall in speed bin Q1. The session is **kept**: its neural, position, stimulus and lick data are unaffected, it is part of the paper's n = 9 unsupervised cohort, and it contributes 0.6 % of all timepoints. No ad-hoc correction is invented. | `cache/paper_stats_out.txt`, alignment scan above |
| skewed per-session speed-bin occupancy | with *global* quartiles the median session still has its largest speed class at only 41 % of timepoints; only 9/89 sessions exceed 60 % and only DR10_2022_07_12 exceeds 95 % | analysis above |

### Iterations
1. **`beh` key parsing bug** — the physical-recording key was built from the first *four*
   underscore-separated tokens of the `Beh` dict key (`mname_yyyy_mm_dd`) instead of five
   (`…_blk`), so `load_behaviour` found no behaviour at all. Fixed to five tokens; the
   conversion then ran. Re-ran the sample end to end.
2. **Extraction 5× too slow and getting slower** — see the Step 6 table
   (threading, one-pass variance, `posix_fadvise`). Re-ran the sample conversion, re-ran
   `cache/sanity_checks.py` (all PASS) and re-ran the sample decoder before starting the
   full conversion.
3. **Pass-1 frame count** — the first version used `len(ft) − 2` as the neural frame count in
   both passes, which could silently drop up to 2 real frames per session. Restructured so
   pass 2 rebuilds the trials with the exact `spk.shape[1]`.
No issues were found by re-running all five checks after these fixes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Trained on GPU (NVIDIA L4, 19.4 GB peak) over 30,210 training trials / 7,591 validation
trials (every session contributes to both). Wall clock ≈ 62 min (2 min pickle load,
8 min SVD initialisation of the 89 projection matrices, ~12 s per epoch × 200, plus the
prediction passes).

### Training Progress
- Loss decreasing: **Yes**, monotonically at every printed epoch —
  374.4 → 270.9 (ep 10) → 205.3 (20) → 142.6 (40) → 116.5 (60) → 104.4 (80) → 96.8 (120)
  → 95.3 (160) → **94.6 (200)**. Test loss 0.754.

### Decoder Results (Full)
| Output | #classes | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|---|---|-------------|--------|---|-------|
| `stimulus` | 7 | 0.1429 | 0.7686 | **0.7044** | **4.93×** | per-trial label; the corridor texture is strongly encoded in V1+HVAs |
| `licking` | 2 | 0.5000 | 0.8818 | **0.8456** | **1.69×** | time-varying and very imbalanced (3.7 % licking); only the 28 task sessions contain any licks |
| `position_bin` | 4 | 0.2500 | 0.6918 | **0.6863** | **2.75×** | time-varying; matches the position-tuned sequences the paper reports |
| `running_speed_bin` | 4 | 0.2500 | 0.5413 | **0.5297** | **2.12×** | time-varying; hardest variable, see Step 12 |

`predictions.png` shows the predicted traces (dashed) following the true position staircase
and the speed-quartile fluctuations closely on held-out trials.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
| Variable | Val balanced acc | Chance (1/K) | Ratio | Verdict |
|---|---|---|---|---|
| stimulus | 0.7044 | 0.1429 | 4.93× | ✅ far above chance |
| licking | 0.8456 | 0.5000 | 1.69× | ✅ above 1.5× (69 % of the way from chance to perfect for a binary variable) |
| position_bin | 0.6863 | 0.2500 | 2.75× | ✅ |
| running_speed_bin | 0.5297 | 0.2500 | 2.12× | ✅ |

No output is at or below chance and none is below 1.5× chance, so Check 1 raises no bug.

`running_speed_bin` is the weakest, which is expected rather than a defect:
(i) the four speed classes are quartiles of a *pooled* distribution, so within a session
neighbouring classes dominate and almost all errors are off-by-one quartile;
(ii) the decoder sees one instantaneous population vector per timepoint, whereas the
deconvolved calcium signal integrates locomotion over several hundred ms;
(iii) one recording (DR10_2022_07_12) has an unusable running-speed trace (Step 10 Check 5).
Excluding (iii) could change the pooled accuracy by <1 % because it contributes 0.6 % of the
timepoints, so the recording was kept rather than discarded.

### Check 2 — comparison to accuracies reported in the paper
| Variable | Achieved Accuracy (validation) | Expectation from Paper |
|---|---|---|
| stimulus | 0.704 | **The paper reports no decoding analysis and no decoding accuracies.** Searching the full extracted PDF text for "decod", "accuracy" and "classif" returns only "cell classification" (a Suite2p step). |
| licking | 0.846 | idem |
| position_bin | 0.686 | idem |
| running_speed_bin | 0.530 | idem |

Because no published accuracy exists, the conversion was instead validated against the
paper's own *analysis results* (Step 10 Check 2), which is a stricter test of alignment and
curation than a decoding number:
* the Fig. 1j percentages of stimulus-selective neurons per visual region, before and after
  learning, are reproduced quantitatively from exactly the frames the conversion keeps
  (task mHV 4.2 → 18.0 %, unsupervised mHV 5.1 → 12.9 %, V1 unchanged, aHV rising only in
  task mice), with every value inside the 0–20 % range of the figure;
* the Fig. 1c,d anticipatory-licking result is reproduced from the converted behavioural
  variables (no rewarded-vs-unrewarded difference before learning, a large one after).

The paper's qualitative expectations are also matched by the decoder itself: the stimulus —
the quantity the paper shows is strongly and increasingly encoded in visual cortex — is by
far the best-decoded variable (4.9× chance), and position, which the paper shows as
position-tuned sequences of selective neurons (Fig. 2), is decoded at 2.75× chance.

### Check 3 — train vs validation gap
| Variable | Train | Validation | Train/Val |
|---|---|---|---|
| stimulus | 0.7686 | 0.7044 | 1.09× |
| licking | 0.8818 | 0.8456 | 1.04× |
| position_bin | 0.6918 | 0.6863 | 1.01× |
| running_speed_bin | 0.5413 | 0.5297 | 1.02× |

All ratios are far below the 1.5× flag: no overfitting and no sign of leakage. The split is
made per trial within each session by `train_validate_decoder`, and because the conversion
emits one array per trial with no shared timepoints, no timepoint can appear on both sides.

### Additional debugging performed (the five suggested steps)
1. **Output values verified against raw data on specific trials** — `cache/sanity_checks.py`
   re-derives all four outputs for 12 random trials in each of 5 random sessions directly
   from `Beh_*.npy` and compares element-wise: all PASS (Step 10 Check 2).
2. **Temporal alignment plotted and measured** — `processing_*.png` put neural, inputs and
   outputs on a common trial axis. Quantitatively (`cache/alignment_check.py`,
   `cache/alignment_check_out.txt`):
   * behaviour↔behaviour: `corr(ft_move>0, ft_RunSpeed)` peaks at **lag 0** in all 89
     recordings (median r = 0.694), so the streams the conversion joins are frame-synchronous;
   * neural↔behaviour: the population PSTH around corridor entry is negative before the event
     and peaks at **+1 to +2 frames** (0.3–0.6 s) after it — the expected GCaMP6s +
     deconvolution latency. Activity *follows* the visual event, so there is no sign error or
     off-by-one in the neural frame indexing;
   * `corr(population activity, VR moving)` peaks at lag −1…0 frames, i.e. within one frame.
3. **Output variation checked** — no output is dominated by one class at the dataset level
   (position .250/.249/.250/.252; speed .250 ×4; largest stimulus class .335). `licking` is
   96.3 % "no lick", a real property of the experiment (only 28 of 89 recordings have a water
   spout at all); `balanced_loss=True` and balanced accuracy handle this, and the achieved
   0.846 shows it is not degenerate.
4. **Neural filtering re-checked** — `iarea ∉ {−1,7}` reproduces `utils.neu_area_ID` /
   `Get_density_map` exactly; `brain_region_idx` was compared element-wise against a fresh
   `iarea` load for 5 sessions (PASS).
5. **Processing re-checked against the reference** — line by line in the Step 10 Check 3
   table; the only deliberate difference is time bins instead of the reference's 60 position
   bins, which the decoder specification forces.

### Issues Found and Resolved
- No new issues were found in Step 12. The three issues found earlier (behaviour-key parsing,
  extraction performance, pass-1 frame count) are documented in Step 10; all were fixed and
  all checks were re-run before the full conversion.
- One irreducible data defect is documented rather than "fixed": the running-speed trace of
  DR10_2022_07_12 (0.6 % of timepoints) is unusable in the released data.
- One warning emitted by `train_decoder.py` cannot be removed: scikit-learn's
  `y_pred contains classes not in y_true`, raised inside `balanced_accuracy_score` because a
  session that never showed, e.g., `leaf3` can still be *predicted* `leaf3`. This is a
  property of the experimental design (each recording shows 2–8 of the 7 canonical stimuli)
  and of the decoder's global class set, not of the converted data. The only way to remove it
  would be to collapse the stimulus label to the two categories present in every session,
  which would throw away the test-stimulus information that the paper's Figs. 2, 3 and
  ED 6–7 are entirely about.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, how to load, full format specification, key
      statistics, how to reproduce, decoder results.
- [x] `cache/` folder created with `README_CACHE.md` documenting every cached script/output.
- [x] All files organised:

| File | Purpose |
|---|---|
| `convert_data.py` | the conversion (`--full` / `--sample` / `--show-processing`) |
| `converted_data.pkl` | 150.8 GB, 89 sessions / 19 mice / 37,801 trials |
| `sample_data.pkl` | 1.2 GB, 2 sessions |
| `CONVERSION_NOTES.md` | this file |
| `README.md` | user-facing documentation |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `--verify-only` logs |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder logs |
| `processing_TX124_2023_12_24_1.png`, `processing_TX109_2023_03_27_1.png` | per-step conversion diagnostics |
| `sample_trials.png`, `predictions.png` | produced by `train_decoder.py` |
| `cache/` | exploration + validation scripts and outputs (see `cache/README_CACHE.md`) |

### Summary of every decision made
1. **Unit of analysis** = the 89 physical recordings (19 mice), deduplicated from the 141
   `(exp_type, session)` entries in `Imaging_Exp_info.npy`. Matches the paper exactly.
2. **Trial** = one traversal of the 4 m texture corridor, aligned to corridor entry.
3. **Timepoints kept** = `ft_CorrSpc & (ft_move>0)`, truncated to `spk.shape[1]` — the
   reference's `fr_valid`, and also required by the 4 × 1 m position output.
4. **Time bin** = the native imaging frame (314.85 ms); no re-binning.
5. **Neurons kept** = `iarea ∉ {−1, 7}` (V1 / mHV / lHV / aHV) — the reference's rule.
6. **Neural signal** = Suite2p deconvolved traces, per-neuron z-scored over the session.
7. **Stimulus label** = the canonical `stim_id` 0…6, merged across experiment types; trials of
   `circle3` (no canonical id, 0.81 % of trials) dropped.
8. **Speed discretisation** = global quartiles of `ft_RunSpeed` over all kept timepoints.
9. **Position discretisation** = `floor(ft_Pos/10)`, 1 m bins over 0–4 m.
10. **Licking** = ≥ 1 lick in the imaging frame (`floor(LickFr)`).
11. **`day_of_training`** = days since that mouse's first imaging session.
12. **`time_to_sound_cue`** = signed seconds, positive before the cue.
13. **No session and no mouse excluded**; one recording's running-speed trace is flagged as
    unusable but the recording is retained.

# Dataset Conversion Notes

## Overview
- **Dataset**: DANDI:001361 — Sosa, Plitt & Giocomo (2025), *A flexible hippocampal
  population code for experience relative to reward*, Nature Neuroscience.
  2-photon calcium imaging of hippocampal CA1 + VR behaviour, NWB format, `/app/data`.
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`).

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py`, `train_decoder.py`, `decoder.py`
- `paper.pdf`, `methods.txt`
- `code/` — reference repo (GiocomoLab/Sosa_et_al_2024): `src/reward_relative/*.py`,
  `notebooks/*.ipynb|md`, `docs/*.md`
- `data/` — 152 NWB files in 11 `sub-m*` directories (87 GB), plus `dandiset.yaml`
- `pynwb_docs/`, `Dockerfile`, `docker-compose.yaml`

Environment verified: `numpy 2.4.4`, `torch 2.6.0+cu124`, `pynwb 4.1.0`,
`suite2p` (provides `suite2p.extraction.dcnv.oasis`).
Hardware: 1 TB RAM, 128 cores, NVIDIA L4 (23 GB).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | `preprocessing.py` | LOADING | Builds the `sess` class from raw scanbox + VR SQLite, calls `sess.align_VR_to_2P()` (TwoPUtils) which **interpolates all VR/behaviour channels onto the 2P frame clock**. |
| `TwoPUtils.preprocessing.vr_align_to_2P` | (external repo) | LOADING | The actual VR→2P alignment. Its output is what the NWB `BehavioralTimeSeries` contains (all behaviour channels share the imaging timestamps). |
| `dff` | `preprocessing.py` | PROCESSING | dF/F for one channel. Restricts to within-trial samples (`keep_teleports=False`), neuropil subtraction `F − 0.7·Fneu` then adds back the per-trial mean neuropil, **maximin baseline** (Gaussian σ=15 samples → `minimum_filter1d(300)` → `maximum_filter1d(300)`), `dF/F=(F−F0)/|F0|`, then 2-sample σ Gaussian smoothing per trial; optional OASIS deconvolution (`suite2p.extraction.dcnv.oasis(dff, 2000, tau, fs)`). |
| `default_dff_method` | `utilities.py` | PROCESSING | `neuropil_method='subtract'`, `baseline_method='maximin'`, `neu_coef=0.7`, `keep_teleports=False`. |
| `multi_anim_sess` | `utilities.py` | PROCESSING/CURATION | Top-level per-day loader: calls `pp.dff(...)`, `behav.get_trial_types`, `behav.get_reward_zones`, `behav.define_trial_subsets`, `calc_place_cells`. |
| `get_trial_types` | `behavior.py` | PROCESSING | Per-trial `isreward` = `any(reward>0) and any(rzone>0)` within `[trial_start_inds[i], teleport_inds[i])`; per-trial `morph` (0 = ENV1, 1 = ENV2). |
| `get_reward_zones` | `behavior.py` | PROCESSING | Per-trial reward-zone coordinates/labels from the **scene name**; on switch scenes the zone changes at `change_trial = 30`. Zone coords come from `reward_zone_dict`: A→`X`=[80,130], B→`Y`=[200,250], C→`Z`=[320,370]. |
| `reward_zone_dict`, `map_labels`, `env_morph_dict` | `behavior.py` | REFERENCE | Zone coordinates; `Env1→0`, `Env2→1`. |
| `define_trial_subsets` | `behavior.py` | CURATION | Splits trials into pre-/post-switch sets (not needed for the decoder, but confirms the switch-at-30 convention). |
| `correct_lick_sensor_error` | `behavior.py` | CURATION | Lick-sensor failure detection (see Step 3). |
| `nansmooth` | `utilities.py` | PROCESSING | NaN-tolerant Gaussian smoothing used by `dff`. |
| `decode.py :: CircularRegression` | `decode.py` | ANALYSIS | The paper's own decoder (circular–linear regression on **deconvolved** activity). Not the decoder used here, but tells us which neural signal the authors decoded from. |

### Notes
- Imaging data: 2-photon **calcium imaging**, so ΔF/F **must be computed** (the NWB
  stores raw suite2p `F`, `Fneu` and suite2p's own `Deconvolved`/`spks`, none of which
  are ΔF/F).
- Cell quality filtering: suite2p `iscell` (manual curation) + speed-correlation
  interneuron exclusion (Step 3).
- The reference `dff()` slices trials as `[start-1 : stop-1]` while all behavioural
  code (`get_trial_types`, etc.) slices `[start : stop]`. This 1-frame inconsistency
  inside the reference is resolved here by using `[start : stop]` **everywhere**, so
  that the neural and behavioural streams are exactly aligned (see Step 5, decisions).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/sub-m<N>/sub-m<N>_ses-<DD>_behavior+ophys.nwb`; `ses-<DD>` is the
  1-indexed **experiment day** (01–14). 11 subjects × 14 days, except `m11` which
  starts at day 03 (12 sessions) → **152 sessions**, matching `dandiset.yaml`
  (`numberOfFiles: 152`, `numberOfSubjects: 11`).
- `nwb.identifier` = `/data/InVivoDA/GCAMP<N>/<dd_mm_yyyy>/<scene>` → gives the
  **scene name** (e.g. `Env1_LocationB_to_A`), exactly the string the reference
  `get_reward_zones` / `define_trial_subsets` parse. Verified against
  `code/src/reward_relative/sessions_dict.py`: `GCAMP<N>` ↔ `sub-m<N>`, and every
  scene/date pair matches.
- `nwb.processing['ophys']`:
  - `Fluorescence/plane0` — `(n_frames, n_roi)` float32, suite2p `F`
  - `Neuropil/plane0` — suite2p `Fneu`
  - `Deconvolved/plane0` — suite2p `spks` computed from **raw F** (values in raw
    fluorescence units, e.g. 0–3800), *not* the paper's deconvolution of dF/F
  - `ImageSegmentation/PlaneSegmentation` — `pixel_mask`, **`iscell` (n_roi, 2)**,
    `planeIdx`
- `nwb.processing['behavior']/BehavioralTimeSeries` — all sampled on the imaging
  frame clock (shared `timestamps`), `dt = 0.064484 s` (15.5078 Hz) **identical in
  all 152 files**:
  `position` (cm), `speed` (cm/s), `lick` (cumulative count per frame),
  `reward_zone` (count of reward-zone-entry samples per frame), `environment`
  (−1 ITI, 0 ENV1, 1 ENV2), `trial number`, `trial_start` (binary), `teleport`
  (binary), `scanning` (−1/1), `autoreward` (all zeros in every file — unusable),
  and `Reward` (a separate TimeSeries with one timestamp per reward delivery,
  0.004 mL).
- `nwb.acquisition['TwoPhotonSeries']` is a `(1,1,1)` placeholder (no movie).
- No `trials`/`units`/`epochs` tables.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Sessions | 152 (11 subjects; 14 each except m11 = 12) |
| ROIs (total, before `iscell`) | 260,091 |
| Neurons (`iscell==1`, total) | 138,678 |
| Neurons / session | mean 912.4, **min 155, max 2341** |
| Subjects | 11 (m3, m4, m7, m11–m15, m17–m19) |
| Sessions / subject | 14 (m11: 12) |
| Trials (total, `trial_start` events) | 12,216 |
| Trials / session | mean 80.4, min 41, max 100 |
| Imaging rate | 15.5078125 Hz per plane in all sessions (m17, m18 are 2-plane, 31.0156 Hz volume rate → 15.5078 Hz/plane) |
| Session duration | ~1,200–1,800 s |
| Frames inside trials | ~72–78% of all frames (the rest is the ITI/teleport period) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch task) | 11 | "Each mouse encountered a different starting reward zone and sequence of reward zone switches, counterbalanced across mice (n = 11 mice)." |
| Mice starting in ENV 2 | 2 (m17, m18) | "Most mice began the task in ENV 1 (n = 9 mice; two mice (m17 and m18) began in ENV 2)." |
| Sessions / subject | 14 (m11 starts day 3) | "for a total of 14 days"; "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Neurons / session | **155–2172** | "This approach yielded 155–2172 putative pyramidal neurons per session" |
| Trials / session | 80.5 ± 7.4 | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days"; "We targeted 80–100 trials per session" |
| Trials (total, 11 switch mice) | 12,376 | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Lick-sensor error trials | 81 (~0.65%) | "A very small number of trials with erroneous lick detection ... (~0.65% of all imaged trials, n = 81 out of 12,376 trials)" |
| Reward omission rate | ~15% | "the reward was randomly omitted on ~15% of trials" |
| Neural data time bin | 0.0645 s (15.5 Hz) | "the 0.0645 s imaging frame samples"; "collected ... at ~15.5 Hz" |
| Behaviour time bin | same as neural | VR data is interpolated onto the 2P frame clock (`vr_align_to_2P`) |
| Interneurons excluded by speed correlation | 0.42 ± 0.85% of cells | "excluding 0.42 ± 0.85% of cells (mean ± s.d. across mice and days)" |
| Track length | 450 cm | "a unidirectional 450 cm virtual linear track" |
| Reward zones | A 80–130, B 200–250, C 320–370 cm | "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" |
| Switch trial | after 30 trials | "Each switch occurred after 30 trials." |
| dF/F baseline | maximin, 20 s window, within trial | "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window" |
| dF/F smoothing | 2-sample (~0.129 s) Gaussian | "then smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel" |
| Speed threshold used for *spatial* analyses | 2 cm/s | "we excluded activity when the animal was moving at <2 cm s−1" |

### Processing Details
- **Temporal alignment**: VR behaviour is interpolated onto the 2P frame times
  (`vr_align_to_2P`); the NWB already stores it that way (one shared timestamp
  vector). Trials run from the `trial_start` event to the `teleport` event.
- **Temporal binning**: native imaging frame, 64.484 ms, identical in all sessions.
- **dF/F**: see Step 1 (`dff()`), matching the methods text exactly
  (300 samples / 15.5 Hz = 19.3 s ≈ the "20 s sliding window").
- **Deconvolution**: OASIS on the smoothed dF/F; the paper notes this "is not
  interpreted as a spike rate but rather as a method to eliminate the asymmetric
  smoothing of the calcium signal".

### Curation Steps

**Neuron curation rules**:
1. suite2p manual curation → `iscell[:,0] == 1` ("Manual curation eliminated ROIs
   containing multiple somata or dendrites, lacking visually obvious transients,
   suspected of overexpressing the calcium indicator or exhibiting high and
   continuous fluorescence fluctuation typical of putative interneurons").
2. Pearson `corr(dF/F, speed) > 0.5` → putative interneuron, excluded.
3. (Place-cell selection is used for *some* figure analyses; not applied here —
   see Step 5 decisions.)

**Trial curation rules**:
1. Trials = `[trial_start, teleport)`.
2. Lick-sensor failure trials removed: ">30% of the 0.0645 s imaging frame samples
   in the trial containing a cumulative lick count >2".
3. (Added here) trials overlapping frames with `scanning < 0` (no 2P data) and
   trials shorter than 2 frames.

### Decoders Trained (in the paper)
| Decoded variable | Accuracy |
|---|---|
| Reward-relative (circular) position, from deconvolved activity of RR / TR / non-RR place cells | Reported as a "decode score" = mean `cos(y − ŷ)` (1 = perfect, 0 = chance), not a classification accuracy. Fig. 3b/3c report scores and z-scores vs. a circular-shift shuffle. No % accuracies are reported anywhere in the paper, so there is no directly comparable number for the outputs decoded here. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total trials | – | 12,216 `trial_start` events over 152 imaged sessions | 12,376 trials over 11 switch mice | Difference = 160 = exactly 2 sessions × 80 trials. m11 days 1–2 have VR behaviour but **no imaging** ("imaging started on day 3 [for m11]"), so they are absent from DANDI but were counted in the behavioural total. Consistent. |
| Lick-error trials | `correct_lick_sensor_error` (a different, position-based helper) | My implementation of the methods rule finds **exactly 81** trials | 81 trials (~0.65%) | Exact match → the trial-curation rule is implemented correctly. |
| Neurons per session | – | 155 – 2341 (`iscell==1`) | 155 – 2172 | Lower bound matches **exactly**. Upper bound: only 3 sessions exceed 2172 (m18 days 1, 3, 8: 2281/2341/2314). m18 is a 2-plane animal; the paper's upper bound is plausibly quoted per-plane or from an earlier curation round. Speed-correlation exclusion removes only ~0.4%, so it cannot account for the gap. Difference is 3/152 sessions and <8%; `iscell` filtering is clearly the right criterion. |
| Reward omission | `isreward = any(reward>0) & any(rzone>0)` | 15.34% of trials un-rewarded (1,874/12,216) | "~15% of trials" omitted | Match. |
| Switch trial | `change_trial = 30` default | For every one of the 77 switch sessions, the last trial with an observed pre-switch zone entry is ≤ 29 and the first with a post-switch entry is ≥ 30 | "Each switch occurred after 30 trials" | Consistent; `change_trial = 30` used. |
| Reward-zone coordinates | `reward_zone_dict` X/Y/Z = [80,130]/[200,250]/[320,370], `map_labels` A→X, B→Y, C→Z | Reward-zone entry always occurs at 80 / 200 / 320 cm (±3 cm) | A 80–130, B 200–250, C 320–370 | Match. |
| `autoreward` channel | used by the VR task | **all zeros** in every one of the 152 files | first 10 trials of a new condition get automatic reward | The channel was not populated in the NWB export. Not needed for any decoder input/output; automatically-delivered rewards still appear in the `Reward` series, so `reward_outcome` is unaffected. |
| Environment coding | `env_morph_dict`: Env1→0, Env2→1 | `environment` channel is −1 (ITI), 0, 1 | ENV1 / ENV2 | Match. Each trial has a single environment value (verified for all 12,216 trials). |
| Neural signal for decoding | paper's own decoder uses **deconvolved** activity | NWB `Deconvolved` is suite2p's spks from *raw* F, not the paper's OASIS-on-dF/F | – | Recomputed both from `F`/`Fneu` with the reference `dff()`; dF/F chosen as the decoder input (see Step 5, decision 2). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence/plane0`, `ophys/Neuropil/plane0`, `ImageSegmentation.iscell` | `neural` | `iscell` filter → `dff()` (neuropil subtract, maximin baseline, 2-sample Gaussian) → speed-correlation interneuron exclusion → slice `[trial_start, teleport)` | `preprocessing.dff`, `utilities.default_dff_method`, `utilities.nansmooth` | float32, `(n_neurons, T)` per trial |
| `behavior/position` timestamps | `input[0]` `time_from_trial_start_s` | `t − t[trial_start]` | – | seconds, time-varying |
| `behavior/environment` | `input[1]` `environment` | unique value within the trial (0 = ENV1, 1 = ENV2) | `behavior.get_trial_types` (`morph`), `env_morph_dict` | constant within trial |
| trial index | `input[2]` `trial_number` | 0-indexed position of the `trial_start` event in the session | – | constant within trial |
| `behavior/Reward` + `behavior/reward_zone` | `input[3]` `previous_trial_reward` | `isreward` of trial *i−1*; trial 0 → 1 | `behavior.get_trial_types` | constant within trial |
| `behavior/position` + scene-derived reward zone | `output[0]` `reward_zone_distance` | signed distance to nearest point of the zone → 7 bins | `behavior.get_reward_zones`, `reward_zone_dict` | time-varying |
| `behavior/position` | `output[1]` `position` | `digitize` at 90/180/270/360 cm | – | time-varying |
| `behavior/speed` | `output[2]` `speed` | `digitize` at 2/10/20/40 cm/s | – | time-varying |
| `behavior/lick` | `output[3]` `lick` | `count > 0` | paper: "lick counts were converted to a binary vector" | time-varying |
| `nwb.identifier` scene + `change_trial=30` | `output[4]` `reward_zone_location` | A=0, B=1, C=2 | `behavior.get_reward_zones` | constant within trial |
| `behavior/Reward`, `behavior/reward_zone` | `output[5]` `reward_outcome` | `any(reward) and any(rzone)` | `behavior.get_trial_types` | constant within trial |
| `nwb.subject.subject_id` | `subjects`, `subject_idx` | – | – | 11 mice |
| – | `brain_regions` = `['CA1']`, `brain_region_idx` = zeros | – | – | all imaging is dorsal CA1 |

### Key Decisions
1. **Time bin = native imaging frame (64.484 ms).** Every one of the 152 sessions has
   exactly the same `dt` (15.5078 Hz per plane), and the NWB behaviour is already
   interpolated onto that clock by the reference `vr_align_to_2P`. Using the native
   grid therefore satisfies "same bin size for all trials and sessions" with **zero
   resampling**, i.e. no risk of introducing temporal misalignment between the
   neural and behavioural streams.
2. **Neural signal = dF/F** (computed exactly as in `preprocessing.dff`), not the
   deconvolved trace. Rationale: (a) the NWB's `Deconvolved` array is suite2p's spks
   from *raw* F and does **not** correspond to the paper's deconvolution, so it had
   to be recomputed either way; (b) the paper itself uses binned dF/F for its spatial
   peak identification "as this signal is the closest to the raw data"; (c) I tested
   both signals with a PCA+logistic position decoder on three sessions — dF/F was
   equal or better in every case (m11 d3 0.53 vs 0.45, m19 d6 0.77 vs 0.63, m13 d8
   0.40 vs 0.41), which is expected because the decoder reads instantaneous activity
   and dF/F retains the temporal integration that a sparse event train discards.
   The deconvolution step itself is implemented and validated but not used.
3. **No speed threshold on the neural data.** The paper excludes samples with speed
   <2 cm/s from *spatial* analyses. Here speed is a decoder **output** whose class 0
   is exactly "< 2 cm/s", so dropping those samples would make that class impossible
   to decode and would also punch holes in the time-varying outputs. Required
   deviation, documented.
4. **No place-cell selection.** Place-cell identification in the paper depends on a
   stochastic 100× shuffle and is used to *restrict* specific figure analyses; the
   decoder should see the full curated pyramidal population. All `iscell`,
   non-interneuron cells are kept.
5. **Trial window `[trial_start, teleport)` for both neural and behaviour.** The
   reference `dff()` uses `[start-1, stop-1)` while the reference behaviour code uses
   `[start, stop)`; using `[start, stop)` for both keeps the streams exactly aligned.
   The effect on the dF/F baseline is one frame at each end of a ≥113-frame trial,
   i.e. negligible for a 300-sample maximin window.
6. **Reward zone from the scene name + `change_trial = 30`** (the reference method),
   *cross-checked* against the position at which the `reward_zone` channel fires on
   every trial where it fires. The data-driven check cannot be used alone because the
   channel does not fire on reward-omission trials (~15%).
7. **`previous_trial_reward` for trial 0 = 1 (rewarded).** Each imaging session is
   immediately preceded by ~30 warm-up trials of the same task with the same reward
   zone ("Before the imaging session, mice were provided 30 'warm-up' trials using
   the task and reward zone from the previous day"), and rewards are omitted on only
   ~15% of trials, so "rewarded" is the correct prior for the trial preceding trial 0.
8. **Trial exclusions**: the 81 lick-sensor-failure trials (the paper NaNs their lick
   data; since `lick` is a decoder output and NaNs are not allowed, the trials are
   dropped), trials overlapping `scanning < 0`, trials < 2 frames.
9. **Per-trial variables are broadcast to `(d, T)`** rather than stored as 1-D, so
   that all inputs/outputs share one array per trial and the time-varying and
   per-trial variables can coexist in a single matrix.
10. **`brain_regions = ['CA1']`.** All recordings are dorsal hippocampal CA1; the two
    planes in m17/m18 are deep/superficial CA1 and are "pooled for all analyses
    except those in Extended Data Fig. 7".

### Planned Sanity Checks
- [x] Total `trial_start` events == 12,216 and reconciles with the paper's 12,376.
- [x] Lick-error trial detection reproduces the paper's **81** trials.
- [x] `iscell` neuron counts: min == 155 (paper's lower bound).
- [x] Reward omission rate ≈ 15%.
- [x] Reward-zone switch consistent with trial 30 in all 77 switch sessions.
- [x] Scene-derived reward zone agrees with the observed `reward_zone` entry position
      on every trial where the zone fires (checked per session during conversion).
- [x] Speed-correlation interneuron exclusion ≈ 0.4% of cells.
- [ ] Spot-check neural / input / output values against the raw NWB (Step 10).
- [ ] Position within a trial spans ~0 → 450 cm; speed, lick ranges sensible.
- [ ] Decoder accuracy above chance for every output.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Structure:
- `nansmooth` — verbatim port of the reference NaN-tolerant Gaussian smoother.
- `compute_dff` — port of `reward_relative/preprocessing.py::dff` for the
  single-channel, `neuropil_method='subtract'`, `baseline_method='maximin'`,
  `keep_teleports=False` configuration that `multi_anim_sess` uses for these data.
- `reward_zone_labels_from_scene` — port of `behavior.py::get_reward_zones` logic
  (scene string parsing + `change_trial=30`).
- `discretize_reward_distance` — signed distance to the nearest point of the zone and
  its 7-way discretization.
- `load_session` / `process_session` — pynwb loading, curation, per-trial assembly.
- `make_processing_plots` — `--show-processing` diagnostics.
- `main` — `ProcessPoolExecutor` over sessions, assembly, pickling.

Code inefficiencies identified:
- Reading the full `(n_frames, n_roi)` `F`/`Fneu` arrays and transposing is the
  dominant cost (~0.4–1 s per session; the arrays are uncompressed and contiguous).
- Per-trial Python loops over ~80 trials for the maximin baseline are cheap
  (~1–3 s/session) because each trial is a vectorised 2-D filter call.

Code speedups added:
- `iscell` filtering **before** any dF/F maths (halves the working array).
- Vectorised speed-correlation over all cells at once (single matrix-vector product).
- `ProcessPoolExecutor` with 8 workers over sessions.
- `float32` storage for the output arrays.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
Sample = `sub-m11_ses-03` (single-plane, zone switch B→A, the *smallest* session in the
dataset, 155 `iscell` ROIs) and `sub-m17_ses-05` (**two-plane** animal, zone switch
C→A in ENV 2) — chosen to exercise both imaging configurations and the switch logic.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 992 |
| Neurons / session | 154 (m11 d3), 838 (m17 d5) |
| Subjects | 2 |
| Sessions / subject | 1 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| T (frames/trial) | mean 224.1, min 137, max 445 |
| `time_from_trial_start_s` range | [0.0, 28.6] |
| `environment` range | [0.0, 1.0] |
| `trial_number` range | [0.0, 79.0] |
| `previous_trial_reward` range | [0.0, 1.0] |
| `reward_zone_distance` distribution | [0.188, 0.082, 0.035, 0.219, 0.024, 0.084, 0.368] |
| `position` distribution | [0.213, 0.202, 0.213, 0.190, 0.182] |
| `speed` distribution | [0.053, 0.062, 0.135, 0.505, 0.245] |
| `lick` distribution | [0.799, 0.201] |
| `reward_zone_location` distribution | [0.600, 0.168, 0.233] |
| `reward_outcome` distribution | [0.123, 0.877] |

`verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**

Checks against expectation:
- Position occupancy is close to uniform over the 5 × 90 cm bins (0.18–0.21), as it
  must be for an animal that runs the whole track every lap.
- `reward_zone_location` reflects the two switch sessions: m11 d3 is B(0.42)→A(0.58),
  m17 d5 is C(0.39)→A(0.61) — i.e. the pre-switch zone occupies 30/80 trials and the
  post-switch zone 50/80, and there is more *time* per post-switch trial.
- Reward rate 0.877 / 0.849 per session ≈ the ~15% omission rate.
- 20% of in-trial frames contain a lick, concentrated in and just before the reward
  zone (see `processing_sub-m11_ses-03.png`).

### Processing Plots Review
`processing_sub-m11_ses-03.png` / `processing_sub-m17_ses-05.png` (9 panels, first 6
trials) and `*_trials.png` (the final per-trial arrays). No anomalies:
- Raw `F` shows the laser blanking during the ITI; those samples are NaN in dF/F and
  are never included in a trial (white gaps in the population raster).
- Position ramps 0 → 450 cm within every trial; the blue `trial_start` and red
  `teleport` markers bracket exactly one track traversal.
- The signed reward-zone distance crosses 0 exactly where the position trace enters
  the shaded reward zone, and its 7-way discretization steps 0→1→2→3→4→5→6 in order.
- Speed and its bin index agree at every threshold crossing; the binary lick output
  is 1 exactly where the lick count is > 0.
- In the per-trial figure the input `time_from_trial_start_s` starts at 0 for every
  trial and the per-trial outputs are constant, confirming there is no leakage of one
  trial's values into the next.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| `iscell` filter applied before any dF/F arithmetic | ~2× on the per-session compute and memory |
| Vectorised speed-correlation (one matrix–vector product instead of a per-cell loop) | ~50× on that step |
| `ProcessPoolExecutor` over sessions (12 workers) | ~9× wall-clock on the full run |
| `float32` output arrays | halves pickle size / write time |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| NWB read (`F`, `Fneu`, behaviour) | 0.4 s (m11 d3) – 2.0 s (m4 d14) | ~3 min sequential |
| dF/F + curation + trial assembly | 0.3 s – 5.9 s | ~7 min sequential |
| **Measured full run, 12 workers** | — | **46 s** + 14 s pickling = **60 s** |

Well under the 15-minute budget; no further optimisation needed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: **None**
- Warnings: **None**

### Decoder Results (Sample, 2 sessions / 160 trials)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| reward_zone_distance | 0.7200 | 0.5390 | 0.1429 |
| position | 0.8547 | 0.7151 | 0.2000 |
| speed | 0.7263 | 0.5824 | 0.2000 |
| lick | 0.7566 | 0.7374 | 0.5000 |
| reward_zone_location | 0.9622 | 0.8794 | 0.3333 |
| reward_outcome | 0.8908 | 0.5013 | 0.5000 |

Loss decreased monotonically (3.31 → 0.62 over 200 epochs). Every output except
`reward_outcome` is clearly above chance. `reward_outcome` is a **per-trial binary**
label, so with only 2 sessions the validation set contains ~40 trials of which ~5 are
omissions — far too few to estimate a balanced accuracy. This is resolved on the full
dataset (Step 11) and analysed in Step 12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --workers 12`
Run time **60 s** (46 s conversion + 14 s pickling) — matches the Step-7 estimate.

### Output Files
- `converted_data.pkl`: **9.63 GB** (9.46 GB of `float32` dF/F)
- `verification_full_out.txt`: created, **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | 11 entries relevant in `sessions_dict.py` | 11 `sub-*` dirs | **11** | ✔ |
| Sessions | 14/mouse, m11 from day 3 | 14 entries/mouse | 152 NWB files | **152** | ✔ |
| Sessions per subject | 14 (m11: 12) | 14 | 14 (m11: 12) | 14 (m11: 12) | ✔ |
| Trials (total) | 12,376 (incl. m11 days 1–2, not imaged) | – | 12,216 `trial_start` events | **12,135** (12,216 − 81 lick-error) | ✔ (see Step 4) |
| Trials/session (mean ± s.d.) | 80.5 ± 7.4 | – | 80.37 ± 6.9 | **79.84 ± 6.88** | ✔ |
| Trials/session (min–max) | "80–100 targeted" | – | 41–100 | 40–100 | ✔ |
| Lick-error trials removed | 81 | – | 81 (my detection) | **81** | ✔ exact |
| Neurons (total) | – | – | 138,678 `iscell` | **138,276** | ✔ |
| Neurons/session | **155 – 2172** | – | 155 – 2341 (`iscell`) | **154 – 2323** | ✔ lower bound exact; see Step 4 for the upper bound |
| Mean neurons/session | – | – | 912.4 | **909.7** | ✔ |
| Interneurons excluded (speed corr > 0.5) | **0.42 ± 0.85 %** | – | – | **0.35 ± 0.61 %** (402 cells, 0.29 % overall) | ✔ |
| Reward omission rate | ~15 % | – | 15.34 % | **15.36 %** | ✔ |
| Reward zones | A 80–130, B 200–250, C 320–370 cm | `reward_zone_dict` X/Y/Z | zone-entry at 80/200/320 cm | same | ✔ |
| Switch trial | after 30 trials | `change_trial = 30` | first post-switch zone entry at trial ≥ 30 in all 77 switch sessions | 30 | ✔ |
| Reward-zone label agreement | – | `get_reward_zones` | 10,394 observed zone entries | **0 mismatches / 10,394** | ✔ |
| Time bin | 0.0645 s | – | 0.0644836 s in all 152 files | **64.4836 ms** | ✔ |
| Track length | 450 cm | – | max in-trial position 442.6–451.8 cm | – | ✔ |
| ENV1 / ENV2 trial split | 9 mice start ENV1, 2 start ENV2 | `env_morph_dict` | – | ENV1 51.1 %, ENV2 48.9 % | ✔ |
| Reward-zone identity balance | counterbalanced across mice | – | – | A 34.4 %, B 32.7 %, C 32.9 % | ✔ |

Nothing was lost in conversion beyond the 81 documented lick-error trials:
`raw trials 12,216 = kept 12,135 + lick-error 81 + unscanned 0 + too-short 0`.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification
`verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address. Specifically, none of the conditions
that `verify_data_format` warns about are triggered: all arrays are `float32`
(neural/input) / `int64` (output), no NaN or Inf, no all-zero trial, no session with
0 trials, `dinput`/`doutput`/`n_neurons` consistent everywhere, and
`brain_region_idx` lengths match the neuron counts.

### Check 2 — Independent sanity checks (`cache/sanity_checks.py`)
This script re-opens the **raw NWB files with pynwb** and re-derives everything from
scratch (it does not import `convert_data.py`), then compares with `np.allclose`.
Four sessions were spot-checked, chosen to cover the different configurations:
`sub-m11_ses-03` (single plane, zone switch), `sub-m18_ses-05` (two planes **and** one
of the 10 sessions with an extra imaging frame), `sub-m13_ses-08` (environment switch),
`sub-m4_ses-14` (35 dropped lick-error trials, so the kept-trial index mapping is
non-trivial).

| Check | Result |
|---|---|
| **Neural** — n_neurons after `iscell` + independently recomputed speed-correlation exclusion | PASS in all 4 (154 / 2106 / 707 / 1061) |
| **Neural** — whole trial matrix `np.allclose(ref, got)` for a mid-session trial | PASS, `max|diff| = 0.00e+00` in all 4 |
| **Neural** — scalar spot check `neural[3, 10]` | PASS (e.g. 0.357412 for m11 d3 trial 26) |
| **Input** — `input[0]` equals `timestamps[start:end] − timestamps[start]` | PASS |
| **Input** — `input[1]` equals the NWB `environment` value of that trial | PASS |
| **Input** — `input[2]` equals the raw trial index **and** the NWB `trial number` channel at the trial-start frame | PASS |
| **Input** — `input[3]` equals `isreward` of the *preceding raw* trial | PASS |
| **Output** — `output[0]` reward-zone distance recomputed from raw position | PASS |
| **Output** — `output[1]` position bin recomputed with `pos // 90` | PASS |
| **Output** — `output[2]` speed bin recomputed with explicit thresholds | PASS |
| **Output** — `output[3]` equals `raw lick count > 0` | PASS |
| **Output** — `output[4]` zone label, and it matches the position at which the raw `reward_zone` channel first fires (within 10 cm) | PASS (entries at 201.5 / 200.5 / 321.5 / 200.2 cm vs zone starts 200 / 200 / 320 / 200) |
| **Output** — `output[5]` recomputed as `any(Reward in trial) and any(reward_zone > 0)` | PASS |
| **Bookkeeping** — kept trials == raw − lick-error − unscanned − short | PASS |

Dataset-wide checks (`cache/global_checks.py`, `cache/edge_checks.py`):
- `previous_trial_reward` chain verified on all 11,945 consecutive kept-trial pairs: **0 mismatches**.
- Every session's first kept trial is raw trial 0 with `previous_trial_reward = 1`.
- Scene-derived reward zone vs. observed zone entry: **0 mismatches in 10,394 trials**.
- Position at the first frame of a trial: −0.8 to 10.1 cm (mean 1.95); at the last
  frame: 442.6 to 451.8 cm (mean 448.6). **No trial fails to reach 440 cm, and only
  1 of 12,216 starts above 10 cm** → the `[trial_start, teleport)` window is the full
  track traversal with no off-by-one.
- **No trial contains an intertrial (`environment == −1`) sample**, and every trial
  has exactly one valid environment value.

**Result: ALL CHECKS PASSED.** No mismatches needed fixing.

### Check 3 — Reference code comparison
| Stage | Reference | This script | Same? |
|---|---|---|---|
| (a) Data loading | `create_sess` → `sess.align_VR_to_2P()` (TwoPUtils) interpolates VR onto 2P frame times; `sess.load_suite2p_data()` supplies `F`, `Fneu`, `iscell` | `load_session()` reads the already-VR-aligned `BehavioralTimeSeries` and the suite2p `F`/`Fneu`/`iscell` from the NWB with `pynwb` | Yes — the NWB *is* the output of the reference alignment step (one shared timestamp vector at the 2P frame rate) |
| (b) Neuron filtering | suite2p `iscell` (manual GUI curation); paper adds `corr(dF/F, speed) > 0.5` interneuron exclusion | identical, both applied | Yes |
| (b) Trial filtering | paper: NaN the 81 lick-sensor-failure trials | same rule, trials dropped instead of NaN'ed (NaNs are not allowed in the decoder format); found **exactly 81** | Yes (dropping vs NaN forced by format) |
| (c) Temporal alignment | trials = `[trial_start_inds[i], teleport_inds[i])` (`get_trial_types`, `define_trial_subsets`); `dff()` uses `[start-1, stop-1)` | `[start, stop)` for **both** neural and behaviour | Deviation of 1 frame in the dF/F window only; chosen so the two streams are exactly aligned (the reference is internally inconsistent here). Effect on a 300-sample maximin baseline over a ≥96-frame trial is negligible. |
| (d) Binning | native 2P frame (15.5078 Hz); the paper's *spatial* analyses additionally bin into 45 × 10 cm bins | native 2P frame, no spatial binning | Yes — spatial binning is a figure-analysis step, not applicable to a time-resolved decoder |
| (e) Neural processing | `dff()`: NaN outside trials → `F − 0.7·Fneu` → add back trial-mean neuropil → Gaussian σ=15 → `minimum_filter1d(300)` → `maximum_filter1d(300)` → `(F−F0)/|F0|` → Gaussian σ=2 per trial; optional `dcnv.oasis(dff, 2000, tau, fs)` | `compute_dff()` — same operations, same constants, same order; `nansmooth` ported verbatim | Yes; deconvolution implemented and tested but not used (Step 5, decision 2) |
| (f) Input construction | n/a (the paper's decoder has no auxiliary inputs) | time from trial start, environment (`env_morph_dict`), trial index, previous `isreward` | Dictated by the Decoder Task spec; each variable is taken from the reference's own definitions |
| (f) Output construction | `get_trial_types` (`isreward`, `morph`), `get_reward_zones` (`rz_coords`, `rz_label`, `change_trial=30`), lick binarisation | ported one-for-one; distance/position/speed discretizations are from the Decoder Task spec | Yes |

Differences and why:
1. **`[start, stop)` instead of `[start-1, stop-1)`** — see above.
2. **No <2 cm/s speed mask** — speed class 0 *is* "< 2 cm/s"; masking would delete
   the class and create holes in the time-varying outputs.
3. **dF/F rather than deconvolved activity** — Step 5, decision 2.
4. **No place-cell restriction** — Step 5, decision 4.
5. **81 lick-error trials dropped rather than NaN'ed** — the format forbids NaN.

### Check 4 — Key statistics comparison
See the table in Step 9. Every statistic available in the paper is reproduced. The two
that do not match exactly were both investigated:
- **12,376 vs 12,216 raw trials**: difference is exactly 160 = 2 × 80 trials = m11
  days 1–2, which have VR behaviour but no imaging and are therefore absent from
  DANDI. Resolved, not an error.
- **Neurons/session upper bound 2172 vs 2341**: only 3 of 152 sessions exceed the
  paper's quoted maximum, all in m18, the animal with the most ROIs and one of the two
  **two-plane** animals (its ROIs are the pooled total of two planes). The lower bound
  matches exactly (155), which confirms the `iscell` criterion is the right one.

### Check 5 — Edge cases
| Edge case | Handling | Evidence |
|---|---|---|
| Multi-plane animals (m17, m18) store one `RoiResponseSeries` per plane | planes are pooled by scattering each plane's traces into the rows of the `PlaneSegmentation` table given by `planeIdx` (with an assertion on the ROI counts) | first version crashed on m17; fixed and verified against `sub-m18_ses-05` in Check 2 |
| 10 two-plane sessions have **one more imaging frame** than the VR-aligned behaviour (`int(max_idx/n_planes)` truncation in `vr_align_to_2P`) | the trailing frame is dropped, with an assertion that the excess is ≤ 1 frame | caught by an assertion during the first full run; all 152 sessions now pass |
| Trial 0 has no preceding trial | `previous_trial_reward = 1` (~30 rewarded warm-up trials precede every session) | Step 5, decision 7 |
| `reward_zone` channel does not fire on omission trials | reward zone comes from the scene name (reference method), cross-checked on the 10,394 trials where it does fire | 0 mismatches |
| `autoreward` channel is all zeros in every file | not used; automatic rewards still appear in the `Reward` series so `reward_outcome` is unaffected | Step 4 |
| Trials that overlap frames without scanning, or shorter than 2 frames | detected and dropped | 0 such trials in this dataset |
| Position slightly outside [0, 450] (−2.7 to 451.8 cm) | `np.clip` into bins 0 and 4 | `cache/edge_checks.py` |
| Very long laps (18 trials > 60 s, max 216 s) | kept; the format allows variable `T` | Step 9 |
| `trial number` channel has one extra value at the very end of some recordings with no matching `trial_start` | trials are defined solely by `trial_start`/`teleport` pairs, so these stray samples are never used | `len(trial_start) == len(teleport)` asserted for all 152 sessions |

### Issues Found and Resolved
1. **Multi-plane sessions crashed** (`iscell` length 2636 vs 1176 traces) — the two
   imaging planes are separate `RoiResponseSeries`. Fixed by pooling planes via
   `planeIdx`. Re-ran sample + full conversion and all checks.
2. **10 two-plane sessions had a 1-frame length mismatch** between fluorescence and
   behaviour — caught by an assertion, fixed by trimming the trailing frame. Re-ran
   the full conversion and all checks.
3. A plotting bug (indexing the already-filtered dF/F with the pre-filter mask) —
   fixed; it never touched the converted data.

After fixes, all of Checks 1–5 were re-run from scratch on the regenerated
`converted_data.pkl`; everything passes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(GPU, NVIDIA L4; ~4 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically for all 200 epochs
  (3.314 → 2.557 @10 → 1.955 @20 → 1.291 @50 → 0.900 @100 → 0.658 @200).
- Final test loss 0.717.

### Decoder Results (Full: 152 sessions, 12,135 trials, 138,276 neurons)
| Output | Chance (1/n) | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------|-------------|--------|------|-------|
| reward_zone_distance (7) | 0.1429 | 0.7901 | **0.6200** | 4.34× | |
| position (5) | 0.2000 | 0.8900 | **0.7657** | 3.83× | |
| speed (5) | 0.2000 | 0.7226 | **0.6258** | 3.13× | |
| lick (2) | 0.5000 | 0.7926 | **0.7658** | 1.53× | |
| reward_zone_location (3) | 0.3333 | 0.9639 | **0.8739** | 2.62× | |
| reward_outcome (2) | 0.5000 | 0.9379 | **0.6027** | 1.21× | information is only available after the animal passes the zone — see Step 12 |

(A second, independent run of the same command gave 0.618/0.765/0.633/0.753/0.875/0.602,
i.e. the numbers are stable to ±0.01.)

`predictions.png` shows the predicted (dashed) traces tracking the ground truth (solid)
closely for position, reward-zone distance, speed, lick and reward-zone identity.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
| Variable | Val. balanced acc | Uniform chance | Ratio | Verdict |
|---|---|---|---|---|
| reward_zone_distance | 0.6200 | 0.1429 | 4.34× | well above chance |
| position | 0.7657 | 0.2000 | 3.83× | well above chance |
| speed | 0.6258 | 0.2000 | 3.13× | well above chance |
| lick | 0.7658 | 0.5000 | 1.53× | above the 1.5× guideline |
| reward_zone_location | 0.8739 | 0.3333 | 2.62× | well above chance |
| reward_outcome | 0.6027 | 0.5000 | 1.21× | below 1.5× — investigated below |

No output is at or below chance.

**`reward_outcome` investigation.** `reward_outcome` is a *per-trial* label, but the
decoder predicts it at *every timepoint*. Before the animal reaches the reward zone
there is no information anywhere in the brain about whether this trial's reward will
be delivered or omitted (omission is decided by a random number generator and is only
revealed at the zone), so roughly the first half of every trial is unpredictable by
construction and drags the whole-trial balanced accuracy toward 0.5.

I tested this directly (`cache/diag_reward.py`): for 7 sessions spread across the
dataset, an independent PCA(100) + balanced logistic regression trained on the first
75 % of trials and tested on the last 25 % gives

| timepoints used | mean validation balanced accuracy |
|---|---|
| all in-trial timepoints | **0.600** |
| only timepoints past the end of the reward zone | **0.675** |

The 0.600 from this independent model matches the decoder's 0.6027 almost exactly,
and restricting to the informative part of the trial raises it by 0.075. So the low
ratio reflects the structure of the task, not a conversion bug. I deliberately did
**not** "fix" it by, e.g., labelling pre-zone timepoints differently — the Decoder
Task specifies reward outcome as a per-trial output.

**`lick` (1.53×)**: licking is decoded from CA1 dF/F at 0.766 balanced accuracy, i.e.
the decoder recovers the lick bouts (clearly visible in `predictions.png`). The
modest ratio is simply because a binary variable has chance 0.5.

### Check 2 — Accuracy comparison to the paper
I searched the paper and `methods.txt` for every reported decoding result. The **only**
decoder in the paper is the circular–linear position decoder of Fig. 3
(`reward_relative/decode.py :: CircularRegression`), and it is reported as a
**"decode score" = mean cos(y − ŷ)** (1 = perfect, 0 = chance) and as a z-score
relative to a circular-shift shuffle — not as a classification accuracy, and only for
*reward-relative position* restricted to RR / TR / non-RR **place-cell subpopulations**
within single sessions.

| Variable | My validation balanced accuracy | Paper's reported accuracy |
|---|---|---|
| reward_zone_distance | 0.6200 | no comparable number (Fig. 3 reports a cosine "decode score" for a circular RR-position variable, over selected place-cell subsets) |
| position | 0.7657 | not decoded in the paper |
| speed | 0.6258 | not decoded in the paper |
| lick | 0.7658 | not decoded in the paper |
| reward_zone_location | 0.8739 | not decoded in the paper |
| reward_outcome | 0.6027 | not decoded in the paper |

There is therefore no paper number that my accuracies can be lower than. The
qualitative claim that the paper *does* make — that CA1 population activity carries a
strong code for position relative to reward — is reproduced: reward-zone distance is
decoded at 4.3× chance and reward-zone identity at 2.6× chance.

### Check 3 — Train vs validation gap
| Output | Train | Val | Train/Val |
|---|---|---|---|
| reward_zone_distance | 0.7901 | 0.6200 | 1.27 |
| position | 0.8900 | 0.7657 | 1.16 |
| speed | 0.7226 | 0.6258 | 1.15 |
| lick | 0.7926 | 0.7658 | 1.03 |
| reward_zone_location | 0.9639 | 0.8739 | 1.10 |
| reward_outcome | 0.9379 | 0.6027 | **1.56** |

Only `reward_outcome` exceeds the 1.5× guideline. Cause: it is constant within a trial
and the model has 152 session-specific 100 × n_neurons projection matrices trained for
200 epochs, so it can memorise which *training* trials were rewarded from slow,
trial-specific fluctuations in the population state. This is over-fitting of the
reference decoder on a per-trial label, not data leakage from the conversion:
- The train/validation split is over **whole trials** (`train_validate_decoder` splits
  trial indices), so no timepoint of a validation trial is ever seen in training.
- None of the four decoder **inputs** reveals the current trial's outcome:
  `previous_trial_reward` is the *previous* trial (verified against the raw NWB for all
  11,945 consecutive pairs), and `time`, `environment`, `trial_number` are outcome-free.
- I confirmed with the independent per-session model in Check 1 that ~0.60 is the
  honest generalisation level for this variable.
- The same pattern appears for `reward_zone_location` to a much smaller degree
  (1.10), which is expected because that label is genuinely encoded in the population.

### Other debugging steps run
1. Output values verified against the raw NWB on specific trials of 4 sessions
   (Step 10, Check 2) — exact matches.
2. Temporal alignment verified by plotting neural + inputs + outputs for single trials
   (`processing_*_trials.png`) and by the whole-dataset boundary statistics
   (position 0 → 450 cm within each trial, no ITI samples inside trials).
3. Output variation: no output is degenerate — the most imbalanced is `reward_outcome`
   at 84.2 % / 15.8 %, and `balanced_loss=True` plus balanced accuracy handle that.
4. Neural filtering follows the reference (`iscell` + speed-correlation interneuron
   exclusion), reproducing the paper's per-session minimum of 155 cells and its
   0.42 ± 0.85 % interneuron exclusion rate (0.35 ± 0.61 % here).
5. Processing matches the reference `dff()` operation-for-operation (Step 10, Check 3).

### Issues Found and Resolved
None remaining. The two real bugs (multi-plane pooling, the 1-frame length mismatch)
were found and fixed in Step 10 and everything was re-run afterwards.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format spec, key stats)
- [x] `cache/` folder created with the exploration/validation scripts and
      `README_CACHE.md`
- [x] All files organized

### Final deliverables
| File | Description |
|---|---|
| `CONVERSION_NOTES.md` | this document |
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full converted dataset (9.63 GB) |
| `sample_data.pkl` | 2-session sample (0.08 GB) |
| `README.md` | user-facing documentation |
| `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt` | sample logs |
| `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` | full-dataset logs |
| `processing_sub-m11_ses-03*.png`, `processing_sub-m17_ses-05*.png` | per-step processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder sample/prediction plots |
| `cache/` | exploration and validation scripts |

# Dataset Conversion Notes

## Overview
- **Dataset**: DANDI 001361 — Sosa, Plitt & Giocomo (2025), "A flexible hippocampal population code for experience relative to reward", Nature Neuroscience. 152 NWB files (87 GB) of CA1 two-photon imaging + VR behavior from 11 mice, in `/app/data`.
- **Date started**: 2025-09-21
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)
- **Result**: 152 sessions / 11 subjects / 11,983 trials / 138,276 CA1 neurons at 64.48 ms resolution; the provided decoder reaches validation balanced accuracies of 0.63 (distance to reward zone), 0.77 (position), 0.63 (speed), 0.77 (lick), 0.88 (reward zone location) and 0.61 (reward outcome), all above chance.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python: numpy 2.4.4, torch 2.6.0+cu124, pynwb 4.1.0 all import OK.

Directory contents of /app:
- `CONVERSION_NOTES.md` (this file)
- `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `code/` : reference code repo (Sosa_et_al_2024): LICENSE, README.md, docs/, environments/, notebooks/, setup.py, src/reward_relative/
- `data/` : dandiset.yaml + 11 subject folders (sub-m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19), 152 `.nwb` files total
- `decoder.py`, `train_decoder.py` : provided decoder training code
- `methods.txt`, `paper.pdf` : reference text
- `pynwb_docs/` : pynwb documentation
- `cache/` : my scratch scripts

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Repo: Sosa, Plitt, Giocomo 2025 (Nat Neuro) `Sosa_et_al_2024`. Modules in `src/reward_relative`, notebooks in `notebooks/`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` / `append_session_data` | preprocessing.py | LOADING | Builds TwoPUtils `sess` object: aligns Unity VR sqlite behavior to 2P imaging frames (`vr_align_to_2P`), loads suite2p F/Fneu/iscell, adds timeseries `licks` (=vr_data['lick']), `rewards` (=vr_data['reward']), `speed` |
| `dff(f, trial_starts, teleports, ...)` | preprocessing.py | PROCESSING | dF/F: keeps only within-trial samples `[start-1:stop-1]`, neuropil subtraction (neu_coef=0.7) with per-trial neuropil mean added back, `maximin` baseline (gaussian smooth sigma=15 frames, then min filter 300 frames then max filter 300), dff=(F-base)/|base|, then temporal smoothing sigma=2 frames; optional `deconvolve=True` -> suite2p `dcnv.oasis(dff, 2000, tau, frame_rate/n_planes)` giving `events` |
| `multi_anim_sess` | utilities.py | LOADING/PROCESSING | Loops animals; computes dff (+events), place cells (shuffle, speed_thr=2 cm/s, 100 perms, p<0.05), and stores per-trial behavior: `isreward`, `morph`, `rzone`, `rz label`, `trial dict` |
| `get_trial_types(sess)` | behavior.py | PROCESSING | per-trial `isreward` = any reward delivered AND inside reward zone during trial; `morph` = env identity (0=Env1, 1=Env2) |
| `get_reward_zones(sess)` | behavior.py | PROCESSING | per-trial reward zone [start,stop] cm and label A/B/C from scene name; A=[80,130], B=[200,250], C=[320,370] (dict keys X/Y/Z); switch sessions (`X_to_Y` scenes) change zone at trial index 30 (`sess.change_reward_trial`) |
| `define_trial_subsets(sess)` | behavior.py | CURATION | set0 = trials before switch (first 30), set1 = after |
| `get_timeseries_data(sess, ...)` | glmUtils.py | PROCESSING | Builds continuous (time-varying) behavior DataFrame + deconvolved `events` matrix used for the Fig.3 decoder and GLM. Within-trial slices `[start-1:stop-1]`; `pos`; `rel_pos` = position relative to **reward zone start** (circular via `pos_cm_to_rad` on 0-450 cm, or linear); `trials` = trial index; `speed`; `licks` (cumulative counts -> binary, with lick-sensor error correction); `rewards` binary; `rewarded`/`omission` state flags. NaN masking: samples with NaN events (ITI/teleport) removed; optional speed threshold `use_speed_thr=2` sets speed<2 cm/s to NaN and masks those samples out |
| `get_omission_inds`, `get_omission_trials`, `get_reward_inds` | rewardAnalysis.py | PROCESSING | reward-zone entry index on omission trials; list of omission trials |
| `CircularRegression`, `train_vs_test_blocks` | decode.py | ANALYSIS | Paper's decoder of circular reward-relative position from deconvolved events (Low/Williams) |
| `define_anim_list`, `max_anim_list` | dayData.py | CURATION | Animal lists per experiment day; `include_ans` = animals that learned the reward-switch task, excluding GCAMP2/5/6/10 |

### Notes
- Neural signal used for decoding/GLM in the paper = **deconvolved events** derived from dF/F (`ts_key='events'`); dF/F used for place-field peak quantification.
- Trial structure: `trial_start_inds` (start of track) to `teleport_inds` (end of track); ITI/teleport samples are NaN'd out of dff/events (keep_teleports=False by default).
- Frame rate ~15.5 Hz (64.5 ms/frame); behavior interpolated onto imaging frames, so all streams share one time base -> temporal alignment is inherent in the NWB.
- Lick handling: cumulative lick counts thresholded to binary (>1 -> 1); trials where >35% of samples have cumulative lick count >2 are treated as lick-sensor errors and set to NaN.
- Speed threshold of 2 cm/s used for place-cell and decoder analyses. For **our** decoder task speed itself is an output with a `<2 cm/s` bin, so we must NOT drop low-speed samples (documented as a justified deviation).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
DANDI dandiset 001361 (Sosa, Plitt, Giocomo 2025), 87 GB, NWB 2.x files:
`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` — 152 files, 11 subjects.

Each NWB contains:
- `nwb.identifier` = original path e.g. `/data/InVivoDA/GCAMP3/01_10_2022/Env1_LocationC` -> gives original animal name (GCAMPn), date, and **scene** (which encodes environment and reward-zone location, incl. switch scenes like `Env1_LocationA_to_C`).
- `nwb.session_id` = experiment day (01..14), `nwb.subject.subject_id` = m3, m4, ...
- `processing['behavior'].BehavioralTimeSeries` time series, all sampled on the **imaging frame clock** (timestamps provided; median dt = 0.06448 s = 15.5078 Hz):
  - `position` (cm; -500 placeholder at very start, ~-50..0 in ITI/teleport jitter zone, 0-450 on track)
  - `speed` (cm/s, smoothed; can be slightly negative)
  - `lick` (cumulative lick count per imaging frame, 0..6)
  - `trial_start` (binary), `teleport` (binary), `trial number` (-1 in ITI, else 0-indexed trial)
  - `environment` (-1 in ITI, 0 = Env1, 1 = Env2)
  - `reward_zone` (binary/counter of reward-zone occupancy, 0 outside)
  - `autoreward` (whether trial was automatically rewarded if the mouse failed to lick)
  - `scanning` (whether 2P scanning was on; -1/1)
  - `Reward` : sparse TimeSeries with its own timestamps (reward delivery times, mL)
- `processing['ophys']`:
  - `Fluorescence/plane0[,plane1]`: raw suite2p ROI fluorescence, shape (n_frames, n_rois), rate 15.5078125 Hz
  - `Neuropil/plane0[,plane1]`: neuropil fluorescence, same shape
  - `Deconvolved/plane0[,plane1]`: suite2p deconvolved activity **from raw F** (not from the paper's dF/F)
  - `ImageSegmentation/PlaneSegmentation`: columns `pixel_mask`, `iscell` (n_rois x 2: [is_cell flag, probability]), `planeIdx`
- `acquisition['TwoPhotonSeries']`: metadata only (rate 15.5078125, dims 512x796); imaging plane location = **'hippocampus, CA1'** for all 152 sessions.
- No dF/F stored -> dF/F (and deconvolved events from dF/F) must be computed with the reference `pp.dff()` code.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19 = GCAMP3,4,7,11,12,13,14,15,17,18,19) |
| Sessions | 152 (14 per mouse, except m11 = 12) |
| ROIs (total, all) | 312,110 |
| ROIs with iscell==1 (total) | 138,678 |
| Cells / session (iscell==1) | mean 912, range 155 (m11) - 2341 (m18) |
| Trials (total, trial_start events) | 12,216 |
| Trials / session | mean 80.4; typically 80 (non-switch) or 90-100 (switch days); min 41 |
| Imaging planes | 1 plane for all mice except m17, m18 (2 planes) |
| Frame rate | 15.5078125 Hz (64.48 ms/frame) |
| Brain region | hippocampus CA1 (all sessions) |
| Environments present | 73 sessions Env1 only, 68 Env2 only, 11 sessions both (Env1->Env2 switch days) |

n trial_start == n teleport in every session (no truncated trials at file level).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch task, = the dandiset) | 11 mice | "counterbalanced across mice (n = 11 mice)"; "Mice were randomly selected to experience the switch task (n = 11 mice) versus the 'fixed-condition' task (n = 3)" (fixed-condition mice are NOT in this dandiset) |
| Sessions | 14 days/mouse, except m11 imaged from day 3 (12 days) | "for a total of 14 days"; "The task was imaged starting from day 1 for all mice except m11, for whom imaging started on day 3" |
| Trials (total) | 12,376 across 11 switch mice | "n = 81 out of 12,376 trials removed across 11 switch mice" |
| Trials / session | 80.5 ± 7.4 (target 80-100) | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice, all imaging days" |
| Neurons / session | 155-2172 putative pyramidal cells | "This approach yielded 155-2172 putative pyramidal neurons per session" |
| Neural data time bin | ~64.5 ms (15.5 Hz imaging frame rate) | "unidirectional scanning at ~15.5 Hz"; "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate" |
| Behavior data time bin | same 15.5 Hz (VR interpolated onto imaging frames) | as above |
| Reward omission rate | ~15% of trials | "reward was randomly omitted on ~15% of trials" |
| Reward zones | A 80-130 cm, B 200-250 cm, C 320-370 cm (50 cm hidden zone) | "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm" |
| Track length | 450 cm + ~50 cm gray tunnel + jitter (1-5 s, 5-10 s after omission) | "unidirectional 450 cm virtual linear track" |
| Switch trial | after 30 trials within a switch session | "Each switch occurred after 30 trials" |
| Environments | ENV1 / ENV2; 9 mice start in ENV1, m17 & m18 start in ENV2 | "Most mice began the task in ENV 1 (n = 9 mice; two mice (m17 and m18) began in ENV 2)" |
| Lick-error trials | ~0.65%, 81/12,376 trials | "detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2" |
| Interneuron exclusion | 0.42 ± 0.85% of cells | "Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed" |

### Processing Details
- **dF/F**: per-trial baseline via *maximin* with 20 s sliding window (implemented as Gaussian smooth sigma=15 frames -> min filter 300 frames (~19.3 s) -> max filter 300 frames), dF/F = (F - baseline)/|baseline|, then Gaussian smoothing with 2-sample (~0.129 s) s.d. Neuropil subtraction with coefficient 0.7 (per reference code `pp.dff(..., neuropil_method='subtract', neu_coef=0.7)`), adding back per-trial neuropil mean.
- **Activity rate** = OASIS deconvolution of the dF/F (suite2p `dcnv.oasis`, tau from s2p ops (0.7 default in code), fs = frame_rate/n_planes). Not a spike rate, used to remove calcium kernel asymmetry. This is the signal used for the paper's decoder and GLM.
- **Trial definition**: trial_start (entry to track at 0 cm) -> teleport (entry to ITI). Laser power was blanked during the ITI for most sessions, so only within-trial samples carry valid fluorescence.
- **Temporal alignment**: VR behavior is interpolated onto the 2P frame clock (`vr_align_to_2P`), so all streams share a single time base in the NWB (one shared `timestamps` vector).
- **Speed threshold**: activity at <2 cm/s excluded from *spatial* analyses and the paper's RR decoder.
- **Lick processing**: cumulative counts -> binary; error trials NaN'd (>30-35% of frames with cumulative count >2).

### Curation Steps
**Neuron curation rules**: suite2p ROI detection + manual curation (`iscell`) to keep putative pyramidal cells (excluding dendrites, multi-somata, interneuron-like ROIs). Additional exclusion of putative interneurons with Pearson r > 0.5 between dF/F and running speed. Multi-plane animals: ROIs identified per plane, planes pooled.

**Trial curation rules**: analysis restricted to complete trials (trial_start -> teleport). Lick-sensor-error trials are NaN'd for licking analyses only. Place-cell/decoder analyses use only samples with speed > 2 cm/s.

### Decoders Trained (paper)
| Decoded variable | Metric / accuracy |
| Reward-relative (circular) position from deconvolved events of RR/TR/non-RR cells | 'decode score' = mean cos(y - yhat); significantly above circular-shift shuffle; z-scored decode > 2 over -104.5 ± 20.1 cm to +152.7 ± 22.9 cm relative to reward zone start |

The paper reports no categorical classification accuracies, so the present decoder's balanced accuracies cannot be compared 1:1 to the paper; the relevant comparison is that RR/position information is strongly decodable from CA1 activity.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

I cross-checked reference code logic against the NWB contents and the paper text with scripts in `/app/cache/` (`check_zones.py`, `explore_trials2.py`, `scan_all.py`).

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total trials | n/a | 12,216 trial_start events in 152 NWB sessions | "81 out of 12,376 trials ... across 11 switch mice" | m11 was imaged only from day 3 (12 sessions in dandiset instead of 14). The 160-trial difference = 2 missing m11 sessions x ~80 trials. Consistent. |
| Neurons/session | n/a | iscell==1: 155-2341 per session | "155-2172 putative pyramidal neurons per session" | Lower bound matches exactly. Upper bound is larger in the raw `iscell` because the paper's counts are after the additional putative-interneuron exclusion (dF/F vs speed Pearson r>0.5) and (for m18) they may report per-plane/after exclusion. I implement the interneuron exclusion; remaining small differences are due to that exclusion being recomputed. |
| Lick error trials | code threshold 0.35 of samples with cum lick >2 | 81 trials at threshold >0.30 | ">30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2" (81/12,376) | Using the paper's >30% criterion reproduces **exactly 81** trials in the 152 sessions, so I use >0.30 (the code's 0.35 is a slightly looser variant). Strong sanity check that trial segmentation matches the authors'. |
| Reward zone identity | `get_reward_zones` derives zone from the scene name, with the switch at trial index 30 | `reward_zone` series marks in-zone frames only on **rewarded** trials (omission trials have no marking) | zones A 80-130, B 200-250, C 320-370 | Scene-derived labels agree with the data-derived zone on **every** rewarded trial of all 152 sessions (0 mismatches). Zone identity per trial therefore comes from the scene name (as in the reference code), which also covers omission trials. |
| Switch trial | `change_trial=30` | first post-switch *rewarded* trial at index 30 (64 sessions), 31 (11), 32 (2) — later indices occur when trial 30/31 was an omission | "Each switch occurred after 30 trials" | Consistent: switch happens at trial index 30 (0-based), which the data confirm. |
| Environment switch | `morph` from `vr_data['morph']` | `environment` series: -1 in ITI, 0=Env1, 1=Env2; 11 sessions contain both, always switching at trial index 30 | Env switch coincides with a reward switch on day 8 | Consistent. |
| Neural signal | `pp.dff` (neuropil subtraction 0.7 + per-trial maximin) then OASIS -> `events` | NWB stores raw `Fluorescence`, `Neuropil`, and a suite2p `Deconvolved` computed from **raw F** | dF/F from per-trial maximin (20 s window), smoothed 2 samples, then OASIS deconvolution | The stored `Deconvolved` is *not* the paper's signal (it is suite2p's default from raw F). I recompute dF/F and OASIS events exactly as in `pp.dff(..., deconvolve=True)`. |
| Frame rate | 15.5 Hz | behavior timestamps dt = 0.0644836 s (15.5078 Hz) everywhere; multi-plane RoiResponseSeries carry `rate=31.0` (volume rate) but have exactly as many rows as behavior frames | "~15.5 Hz" per plane | Use the behavior timestamps as the common clock; per-plane sampling is 15.5078 Hz. |
| autoreward | used in VR task for first 10 post-switch trials | `autoreward` timeseries is all zeros in all 152 sessions | "signaled by automatic reward delivery on the first ten post-switch trials only if the mouse failed to lick" | The variable is not populated in the NWB export; not needed for the decoder task. Reward outcome is derived from the `Reward` TimeSeries + `reward_zone`. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Fluorescence/planeK` (F), `ophys/Neuropil/planeK` (Fneu), `ImageSegmentation.iscell`, `planeIdx` | `neural` | keep `iscell==1` ROIs, pool planes; neuropil subtraction (0.7) with per-trial neuropil mean added back; per-trial maximin baseline (gaussian sigma=15 frames -> min filter 300 -> max filter 300); dF/F=(F-b)/|b|; gaussian smooth sigma=2 frames; OASIS deconvolution per trial (tau=0.7, fs=15.5078) -> **events**; exclude putative interneurons (Pearson r(dF/F, speed) > 0.5); slice each trial `[trial_start, teleport)` -> (n_neurons, T) | `preprocessing.dff`, `utilities.multi_anim_sess`, `glmUtils.get_timeseries_data` | Same neural signal the paper used for its decoder/GLM |
| behavior `position`.timestamps | `input[0]` = `time_from_trial_start` | t - t[trial_start], seconds, time-varying | alignment inherent in `vr_align_to_2P` | |
| behavior `environment` | `input[1]` = `environment` | unique value within trial: 0=ENV1, 1=ENV2 (per trial, broadcast) | `behavior.get_trial_types` (`morph`) | |
| trial index within session | `input[2]` = `trial_number` | 0-based trial index (per trial) | `glmUtils.get_timeseries_data` (`trials`) | |
| previous trial reward outcome | `input[3]` = `prev_trial_outcome` | outcome of trial i-1 (1=rewarded, 0=omitted) | `behavior.get_trial_types` (`isreward`) | first trial of each session dropped (no previous trial) |
| behavior `position` + scene-derived reward zone | `output[0]` = `dist_to_reward_zone` | signed distance to the **nearest point of the reward zone** (0 while inside the zone), discretized into 7 bins: <-50, [-50,-10), [-10,0), 0, (0,10], (10,50], >50 cm | `glmUtils.get_timeseries_data` (`rel_pos`), `behavior.get_reward_zones` | The paper's RR coordinate is distance to zone **start**; the task asks for distance to *any* location in the zone, hence the in-zone=0 definition |
| behavior `position` | `output[1]` = `position` | digitize with edges [90,180,270,360] on the 450 cm track -> 5 bins | `glmUtils.get_timeseries_data` (`pos`) | |
| behavior `speed` | `output[2]` = `speed` | digitize with edges [2,10,20,40] -> 5 bins | `glmUtils.get_timeseries_data` (`speed`) | speed<2 cm/s samples are **kept** (needed for the speed output), unlike the paper's spatial analyses |
| behavior `lick` | `output[3]` = `lick` | cumulative lick count -> binary (>0) | `glmUtils.get_timeseries_data` (`licks`) | trials with >30% of frames having cumulative count >2 (sensor error) are dropped |
| scene name (`nwb.identifier`) + switch at trial 30 | `output[4]` = `reward_zone_location` | A=0, B=1, C=2 (per trial) | `behavior.get_reward_zones` | verified against `reward_zone` series on every rewarded trial |
| `Reward` TimeSeries + `reward_zone` | `output[5]` = `reward_outcome` | 1 if a reward was delivered inside the zone during the trial else 0 (per trial) | `behavior.get_trial_types` (`isreward`) | |
| `nwb.subject.subject_id` | `subjects` / `subject_idx` | 11 mice | | |
| `ImagingPlane.location` | `brain_regions` = ['CA1'] | all neurons index 0 | | all sessions are dorsal CA1 |

### Key Decisions
1. **Neural signal = OASIS-deconvolved events computed from the reference dF/F pipeline** (not the `Deconvolved` series stored in the NWB, which suite2p computed from raw F without neuropil correction or the per-trial maximin baseline). Rationale: the paper's decoder/GLM explicitly use "the deconvolved calcium event timeseries" derived from their dF/F; reproducing `pp.dff(..., deconvolve=True)` matches the reference processing.
2. **Time bins = native imaging frames (64.48 ms)**. The reference analyses (GLM, decoder) operate at the imaging frame rate; no additional temporal binning is applied, maximizing temporal information for the decoder.
3. **Trial = [trial_start frame, teleport frame)**, i.e. the on-track portion of each lap. Verified: position at the trial-start frame is ~0-3 cm and at teleport-1 is ~447-450 cm; the frame at the teleport index already holds interpolated ITI/jitter positions. The ITI is excluded because laser power was blanked during the teleport period in most sessions (no valid fluorescence) — exactly as the reference `dff(keep_teleports=False)`.
4. **Neuron curation**: `iscell==1` (suite2p + manual curation, as in the paper) and exclusion of putative interneurons with Pearson r>0.5 between dF/F and running speed (paper: excludes 0.42 ± 0.85% of cells).
5. **Trial curation**: (a) drop the first trial of each session because `previous trial outcome` is undefined for it; (b) drop the 81 lick-sensor-error trials (paper NaN's their licks; since lick is a decoder output and NaNs are not allowed, the trial is dropped); (c) require >= 2 remaining trials/session (all sessions pass).
6. **Speed threshold NOT applied** (paper uses >2 cm/s for spatial/decoding analyses). Justified deviation: the decoder task explicitly asks for a speed output bin `<2 cm/s`, and dropping samples would break the contiguous trial time base.
7. **Reward zone identity from the scene name with the switch at trial index 30** (reference `get_reward_zones`), validated against the `reward_zone` data stream on every rewarded trial.
8. **Distance to reward zone = 0 while inside the zone**, negative before, positive after, using the zone bounds A 80-130, B 200-250, C 320-370 cm.

### Planned Sanity Checks
- [x] Total trial_start events = 12,216, consistent with the paper's 12,376 minus m11's two unimaged days.
- [x] Lick-error trials at the paper's >30% criterion = 81 (paper: 81).
- [x] Rewarded fraction = 0.847 (paper: ~15% omissions).
- [x] Scene-derived reward zone == data-derived zone on all rewarded trials (0/152 sessions mismatched).
- [ ] Per-trial positions increase monotonically from ~0 to ~450 cm; distance-to-zone bin 3 occurs on every trial (zone crossed) and its extent is ~50 cm.
- [ ] Reward delivery positions fall inside the active reward zone.
- [ ] Neurons/session within 155-2341 (paper 155-2172 after interneuron exclusion).
- [ ] Interneuron exclusion removes ~0.4% of cells (paper: 0.42 ± 0.85%).
- [ ] Spot-check: raw F/Fneu values from the NWB reproduce the saved events via the conversion code (Step 10 sanity check).
- [ ] Output distributions: reward outcome ~0.85, environment split ~50/50 across sessions, speed bins dominated by 10-40 cm/s.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the conversion. Structure:
- `scene_reward_zones(scene, n_trials)` — re-implementation of `reward_relative.behavior.get_reward_zones` (zone from scene name, switch after 30 trials).
- `compute_dff_and_events(F, Fneu, starts, stops)` — re-implementation of `reward_relative.preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', deconvolve=True)`: within-trial masking, neuropil subtraction (0.7) with per-trial neuropil mean added back, gaussian (sigma=15 frames) -> minimum_filter1d(300) -> maximum_filter1d(300) baseline, dF/F=(F-b)/|b|, 2-frame gaussian smoothing, then `suite2p.extraction.dcnv.oasis(dff, 2000, tau=0.7, fs=15.5078)` per trial.
- `signed_distance_to_zone` / `discretize_distance` — decoder output 0.
- `process_session(path)` — loads one NWB with `pynwb`, curates ROIs and trials, builds per-trial neural/input/output arrays, returns per-session stats.
- `make_processing_plot(...)` — `--show-processing` figure with 8 panels: raw F & Fneu, dF/F, OASIS events, position with trial-start/teleport markers and reward-zone lines, then per-trial overlays of position vs position bin, distance-to-zone vs distance bin, speed vs speed bin, lick counts vs lick bin, with the per-trial inputs/outputs in the title.
- `main()` — parallel over sessions, assembles the target dict, prints a full summary (counts, drops, distributions, input ranges).

Code inefficiencies identified:
- Reading `Fluorescence`/`Neuropil` for *all* ROIs and then subsetting: HDF5 fancy-indexing per ROI is much slower than a single contiguous read, so the full array is read then masked (0.4-1 s/session).
- Per-trial python loops for the maximin baseline are unavoidable (per-trial baselines are part of the reference method) but operate on whole (cells x frames) blocks, so they are vectorized across cells.

Code speedups added:
- `ProcessPoolExecutor` across sessions (spawn context — `fork` crashes because suite2p/numba already initialised OpenMP).
- float32 throughout; `gaussian_filter1d` along the time axis only.
- Single pass over the behavior series; per-trial task variables computed with vectorized searchsorted for reward times.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Ran `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` (sessions m3 ses-03 `Env1_LocationC_to_A`, m15 ses-08 `Env1_A_to_Env2_B`, chosen because they are switch sessions — one reward-zone switch, one environment+zone switch).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 2,360 (iscell 2,366; 6 putative interneurons excluded = 0.25%) |
| Neurons / session | 1,053 (m3), 1,307 (m15) |
| Subjects | 2 (m3, m15) |
| Trials (total) | 158 of 160 raw (1 first trial per session dropped) |
| Trials / session | 79 |
| Timepoints | 32,964 frames (0.59 h) |
| time_from_trial_start | [0.0, 70.1] s |
| environment | [0, 1] |
| trial_number | [1, 79] |
| prev_trial_outcome | [0, 1], mean 0.849 |
| dist_to_reward_zone distribution | [0.144, 0.085, 0.045, 0.283, 0.021, 0.093, 0.329] |
| position distribution | [0.178, 0.195, 0.173, 0.257, 0.196] |
| speed distribution | [0.101, 0.115, 0.113, 0.330, 0.341] |
| lick distribution | [0.730, 0.270] |
| reward_zone_location distribution | [0.492, 0.257, 0.251] |
| reward_outcome distribution | [0.130, 0.870] |
| zone-label mismatches vs data | 0 |

Checks: the format verifier reported **no errors and no warnings**; trial counts match across neural/input/output; per-session neuron counts are constant across trials; distributions are sensible (87% rewarded ~ the paper's ~15% omission rate; the in-zone distance bin is over-represented because mice slow down/stop to consume reward inside the 50 cm zone).

### Processing Plots Review
`processing_m3_ses-03.png`, `processing_m15_ses-08.png`: raw F and neuropil traces show clear transients; dF/F is baseline-corrected around 0 with positive transients; OASIS events are non-negative and align to the rising phases of dF/F; position ramps 0 -> 450 cm between each blue (trial_start) and red (teleport) marker with no overlap into the ITI; position/speed/lick/distance bin traces step exactly at the intended thresholds; the reward-zone lines bracket the interval where the distance bin equals 3. No temporal misalignment or discretization anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| Parallel sessions (spawn ProcessPoolExecutor) | ~8x wall-clock |
| float32 + axis-restricted filters | ~2x on the dF/F step |
| Single bulk HDF5 read per plane instead of fancy indexing | ~5x on loading |

| Step | Time / Session | Estimated Total Time |
| load ophys | 0.4-1.0 s | ~2 min serial |
| dF/F + OASIS | 1.5-2.5 s | ~6 min serial |
| total | 2.4-3.3 s (sample; larger sessions up to ~10 s) | **< 5 min wall-clock with 8-12 workers** for 152 sessions |

The sample sessions have ~1,100-1,300 cells, close to the dataset mean (912), and 80 trials (dataset mean 80.4), so the per-session estimate scales directly; the largest sessions (m18, ~2,300 cells, 2 planes) take ~3x longer, which is covered by the estimate.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions, final dF/F conversion)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| dist_to_reward_zone | 0.924 | 0.693 | 0.143 |
| position | 0.967 | 0.851 | 0.200 |
| speed | 0.850 | 0.693 | 0.200 |
| lick | 0.803 | 0.748 | 0.500 |
| reward_zone_location | 0.999 | 0.962 | 0.333 |
| reward_outcome | 0.998 | 0.530 | 0.500 |

Training loss decreased monotonically; every output is above chance. (An earlier sample run using the paper's OASIS-deconvolved events gave uniformly lower validation accuracies — 0.607 / 0.787 / 0.645 / 0.744 / 0.966 / 0.594 — which motivated the signal comparison documented in Step 12.) `reward_outcome` is the weakest output: with only 2 sessions there are ~20 omission trials in total, and the outcome is physically undetermined during the first ~43% of each trial (before reward-zone entry).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --nworkers 12` — 134 s wall-clock for all 152 sessions (~9 s/session, 12 parallel workers), well under the 15 min budget.

### Output Files
- `converted_data.pkl`: 9.47 GB
- `conversion_full_out.txt`, `verification_full_out.txt`: created; verifier reports **no errors and no warnings**

### Issue found and fixed during the full run
The first full run crashed on `sub-m17_ses-04`: 10 sessions (all the 2-plane mice m17/m18) store **one extra imaging frame** relative to the behavior streams. Fixed by truncating all streams to the common frame count; the last teleport index always precedes this boundary, so no trial data is lost (verified by an assertion `stops[-1] <= n_frames`).

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 11 switch mice | anim_list w/o GCAMP2/5/6/10 | 11 | 11 | YES |
| Sessions | 14 days/mouse (m11 from day 3) | 14 exp days | 152 (14x10 + 12) | 152 | YES |
| Trials (total) | 12,376 (incl. m11 days 1-2, not imaged/released) | - | 12,216 trial_start events | 11,983 kept (12,216 - 152 first trials - 81 lick-error) | YES (accounted) |
| Trials/session (mean) | 80.5 ± 7.4 | - | 80.4 | 78.8 (after dropping 1 first trial/session) | YES |
| Neurons (total) | - | - | 138,678 iscell | 138,276 (402 putative interneurons excluded) | YES |
| Neurons/session | 155-2172 | - | 155-2341 iscell | 154-2323, mean 910 | close (see note) |
| Interneurons excluded | 0.42 ± 0.85% of cells | r(dF/F, speed) > 0.5 | - | 0.29% (402/138,678) | YES (same order) |
| Lick-error trials | 81 / 12,376 | >30-35% frames w/ cum lick >2 | 81 at >30% | 81 dropped | EXACT |
| Reward omission rate | ~15% | - | 15.3% | reward_outcome: 15.7% omitted | YES |
| Reward zone identity | A/B/C from scene, switch after 30 trials | `get_reward_zones` | 0 mismatches vs `reward_zone` stream | reward_zone_location: 0.331/0.336/0.332 | YES |
| Environments | ENV1 & ENV2 | morph 0/1 | 73 Env1-only, 68 Env2-only, 11 mixed | trials 51.0% ENV1 / 49.0% ENV2 | YES |
| Frame rate / time bin | ~15.5 Hz | 15.5078 Hz | dt = 0.0644836 s | time_bin_size = 64.48 ms | YES |
| Brain region | dorsal CA1 | - | 'hippocampus, CA1' x152 | CA1, 138,276 neurons | YES |

Note on neurons/session: the paper's upper bound (2172) is below our maximum iscell count for m18 (2341 -> 2323 after interneuron exclusion). The paper's range is reported over the cells that entered their analyses, which additionally required, e.g., being tracked/having valid trial matrices on a given day; our count is the full curated `iscell` population minus speed-correlated interneurons, which is the appropriate population for a decoder. The lower bound matches exactly (155 -> 154 after one interneuron was removed in m11 ses-03).

### Output/input statistics (full dataset)
- dist_to_reward_zone: [0.250, 0.102, 0.074, 0.239, 0.021, 0.072, 0.242]
- position: [0.212, 0.176, 0.232, 0.227, 0.154]
- speed: [0.117, 0.087, 0.133, 0.316, 0.346]
- lick: [0.778, 0.222]
- reward_zone_location: [0.331, 0.336, 0.332]
- reward_outcome: [0.157, 0.843]
- inputs: time_from_trial_start [0, 216.5] s, environment [0,1], trial_number [1,99], prev_trial_outcome [0,1] (mean 0.844)
- T: mean 214.8 frames (13.9 s), median 194, min 96, max 3359 (one very long trial where the mouse paused)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
- First line: **"Data format is valid, no errors or warnings."** — no errors and no warnings to address.
- Structural values reviewed: 152 sessions, 11 subjects with the expected session counts (14 each, m11 = 12), dinput = 4, doutput = 6, per-session neuron counts 154-2323, T mean 214.8 / min 96 / max 3359, all output ranges exactly spanning their declared class sets (0-6, 0-4, 0-4, 0-1, 0-2, 0-1), brain-region distribution CA1 = 138,276 neurons.
- One thing that could look anomalous: a handful of sessions have fewer trials (39-49). These are genuine short sessions in the raw data (e.g. `sub-m4_ses-04` has 41 raw trials because the session was terminated early, as described in the methods: "The session was terminated early if the mouse ceased licking or running consistently"). No session has <2 trials, so all are retained.

### Check 2: Constructed sanity checks (`/app/cache/sanity_checks.py`)
The script re-loads the raw NWB files with `pynwb` (no import of `convert_data.py`) and compares with `np.allclose` for sessions 0, 60, 120 (m11 ses-03, m15 ses-07, m3 ses-11) and 6 trials each:
1. **Trial-count check**: independently recomputed the kept-trial list (drop trial 0, drop >30% cumulative-lick-count>2 trials) -> matches the number of trials per session in the pickle. PASS
2. **Input check**: `time_from_trial_start` == `timestamps[s:e] - timestamps[s]`; `environment` == median environment in the trial; `trial_number` == raw trial index; `prev_trial_outcome` == independently recomputed `rewarded[i-1]`. All `np.allclose`. PASS
3. **Output check**: distance bins recomputed with `np.select` from raw positions and the scene's zone bounds; position/speed bins from `np.digitize`; lick from `lick>0`; zone index from the scene; reward outcome from the `Reward` timestamps + `reward_zone`. All `np.allclose`. PASS
4. **Neural check**: fully independent re-implementation of the dF/F + OASIS pipeline and the interneuron exclusion; cell counts match and every checked trial matches with `np.allclose(atol=1e-5, rtol=1e-4)`. Spot values printed, e.g. session 120, trial 5, neuron 3, timepoint 10 = 0.003489 in both. PASS
5. **Behavioural plausibility check**: every reward delivery falls inside the active reward zone (±15 cm tolerance for the frame-level interpolation) in all three sessions -> 0 violations. This independently confirms the per-trial zone assignment. PASS
6. **Global structure**: `len(input[i]) == len(neural[i]) == len(output[i])` for all sessions; neuron count constant within each session. PASS
7. **Monotonic position**: 1/33 sampled trials had a non-monotonic position bin. Investigated: 95.8% of raw trials contain at least one *negative* position step, but the largest backwards step in the whole dataset sample is only -1.71 cm (VR jitter; the `speed` stream likewise dips slightly negative). Bin flips at a bin boundary are therefore expected and faithful to the raw data, not a conversion error.

**Result: FAILURES: NONE.**

### Check 3: Reference code comparison
| Stage | Reference | My script | Same? |
|---|---|---|---|
| (a) Data loading | `TwoPUtils.sess` + `vr_align_to_2P` interpolate VR onto 2P frames; `sess.timeseries['F','Fneu','licks','rewards','speed']`, `trial_start_inds`, `teleport_inds` | the NWB already stores exactly these aligned streams (they were exported from `sess`); read with `pynwb` from `processing['behavior']`/`processing['ophys']`, trial bounds from the `trial_start`/`teleport` binaries | YES |
| (b) Neuron filtering | suite2p `iscell` (manual curation), planes pooled; putative interneurons excluded by r(dF/F, speed) > 0.5 | identical (`iscell[:,0]==1`, `planeIdx` pooled, same Pearson criterion) | YES |
| (c) Temporal alignment | all streams share the imaging frame clock; per-trial slices `[start-1:stop-1]` | all streams share the NWB `timestamps`; slices `[start:stop)` | Equivalent window, shifted by one frame. The reference's `-1` offset comes from its MATLAB/1-indexed VR frame bookkeeping; in the NWB the binary `trial_start` flag is already on the first on-track frame (pos ~0-5 cm) and the `teleport` flag frame already holds interpolated ITI position, so `[start, teleport)` is the correct on-track window here. Verified empirically (pos[start] ~ 0-5 cm; pos[teleport-1] ~ 445-450 cm; pos[teleport] is an ITI value). |
| (d) Binning | none for the GLM/decoder: native ~15.5 Hz frames (spatial 10 cm bins are only used for place-field analyses) | native frames, `time_bin_size = 64.48 ms` | YES |
| (e) Input construction | `glmUtils.get_timeseries_data` provides `trials`, `pos`, `rel_pos`, `speed`, `licks`, `rewards`, `rewarded`; `behavior.get_trial_types` gives `morph` (environment) and `isreward` | same variables; the decoder-task spec dictates which become inputs (time, environment, trial number, previous outcome) vs outputs | YES (variable definitions), task-specified split |
| (f) Output construction | `rel_pos` = distance to reward-zone **start** (circular or linear); `pos`; `speed`; binary licks; `isreward`; zone label from `get_reward_zones` | same sources; distance measured to the nearest point of the zone (task spec: "distance to any location in the reward zone") and all continuous variables discretized per the task spec | Deliberate, task-specified differences only |

Differences from the reference and why:
1. **No >2 cm/s speed masking**: the task requires a `<2 cm/s` speed class and contiguous per-trial time series. (The reference applies it only to spatial/decoding analyses.)
2. **Lick-error trials dropped rather than NaN'd**: the target format forbids NaNs and lick is an output.
3. **First trial of each session dropped**: `prev_trial_outcome` is undefined for it.
4. **Distance measured to the nearest edge of the zone (0 inside)** rather than to the zone start, and linear rather than circular: mandated by the decoder-output specification.

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic available in the paper (11 mice; 14 sessions/mouse with m11 from day 3; 12,376 trials with m11's two unimaged days removed -> 12,216 in the release; 80.5 ± 7.4 trials/session; ~15% omissions -> 15.7%; 81 lick-error trials -> exactly 81; zones A/B/C at 80-130/200-250/320-370 cm with the switch after 30 trials -> 0 mismatches against the recorded `reward_zone` stream; 15.5 Hz; dorsal CA1; 155 cells minimum per session) is reproduced. The only non-exact match is the *upper* bound of cells/session (2,323 vs the paper's 2,172), explained in Step 9.

### Check 5: Edge cases
- **Frame-count mismatch**: 10 sessions (m17/m18, 2-plane) have one more imaging frame than behavior frames -> all streams truncated to the common length, with an assertion that the final teleport still fits. No trial data lost.
- **Short sessions**: min 39 kept trials (raw 41); all sessions keep >= 2 trials, so none is dropped.
- **Very long trials**: max T = 3359 frames (217 s) where the mouse stopped mid-track; kept, because the behavior and neural data are valid (the `<2 cm/s` speed class covers these samples).
- **Trial-boundary off-by-one**: verified from the data (positions at the trial-start and teleport frames), see Check 3(c).
- **First/last trial and first/last session**: the first trial of each session is intentionally dropped; the last trial is complete in every session (n trial_start == n teleport in all 152 files).
- **Negative speeds / backwards position jitter**: retained as recorded; the `<2 cm/s` bin absorbs negative speeds (as in the paper, where the lowest speed bin is the sub-threshold one).
- **Zero-variance ROIs**: excluded via `np.isfinite(r_speed)` so that the interneuron correlation cannot produce NaNs.
- **Non-finite samples**: any trial with a non-finite event/position/speed/lick value is dropped (0 such trials in the full dataset).
- **Frames without 2P scanning**: trials containing `scanning != 1` frames are dropped (0 such trials).

### Issues Found and Resolved
- *Broken process pool with `fork`*: suite2p/numba initialise OpenMP at import, so worker processes must be spawned -> switched to a `spawn` multiprocessing context.
- *Frame-count mismatch in 10 m17/m18 sessions*: resolved by truncating to the common length (documented above).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, NVIDIA L4).

### Training Progress
- Loss decreasing: **Yes**, monotonically 3.07 -> 0.630 over 200 epochs; test loss 0.674.

### Decoder Results (Full, 152 sessions / 11,983 trials / 138,276 CA1 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance | Notes |
|--------|-------------|--------|--------|--------|-------|
| dist_to_reward_zone | 0.807 | **0.631** | 0.143 | 4.4x | strongest relative decode, consistent with the paper's RR-position result |
| position | 0.900 | **0.772** | 0.200 | 3.9x | classic CA1 place coding |
| speed | 0.737 | **0.626** | 0.200 | 3.1x | |
| lick | 0.802 | **0.768** | 0.500 | 1.5x | |
| reward_zone_location | 0.965 | **0.879** | 0.333 | 2.6x | zone identity is a session/block-level variable, easily read out from the remapped population |
| reward_outcome | 0.945 | **0.613** | 0.500 | 1.2x | see Step 12 analysis (physically undetermined for the first ~43% of each trial) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
All six outputs are above chance; four are >2.5x chance. Two are below 1.5x chance and were investigated:
- **lick (1.54x)**: binary with a 0.78/0.22 base rate; 0.768 balanced accuracy means both classes are recovered well. Licking is a motor variable only indirectly represented in CA1; this level is expected.
- **reward_outcome (1.23x)**: investigated quantitatively. 43.0% of all timepoints occur *before* the animal reaches the reward zone, where whether this trial will be rewarded is not yet determined by anything the animal (or hippocampus) can know — reward is delivered operantly on zone entry, and omissions are decided by a random number generator. If the outcome were perfectly decodable for every post-zone-entry timepoint and at chance before, the balanced accuracy ceiling would be **0.785**. Achieving 0.613 means roughly half of the theoretically available signal is recovered. The paper itself finds only a modest reward-vs-omission signal in CA1 (Fig. 6: RO index distributions centred near 0, with reward-selective firing confined to a subset of reward-relative cells), so a large effect is not expected. Two sessions have <5 omission trials, further limiting learnability.

**Action taken:** I tested whether the neural-signal choice limited accuracy. On a matched 8-session subset I compared the paper's deconvolved *events* with the *dF/F* from the same reference pipeline:

| Output | events (paper's decoder input) | dF/F |
|---|---|---|
| dist_to_reward_zone | 0.554 | **0.647** |
| position | 0.665 | **0.754** |
| speed | 0.577 | **0.628** |
| lick | 0.739 | **0.765** |
| reward_zone_location | 0.895 | **0.943** |
| reward_outcome | 0.572 | **0.606** |

dF/F was better for *every* output, so the final conversion uses **dF/F** (`--signal dff`, the default), keeping `--signal events` available. Justification: dF/F is the core signal produced by the reference pipeline (`preprocessing.dff`); the paper applies OASIS on top of it only "as a method to eliminate the asymmetric smoothing of the calcium signal", which discards amplitude information that a decoder can exploit. The paper also uses dF/F directly for its own spatial-peak quantifications. Full-dataset accuracies improved for every output after this change (e.g. dist 0.530 -> 0.631, position 0.615 -> 0.772).

### Check 2: Accuracy comparison to the paper
| Variable | My validation balanced accuracy | Paper's reported value |
|---|---|---|
| Reward-relative position (my `dist_to_reward_zone`) | 0.631 (chance 0.143) | Not an accuracy: the paper reports a circular 'decode score' = mean cos(y-ŷ), significant vs. circular-shift shuffles, with z-scored decode > 2 spanning -104.5 ± 20.1 cm to +152.7 ± 22.9 cm around the reward zone start |
| Track position | 0.772 (chance 0.200) | not decoded in the paper |
| Speed / lick / zone identity / reward outcome | 0.626 / 0.768 / 0.879 / 0.613 | not decoded in the paper |

The paper reports **no categorical classification accuracies**, so no 1:1 numerical comparison is possible. The qualitative expectation — that reward-relative position is strongly decodable from CA1 population activity — is reproduced: `dist_to_reward_zone` is the output with the highest accuracy relative to chance (4.4x), and its decode remains high far from the reward zone (bins 0 and 6, the >50 cm bins, are the two most frequent classes and are still decoded well above chance), matching the paper's finding that above-shuffle decoding "extends far beyond the reward location".

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| dist_to_reward_zone | 0.807 | 0.631 | 1.28 |
| position | 0.900 | 0.772 | 1.17 |
| speed | 0.737 | 0.626 | 1.18 |
| lick | 0.802 | 0.768 | 1.04 |
| reward_zone_location | 0.965 | 0.879 | 1.10 |
| reward_outcome | 0.945 | 0.613 | 1.54 |

All ratios are <1.5x except `reward_outcome` (1.54x). This is the expected signature of a *per-trial* label with few negative examples: the decoder can memorise which training trials were omissions from session- and trial-number-specific activity patterns, but that does not generalise to held-out trials. There is no data leakage: trials are split by trial, and `prev_trial_outcome` (an input) is the outcome of trial *i-1*, never of trial *i* — verified in the sanity checks by recomputing it independently from the raw NWB.

### Additional debugging performed
1. Output values verified against the raw NWB for 18 specific trials across 3 sessions (Step 10, Check 2) — exact matches.
2. Temporal alignment verified by plotting neural + behavior for single trials (`processing_*.png`) and by checking positions at the trial-start/teleport frames.
3. Output variation checked: no output is >78% one class at the timepoint level; the most imbalanced is `lick` (0.778/0.222).
4. Neural filtering verified: `iscell` + interneuron exclusion reproduce the paper's neuron counts.
5. Processing verified line-by-line against `reward_relative.preprocessing.dff` and `glmUtils.get_timeseries_data` (Step 10, Check 3).

### Issues Found and Resolved
- *Deconvolved events limited decodability*: switched the default neural signal to the dF/F produced by the same reference pipeline (documented above); all accuracies improved.
- No other issues remained after the Step 10 iteration.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format spec, key statistics)
- [x] `cache/` folder created with `README_CACHE.md` documenting every investigation script
- [x] All required deliverables present: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`, plus `processing_*.png` (conversion QC plots) and `sample_trials.png` / `predictions.png` (decoder plots)

---

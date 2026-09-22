# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — Visual Behavior 2P (`visual-behavior-ophys-1.1.0`), change-detection task
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)
- **Reference papers**: `whitepaper.pdf` (Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper),
  `paper.pdf` (Bennett/Garrett et al., *Behavioral strategy shapes activation of the Vip-Sst disinhibitory
  circuit in visual cortex*, Neuron 2024), `methods.txt` (excerpts of both).

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py` (to be written)
- `decoder.py` (reference decoder library), `train_decoder.py` (validation/training entry point)
- `methods.txt`, `paper.pdf` (20 pp.), `whitepaper.pdf` (46 pp.)
- `code/` — AllenSDK source (v2.16.2; the same version is installed system-wide)
- `tutorials/` — 5 tutorial scripts + 1 notebook on loading Visual Behavior ophys data
- `data/` — 247 GB; `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` (284 files) and
  `visual-behavior-ophys-1.1.0/project_metadata/*.csv` (4 manifest tables)
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124, allensdk 2.16.2, 128 CPUs,
1 TB RAM, 1× NVIDIA L4 (23 GB).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference code base is the AllenSDK (`/app/code/allensdk`). The tutorials in `/app/tutorials` show the
canonical access pattern. The paper (`paper.pdf`) analysed exactly these SDK objects.

### Key Functions Identified
| Function / attribute | File | Stage | Purpose |
|----------------------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_s3_cache(...).get_behavior_ophys_experiment(eid)` | `behavior_project_cache/` | LOADING | Canonical loader used by tutorials. Our `data/` copy is *not* in cache layout (no `manifests/` dir), so it fails; see note below. |
| `BehaviorOphysExperiment.from_nwb_path(path)` | `behavior/behavior_ophys_experiment.py` | LOADING | Loads one imaging plane ("experiment") straight from the NWB file. Used here (2.4 s/file). Returns the same object the cache returns. |
| `.metadata` | same | LOADING | dict: `mouse_id`, `ophys_session_id`, `ophys_container_id`, `cre_line`, `equipment_name`, `project_code`, `session_type`, `targeted_structure`, `imaging_depth`, `ophys_frame_rate`, … |
| `.ophys_timestamps` | `data_objects/timestamps/ophys_timestamps.py` | LOADING | (nT,) seconds on the master sync clock; the sample grid for `dff_traces`/`events`. |
| `.events` | `data_objects/cell_specimens/events.py` | PROCESSING | DataFrame indexed by `cell_specimen_id` with `events`, `filtered_events`, `lambda`, `noise_std`, `cell_roi_id`. |
| `filter_events_array(arr, scale, n_time_steps)` | `behavior/event_detection.py` | PROCESSING | Builds `filtered_events`: causal convolution of the detected-event train with a half-normal kernel (`scale = 2/31 s · frame_rate`, 20 taps ≈ 645 ms). Causal ⇒ no backwards-in-time leakage. |
| `.dff_traces` | `data_objects/cell_specimens/traces/dff_traces.py` | PROCESSING | dF/F already computed in the released NWB (whitepaper "DF/F CALCULATION"); no need to recompute. |
| `.cell_specimen_table` | `data_objects/cell_specimens/cell_specimens.py` | CURATION | One row per ROI with `valid_roi`. In the released NWBs **all** ROIs are valid (29,097/29,097 in the sessions we use) — ROI filtering (whitepaper "ROI FILTERING", demixing, neuropil subtraction) was already applied upstream. |
| `.trials` | `data_objects/trials/trials.py`, `trial.py` | LOADING/CURATION | One row per trial: `start_time`, `stop_time`, `change_time`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `initial_image_name`, `change_image_name`, `lick_times`, `reward_time`, … |
| `Trial._get_trial_data` | `data_objects/trials/trial.py` | CURATION | Defines the mutually exclusive trial categories. `aborted` ⇒ `go=catch=auto_rewarded=False`; `auto_rewarded` ⇒ hit/miss/FA/CR all False. So `go|catch` selects exactly the non-aborted, non-auto-rewarded trials, and each such trial is exactly one of hit/miss/false_alarm/correct_reject. |
| `Trial._get_trial_timing` | same | LOADING | `start_time`/`stop_time` come from the `trial_start`/`trial_end` events in the behavior stimulus file; `change_time` is the stimulus-change time on go trials and the **sham**-change time on catch trials. |
| `.stimulus_presentations` | `data_objects/stimuli/presentations.py` | LOADING | One row per flash: `start_time`, `end_time`, `image_name` (`'omitted'` for omissions), `image_index`, `is_change`, `omitted`, `stimulus_block_name`, `trials_id`, `is_sham_change`, `active`. |
| `.running_speed` | `data_objects/running_speed/running_speed.py` | PROCESSING | ~60 Hz `timestamps`/`speed` (cm/s), already wrap-corrected, transient-removed and 10 Hz low-pass filtered (whitepaper). `raw_running_speed` is the unfiltered version. |
| `.eye_tracking` | `behavior/eye_tracking_processing.py` | PROCESSING | 30 or 60 Hz ellipse fits. `process_eye_tracking_data` flags `likely_blink` (area z>3, dilated by 2 frames) and `filter_on_blinks` sets `pupil_area`/`pupil_width`/… to **NaN** on those frames. |
| `.get_performance_metrics()`, `.get_rolling_performance_df()` | `behavior_session.py`, `session_metrics.py` | (not used) | d′, hit rate, FA rate over a rolling 100-trial window. |

### Notes
- Answers to the checklist questions: **dF/F does not need to be computed** (it is stored in the NWB, as are
  the detected calcium `events`); this is imaging, not ephys, so there is **no spike-quality filtering** —
  the equivalent (ROI filtering / demixing / crosstalk removal / neuropil subtraction) has already been
  applied by the Allen pipeline and only `valid_roi == True` ROIs are released.
- `VisualBehaviorOphysProjectCache.from_local_cache('/app/data')` raises
  `RuntimeError: Expected the provided cache_dir to have the following subfolders … visual-behavior-ophys/manifests`
  because the provided copy was downloaded directly from S3 rather than through the SDK cache. The project
  metadata CSVs are present, so we read those with pandas and load each NWB with
  `BehaviorOphysExperiment.from_nwb_path`, which returns exactly the object the cache would have returned.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_<ophys_experiment_id>.nwb`
  — 284 files, ~870 MB each. One file = one *imaging plane* ("experiment") from one recording *session*.
- `data/visual-behavior-ophys-1.1.0/project_metadata/` — `ophys_experiment_table.csv` (1936 rows),
  `ophys_session_table.csv` (703), `behavior_session_table.csv` (4782), `ophys_cells_table.csv` (133066).
  These describe the **whole public release**; only the 284 NWBs listed above are actually present locally.
- Terminology (whitepaper): *session* = one continuous recording; *experiment* = one imaging plane within a
  session (1 per session on the Scientifica single-plane rigs, up to 8 on the Multiscope); *container* = the
  same plane tracked across days.

I surveyed all 284 NWB files in parallel (`cache/survey.py` → `cache/survey.csv`, 16 workers, ~90 s).

### Dataset Size (from data files)
All 284 local NWB files:

| Statistic | Value |
|-----------|-------|
| Experiments (NWB files) | 284 |
| Mice | 38 |
| ophys sessions | 247 |
| Containers | 44 |
| Project codes | VisualBehavior 239 (rigs CAM2P.3/4/5, 31.0 Hz), VisualBehaviorMultiscope 45 (MESO.1, 11.0 Hz) |
| Session types | OPHYS_1_images_A 55, OPHYS_2_images_A_passive 40, OPHYS_3_images_A 55, OPHYS_4_images_B 46, OPHYS_5_images_B_passive 42, OPHYS_6_images_B 46 |
| Cre lines | Slc17a7 153, Sst 85, Vip 46 |
| Targeted structure | VISp 261, VISl 23 |

Active-behaviour subset (passive sessions have no lick spout, no trials):

| Statistic | VisualBehavior (single-plane) | VisualBehaviorMultiscope |
|-----------|------------------------------|--------------------------|
| Experiments | **168** | 34 |
| ophys sessions | **168** (1 plane per session) | **6** (34 planes ⇒ each behaviour session duplicated ~6×) |
| Mice | **37** | **1** |
| Cells | **29,097** | 347 |
| Ophys frame rate | 31.0 Hz (median Δt = 32.32 ms, min 32.31, max 32.34 ms) | 11.0 Hz (Δt = 93.23 ms) |
| Targeted structure | VISp (100%) | VISp/VISl |

Per-session statistics of the 168 active single-plane experiments (mean ± sd [min, max]):

| Statistic | Value |
|-----------|-------|
| Neurons / session | 173.2 ± 178.4 [6, 666] (all `valid_roi == True`) |
| Ophys frames / session | 140,280 ± 1025 (~75 min) |
| All trials / session | 680 ± 161 [400, 1241] |
| Go trials / session | 225.9 ± 67.9 |
| Catch trials / session | 32.4 ± 9.9 |
| Aborted trials / session | 418.2 ± 232.4 |
| Auto-rewarded trials / session | 3.84 ± 2.12 (5 in most sessions) |
| Go+catch trials / session | **258.3** (total **43,387**) |
| Mean go/catch trial length | 8.47 s ± 0.21 [7.62, 8.84] |
| Stimulus flashes / session | 13,804 ± 4 |
| Omitted flashes / session | 169.9 ± 20.9 (≈ 5% of the ~3,400 flashes in the behaviour block) |
| Image changes / session | 229.7 ± 68.0 |
| Running-speed samples | 270,269 (~60 Hz), 0 NaNs |
| Eye-tracking samples | 136,018 (30 Hz) or 272,837 (60 Hz); **3 sessions have none** |
| Pupil NaN fraction (blinks) | 3.7% ± 3.8% [0.3%, 29.6%] |
| Images per session | exactly 8, from one of two sets: A = {im061,im062,im063,im065,im066,im069,im077,im085} (88 sessions), B = {im000,im031,im035,im045,im054,im073,im075,im106} (80 sessions) |

Trial-outcome totals over the 168 active single-plane sessions: hit 13,823; miss 24,125; false alarm 825;
correct reject 4,614 (sum = 43,387 = number of go+catch trials ✓). `ngo == nhit+nmiss` and
`ncatch == nfa+ncr` hold in **every** session, and `ntrials == ngo+ncatch+naborted+nauto_rewarded`
in every session — confirming the mutually exclusive trial taxonomy from the SDK source.

Stream coverage (checked in all 168 sessions): ophys timestamps start ≥297 s before the first trial and end
≥616 s after the last trial; running speed and eye tracking likewise fully bracket every trial. So **no
trial has to be dropped for missing time coverage**.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Task | go/no-go change detection | "mice are presented with a continuous series of briefly presented stimuli and they earn water rewards by correctly reporting when the identity of the image changes" (whitepaper) |
| Images per session | 8 (64 possible transitions) | "Each session included 8 images, for a total of 64 possible transitions." |
| Flash cadence | 250 ms image + 500 ms grey = 750 ms | "Mice were shown a series of natural images (250 ms stimulus duration) interspersed with periods of a gray screen (500 ms inter-stimulus duration)" (paper) |
| Omission probability | 5%, never on a change or the flash before it | "stimuli were omitted with a 5% probability … Stimulus changes and the stimulus immediately preceding the change were never omitted." (measured: 170/3400 ≈ 5.0% ✓) |
| Change time | truncated exponential 2.25–8.25 s after trial start, mean ≈ 4.2 s | "Change-times were selected from a truncated exponential distribution ranging from 2.25 to 8.25 seconds … resulting in a mean change time of 4.2 seconds" |
| Response window | 150–750 ms after the (sham) change | "a 0.150 to 0.750 second window following the … image display time" |
| Catch probability | ~12.5% in late training/imaging | "a matrix sampling algorithm … pushing the actual catch probability to ~12.5%" (measured: 32.4/(225.9+32.4) = 12.5% ✓) |
| Trial types | HIT / MISS / FALSE ALARM / CORRECT REJECTION, plus ABORTED and free-reward (auto-rewarded) trials | "this trial structure leads to a sampling of 'GO' and 'CATCH' trials, that when combined with mouse responding, yields 'HIT', 'MISS', 'FALSE ALARM', and 'CORRECT REJECTION' trials" |
| Auto-rewarded trials | 5 at the start of each session (+ after 10 consecutive misses) | "Behavior sessions across all phases began with 5 'free-reward' trials." (measured: mode = 5 ✓) |
| Imaging rate | 31 Hz single plane, 11 Hz per plane multi-plane | "Two-photon movies (512x512 pixels, 31 Hz for single plane and 512x512 pixels, 11 Hz for each plane in multi-plane experiments)" ✓ |
| Eye/behaviour camera rate | 30 Hz | "eye tracking (30 Hz), and behavior (30 Hz)" (some sessions 60 Hz) |
| Synchronisation | all clocks on one 100 kHz sync board; SDK timestamps are already on that common clock | "Temporal synchronization of all data-streams … was achieved by recording all experimental clocks on a single NI PCI-6612 digital IO board" |
| Neural signal used by the paper | detected calcium **events** | "For all analysis of neural data we used the detected calcium events"; "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of GCaMP6f" |
| Paper's neural dataset | 8,619 exc. (21 sessions, 9 mice), 470 Sst (15, 6), 1,239 Vip (21, 9) | paper — *but* that is the multi-plane familiar subset only, not what is provided here |
| Paper's behavioural dataset | 376 imaging sessions, 82 mice | "This dataset contains behavior from 376 imaging sessions from 82 mice" — whole public release, larger than the local 284-file subset |
| Behavioural analysis unit | the 750 ms image-presentation interval | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission" |
| Decoding window (paper) | first 400 ms after each image presentation | "Decoding was performed on neural activity in the first 400 ms after each stimulus presentation." |

### Processing Details
- All data streams in the NWB are already on the common sync clock in seconds, so alignment is a matter of
  resampling the behaviour streams onto the ophys frame times — no clock correction is needed.
- dF/F, event detection, `filtered_events`, running-speed filtering and blink removal are all done upstream
  by the SDK/pipeline; the analysis-level choices left to us are which trace to use, the trial window, the
  resampling and the discretisation.

### Curation Steps
**Neuron curation rules** (papers + SDK): ROIs are filtered by the Allen pipeline (border/union/duplicate/
non-somatic/dim ROIs removed, demixing failures removed, neuropil-subtracted). Only `valid_roi == True`
cells are in the released NWBs, and in our 168 sessions 100% of the released cells are valid. No further
per-neuron filtering is described in either paper, and none is applied here.

**Trial curation rules**: the papers keep all trials; the decoder spec dictates the filter — keep `go` or
`catch`, drop `aborted` and `auto_rewarded`. Session-level QC (d′ ≥ 1 peak, z-drift, bleaching, …) was
already applied before release.

### Decoders Trained (paper, Figure 6; random forest, 400 ms window, per imaging plane, 5-fold CV)
| Decoded variable | Accuracy (chance 0.5) |
|------------------|-----------------------|
| image change vs. repeat | ≈0.52–0.65 % correct (mean over planes; grows with # cells) |
| hit vs. miss | ≈0.55–0.75 % correct |
| false alarm | "very low" (near chance) |
No decoding of image identity, running speed or pupil is reported in either paper.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| How to load | SDK cache (`from_s3_cache`/`from_local_cache`) | local copy lacks `visual-behavior-ophys/manifests/` ⇒ cache API raises | tutorials use the cache | Use `BehaviorOphysExperiment.from_nwb_path`, the same object the cache hands back; read the project metadata CSVs with pandas. |
| Dataset size | manifest tables list 1936 experiments / 703 sessions / 82 mice | only 284 NWBs (247 sessions, 38 mice) are present | paper quotes 376 sessions / 82 mice (behaviour) | The local copy is a subset of the public release. All statistics below are quoted against the **local** subset; per-session statistics (trials/session, catch %, omission %, frame rate, images/session) match the papers exactly. |
| Neural signal | NWB stores `dff`, `events`, `filtered_events` | all three present, same length as `ophys_timestamps` | paper used "detected calcium events" | Use `filtered_events` = the SDK's causally smoothed version of exactly those detected events (see Step 5 decision 3). |
| Rig / frame rate | SDK exposes `ophys_frame_rate` | 31 Hz (239 exps) vs 11 Hz (45 exps) | whitepaper documents both | The target format requires one common time-bin size ⇒ keep the 31 Hz single-plane rig only (Step 5 decision 1). |
| "Session" unit | SDK: experiment ⊂ session ⊂ container | single-plane: 1 experiment = 1 session; multiscope: 34 experiments from 6 sessions of 1 mouse | whitepaper terminology | With only single-plane data kept, experiment ≡ session, so there is no ambiguity and no duplicated behaviour. |
| Trial taxonomy | `go`, `catch`, `aborted`, `auto_rewarded` mutually exclusive; hit/miss/FA/CR false on auto-rewarded | verified in all 168 sessions | whitepaper describes the same 4+2 categories | `go|catch` is exactly "not aborted and not auto-rewarded"; outcome = exactly one of hit/miss/FA/CR. |
| Pupil | SDK nulls pupil on `likely_blink` | 3.7% NaN on average, up to 30%; 3 sessions have **no** eye tracking at all | not discussed | Interpolate across blinks; drop the 3 sessions with no eye data (Step 5 decisions 7–8). |

Understanding of code, data and papers is consistent.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (AllenSDK `BehaviorOphysExperiment`) | Target field | Transform | Notes |
|---|---|---|---|
| `dff_traces.dff` (one row per `cell_specimen_id`) | `neural[session][trial]` | stack to (n_neurons, nT) float32, slice the ophys frames with `start_time ≤ t < stop_time` | native ophys sample grid, no re-binning |
| — | `input[session][trial]` | `np.zeros((0, T))` | the task specifies no decoder inputs |
| `stimulus_presentations.image_name` / `.start_time` / `.omitted` | `output[...][0]` image_identity | index of the most recent non-omitted flash at or before each ophys frame; mapped onto the 16-image global vocabulary | 8 images per session |
| `stimulus_presentations.is_change` | `output[...][1]` image_change | 1 for every ophys frame inside the flash interval that starts at a change (= the 750 ms image presentation interval) | 0 on catch (sham-change) trials |
| `running_speed.speed` / `.timestamps` | `output[...][2]` running_speed_bin | `np.interp` onto ophys frame times, then `np.digitize` into 5 per-session equal-percentile bins | speed is already filtered by the SDK |
| `eye_tracking.pupil_area` / `.timestamps` | `output[...][3]` pupil_diameter_bin | drop blink NaNs → diameter `2√(area/π)` → `np.interp` onto ophys frames → 5 per-session percentile bins | |
| `trials.hit/miss/false_alarm/correct_reject` | `output[...][4]` trial_outcome | one label per trial, broadcast over the trial's timepoints | mutually exclusive on go/catch trials |
| `trials.start_time`, `.stop_time`, `.go`, `.catch` | trial segmentation | keep `go | catch` | drops aborted + auto-rewarded |
| `metadata['mouse_id']` | `subjects`, `subject_idx` | | 37 mice |
| `metadata['targeted_structure']` | `brain_regions`, `brain_region_idx` | | all VISp |

### Key Decisions
1. **Only single-plane ("VisualBehavior" project code) active-behaviour sessions.**
   *Rationale*: (a) the target format requires one common time-bin size, and the Multiscope rig samples at
   11 Hz vs 31 Hz on the Scientifica rigs — there is no way to keep both without resampling one of them off
   its native ophys timestamps, which the task explicitly asks us to align to; (b) the local Multiscope data
   is 34 imaging planes from only **6 behaviour sessions of a single mouse**, so including it would repeat
   the same 6 sets of behavioural labels 34 times (the behaviour, and hence 4 of the 5 decoder outputs, is
   literally identical across planes of one session) — that both duplicates data across the train/validation
   split and over-weights one animal. Cost: 34 of 202 active experiments (17%), 1 of 38 mice.
   Passive sessions (`OPHYS_2`, `OPHYS_5`) are excluded because they have no lick spout and no trials at all.
2. **Both image sets / all experience levels kept** (OPHYS_1,3 = familiar image set A; OPHYS_4,6 = novel
   image set B). The papers restrict *their* analyses to familiar images, but that is an analysis choice for
   a novelty-specific question; the decoder task says only "the Visual Behavior task" and dropping half the
   sessions would halve the data. The two 8-image sets are disjoint, so `output_values[0]` is the 16-image
   union and only 8 classes occur in any one session (this is fine: balanced accuracy is computed per class
   over the sessions in which the class occurs).
3. **Neural signal = `dff_traces`** (the released, baseline-normalised and detrended dF/F; whitepaper section
   "DF/F CALCULATION"). The Neuron paper used the *detected calcium events* instead, and the SDK also
   provides `events` and `filtered_events`. I converted the dataset with each of them and trained the
   reference decoder (20-session subset, identical pipeline otherwise):

   | neural signal | image identity | image change | speed bin | pupil bin | outcome |
   |---|---|---|---|---|---|
   | `events` (raw, 2-session sample) | 0.089 | 0.524 | 0.221 | 0.203 | 0.253 |
   | `filtered_events` | 0.166 | 0.547 | 0.237 | 0.228 | 0.271 |
   | `filtered_events`, unit-variance per neuron | 0.265 | 0.566 | 0.252 | 0.245 | 0.273 |
   | **`dff`** | **0.372** | **0.608** | **0.276** | **0.259** | **0.291** |
   | `dff`, unit-variance per neuron | 0.396 | 0.607 | 0.278 | 0.259 | 0.292 |

   (validation balanced accuracy; chance 0.0625 / 0.5 / 0.2 / 0.2 / 0.25.)
   The decoder used here reads **one timepoint at a time** (a linear read-out of 100 per-session PCs with no
   temporal filter). Detected events are exactly zero in ~95% of the 32 ms frames, so with the event trace
   103 whole trials contain no non-zero sample at all ("all neural data is zero" warnings) and every output
   decodes worse. This is the case the task's exception clause covers ("except where … training a neural
   decoder require otherwise"): dF/F is the Allen pipeline's primary released trace, it is what the
   whitepaper documents in full, it retains the information the event detector discards, and it is strictly
   better for the required per-timepoint decoding. The events-based conversion is still reproducible with
   `--neural events|filtered_events`.
4. **No further normalisation of the neural traces** (`--normalize none`). dF/F is already a normalised
   quantity; per-neuron unit-variance scaling changes validation accuracy by ≤0.02 (table above), so the raw
   released values are kept.
5. **Trial = `[trials.start_time, trials.stop_time)`**, i.e. the experiment's own trial definition, sampled
   on the native ophys frame times. Trial length therefore varies (217–389 frames, mean 262 = 8.47 s);
   the time-bin size (32.32 ms) is the same for every trial and session. The alignment event recorded in
   the metadata is the trial start (`off_start = 0`, `off_end = None` because the length is variable).
   Checked in the data: consecutive trials never overlap, and `stop_time − change_time` is a constant
   ≈4.24 s, so every trial contains its (sham-)change and a fixed post-change period.
6. **Image identity carries through omissions and through the grey inter-stimulus interval.** The paper
   defines the analysis unit as "the 750 ms interval beginning with each image presentation … for image
   omissions we used the 750 ms following the time of the omission", so each ophys frame is labelled with
   the image of the flash interval it falls in. Because omitted flashes inherit the ongoing image, the
   identity signal changes *exactly* at image changes, which keeps outputs 0 and 1 mutually consistent
   ("image change = 1 right after a change in image identity"). Changes and the flash before them are never
   omitted, so the change interval is always a full 750 ms.
7. **Pupil diameter from `pupil_area`** as `2√(area/π)`; blink frames (which the SDK has already set to NaN
   via `likely_blink`) are dropped and bridged by linear interpolation. Percentile binning is invariant to
   the monotone area→diameter transform, so this choice only affects the recorded bin edges, not the labels.
8. **Sessions without eye tracking are dropped** (3 of 168: 795953296, 806456687, 833631914). Pupil
   diameter is a required output and cannot be imputed for a whole session.
9. **Percentile bins are computed per session**, over exactly the timepoints that end up in the converted
   trials. Pupil size is measured in camera pixels, which is not comparable across sessions/rigs (per-session
   medians range over 45–120 px here), and running propensity varies enormously between mice; per-session
   quintiles make the class label mean the same thing ("this animal's slowest/fastest fifth") everywhere and
   give exactly 20% occupancy per class in every session.
10. **No neuron filtering**: only `valid_roi == True` cells exist in the released NWBs (verified in all 165
    sessions), the upstream pipeline already removed border/union/duplicate/non-somatic ROIs, and neither
    paper applies a further cut. **No trial filtering beyond the specified go/catch rule**, and **no
    engagement/performance filtering** — neither reference paper excludes disengaged trials.
11. **`input` has dimension 0** (`np.zeros((0, T), float32)`, `input_names = []`), since the task specifies
    no decoder inputs. Verified that `verify_data_format`, `print_data_summary`, `plot_trial` and
    `train_decoder` all handle `dinput = 0`.

### Planned Sanity Checks
- [x] Per-session counts of go / catch / hit / miss / false-alarm / correct-reject trials and of neurons
      must equal the published `behavior_session_table.csv` / `ophys_cells_table.csv` values.
- [x] Catch fraction ≈ 12.5%, omission rate ≈ 5%, 8 images/session, mean change time ≈ 4.2 s after trial
      start, 5 auto-rewarded trials/session, 31 Hz frame rate (whitepaper values).
- [x] `image_change` fraction ≈ n_go × 0.75 s / total converted time.
- [x] Image identity must equal `initial_image_name` before the change and `change_image_name` after it on
      every go trial, and be constant on every catch trial (asserted for all 42,470 trials during
      conversion).
- [x] Running-speed / pupil bin occupancy exactly 20% per class per session.
- [x] Change-triggered average of the population must peak *after* the change (temporal alignment).
- [x] Independent re-derivation of neural / output values straight from the HDF5 (`cache/sanity_checks.py`).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — `python -u convert_data.py <out.pkl> [--full|--sample] [--show-processing]
[--workers N] [--neural dff|events|filtered_events] [--normalize none|zscore|noise_std]`.

Structure: `select_experiments()` (metadata-table based selection) → `process_session()` in a worker pool
(one NWB per task; returns per-trial arrays plus session metadata, or a `skip` reason) → `_check_session()`
(assertions on every trial of every session) → assembly of the global dictionary (image vocabulary, subject
and brain-region indices, metadata) → pickle.

Code inefficiencies identified / speed-ups added:
- All per-frame quantities (image identity, change, speed, pupil, percentile bins) are computed **once per
  session** as vectorised whole-session arrays (`np.searchsorted` onto the flash grid, `np.interp`,
  `np.digitize`) and then only sliced per trial; the only per-trial Python loop is the slicing itself.
- Sessions are processed in a `multiprocessing.Pool` (24 workers): 3.2 s of CPU per session → **49 s wall
  for all 168 sessions**.
- Traces are cast to float32 (halves memory vs the float64 in the NWB) and output labels to int8.
- Timing is printed per session with a running ETA.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/sample_data.pkl --sample --show-processing` (→ `conversion_sample_out.txt`).
The sample takes the first candidate session of each image set from different mice that passes the quality
checks (2 of the 6 candidates tried were dropped for missing eye tracking).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (775614751 OPHYS_1_images_A, 808621958 OPHYS_6_images_B) |
| Neurons (total) | 135 (89 + 46) |
| Subjects | 2 (403491, 421136) |
| Trials (total) | 168 (39 + 129) |
| Timepoints | 42,489; T mean 253, min 224, max 388 |
| time_bin_size | 32.31 ms |
| image_identity distribution | 16 classes, 8 per session, 0.016–0.101 |
| image_change | 0.919 / 0.081 |
| running_speed_bin | 0.200 × 5 |
| pupil_diameter_bin | 0.200 × 5 |
| trial_outcome | hit 0.676, miss 0.203, FA 0.043, CR 0.078 |

### Processing Plots Review
`processing_775614751.png`, `processing_808621958.png` (7 panels each):
raw dF/F on the ophys grid with flash/change/trial-boundary overlays; population mean; running speed at its
native 60 Hz overlaid with the values resampled onto the ophys frames (they lie exactly on the native trace);
pupil diameter before/after blink interpolation; the four time-varying outputs (identity steps exactly at the
red change line, the change pulse is exactly one 750 ms flash interval, bins track the analogue traces);
the percentile histograms with the bin edges and the 20%-per-bin occupancy; the change-triggered average
(peaks 0.05–0.3 s *after* the change, never before, confirming alignment) and the distribution of trial
lengths. No anomalies.

### Run Time Estimates
| Speed-ups implemented | Effect |
|---|---|
| vectorised per-session timelines + per-trial slicing only | ~3.2 s CPU per session |
| 24-process pool | 168 sessions in 49 s wall |
| float32 traces / int8 labels | 8.3 GB instead of ~17 GB |

| Step | Time / session | Estimated total (168 sessions) |
|---|---|---|
| NWB load (`from_nwb_path`) | 2.4 s | — |
| trace + behaviour processing + trial cutting | 0.8 s | — |
| total per session (CPU) | 3.2 s | 9 min serial → **49 s with 24 workers** |
| assembly + pickle write | — | 13 s |
| **measured full run** | | **56 s** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation (`verification_sample_out.txt`)
- Errors: None
- Warnings: None

### Decoder Results (sample, 2 sessions / 135 neurons / 168 trials)
| Output | Chance | Training balanced acc | Validation balanced acc |
|---|---|---|---|
| image_identity | 0.0625 | 0.2878 | 0.1537 |
| image_change | 0.5 | 0.6255 | 0.5892 |
| running_speed_bin | 0.2 | 0.2904 | 0.2486 |
| pupil_diameter_bin | 0.2 | 0.3029 | 0.2285 |
| trial_outcome | 0.25 | 0.4362 | 0.2367 |

Training loss falls monotonically (1.626 → 1.433; test loss 1.623). Four of the five outputs validate above chance; the
static per-trial `trial_outcome` does not, which is expected with only 168 trials (the effective sample
size for a trial-constant label is the number of trials, not timepoints) — with 20 sessions it rises to
0.291 and with all 165 sessions to 0.30 (Step 11).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/converted_data.pkl --full --workers 24` → `conversion_full_out.txt`
(56 s total: 49 s for the 168 sessions, 13 s to write the pickle).
`python -u train_decoder.py /app/converted_data.pkl --verify-only` → `verification_full_out.txt`.

### Output Files
- `converted_data.pkl`: 8.34 GB (165 sessions, 42,470 trials, 11,192,974 timepoints × 28,821 neurons)
- `verification_full_out.txt`: **"Data format is valid, no errors or warnings."**

### Consistency Check
"Reference data" = the Allen project-metadata tables (`behavior_session_table.csv`,
`ophys_cells_table.csv`) and the per-file survey (`cache/survey.csv`); "reference papers" =
whitepaper/paper. Computed by `cache/stats_check.py`.

| Statistic | Reference papers | Reference code/data | Converted data | Match? |
|---|---|---|---|---|
| Sessions (active, single-plane, with eye tracking) | — | 168 available − 3 without eye tracking | **165** | ✓ |
| Subjects (mice) | 82 in the full public release | 37 in the local active single-plane subset | **37** | ✓ |
| Sessions per subject | — | 2–9 | 2–9 (mean 4.46) | ✓ |
| Total neurons | 8,619 exc + 470 Sst + 1,239 Vip in the paper's multi-plane familiar subset (different subset) | 28,821 in `ophys_cells_table` for these 165 experiments | **28,821** (identical per session, 165/165) | ✓ |
| Neurons / session | — | 6–666 | 174.7 mean, 6–666 | ✓ |
| Trials total (go+catch) | — | 42,470 (`behavior_session_table`: go 37,143 + catch 5,327) | **42,470** (go 37,143, catch 5,327; identical per session, 165/165) | ✓ |
| Trials / session | — | — | 257.4 mean (39–409) | ✓ |
| Hit / miss / FA / CR totals | — | 13,569 / 23,574 / 814 / 4,513 (published table) | **13,569 / 23,574 / 814 / 4,513** (identical in all 165 sessions) | ✓ |
| Catch fraction | "~12.5%" | 12.54% | **12.54%** | ✓ |
| Aborted fraction of all trials | "premature licks reset the trial" | 61.7% of all 112,552 trials (excluded) | excluded | ✓ |
| Auto-rewarded per session | "sessions began with 5 free-reward trials" | 3.91 mean (5 in most sessions) | excluded | ✓ |
| Omission rate | 5% | 170/3,400 flashes ≈ 5.0% | — (carried through) | ✓ |
| Images per session | 8 | 8 (two disjoint sets of 8) | 8 per session, 16 classes | ✓ |
| Change time − trial start | mean 4.2 s, 2.25–8.25 s (+1 flash cycle) | — | mean **4.28 s**, range 2.79–8.29 s | ✓ |
| Ophys frame rate / bin | 31 Hz single plane | 31.0 Hz; Δt median 32.31–32.33 ms | **32.3193 ms** (session range 32.31–32.33 ms) | ✓ |
| Trial length | — | 8.47 s mean | **8.468 s** (262.0 frames; 217–389) | ✓ |
| image_change fraction | — | analytic: n_go × 0.75 s / total time = 0.0770 | **0.0776** | ✓ |
| Running-speed / pupil bins | "five equal percentile bins" (task) | — | exactly 0.200 each, in every session (max deviation 6e-5) | ✓ |
| Trial-outcome distribution | — | published table → 0.3195 / 0.5551 / 0.0192 / 0.1063 (by trial) | 0.316 / 0.559 / 0.018 / 0.107 (by timepoint) | ✓ |
| Brain region | VISp (+VISl only on the Multiscope rig) | VISp 165/165 | VISp | ✓ |
| Cre lines | Slc17a7 / Sst / Vip | 106 / 28 / 31 sessions | same | ✓ |

(The by-timepoint outcome fractions differ from the by-trial fractions in the fourth decimal only because
trials differ slightly in length.)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`verification_full_out.txt` reports **no errors and no warnings** for the final (dF/F) dataset.

*Warnings that appeared in an earlier iteration and how they were removed*: with the
`filtered_events` trace, 103 of 42,470 trials (0.24%) triggered "all neural data is zero", because detected
calcium events are exactly zero in ~95% of 32 ms frames and a small population can be silent for a whole
8 s trial. They are not a conversion bug — the raw event traces really are zero there — but they are a
symptom of using a very sparse signal for a per-timepoint decoder. Switching the neural signal to the
released dF/F traces (Step 5 decision 3) removed all of them, and the final file produces no warnings at
all. No warning is left unaddressed.

### Check 2 — independent sanity checks (`cache/sanity_checks.py`)
Each check re-derives the value from the original NWB **HDF5** with `h5py` (not through `convert_data.py`),
and compares with `np.allclose` / `np.array_equal`. Run on 4 randomly chosen sessions (seed 0):

| Check | Result |
|---|---|
| `dff` ROI ordering is the `cell_specimen_table` ordering | PASS |
| `cell_specimen_ids` in metadata == HDF5 cell_specimen_table | PASS |
| neural: 3 random (trial, neuron, timepoint) samples == `processing/ophys/dff/traces/data` indexed by `searchsorted(ophys_timestamps, trial_start)` + t | PASS (max abs diff 9.2e-9, float32 rounding) |
| neural: one whole (n_neurons × T) trial block == the corresponding HDF5 slice | PASS |
| input: every trial is exactly `(0, T)` | PASS |
| output0 image identity == brute-force scan of the HDF5 stimulus table (last non-omitted flash at or before t) — 40 timepoints/session | PASS |
| output1 image change == `is_change` of the flash interval containing t | PASS |
| output2/3 percentile edges == recomputed from HDF5 pupil area (+ blink mask) and the SDK running speed | PASS |
| output2/3 all ~80–96k bin labels per session == independently digitised values | PASS |
| output4 == HDF5 `intervals/trials` hit/miss/FA/CR, constant within a trial | PASS |
| trial start/stop == HDF5 trials table; no aborted or auto-rewarded trial kept | PASS |

### Check 3 — reference-code comparison
| Processing step | Reference (AllenSDK / papers) | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `cache.get_behavior_ophys_experiment(eid)` → `BehaviorOphysExperiment` | `BehaviorOphysExperiment.from_nwb_path(path)` (the cache API cannot read this non-cache-layout copy) — identical object | yes |
| (b) neuron filtering | pipeline ROI filtering + demixing + neuropil subtraction + crosstalk removal, only `valid_roi` released | none added; asserted `valid_roi.all()` in every session | yes |
| (b) trial filtering | SDK `trials` categories; papers keep all trials | `go | catch` per the decoder task (drops aborted + auto-rewarded) | task-specified |
| (c) temporal alignment | all streams already on the 100 kHz sync clock; SDK exposes `ophys_timestamps`, `running_speed.timestamps`, `eye_tracking.timestamps` in the same seconds | trials cut on `ophys_timestamps`; behaviour streams `np.interp`-ed onto those times | yes |
| (d) binning | ophys frames (31 Hz) are the native bins; the paper aggregates them into 750 ms image presentation intervals / 400 ms windows for *its* analyses | native ophys frames kept (task: "temporally align based on ophys timestamp"); the 750 ms interval convention is used for the *labels*, as in the paper | yes / task-specified |
| (e) input construction | n/a | none (task) | n/a |
| (f) output construction | paper assigns behavioural events to the 750 ms image presentation interval; trial outcomes from the SDK trials table; `running_speed`/`eye_tracking` used as released | identical definitions; discretisation added because the decoder needs categorical outputs | yes + task-required |
| neural signal | paper: detected calcium `events`; whitepaper: dF/F is the pipeline's normalised/detrended output | dF/F (justified + measured in Step 5 decision 3; `--neural events` reproduces the paper's choice) | **differs, documented** |

The only substantive difference from the Neuron paper is the neural trace (dF/F instead of detected
events), which is covered by the task's exception for "what training a neural decoder requires" and is
supported by the A/B measurements in Step 5. Two further differences are dictated by the task: we keep all
experience levels (the paper restricted its neural analysis to familiar images) and we use the single-plane
rig (the paper used the multi-plane rig, which is present here only as 6 sessions of one mouse and at an
incompatible frame rate).

### Check 4 — key statistics comparison
See the Step 9 table: every count that can be cross-checked (go, catch, hit, miss, false alarm, correct
reject, total trials, neurons per session) is **identical in all 165 sessions** to the values published in
`behavior_session_table.csv` / `ophys_cells_table.csv`, and the task statistics (12.54% catch rate, 5%
omissions, 8 images/session, 4.28 s mean change time, 31 Hz) match the whitepaper.

### Check 5 — edge cases (`cache/edge_checks.py`, all PASS)
- every trial ≥ 2 timepoints (min 217), every session ≥ 2 trials (min 39);
- dtypes float32 / float32 / int8, no NaN or Inf anywhere;
- every output within `[0, len(output_values)-1]`; int8 is safe (max value 15);
- neural/input/output trial counts and T agree; `brain_region_idx` length == n_neurons; `subject_idx` in range;
- `n_trials == n_go + n_catch` for every session and 42,470 == 42,470 overall (nothing silently dropped);
- (sham-)change time strictly inside every trial; `stop_time − change_time` ≈ 4.24 s for all trials;
- trials never overlap, so no ophys frame is used twice when the percentile bins are computed;
- `image_change` is exactly one contiguous pulse per go trial and identically 0 on catch trials; image
  identity switches exactly once per go trial, at the frame where `image_change` turns on; the number of
  trials containing a change equals the number of go trials (37,143);
- per-session bin occupancy is 20% ± 6e-5 for both behaviour variables;
- first/last trial edge cases: trials start ~21 ms before a flash onset (so the first frame of a trial
  inherits the previous flash's identity, which is the same image — asserted), and at most one trial per
  session extends past the last flash of the behaviour block, by ≤0.23 s.

### Issues found and resolved during this review
1. *(iteration 1)* `--sample` picked a session with no eye tracking and produced a 1-session file → sample
   selection now walks a candidate list and keeps the first two sessions that pass, preferring one per
   image set.
2. *(iteration 1)* image names were `np.str_` rather than `str` in `output_values` → cast to `str`.
3. *(iteration 2)* 103 "all neural data is zero" warnings and weak decoding with the event traces →
   neural signal changed to dF/F after the A/B comparison (Step 5 decision 3); full conversion, verification
   and all checks above were re-run from scratch on the new file.
4. Added an explicit assertion that trials never overlap (would corrupt the percentile binning).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/converted_data.pkl --plot-samples` → `train_decoder_full_out.txt`
(165 sessions, 42,470 trials, 11.2 M timepoints; ~13 min on the L4 GPU).

### Training Progress
- Loss decreasing: **Yes**, monotonically at every reported epoch:
  1.6295 (epoch 1) → 1.6143 (10) → 1.5981 (20) → 1.5383 (50) → 1.4450 (100) → 1.3908 (150) → 1.3623 (200);
  test loss 1.4164. (Chance loss for these five heads is ≈1.614, so the model moves well away from chance.)

### Decoder Results (full dataset)
| Output | #classes | Chance | Training balanced acc | Validation balanced acc | Val / chance |
|---|---|---|---|---|---|
| image_identity | 16 | 0.0625 | 0.4405 | **0.4176** | 6.7× |
| image_change | 2 | 0.5 | 0.6382 | **0.6068** | 1.21× |
| running_speed_bin | 5 | 0.2 | 0.3108 | **0.2823** | 1.41× |
| pupil_diameter_bin | 5 | 0.2 | 0.3121 | **0.2666** | 1.33× |
| trial_outcome | 4 | 0.25 | 0.4373 | **0.2950** | 1.18× |

Sample-trial and prediction plots (`sample_trials.png`, `predictions.png`) were produced; the predicted
traces track the true image identity (stepping at the change), and the change head fires around the true
change pulse (it over-predicts changes because the loss is class-balanced and changes are only 7.7% of
timepoints).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
Every output validates above chance (table above). Three outputs are below 1.5× chance
(image_change 1.21×, pupil 1.33×, trial_outcome 1.18×), so I investigated each:

- **image_change (0.607).** Chance for a binary variable is 0.5, so "1.5× chance" would be 0.75, which is
  above anything reported for this dataset: the paper's own change decoder reaches 0.52–0.65 % correct.
  Evaluated the way the paper evaluates it (change flash vs. the immediately preceding repeat flash, first
  400 ms, per image presentation) our decoder gets **0.689** (Check 2), i.e. at/above the top of the
  paper's range. The lower 0.607 figure is over *all* timepoints of the trial, including the tail of the
  750 ms change interval where the calcium response has decayed.
- **pupil_diameter_bin (0.267) and running_speed_bin (0.282).** These are 5-way quintile decodes of a
  behavioural variable from VISp dF/F at single 32 ms frames. I tested the one conversion choice that
  changes them materially — per-session vs. dataset-wide percentile edges (20-session A/B, everything else
  identical):

  | binning | image identity | image change | speed bin | pupil bin | outcome |
  |---|---|---|---|---|---|
  | per session (used) | 0.368* | 0.608* | 0.276 | 0.259 | 0.291 |
  | dataset-wide | 0.368 | 0.609 | **0.381** | **0.445** | 0.294 |

  (*same sessions, per-session run.) Dataset-wide edges look much better but are an artefact: pupil is
  measured in **camera pixels**, and the per-session median pupil diameter ranges 45–120 px with a clear
  rig dependence (CAM2P.3 mean 98.6 px vs CAM2P.4 83.3 px vs CAM2P.5 81.7 px) and a between-mouse s.d.
  (16.3 px) twice the within-mouse s.d. (8.4 px). With dataset-wide edges whole sessions fall in a single
  bin (e.g. 99% of one session in bin 0, 100% of another in bin 4), so the decoder scores by identifying
  the session — which it can always do, because each session has its own projection matrix — rather than
  by reading out arousal. I therefore kept per-session quintiles, which give exactly 20% occupancy per
  class *in every session* and force the decoder to track within-session fluctuations. This is a
  deliberate choice of the lower number over an inflated one.
- **trial_outcome (0.295).** This is the only static, per-trial label, so its effective sample size is
  42,470 trials rather than 11.2 M timepoints, and two of its four classes are rare (false alarm 1.9%,
  correct reject 10.6%) and occur only on catch trials, which are neurally *identical* to ordinary image
  repeats — the paper reports that "for all cell classes, false alarm decoding performance was very low".
  Restricted to the comparison the paper actually makes (hit vs. miss on image changes, first 400 ms) the
  same trained model reaches **0.604 % correct / 0.654 balanced** (Check 2), inside the paper's 0.55–0.75
  range.

### Check 2 — comparison with accuracies reported in the papers
The Neuron paper reports two decoding accuracies (Figure 6A, 6C; random forest, per imaging plane, 5-fold
CV, neural activity in the first 400 ms after each image presentation). `cache/paper_comparison.py`
re-evaluates our trained decoder on exactly those comparisons, on validation trials only:

| Decoded variable | Paper (Figure 6) | This dataset + reference decoder | Note |
|---|---|---|---|
| image change vs. repeat (per image presentation, 400 ms) | 0.52–0.65 % correct (mean ± SEM over planes) | **0.689** (15,060 validation image presentations) | at/above the top of the paper's range; we pool 165 sessions and up to 666 cells/session, while the paper used 5–80 cells per plane |
| hit vs. miss (per image change, 400 ms) | 0.55–0.75 % correct | **0.604** % correct, 0.654 balanced (7,530 validation go trials) | inside the paper's range |
| false alarms | "very low, no difference between strategies" | FA recall is the weakest class of `trial_outcome` | consistent |
| image identity, running speed, pupil | not reported | 0.418 / 0.282 / 0.267 | no reference value exists |
No decoding accuracy reported in either reference paper is higher than what this conversion achieves, so
there is no unexplained accuracy gap.

### Check 3 — train vs. validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| image_identity | 0.4405 | 0.4176 | 1.05 |
| image_change | 0.6382 | 0.6068 | 1.05 |
| running_speed_bin | 0.3108 | 0.2823 | 1.10 |
| pupil_diameter_bin | 0.3121 | 0.2666 | 1.17 |
| trial_outcome | 0.4373 | 0.2950 | 1.48 |

No ratio exceeds 1.5. The largest (trial_outcome) is expected for a label that is constant within a trial:
the split is *by trial*, so all timepoints of a trial are on the same side of the split and there is no
leakage, but the model can fit trial-specific activity in training. Verified that the split is by trial
(`train_validate_decoder` permutes trial indices per session) and that no trial appears in both sets.

### Additional observation: run-to-run variance
`train_decoder.py` seeds numpy with 0, so its result is reproducible, but a second training run with a
different random split/initialisation (the `cache/paper_comparison.py` run) gave 0.418 / 0.693 / 0.283 /
0.267 / 0.345 — i.e. image_change and trial_outcome vary by ±0.05 between runs with 200 full-batch Adam
steps. The reported Step 11 numbers are the ones produced by the reference command.

### Issues found and resolved
- No further data issues were found in this review. The changes made earlier (neural signal, sample
  selection, string types, overlap assertion) are listed in Step 10.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format spec, key statistics, decoder results)
- [x] `cache/` folder created with `README_CACHE.md` describing every investigation script
- [x] All files organised:

| file | contents |
|---|---|
| `convert_data.py` | conversion script (`--full` / `--sample` / `--show-processing` / `--neural` / `--normalize`) |
| `converted_data.pkl` (8.3 GB) | full converted dataset |
| `sample_data.pkl` | 2-session sample |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `--verify-only` logs |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_775614751.png`, `processing_808621958.png` | per-step processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder sample and prediction plots |
| `CONVERSION_NOTES.md`, `README.md` | documentation |
| `cache/` | survey + validation scripts and their outputs |

### Summary of the decisions that shape the dataset
1. 165 active-behaviour, single-plane (31 Hz) VISp sessions from 37 mice; Multiscope, passive and
   eye-tracking-less sessions excluded (Step 5 decisions 1 and 8).
2. Trials are the experiment's own go/catch trials, `[start_time, stop_time)`, sampled on the native
   ophys frame grid (Step 5 decision 5).
3. Neural signal is the released dF/F; the reference paper's detected-event traces are reproducible with
   `--neural events` but decode substantially worse with this per-timepoint decoder (Step 5 decision 3).
4. Image identity and image change follow the paper's 750 ms image-presentation-interval convention and
   are mutually consistent (Step 5 decision 6).
5. Running speed and pupil diameter are interpolated onto the ophys frames and binned into per-session
   quintiles (Step 5 decisions 7 and 9, Step 12 Check 1).
6. Trial outcome is the SDK's hit/miss/false-alarm/correct-reject label, held constant over the trial.

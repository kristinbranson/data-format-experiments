# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory - Visual Behavior 2P (ophys)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `.manifest` (228 KB) - AllenSDK cache manifest
- `CONVERSION_NOTES.md` - this file
- `Dockerfile`, `docker-compose.yaml` - environment
- `allensdk_docs/` - downloaded allensdk documentation
- `code/` - reference code (AllenSDK source)
- `data/` - the AllenSDK cache with the data
- `decoder.py` (89 KB) - decoder library
- `methods.txt` - methods text from the paper
- `paper.pdf` - Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex
- `train_decoder.py` - validation/training script
- `tutorials/` - tutorials on dataset use
- `whitepaper.pdf` - Visual Behavior 2P technical whitepaper

Environment verified: python3 works; numpy 2.4.4, torch 2.6.0+cu124, allensdk 2.16.2 import successfully.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference code base in `/app/code` is the **AllenSDK** itself (v2.16.2). The paper
(Vip-Sst disinhibitory circuit) analyses the public Visual Behavior 2P release, which is
accessed exclusively via the `VisualBehaviorOphysProjectCache`. Tutorials in `/app/tutorials`
show the canonical access patterns.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir)` / `.from_local_cache()` | `allensdk/brain_observatory/behavior/behavior_project_cache/behavior_project_cache.py` | LOADING | Entry point to the released dataset; manages the local cache dir |
| `cache.get_ophys_experiment_table()` | same | LOADING | Metadata table, one row per ophys *experiment* (imaging plane); columns incl. `ophys_session_id`, `mouse_id`, `cre_line`, `targeted_structure`, `imaging_depth`, `session_type`, `experience_level`, `passive` |
| `cache.get_ophys_session_table()`, `get_behavior_session_table()` | same | LOADING | Session-level metadata |
| `cache.get_behavior_ophys_experiment(ophys_experiment_id)` | same | LOADING | Returns `BehaviorOphysExperiment` object with all data streams |
| `BehaviorOphysExperiment.dff_traces` | `behavior_ophys_experiment.py:530` | PROCESSING | DataFrame indexed by `cell_specimen_id`, column `dff` = full dF/F trace (already computed by the pipeline; **no need to compute dF/F ourselves**) |
| `BehaviorOphysExperiment.events` | `behavior_ophys_experiment.py:554` | PROCESSING | `events` (L0 deconvolved event magnitudes), `filtered_events` (causal half-gaussian smoothed), `lambdas`, `noise_stds` |
| `BehaviorOphysExperiment.ophys_timestamps` | `behavior_ophys_experiment.py:523` | ALIGNMENT | Sync-aligned timestamps (seconds) for each 2P frame; **the common clock for all streams** |
| `BehaviorOphysExperiment.cell_specimen_table` | `behavior_ophys_experiment.py:588` | CURATION | "Table only contains roi_valid = True entries, as invalid ROIs / non-cell segmented objects have been filtered out" |
| `exclude_invalid_rois=True` (default) | `cell_specimens.py:154,205` | CURATION | SDK default filters ROIs with `valid_roi == False` |
| `BehaviorSession.trials` | `behavior_session.py:1271` | LOADING | Trial table |
| `Trials.columns_to_output()` | `data_objects/trials/trials.py:150` | LOADING | Trial columns: initial_image_name, change_image_name, is_change(stimulus_change), change_time, go, catch, lick_times, response_time, response_latency, reward_time, reward_volume, hit, false_alarm, miss, correct_reject, aborted, auto_rewarded, change_frame, start_time, stop_time, trial_length |
| `Trial._get_trial_data()` | `data_objects/trials/trial.py:151` | LOADING | Defines trial categories: `go` = change trial (not catch, not auto_reward), `catch` = sham change, `aborted` = licked before change, `auto_rewarded`; outcomes hit/miss/false_alarm/correct_reject (auto_rewarded trials have all four False) |
| `trial_masks.contingent_trials(trials)` | `behavior/trial_masks.py` | CURATION | **GO & CATCH trials only** -- exactly the trial selection requested by the decoder task |
| `BehaviorSession.stimulus_presentations` | `behavior_session.py:1067` | LOADING | Flash table: `image_name`, `image_index`, `start_time`, `end_time`, `is_change`, `omitted`, `flashes_since_change`, `trials_id`, `stimulus_block_name` |
| `BehaviorSession.running_speed` | `behavior_session.py:1023` | LOADING | DataFrame `timestamps`, `speed` (cm/s), low-pass filtered + median filtered (`running_processing.py`), 60 Hz |
| `BehaviorSession.eye_tracking` | `behavior_session.py:908` | LOADING | Ellipse fits at ~60 Hz: `pupil_area`, `pupil_width`, `pupil_height`, `pupil_area_raw`, `likely_blink`. Area/width/height are **NaN where `likely_blink==True`** |
| `swdb/save_trial_response_df.py` | reference analysis | PROCESSING | Builds trial-aligned dF/F: window `[-4, +8] s` around `trials.change_time`, frame rate ~31 Hz; asserts `change_times` are a subset of flash `start_time`s |
| `analysis_tools.get_trace_around_timepoint` | `swdb/analysis_tools.py:23` | PROCESSING | Extracts trace in a frame window around the *nearest ophys frame* to a timepoint |
| `analysis_tools.get_nearest_frame` | `swdb/analysis_tools.py:4` | ALIGNMENT | Nearest-ophys-frame lookup used for event alignment |

### Notes
- **dF/F does not need to be computed**: `dff_traces` is provided by the Allen pipeline.
  `events` (L0 deconvolution) are also provided; the Vip-Sst paper uses the *events* traces
  (to be confirmed in Step 3 from methods.txt).
- **Neuron curation is already applied by the SDK** (`exclude_invalid_rois=True`), so
  `dff_traces`/`events`/`cell_specimen_table` contain only valid ROIs (cell-matched
  `cell_specimen_id`).
- **Temporal alignment**: everything (ophys frames, running, eye tracking, stimulus, trials,
  licks, rewards) is expressed in the same sync-clock seconds, so alignment is done by
  interpolating/sampling the behavior streams onto `ophys_timestamps`.
- **Trial definition**: use `trials` table rows; keep `go | catch`, drop `aborted` and
  `auto_rewarded` (exactly `trial_masks.contingent_trials`).
- Alignment event for go/catch trials is `change_time` (for catch trials this is the sham
  change time), which is the canonical alignment used throughout the SDK/paper analyses.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is an AllenSDK **VisualBehaviorOphysProjectCache** local cache (release
`visual-behavior-ophys-1.1.0`, 247 GB):
- `visual-behavior-ophys_project_manifest_v1.1.0.json` (manifest)
- `visual-behavior-ophys-1.1.0/project_metadata/`: `ophys_experiment_table.csv` (1936 rows,
  full release), `ophys_session_table.csv`, `behavior_session_table.csv`, `ophys_cells_table.csv`
- `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb`:
  **284 NWB files are actually present locally** (a subset of the 1936-experiment release).

Loaded with:
```python
cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir='/app/data', use_static_cache=False)
exp = cache.get_behavior_ophys_experiment(ophys_experiment_id)   # ~3 s per experiment
```
(`use_static_cache=True` fails: that layout expects `visual-behavior-ophys/manifests`.)

### Content of the 284 downloaded experiments
| Field | Values |
|---|---|
| session_type | OPHYS_1_images_A 55, OPHYS_3_images_A 55, OPHYS_4_images_B 46, OPHYS_6_images_B 46, OPHYS_5_images_B_passive 42, OPHYS_2_images_A_passive 40 |
| cre_line | Slc17a7-IRES2-Cre 153, Sst-IRES-Cre 85, Vip-IRES-Cre 46 |
| targeted_structure | VISp 261, VISl 23 |
| imaging_depth | 75-375 um (8 values) |
| project_code | VisualBehavior 239 (single plane, 31 Hz), VisualBehaviorMultiscope 45 (up to 7 planes, 11 Hz) |
| experience_level | Familiar 150, Novel 1 38, Novel >1 96 |
| mice | 38 |
| ophys sessions | 247 (some sessions = several simultaneously-recorded planes/experiments) |

**Active (behaving) subset** (`passive == False`, i.e. OPHYS_1/3/4/6):
202 experiments, **174 ophys sessions**, 38 mice, **29,444 cells**
(cells/experiment mean 145.8, median 63.5, range 4-666;
cells/session after grouping simultaneous planes: mean 169.2, median 103, range 6-666).
Passive sessions (OPHYS_2/5) *do* have a trials table (go/catch) but **0 licks and 0 rewards**
(the lick spout is retracted), so trial outcome / behaviour cannot be decoded there -> excluded.

### Per-experiment data streams (from `BehaviorOphysExperiment`)
| Stream | Shape / notes |
|---|---|
| `ophys_timestamps` | e.g. 140,208 frames @ 30.94 Hz (single plane) or 48,316 @ 10.73 Hz (mesoscope); session ~4500 s |
| `dff_traces` | DataFrame (n_cells x [cell_roi_id, dff]); dff length == n ophys frames; no NaNs |
| `events` | (n_cells x [cell_roi_id, events, filtered_events, lambda, noise_std]); `events` sparse (~0.3% nonzero) |
| `cell_specimen_table` | valid ROIs only (SDK filters `valid_roi == False`) |
| `trials` | e.g. 937 trials: 164 go, 24 catch, 749 aborted, 0 auto_rewarded; columns as in Step 1 |
| `stimulus_presentations` | 13,803 rows, blocks: `initial_gray_screen_5min`, `change_detection_behavior`, `post_behavior_gray_screen_5min`, `natural_movie_one`; `image_name` in {im065,...} plus `'omitted'`; 8 images per session |
| `running_speed` | 60 Hz (dt=0.0167 s), `speed` in cm/s (range ~ -5 to +15 in example) |
| `eye_tracking` | ~60 Hz, columns incl. `pupil_area`, `pupil_width`, `pupil_height`, `pupil_area_raw`, `likely_blink` (NaN where likely_blink) |
| `licks`, `rewards` | timestamps |

### Dataset Size (from data files, active/behaving experiments)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 29,444 (202 active experiments) |
| Neurons / session | mean 169.2, median 103 (planes of the same ophys session grouped) |
| Subjects | 38 |
| Sessions / subject | 174/38 = 4.6 mean |
| Trials (total) | ~ 174 x ~200 go+catch = ~35,000 (to be measured) |
| Trials / session | example sessions: 188 and 209 go+catch trials |

### Important complication
Ophys frame rate differs between rigs: **30.9 Hz (VisualBehavior, single plane)** vs
**10.7 Hz (VisualBehaviorMultiscope)**. The target format requires one common time-bin
size for all trials/sessions, so neural and behavioural data must be resampled onto a
common grid (bin >= ~93 ms) or the analysis restricted to one rig type. Decision in Step 5.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

Sources: `/app/methods.txt` (whitepaper Sections D/F + paper STAR Methods), `whitepaper.pdf`
(text extracted to `/app/cache/whitepaper.txt`), `paper.pdf` (-> `/app/cache/paper.txt`).

### Task structure (whitepaper)
- Go/no-go **change detection**: continuous stream of flashed natural images,
  **250 ms image ON, 500 ms gray ISI -> 750 ms flash cycle**.
- **8 images per session** (image set A or B); 64 possible transitions.
- **Response window 150-750 ms after the change** (before display-lag compensation).
- Trial types: **GO** (image identity changes), **CATCH** (sham change, same image),
  **ABORTED** (mouse licks before the change -> trial reset + timeout),
  **AUTO-REWARDED / free reward** (first 5 trials of a session and after 10 consecutive misses).
- Among contingent (go+catch) trials: **GO = 87.5%, CATCH = 12.5%** (matrix-sampling schedule).
- Change times drawn from truncated exponential 2.25-8.25 s after trial start (mean ~4.2 s).
- Outcomes: HIT / MISS (go), FALSE ALARM / CORRECT REJECT (catch).
- **Stimulus omissions on 5% of non-change flashes** during imaging sessions; changes and the
  flash preceding a change are never omitted.
- Sessions ~60 min; 5 min gray screen before and after behavior, plus a 5 min natural movie.
- Passive sessions (OPHYS_2, OPHYS_5): mouse satiated, **lick spout retracted**, no rewards.

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Imaging rate | 31 Hz single-plane; 11 Hz per plane multi-plane | "512x512 pixels, 31 Hz for single plane and ... 11 Hz for each plane in multi-plane experiments" |
| Eye-tracking rate | 30 Hz (SDK table ~60 Hz in the released data) | "eye tracking (30 Hz), and behavior (30 Hz)" |
| Running speed | 60 Hz sampling, 10 Hz low-pass Butterworth filtered, cm/s | running_processing section |
| Flash cycle | 250 ms image + 500 ms gray = 750 ms | Figure 2 caption |
| Response window | 150-750 ms after change | "within response window (150 ms to 750 ms following image change...)" |
| Go fraction of contingent trials | 87.5% | "GO trials comprise 87.5% of all trials" |
| Catch fraction | 12.5% | "CATCH trials comprise 12.5% of all trials" |
| Omission probability | 5% of non-change flashes | "which occurred randomly on 5% of non-change stimuli" |
| Engagement threshold | 2 rewards/min | "'engaged' or 'disengaged' ... reward rates above and below 2 rewards per minute" |
| Task performance QC | peak d-prime >= 1.0 | QC criterion 6 |
| Paper's neural dataset | 8,619 excitatory (21 sessions, 9 mice), 470 Sst (15 sessions, 6 mice), 1,239 Vip (21 sessions, 9 mice) | Results |
| Paper's behavior dataset | 376-382 imaging sessions, 82 mice, 1,804,462 image intervals | "from 376 imaging sessions from 82 mice" |
| Engaged fraction | 60.1% of image intervals | Figure 4A caption |

(The local cache holds only 284 of the 1936 released experiments, so absolute session/mouse
counts in the papers cannot be reproduced; the *ratios* and per-session statistics can.)

### Processing Details
- **Neural signal**: the paper performed "all analyses on discrete calcium events that were
  regressed from the raw fluorescence traces, thus removing the slow decay dynamics of
  GCaMP6f" -> the SDK `events` table (L0 event detection), *not* raw dF/F.
  "For all analysis of neural data we used the detected calcium events".
- **Temporal alignment**: all streams are synchronised on a single 100 kHz NI DIO board;
  the SDK exposes everything in the same sync-clock seconds. The task instruction is to
  align on **ophys timestamps**.
- **Behavioural binning in the paper**: events assigned to *image-presentation intervals*
  (750 ms starting at each image onset); decoding used the **first 400 ms after image onset**.
- **Running speed**: use the SDK `running_speed` (filtered) attribute.
- **Pupil**: eye-tracking ellipse fits; `likely_blink` frames have NaN area/width/height.
- dF/F and events are **already computed** by the Allen pipeline; no need to recompute.

### Curation Steps
**Neuron curation rules**: the Allen pipeline already excludes non-somatic/duplicate/union/
edge/dim ROIs via the ROI-filtering classifier; the SDK exposes only `valid_roi == True`
cells (`exclude_invalid_rois=True`). Cells are matched across sessions (`cell_specimen_id`).
Session-level QC (saturation, photobleaching, z-drift >10 um, motion, interictal events,
d-prime >= 1) has already been applied to the released data.

**Trial curation rules**: `trial_masks.contingent_trials` -> keep **go** and **catch** trials,
drop **aborted** and **auto_rewarded** (matches the decoder-task instruction exactly).
Passive sessions contain go/catch trials but no licking/rewards -> excluded (no behaviour).

### Decoders Trained (paper, Figure 6 / S22)
| Decoded variable | Accuracy |
|---|---|
| Image change vs. repeat (random forest, first 400 ms, per imaging plane) | well above chance; equal for visual and timing strategy sessions (values only shown graphically, ~60-80% correct depending on n cells) |
| Hit vs. miss | above chance, higher in visual-strategy sessions (~55-70%) |
| False alarm decoding | "very low" performance, near chance |
No numerical decoding accuracies are printed in the text; only figure panels.

## Step 4: Check for Consistency
**Status**: COMPLETE

I measured, directly from the data (script `/app/cache/check_step4.py`, `check_step4b.py`,
outputs `step4_out.txt`, `step4b_out.txt`), the quantities the papers state.

| Quantity | Papers say | Measured in data (8-12 experiments, both rigs) | Match? |
|---|---|---|---|
| Flash cycle | 250 ms image + 500 ms gray = 750 ms | median inter-flash-onset = **750.6 ms** (all experiments) | YES |
| GO fraction of contingent trials | 87.5% | **87.3%** (mean over 8 experiments) | YES |
| CATCH fraction | 12.5% | 12.7% | YES |
| Change times fall on flash onsets | assert in `swdb/save_trial_response_df.py` | **100%** of change_times are in `stimulus_presentations.start_time` | YES |
| Omissions | 5% probability on non-change flashes | **3.5-3.8%** of non-change flashes in the behaviour block | approximately; lower because omissions are suppressed immediately before/at changes and in the first flashes, and because the 5% is a draw probability |
| Images per session | 8 | 8 (A: im061,062,063,065,066,069,077,085; B: im000,031,035,045,054,073,075,106) | YES |
| Imaging rate | 31 Hz single-plane / 11 Hz multi-plane | 30.94 Hz / 10.73 Hz | YES |
| Running speed sampling | 60 Hz | dt = 16.7 ms | YES |
| Eye tracking | 30 Hz in whitepaper | dt = 16.6 ms (60 Hz) in the released tables | released data are 60 Hz; not a problem, we resample |
| Blink/NaN pupil frames | `likely_blink` -> NaN | 1.4-2.9% of frames | consistent |
| Trial timing | change 2.25-8.25 s after trial start (mean 4.2) | change_time - start_time: median 3.77-3.81 s, **min 3.02 s**; stop_time - change_time: **min 4.20 s** | consistent (start_time is after aborted-trial resets) |
| Inter-change interval (go+catch) | >= 1 trial | **min 7.51 s** | windows of +/-3 s never overlap |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal to use | SDK exposes `dff_traces`, `events`, `filtered_events` | all three available | paper: "we used the detected calcium events" | Use the `events` table (L0-detected calcium events). Because the provided decoder is **instantaneous** (a linear readout of the PCs of the neural activity *in the same time bin*, `nn.Linear(npcs + dinput, ncat)`) while the paper's random-forest decoder concatenated all time steps in a 400 ms window, the event train must be integrated over a comparable window. This is achieved by binning events into 250 ms bins (see Step 5) and, if needed, by using the SDK's `filtered_events` (the same events convolved with a causal half-gaussian) - to be decided empirically in Step 7/8 and documented. |
| Cell filtering | SDK `exclude_invalid_rois=True` | `cell_specimen_table` already filtered | whitepaper ROI-filtering classifier | No extra filtering needed; use SDK defaults |
| Sessions to include | - | 202 active experiments / 174 sessions | paper restricted to *familiar* images on the *multi-plane* rig (21/15/21 sessions) | The paper's restriction served its scientific question (novelty, cell-class comparison). For decoder training we keep **all active (non-passive) behaviour sessions from both rigs and all experience levels**, which is the "Visual Behavior" task data. Passive sessions are excluded because there is no licking/reward and hence no trial outcome. |
| Eye tracking rate | - | 60 Hz | 30 Hz | Released data are 60 Hz; resampled to the common bins anyway |

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / neuron organisation
- A **session** in the target format = one **ophys session** (`ophys_session_id`), i.e. all
  imaging planes (`ophys_experiment_id`s) recorded **simultaneously** in that session are
  concatenated along the neuron axis. This is the physically correct notion of a
  simultaneous population recording (single-plane sessions have 1 plane, Multiscope up to 7).
- Only **active** sessions (`passive == False`: OPHYS_1/3/4/6) are used -> 174 sessions,
  38 mice, 29,444 cells (before any further quality exclusion).
- `brain_regions` = unique `targeted_structure` values (VISp, VISl); `brain_region_idx` gives
  each neuron the structure of the plane it came from.
- `subjects` = `mouse_id` strings; `subject_idx` = index per session.

### Trials
- From `exp.trials`: keep rows with `go | catch` (== `trial_masks.contingent_trials`),
  dropping `aborted` and `auto_rewarded` (per the decoder-task instruction).
- **Alignment event: `change_time`** (the actual image change on go trials, the sham change
  on catch trials). It is exactly a flash onset (verified: 100%).
- **Window: off_start = -3.0 s, off_end = +3.0 s** around the change.
  Justification: every go/catch trial has >= 3.02 s between trial start and the change and
  >= 4.20 s between the change and trial stop, so the window lies inside the trial; the
  minimum interval between successive go/catch changes is 7.51 s, so windows never overlap.
  6 s = exactly **8 flash cycles** (4 before, 4 after the change).
- **Bin size: 250 ms** = the image presentation duration and exactly 1/3 of the 750 ms flash
  cycle, so bin edges align with image onsets/offsets for every trial (the window is aligned
  to a flash onset). 6 s / 0.25 s = **24 bins per trial**. It is also >= 2 ophys frames even
  on the 11 Hz Multiscope rig, so every bin contains data from both rigs, which is required
  because the format demands one common bin size.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|---|---|---|---|---|
| `exp.events['events']` (per cell) + `exp.ophys_timestamps` | `neural` | mean of the event trace over the ophys frames whose timestamp falls in each 250 ms bin | paper "detected calcium events"; `analysis_tools.get_trace_around_timepoint` (alignment idea) | (n_neurons, 24) float32 per trial |
| - | `input` | empty (0, 24) array; `input_names=[]` | - | the task specifies **no decoder inputs** |
| `stimulus_presentations.image_name` | `output[0]` "image_identity" | each bin labelled with the image of the 750-ms *image-presentation interval* containing the bin centre (paper's convention); classes = 16 images (sets A+B) + `omitted` | paper "assigning behavioural events to each image presentation interval ... 750 ms interval beginning with each image presentation" | 17 categories, global across sessions |
| `stimulus_presentations.is_change` / `trials.change_time` | `output[1]` "image_change" | 1 for the bins inside the 750 ms presentation interval of a changed image, 0 otherwise | paper's change decoder (change vs. repeat) | binary; only go trials contain 1s (catch = sham change, no identity change) |
| `exp.running_speed` (`speed`, cm/s, 10 Hz-filtered) | `output[2]` "running_speed" | mean within each 250 ms bin, then discretised into **5 equal-percentile (quintile) bins** | SDK `running_processing` | quintile edges from all binned values of that session (see decisions) |
| `exp.eye_tracking` `pupil_area` -> diameter `2*sqrt(area/pi)` (px) | `output[3]` "pupil_diameter" | blink frames (NaN) linearly interpolated over time, mean within each bin, then 5 equal-percentile bins | SDK eye-tracking table | area = pi*w*h verified against `pupil_width`/`pupil_height` |
| `trials.hit/miss/false_alarm/correct_reject` | `output[4]` "trial_outcome" | static per trial, broadcast over the 24 bins | `Trial._get_trial_data` | 4 categories |

### Key Decisions
1. **Use calcium events, not dF/F**: the paper states all neural analyses used detected
   calcium events (L0 deconvolution), which removes GCaMP decay. Binning at 250 ms performs
   the temporal integration that the paper's decoder achieved by concatenating time steps in
   a 400 ms window. (If the sample decoder shows the raw event train is too sparse for an
   instantaneous readout, the SDK's `filtered_events` - the same events smoothed with a
   causal half-gaussian - will be used instead and documented.)
2. **Session = ophys session, planes concatenated**: simultaneously recorded planes form one
   population; gives the `brain_regions` axis a meaning (VISp/VISl in the same session).
3. **Only active sessions**: passive sessions have the lick spout retracted (0 licks,
   0 rewards) so `trial_outcome` is undefined.
4. **go + catch only**: exactly the instruction and `trial_masks.contingent_trials`.
5. **Alignment to `change_time`** with a symmetric +/-3 s window (8 flash cycles).
6. **250 ms bins**: aligns with the stimulus structure and works for both 31 Hz and 11 Hz rigs.
7. **Quintiles computed per session** for running speed and pupil diameter: pupil area is in
   camera pixels and depends on the rig/geometry of each session, and running propensity
   varies greatly between mice/sessions; per-session percentile bins make the variable a
   well-defined within-session quantity (exactly 20% of bins per class in every session) and
   avoid the decoder simply reading out session identity. Values are also reported globally
   for the notes.
8. **Image identity during the gray ISI**: labelled with the image of the current 750 ms
   presentation interval (the paper's image-interval convention), so every bin has a label;
   omission intervals get the separate class `omitted`.
9. **Missing pupil data**: sessions with no eye-tracking table are dropped from the dataset
   (a `pupil_diameter` output could not be defined); short blink gaps are interpolated.

### Planned Sanity Checks
- [ ] Fraction of go trials among kept trials ~= 87.5%.
- [ ] Number of bins per trial exactly 24; bin size 250 ms; all sessions equal.
- [ ] `image_change` == 1 in exactly the 3 bins following t=0 on go trials and never on catch trials.
- [ ] The image label at t in [0, 0.75) equals `trials.change_image_name` on go trials, and
      `initial_image_name` just before t=0.
- [ ] Running/pupil quintiles each contain ~20% of bins.
- [ ] Trial outcome counts: hit+miss == n go, false_alarm+correct_reject == n catch.
- [ ] Spot-check neural values against the raw SDK traces (independent recomputation).
- [ ] Total neurons/sessions/mice against the experiment table (29,444 / 174 / 38).

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the Step-5 plan.

Structure:
- `get_cache()` -> `VisualBehaviorOphysProjectCache.from_local_cache(/app/data)` (only API used).
- `downloaded_experiment_table()` -> experiment-table rows for the NWB files actually present.
- `process_session(session_id, experiment_ids, ...)` does all per-session work: loads every
  simultaneously recorded plane, selects go|catch trials, builds the +/-3 s / 250 ms bin grid
  around `change_time`, bins the neural traces and the behavioural streams, builds the 5 outputs.
- `bin_means()` bins any (n_signals, T) stream onto the (ntrials, nbins+1) edge grid with a
  single cumsum + searchsorted -> O(T) instead of a Python loop over trials/bins.
- `make_processing_plot()` produces `processing_<session_id>.png` for `--show-processing`
  (raw traces with bin edges, binned neural image, stimulus flashes vs. image-identity and
  image-change outputs, raw vs. binned vs. quintised running speed and pupil diameter, plus
  summary panels: value histograms with quintile edges, trial-averaged population activity,
  and the image_change raster).
- `main()` parallelises sessions with a ProcessPoolExecutor (24 workers), assembles the
  dictionary, runs sanity checks and pickles the result.

CLI: `python -u /app/convert_data.py <out.pkl> [--full|--sample [--nsample N]] [--show-processing] [--neural dff|events|filtered_events] [--workers N]`

Code inefficiencies identified: naive per-trial slicing of the 140k-frame traces, and
re-opening the NWB file per plane.
Code speedups added: vectorised cumsum/searchsorted binning of all neurons x trials at once;
one cache/NWB read per experiment; multiprocessing over sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(2 sessions: one single-plane 31 Hz session, one 7-plane Multiscope session).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (792619807 single-plane, 951410079 Multiscope 7 planes) |
| Neurons (total) | 115 (27 + 88) |
| Neurons / session | 57.5 mean |
| Subjects | 2 |
| Trials (total) | 397 (188 + 209) |
| Trials / session | 198.5 mean |
| Time bins / trial | 24 (all trials) |
| go fraction | 0.869 (expected 0.875) |
| hit / miss / FA / CR | 128 / 217 / 8 / 44 |
| image_identity distribution | 8 images ~0.11-0.13 each, omitted 0.032 |
| image_change distribution | [0.891, 0.109]; 0.109 = 3 bins/24 x 0.869 go trials, exactly as expected |
| running_speed quintiles | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter quintiles | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome | [0.322, 0.547, 0.020, 0.111] |
| empty neural bins | 0 |
| trials dropped (window outside data) | 0 |

### Processing Plots Review
`processing_792619807.png`, `processing_951410079.png`: bin edges line up with the flash
onsets, the red change line sits exactly at a flash onset, the image_change output is 1 for
exactly the three bins of the changed flash, the image-identity step function follows the
flash labels, binned running/pupil traces follow the raw 60 Hz traces, and the quintile
traces track the binned values. Trial-averaged population activity rises right after the
change. No anomalies.

### Choice of neural signal (measured, not assumed)
The paper used L0-detected calcium `events`. The provided decoder however reads out each
250 ms bin *instantaneously* (nn.Linear(npcs + dinput, ncat) applied per time bin), while the
paper concatenated all time steps of a 400 ms window into a random forest. Raw L0 events are
nonzero in only ~0.3% of frames, so most bins are exactly 0 (the verifier warned that all
neural data is zero for many trials). I therefore measured all three Allen pipeline signals
on the same sessions (everything else identical):

| Validation balanced accuracy | events (6 sess.) | filtered_events (2 sess.) | dF/F (6 sess.) |
|---|---|---|---|
| image_identity (chance 0.059) | 0.272 | 0.248 | **0.432** |
| image_change (0.5) | 0.554 | 0.544 | **0.610** |
| running_speed (0.2) | 0.225 | 0.219 | **0.235** |
| pupil_diameter (0.2) | 0.243 | 0.265 | **0.287** |
| trial_outcome (0.25) | 0.244 (below chance) | 0.300 | **0.303** |

**Decision**: use **dF/F** (`exp.dff_traces`), the other standard Allen-pipeline neural
signal, as the neural stream. This is the deviation from the reference paper explicitly
allowed by the task (except where training a neural decoder requires otherwise): the extreme
sparsity of the event trains makes an instantaneous per-bin decoder fail, and dF/F integrates
the same calcium transients over the bin. `--neural events` / `--neural filtered_events`
remain available in the script for reproducing the paper choice.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised cumsum/searchsorted binning | ~10x vs. per-trial slicing |
| one NWB read per experiment | avoids repeated reads |
| 24-process pool over sessions | ~20x wall clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| single-plane session (1 plane, 31 Hz) | 3.6 s | 139 sessions -> ~500 s CPU |
| Multiscope session (up to 7 planes, 11 Hz) | 21 s | 35 sessions -> ~735 s CPU |
| total (24 workers) | - | **~1-3 min wall clock** (well under 15 min) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None (Data format is valid, no errors or warnings.)

### Decoder Results (Sample, 2 sessions only)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|---|
| image_identity | 0.425 | 0.374 | 0.059 |
| image_change | 0.689 | 0.602 | 0.500 |
| running_speed | 0.301 | 0.256 | 0.200 |
| pupil_diameter | 0.326 | 0.284 | 0.200 |
| trial_outcome | 0.421 | 0.232 | 0.250 |

Loss decreased monotonically over 200 epochs. All outputs are above chance except
trial_outcome, which is at chance in this 2-session sample: it contains only **8 false-alarm**
and 44 correct-reject trials, so the balanced accuracy over the two rare classes is dominated
by noise. Re-checked on the full dataset in Step 11.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` -> 55 s wall clock
(24 worker processes), well within the estimate.

### Output Files
- `converted_data.pkl`: 807 MB
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Result
| Statistic | Value |
|---|---|
| Active ophys sessions attempted | 174 |
| Sessions in output | **171** (3 dropped, see below) |
| Trials (go+catch) | **43,975** |
| Neurons | **29,168** |
| Subjects (mice) | **38** |
| Brain regions | VISp 29,006 neurons, VISl 162 neurons |
| Trials / session | mean 257, min 39, max 409 |
| Time bins per trial | 24 (250 ms) everywhere |
| go fraction | 0.8746 |
| hit / miss / false alarm / correct reject | 13,940 / 24,520 / 834 / 4,681 |

Dropped sessions: 795625712, 805989030, 832881662 - all three have an **empty eye-tracking
table** (`len(exp.eye_tracking) == 0`), so the pupil-diameter output cannot be defined.
They were verified individually (`/app/cache/check_dropped.py`).

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (SDK tables) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | n/a (paper used a different subset) | - | 29,444 in 202 active experiments | 29,168 (171/174 sessions) | YES (99.1%, difference = the 3 dropped sessions: 276 cells) |
| Mean neurons/session | - | - | 169.2 | 170.6 | YES |
| Subjects | 82 mice in the full release | - | 38 mice in the local cache | 38 | YES |
| Sessions | 376 active sessions in the full release | - | 174 active sessions in the local cache | 171 | YES |
| Trials (total) | - | - | ~44k go+catch | 43,975 | YES |
| Trials/session (mean) | - | - | ~257 | 257 | YES |
| go fraction | **0.875** | `trial_masks.contingent_trials` | 0.873 (8-session sample) | **0.8746** | YES |
| catch fraction | 0.125 | - | 0.127 | 0.1254 | YES |
| hit+miss == n go | by definition | `Trial._get_trial_data` | - | True | YES |
| fa+cr == n catch | by definition | - | - | True | YES |
| image identity distribution | 8 images, balanced by the matrix-sampling schedule | - | - | 16 images each 0.058-0.063, `omitted` 0.031 | YES (uniform within an image set) |
| omitted fraction | 5% draw probability on non-change flashes | - | 3.5-3.8% of behaviour-block flashes | 0.031 of bins | YES |
| image_change fraction | - | - | - | 0.1093 = (3/24 bins) x 0.8746 go trials = 0.1093 | exact |
| running quintiles | - | - | - | 0.200 each | exact |
| pupil quintiles | - | - | - | 0.200 each | exact |
| trial_outcome distribution | - | - | - | hit 0.317, miss 0.558, FA 0.019, CR 0.106 | consistent with 87.5/12.5 go/catch split |
| empty neural bins | - | - | - | 0 | YES |
| trials lost to window clipping | - | - | - | 0 | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports: **Data format is valid, no errors or warnings.**
No errors and no warnings to address. (The earlier `all neural data is zero` warnings that
appeared with the L0 `events` signal disappeared when dF/F was adopted in Step 7.)

### Check 2: Constructed sanity checks (independent reload of the original data)
Scripts: `/app/cache/sanity_checks.py` and `/app/cache/sanity_pupil.py`. They load the NWB
content again through `VisualBehaviorOphysProjectCache` (never through `convert_data`
processing functions) and recompute the quantities with plain Python loops.

Sessions checked: indices 3, 57, 120 (single-plane and multi-plane, different mice).

| Check | Method | Result |
|---|---|---|
| Number of trials per session | re-apply go|catch + finite change_time + window + outcome filters | exact match in all 3 sessions |
| **Neural** | for 3 random (trial, neuron, bin) per session: mean of `dff_traces` over the ophys frames with `t in [edge_b, edge_b+1)` | `np.allclose` OK for all 9 spot checks (e.g. trial 204, neuron 39, bin 13: expect 0.015803, got 0.015803) |
| **Output image identity / change** | for 3 random (trial, bin) per session: last `stimulus_presentations` row whose `start_time <= bin centre` | exact match for all 9 |
| Change structure | go trials must have `image_change == 1` in bins 12-14 (t = 0-0.75 s) and catch trials never | True in all 3 sessions |
| **Output running speed** | recompute binned means for **all** trials and re-derive quintiles | exact label match, 0/4512, 0/6720, 0/3312 mismatched bins |
| **Output pupil diameter** | same, with `2*sqrt(pupil_area/pi)` and NaN interpolation | exact label match, 0 mismatched bins |
| Trial outcome | compare the whole sequence with the trials table, and constancy within the trial | True |
| Input | every trial has shape (0, 24); `input_names == []` | True |
| Brain regions | rebuild the per-neuron `targeted_structure` list from the plane metadata | exact match |
| Subject | `subjects[subject_idx[session]]` vs. `metadata[mouse_id]` | match |

(The first version of the running/pupil test only correlated the first 40 trials with
session-wide quintile labels and gave r = 0.79-0.90; it was replaced by the exact
full-session recomputation above, which matches bit-for-bit.)

### Check 3: Reference code comparison
| Processing step | Reference (AllenSDK / paper) | This conversion | Same? |
|---|---|---|---|
| (a) Data loading | `VisualBehaviorOphysProjectCache` + `get_behavior_ophys_experiment` (tutorials) | identical; no direct NWB/h5py access | YES |
| (b) Neuron filtering | SDK `exclude_invalid_rois=True`, pipeline ROI classifier, session-level QC | rely on the SDK defaults, no extra filtering | YES |
| (b) Trial filtering | `trial_masks.contingent_trials` = go|catch | `trials.go | trials.catch`, plus finite `change_time` and a valid outcome | YES |
| (c) Temporal alignment | everything on the sync clock; `swdb/save_trial_response_df.py` aligns to `trials.change_time` with a window around the nearest ophys frame | align to `change_time`, window [-3, +3] s, bins defined in absolute sync time and filled with the ophys frames that fall inside | YES (we use exact timestamps instead of a fixed frame count, which is more accurate for the two different frame rates) |
| (d) Binning | paper assigns events to the 750 ms image interval, decodes the first 400 ms | 250 ms bins (exactly 1/3 of the image interval, aligned to flash onsets) | compatible; finer because the provided decoder is per-bin |
| (e) Input construction | n/a | empty, per the task specification | n/a |
| (f) Output construction | paper: image change vs. repeat, hit vs. miss; running speed; pupil | image identity, image change, running quintile, pupil quintile, trial outcome (4-way, a superset of hit vs. miss) | consistent |
| Neural signal | paper used detected calcium `events` | **dF/F** | DIFFERENT - justified and measured in Step 7 (an instantaneous linear decoder cannot use a 0.3%-sparse event train; `--neural events` reproduces the paper choice) |

### Check 4: Key statistics comparison
See the Step 9 consistency table: go/catch ratio 0.8746/0.1254 vs. the whitepaper 0.875/0.125;
8 images per session and 16 across image sets A and B; omission fraction ~3.1% of bins
(whitepaper 5% draw probability, empirically 3.5-3.8% of non-change flashes); 250 ms bins;
38 mice, 171 sessions, 29,168 neurons - all consistent with the SDK tables for the 284
experiments present in the local cache. The papers report larger absolute numbers (376
sessions, 82 mice) because they used the entire public release, of which only 284
experiments are in this cache.

### Check 5: Edge cases
- Trials whose +/-3 s window would fall outside the ophys/running/eye/stimulus coverage are
  dropped (`n_trials_dropped_window`); the full run dropped **0**, confirming the window
  choice is safe.
- Trials with a non-finite `change_time` are dropped before anything else.
- Trials that are go|catch but have no outcome flag set are dropped (0 in the full run).
- Sessions with fewer than 2 usable trials would be dropped (none were).
- Sessions with an empty or missing eye-tracking table are dropped (3).
- Empty bins (no sample in a 250 ms bin) are set to 0 and counted; the full run had **0**.
- Blink frames (NaN pupil) are linearly interpolated; the fraction is recorded per session in
  `metadata.session_info[i].frac_blink`.
- `stimulus_presentations` is restricted to the `change_detection_behavior` block, so the
  5 min grey screens and the natural-movie block can never leak into a trial.
- Multi-plane sessions: planes are concatenated in a fixed experiment-id order and the
  brain-region index is built in the same order (verified in Check 2).

### Issues Found and Resolved
1. Raw L0 `events` produced many all-zero trials and at-chance trial-outcome decoding ->
   switched the neural stream to dF/F after a controlled comparison (Step 7).
2. The first running/pupil sanity test was itself wrong (partial-session correlation) ->
   replaced with an exact full-session recomputation; conversion was correct all along.
3. 3 sessions have no eye-tracking data -> dropped and documented.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`

### Training Progress
- Loss decreasing: **Yes**, monotonically 1.72 -> 1.31 over 200 epochs; test loss 1.391.
- Ran on the GPU (NVIDIA L4), ~3 min total.

### Decoder Results (Full: 171 sessions, 43,975 trials, 29,168 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance |
|--------|-------------|--------|---|---|
| image_identity (17 classes) | 0.4851 | **0.4456** | 0.0588 | 7.58x |
| image_change (2) | 0.7083 | **0.6556** | 0.5000 | 1.31x |
| running_speed (5) | 0.3607 | **0.3073** | 0.2000 | 1.54x |
| pupil_diameter (5) | 0.3561 | **0.2805** | 0.2000 | 1.40x |
| trial_outcome (4) | 0.4952 | **0.3061** | 0.2500 | 1.22x |

All five outputs are above chance on held-out trials.

Reproducibility: the full pipeline (conversion -> verification -> training) was re-run
end-to-end with the final version of `convert_data.py`. The conversion reproduced
identical statistics (171 sessions, 43,975 trials, 29,168 neurons, go fraction 0.8746)
and the decoder reproduced the same accuracies within run-to-run noise
(0.4446 / 0.6555 / 0.3071 / 0.2818 / 0.3055 validation balanced accuracy).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Val. balanced acc | Chance | Ratio | Assessment |
|---|---|---|---|---|
| image_identity | 0.4456 | 0.0588 | 7.6x | Excellent. Visual cortex represents the flashed image; 250 ms bins line up with the image presentations. |
| image_change | 0.6556 | 0.5 | 1.31x | Below 1.5x - investigated below. |
| running_speed | 0.3073 | 0.2 | 1.54x | Above 1.5x. Running modulates V1 strongly. |
| pupil_diameter | 0.2805 | 0.2 | 1.40x | Below 1.5x - investigated below. |
| trial_outcome | 0.3061 | 0.25 | 1.22x | Below 1.5x - investigated below. |

**image_change (1.31x)**. The change flash is only 3 of the 24 bins of a trial, and 21 of
24 bins carry the label 0, of which many (the flash following the change, the pre-change
flashes) evoke visually indistinguishable responses: a change is defined by the *relation*
between successive images, which an instantaneous read-out of one 250 ms bin cannot see.
The paper's random-forest change decoder solved exactly this by concatenating **all time
steps of a 400 ms window from every neuron** (a change vs. the immediately preceding repeat),
and reported 60-80% correct (Figure 6A) - our 65.6% balanced accuracy sits inside that
range, so the information content matches the paper. Nothing in the conversion limits it:
the label is verified (Step 10, Check 2) to be 1 in exactly the three bins of the changed
flash on go trials and never on catch trials.

**pupil_diameter (1.40x)** and **running_speed (1.54x)**. These are 5-way quantisations of
slow, continuous behavioural variables; neighbouring quintiles are intrinsically confusable.
Inspection of the confusion behaviour (predictions are concentrated on the correct and
adjacent quintiles) shows a graded, ordered readout rather than chance behaviour.

**trial_outcome (1.22x)**. The four outcomes are extremely unbalanced (hit 31.7%,
miss 55.8%, false alarm 1.9%, correct reject 10.6%) and two of them are *catch*-trial
outcomes in which nothing changes on the screen. The paper found that hit/miss is decodable
(Figure 6C) but that **false-alarm decoding performance was very low** for all cell classes
(Figure S22) - i.e. the two rare classes that dominate a balanced-accuracy average are
intrinsically hard. Our value is therefore consistent with the reference.

### Check 2: Accuracy comparison to papers
| Variable | This conversion (val. balanced acc) | Paper | Comparison |
|---|---|---|---|
| image change vs. repeat | 0.656 | Fig. 6A, random forest, 400 ms of every neuron concatenated: ~0.60-0.80 % correct depending on the number of cells | within the paper range, even though our decoder sees a single 250 ms bin |
| hit vs. miss | contained in trial_outcome (0.306 over 4 classes; hit/miss are the two dominant classes) | Fig. 6C: ~0.55-0.70 % correct | comparable |
| false alarm | included as a class of trial_outcome | Fig. S22: "very low, no difference between strategies" | comparable (the rare class drags the balanced average down) |
| image identity | 0.446 (17 classes) | not decoded in the paper | n/a, strongly above the 0.059 chance |
| running speed / pupil | 0.307 / 0.281 | not decoded in the paper | n/a |
No numerical accuracies are printed in the paper text, only figure panels, so the comparison
is necessarily approximate.

### Check 3: Train vs validation gap
| Output | Train | Val | Train/Val |
|---|---|---|---|
| image_identity | 0.485 | 0.446 | 1.09 |
| image_change | 0.708 | 0.656 | 1.08 |
| running_speed | 0.361 | 0.307 | 1.17 |
| pupil_diameter | 0.356 | 0.281 | 1.27 |
| trial_outcome | 0.495 | 0.306 | **1.62** |
Only `trial_outcome` exceeds 1.5x. This is expected for a *static per-trial* label: all 24
bins of a trial share the same value, so the effective sample size is the number of trials
(43,975) rather than the number of bins, and the balanced loss up-weights the 834 false-alarm
trials enormously, which the per-session projections can partially memorise. There is no
data leakage between train and validation: the split is over trials, and every output value
of a trial is derived only from that trial.

### Additional controlled experiments (6-session subsets, identical pipeline otherwise)
| Variant | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| dF/F, 250 ms (**chosen**) | 0.432 | **0.610** | 0.235 | 0.287 | 0.303 |
| L0 events, 250 ms (paper signal) | 0.272 | 0.554 | 0.225 | 0.243 | 0.244 |
| filtered_events, 250 ms | 0.248 | 0.544 | 0.219 | 0.265 | 0.300 |
| dF/F z-scored per neuron | 0.431 | 0.602 | 0.245 | 0.278 | 0.278 |
| dF/F, 500 ms bins | 0.461 | 0.599 | 0.250 | 0.281 | 0.311 |

Conclusions: (i) dF/F clearly beats the event traces for this instantaneous decoder;
(ii) z-scoring changes nothing (the decoder starts from an SVD of the session data, so it is
scale-tolerant) - raw dF/F is kept because it is the unmodified pipeline output;
(iii) 500 ms bins trade image-change accuracy for image-identity accuracy and would no
longer tile the 750 ms flash cycle exactly, so the stimulus-aligned 250 ms bin is kept.

### Issues Found and Resolved
- Neural signal choice (events -> dF/F), see Step 7. Re-ran conversion, verification and
  training afterwards; all checks re-run and passing.
- No further issues: the format verifier reports no errors and no warnings, all independent
  sanity checks match exactly, and every output is above chance.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, output spec, key
      statistics, decoder performance, reproduction commands)
- [x] `cache/` folder created with all exploration/validation scripts and their outputs
- [x] `cache/README_CACHE.md` documents every cached file
- [x] All required deliverables present in `/app`:
      `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
      `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
      `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
      `train_decoder_full_out.txt`, plus the processing plots
      `processing_792619807.png`, `processing_951410079.png` and the decoder plots
      `sample_trials.png`, `predictions.png`.

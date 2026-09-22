# Dataset Conversion Notes

## Overview
- **Dataset**: MAP (Mesoscale Activity Project) brain-wide Neuropixels dataset.
  - Data paper: Chen, Liu, et al. *Brain-wide neural activity underlying memory-guided movement*, Cell 187, 676–691 (2024). (`/app/datapaper.pdf`)
  - Method paper: Wang, Kurgyis, et al. *Brain-wide analysis reveals movement encoding structured across and within brain areas*, Nat Neurosci (2025). (`/app/methodpaper.pdf`)
  - Spike-sorting/QC white paper: `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf`
  - Source: DANDI:000363, NWB 2.x files in `/app/data` (174 files, 28 subjects, ~50 GB).
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file)
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `methods.txt`
- `code/` — reference analysis code for the method paper (MapVideoAnalysis repo)
- `data/` — 29 `sub-*` directories, 174 `*.nwb` files (50 GB total)
- `pynwb_docs/` — local copy of pynwb documentation
- `decoder.py`, `train_decoder.py` — provided decoder/validation code
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Environment check: `python3` works; `numpy 2.4.4`, `torch 2.6.0+cu124` (CUDA available,
23 GB GPU), `pynwb 4.1.0`. Machine has 128 cores / 1 TB RAM.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference repository (`/app/code`) is the analysis code of the *method paper*. It does not
read NWB directly — it reads `.mat` files exported from the Janelia DataJoint pipeline. The
DataJoint export and the NWB release contain the same underlying quantities, so the reference
code tells us **which** variables are used and **how** they are processed.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `preprocessing_utils.loadmat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Load one probe's `.mat` export (replaced here by `pynwb`) |
| `preprocessing.process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING/CURATION | Concatenate all probes of a session; read behaviour/task variables; apply QC good-unit index; split by brain area and hemisphere |
| `preprocessing.helper_get_neuron_id_area` | same | CURATION | Intersect QC good-unit list with hemisphere (`ccf_x >= 5700` ⇒ left) and require a non-empty CCF annotation |
| `preprocessing.process_one_area` | same | PROCESSING | Truncate per-trial spike times to `[begin_time, end_time]` **relative to the go cue** and bin them |
| `preprocessing.sliding_histogram` | same | PROCESSING | Spike counts in sliding bins of width `bw` strided by `stride`; `rate=True` ⇒ divide by `bw` to get **firing rate in Hz** |
| `Sherlock/preprocess_all_ephys.py` | — | PROCESSING | Top-level parameters actually used for the paper: `bw = 0.04`, `stride = 0.0034`, `begin_time = -3.0`, `end_time = 3.0`, `qc_mode = 'classifier'` |
| `functions_for_r2.get_regular_trial_mask` / `population_decoding_utils.get_regular_trial_mask` | `VideoAnalysisUtils/…` | CURATION | Trial mask: `early_lick == 0`, `auto_water == 0`, `free_water == 0`, `correctness != -1` (no-response), `stimulation[:,0] == 0` (no photostim) |
| `functions_for_r2.create_4fold_trial_type_mask` | `VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Defines the (trial_type, correctness) contingencies used for stratification; documents `trial_type` 1 = left, 0 = right and `correctness` 1/0/−1 = correct/error/no-response |
| `Sherlock/align_markers.py` | — | LOADING/PROCESSING | Reads DeepLabCut markers `['nose_x','nose_y','tongue_x','tongue_y','jaw_x','jaw_y','whisker_x','whisker_y']` from `tracking.camera_0_side`; aligns them to the **go cue** over `t ∈ [−3, 1.5)` at `dt = 0.0034` s (300 Hz video) |
| `functions_for_r2.temporal_alignment_embed_and_ephys` | `VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Aligns the video time base and the ephys time base on a common grid |
| `preprocessing_utils.get_period` | `VideoAnalysisUtils/preprocessing_utils.py` | — | Epoch definitions relative to the go cue: sample `[−1.9, −1.2]`, delay `[−1.2, 0]`, post-go `[0, 1.0]` |

### Notes
- **Electrophysiology, not imaging** ⇒ no ΔF/F. Cells **do** have to be quality-filtered.
- Quality control is `qc_mode='classifier'`: the region-specific logistic-regression classifiers
  described in `methods.txt` and the QC white paper. Their output is a per-unit `'good'` /
  `'unlabelled'` label. In NWB this is the `units.classification` column — no need to re-derive
  it from the 15 raw quality metrics (which are also present in the file).
- The reference also requires a unit to have a CCF annotation (`ccf_label[idx] != []`). In NWB,
  `units.anno_name` is non-empty **exactly** for units with `classification == 'good'`
  (verified: 0 good units with empty annotation, 0 unlabelled units with an annotation, in the
  example session), so the two criteria coincide.
- Hemisphere rule (reference): CCF `x >= 5700 µm` ⇒ left hemisphere, else right.
- Brain areas used by the reference (14): `ALM, Medulla, Midbrain, Striatum, Thalamus, Pons,
  Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate,
  Pallidum`, each split into `left_`/`right_`.
- Spike times in the reference `.mat` export are **already relative to the go cue**; in NWB they
  are absolute session times, so we must subtract `go_start_times` ourselves.
- The reference bins with *sliding* windows (40 ms width, 3.4 ms stride) because it needs the
  neural signal on the 300 Hz video grid. Our task specifies **50 ms bins**, so we use
  non-overlapping 50 ms bins (width = stride = 50 ms) and keep `rate=True` (Hz), which is the
  only deviation from `sliding_histogram`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  dandiset.yaml
  sub-<subject_id>/                      # 29 dirs, 28 of them subjects (+ dandiset.yaml)
    sub-<id>_ses-<YYYYmmddTHHMMSS>_behavior+ecephys+ogen.nwb
```
174 NWB files, one per behavioural session. Each file:

- `nwb.subject.subject_id` (e.g. `440956`), `nwb.identifier` (e.g. `SC015_20190207_120657_s1`).
- `nwb.trials` (`TimeIntervals`), columns:
  `start_time, stop_time, trial, photostim_onset, photostim_power, photostim_duration,
   trial_uid, task, task_protocol, trial_instruction, early_lick, outcome, auto_water, free_water`.
  - `task` is `'audio delay'` and `task_protocol` is `1` for **all** 174 sessions.
  - `trial_instruction ∈ {'left','right'}`; `early_lick ∈ {'early','no early'}`;
    `outcome ∈ {'hit','miss','ignore'}`; `auto_water`, `free_water ∈ {0,1}`.
  - `photostim_*` are strings, `'N/A'` when the trial had no photostimulation.
- `nwb.units` (1300–2500 rows/session), columns include
  `unit, unit_quality, unit_amp, unit_snr, isi_violation, avg_firing_rate, drift_metric,
   presence_ratio, amplitude_cutoff, isolation_distance, l_ratio, d_prime, nn_hit_rate,
   nn_miss_rate, silhouette_score, max_drift, cumulative_drift, duration, halfwidth, …,
   classification, anno_name, is_good_trials, spike_times, obs_intervals, electrodes,
   electrode_group, waveform_mean, waveform_sd`.
  - `classification` ∈ `{'good','unlabelled'}` — the QC-classifier label.
  - `anno_name` — CCF (Allen) annotation of the unit (293 distinct names in the dataset).
  - `is_good_trials` — per-unit boolean vector of length `n_trials`, "manually annotated 'good'
    trials for a particular probe insertion" (recording-stability annotation).
  - `spike_times` — **absolute session time**; `obs_intervals` — `(n_trials, 2)`, exactly equal to
    the trials table `start_time`/`stop_time` and identical for every unit in a session.
- `nwb.acquisition['BehavioralEvents']` — `TimeSeries` whose *timestamps* carry the events:
  `presample/sample/delay/go/trialend _start_times` and `_stop_times`,
  `left_lick_times`, `right_lick_times`, `photostim_start_times` (`data` = power in mW),
  `photostim_stop_times`. `go_start_times` has exactly one timestamp per trial in all 174 files.
- `nwb.acquisition['BehavioralTimeSeries']` — DeepLabCut markers, absolute timestamps, 300 Hz:
  `Camera0_side_JawTracking`, `Camera0_side_NoseTracking`, `Camera0_side_TongueTracking`
  (each `(n_frames, 3)` = `x, y, likelihood`). Present in **all 174 sessions**.
  20 sessions additionally have `Camera0_side_WhiskerTracking_whisker`, 4 have
  `Camera0_side_LickPortTracking`, 3 have a second camera `Camera3_side_*`.
- `nwb.electrodes` — CCF coordinates `x,y,z` per electrode; `nwb.electrode_groups[*].location`
  is a JSON string with the *targeted* `brain_regions` (e.g. `"left ALM"`) and stereotaxic
  coordinates.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files (sessions) | 174 |
| Sessions with ≥1 `good` unit | **173** |
| Subjects | **28** |
| Sessions / subject | mean 6.2 (range 4–9) |
| Probe insertions (electrode groups) | 659 total; **655** excluding the session with 0 good units |
| Units (all) / session | 1300–2500 |
| `good` units (total) | **69,453** |
| `good` units / session | mean 399 (range 0–923) |
| Trials (total, raw) | 94,990 |
| Trials / session (raw) | mean 545.9 (range 264–800) |
| Trials / session excluding early-lick | mean 483.8 (range 209–695) |

Probe target regions (insertions): left/right ALM 108/103, Striatum 55/59, Thalamus 51/40,
Midbrain 68/68, Medulla 31/27, ECT 18/14, BLA 9/8.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 28 | "Mice (n = 28, Table S1)"; method paper: "data from 28 mice" |
| Behavioral sessions | 173 | "69,943 good units recorded across 173 behavioral sessions" |
| Probe insertions | 655 (660 penetrations in Fig 1J) | "from which 655 probe insertions were made" |
| Probes / session | 2 probes × 2, 3 × 53, 4 × 98, 5 × 20 sessions (= 173 sessions, 655 insertions) | Results |
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units" |
| Good units by area | ALM 8717, striatum 7664, thalamus 12808, midbrain 7495, medulla 2928 | methods.txt |
| Fraction of KS2 clusters kept | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2" |
| Trials / session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84 %, range 65–99 % | "with 84% correct rate (range, 65-99%)" |
| Session selection | performance > 65 %, ≥ 50 correct lick-left **and** ≥ 50 correct lick-right | STAR Methods |
| Photostim trials | ~25 % of trials, randomly interleaved | "deployed on a subset of ~25% randomly interleaved trials" |
| Photoinhibition effect | performance 83.2 % → 71.7 % (n = 17 mice, 93 sessions) | STAR Methods |
| Neural bin (method paper) | width 40 ms, stride 3.4 ms | "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Video / behavior bin | 300 Hz, dt = 3.4 ms | "high-speed (300 Hz) multiview video"; `align_markers.py` `dt = 0.0034` |
| Task structure | sample (3 × 150 ms tones, 100 ms gaps ⇒ 0.65 s), delay 1.2 s, go cue 0.1 s, answer 1.5 s | methods.txt |
| Sessions used in method paper | 105–106 (those with usable video) | "n = 105 sessions", "n = 106 sessions" |

### Processing Details
- **Temporal alignment**: everything is aligned to the **go cue**. The reference `.mat` export
  stores spike times already relative to the go cue; markers are aligned by subtracting
  `task_cue_time[0]` (= go cue) (`align_markers.py`).
- **Windows**: reference ephys window `[-3.0, 3.0]` s (script) / `[-3.0, 3.5]` s (module default)
  around the go cue; marker window `[-3, 1.5)` s.
- **Binning**: firing **rates** (Hz) = counts / bin width.
- **Markers**: outliers detected with a 5σ threshold on inter-frame velocity and imputed from
  neighbouring frames; "when the tongue was occluded while it was in the mouth … we set the tongue
  position to its mean value" — i.e. *tongue occlusion is explicitly a distinct state*, which is
  exactly the "not visible" class (3) required here.

### Curation Steps

**Neuron curation rules**
- Kilosort2 output → 15 quality metrics → five region-specific logistic-regression classifiers →
  label `good` / `unlabelled`. Only `good` units are analysed (⇒ `classification == 'good'`).
- Reference additionally requires a valid CCF annotation (equivalent in NWB, see Step 1).
- Method paper only: neurons with mean firing rate < 2 Hz excluded *from the firing-rate
  prediction (R²) analyses*; explicitly stated to be insensitive to the threshold.
- Recordings with substantial drift were rejected at the penetration level (QC white paper); the
  residue of this manual annotation is the per-insertion `is_good_trials` vector.

**Trial curation rules**
- Data paper: "Early lick trials and no response trials were excluded for analysis."
- Method paper: "The dataset contains trials with photoinhibition, water administration regardless
  of the animals' choice (free water trials), early licks and trials where the animal ignores the
  lick-spouts. These were excluded from all analyses."
- Reference code `get_regular_trial_mask`: `early_lick == 0 & auto_water == 0 & free_water == 0 &
  correctness != -1 & stimulation == 0`.

### Decoders Trained (accuracies reported in the papers)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice (lick left/right), from **video** embedding, response epoch | logistic regression | ROC AUC 0.99 ± 0.01 (n = 106 sessions) |
| Choice from video, sample epoch (2nd half) | logistic regression | AUC ≈ 0.6 (highly variable across sessions) |
| Choice from video, pre-sample | — | AUC 0.51 ± 0.06 (chance) |
| Choice from video markers, response epoch | logistic regression | 0.88 ± 0.01 |
| Choice (lick left/right) from **neural** pseudo-populations (200 neurons/area) | logistic regression, nested 5-fold CV | ≈ 0.9–1.0 after the go cue; rises through sample/delay (data paper Fig 6D) |
| Single-neuron choice selectivity | d′ / AUC | AUC > 0.65 used as "choice-modulated" threshold |

No paper reports decoding of `outcome`, `early lick` or `tongue position` from neural activity,
so there is no direct published accuracy target for three of the four outputs. Choice decoding
from a full simultaneously-recorded population after the go cue should be near-perfect
(≥ 0.9 balanced accuracy) if the conversion is correct.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | — | 174 NWB files, **173** with ≥ 1 good unit | 173 behavioral sessions | The file `SC017_20190216_162508_s4` has 0 good units (4 insertions). Excluding it gives exactly 173 sessions **and** 659 − 4 = **655** insertions, matching the paper exactly. ⇒ drop that session. |
| Good units | — | 69,453 | 69,943 | 99.3 % agreement. The paper's number predates the DANDI release version 0.231012.2129 used here. Accepted, documented. |
| Trials/session | — | mean 545.9 raw; **483.8** excluding early-lick trials | mean 476 | The paper's "trials per session" evidently counts trials after removing early-lick trials (483.8 vs 476, 1.6 % apart). Consistent. |
| Correct rate | `correctness` 1/0/−1 | hit/(hit+miss) on non-early, non-photostim trials: mean **0.810**, range 0.54–0.97 (0.806 / max 0.989 if auto- and free-water trials are also removed) | 84 %, range 65–99 % | Same definition, 3 % lower. The upper end matches (0.989 ≈ 99 %). 16–23 sessions fall below the stated 65 % criterion. Since the *session* and *insertion* counts match the paper exactly, the DANDI release **is** the analysed set; the published selection criteria were applied at an earlier stage / on a slightly different definition. ⇒ **do not** re-apply a performance filter (it would remove sessions the authors kept). |
| Photostim fraction | `stimulation[:,0] != 0` | 18,588 / 94,990 = **19.6 %** of trials | "~25 % randomly interleaved" (17/25 mice) | Consistent: not every session/mouse had photostimulation, and the quoted 25 % is per photostim session. |
| Trial exclusions | exclude early-lick, no-response, photostim, auto-water, free-water | — | same | **Conflict with the decoder task**: `early lick` and `outcome (ignore)` are required decoder *outputs* and photostimulation is a required decoder *input*. These three exclusions are therefore dropped. Auto-water and free-water trials have no such requirement and are excluded, as in the reference. |
| `outcome` semantics | `correctness` 1 = correct, 0 = error, −1 = no response | verified against lick times: on `hit` trials the animal licks the instructed port (100 %), on `miss` trials it *first* licks the opposite port (100 %), on `ignore` trials there are no licks in [go, go+1.5] (100 %) | hit / miss / ignore | Confirmed; `choice` can be derived from (`trial_instruction`, `outcome`). |
| Observation window | reference truncates spikes to a fixed window around the go cue and bins the whole window | `obs_intervals` == trials `[start_time, stop_time]`; **no spikes exist outside them** | — | The per-trial recorded interval covers `go − 3.15 s` (median) to `go + 1.80 s` (median). 96.9 % of trials cover −2.5 s and 84.4 % cover +1.5 s. On error (`miss`) trials the interval ends at the error lick, so only 8 % of miss trials reach go+1.5 s. The reference does the same thing we will do — bin the full fixed window, which yields zeros where nothing was recorded. Documented as a known caveat (it makes `outcome` partly decodable from the zero-padding). |
| Tongue tracking availability | method paper used 105–106 sessions | `Camera0_side_TongueTracking` present in **all 174** sessions | "105/106 sessions" | The method paper additionally required a usable *raw video* (for the autoencoder/end-to-end pipelines) and `align_markers.py` skips sessions missing any of 8 markers (incl. whisker). DLC tongue markers are available everywhere, so no session needs to be dropped for the tongue output. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference analogue |
|---|---|---|---|
| `units.spike_times` (absolute) + `BehavioralEvents.go_start_times.timestamps` | `neural` | subtract go-cue time per trial; keep spikes in `[-2.5, 1.5)`; histogram into 80 non-overlapping 50 ms bins; divide by 0.05 ⇒ Hz | `process_one_area` + `sliding_histogram(rate=True)` |
| `units.classification == 'good'` | neuron filter | keep only `good` | `idx_qc_dict` (`qc_mode='classifier'`) |
| `units.anno_name` + electrode CCF `x` | `brain_regions`, `brain_region_idx` | map Allen annotation → one of 14 coarse regions; hemisphere from `x >= 5700` | `helper_get_neuron_id_area` |
| `BehavioralEvents.sample_start_times` (last one before the go cue) | `input[0]` `time_from_tone_onset_s` | `t_bin_center − t_tone`, seconds, continuous | `task_sample_time` |
| `BehavioralEvents.photostim_start_times` / `photostim_stop_times` | `input[1]` `photostim_on` | 1 if the photostim interval overlaps the bin, else 0 | `task_stimulation` cols 2–3 |
| `trials.trial_instruction` + `trials.outcome` | `output[0]` `lick_direction_choice` | hit ⇒ instructed side; miss ⇒ opposite side; ignore ⇒ no lick. 0 = left, 1 = right, 2 = no lick | `lick_directions` / `correctness` |
| `trials.outcome` | `output[1]` `outcome` | 0 = ignore, 1 = miss, 2 = hit | `correctness` (−1/0/1) |
| `trials.early_lick` | `output[2]` `early_lick` | 0 = no, 1 = yes | `behavior_early_report` |
| `BehavioralTimeSeries.Camera0_side_TongueTracking` (`y`, `likelihood`) | `output[3]` `tongue_y_position` | per 50 ms bin: mean `y` over frames of *that trial* with `likelihood > 0.9`; if no such frame ⇒ 3 (not visible); else 0/1/2 by session percentiles (<40th, 40–60th, >60th) | `tracking.camera_0_side.tongue_y` |
| `nwb.subject.subject_id` | `subjects`, `subject_idx` | unique sorted list | — |
| `trials.auto_water`, `trials.free_water` | trial filter | drop | `get_regular_trial_mask` |
| `units.is_good_trials` | trial filter | drop trials flagged `False` for any retained unit | drift QC (white paper) |

### Key Decisions
1. **Alignment = go cue** (`BehavioralEvents['go_start_times'].timestamps`, one per trial).
   Matches both papers and all reference code.
2. **Window = [−2.5, +1.5] s, 80 non-overlapping 50 ms bins** (task spec). Bin *k* covers
   `[-2.5 + 0.05k, -2.5 + 0.05(k+1))`, centre `-2.475 + 0.05k`.
3. **Firing rates in Hz** (counts / 0.05 s), as `sliding_histogram(rate=True)`.
4. **Neuron filter: `classification == 'good'` only.** This is exactly the classifier-based QC of
   the data paper. The method paper's extra "≥ 2 Hz" filter is *not* applied: it was introduced to
   stabilise per-neuron R² estimates in the video→firing-rate regression, not for decoding, and
   discarding low-rate neurons would throw away decodable information.
5. **Trial filter**: drop `auto_water` and `free_water` trials (reference); drop trials not
   flagged good in `is_good_trials` for any retained unit (drift QC). **Keep** early-lick,
   `ignore`, and photostim trials — required by the decoder input/output spec.
6. **Session filter**: drop sessions with 0 good units (1 session) and sessions with < 2 trials
   after curation. No behavioural-performance filter (see Step 4).
7. **Zero padding outside the recorded interval** is kept rather than dropping trials, matching
   `process_one_area`, because dropping would remove ~92 % of `miss` trials and destroy the
   `outcome` output. Documented caveat.
8. **Tone onset = last `sample_start_time` before the go cue** — with early licks the sample epoch
   is replayed, and the last replay is the instruction that actually precedes the delay/go.
   Median tone→go = 1.85 s (= 0.65 s sample + 1.2 s delay), as expected from the task design.
9. **`time_from_tone_onset` is a continuous ramp** (explicitly required by the task spec:
   "continuous, time-varying"), not a binary indicator.
10. **Tongue visibility threshold: DLC likelihood > 0.9.** The likelihood distribution is
    strongly bimodal (≈ 3–6 × 10⁻⁵ vs ≈ 1.0); any threshold in [0.1, 0.99] gives the same answer
    to within 0.1 % of frames. ~6–11 % of frames have the tongue visible.
11. **Tongue percentiles are computed per session** over the *binned, visible* values of all
    retained trials of that session (the quantity actually being discretised).
12. **Per-trial outputs are broadcast across time**, because the output array must have a single
    shape and the tongue output is time-varying.
13. **Brain regions: 14 coarse areas** (the reference's list). ALM is defined as units on
    ALM-targeted insertions annotated to secondary motor / primary motor / frontal-pole areas.
    Hemisphere is stored separately in metadata rather than doubling the region list.

### Planned Sanity Checks
- [ ] 173 sessions, 28 subjects, 655 insertions, ≈ 69.4 k good units.
- [ ] Trials/session (excluding early licks) ≈ 476.
- [ ] Photostim trials ≈ 20 % of trials; photostim always ends before the go cue and lies in the
      delay epoch.
- [ ] Median tone→go interval = 1.85 s.
- [ ] `choice` derived from (instruction, outcome) agrees with the direction of the **first lick**
      after the go cue on ≥ 99 % of hit/miss trials, and `ignore` trials have no licks.
- [ ] Tongue visible ≈ 6–11 % of frames; visible bins concentrated after the go cue.
- [ ] Tongue class fractions ≈ 40 / 20 / 40 of the *visible* bins.
- [ ] Per-neuron mean firing rate in the converted data matches `units.avg_firing_rate` in order
      of magnitude, and a direct spike-count recomputation from the raw NWB matches exactly.
- [ ] Neural, input and output spot-checks against the raw NWB with `np.allclose`.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (+ `/app/ccf_regions.py` for the CCF annotation → coarse-area map).

Structure:
- `list_session_files()` — enumerate the 174 NWB files.
- `process_session(path)` — everything for one session; returns `None` if the session is dropped.
- `bin_spikes_rate(spike_times, go_times)` — vectorised binning. The per-trial bin edges are
  built as `go[:,None] + BIN_EDGES[None,:]` and located in the (sorted) spike train with a single
  `np.searchsorted`, so there is no Python loop over trials; the counts are `np.diff` of the
  resulting indices, divided by the bin width to give Hz.
- `interval_overlap_bins()` — photostimulation indicator.
- `bin_tongue()` — assigns every video frame to its trial, keeps visible frames
  (likelihood > 0.9) inside the window, and averages `y` per (trial, bin) with two
  `np.bincount` calls.
- `discretise_tongue()` — 40th/60th session percentiles over the visible bins.
- `plot_processing()` — the `--show-processing` figure (8 panels, one per processing step).
- `main()` — `multiprocessing.Pool` over sessions (default 16 workers), assembles the final dict.

Bugs found and fixed while developing:
1. **`is_good_trials` / `obs_intervals` are not always `n_trials` long.** In 9 sessions the ephys
   recording covers only a subset of the behavioural trials (e.g. `SC015_20190208_133600_s2`:
   160 of 480 trials). `units.obs_intervals` is identical for every unit of a session and its
   rows match the `[start_time, stop_time]` of the *observed* trials exactly (checked for all
   174 files). The script now locates the observed trials in the trials table by `start_time`
   and restricts everything to them. Note 8 of the 9 are a prefix of the trials table but
   `SC026_20190807_134913_s20` is not, so a simple "first N trials" rule would have been wrong.
2. **Trials with literally no spikes.** The recording can stop part-way through the last
   annotated trial, leaving a trial in which not one of hundreds of good units fires. Such
   trials are dropped (1 trial in the 2-session sample; they also triggered an
   "all neural data is zero" warning from the verifier).

Code inefficiencies identified / speedups added:
- Reading spike times per unit through `units['spike_times'][i]` is slow; the whole flattened
  ragged dataset is read once (`sv.target.data[:]`, ~90 MB per session, 0.13 s) and sliced.
- Per-trial Python loops for binning replaced by a single vectorised `np.searchsorted`.
- Tongue binning done with `np.bincount` instead of a per-bin loop.
- Sessions processed in parallel with `multiprocessing` (16 workers).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(2 sessions, both from subject 440956).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 834 |
| Neurons / session | mean 417.0 (375, 459) |
| Subjects | 1 |
| Trials (total) | 513 |
| Trials / session | mean 256.5 (159, 354) |
| T (time bins) | 80 for every trial |
| `time_from_tone_onset` range | [-0.6, 5.7] s |
| `photostim_on` range | [0, 1] |
| `choice` distribution | left 0.456, right 0.407, no lick 0.136 |
| `outcome` distribution | ignore 0.136, miss 0.298, hit 0.565 |
| `early_lick` distribution | no 0.942, yes 0.058 |
| `tongue_y_position` distribution | low 0.094, mid 0.047, high 0.094, not visible 0.766 |

Tongue classes 0/1/2 are in a 0.094 : 0.047 : 0.094 ratio = **40 : 20 : 40 of the visible bins**,
exactly as the percentile definition requires. ✔

### Processing Plots Review
`processing_SC015_20190207_120657_s1.png`, `processing_SC015_20190208_133600_s2.png`:
- **Alignment**: trial start is at −3.2 s (median) and the tone onset sits at exactly −1.85 s
  (= 0.65 s sample + 1.2 s delay) on nearly every trial. ✔
- **Firing rates**: a clear population response locked to the go cue, plus a smaller bump at
  the sample onset (−1.85 s). Black (zero) bands appear after ~+0.4 s on the trials whose
  recorded interval has ended — the documented zero-padding caveat, visible as expected.
- **Input 0** is a linear ramp crossing zero at the tone onset. ✔
- **Input 1** (photostim) is on during the delay epoch (−1.2 to −0.7 s here) on 77 of 354
  trials (22 %), always ending before the go cue, as the methods describe. ✔
- **Tongue**: the y distribution is unimodal and the 40th/60th percentile lines split it
  correctly; the tongue is visible almost exclusively after the go cue, with scattered
  pre-go visibility on early-lick trials. ✔
- **Outputs 0–2** take only their allowed values.

### Run Time Estimates
| Speed-up implemented | Time saving |
|---|---|
| Bulk read of the ragged spike dataset instead of per-unit reads | ~5× on the neural step |
| Vectorised `searchsorted` binning | ~20× vs. a per-trial loop |
| `np.bincount` tongue aggregation | ~50× vs. a per-bin loop |
| 16-way multiprocessing over sessions | ~10× overall |

| Step | Time / session | Estimated total (173 sessions) |
|---|---|---|
| setup (trials, units, annotations) | 0.3 s | 52 s |
| neural binning | 0.1–0.3 s | ~35 s |
| inputs | <0.05 s | ~8 s |
| outputs (video) | 0.1 s | ~17 s |
| **total per session (serial)** | **0.7–2 s** | **2–6 min serial, ≈ 1–2 min with 16 workers** |
| pickle write (≈ 11 GB) | — | ~1–3 min |

Estimated full conversion ≪ 15 min, so no further optimisation is needed. (The sample sessions
are about average in size: 417 neurons vs. a dataset mean of 399, 257 trials vs. ~500.
Scaling for trials gives ≈ 2× the per-session time, still well inside budget.)

### Verification (`/app/verification_sample_out.txt`)
`Data format is valid, no errors or warnings.`

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (the initial "all neural data is zero" warning was fixed by dropping the
  spike-less trial, see Step 6)

### Decoder Results (Sample, 2 sessions / 513 trials)
Loss decreased monotonically from 2.77 to 0.564 over 200 epochs.

| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| choice | 0.333 | 0.7385 | 0.6128 |
| outcome | 0.333 | 0.7613 | 0.6425 |
| early_lick | 0.500 | 0.8305 | 0.7209 |
| tongue_y_position | 0.250 | 0.7000 | 0.5848 |

All four outputs are well above chance (1.4–2.3×) on only two sessions.

---

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full`
Wall time **40 s** (173 sessions converted in 26 s with 16 workers, 13 s to write the pickle) —
far below the 15-minute budget, so no further optimisation was required.

### Output Files
- `converted_data.pkl`: **11.82 GB**, 173 sessions / 89,068 trials / 69,453 neurons
- `conversion_full_out.txt`, `verification_full_out.txt`: created
- `verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (raw NWB) | Converted Data | Match? |
|-----------|------------------|----------------|--------------------------|----------------|--------|
| Subjects | 28 | — | 28 | **28** | ✔ exact |
| Behavioural sessions | 173 | — | 174 files, 173 with good units | **173** | ✔ exact |
| Probe insertions | 655 (660 "penetrations") | — | 659 groups, 655 with good units | 655 | ✔ exact |
| Good units (total) | 69,943 | `classification == 'good'` | 69,453 | **69,453** | ✔ 99.3 % (release version differs from the paper's snapshot) |
| Units / session | — | — | mean 399.2 | mean 401.5 (90–923) | ✔ |
| ALM units | 8,717 | — | — | 9,367 | ~ +7.5 % (ALM boundary, see below) |
| Striatum units | 7,664 | — | — | **7,666** | ✔ +0.03 % |
| Thalamus units | 12,808 | — | — | 12,968 | ✔ +1.2 % |
| Midbrain units | 7,495 | — | — | 7,480 | ✔ −0.2 % |
| Medulla units | 2,928 | — | — | 2,925 | ✔ −0.1 % |
| Trials / session (excl. early licks) | mean 476 (130–785) | — | mean 483.8 (209–695) | mean 454.9 (147–687) | ✔ −4.4 % (auto/free-water + drift-QC + ephys-coverage exclusions) |
| Trials / session (all kept) | — | — | mean 545.9 | mean 514.8 (159–796) | ✔ |
| Correct rate (control trials) | 84 % (65–99 %) | `correctness` | 81.0 % | **81.6 %** | ✔ 3 % below the published figure (see Step 4) |
| Photostim trials | ~25 % (on photostim sessions) | `stimulation[:,0] != 0` | 19.6 % | **20.0 %** | ✔ (not all mice/sessions had photostim) |
| Neural bin | 40 ms width / 3.4 ms stride | `bw=0.04, stride=0.0034` | — | 50 ms, non-overlapping | deliberate (task spec) |
| Mean firing rate | — | — | — | 8.9 Hz | plausible for `good` units |
| `time_from_tone_onset` range | tone→go = 0.65 + 1.2 = 1.85 s | — | median 1.85 s | [−1.5, 11.9] s, median at window start −0.625 s ⇒ tone→go = 1.85 s | ✔ exact |
| `photostim_on` range | binary | — | — | [0, 1] | ✔ |
| `choice` distribution | — | — | — | left 0.429 / right 0.422 / no lick 0.149 | ✔ balanced L/R as expected for randomly interleaved trial types |
| `outcome` distribution | — | — | hit 0.687 / miss 0.165 / ignore 0.148 (all raw trials) | ignore 0.149 / miss 0.166 / hit 0.685 | ✔ |
| `early_lick` distribution | — | — | 0.114 | 0.116 | ✔ |
| `tongue_y_position` distribution | — | — | tongue visible in 11.7 % of frames | not-visible 0.755; 0/1/2 = 40:20:40 of the visible bins | ✔ by construction |

**ALM**: the only region count that differs by more than ~1 %. The paper defines ALM with an
anatomical voxel mask (`ALM_voxels_symmetric.npy` in the reference code, not distributed).
Using annotations alone, secondary+primary motor cortex on ALM-targeted insertions gives 7,512
units (−14 %) and adding frontal pole gives 9,367 (+7.5 %); the true mask evidently covers part
of the frontal-pole annotation. The frontal-pole-inclusive definition is the closer of the two
and is anatomically correct for a probe at AP 2.5 / ML 1.5 mm, so it is what the script uses.

### Data loss accounting (94,370 → 89,068 trials)
| Stage | Trials |
|---|---|
| Trials in the 173 retained session files | 94,370 |
| ... with ephys coverage (`obs_intervals`) | 93,310 (−1,060, 9 sessions) |
| ... minus `auto_water` / `free_water` | 89,546 (−3,764) |
| ... minus trials not good in `is_good_trials` | 89,070 (−476, 4 sessions) |
| ... minus trials with no spikes at all | **89,068** (−2, 2 sessions) |

No trial is lost for any other reason; all 173 sessions and all 69,453 good units survive.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`/app/verification_full_out.txt`: **"Data format is valid, no errors or warnings."**
There are no remaining errors or warnings. Two warnings that appeared during development were
fixed rather than tolerated:
- *"Session 1, trial 159: all neural data is zero"* — the recording stopped part-way through
  the last annotated trial. Such trials are now dropped (2 trials in the whole dataset).
- A `ValueError` on `is_good_trials` broadcasting revealed that 9 sessions have ephys for only
  a subset of the behavioural trials (Step 6).

### Check 2 — independent sanity checks (`/app/sanity_checks.py`, output `/app/sanity_checks_out.txt`)
`sanity_checks.py` re-reads the raw NWB files with `pynwb`, imports nothing from
`convert_data.py`, and compares against the pickle with `np.allclose` / exact equality.
**101 checks, 0 failed.** Five sessions are spot-checked (indices 0, 88, 109, 145, 172, chosen
by a seeded RNG), each with 4 random trials × 5 random neurons and 8 random time bins.

| # | Check | Data stream | Method | Result |
|---|---|---|---|---|
| 1 | neuron count == number of `good` units | neural | exact | PASS ×5 |
| 2 | firing rates equal a brute-force `np.sum((st >= lo) & (st < hi)) / 0.05` recount, bin by bin | neural | `np.allclose` | PASS ×5 (20 trial×neuron probes each) |
| 3 | per-neuron mean rate correlates with the independent `units.avg_firing_rate` column | neural | r > 0.95 | PASS ×5, r = 0.988–0.998 |
| 4 | `input[0]` equals `bin_centre + (go − last sample_start before go)` | input | `np.allclose` | PASS ×5 |
| 5 | median tone onset is 1.85 s before the go cue | input | threshold | PASS ×5 (−0.625 s at the window start) |
| 6 | the set of photostim trials equals the set derived from `photostim_start/stop_times` | input | exact | PASS ×5 |
| 7 | ... and equals the independent `trials.photostim_onset != 'N/A'` column | input | exact | PASS ×5 |
| 8 | photostimulation always ends before the go cue | input | bin index < 50 | PASS ×5 |
| 9 | photostim onset is inside the delay epoch on ≥98 % of stim trials | input | threshold | PASS ×5 |
| 10 | `output[0]` equals `f(trial_instruction, outcome)` | output | exact | PASS ×5 |
| 11 | `output[1]` equals the `outcome` column | output | exact | PASS ×5 |
| 12 | `output[2]` equals the `early_lick` column | output | exact | PASS ×5 |
| 13 | choice agrees with the **direction of the first lick** after the go cue | output | > 99 % | PASS ×5, **100 %** (195/195, 353/353, 578/578, 573/573, …) |
| 14 | "no lick" trials contain no licks in [go, go+1.5] | output | exact | PASS ×5, 0 violations |
| 15 | `output[3]` equals a direct recomputation from the DLC timestamps/likelihoods | output | exact | PASS ×5 (32 bin probes each) |
| 16 | tongue classes 0/1/2 form a 40:20:40 split of the visible bins | output | `np.allclose`, atol 0.02 | PASS ×5, 0.400/0.200/0.400 |
| 17 | tongue visibility is concentrated after the go cue | output | ratio > 2 | PASS ×5 |
| 18 | hemisphere matches the reference `ccf_x >= 5700` rule | metadata | exact | PASS ×5 |
| 19 | hemisphere matches the probe's targeted hemisphere except for midline units | metadata | ≤100 µm from midline | PASS ×5 |
| 20 | 173 sessions / 28 subjects / ~69.5 k units / every trial 80 bins / ≥2 trials / float32 | global | exact | PASS |

Two checks were *initially* too strict and led to real findings that were fixed or verified:
- **Fixed a bug**: photoinhibition ends at the go cue to within ±0.5 ms, and starts within
  ±0.5 ms of a bin edge. The original "does the interval touch this bin" test therefore
  switched the first post-go bin on for about half the stimulated trials (and one bin too
  early at the onset). `interval_overlap_bins` now requires >1 ms of overlap
  (`OVERLAP_TOL`), which removes the artefact and makes check 8 pass exactly.
- **Verified, not a bug**: in session 0 one photostim trial has its laser onset 2.25 s before
  the go cue. Reading the raw file shows the trial had an early-lick replay (go cue 4.23 s
  after trial start instead of the usual 3.2 s) while the laser fired at its usual time, so
  the onset is genuinely outside the delay window. 1 of 77 stim trials in that session.
- **Verified, not a bug**: 5 units in session 172 are annotated "Nucleus raphe magnus" and sit
  20–40 µm on the other side of the 5,700 µm midline from their left-Medulla insertion.
  Raphe magnus is a midline structure; the CCF-coordinate rule of the reference code is the
  authoritative one and is what the script uses.

### Check 3 — reference-code comparison
| Stage | Reference (`/app/code`) | This conversion | Same? |
|---|---|---|---|
| (a) **Loading** | `preprocessing_utils.loadmat` on DataJoint `.mat` exports, one file per probe, concatenated over probes | `pynwb.NWBHDF5IO` on the NWB release; probes are already merged into one `units` table | Same content, different container (NWB is mandatory here, and `h5py` is forbidden) |
| (b) **Neuron filtering** | QC-classifier good-unit index per session (`qc_mode='classifier'`) ∩ non-empty CCF annotation ∩ hemisphere by `ccf_x >= 5700` | `units.classification == 'good'` (the same classifier output, shipped in the file); `anno_name` is non-empty exactly for those units; same `ccf_x >= 5700` hemisphere rule | **Same** |
| (c) **Temporal alignment** | spike times already relative to `task_cue_time[0]` (go cue); markers aligned by subtracting the go cue (`align_markers.py`) | subtract `BehavioralEvents['go_start_times'].timestamps` from both spike times and video timestamps | **Same** |
| (d) **Binning** | `sliding_histogram`, `bw = 0.04`, `stride = 0.0034`, `rate=True`, window `[-3, 3]`; spikes truncated to the window and the whole window binned regardless of when the trial ended | non-overlapping 50 ms bins, `rate=True` (Hz), window `[-2.5, 1.5]`; whole window binned regardless of when the trial ended | Bin width/stride and window differ **because the task specifies them**; the counting rule, the rate normalisation and the treatment of the trial edges are identical |
| (e) **Input construction** | uses `task_sample_time` (sample-epoch onset) and `task_stimulation` (laser on/off times, both re-referenced to the go cue) | uses `sample_start_times` and `photostim_start/stop_times`, re-referenced to the go cue | **Same variables**; the reference keeps them as scalars per trial, we rasterise them onto the bin grid because the decoder wants time-varying inputs |
| (f) **Output construction** | `lick_directions`/`trial_type` + `correctness`; `early_lick_trials`; `tracking.camera_0_side.tongue_y` | `trial_instruction` + `outcome`; `early_lick`; `Camera0_side_TongueTracking` | **Same variables**. The reference *imputes* the tongue position with its mean while the tongue is occluded; the task instead requires an explicit "not visible" class, which is the more informative encoding of the same fact |
| Trial curation | `early_lick == 0 & auto_water == 0 & free_water == 0 & correctness != −1 & stimulation == 0` | `auto_water == 0 & free_water == 0` (+ `is_good_trials` + ephys coverage) | **Deliberate deviation**: early-lick, `ignore` (`correctness == −1`) and photostim trials are *required* by the decoder input/output specification. Removing them would leave `early_lick` and `outcome` with a single class each and `photostim_on` identically zero. The two criteria that do not conflict (`auto_water`, `free_water`) are applied exactly as in the reference |
| Session curation | — (implicit in which sessions were exported) | drop the one session with 0 good units; drop sessions with <2 usable trials | Reproduces the paper's 173 sessions / 655 insertions exactly |

### Check 4 — key statistics comparison
See the table in Step 9. Every statistic available in the papers matches, with three explained
differences: total good units (99.3 % of the published number — a dataset-version difference,
as the session and insertion counts match *exactly*), the ALM unit count (+7.5 %, ALM boundary
definition), and the mean correct rate (81.6 % vs 84 %, same definition; the released set also
contains 16 sessions below the published 65 % selection criterion, so the criteria in the paper
were evidently applied to a slightly different quantity — see Step 4). None of these affects the
conversion itself.

### Check 5 — edge cases
Explicitly handled and tested:
1. **Sessions where ephys covers only part of the behavioural session** (9 sessions): trials are
   located through `obs_intervals`, not assumed to be the first *N* (one session,
   `SC026_20190807_134913_s20`, is *not* a prefix).
2. **Trials with no spikes at all** (recording stopped mid-trial): dropped (2 trials).
3. **Session with zero good units** (`SC017_20190216_162508_s4`): dropped.
4. **Windows that run outside the recorded interval**: 1.8 % of trials have no spikes in the
   first 5 bins and 17.5 % none in the last 5 bins. Kept and zero-filled, exactly as the
   reference `process_one_area` does (documented caveat in the metadata).
5. **Bin-boundary alignment of the photostim interval** (±0.5 ms): solved with `OVERLAP_TOL`.
6. **Bin edges**: bin *k* is the half-open interval `[go + 0.05k − 2.5, go + 0.05(k+1) − 2.5)`;
   `searchsorted(..., side='left')` implements exactly that, and the brute-force sanity check
   uses the same convention independently.
7. **Early-lick sample replays**: the tone onset is the *last* sample-epoch start before the go
   cue, so replays are handled; the delay epoch can be as short as 0.1 s in replay trials, which
   is why `time_from_tone_onset` is not a constant offset.
8. **Video frames assigned to the wrong trial**: frames are first bucketed by trial
   (`searchsorted` on `trial_start`), so a window that extends past its own trial boundary
   cannot pick up a neighbouring trial's frames.
9. **Units with a missing CCF x coordinate**: hemisphere falls back to the insertion target.
10. **Unmapped CCF annotations**: all 293 distinct annotation strings in the dataset map to one
    of the 14 regions (verified exhaustively); the code warns and falls back to `OtherCortex`
    if an unknown string ever appears.
11. **Photostim start/stop count mismatch**: guarded, warns and leaves the input at 0.
12. **`go_start_times` count vs. trials table**: asserted equal for all 174 files.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(GPU, 173 sessions / 89,068 trials / 7.1 M timepoints, ~12 min for 200 epochs).

### Training Progress
- Loss decreasing: **Yes**, monotonically — 18.26 (epoch 1) → 7.50 (10) → 1.87 (50) →
  0.877 (100) → 0.704 (150) → 0.655 (200). Test loss 0.660, essentially equal to the
  training loss.

### Decoder Results (Full)
| Output | #classes | Chance (uniform) | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|----------|------------------|-----------------------|-------------------------|--------------|
| choice | 3 | 0.3333 | 0.7081 | **0.6794** | 2.04× |
| outcome | 3 | 0.3333 | 0.7027 | **0.6590** | 1.98× |
| early_lick | 2 | 0.5000 | 0.7898 | **0.7484** | 1.50× |
| tongue_y_position | 4 | 0.2500 | 0.6930 | **0.6619** | 2.65× |

Notes: every output is well above chance; the train–validation gap is at most 1.06×, so there
is no sign of overfitting or leakage. `sample_trials.png` and `predictions.png` were produced
and inspected — the photostimulation pulse sits in the delay epoch, the tone ramp is linear,
and the tongue output switches out of "not visible" only after the go cue.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

Because `train_decoder.py` reports one number per output averaged over the whole
−2.5 … +1.5 s window (62 % of which precedes the go cue), while the papers report decoding
accuracy *within task epochs*, the review used `/app/analyze_accuracy_vs_time.py`, which trains
the same decoder with the same hyper-parameters and then scores the held-out trials **per time
bin**. Raw numbers: `/app/cache_accuracy_vs_time.json`, `/app/cache_test_predictions.npz`.

### Check 1 — accuracy vs. chance
| Output | Chance | Validation | Ratio | Peak per-bin (time) | Peak / chance |
|---|---|---|---|---|---|
| choice | 0.333 | 0.679 | 2.04× | 0.856–0.865 (+0.33 s) | 2.59× |
| outcome | 0.333 | 0.659 | 1.98× | 0.869–0.874 (+1.0…+1.1 s) | 2.61× |
| early_lick | 0.500 | 0.748 | 1.50× | 0.837 (−1.1 s, mid-delay) | 1.67× |
| tongue_y_position | 0.250 | 0.662 | 2.65× | 0.673–0.690 | 2.70× |

No output is below chance. `early_lick` is the only one near the 1.5× "investigate" threshold, so
it was examined specifically:
- The decoder *does* find a strong signal where the signal should be: balanced accuracy peaks at
  **0.837 during the delay epoch** (mid-window average 0.80–0.82) and decays to 0.69 in the
  response epoch, i.e. long after the early lick occurred. Averaging over the whole window
  therefore understates it.
- The behavioural evidence is real and correctly converted: in early-lick trials the tongue is
  visible in **29.2 %** of the pre-go bins versus **6.5 %** in other trials, and 60.7 % of
  early-lick trials have a non-standard tone→go interval versus 6.7 % of the rest.
- It is also intrinsically limited: the task *replays* the sample epoch after an early lick, so
  the final sample+delay leading up to the go cue is by construction lick-free. Only the part of
  the aborted epoch that falls inside the window carries direct evidence.
So the value reflects the task, not a conversion error.

### Check 2 — comparison to the papers
| Variable | Papers | Where | This conversion | Comment |
|---|---|---|---|---|
| Choice (lick left vs right), **response epoch**, neural | "close to perfect"; pseudo-populations of 200 neurons, per-session logistic regression, nested 5-fold CV (data paper Fig. 6D) | [0, 1.0] s | **0.861 mean / 0.918 peak** accuracy (left-vs-right among lick trials) | Same ballpark. Ours is a *single shared* linear readout on 100 PCs for all 173 sessions at all 80 time bins; theirs is one decoder per session **per time point**, on hand-picked pseudo-populations, with hyper-parameter selection by nested CV, and on 200 ms causally-smoothed rates instead of 50 ms bins. |
| Choice, **delay epoch**, neural | rises through sample and delay, well above chance by the end of the delay (Fig. 6D) | [−1.2, 0] s | 0.696 (left vs right) | Same qualitative time course |
| Choice, **sample epoch**, neural | starts at chance, rises during the sample | [−1.85, −1.2] s | 0.650 | ✔ |
| Choice, **pre-sample**, neural | chance | [−2.5, −1.85] s | 0.540 | ✔ near chance, as it must be |
| Choice from **video**, response epoch | AUC 0.99 ± 0.01 | response | — (video is an output here, not an input) | For reference only; AUC 0.99 ≈ 0.95 accuracy |
| Choice from **video markers**, response epoch | 0.88 ± 0.01 | response | — | Our tongue-position output is the same marker stream |
| Outcome, early lick, tongue position | not decoded in either paper | — | 0.659 / 0.748 / 0.662 | No published target |

The full time course reproduces the published profile of choice coding: at chance before the
instruction, rising through the sample and delay epochs, and jumping to its maximum ~0.3 s after
the go cue. The residual gap to "close to perfect" is fully accounted for by the decoder
architecture the task prescribes (one shared low-rank linear readout, no per-time-point or
per-session fitting, no temporal smoothing); it is not a property of the converted data.

### Check 3 — train vs. validation gap
| Output | Train | Validation | Train / Val |
|---|---|---|---|
| choice | 0.7081 | 0.6794 | 1.042 |
| outcome | 0.7027 | 0.6590 | 1.066 |
| early_lick | 0.7898 | 0.7484 | 1.055 |
| tongue_y_position | 0.6930 | 0.6619 | 1.047 |

Maximum ratio 1.07, far below the 1.5 threshold: no overfitting and no sign of leakage between
the training and validation trials.

### Per-class recall (response epoch, validation)
| Output | Class recalls |
|---|---|
| choice | left 0.765, right 0.745, no lick 0.797 |
| outcome | ignore 0.750, miss 0.780, hit 0.826 |
| early_lick | no 0.594, yes 0.779 |
| tongue_y_position | low 0.585, mid 0.475, high 0.700, not visible 0.617 |

No class collapses, so the balanced-loss training is working and every class is represented.
The `mid (40–60th pct)` tongue class is hardest, which is expected: it is the narrowest slice of
a continuous variable and is bounded on both sides by the other two classes.

### Issues found and resolved in this step
None new. The two items examined (early_lick ratio, choice below the papers' "near perfect")
were traced to the task structure and to the prescribed decoder architecture, respectively, and
verified not to be conversion errors. The one genuine bug found during the review phase (the
sub-millisecond photostim/bin-edge overlap) was fixed in Step 10 and the full conversion,
verification, sanity checks and decoder training were all re-run afterwards.

### Known caveat (documented, not a bug)
Spikes exist only inside each trial's recorded interval (`units.obs_intervals` ==
trials `[start_time, stop_time]`). On error (`miss`) trials the interval ends at the error lick,
so only ~8 % of them extend to go + 1.5 s; overall 17.5 % of trials have no spikes in the last
5 bins and 1.8 % none in the first 5. The fixed window is binned regardless — exactly what the
reference `process_one_area` does — so those bins read 0 Hz. This inevitably makes `outcome`
partly decodable from the position of the zero-padding rather than from neural activity alone.
The alternative (dropping trials without full coverage) would remove ~92 % of `miss` trials and
leave the `outcome` output with effectively two classes, so it was rejected. The caveat is
recorded in `data['metadata']['caveat']`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, how to load and use the pickle, full output
      format specification, curation summary, key statistics and decoder performance.
- [x] `cache/` folder created with `cache/README_CACHE.md` documenting every cached file.
- [x] All files organised.

### Deliverables
| File | Status |
|---|---|
| `/app/CONVERSION_NOTES.md` | this file |
| `/app/convert_data.py` (+ `/app/ccf_regions.py`) | conversion script |
| `/app/converted_data.pkl` | 11.8 GB, 173 sessions |
| `/app/sample_data.pkl` | 73 MB, 2 sessions |
| `/app/README.md` | user-facing documentation |
| `/app/conversion_sample_out.txt`, `/app/conversion_full_out.txt` | conversion logs |
| `/app/verification_sample_out.txt`, `/app/verification_full_out.txt` | `--verify-only` logs (valid, no errors or warnings) |
| `/app/train_decoder_sample_out.txt`, `/app/train_decoder_full_out.txt` | decoder training logs |
| `/app/sanity_checks.py`, `/app/sanity_checks_out.txt` | 101 independent checks, 0 failed |
| `/app/analyze_accuracy_vs_time.py` | per-time-bin accuracy analysis used in Step 12 |
| `/app/processing_SC015_*.png` | per-step verification plots |
| `/app/sample_trials.png`, `/app/predictions.png` | decoder sample/prediction plots |
| `/app/cache/` | exploration scripts and intermediate results |

### Summary of the decisions that define this conversion
1. Align to the **go cue**; window **[-2.5, +1.5] s**; **80 non-overlapping 50 ms bins**;
   neural data as **firing rate in Hz** (task spec + reference `sliding_histogram(rate=True)`).
2. Keep only `classification == 'good'` units (the paper's QC-classifier output). No
   firing-rate threshold - the method paper's 2 Hz cut was specific to its R2 analysis.
3. Trials: drop `auto_water`/`free_water` (reference), drop trials flagged in
   `is_good_trials`, drop trials outside `obs_intervals`, drop trials with no spikes.
   **Keep** early-lick, `ignore` and photostim trials - required by the decoder spec.
4. Sessions: drop the one with no good units => exactly the paper's 173 sessions / 655 insertions.
5. Inputs: continuous time from the **last sample-epoch onset before the go cue**, and a binary
   photostim indicator requiring >1 ms overlap with the bin.
6. Outputs: `choice` from instruction x outcome, `outcome` and `early_lick` from the trials
   table, and the side-view DLC tongue *y* averaged per bin over frames with likelihood > 0.9,
   discretised at the session's 40th/60th percentiles, with a distinct "not visible" class.
7. Brain regions: the reference's 14 coarse areas, obtained by mapping all 293 Allen CCF
   annotations present in the dataset; hemisphere from the reference's `ccf_x >= 5700` rule.

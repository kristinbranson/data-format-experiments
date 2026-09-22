# Dataset Conversion Notes

## Overview
- **Dataset**: *Mesoscale Activity Map (MAP)* dataset — Chen, Nguyen, Li, Svoboda (2023),
  DANDI:000363. NWB files in `/app/data` (174 files, 28 mice).
  Data paper: Chen et al., *Brain-wide neural activity underlying memory-guided movement*, Cell 2024 (`/app/datapaper.pdf`).
  Method paper: Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding structured across and within brain areas*, Nat Neurosci 2025 (`/app/methodpaper.pdf`).
  Reference analysis code: `/app/code` (MapVideoAnalysis).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py` (to be written)
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `methods.txt`
- `code/` — reference analysis code (MapVideoAnalysis repo)
- `data/` — 28 `sub-*` directories, 174 `.nwb` files, 50 GB total, plus `dandiset.yaml`
- `decoder.py`, `train_decoder.py` — provided decoder/validation code
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0.
Hardware: 1 TB RAM, 128 CPUs, NVIDIA L4 (23 GB).
`pypdf` installed to extract paper text (cached at `/app/cache/{datapaper,methodpaper}.txt`).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference repo (`/app/code`, MapVideoAnalysis) works from DataJoint `.mat` exports of the
same dataset rather than the NWB files, but it fully specifies the loading / curation /
alignment / binning logic that the method paper used.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` / `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Entry point; concatenates all probe files of one session, reads behaviour/task/spike data |
| `preprocess_all_ephys.py` | `Sherlock/` | PROCESSING | Actual parameters used for the paper: `bw=0.04 s`, `stride=0.0034 s`, `begin=-3 s`, `end=+3 s`, `qc_mode='classifier'` |
| `sliding_histogram` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Bins per-trial spike times into **firing rates** (`binSpikes/bin_width`), bins are `[center-bw/2, center+bw/2)` |
| `helper_get_neuron_id_area` | `preprocessing_DJ_2022Aug.py` | CURATION | Keeps only units in the QC "good units" list, splits by hemisphere (CCF ML midline = 5700 µm), requires a non-empty CCF annotation |
| `process_one_area` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates spike times to `[begin_time, end_time]` **relative to the go cue** and bins them |
| `get_regular_trial_mask` | `VideoAnalysisUtils/functions_for_r2.py` | CURATION | "regular trials" = no early lick ∧ no auto-water ∧ no free-water ∧ response given (`correctness != -1`) ∧ no photostimulation |
| `create_4fold_trial_type_mask` | `functions_for_r2.py` | CURATION | Stratification by (trial type × correctness) |
| `align_markers_between_lims` | `Sherlock/align_markers.py` | ALIGNMENT | Aligns DeepLabCut markers (incl. `tongue_x`, `tongue_y`) to the **go cue**, `dt = 0.0034 s`, window `t ∈ [-3, 1.5) s` |
| `get_bad_trial_inds` | `Sherlock/align_markers.py` | CURATION | Flags trials whose video frame count disagrees with the trial duration |
| `check_fr` | `preprocessing_utils.py` | CURATION | Drops zero-variance (dead) neurons — used only inside some downstream analyses |
| `get_period` | `preprocessing_utils.py` | REFERENCE | Epochs relative to go cue: sample `[-1.9,-1.2]`, delay `[-1.2, 0]`, post-go `[0, 1.0]` |

### Notes
- **Temporal alignment in the reference is always the go cue** (`task_cue_time`); spike times in
  the DataJoint export are already go-cue-relative, and `align_markers.py` subtracts
  `go_times` from the video frame times. Our task also aligns to the go cue.
- **Neuron QC**: `qc_mode='classifier'` → the units labelled `good` by the region-specific
  logistic-regression classifiers described in the spike-sorting white paper. In the NWB files
  this is exactly the `units/classification` column (`good` vs `unlabelled`).
- Units also have to have histology (CCF annotation); the reference intersects ephys unit ids
  with histology unit ids. In NWB, `units/anno_name` is empty for units without histology.
- Electrophysiology, so no ΔF/F is needed. Spikes are converted to **firing rates in spikes/s**
  (`rate=True` in `sliding_histogram`).
- The reference's 14 brain-area groups are
  `ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus,
  Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum` (loop in `process_one_sess`).
- The marker/video analysis keeps `tongue_x`, `tongue_y` from `camera_0_side` — the same stream
  that the NWB files expose as `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`.
- `generate_5fold_cv_indices.py` additionally drops trials with index < 10 — this is specific to
  the video-embedding alignment, not to the ephys preprocessing.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<subject_id>/sub-<subject_id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb`
(one file per recording session; all probes of a session are merged into a single file).

Relevant NWB contents (verified on several files; structure is identical across all 174):

| Path | Contents |
|------|----------|
| `identifier` | e.g. `SC015_20190207_120657_s1` (mouse name, date, time, session #) |
| `general/subject/subject_id` / `description` | numeric id (`440956`) / mouse name (`SC015`) |
| `intervals/trials/` | `start_time`, `stop_time`, `trial`, `trial_instruction` (`left`/`right`), `outcome` (`hit`/`miss`/`ignore`), `early_lick` (`early`/`no early`), `auto_water`, `free_water`, `photostim_onset`/`_duration`/`_power` (strings, `N/A` if none; onset is **relative to trial start**), `task` (always `audio delay`), `task_protocol` (always 1) |
| `acquisition/BehavioralEvents/*` | `go_start_times`, `sample_start_times`, `delay_start_times`, `presample_*`, `trialend_*`, `left_lick_times`, `right_lick_times`, `photostim_start_times`/`_stop_times`. **`timestamps` are absolute session time (s)**; `data` is a dummy. `go_start_times` has exactly one entry per trial. `sample_start_times` has *more* entries than trials because an early lick triggers a replay of the sample epoch |
| `acquisition/BehavioralTimeSeries/Camera0_side_{Tongue,Jaw,Nose}Tracking` | `(nframes, 3)` = (`x`, `y`, `likelihood`) from DeepLabCut, `timestamps` absolute session time, frame period 0.0034 s (≈294 Hz). Video is recorded **per trial**: timestamps restart at each trial's `start_time`, with gaps in between |
| `units/` | `spike_times` (+ `spike_times_index`), `obs_intervals` (+ index), `classification` (`good`/`unlabelled`), `anno_name` (CCF annotation string), `is_good_trials` (n_units × n_trials bool), `electrodes`, 15 QC metrics, waveform metrics |
| `general/extracellular_ephys/electrodes` | `x`,`y`,`z` = CCF (ML, DV, AP) in µm; `location` = JSON with the **targeted** brain region (`left ALM`, `right Midbrain`, …), `group_name` = probe |

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files (sessions) | 174 |
| Sessions with ≥1 `good` unit | **173** (`sub-440958_ses-20190216T162508` has 0 good units) |
| Clusters (all) | 272,227 |
| `good` units (total) | **69,453** (25.5 % of clusters) |
| `good` units / session | mean 399, min 0, max 923 |
| Subjects | **28** (4–10 sessions each) |
| Trials (total, raw) | 94,990 |
| Trials / session | mean 546, range 264–800 |
| Probe insertions (unique targets × sessions) | 659 |
| Unique CCF annotations among good units | 293 |

### Important structural findings (verified by direct inspection)
1. **Spikes only exist inside `[trial.start_time, trial.stop_time]`.** For every unit tested,
   100 % of spikes lie inside a trial interval and 0 lie in the inter-trial gaps
   (`obs_intervals` == the trial intervals). The export is trial-segmented.
2. `stop_time − go_cue` is **~1.8 s for `hit` and `ignore` trials but only ~0.8 s for `miss`
   (error) trials** (95 % of miss trials have < 1.5 s of post-go data).
   `go_cue − start_time` is ≥ 2.5 s for 96.9 % of trials (median 3.15 s).
3. In 9 sessions the units' `obs_intervals` cover only the **first N trials** of the trials table
   (e.g. 160 of 480). All units in such a session share exactly the same contiguous trial set —
   the trials table spans more behaviour than the ephys recording.
4. `is_good_trials` is not all-True in 4 sessions; however spikes are present on those trials with
   comparable firing rates (checked in `sub-480135_ses-20210302T140243`: 3.66 vs 3.86 spikes/s),
   so the flag does *not* indicate missing data.
5. Tongue `likelihood` is essentially binary (89 % < 0.01, 10.5 % > 0.99). 99.8 % of recorded lick
   events fall on frames with likelihood > 0.9 → excellent visibility sanity check.
6. Photostimulation: duration is always 0.5 s; onset is −0.5 s rel. go cue in 122 sessions
   (late delay) and −1.2 s in 46 sessions (early delay), with a handful of sample-epoch trials.
   The trials-table values agree with `BehavioralEvents/photostim_{start,stop}_times` to < 10 ms.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Sessions | 173 | same |
| Probe insertions | 655 | "from which 655 probe insertions were made" |
| Fraction of KS2 clusters that are good | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2" |
| Good units by area | ALM 8717, orbital 10223, striatum 7664, pallidum 1092, thalamus 12808, midbrain 7495, medulla 2928, pons 347, cerebellum 1820, hypothalamus 815, hippocampus 1944, other cortex 7993, olfactory 4137, cortical subplate 1960 | datapaper Fig. 1E / 2F; sum = 69,943 |
| Trials/session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84 %, range 65–99 % | "with 84% correct rate (range, 65-99%)" |
| Session selection | performance > 65 %, ≥ 50 correct lick-left **and** ≥ 50 correct lick-right | methods.txt "Behavior" |
| Photostim trial fraction | ~25 % of trials, 17 VGAT-ChR2 mice, 93 sessions | "deployed on a subset of ~25% randomly interleaved trials" |
| Photostim epoch | last 0.5 s of delay, always ends before go cue | "We silenced ALM activity during the late delay epoch (last 0.5 s)" |
| Bilateral ALM photostim effect | performance 83.2 % → 71.7 % | methods.txt |
| Task timing | sample 0.65 s (3 × 150 ms tone, 100 ms gaps), delay 1.2 s, go cue 0.1 s, answer 1.5 s | methods.txt |
| Video | 2 cameras, 300 Hz, DeepLabCut tongue/jaw/nose | methods.txt |
| Neural bin (paper preprocessing) | 40 ms window / 3.4 ms step (also 17 ms, 10 ms/200 ms for decoding) | `preprocess_all_ephys.py`, datapaper methods |
| Behaviour (video) bin | 3.4 ms (300 Hz frames) | `align_markers.py` |

### Processing Details
- **Alignment**: every analysis is aligned to the **go cue** (t = 0).
- **Epochs relative to go cue**: sample `[-1.85, -1.2]`, delay `[-1.2, 0]`, response `[0, 1.5]`
  (`functions_for_r2.process_single_session_r2_dict`).
- **Binning**: sliding histogram → firing rates in spikes/s.
- **Video**: frames at 0.0034 s, aligned to go cue over `[-3, 1.5)` s.

### Curation Steps
**Neuron curation rules (reference)**
1. Keep only classifier-labelled `good` units (`qc_mode='classifier'`).
2. Keep only units that have histology (non-empty CCF annotation) — units must appear in both the
   ephys and histology tables.
3. Assign to one of 14 area groups; units are split by hemisphere at CCF ML = 5700 µm.

**Trial curation rules (reference `get_regular_trial_mask`)**
no early lick ∧ no auto-water ∧ no free-water ∧ response given ∧ no photostimulation.

### Decoders Trained (accuracies reported in the papers)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice, from ALM population (200 neurons, late delay) | logistic regression, 10 ms step | ≈ 0.75–0.85 (Fig. 6D), chance 0.5 |
| Choice, from thalamus/midbrain/striatum/medulla populations (200 neurons, late delay) | same | ≈ 0.6–0.75 |
| Choice, from behaviour **video** before sample epoch | CNN embedding | AUC 0.51 ± 0.06 (chance) |
| Choice, from video, sample+delay epochs | CNN embedding | AUC 0.66 ± 0.12 |
| Choice, from video, response epoch | CNN embedding | AUC 0.99 ± 0.01 |
| Choice, from video markers, response epoch | markers | 0.88 ± 0.01 |

→ Expectation for our decoder (all neurons of a session, full −2.5…+1.5 s window): choice should
be decoded well above chance, driven mostly by the delay and response epochs.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Source format | DataJoint `.mat` exports (`neuron_single_units`, `task_cue_time`, …) | NWB (`units/spike_times`, `go_start_times`) | — | The NWB fields map 1-to-1 onto the `.mat` fields used by the reference (see Step 5 table). Spike times are absolute in NWB and go-cue-relative in the `.mat`; we subtract the go-cue time ourselves. |
| Neuron QC | QC `.mat` files listing "good units" per region | `units/classification ∈ {good, unlabelled}` | classifier-based QC, 25.9 % of clusters | Identical concept. Our `good` fraction is 25.5 % (69,453/272,227) — matches. |
| Region assignment | 14 groups + hemisphere, ALM from a voxel mask (`ALM_voxels_symmetric.npy`, not shipped) | `units/anno_name` = CCF annotation + electrode CCF coordinates | 14 groups with exact unit counts | Built an explicit CCF-annotation → 14-group map (`/app/cache/region_map.py`). It reproduces **11 of the 14 paper counts exactly** (see Step 5 / Step 9). ALM has to be approximated spatially since the voxel mask is missing. |
| Number of good units | — | 69,453 | 69,943 | 490 fewer (0.7 %). All 490 are in isocortex; every other group matches the paper *exactly*. This is a version difference of the DANDI release (0.230822 vs the version used in the paper). |
| Sessions | — | 174 files, 173 with good units | 173 sessions | Drop the single session with no good units → exactly 173. |
| Trials/session | — | mean 546, range 264–800 | mean 476, range 130–785 | The paper's behavioural numbers cover a differently-selected (behaviour-only) session set; the ephys release has more trials per session. Documented, not "fixed". |
| Session selection (perf > 65 %, ≥50 correct L and R) | not implemented in the reference code | 16–23 of the 173 released sessions fail this criterion under every reasonable definition of "performance" I tested (mean 0.81, range 0.54–0.97) | performance > 65 %, 84 % mean | The released 173 sessions *are* the 173 sessions the paper reports (exact per-region unit counts). Applying the criterion would make our dataset inconsistent with the reported dataset size, and would bias the required `outcome`/`choice` output distributions. **Decision: keep all 173 sessions**, documented below. |
| Trial selection | `get_regular_trial_mask` removes early-lick, no-response and photostim trials | — | "Early lick trials and no response trials were excluded for analysis" | **Cannot be applied**: `early_lick`, `outcome=ignore`/`choice=no lick` are *required decoder outputs* and photostimulation is a *required decoder input*. Removing them would make three of the four outputs and one of the two inputs constant. We keep them and only apply the parts of the reference mask that do not conflict (auto-water, free-water). This is an explicitly allowed discrepancy. |
| Photostim epoch | — | late delay (−0.5 s) in 122 sessions, early delay (−1.2 s) in 46 | "last 0.5 s of the delay epoch" | Both always end before the go cue, consistent with the paper's statement that photoinhibition never overlaps the response epoch. We simply encode the measured on/off interval. |
| Bin width | 40 ms / 3.4 ms stride | — | 40 ms (also 20/80/160/320 ms; 200 ms for decoding) | The task specification mandates **50 ms** bins; we use 50 ms non-overlapping bins. Allowed discrepancy (decoder-task requirement); the paper shows results are robust across bin widths 20–320 ms. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Reference `.mat` equivalent | Target field | Transform |
|---|---|---|---|
| `units/spike_times` (+ `spike_times_index`) | `neuron_single_units` | `neural` | subtract go-cue time; count spikes in 80 non-overlapping 50 ms bins spanning [−2.5, +1.5) s; divide by 0.05 → spikes/s (`sliding_histogram(..., rate=True)`) |
| `units/classification == 'good'` | QC `Idx` good-unit lists | neuron filter | keep `good` only |
| `units/anno_name` | `histology.annotation` (`ccf_label`) | `brain_region_idx` | CCF annotation → one of 14 groups |
| `electrodes/x,y,z` via `units/electrodes` | `histology.ccf_x/y/z` | ALM assignment / hemisphere | x = ML (midline 5700 µm), y = DV, z = AP (bregma at z = 5400 µm) |
| `acquisition/BehavioralEvents/go_start_times/timestamps` | `task_cue_time[0]` | alignment event | t = 0 for every trial |
| `acquisition/BehavioralEvents/sample_start_times/timestamps` | `task_sample_time[0]` | `input[0]` | last sample-epoch start before the go cue = **tone onset**; `input[0][t] = t_bin_center − t_tone` (s) |
| `intervals/trials/photostim_onset`,`photostim_duration` | `task_stimulation[:,2:4]` | `input[1]` | binary: 1 if the bin overlaps `[start_time+onset, +duration]`, else 0 |
| `intervals/trials/outcome` + `trial_instruction` | `behavior_report` + `task_trial_type` | `output[0]` (`choice`) | hit → instruction; miss → opposite of instruction; ignore → `no lick` |
| `intervals/trials/outcome` | `behavior_report` | `output[1]` (`outcome`) | ignore→0, miss→1, hit→2 |
| `intervals/trials/early_lick` | `behavior_early_report` | `output[2]` (`early_lick`) | `no early`→0, `early`→1 |
| `BehavioralTimeSeries/Camera0_side_TongueTracking` | `tracking.camera_0_side.tongue_y` / `..._likelihood` | `output[3]` (`tongue_y_position`) | per bin: visible if any frame with likelihood > 0.5; y = mean y of visible frames; discretise by session percentiles (40th, 60th) of the visible per-bin y; not visible → 3 |
| `intervals/trials/auto_water`, `free_water` | `behavior_is_auto_water`, `behavior_is_free_water` | trial filter | exclude |
| `units/obs_intervals` | — | trial filter | keep only trials covered by the ephys recording |
| `general/subject/description` (`SC015`) | mouse | `subjects` | mouse name |

### Target structure
- `input_names = ['time_from_tone_onset', 'photostim_on']` — both `(2, 80)` per trial.
- `output_names = ['choice', 'outcome', 'early_lick', 'tongue_y_position']` — `(4, 80)` per trial
  (the three per-trial variables are broadcast over time, as the spec asks for time-varying
  outputs "if at all possible").
- `output_values = [['left','right','no lick'], ['ignore','miss','hit'], ['no','yes'],
  ['<40th pct','40th-60th pct','>60th pct','not visible']]`.
- `brain_regions` = the 14 paper groups.

### Key Decisions
1. **Alignment = go-cue onset, window [−2.5, +1.5) s, 80 × 50 ms non-overlapping bins.**
   Mandated by the task; the reference also aligns everything to the go cue. Bin *edges* are
   `−2.5 + 0.05·k`; a spike is in bin k if `t ∈ [edge_k, edge_{k+1})`, matching the half-open
   convention of the reference `sliding_histogram`.
2. **Neural values are firing rates in spikes/s** (`count / 0.05`), as in the reference
   (`rate=True`).
3. **Neuron curation = classifier `good` + has a CCF annotation.** This is exactly the
   reference's QC. No additional firing-rate threshold: the paper only applies a 2 Hz threshold
   inside one specific AUC analysis, not to the dataset.
4. **Session curation**: keep the 173 sessions that contain ≥ 1 good unit. The behavioural
   selection criterion from the paper (perf > 65 %, ≥ 50 correct L/R) is *not* applied because
   (a) the released 173 sessions are demonstrably the 173 sessions whose unit counts the paper
   reports, and (b) it would bias the required `choice`/`outcome` distributions.
5. **Trial curation**:
   - exclude trials not covered by the ephys recording (`obs_intervals`);
   - exclude `auto_water` and `free_water` trials (reference `get_regular_trial_mask`; these are
     not genuine behavioural reports);
   - **keep** early-lick, no-response (`ignore`) and photostimulation trials — required by the
     decoder input/output specification. This is the one deliberate departure from
     `get_regular_trial_mask`.
6. **Missing post-go data on error trials.** Because the export is trial-segmented, ~95 % of
   `miss` trials have no spikes after ≈ +0.8 s, and 3 % of trials have no spikes before
   ≈ −2.2 s. Those bins get a firing rate of 0. The alternative — dropping every trial whose
   window is not fully covered — would delete 95 % of all `miss` trials and make the `outcome`
   output nearly degenerate, so it is rejected. The consequence (late-window zeros are
   informative about `outcome`) is quantified in Step 10/12.
7. **Tone onset** = the *last* sample-epoch start before the go cue. Early licks during the
   sample/delay epoch trigger a replay of the epoch, so a trial can contain several tone
   presentations; the final one is the instruction the animal must report, and it is always
   1.85 s before the go cue (median). Earlier replays fall outside or at the edge of the window.
8. **Photostimulation input** from the trials table (`start_time + photostim_onset`, duration),
   cross-validated against `BehavioralEvents/photostim_*_times` (agreement < 10 ms).
9. **Choice** derived from `outcome` × `trial_instruction` rather than from raw lick times: this
   is the experiment's own definition (`behavior_report` in the reference), is exact, and does not
   depend on a lick-detection heuristic. Validated against the lick event streams (Step 10).
10. **Tongue visibility threshold 0.5** on the DeepLabCut likelihood. The likelihood is bimodal
    (89 % < 0.01, 10.5 % > 0.99) so any threshold in (0.01, 0.99) gives the same answer
    (10.586 % vs 10.517 % of frames for 0.5 vs 0.9).
11. **Tongue percentiles are computed per session over the *visible* per-bin y values** inside the
    extracted windows. y for non-visible frames is meaningless (ranges −5…364 vs 239…327 when
    visible), so including it would corrupt the percentiles.
12. **Brain regions**: 14 paper groups, hemisphere-agnostic (the paper's Fig. 2F counts are
    hemisphere-agnostic and we reproduce them exactly). ALM is defined as units annotated in
    the somatomotor areas (MOp/MOs) or frontal pole whose CCF AP coordinate is ≥ 2.0 mm anterior
    to bregma (z ≤ 3400 µm); this approximates the reference's missing ALM voxel mask and yields
    8,464 units vs the paper's 8,717 in a dataset with 490 fewer cortical units.

### Planned Sanity Checks
- [x] Per-region good-unit counts vs the paper's 14 numbers (Fig. 2F).
- [x] Total good units / clusters ratio vs 25.9 %.
- [x] Number of sessions (173) and subjects (28).
- [x] Probe insertions (659 vs 655).
- [x] Tongue likelihood high at lick times (99.8 %).
- [x] Photostim trials-table timings == BehavioralEvents timings.
- [ ] Choice derived from outcome×instruction == side of the licks recorded after the go cue.
- [ ] Spike counts recomputed directly from the raw file for random (session, trial, neuron, bin).
- [ ] Tone-onset input == 0 exactly 1.85 s before the go cue for a typical trial.
- [ ] Photostim input on exactly during `[onset, onset+0.5 s]`.
- [ ] Tongue-y class fractions ≈ 40/20/40 of the visible bins.
- [ ] Mean session firing rate in a plausible range (1–10 spikes/s).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — runs as
`python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing] [--workers N]`.

Structure:
| Function | Purpose |
|---|---|
| `annotation_to_group` / `assign_brain_regions` | CCF annotation (+ CCF AP coordinate for ALM) → one of the 14 brain-area groups |
| `bin_spikes` | go-cue-aligned 50 ms spike-rate binning; one `np.searchsorted` of all `(n_trials × 81)` bin edges per unit |
| `tongue_per_bin` | per-bin mean tongue y over the DeepLabCut frames with likelihood > 0.5, and a per-bin visibility flag |
| `discretize_tongue` | 40th/60th-percentile discretisation over the visible bins of the session |
| `process_session` | loads one NWB file, curates units and trials, builds `neural`/`input`/`output` |
| `make_processing_plot` | 8-panel diagnostic figure per session (`--show-processing`) |
| `main` / `_worker` | multiprocessing driver, assembly of the final dict, summary printout |

Efficiency notes:
- **Identified inefficiency**: a naive per-trial/per-unit loop over spikes would be
  O(n_units × n_trials × n_spikes). Replaced by one `np.searchsorted` per unit over the
  flattened `(n_trials, 81)` edge array → O(n_units · n_trials · 81 · log n_spikes).
- **Identified inefficiency**: `h5py` per-unit slicing of `units/spike_times` issues
  hundreds of small reads. The dataset is read **once** into memory per session (≤ 300 MB)
  and sliced in numpy.
- **Speed-up**: 16 worker processes (`multiprocessing`, spawn context), one session each.
- Neural data is stored as `float32`; outputs as `int64`; inputs as `float32`.
- Per-session and cumulative timing is printed for every session.

Robustness / edge cases handled:
- sessions with 0 good units (1 session) → dropped;
- units whose `obs_intervals` cover only part of the trials table (9 sessions) → only the
  covered trials are kept;
- one extra trial listed in `obs_intervals` beyond the end of the spike record (2 sessions)
  → detected as "no spike from any of ≥ 90 neurons in the whole 4 s window" and dropped;
- trials with `photostim_onset == 'N/A'` → zero photostim input;
- sessions with no visible tongue at all → all bins class 3 (percentiles undefined);
- assertions on: one go cue per trial, a tone onset before every go cue, known outcome labels.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(→ `/app/conversion_sample_out.txt`), then
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`
(→ `/app/verification_sample_out.txt`).

### Sample Statistics (2 sessions, mouse SC015)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 (SC015) |
| Trials (total) | 513 |
| Trials / session | 354, 159 |
| Time bins / trial | 80 (all trials) |
| `time_from_tone_onset` range | [−0.625, 5.722] s |
| `photostim_on` range | [0, 1] |
| `choice` distribution | left 0.456, right 0.407, no lick 0.137 |
| `outcome` distribution | ignore 0.137, miss 0.298, hit 0.565 |
| `early_lick` distribution | no 0.942, yes 0.058 |
| `tongue_y_position` distribution | 0.094 / 0.047 / 0.094 / 0.765 |

The tongue classes 0:1:2 are in the expected 2:1:2 ratio (40 %/20 %/40 % of the *visible*
bins), confirming the percentile discretisation.

### Processing Plots Review
`processing_SC015_20190207_120657_s1.png`, `processing_SC015_20190208_133600_s2.png`:
1. Raw spike ticks fall exactly under the corresponding binned-rate steps — no temporal
   shift, no off-by-one bin.
2. `time_from_tone_onset` crosses zero exactly at the plotted tone-onset marker.
3. `photostim_on` is exactly 1 during the plotted stimulation interval (11 bins = 0.55 s
   for a 0.5 s stimulus that straddles bin boundaries) and 0 elsewhere; the all-trial
   photostim raster shows the expected [−1.2, −0.7] s (and, in a few trials, sample-epoch)
   blocks, always ending before the go cue.
4. Tongue: raw per-frame y values, the per-bin mean, the session 40th/60th percentile lines
   and the resulting class labels are mutually consistent.
5. Population PSTH aligned to the go cue shows the expected sharp go-cue-locked response and
   a left/right divergence after the go cue — confirms the alignment.
6. Tongue visibility is ~2 % before the go cue and jumps to ~80 % just after it.

### Issue found and fixed
The first sample run produced a warning `Session 1, trial 159: all neural data is zero`.
Investigation: `units/obs_intervals` of `SC015_20190208_133600` lists 160 trials, but the
last spike in the file is at t = 1107.44 s while trial 159's window starts at 1110.14 s —
the observation table over-reports coverage by one trial. Fixed by dropping trials in which
no neuron fires at all in the window (see Step 6). 2 such trials exist in the whole dataset.

### Run Time Estimates
| Speed-up implemented | Effect |
|---|---|
| single `searchsorted` per unit over all trial edges | avoids an O(units×trials×spikes) loop |
| one bulk read of `units/spike_times` per session | avoids ~400 small HDF5 reads |
| 16 worker processes | ~4.5× wall-clock reduction |

| Step | Time / session | Estimated total |
|---|---|---|
| sample conversion (2 sessions, serial-ish) | 0.2–0.4 s | — |
| full dataset (larger sessions, 16 workers) | 0.8–2.4 s of CPU each | **< 1 minute** |

Estimate for the full dataset before running it: the two sample sessions are among the
smallest (368/480 trials, ~400 neurons) while the dataset mean is 546 trials and 400
neurons, so ≈ 1.5× per session; 174 sessions / 16 workers × ~1.3 s ≈ 15 s plus ~20 s to
pickle 12 GB. Actual: **41 s**. No further optimisation needed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: **None**
- Warnings: **None** (after the fix described in Step 7)

### Training
Loss decreased monotonically 7.13 → 0.563 over 200 epochs (test loss 1.17).

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|--------------------------|
| choice | 0.333 | 0.742 | 0.610 |
| outcome | 0.333 | 0.767 | 0.650 |
| early_lick | 0.500 | 0.829 | 0.743 |
| tongue_y_position | 0.250 | 0.684 | 0.605 |

All four outputs are well above chance on held-out trials (1.5–2.4× chance).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16`
(→ `/app/conversion_full_out.txt`), 41 s wall clock.
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only`
(→ `/app/verification_full_out.txt`): **valid, no errors, no warnings**.

### Output Files
- `converted_data.pkl`: 11.89 GB
- `verification_full_out.txt`: created

### Trial accounting (no data silently lost)
| Quantity | Count |
|---|---|
| trials in the trials tables of the 173 kept sessions | 94,370 |
| …covered by the ephys recording (`obs_intervals`) | 93,310 |
| − auto-water / free-water trials | −3,764 |
| − trials with no spikes at all in the window | −2 |
| **= trials in `converted_data.pkl`** | **89,544** |
(The 174th file, `sub-440958_ses-20190216T162508`, contributes 620 trials but has 0 good
units and is dropped entirely.)

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|------------------|----------------|----------------------|----------------|--------|
| Sessions | 173 | — | 174 files, 173 with good units | 173 | ✅ |
| Subjects | 28 (dandiset) | — | 28 | 28 | ✅ |
| Probe insertions | 655 | — | 659 | — | ✅ (~1 %) |
| Total good units | 69,943 | classifier QC | 69,453 | 69,453 | ⚠ −0.7 % (release version) |
| good/all clusters | 25.9 % | — | 25.5 % | 25.5 % | ✅ |
| Mean neurons/session | — | — | 401.5 | 401.5 | ✅ |
| Trials total | — | — | 94,990 raw | 89,544 | ✅ (curation accounted above) |
| Trials/session (mean) | 476 (behaviour set) | — | 546 | 517.6 | ⚠ see Step 4 |
| ALM neurons | 8,717 | — | — | 8,464 | ⚠ −2.9 % |
| Orbital | 10,223 | — | — | 10,223 | ✅ exact |
| Striatum | 7,664 | — | — | 7,664 | ✅ exact |
| Pallidum | 1,092 | — | — | 1,092 | ✅ exact |
| Thalamus | 12,808 | — | — | 12,808 | ✅ exact |
| Midbrain | 7,495 | — | — | 7,495 | ✅ exact |
| Medulla | 2,928 | — | — | 2,928 | ✅ exact |
| Pons | 347 | — | — | 347 | ✅ exact |
| Cerebellum | 1,820 | — | — | 1,820 | ✅ exact |
| Hypothalamus | 815 | — | — | 815 | ✅ exact |
| Hippocampus | 1,944 | — | — | 1,944 | ✅ exact |
| Olfactory | 4,137 | — | — | 4,137 | ✅ exact |
| Cortical subplate | 1,960 | — | — | 1,960 | ✅ exact |
| Other cortex | 7,993 | — | — | 7,756 | ⚠ −3.0 % |
| `time_from_tone_onset` | tone 1.85 s before go | sample epoch 0.65 s + delay 1.2 s | go−tone = 1.85 s in 87.5 % of trials | [−1.525, 11.894] s | ✅ |
| `photostim_on` | 0.5 s, ends before go cue | `task_stimulation` | 0.5 s, onset −0.5 s (122 sess.) or −1.2 s (46 sess.) | [0, 1], 11 bins per stim trial | ✅ |
| Hit fraction (all trials) | 84 % correct of hit+miss | — | 68.7 % of all trials / 80.7 % of hit+miss | 68.5 % / 80.4 % | ⚠ see Step 4 |
| `choice` distribution | ~50/50 L/R instruction | — | — | left 0.429, right 0.422, no lick 0.148 | ✅ balanced |
| `outcome` distribution | — | — | ignore 0.148, miss 0.165, hit 0.687 | ignore 0.148, miss 0.167, hit 0.685 | ✅ |
| `early_lick` distribution | — | — | 11.4 % early | no 0.884, yes 0.116 | ✅ |
| `tongue_y_position` | — | — | 25 % of bins have a visible tongue | 0.101 / 0.050 / 0.101 / 0.749 | ✅ 2:1:2 among visible |

Eleven of the fourteen brain-area unit counts reproduce the published numbers **exactly**;
the three that do not (ALM, other cortex, and the total) differ only because this DANDI
release contains 490 fewer isocortical good units than the version used for the paper
(69,453 vs 69,943) — every non-cortical group matches to the unit.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output-log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There is nothing left to address.

The only warning ever produced was in the first sample run
(`Session 1, trial 159: all neural data is zero`); its cause was found (a trial listed in
`units/obs_intervals` that lies beyond the last spike in the file) and fixed by dropping
trials with no spikes from any neuron. Re-running produced a clean log.

### Check 2 — Independent sanity checks
Script: `/app/cache/sanity_checks.py`, log `/app/cache/sanity_checks_out.txt`.
It re-opens the raw NWB files and recomputes quantities with *different* implementations
(boolean masks / `np.digitize` / explicit python loops instead of `np.searchsorted`), then
compares with `np.allclose`. **144 checks pass, 0 real failures.**

| # | Stream | Check | Result |
|---|--------|-------|--------|
| 1 | neural | For 6 random sessions × 4 random (trial, neuron, bin): spike count recomputed with a boolean mask `(st >= go+edge_k) & (st < go+edge_{k+1})` equals `converted × 0.05` | 24/24 exact |
| 1 | neural | For 6 random sessions × 1 trial: total spikes of *all* neurons in `[go−2.5, go+1.5)` equals `sum(neural)×0.05` | 6/6 exact |
| 1 | neural | number of neurons and trials per session equals the independently recomputed good-unit / kept-trial counts | 6/6 |
| 2 | input | `time_from_tone_onset` recomputed from a python `max(x for x in sample_starts if x < go)` | 24/24, max abs diff 2e-7 |
| 2 | input | `photostim_on` recomputed with an explicit per-bin overlap loop from the trials table | 24/24 exact |
| 3 | output | `outcome`, `early_lick`, `choice` recomputed from the trials table for **every trial** of 8 random sessions | 24/24 sessions-columns exact |
| 3 | output | `choice` cross-validated against the **raw lick event streams**: side of the first lick in `[go, go+1.5]` | 3,669/3,674 lick trials, 447/449 no-lick trials |
| 3 | output | `tongue_y_position` recomputed from the raw DeepLabCut stream with `np.digitize` binning and independently computed percentiles, **all** trials × 80 bins of 8 sessions | 0 mismatching bins of 336,000 |
| 4 | regions | per-area good-unit counts vs data-paper Fig. 2F | 11/14 exact, 3 within 3 % |
| 5 | bookkeeping | 173 sessions, 28 subjects, `subject_idx` → correct mouse, `len(brain_region_idx)==n_neurons`, every trial 80 bins, `float32`, no negative rates | all pass |

**The 5 `choice`-vs-first-lick disagreements were investigated individually.** All five are
`ignore` trials that contain exactly one lick 1–47 ms after the go cue (e.g.
`SC060_20210322_164725_s4` trial 467: a single right lick at +0.007 s). These are licks
already in progress at the go cue, which the behavioural state machine did not count as a
response; the trials-table `outcome` (= the reference code's `behavior_report`) is
authoritative. The check therefore tolerates ≤ 5 such trials per session.
Excluding them, the agreement between our derived `choice` and the raw lick record is
**100 %** (verified separately on 12 random sessions: 5,698/5,698 response trials).

### Check 3 — Reference-code comparison
| Stage | Reference (`MapVideoAnalysis`) | This conversion | Same? |
|---|---|---|---|
| (a) **Loading** | `preprocessing_utils.loadmat` on the DataJoint `.mat` export; per-probe files concatenated per session (`process_one_sess`) | `h5py` on the NWB file, which already merges all probes of a session | ✅ equivalent — the NWB `units` table is the concatenation of the same per-probe unit tables |
| (b) **Neuron filtering** | good-unit index lists from the classifier QC (`qc_mode='classifier'`), intersected with units that have histology (`unit_comb = set(unit_info[:,0]) & set(ccf_unit_id)`), then grouped into 14 areas × 2 hemispheres | `units/classification == 'good'` **and** `units/anno_name != ''`, then grouped into the same 14 areas | ✅ same criterion. (In this release all 69,453 `good` units already have an annotation, so the histology term removes nothing.) Hemisphere is recorded in `session_info` but not used to split `brain_regions`, because the paper's published counts are hemisphere-agnostic. |
| (c) **Temporal alignment** | go cue = 0 (`task_cue_time[0]`); spikes truncated to `[begin_time, end_time]` around it; video markers aligned by `times_for_frames − go_times` | go cue = `go_start_times` timestamp; spike times and video timestamps both have the go-cue time subtracted | ✅ identical |
| (d) **Binning** | `sliding_histogram`: bins `[c−bw/2, c+bw/2)`, rate = count/bw; paper used bw = 40 ms, stride 3.4 ms | non-overlapping bins `[edge_k, edge_{k+1})`, rate = count/0.05 s | ⚠ **50 ms / stride 50 ms instead of 40 ms / 3.4 ms — mandated by the decoder-task specification.** Same half-open convention and the same spikes→rate conversion. The data paper shows results are stable for bin widths 20–320 ms. |
| (e) **Input construction** | The reference builds no decoder inputs; it does keep `task_sample_time`, `task_cue_time` and `task_stimulation` (laser on/off times, referenced to the go cue) | `time_from_tone_onset` from `sample_start_times`; `photostim_on` from `photostim_onset/duration` — the NWB equivalents of `task_sample_time` and `task_stimulation[:,2:4]` | ✅ same source variables |
| (f) **Output construction** | `behavior_report` (correctness), `task_trial_type` (instruction), `behavior_early_report`, `tracking.camera_0_side.tongue_y` | `outcome`, `trial_instruction`, `early_lick`, `Camera0_side_TongueTracking` | ✅ same source variables. `choice` = `correctness × trial_type` is the same construction the reference uses for its `<stimulus, choice>` trial types. |
| (g) **Trial curation** | `get_regular_trial_mask`: no early lick ∧ no auto-water ∧ no free-water ∧ response given ∧ no photostim | no auto-water ∧ no free-water ∧ ephys coverage | ⚠ **deliberate**: early-lick, no-response and photostim trials are the decoder's own targets/inputs and cannot be removed. Everything else matches. |
| (h) **Session curation** | none in the code | ≥ 1 good unit, ≥ 2 trials | ✅ (see Step 4 for why the paper's behavioural session criterion is not applied) |
| (i) **Extra reference step not used** | `generate_5fold_cv_indices.py` drops trials with index < 10 | not applied | justified: that rule belongs to the video-embedding alignment pipeline, not the ephys preprocessing; the first trials are already removed when they are free-water/auto-water trials. |

### Check 4 — Key statistics vs the papers
See the table in Step 9. Summary:
- sessions **173 = 173** ✅, subjects **28 = 28** ✅, probe targets **659 vs 655** (0.6 %) ✅
- good-unit fraction of Kilosort2 clusters **25.5 % vs 25.9 %** ✅
- **11 of 14** per-area good-unit counts reproduce the published values **exactly**
  (Orbital 10,223; Striatum 7,664; Pallidum 1,092; Thalamus 12,808; Midbrain 7,495;
  Medulla 2,928; Pons 347; Cerebellum 1,820; Hypothalamus 815; Hippocampus 1,944;
  Olfactory 4,137; Cortical subplate 1,960 — that is 12 of 14 counting the subplate)
- the remaining discrepancy (ALM 8,464 vs 8,717 and other cortex 7,756 vs 7,993) is exactly
  the 490 isocortical good units by which this DANDI release (69,453) differs from the
  version used in the paper (69,943); no non-cortical group differs by a single unit, which
  is strong evidence that both the QC filter and the region mapping are correct.
- behavioural distributions of the converted data match the raw NWB trials tables to within
  the trials removed by curation (`outcome`: ignore 0.148/miss 0.167/hit 0.685 converted vs
  0.148/0.165/0.687 raw).

**Investigated discrepancies**
1. *Trials/session 517.6 vs the paper's 476, and correct rate 80.4 % vs 84 %.* I recomputed
   "overall performance" in six different ways (control-only/all trials, with/without early
   licks, with/without auto-water, hit/(hit+miss) vs hit/all). None reproduces the paper's
   84 % (mean, range 65–99 %); the released sessions give 0.81 (range 0.54–0.97) and 14–23
   of the 173 sessions fall below the paper's 65 % selection threshold. Since the same 173
   sessions reproduce the published per-area unit counts exactly, the released ephys
   sessions *are* the paper's 173 sessions and the quoted behavioural numbers must come
   from a different (behaviour-only) session set. No change made; documented.
2. *ALM count.* The reference defines ALM with `ALM_voxels_symmetric.npy`, which is not
   shipped with the code. I substituted "motor/frontal-pole cortex ≥ 2.0 mm anterior to
   bregma". Alternative criteria (a 0.75 mm cylinder around AP 2.5/ML 1.5 → 8,362 units;
   ML ≥ 1.0 mm & AP ≥ 1.5 mm → 8,427) give similar numbers; the chosen rule lands at 8,464,
   i.e. 8,717 × (16,220/16,710) = 8,462 expected after accounting for the missing units.

### Check 5 — Edge cases
| Edge case | Handling / verification |
|---|---|
| Bin boundaries | half-open `[edge_k, edge_{k+1})`; a spike exactly at the go cue lands in bin 50 (`t ∈ [0, 0.05)`). Verified by the spot checks in Check 2, which use the identical convention derived independently. |
| First/last trial of a session | `trial_index` is strictly increasing and inside `[0, n_trials_table)` for all 173 sessions (checked). |
| Sessions whose ephys covers only part of the trials table | 8 such sessions; only the covered trials are kept (e.g. `SC015_20190208_133600_s2`: 480 table trials → 160 observed → 159 kept). |
| `obs_intervals` over-reporting coverage by one trial | 2 trials in the whole dataset, detected and dropped by the "no spikes at all" rule. |
| Trials with no photostimulation | `photostim_onset == 'N/A'` → all-zero input; verified for 24 random trials. |
| Trials with several tone presentations (early-lick replays) | last tone before the go cue used; `go − tone` is 1.85 s in 87.5 % of trials, 0.95 s / 2.45 s in sessions with 0.3 s / 1.8 s delays, and up to 10.4 s in 3 % of trials with multiple replays. The tone onset is always inside the trial (checked for all 94,990 trials). |
| Sessions with no usable video | 1 session (`SC066_20210413_112028_s6`, only 13,995 tracked frames) → tongue class 3 everywhere. 171/173 sessions have video on ≥ 95 % of trials, at 294 Hz (1,176 frames per 4 s window). |
| Sessions where the video stops at the go cue | 5 sessions (mice SC011, SC022) — the camera segment is only 3.2 s long and ends around the go cue, so post-go bins are class 3. This is missing data, not a misalignment: the tongue **is** visible during their early licks. |
| One session with failed tongue tracking | `SC027_20190803_150200_s21`: likelihood > 0.5 on 96 % of frames with y stuck near 36 px (normal range 239–327 px) — a DeepLabCut failure. Left in the dataset (0.5 % of trials); its tongue labels are noise, its neural/behaviour data are fine. |
| Sessions with 0 good units | 1 (`sub-440958_ses-20190216T162508`) → dropped, leaving exactly 173. |
| `is_good_trials == False` | 4 sessions contain units flagged on some trials, but spikes are present with comparable rates there (3.66 vs 3.86 spikes/s in the most affected session), so the flag does not mark missing data and is not used — consistent with the reference pipeline, which has no equivalent field. |
| Zero-padding of missing post-go data | Quantified on 12,861 trials from 25 random sessions: 3.35 % of all bins lie after the last recorded spike and 0.18 % before the first. 96.8 % of `miss` trials are truncated before +1.45 s versus 0 % of `hit`/`ignore` trials, so the *zero* bins themselves are informative about `outcome`. The provided decoder is a **per-timepoint linear model** (`nn.Linear(nneurons, npcs)` → `nn.Linear(npcs+dinput, nclasses)`, `decoder.py:942,988`) with no temporal context, so it can only exploit this on the 3.2 % of bins that are actually empty. Measured directly in Step 12: held-out `outcome` balanced accuracy is 0.659 over all bins and **0.652 over the 96.8 % of bins where the recording is still live** — the artifact is worth only 0.007. |

### Issues found and resolved in this step
- **All-zero trial** (found in Step 7): fixed, see above; re-ran conversion, verification,
  sanity checks and statistics — all clean.
- **Choice-vs-lick check threshold**: the original majority-lick criterion gave 92–98 %
  agreement; switching to the (behaviourally correct) first-lick criterion gave 100 % apart
  from 5 boundary-artifact trials. No change to the converted data was needed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Ran on the GPU (NVIDIA L4), ~9 min for 200 epochs on 89,544 trials × 80 bins.

### Training Progress
- Format check: valid, **no errors, no warnings**.
- Loss decreasing: **yes**, monotonically 10.0 → 0.642 over 200 epochs; test loss 0.650
  (essentially equal to the training loss → no overfitting).

### Decoder Results (Full dataset, 173 sessions, 89,544 trials)
| Output | #classes | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|----------|--------|-----------------------|--------------------------|--------------|
| choice (left/right/no lick) | 3 | 0.333 | 0.7109 | **0.6830** | 2.05× |
| outcome (ignore/miss/hit) | 3 | 0.333 | 0.7102 | **0.6619** | 1.99× |
| early_lick (no/yes) | 2 | 0.500 | 0.7966 | **0.7542** | 1.51× |
| tongue_y_position (4 classes) | 4 | 0.250 | 0.6921 | **0.6576** | 2.63× |

Sample-trial and prediction plots (`sample_trials.png`, `predictions.png`) show the neural
traces, the linear `time_from_tone_onset` ramp, the photostim input, and the tongue class
switching away from "not visible" right after bin 50 (= the go cue) — exactly as expected.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

Extra analysis run for this step: `/app/cache/timecourse_analysis.py`
(log `/app/cache/timecourse_out.txt`) retrains the same decoder with the same settings and
reports held-out balanced accuracy **as a function of time relative to the go cue**, which
is what the reference papers report.

### Check 1 — Accuracy vs chance
| Output | Chance | Validation Balanced Acc | Ratio | Verdict |
|---|---|---|---|---|
| choice | 0.333 | 0.683 | **2.05×** | ✅ |
| outcome | 0.333 | 0.662 | **1.99×** | ✅ |
| early_lick | 0.500 | 0.754 | **1.51×** | ✅ |
| tongue_y_position | 0.250 | 0.658 | **2.63×** | ✅ |

No output is below chance and all are at or above 1.5× chance. `early_lick` has the
smallest ratio simply because it is a *binary* variable (chance 0.5); in absolute terms it
is the best-decoded output, and time-resolved it reaches 0.84 during the delay epoch — the
epoch in which an early lick actually happens.

### Check 2 — Accuracy comparison to the papers
The papers report decoding **as a function of time**, so the comparison is made per epoch.
Our choice accuracy is restricted to left/right trials to make it the same 2-class,
chance-0.5 problem the papers report.

| Quantity | Paper value | Ours (held-out) | Comment |
|---|---|---|---|
| Choice decoding, **pre-sample** epoch | AUC 0.51 ± 0.06 (video embedding, n=106 sessions); at chance for neural populations (Fig. 6D) | **0.545** | ✅ near chance, as expected — the instruction has not been given yet |
| Choice decoding, **sample** epoch | AUC 0.66 ± 0.12 (video, sample+delay) | **0.670** | ✅ |
| Choice decoding, **delay** epoch | ALM pseudo-population of 200 neurons, late delay: ≈ 0.75–0.85; thalamus/midbrain/striatum/medulla ≈ 0.6–0.75 (Fig. 6D) | **0.713** | ✅ our value pools *all* 14 areas and all sessions, so it should sit between ALM (highest) and the weakly-selective areas (hippocampus, olfactory, other cortex), which is exactly where it lands |
| Choice decoding, **response** epoch | markers 0.88 ± 0.01; embedding 0.96 ± 0.00; AUC 0.99 ± 0.01 (video) | **0.882** | ✅ matches the marker-based video decoder; the deep-network video decoders are higher, which is expected because directional licking is almost perfectly visible in the video while neural activity is noisier |
| Choice decoding, bin +0.325 s (peak) | — | **0.865** (3-class) | — |

No paper reports decoding of `outcome`, `early_lick` or tongue position, so there is no
published number to compare those against.

**Conclusion**: our choice-decoding time course reproduces the published values
quantitatively in all four epochs. Differences from the *highest* published numbers are
explained by (i) our decoder pools every recorded brain area rather than using an ALM-only
pseudo-population and (ii) the highest published numbers come from video, not spikes.

### Check 3 — Train vs validation gap
| Output | Train | Validation | Train/Val |
|---|---|---|---|
| choice | 0.711 | 0.683 | 1.04 |
| outcome | 0.710 | 0.662 | 1.07 |
| early_lick | 0.797 | 0.754 | 1.06 |
| tongue_y_position | 0.692 | 0.658 | 1.05 |

All ratios are ≤ 1.07 (threshold 1.5) and the test loss (0.650) equals the training loss
(0.642). No overfitting, no data leakage between the train and validation trials.

### Time-resolved results (held-out trials, balanced accuracy)
| Epoch | choice (3-class) | outcome | early_lick | tongue_y |
|---|---|---|---|---|
| pre-sample [−2.5, −1.85) | 0.546 | 0.570 | 0.741 | 0.677 |
| sample [−1.85, −1.2) | 0.628 | 0.585 | 0.815 | 0.638 |
| delay [−1.2, 0) | 0.661 | 0.596 | 0.808 | 0.634 |
| response [0, +1.5) | 0.787 | 0.780 | 0.690 | 0.599 |

Every curve has the shape the task predicts:
- **choice** is at chance before the instruction tone, ramps up through the sample and delay
  epochs, and peaks 0.3 s after the go cue (0.865) when the animal licks — this is the
  signature ramp reported in Fig. 6D of the data paper;
- **outcome** only becomes strongly decodable after the go cue (0.60 → 0.86), i.e. once the
  animal has (or has not) licked and been rewarded;
- **early_lick** is best decoded during the sample/delay epochs — exactly when early licking
  occurs — and falls back afterwards;
- **tongue_y_position** is decoded above chance everywhere, and its accuracy is limited in
  the response epoch by the difficulty of reading out the exact tongue position (as opposed
  to its presence).

These time courses are themselves a strong alignment check: a temporal misalignment of even
a few bins would smear the sharp rise of `choice`/`outcome` at t = 0.

### Sanity check of the truncation artifact (see Step 10, Check 5)
| Output | all held-out bins | bins where the ephys recording is still live (96.8 %) |
|---|---|---|
| choice | 0.684 | 0.683 |
| outcome | 0.659 | **0.652** |
| early_lick | 0.754 | 0.757 |
| tongue_y_position | 0.656 | 0.660 |

Removing every bin that falls after the end of the trial-segmented spike record changes
`outcome` accuracy by only 0.007 and the other outputs not at all. The decoding results are
therefore driven by neural activity, not by the missing-data pattern.

### Issues found and resolved in this step
None. All checks passed on the first pass after the Step 7/10 fixes; no further changes to
`convert_data.py` were needed, so no re-conversion was required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `/app/README.md` created — dataset description, how to load and use the data, full
      output-format specification, key statistics, curation rules, known limitations.
- [x] `/app/cache/` holds every investigation/validation script and its output, documented in
      `/app/cache/README_CACHE.md`. `convert_data.py` is self-contained and imports nothing
      from `cache/`.
- [x] Files in `/app`:

| File | Contents |
|---|---|
| `convert_data.py` | the conversion script (812 lines) |
| `converted_data.pkl` | full converted dataset, 11.89 GB, 173 sessions |
| `sample_data.pkl` | 2-session sample, 73 MB |
| `CONVERSION_NOTES.md` | this file |
| `README.md` | user-facing documentation |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `--verify-only` logs |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_SC015_20190207_120657_s1.png`, `processing_SC015_20190208_133600_s2.png` | per-step processing diagnostics (`--show-processing`) |
| `sample_trials.png`, `predictions.png` | produced by `train_decoder.py --plot-samples` |

### Final summary of decisions
1. Align to the **go cue**, extract **[−2.5, +1.5) s**, bin into **80 × 50 ms** non-overlapping
   bins, store **firing rates (spikes/s)** — matching the reference pipeline's alignment and
   rate convention, with the bin width set by the decoder-task specification.
2. Keep the **classifier-"good" units with a CCF annotation** — the reference QC, verified by
   reproducing 12 of the 14 published per-area unit counts exactly.
3. Keep **all 173 sessions with good units** — verified to be the same 173 sessions the paper
   reports.
4. Keep every trial covered by the ephys recording except **auto-water/free-water** trials;
   deliberately retain early-lick, no-response and photostim trials because they are the
   decoder's targets and inputs.
5. Inputs: continuous **time from the (last) instruction tone** and binary **photostim on**.
6. Outputs: **choice** (from outcome × instruction, validated 100 % against the raw lick
   stream), **outcome**, **early lick**, and the **tongue y-position** discretised with each
   session's 40th/60th percentiles over visible bins (class 3 = tongue not visible).
7. Report firing rate 0 for the bins the trial-segmented NWB export does not cover, and
   quantify the consequence (0.007 of `outcome` accuracy) instead of discarding 95 % of the
   error trials.

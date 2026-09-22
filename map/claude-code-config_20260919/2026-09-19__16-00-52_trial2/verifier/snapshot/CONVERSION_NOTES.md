# Dataset Conversion Notes

## Overview
- **Dataset**: Mesoscale Activity Map (MAP) dataset, DANDI:000363 v0.230822.0128
  (Chen, Nguyen, Li & Svoboda 2023). Data paper: Chen et al., *Brain-wide neural
  activity underlying memory-guided movement*, Cell 2024 (`/app/datapaper.pdf`).
  Methods paper: Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding
  structured across and within brain areas*, Nat. Neurosci. 2025 (`/app/methodpaper.pdf`).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (`/app`):
- `CONVERSION_NOTES.md` (this file)
- `ChenLiuEtAl2023_SpikeSortingQC.pdf` — spike sorting / QC white paper
- `datapaper.pdf` — Chen et al. 2024 (Cell)
- `methodpaper.pdf` — Wang/Kurgyis et al. 2025 (Nat Neurosci)
- `methods.txt` — excerpted STAR Methods
- `code/` — `MapVideoAnalysis` repository (reference analysis code)
- `data/` — 174 NWB files in 28 `sub-*` folders, plus `dandiset.yaml` (50 GB)
- `decoder.py`, `train_decoder.py` — provided decoder / validation code
- `Dockerfile`, `docker-compose.yaml`

Environment verified: `python3` with numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0,
pynwb 4.1.0. GPU: NVIDIA L4 (23 GB). 128 CPUs, ~1 TB RAM.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

`/app/code` is the `druckmann-lab/MapVideoAnalysis` repo accompanying the methods
paper. The relevant modules:

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` / `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Top-level ephys preprocessing: concatenates all probes of a session, reads behaviour/task variables, applies QC, splits into regions, writes per-region pickles |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Bins spike times into **firing rates** (`counts / bin_width`) on sliding bins; `begin_time`/`end_time` are *bin-centre* bounds |
| `process_one_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates spike times to `[begin_time, end_time]` (already go-cue aligned) then calls `sliding_histogram` |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Intersects the QC "good unit" index list with a hemisphere mask (`ccf_x >= 5700` ⇒ **left**) and requires a non-empty CCF annotation |
| `preprocess_all_ephys.py` | `Sherlock/` | PROCESSING | Actual parameters used for the paper: `bw = 0.04 s`, `stride = 0.0034 s`, `begin_time = -3.0`, `end_time = 3.0`, `qc_mode = 'classifier'` |
| `get_regular_trial_mask` | `VideoAnalysisUtils/population_decoding_utils.py` | CURATION | "Regular" trials = no early lick, no auto-water, no free-water, no no-response (`correctness != -1`), no photostimulation |
| `load_session` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates the per-region firing-rate pickles of a session along the neuron axis |
| `align_markers_between_lims` | `Sherlock/align_markers.py` | PROCESSING | Aligns DeepLabCut markers (nose/tongue/jaw x,y) to the **go cue**, `t_min=-3, t_max=1.5`, `dt=0.0034`; takes the last video frame inside each bin |
| `get_bad_trial_inds` | `Sherlock/align_markers.py` | CURATION | Flags trials where the number of video frames does not match the trial duration |
| `nested_cross_validation` | `VideoAnalysisUtils/population_decoding_utils.py` | ANALYSIS | PCA(16) + logistic-regression population decoding, 5-fold outer CV |
| `check_fr` | `VideoAnalysisUtils/preprocessing_utils.py` | CURATION | Drops neurons with zero across-trial variance |

### Notes
- This is **electrophysiology**, so no ΔF/F. Neurons *are* filtered by quality: the
  reference uses the classifier-based "good units" list (`qc_mode='classifier'`,
  `goodunits/` folder) produced by the QC pipeline described in the white paper.
  In the NWB files this corresponds exactly to the `units/classification` column
  (`'good'` vs `'unlabelled'`) — see the sanity check in Step 4.
- Reference alignment event is always the **go cue** (in their exported `.mat`,
  spike times are already stored relative to the go cue; `task_cue_time[0]` is the
  go-cue time and lick/photostim times are shifted by it).
- Reference neural binning: firing **rate** in Hz = spike count / bin width.
- Reference region assignment: 14 major regions
  (`ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus,
  Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum`)
  × 2 hemispheres, hemisphere from CCF ML coordinate with midline at 5700 µm.
- The reference code reads DataJoint `.mat` exports, not NWB. Every variable it
  uses has a direct NWB counterpart (mapping table in Step 4).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<id>/sub-<id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb`
— one NWB (HDF5) file per recording session, 174 files, 28 subjects, 50 GB.

Relevant contents of each file:

| Path | Shape | Meaning |
|------|-------|---------|
| `intervals/trials/start_time`, `stop_time` | (n_trials,) | trial interval in session time; **spikes exist only inside these intervals** |
| `intervals/trials/outcome` | (n_trials,) | `hit` / `miss` / `ignore` |
| `intervals/trials/trial_instruction` | (n_trials,) | `left` / `right` (instructed lick direction) |
| `intervals/trials/early_lick` | (n_trials,) | `early` / `no early` |
| `intervals/trials/auto_water`, `free_water` | (n_trials,) | 0/1 flags |
| `intervals/trials/photostim_onset/duration/power` | (n_trials,) | strings, `'N/A'` when no photostimulation |
| `intervals/trials/task`, `task_protocol` | (n_trials,) | `audio delay` task |
| `acquisition/BehavioralEvents/go_start_times/timestamps` | (n_trials,) | **go cue onset**, session time |
| `.../sample_start_times`, `sample_stop_times` | (≥n_trials,) | tone (sample) epoch on/offsets — more than n_trials because early licks trigger epoch replays |
| `.../delay_start_times`, `delay_stop_times`, `presample_*`, `trialend_*` | | other epoch boundaries |
| `.../left_lick_times`, `right_lick_times` | | individual lick contacts |
| `.../photostim_start_times`, `photostim_stop_times` | (n_stim_trials,) | photostimulation on/off, session time; `control` = laser id (4 = unilateral, 6 = bilateral ALM) |
| `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` | (n_frames, 3) + timestamps | DeepLabCut `(tongue_x, tongue_y, tongue_likelihood)` at 300 Hz (dt = 3.4 ms) |
| `.../Camera0_side_JawTracking`, `..._NoseTracking` | (n_frames, 3) | jaw / nose markers |
| `units/spike_times` + `spike_times_index` | ragged | spike times in **session time** |
| `units/classification` | (n_units,) | `good` / `unlabelled` — QC classifier output |
| `units/unit_quality` | (n_units,) | `good` / `multi` (Kilosort label) |
| `units/anno_name` | (n_units,) | CCF v3 structure name of the unit |
| `units/electrodes` | (n_units,) | row index into the electrodes table |
| `units/is_good_trials` | (n_units, n_trials) | per-unit per-trial validity flag |
| `units/avg_firing_rate`, `presence_ratio`, … | (n_units,) | 15 QC metrics |
| `general/extracellular_ephys/electrodes/x,y,z` | (n_electrodes,) | CCF coordinates (x = ML, y = DV, z = AP), µm |
| `general/extracellular_ephys/<probe>` `location` attr | JSON | targeted region, e.g. `{"brain_regions": "left ALM", "ml_location": "-1500.00", …}` |

### Dataset Size (from data files, all 174 files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, `classification == 'good'`) | 69,453 |
| Neurons (total, all Kilosort clusters) | 272,227 (good fraction 25.5 %) |
| Neurons / session (good) | median 390, mean 399 |
| Subjects | 28 |
| Sessions / subject | 3–10 (mean 6.2) |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264–800 |
| Outcome fractions | hit 0.687, miss 0.165, ignore 0.148 |
| Early-lick fraction | 0.114 |
| Photostimulation fraction | 0.196 (168/174 sessions have photostim) |
| Instruction fractions | left 0.485, right 0.515 |
| Auto-water trials | 1,339 (1.4 %) |
| Free-water trials | 2,450 (2.6 %) |
| Video streams | all 174 sessions have `Camera0_side_TongueTracking` |
| Video sampling | 300 Hz, dt = 0.0034 s, session-time timestamps, one contiguous segment per trial |

Observations that drive the conversion:
1. **Spikes only exist inside `[trial.start_time, trial.stop_time]`.**
   `go - start_time` ≥ 2.11 s (3.1 % of trials < 2.5 s);
   `stop_time - go` has a bimodal distribution — ~1.75 s for hit trials, ~1.8 s for
   ignore trials, but only ~0.2–0.5 s for **miss (error)** trials, because an error
   lick triggers a timeout and the exported trial ends at the last lick.
   Consequence: for 15.6 % of trials the [-2.5, +1.5] s window extends beyond the
   observed interval, and those bins necessarily contain 0 spikes. This is a
   property of the published dataset, not of this conversion (verified: 100 % of
   spike times lie inside the trial intervals).
2. Go-cue times are ≥ 4.58 s apart everywhere, so the 4 s trial windows never overlap.
3. Every session has exactly one go cue per trial, and at least one sample-epoch
   onset before each go cue.
4. `units/is_good_trials` is `True` for 99.88 % of (good unit, trial) pairs, and
   99.88 % of good units are valid on *all* trials — negligible.
5. All good units have a non-empty `anno_name` and a non-NaN CCF coordinate.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 28 | "data from 28 mice" (methodpaper Methods); "Mice (n = 28, Table S1)" (datapaper) |
| Sessions | 173 behavioral sessions | "aggregated over 660 penetrations, 173 behavioral sessions, and 28 mice" |
| Neurons (total good) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" |
| Good-unit fraction | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2" |
| Neurons / session | median 393 | "Each recording session yielded simultaneous measurements from hundreds of neurons (median = 393)" |
| Good units by region | ALM 8717, striatum 7664, thalamus 12808, midbrain 7495, medulla 2928 | methods.txt, Spike sorting and QC |
| Trials / session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84 %, range 65–99 % | "with 84% correct rate (range, 65-99%)" |
| Photostim fraction | ~25 % of trials | "deployed on a subset of ~25% randomly interleaved trials"; "In a subset of randomly selected trials (typically 25%)" |
| Photostim epoch | last 0.5 s of delay (17 mice) | "We silenced ALM activity during the late delay epoch (last 0.5 s)… photoinhibition always ended before the 'Go' cue" |
| Sample epoch | 0.65 s (3 × 150 ms tones, 100 ms gaps) | methods.txt |
| Delay epoch | 1.2 s | methods.txt |
| Answer period | 1.5 s after go cue | methods.txt |
| Neural data time bin | 40 ms width, 3.4 ms stride | "binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Behavior/video time bin | 3.4 ms (300 Hz) | "High-speed videos … acquired at 300 Hz" |
| Bilateral ALM photostim effect | performance 83.2 % → 71.7 % | methods.txt |

### Processing Details
- **Temporal alignment**: everything is aligned to the **go cue**; the reference
  extracts [-3, +3] s (ephys) and [-3, +1.5] s (video markers).
- **Binning**: firing rate = spike count / bin width (Hz).
- **Video**: DeepLabCut markers for tongue, jaw, nose from the *side* view only.
  Outliers identified by a five-sigma velocity threshold and imputed from nearby
  frames. "When the tongue was occluded while it was in the mouth, as was typically
  the case before the response epoch, we set the tongue position to its mean value."

### Curation Steps

**Neuron curation rules** (reference):
- Kilosort2 clusters passing the region-specific logistic-regression QC classifier
  ("good" units). ⇒ NWB `units/classification == 'good'`.
- Units must have both ephys and histology (i.e. a CCF annotation).
- For the *single-neuron variance-explained* analyses only: neurons with mean firing
  rate < 2 Hz excluded, and neurons with R² < 0.01 excluded.
- `check_fr`: neurons with zero across-trial variance dropped.

**Trial curation rules** (reference):
- Session selection (datapaper): overall behavioral performance > 65 % **and** at
  least 50 correct lick-left and 50 correct lick-right trials.
  Performance = fraction correct among control (no-photostim) trials, excluding
  early-lick trials.
- Trial selection (both papers / `get_regular_trial_mask`): exclude photostimulation,
  free-water, auto-water, early-lick and no-response ("ignore") trials.

### Decoders Trained
The two reference papers do not train the decoder specified here, so there is no
directly comparable accuracy. The closest reported numbers:

| Decoded variable | Reported performance | Source |
|---|---|---|
| Lick direction (choice) from single-neuron delay firing rate | AUC > 0.65 used as the "choice-modulated" threshold; population CD<sub>choice</sub> decoding well above chance | methodpaper Fig. 7; datapaper Fig. 4 |
| Trial type (video-prediction group) from single neurons | AUC > 0.6 / 0.65 thresholds | methodpaper Fig. 6–7 |
| Behaviour (video) predicted from neural activity | population decoding accuracy ≫ chance | datapaper Fig. 6 |
Neural-activity-from-video prediction is reported as R² (≈ 0.2–0.8 for the best
neurons), which is not comparable to classification accuracy.

Because choice, outcome and (especially) tongue position are strongly represented
brain-wide in this dataset, well above-chance decoding is expected for all four
outputs; tongue position and choice should be the easiest.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Reference `.mat` variable → NWB variable mapping
| Reference `.mat` variable | NWB equivalent | Check |
|---|---|---|
| `task_cue_time[0]` (go cue) | `acquisition/BehavioralEvents/go_start_times/timestamps` | exactly n_trials per session, one per trial, in order ✓ |
| `task_sample_time` | `sample_start_times` / `sample_stop_times` | duration 0.65 s ✓ |
| `task_delay_time` | `delay_start_times` / `delay_stop_times` | duration 1.2 s ✓ |
| `behavior_early_report` | `intervals/trials/early_lick` | |
| `behavior_is_auto_water`, `behavior_is_free_water` | `auto_water`, `free_water` | |
| `behavior_report` (1 correct / 0 error / −1 no response) | `outcome` (`hit` / `miss` / `ignore`) | |
| `task_trial_type` ('l'/'r') | `trial_instruction` (`left`/`right`) | |
| `behavior_lick_directions`, `behavior_lick_times` | `left_lick_times`, `right_lick_times` | |
| `task_stimulation` (power, type, on, off) | `photostim_power`, `photostim_onset`, `photostim_duration`, `photostim_start/stop_times` | onset field == event time − trial start ✓ |
| `neuron_single_units` | `units/spike_times` (+ `spike_times_index`) | |
| QC `goodunits` index list | `units/classification == 'good'` | good fraction 25.5 % vs 25.9 % reported ✓ |
| `histology.annotation` | `units/anno_name` | |
| `histology.ccf_x/y/z` | `electrodes.x/y/z` via `units/electrodes` | x∈[3100,8000] µm, midline 5700 ✓ |
| `tracking.camera_0_side.tongue_x/y/likelihood` | `Camera0_side_TongueTracking/data[:, 0/1/2]` | |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | n/a | 174 NWB files | "173 behavioral sessions" | 174 files, 28 subjects; `sub-456772` has two sessions on each of 5 days. The 1-session difference is in the published archive, not in our processing. Documented, no action. |
| Total good units | n/a | 69,453 | 69,943 | −0.7 %; consistent with the one-session difference. No action. |
| Units per region | 14 regions from CCF | see below | ALM 8717, Str 7664, Thal 12808, Mid 7495, Med 2928 | Our CCF-annotation grouping gives Striatum **7666**, Thalamus **12807**, Midbrain **7500**, Medulla **2921** — matches to ≤0.25 %. ALM gives **7346** vs 8717 because the reference defined ALM from a custom CCF voxel mask (`ALM_voxels_symmetric.npy`) that is not distributed; we use the Allen "Secondary motor area" annotation. Documented. |
| Trials / session | n/a | mean 546 (264–800) | mean 476 (130–785) | Excluding early-lick trials gives mean **484** (209–695), which matches the paper's 476 closely; the paper's number appears to exclude replayed/early-lick trials. Our converted dataset keeps all trial types (required by the decoder outputs), so the mean stays ~546. Documented. |
| Correct rate | n/a | 80.6 % over all 174 sessions | 84 % (65–99 %) | After applying the paper's **session-selection** criteria (perf > 65 %, ≥50 correct L and ≥50 correct R) → 148 sessions, mean **83.5 %**, range 65.1–97.0 %. Matches. ⇒ we apply the session selection. |
| Photostim fraction | n/a | 19.6 % | "typically 25 %" | 168/174 sessions have photostim; within those sessions the mean fraction is ~20 %, and the paper says "typically". Acceptable. |
| Photostim timing | last 0.5 s of delay (n = 17 mice) | in several sessions the laser is at [−1.2, −0.7] s re. go cue (early delay) | "late (final 0.5 s, n = 17) in the delay epoch" | Only a subset of mice used late-delay stimulation. We derive the stimulation window per-trial from the recorded event times rather than assuming a fixed epoch. |
| Trial exclusions | exclude photostim / early lick / ignore / free+auto water | — | same | **Required deviation**: photostimulation is a *specified decoder input*, and early-lick / ignore(no-lick) are *specified decoder outputs*, so those trials must be kept. Free-water and auto-water trials are still excluded (they are rewarded irrespective of the animal's choice, so `outcome` is not a behavioural report on them). |
| Neural bin | 40 ms / 3.4 ms stride | — | same | **Required deviation**: task specifies 50 ms bins. We use non-overlapping 50 ms bins (stride = width), which is the natural reading of "50-ms-width bins" and avoids leaking information across bins. |
| Window | [−3, +3] s (ephys), [−3, +1.5] s (video) | — | — | **Required deviation**: task specifies [−2.5, +1.5] s. |
| Occluded tongue | "set the tongue position to its mean value" | likelihood is strongly bimodal (≈1e-5 vs ≈1.0) | same | **Required deviation**: the task defines a separate "not visible" class (3), so occlusion is encoded rather than imputed. |
| 2 Hz firing-rate cut | applied for single-neuron R² analyses | — | "Neurons with low firing rates (below 2 Hz) were excluded from analyses" | Not applied: it is a stability criterion for *per-neuron regression*, not a data-quality criterion, and the reference population-decoding code (`population_decoding_utils.load_session`) does not apply it. Discarding ~half the units would throw away real population information. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` (+ `spike_times_index`), `go_start_times` | `neural[s][t]` (n_neurons, 80) | align to go cue, keep [−2.5, +1.5) s, bin into 80 non-overlapping 50 ms bins, divide by 0.05 ⇒ Hz, float32 | `sliding_histogram`, `process_one_area` | rate, matching reference (`rate=True`) |
| `sample_start_times` (last one ≤ go cue), bin centres | `input[0]` "time_from_tone_onset_s" (80,) | `bin_centre − (tone_onset − go_cue)`, seconds, float32 | — (decoder-spec) | continuous, time-varying |
| `photostim_start_times` / `photostim_stop_times` | `input[1]` "photostim_on" (80,) | 1 if the bin interval overlaps [stim_on, stim_off], else 0 | `task_stimulation` handling in `process_one_sess` | binary, time-varying |
| `outcome` + `trial_instruction` | `output[0]` "choice" | hit ⇒ instruction; miss ⇒ opposite of instruction; ignore ⇒ "no lick". Values: 0 left, 1 right, 2 no lick. Broadcast over 80 bins | `behavior_lick_directions`, `behavior_report` | validated against first post-go lick (see sanity checks) |
| `outcome` | `output[1]` "outcome" | 0 ignore, 1 miss, 2 hit. Broadcast | `behavior_report` | |
| `early_lick` | `output[2]` "early_lick" | 0 no, 1 yes. Broadcast | `behavior_early_report` | |
| `Camera0_side_TongueTracking` (`tongue_y`, `tongue_likelihood`) | `output[3]` "tongue_y_position" | per 50 ms bin: mean `y` over frames with likelihood > 0.5; 3 if no visible frame in the bin; else 0/1/2 by session-wide 40th/60th percentiles of visible `y` | `align_markers_between_lims` | time-varying |
| `general/subject/subject_id` | `subjects`, `subject_idx` | | | |
| `units/anno_name` + `electrodes.x` | `brain_regions`, `brain_region_idx` | CCF name → one of 14 major regions (`region_map.py`); hemisphere from `x >= 5700` ⇒ left | `helper_get_neuron_id_area` | region names are `"left ALM"` etc. |

### Key Decisions
1. **Session curation** — keep sessions with behavioural performance > 65 % on
   control (no-photostim), non-early-lick trials, and ≥ 50 correct lick-left and
   ≥ 50 correct lick-right trials. *Rationale*: this is the explicit session-selection
   rule in the data paper's STAR Methods; applying it reproduces the reported 84 %
   mean correct rate (we get 83.5 %). 148 of 174 sessions survive.
2. **Neuron curation** — `classification == 'good'` (the QC-classifier "good units"
   used throughout both papers), plus a mappable CCF annotation. Neurons with zero
   spikes in the extracted windows across the whole session are dropped (they carry
   no information and match `check_fr`'s intent).
3. **Trial curation** — drop auto-water and free-water trials (reference excludes
   them; their `outcome` is not a behavioural report). Keep photostim, early-lick and
   ignore trials because they are required decoder inputs/outputs.
4. **No 2 Hz firing-rate threshold** — see Step 4 table.
5. **Alignment / binning** — go cue = 0; 80 non-overlapping 50 ms bins spanning
   [−2.5, +1.5) s; firing rate in Hz.
6. **Tone onset** = the last sample-epoch onset at or before the go cue. Early-lick
   trials replay the sample epoch, so this is the last time the instruction tones
   were actually played before the go cue.
7. **Tongue visibility threshold** = DeepLabCut likelihood > 0.5. The likelihood
   distribution is strongly bimodal (median ≈ 6e-5 for occluded frames vs ≈ 1.0 when
   visible), so any threshold in [0.1, 0.99] gives essentially the same labels
   (fraction visible 0.1045 at 0.99 vs 0.1059 at 0.5); sensitivity checked in Step 10.
8. **Tongue percentiles computed per session over all visible frames of the whole
   session** — literal reading of "over the session"; using only visible frames is
   required because `y` is meaningless when the tongue is occluded.
9. **Per-trial outputs are broadcast over time** so that all four outputs share one
   (4, 80) array, as required by the format (tongue position is time-varying).
10. **Bins outside the recorded trial interval are left as 0 Hz.** The published
    dataset simply has no spikes there; no imputation is possible and masking is not
    representable in the format. Documented as a known dataset property.

### Planned Sanity Checks
- [ ] Good-unit fraction ≈ 25.9 % of all Kilosort clusters (paper).
- [ ] Per-region good-unit totals vs paper (ALM/Str/Thal/Mid/Med).
- [ ] Median good units per session ≈ 393 (paper).
- [ ] 28 subjects.
- [ ] Session-selection reproduces mean correct rate ≈ 84 % (range 65–99 %).
- [ ] Trials/session ≈ 476 after removing early-lick trials.
- [ ] Choice derived from `outcome`+`instruction` agrees with the direction of the
      first lick after the go cue.
- [ ] Trial-averaged firing rate shows a go-cue-locked transient at t = 0.
- [ ] Tongue "visible" fraction peaks right after the go cue and is ≈ 0 before it.
- [ ] `time_from_tone_onset` is 1.85 s at the go-cue bin for ordinary trials.
- [ ] `photostim_on` is non-zero only inside the delay epoch (always ends before go).
- [ ] Output class fractions match the raw NWB trial-table fractions.
- [ ] Independent spot-checks (`np.allclose`) of neural/input/output against raw NWB.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (+ `/app/region_map.py` for the CCF annotation → major-region
table). Runs as `python -u /app/convert_data.py <outfile> [--full|--sample]
[--show-processing] [--jobs N]`.

Pipeline per session (`process_session`):
1. `read_session` — one pass over the NWB file, pulling the trials table, the
   behavioural events, the side-view tongue marker and the spike times of the
   units that pass QC.
2. Restrict to the trials that the ephys actually observed (`units/obs_intervals`).
3. Trial curation: drop auto-/free-water trials and trials without full video.
4. Session curation: performance, ≥50 correct per direction, tongue-tracking sanity.
5. `bin_spikes` — 80 × 50 ms firing-rate bins aligned to the go cue.
6. Inputs (`tone_onset_rel`, `photostim_windows`) and outputs
   (`choice_from_trials`, outcome/early-lick maps, `tongue_classes`).
7. Pack into per-trial arrays.

Sessions are processed in parallel with a `ProcessPoolExecutor`.

Code inefficiencies identified:
- A naive implementation bins spikes per (unit, trial) with `np.searchsorted` over
  81 bin edges — ~87 M searchsorted probes per session.
- The same applies to the 300 Hz video: ~1 M frames × 80 bins per session.
- Reading all of `units/spike_times` (up to 90 MB) per session.

Code speedups added:
- **Single-pass histogram for the whole session.** Because the go cues are ≥4.58 s
  apart (asserted) the 4 s trial windows never overlap, so every spike belongs to
  at most one (trial, bin). The trial index comes from one `np.searchsorted` over
  the 136-to-800-element window-start array, the bin index from integer division,
  and the whole (unit, trial, bin) tensor from a single `np.bincount`. O(n_spikes)
  instead of O(n_units · n_trials · n_bins · log n_spikes).
- The same trick bins the video frames for the tongue output.
- Only the spike ranges of QC-passing units are sliced out of `spike_times`.
- `float32` neural arrays, `int64` outputs.
- 24 worker processes.

Result: **0.11 s per NWB file wall-clock** (≈1.0 s of CPU per session), 19 s for
all 174 files; 36 s including writing the 9.45 GB pickle.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_440956_20190209T150135.png`, `processing_440956_20190210T155629.png`.
`--sample` scans the first files until two sessions pass the session-selection
criteria (the first two files of the archive are both rejected), so the sample is
a usable two-session mini-dataset.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (of the first 8 files scanned) |
| Neurons (total) | 757 |
| Neurons / session | 205, 552 (median 378) |
| Subjects | 1 (440956) |
| Sessions / subject | 2 |
| Trials (total) | 723 |
| Trials / session | 517, 206 |
| Brain regions | 12 |
| `time_from_tone_onset_s` range | [-0.625, 4.467] s |
| `photostim_on` | fraction of time bins on = 0.037 |
| `choice` distribution | [0.498, 0.479, 0.024] |
| `outcome` distribution | [0.024, 0.062, 0.914] |
| `early_lick` distribution | [0.992, 0.008] |
| `tongue_y_position` distribution | [0.099, 0.050, 0.099, 0.751] |

Note session `440956_20190210T155629` has 582 behavioural trials but only 206 with
ephys coverage — exactly the `obs_intervals` restriction from Step 2.

### Processing Plots Review
Each `processing_<session>.png` has 12 panels covering every processing stage. What
they show (no anomalies found):
- **Population PSTH**: flat baseline, a bump beginning exactly at the median tone
  onset (−1.85 s) and ending at the end of the sample epoch (−1.2 s), and a large
  transient starting exactly at t = 0. Independent confirmation that the neural
  data are correctly aligned to the go cue.
- **Trial-interval coverage**: the fraction of trials with no population spikes is
  0 for every bin up to +0.6 s, then rises to the fraction of error trials — the
  `stop_time` truncation described in Step 2, and nothing else.
- **Neural heat map** for one trial: firing rates are sensible (0 – a few hundred Hz).
- **input 0**: a straight ramp of slope 1 crossing zero at the tone onset.
- **input 1**: photostimulation confined to the delay epoch, always ending at or
  before the go cue; a raster over trials shows the ~20 % randomly interleaved
  stimulation trials.
- **raw photostim event histogram**: on at −1.2 s or −0.5 s, off 0.5 s later.
- **tongue raw y vs derived class** for one trial: the step trace of the derived
  class follows the raw visible-frame positions and switches at the 40th/60th
  percentile lines; class 3 exactly where there are no visible frames.
- **tongue class raster / class fractions over time**: ≈0 visible before the go cue,
  rising abruptly at t = 0 — the licking response. Correct temporal alignment of
  the video stream against the neural stream.
- **per-trial outputs, class fractions, PSTH split by choice**: lick-left, lick-right
  and no-lick trials have clearly different population responses.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|---|---|
| single `np.bincount` histogram instead of per-(unit,trial) `searchsorted` | ≈100× on the binning step |
| same for the 300 Hz video frames | ≈50× on the tongue step |
| slice only QC-passing units out of `spike_times` | ≈2× on I/O |
| 24 worker processes | ≈10× wall-clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| read NWB | 0.45 s | 78 s CPU |
| bin spikes | 0.50 s | 87 s CPU |
| inputs + outputs | 0.02 s | 3 s CPU |
| pack | 0.02 s | 3 s CPU |
| **total (24 workers)** | **0.11 s wall** | **19 s** + 17 s to write the pickle |

The two sample sessions are close to the dataset median in neurons and trials, and
the full run (36 s) confirmed the estimate. No further optimisation needed.

### Format verification (`--verify-only`)
`/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**
All trials have T = 80; input range and output distributions as tabulated above.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 723 trials)
Training loss falls monotonically from 11.9 (epoch 1) to 0.49 (epoch 200);
test loss 0.53 — no divergence, no overfitting collapse.

| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|--------------------------|
| choice | 0.333 | 0.788 | 0.667 |
| outcome | 0.333 | 0.779 | 0.558 |
| early_lick | 0.500 | 0.975 | 0.960 |
| tongue_y_position | 0.250 | 0.636 | 0.575 |

Every output is above chance. (`early_lick` is nearly degenerate in these two
sessions — only 0.8 % "yes" — so its balanced accuracy is based on very few
positive trials; the full dataset has 11.6 % early-lick trials.)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --jobs 24`
→ `/app/conversion_full_out.txt` (36 s wall-clock).
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only`
→ `/app/verification_full_out.txt`: **"Data format is valid, no errors or warnings."**

### Output Files
- `converted_data.pkl`: 9.45 GB
- `converted_data_session_info.json`: per-file record incl. rejection reasons
- `verification_full_out.txt`: created

### Session accounting (174 source files)
| | sessions |
|---|---|
| available | 174 |
| rejected — behavioural performance ≤ 65 % | 25 |
| rejected — < 50 correct lick-left or lick-right control trials | 12 (4 of which lost nearly all trials to missing video) |
| rejected — DeepLabCut tongue tracking failure | 1 |
| **kept** | **136** |

Trial accounting inside the 136 kept sessions: 74,689 behavioural trials →
73,953 with ephys coverage → −2,759 auto/free-water → −254 without full video →
**70,949 converted trials**.

### Consistency Check
"Reference data" = statistics recomputed directly from all 174 NWB files;
"Converted data" = the 136 kept sessions in `converted_data.pkl`.

| Statistic | Reference Papers | Reference Code | Reference Data (all 174 files) | Converted Data | Match? |
|-----------|------------------|----------------|-------------------------------|----------------|--------|
| Subjects | 28 | — | 28 | 28 | ✅ |
| Sessions | 173 behavioural | — | 174 files | 136 after curation | ✅ (see note 1) |
| Good-unit fraction of all KS2 clusters | 25.9 % | `classification=='good'` | 25.51 % | **25.90 %** | ✅ |
| Total good units | 69,943 | — | 69,453 | 54,629 (kept sessions) | ✅ (note 1) |
| Median neurons / session | 393 | — | 390 | **392** | ✅ |
| Mean neurons / session | — | — | 399 | 402 | ✅ |
| Trials total | — | — | 94,990 (93,450 with ephys) | 70,949 | ✅ |
| Trials / session (mean) | 476 (range 130–785) | — | 546 all / **484 excluding early licks** | 522 (206–796) | ✅ (note 2) |
| Correct rate (control, non-early) | **84 %** (range 65–99 %) | — | 80.6 % over all 174 | **83.8 %** (range 65.8–98.9 %) | ✅ |
| Photostimulation fraction | "typically 25 %" | — | 19.6 % | 20.3 % of trials, 133/136 sessions | ✅ |
| Photostimulation duration | 0.5 s | — | 0.5 s for all 18,588 events | 10 bins × 50 ms = 0.5 s for 14,368/14,382 trials | ✅ |
| Photostim always ends before go cue | yes | — | yes (±1 ms timestamp jitter) | latest "on" bin centre = −0.025 s | ✅ |
| Sample epoch duration | 0.65 s | — | 0.65 s for every trial | tone onset at −1.85 s for 87.5 % of trials | ✅ |
| Delay epoch duration | 1.2 s | — | 1.2 s (0.3 s / 1.8 s in a few % of trials) | reflected per trial in input 0 | ✅ |
| Good units by region (all 174 files) | ALM 8717, Str 7664, Thal 12808, Mid 7495, Med 2928 | 14-region scheme | **ALM 7346, Str 7666, Thal 12807, Mid 7500, Med 2921** | — | ✅ except ALM (note 3) |
| `choice` distribution | — | — | — | [left 0.448, right 0.442, no lick 0.110] | ✅ |
| `outcome` distribution | hit-rate 84 % on control trials | — | hit 0.687, miss 0.165, ignore 0.148 | hit 0.737, miss 0.153, ignore 0.110 | ✅ |
| `early_lick` distribution | — | — | 0.114 early | 0.116 early | ✅ |
| `tongue_y_position` distribution | — | — | tongue visible ≈11 % of a session | [0.106, 0.053, 0.106, 0.734]; of *visible* bins exactly 40/20/40 | ✅ |
| `time_from_tone_onset_s` range | — | — | tone at −1.85 s (87.5 %) | [−1.525, 11.894] s; 1.875 s at the go-cue bin for 87.4 % of trials | ✅ |
| `photostim_on` range | — | — | — | [0, 1], on in 2.53 % of all time bins | ✅ |

Notes:
1. The archive holds 174 files (28 subjects); `sub-456772` contributed two sessions
   on each of five days. The papers report 173 behavioural sessions and 69,943 good
   units, 0.7 % more than the 69,453 in the released files — a discrepancy in the
   published archive, not in this conversion.
2. The papers' "476 trials per session" matches the mean number of **non-early-lick**
   trials in the data (484). Our converted mean (522) is higher because early-lick
   trials must be kept (they are a required decoder output).
3. Four of the five region counts reproduce the paper to ≤0.25 %; ALM differs
   because the reference used a custom CCF voxel mask that is not distributed
   (see Step 4).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or
warnings."** There are no errors and no warnings to address.
(`train_decoder.py` emits one `sklearn` `UserWarning: y_pred contains classes not
in y_true` during scoring. That is produced by `balanced_accuracy_score` inside the
decoder when a held-out split of one session happens to contain no trial of some
class — e.g. a session with two no-lick trials both landing in the training split.
It is a property of the random train/test split, not of the data, and cannot be
removed without dropping legitimate rare classes.)

### Check 2 — Independent sanity checks
`/app/cache/sanity_checks.py` reloads the source NWB files with code that does not
import `convert_data.py` and compares against the pickle with `np.allclose` /
exact equality, for 4 randomly chosen sessions. Output:
`/app/cache/sanity_checks_out.txt` — **56/56 PASS**.

| # | Stream | Check | Result |
|---|--------|-------|--------|
| 1 | provenance | converted trial count == length of `metadata['raw_trial_index'][s]` | PASS |
| 2 | neural | neuron count == #(good ∧ CCF-mappable) units in the NWB | PASS (e.g. 304 vs 304) |
| 3 | neural | spot-check 3 random (trial, neuron, bin) cells: recount spikes in `[go−2.5+b·0.05, +0.05)` from the ragged `units/spike_times` → firing rate | PASS (e.g. 2 spikes → 40.0 Hz, got 40.0) |
| 4 | neural | total spikes of a whole random trial across all neurons, computed without any binning | PASS (14519 vs 14519) |
| 5 | input 0 | recompute `bin_centre − (last sample_start ≤ go − go)` for 3 random trials | PASS, max abs error 9.5e-8 |
| 6 | input 1 | recompute the photostim trace for **all** trials of the session from `photostim_start/stop_times` | PASS (0 mismatches over 1,535 trials) |
| 7 | output 0 | recompute `choice` from `outcome` + `trial_instruction` for all trials | PASS |
| 8 | output 0 | cross-validate `choice` against the direction of the first lick inside the 1.5 s answer period (non-early trials) | PASS, 1,318/1,319 = 99.92 % |
| 9 | output 1 | recompute `outcome` for all trials | PASS |
| 10 | output 2 | recompute `early_lick` for all trials | PASS |
| 11 | outputs 0–2 | are constant across the 80 bins of each trial | PASS |
| 12 | output 3 | recompute the tongue class for **every** (trial, bin) from the raw 300 Hz `Camera0_side_TongueTracking` stream, including the session percentiles | PASS, 0 mismatches |
| 13 | regions | `brain_regions[brain_region_idx[s][i]]` == hemisphere(CCF x) + region(`anno_name`) for 3 random neurons | PASS |
| 14 | subjects | `subjects[subject_idx[s]]` == `general/subject/subject_id` | PASS |

The single disagreement in check 8 is a non-early 'miss' trial in which a left lick
was recorded 2.7 ms after the go cue (a lick already in flight) before the scored
right lick at 187 ms; the rig's `outcome` field is authoritative.

### Check 3 — Reference code comparison
| Stage | Reference code | This conversion | Same? |
|---|---|---|---|
| (a) loading | `preprocessing_DJ_2022Aug.process_one_sess` reads DataJoint `.mat` exports, concatenating all probes of a session | one NWB file per session already contains all probes; variable-by-variable mapping table in Step 4 | Yes — same variables |
| (b) neuron filtering | `qc_mode='classifier'` "good units" index lists ∩ hemisphere mask ∩ non-empty CCF annotation; `check_fr` drops zero-variance neurons | `units/classification == 'good'` ∩ CCF annotation that maps to one of the same 14 regions; silent neurons dropped | Yes. Verified: our good-unit fraction is 25.90 % vs the paper's 25.9 %, and 4/5 regional unit counts reproduce to ≤0.25 % |
| | the 2 Hz firing-rate cut of the methods paper | **not applied** | Deliberate: that cut belongs to the single-neuron ridge-regression analysis (unstable R² at low rates), not to data quality; the reference population-decoding loader (`population_decoding_utils.load_session`) does not apply it, and it would discard a large share of the population |
| (c) temporal alignment | go cue = 0 for spikes, licks and photostimulation (`process_one_sess` subtracts `task_cue_time[0]`); markers aligned to the go cue in `align_markers_between_lims` | identical: `go_start_times` is the origin for spikes, tone onset, photostimulation and video | Yes |
| (d) binning | `sliding_histogram`, bin width 0.04 s, stride 0.0034 s, rate = count / bin width | 80 non-overlapping bins of width 0.05 s, rate = count / 0.05 | Width/stride differ **because the task specifies 50 ms bins**; the rate convention and the bin-centre grid construction are the same |
| (e) input construction | the reference has no decoder; it stores `task_stimulation` (power, type, on, off, relative to the go cue) and the epoch times | photostim trace built from the same event times; tone onset from the same sample-epoch events | Same source variables |
| (f) output construction | `behavior_report` (correct/error/no-response), `task_trial_type` (l/r), `behavior_early_report`, `tracking.camera_0_side.tongue_y/likelihood` | `outcome`, `trial_instruction`, `early_lick`, `Camera0_side_TongueTracking` — the NWB counterparts of exactly those four | Yes |
| trial curation | `get_regular_trial_mask`: no early lick, no auto-water, no free-water, no no-response, no photostimulation | auto-/free-water excluded; early-lick, no-response and photostimulation trials **kept** | Deliberate: those three are required decoder inputs/outputs (Step 4) |
| video trial curation | `get_bad_trial_inds` drops trials whose frame count does not match the trial duration | drop trials with < 90 % of the expected 1,176 frames in the window (254 trials) | Same intent |
| session curation | `align_markers.py` skips sessions without full marker data | plus the data paper's performance / ≥50-correct criteria, plus one session whose tongue tracking failed | Same intent, extended with the paper's stated criteria |

### Check 4 — Key statistics comparison
See the table in Step 9. Every statistic that the papers report is reproduced:
28 subjects; 25.9 % good-unit fraction (exact); median 392 vs 393 neurons per
session; 84 % vs 83.8 % correct rate with range 65–99 % vs 65.8–98.9 %; 0.65 s
sample and 1.2 s delay epochs; 0.5 s photostimulation always ending before the go
cue; ~20 % photostimulation trials vs "typically 25 %"; regional unit counts for
striatum/thalamus/midbrain/medulla within 0.25 %. The only two residual differences
(ALM unit count, 173 vs 174 sessions) are explained above and are not caused by the
conversion.

### Check 5 — Edge cases found and handled
| Issue | Handling |
|---|---|
| **Ephys covers only part of the behavioural session** in 9 of 174 files (`units/obs_intervals` and `is_good_trials` span only a prefix of the trials table). Found because 321/480 trials of one session came out with zero spikes. | The observed trials are recovered by matching `obs_intervals` against the trials table (asserted with `np.allclose`) and only those trials are converted. Verified afterwards: **no converted trial has zero total spikes**. |
| Spikes exist only inside `[trial.start_time, trial.stop_time]`; for error trials that interval ends ~0.2–0.5 s after the go cue. | Unavoidable property of the published data (100 % of spikes lie inside the trial intervals). Bins beyond the interval stay at 0 Hz; documented in `metadata['known_limitations']` and visualised in the `--show-processing` "trial-interval coverage" panel. |
| **Photostim off-timestamps overshoot the go cue by ~1 ms** in 35 % of stimulation events, which lit up the go-cue bin under an interval-overlap rule. | The binary laser signal is sampled at the bin centres instead. Each 0.5 s stimulation now covers exactly 10 bins and never reaches the go-cue bin, as the papers state. |
| **Degenerate DeepLabCut tongue tracking** in `455220_20190803T150200`: likelihood > 0.5 on 96 % of frames with all y values within 1 px, so the 40th/60th percentiles were 0.6 px apart. | Session rejected (`MAX_TONGUE_VISIBLE_FRACTION = 0.5`); the tongue is physically out of the mouth on ~11 % of frames (99th percentile across sessions 25 %). |
| **Missing video** in 7 sessions (as few as 14 k frames for a whole session). | Trials without ≥90 % window coverage are dropped; 4 of those sessions consequently fall below the ≥50-correct-trials criterion and are rejected outright. |
| Trials whose window starts before the trial's own start (3.1 % of trials) or whose tone onset precedes the window (5.0 %). | Handled correctly by construction: the window is defined on the session clock, so it simply extends into the previous trial's recording; `time_from_tone_onset_s` takes the correct (larger) value. |
| Sample-epoch replays after early licks give several sample onsets per trial. | The tone onset is the **last** sample onset at or before the go cue. Verified that this epoch always has the full 0.65 s duration in all 94,990 trials, i.e. it is never a truncated/aborted presentation. |
| Trials with a 0.3 s or 1.8 s delay instead of 1.2 s (a few %). | Not special-cased: `time_from_tone_onset_s` is computed per trial from the actual events, so it reflects the true delay. |
| First/last trial of a session: `raw_trial_index + 1` used when searching for a trial's photostim event. | Guarded (`if raw+1 < len(trial_start) else inf`) in the check script; the converter maps stimulation events to trials with `searchsorted` and never indexes past the end. |
| Floating-point bin index at the window edge. | `np.clip(b, 0, 79)` after the explicit `t < win_end` mask; spikes exactly at `go+1.5` are excluded, spikes exactly at `go−2.5` fall in bin 0. |
| `units/electrodes` assumed to index electrode rows directly. | Asserted that `electrodes/id == arange(n)` in every file. |
| `units/is_good_trials` False for some (unit, trial) pairs. | 99.88 % of pairs are True and 99.88 % of good units are valid on *all* trials, so no action is taken; documented. |
| Neurons with a NaN CCF coordinate. | None occur among good units (checked over the whole archive); a fall-back to the probe's targeted ML sign is implemented anyway. |
| Empty / unmappable CCF annotation. | All 293 distinct annotations of the good units map to one of the 14 regions (0 unmapped); a unit without a mapping would be dropped. |

### Iterations performed
1. **Zero-spike trials** (found by the "trial-interval coverage" diagnostic panel) →
   `obs_intervals` restriction added → re-ran conversion, verification and sanity
   checks → 0 zero-spike trials.
2. **Tongue percentiles computed over whole-session raw frames** gave an unbalanced
   44/31/24 split of the visible bins → switched to percentiles of the per-bin
   tongue position over the session's converted trials → exactly 40/20/40.
3. **Video quality** audit → per-trial video-coverage filter and the tongue-tracking
   session filter → re-ran everything.
4. **Photostim go-cue bin artefact** → bin-centre sampling → re-ran everything;
   all 56 sanity checks still pass and the photostim duration is now exactly 10 bins.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`
(4 min 39 s on the L4 GPU). 56,705 training trials / 14,244 validation trials.

### Training Progress
- Loss decreasing: **Yes**, monotonically — 18.4 (epoch 1) → 12.6 (3) → 7.6 (10)
  → 4.7 (20) → 1.8 (50) → 0.87 (100) → 0.65 (200). Test loss 0.665, essentially
  equal to the final training loss, so the model is not overfitting.

### Decoder Results (Full, 136 sessions / 70,949 trials)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Train / Val |
|--------|--------|-----------------------|-------------------------|--------------|-------------|
| choice | 0.3333 | 0.7187 | **0.6759** | 2.03× | 1.06 |
| outcome | 0.3333 | 0.7048 | **0.6451** | 1.94× | 1.09 |
| early_lick | 0.5000 | 0.7868 | **0.7521** | 1.50× | 1.05 |
| tongue_y_position | 0.2500 | 0.6954 | **0.6657** | 2.66× | 1.04 |

`sample_trials.png` and `predictions.png` show individual trials: the tongue output
starts varying exactly at bin 50 (t = 0, the go cue) on lick trials, stays at "not
visible" on no-lick trials, and varies before the go cue on early-lick trials.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
Every output is far above chance and none is below 1.5× chance:

| Output | Chance (1/n_classes) | Validation | Ratio |
|---|---|---|---|
| choice | 0.333 | 0.676 | **2.03×** |
| outcome | 0.333 | 0.645 | **1.94×** |
| early_lick | 0.500 | 0.752 | **1.50×** |
| tongue_y_position | 0.250 | 0.666 | **2.66×** |

`early_lick` sits exactly at the 1.5× line, so it was investigated separately
(`/app/cache/time_resolved_accuracy.py`). It is a *binary* variable, for which
1.5× chance is 0.75 balanced accuracy — a strong effect, not a weak one; the same
ratio for a 4-class variable would be 0.375. Two further facts rule out a
conversion bug:
- 99.1 % of early-lick trials contain at least one recorded lick inside the
  extracted window, and their pre-go-cue tongue-visible fraction is 26.3 % versus
  4.7 % on non-early trials — the event really is inside the window and really is
  visible in the behavioural data.
- Resolved in time, `early_lick` decoding **peaks at 0.818 during the sample epoch
  and 0.812 during the delay** (exactly when an early lick can occur) and falls to
  0.696 in the response epoch (after the go cue, when every mouse is licking and the
  distinction is no longer visible). That is the physiologically correct time course;
  the whole-trial average of 0.752 is diluted by the response epoch.

### Check 2 — Accuracy comparison to the papers
Neither reference paper trains this decoder, so the comparable quantities are the
choice decoders of the data paper. To compare like with like, the 3-way `choice`
predictions were restricted to lick-left/lick-right trials and scored as binary
(`/app/cache/time_resolved_accuracy.py`):

| Quantity | Paper | This conversion | Comment |
|---|---|---|---|
| Single-session population choice decoding, all neurons in an area, mean over late delay [−0.6, −0.1] s (Fig. S6B) | dots scattered ≈0.6–0.9, mean ≈0.75 for ALM and lower for other areas | **0.691** over [−0.6, 0) s | Matching regime (one session's neurons, late delay). Our sessions mix all 14 regions, most far less choice-selective than ALM |
| Population choice decoding at t = −0.1 s vs neuron count (Fig. S6E) | 0.55–0.6 at 20 neurons, rising with neuron count | **0.699** at t = −0.075 s | In range |
| Pseudo-population choice decoding, 200 ALM neurons pooled across mice, 200 ms causal window, end of delay (Fig. 6D) | ≈0.90 (ALM); ≈0.6–0.75 for striatum/thalamus/midbrain/medulla | 0.691 late delay, **0.927 peak (response epoch)** | Not directly comparable — see below |
| Single-neuron choice AUC threshold used to call a neuron "choice-modulated" (methodpaper Fig. 7) | 0.65 | — | population decoding, not per neuron |
| Video→spike-rate prediction (methodpaper Fig. 2) | R² 0.2–0.8 | — | regression, not classification |

Why the Fig. 6D number is higher than our late-delay value, and why this is not a
conversion defect:
1. **Neuron pooling.** Fig. 6D builds 200-neuron pseudo-populations *per brain area*
   by hierarchical bootstrapping across mice and sessions. Our decoder sees one real
   session at a time, projected to 100 dimensions, with all regions mixed. The
   paper's own Fig. S6E shows accuracy at t = −0.1 s climbing steeply with neuron
   count; the single-session version of the same measurement (Fig. S6B) is
   substantially lower and is what our 0.691 matches.
2. **Smoothing.** The paper uses 200 ms causal windows; the task here specifies
   50 ms bins, i.e. a quarter of the spikes per sample.
3. **Trial composition.** The paper excludes photostimulation trials; we must keep
   them (photostimulation is a required decoder input). Bilateral ALM inactivation
   drops behavioural performance from 83.2 % to 71.7 % and is designed to disrupt
   the delay-epoch choice code, so those 20 % of trials lower late-delay choice
   decoding by construction.
4. **A single shared decoder** is fitted jointly across 136 sessions and all 80 time
   points, rather than a separate nested-cross-validated logistic regression per
   brain area and per time point.

Crucially, the **peak binary choice accuracy of 0.927** (in the response epoch)
exceeds the paper's best late-delay ALM value, which rules out any temporal
misalignment or label error: the information is there, it simply emerges with the
task's own time course.

### Time-resolved validation accuracy (`/app/cache/time_resolved_accuracy_out.txt`)
| Output | pre-sample [−2.5,−1.85) | sample [−1.85,−1.2) | delay [−1.2,0) | response [0,1.5) | peak |
|---|---|---|---|---|---|
| choice (3-way) | 0.534 | 0.623 | 0.660 | **0.790** | 0.874 |
| choice (binary L/R) | 0.531 | 0.639 | 0.678 | **0.838** | 0.927 |
| outcome | 0.556 | 0.575 | 0.591 | **0.795** | 0.881 |
| early_lick | 0.736 | **0.819** | **0.812** | 0.696 | 0.845 |
| tongue_y_position | 0.681 | 0.666 | 0.644 | 0.589 | 0.698 |

Every time course is the physiologically expected one, which is the strongest
available evidence that the streams are correctly aligned:
- choice and outcome are near chance before the instruction tone, rise through the
  sample and delay epochs, and jump after the go cue;
- early-lick decoding peaks precisely in the sample/delay epochs where early licks
  occur;
- tongue position is decodable throughout (licking also occurs before the go cue on
  early-lick trials and at the start of the window, which overlaps the previous
  trial's consumption licking).

### Check 3 — Train vs validation gap
Ratios train/validation: choice 1.06, outcome 1.09, early_lick 1.05,
tongue_y_position 1.04 — all far below the 1.5 threshold. The final training loss
(0.654) and test loss (0.665) are nearly identical. No overfitting and no data
leakage (the train/test split is over trials, and no per-trial information is shared
between trials other than the session-level tongue percentiles, which are a property
of the session rather than of any individual trial).

### Issues found and resolved in this step
- `early_lick` initially looked like the weakest output (1.50× chance). Investigated
  as above; no conversion defect — the whole-trial average is diluted by the
  response epoch, where the variable is genuinely not decodable.
- `choice` late-delay accuracy is below the headline number of Fig. 6D. Investigated
  as above; fully explained by pseudo-population pooling, smoothing window and the
  inclusion of photostimulation trials, all of which are forced by the decoder
  specification. No change made.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, how to load and use the data, full
      output-format specification, key statistics, filtering rationale, file list.
- [x] `CONVERSION_NOTES.md` complete (this file).
- [x] `cache/` folder created with `README_CACHE.md`.
- [x] All files organised.

### Deliverables
| File | Purpose |
|---|---|
| `/app/converted_data.pkl` | full converted dataset (9.45 GB, 136 sessions) |
| `/app/sample_data.pkl` | two-session sample (0.07 GB) |
| `/app/convert_data.py` | conversion script |
| `/app/region_map.py` | CCF annotation → 14-region table |
| `/app/CONVERSION_NOTES.md` | this record |
| `/app/README.md` | user-facing documentation |
| `/app/conversion_sample_out.txt`, `/app/conversion_full_out.txt` | conversion logs |
| `/app/verification_sample_out.txt`, `/app/verification_full_out.txt` | format verification |
| `/app/train_decoder_sample_out.txt`, `/app/train_decoder_full_out.txt` | decoder training |
| `/app/processing_*.png` | per-step diagnostic plots |
| `/app/sample_trials.png`, `/app/predictions.png` | decoder sample/prediction plots |
| `/app/cache/` | investigation and verification scripts and their outputs |

# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (Chen et al.), NWB files in /app/data
- **Date started**: (see file timestamps)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `ChenLiuEtAl2023_SpikeSortingQC.pdf` (spike sorting/QC white paper)
- `datapaper.pdf` (Chen et al., Brain-wide neural activity underlying memory-guided movement)
- `methodpaper.pdf` (Wang, Kurgyis et al. 2025, Brain-wide analysis reveals movement encoding...)
- `methods.txt` (methods excerpts)
- `code/` (MapVideoAnalysis repo: VideoAnalysisUtils/, Sherlock/, Notebooks/, Archive/)
- `data/` (DANDI 000363 subset: 28 `sub-*` folders, 174 NWB files, `dandiset.yaml`)
- `decoder.py`, `train_decoder.py` (decoder validation code)

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel(bw, stride, begin_time, end_time, data_folder, save_folder, qc_mode, n_cpu)` | `Sherlock/preprocess_all_ephys.py` -> `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Top-level entry. Called with **bw = 0.04 s, stride = 0.0034 s, begin = -3.0 s, end = +3.0 s, qc_mode = 'classifier'** |
| `process_one_sess(...)` | `preprocessing_DJ_2022Aug.py` | LOADING/CURATION | Concatenates probes of a session, reads behavior/task variables, keeps only units that have **both ephys and histology**, applies the **classifier-based good-unit list** (`goodunits` QC files), splits neurons by hemisphere and coarse region |
| `sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True)` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Sliding-window spike counts -> **firing rates (spikes/s)** = count / bin_width. Bin centers from begin to end |
| `process_one_area(...)` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates spike times (already **aligned to go cue = 0**) to [begin,end], bins them, saves `fr` of shape (n_bins, n_trials, n_neurons) |
| `helper_get_neuron_id_area(...)` | `preprocessing_DJ_2022Aug.py` | CURATION | Hemisphere assignment using CCF ML coordinate, midline **ML = 5700 um** (`>=5700` -> left); intersects with QC good-unit ids; drops units with empty CCF label |
| `get_regular_trial_mask(ephys_data)` | `population_decoding_utils.py`, `functions_for_r2.py` | CURATION | Regular trials = no early lick, no auto water, no free water, **not** no-response, no photostimulation |
| `align_markers_between_lims(marker_data, go_times, t_min=-3, t_max=1.5)` | `Sherlock/align_markers.py` | PROCESSING | Aligns DeepLabCut markers (nose/tongue/jaw x,y) to **go cue**, dt = 0.0034 s, takes the **last frame within each bin** |
| `get_bad_trial_inds(...)` | `Sherlock/align_markers.py` | CURATION | Drops trials whose number of video frames does not match the trial duration (missing/extra frames) |
| `load_session(session_path, area)` | `population_decoding_utils.py` | LOADING | Reassembles per-area pickles into a session: concatenates `fr` over neurons |
| `nested_cross_validation(X, y, ...)` | `population_decoding_utils.py` | ANALYSIS | PCA(16) + logistic regression decoding of binary trial labels per time point |

### Notes
- The reference preprocessing operates on **MATLAB DataJoint exports** of the same dataset that is distributed here as NWB (DANDI 000363). Variable names map 1:1 onto NWB fields (see Step 4).
- Neural processing: spikes aligned to the **go cue**, binned with a **40 ms sliding window** and converted to **rates in Hz** (divide by bin width). The stride (3.4 ms) equals the video frame period, chosen so that neural bins align with video frames. Our task specifies **50 ms non-overlapping bins**, i.e. bw = stride = 50 ms (documented deviation required by the decoder spec).
- Neuron curation: **only classifier-'good' units** (region-specific logistic-regression classifiers described in the QC white paper) **that also have a CCF annotation** are used.
- Trial curation for the *reference analyses*: photostim, free/auto water, early-lick and no-response (ignore) trials are excluded. Our decoder must *predict* early lick and outcome (ignore/miss/hit) and *receives photostimulation as an input*, so those trials must be **kept** (documented deviation).
- No dF/F is needed (electrophysiology).
- Additional neuron filter used in the method paper analyses: neurons with mean firing rate **< 2 Hz excluded**; in this conversion this would discard 33% of good units and remove the sparse-firing population that still carries decodable information, so it is **not** applied (documented in Step 5).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` + 28 subject folders `sub-<id>/`, 174 NWB (HDF5, NWB 2.6) files named
  `sub-<id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb` (DANDI 000363, Chen et al. MAP dataset).
- Contents of each file (verified by `h5py.visititems`):
  - `intervals/trials` (one row per trial): `start_time`, `stop_time`, `outcome` {hit, miss, ignore},
    `early_lick` {early, no early}, `trial_instruction` {left, right}, `auto_water`, `free_water`,
    `photostim_onset` / `photostim_duration` / `photostim_power` (strings, `'N/A'` when no stim, onset given
    relative to trial start), `task` = 'audio delay', `task_protocol` = 1, `id`, `trial`, `trial_uid`.
  - `acquisition/BehavioralEvents/*`: TimeSeries with **absolute session timestamps**:
    `presample/sample/delay/go/trialend_start_times` and `_stop_times`, `left_lick_times`, `right_lick_times`,
    `photostim_start_times` (data = laser power in mW, `control` = stim site code 1/2/4/6), `photostim_stop_times`.
    `sample_start_times` has MORE entries than trials because the sample epoch is **replayed after early licks**.
  - `acquisition/BehavioralTimeSeries/Camera0_side_{Jaw,Nose,Tongue}Tracking`: DeepLabCut markers,
    `data` = (n_frames, 3) = (x, y, likelihood), `timestamps` = absolute session time, **dt = 0.0034 s (~294 Hz)**.
    Some sessions additionally contain `Camera0_side_WhiskerTracking_whisker` (20), `Camera0_side_LickPortTracking` (4)
    or a second camera `Camera3_side_*` (3). All 174 sessions have the side Jaw/Nose/Tongue markers.
  - `units` table: `spike_times` (+`spike_times_index`) with **absolute session times**, `obs_intervals`
    (= exactly the [trial start, trial stop] windows; **spikes exist only inside trials**),
    `classification` {good, unlabelled} = output of the region-specific QC classifiers, `unit_quality` {good, multi},
    `anno_name` = CCF annotation (empty string exactly for the `unlabelled` units), `electrodes` /`electrodes_index`,
    `electrode_group`, 15+ QC metrics (`presence_ratio`, `amplitude_cutoff`, `isi_violation`, `drift_metric`,
    `avg_firing_rate`, `unit_amp`, `unit_snr`, ...), `is_good_trials` (n_units x n_trials bool).
  - `general/extracellular_ephys`: one `ElectrodeGroup` per probe insertion, whose `location` attribute is a JSON
    string with the targeted `brain_regions` (e.g. 'left ALM', 'right Medulla') and stereotaxic coordinates;
    `electrodes` table with CCF coordinates `x` (ML), `y` (DV), `z` (AP) in um.
  - `general/subject/subject_id`, `session_start_time`, `identifier` (e.g. `SC015_20190207_120657_s1`).
- No `processing` module, no `stimulus` content.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files (sessions) | 174 (173 with >=1 good unit; `SC017_20190216_162508_s4` has 0 good units) |
| Units (all clusters) | 272,227 |
| Units classified 'good' | **69,453** (25.5% of clusters) |
| Good units / session | mean 399, range 0-923 |
| Subjects | 28 |
| Sessions / subject | mean 6.2 (range 1-13) |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264-800 |
| Probe insertions (ElectrodeGroups) | 659 |
| Outcome distribution | hit 68.7%, miss 16.5%, ignore 14.8% |
| Early lick | 11.4% |
| Instruction | right 51.5% / left 48.5% |
| Photostim trials | 18,588 (19.6%); 168/174 sessions have stim |
| Auto-water trials | 1,339 ; free-water trials 2,450 |
| CCF annotations (good units) | 293 distinct leaf structures |
| Video frame period | 0.0034 s in every session |

### Timing facts measured from the data (critical for conversion)
| Quantity | Value |
|----------|-------|
| go cue - trial start | median 3.15 s; 1st percentile 2.19 s; min 2.11 s; **3.1% of trials < 2.5 s** |
| trial stop - go cue, hit trials | median 1.82 s, min 1.58 s (100% >= 1.5 s) |
| trial stop - go cue, ignore trials | 1.80 s (100% >= 1.5 s) |
| trial stop - go cue, **miss (error) trials** | median 0.79 s; **only 5.1% >= 1.5 s** (timeout ends the trial early) |
| go cue - sample (tone) onset | 1.85 s = 0.65 s sample + 1.2 s delay (replayed sample epochs are later) |
| photostim (when present) | starts ~1.2 s before go cue, duration 0.5 s -> ends before the go cue |

Consequence: spike data and video frames are only defined inside [trial start, trial stop]. Requiring the full
[-2.5, +1.5] s window to be observed would delete ~95% of error (miss) trials and make the `outcome` output
nearly degenerate; unobserved bins are therefore zero-filled for neural data (no spikes recorded) and the
'not visible' class is used for the tongue (no video frames).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt / datapaper) |
| Fraction of KS2 clusters kept | 25.9% | "This corresponds to 25.9 % of clusters reported by Kilosort2." |
| Good units by area | ALM 8717, striatum 7664, thalamus 12808, midbrain 7495, medulla 2928 | methods.txt |
| Sessions | 173 behavioral sessions | methods.txt, datapaper Fig. 1J |
| Probe insertions | 655 (Fig 1J caption: 660 penetrations) | methods.txt / datapaper |
| Subjects | 28 mice | "aggregated over 660 penetrations, 173 behavioral sessions, and 28 mice"; method paper: "data from 28 mice" |
| Trials / session | mean 476, range 130-785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84% (range 65-99%) | "with 84% correct rate (range, 65-99%)" (control, non-early-lick trials) |
| Session inclusion (data paper) | performance > 65% and >= 50 correct lick-left and lick-right trials | methods.txt |
| Photostim trials | ~25% randomly interleaved, 17 VGAT-ChR2 mice | methods.txt |
| Photostim effect | performance 83.2% -> 71.7% under bilateral ALM stim (93 sessions) | methods.txt |
| Neural bin width / stride | 40 ms width, 3.4 ms stride | method paper Methods: "binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Video rate | 300 Hz (actual dt in files 3.4 ms) | method paper |
| Markers | nose, tongue, jaw tracked with DeepLabCut | method paper |
| Task epochs | sample 2 x (3 x 150 ms tones) ~0.65 s, delay 1.2 s, answer 1.5 s | methods.txt |
| Sessions usable for video analyses | 105-106 | method paper Fig. 2 statistics (n = 105 sessions) |

### Processing Details
- Spike sorting: Kilosort2, then 15 QC metrics + **region-specific logistic-regression classifiers** trained on manual
  Phy labels -> units labelled 'good'. In NWB this is the `units/classification` column.
- Alignment: reference DataJoint export stores spike times **relative to the go cue**; the reference marker alignment
  script (`align_markers.py`) also aligns video to the **go cue** over [-3, 1.5] s.
- Binning: 40 ms sliding window, 3.4 ms stride, rates in Hz (`sliding_histogram`).
- Trial selection for the reference analyses: exclude photostim, free water, early lick and ignore (no-response) trials.
- Neuron selection: classifier-good + CCF-annotated; additionally FR < 2 Hz excluded for the regression analyses.

### Curation Steps

**Neuron curation rules** (reference): `units/classification == 'good'` (classifier QC) AND a non-empty CCF annotation
(in NWB these two are equivalent: every unlabelled unit has an empty `anno_name`).

**Trial curation rules** (reference): regular trials = no early lick, no auto water, no free water, no ignore,
no photostim. For this decoding task early lick / outcome are *targets* and photostim is an *input*, so only
auto-water and free-water trials (where reward delivery is not driven by the animal's choice) are dropped, plus
trials whose pre-go-cue window is not observed.

### Decoders Trained (accuracies reported in the papers)
| Decoded variable | Accuracy |
|---|---|
| Choice (lick direction) from **video** during delay epoch | AUC 0.66 +- 0.12 s.d. (n = 106 sessions) |
| Choice from video, response epoch | AUC ~0.9 ("choice being easily decodable from video of directional licking") |
| Choice from video, sample epoch | at chance (AUC ~0.5) |
| Single-neuron choice selectivity criterion | AUC > 0.65 |
| Neural population choice decoding (data paper, CD-based) | not given as accuracy; choice-selective activity widespread |
No paper reports accuracy for decoding outcome, early lick, or tongue position from neural activity; the neural
choice decoding in the data paper (Fig. 2) is near-perfect during the response epoch for ALM populations.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Mapping of reference (DataJoint .mat) variables to NWB fields
| Reference code variable | NWB field | Verified |
|---|---|---|
| `behavior_early_report` | `intervals/trials/early_lick` ('early'/'no early') | yes |
| `behavior_is_auto_water` / `behavior_is_free_water` | `intervals/trials/auto_water` / `free_water` | yes |
| `behavior_report` (1 correct / 0 error / -1 no response) | `intervals/trials/outcome` (hit / miss / ignore) | yes |
| `task_trial_type` ('l'/'r') | `intervals/trials/trial_instruction` (left/right) | yes |
| `task_cue_time[0]` (go cue) | `acquisition/BehavioralEvents/go_start_times/timestamps` | yes, 1 per trial |
| `task_sample_time` | `sample_start_times` / `sample_stop_times` (more events than trials because of early-lick replays) | yes |
| `task_delay_time` | `delay_start_times` / `delay_stop_times` | yes |
| `task_stimulation` [power, type, on, off] | `trials/photostim_power`, `photostim_onset`, `photostim_duration` and `BehavioralEvents/photostim_start_times` (`control` = site code) | yes |
| `behavior_lick_times` / `lick_directions` | `left_lick_times`, `right_lick_times` | yes |
| `neuron_single_units` (spike times per trial, already go-cue aligned) | `units/spike_times` (absolute session time) + `units/obs_intervals` | yes |
| QC `goodunits` classifier lists | `units/classification == 'good'` | yes |
| `histology.annotation` (CCF label) | `units/anno_name` | yes |
| `histology.ccf_x/y/z` | `general/extracellular_ephys/electrodes/x,y,z` via `units/electrodes` | yes |
| DLC markers `tongue_x/y`, `jaw_*`, `nose_*` | `acquisition/BehavioralTimeSeries/Camera0_side_*Tracking/data` = (x, y, likelihood) | yes (NWB adds the DLC likelihood column) |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Good units | classifier QC list per session | 69,453 units with `classification=='good'` (25.5% of 272,227 clusters) | 69,943 good units, 25.9% of KS2 clusters | Agree within 0.7%; the small difference is the published-vs-archived QC list. Use `classification=='good'`. |
| Sessions | one folder per session | 174 NWB files, **173** with >=1 good unit | 173 behavioral sessions | The extra file (`SC017_20190216_162508_s4`) has 0 good units -> naturally dropped. Consistent. |
| Insertions | per-probe .mat files | 659 ElectrodeGroups | 655 (text) / 660 (Fig. 1J) | Consistent within rounding of the published numbers. |
| Per-area unit counts | region split by CCF label + hemisphere | ALM 7,885; Striatum 7,839; Thalamus 12,917; Midbrain 7,500; Medulla 2,918 (my CCF keyword mapping) | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | Excellent agreement (<2% for 4/5 areas; ALM differs by 9% because the paper's ALM is a stereotaxic volume, mine is MOp/MOs units). Mapping accepted. |
| Spike alignment | spike times already relative to go cue | absolute session times; `obs_intervals` = [trial start, stop] | go-cue alignment | Subtract the go cue time per trial; identical result. |
| Behavioral performance | - | control, non-early trials: hit/(hit+miss) = 81.0%, range 54-97% | 84% correct, range 65-99% | Same statistic to within 3%; the paper additionally applied session-selection criteria (>65% performance, >=50 correct trials of each type) that the DANDI release does not apply. |
| Trials/session | - | 546 (all trials), 484 (non-early), range 264-800 | mean 476, range 130-785 | Consistent (the paper counts trials after removing early-lick/ignore trials in some sessions). |
| Trial exclusions | drop early-lick, ignore, photostim, free/auto water | those trials exist and are labelled | same | Decoder spec requires early lick / outcome as targets and photostim as an input -> only auto-water and free-water trials are dropped (see Step 5). |
| Binning | 40 ms window, 3.4 ms stride | - | same | Decoder spec requires 50 ms bins -> 50 ms non-overlapping bins (rates in Hz, same formula count/bin width). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` (good units) | `neural[session][trial]` (n_neurons, 80) | absolute spike times -> histogram in 80 non-overlapping 50-ms bins spanning [go-2.5 s, go+1.5 s]; divided by bin width -> **rate in Hz**, float32 | `sliding_histogram`, `process_one_area` | same rate definition as reference, bin width/stride set to 50 ms by the task spec |
| `sample_start_times` (last one before the go cue) | `input[0]` = 'time from tone onset (s)' | bin-center time minus tone onset time, seconds, float32 | new (spec) | continuous, time-varying |
| `photostim_start_times` / `photostim_stop_times` | `input[1]` = 'photostimulation on' | 1 if the bin overlaps a stim interval else 0 | `task_stimulation` in reference | binary, time-varying |
| `trials/outcome` + `trials/trial_instruction` | `output[0]` = 'lick direction choice' | ignore->'no lick'; hit->instruction; miss->opposite of instruction | `behavior_report`, `task_trial_type` | cross-checked against first lick after the go cue (99.4% agreement) |
| `trials/outcome` | `output[1]` = 'outcome' | ignore=0, miss=1, hit=2 | `behavior_report` | |
| `trials/early_lick` | `output[2]` = 'early lick' | 'no early'=0, 'early'=1 | `behavior_early_report` | |
| `Camera0_side_TongueTracking/data` (y, likelihood) | `output[3]` = 'tongue y-position' | per bin: visible frames = likelihood > 0.9; if none -> 3 ('not visible'); else mean y -> 0/1/2 using the session's 40th/60th percentiles | `align_markers.py` (marker/go-cue alignment) | percentiles computed per session over all visible bins of that session |
| `units/anno_name` + electrode CCF `x` | `brain_region_idx` | CCF leaf -> coarse group (ALM, Striatum, Thalamus, Midbrain, Medulla, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum) x hemisphere (ML >= 5700 um = left) | `helper_get_neuron_id_area` | same 14 groups and same midline as the reference code |
| `general/subject/subject_id` | `subjects`, `subject_idx` | string id per session | | 28 mice |

### Key Decisions
1. **Neuron curation = `units/classification == 'good'`**: this column is exactly the output of the region-specific QC classifiers described in the white paper and used by the reference code (`qc_mode='classifier'`). No further firing-rate threshold is applied: the method paper's 2-Hz cut was specific to its ridge-regression encoding analysis, would remove 33% of the units, and the decoder benefits from all recorded units.
2. **Trial curation**: drop only `auto_water` and `free_water` trials (1,339 + 2,450; reward not driven by the animal's choice, and the reference `get_regular_trial_mask` also drops them). Early-lick, ignore/miss and photostim trials are **kept** because the decoder must predict early lick and outcome and receives photostim as an input; dropping them would make three of the four outputs degenerate.
3. **No coverage-based trial filtering**: spikes and video exist only inside [trial start, trial stop]. Error (miss) trials are terminated ~0.8 s after the go cue, so requiring the whole [-2.5, +1.5] s window to be observed would delete 95% of miss trials. Instead the window is cut from the **absolute** session time base, so every spike/frame that was recorded in the window (including from the neighbouring trial) is used, and the short inter-trial gaps contribute zero spikes / 'not visible'. Worst-case coverage is 66% of the window, mean ~97%.
4. **Rates in Hz** (spike count / 0.05 s), as in `sliding_histogram(rate=True)`.
5. **Choice from the trial table** (outcome x instruction) rather than from lick times: it is the same variable the reference code uses (`behavior_report`), is defined for every trial, and agrees with the first-lick-after-go-cue definition on 99.4% of trials.
6. **Tongue visibility threshold 0.9** on the DLC likelihood: the likelihood distribution is strongly bimodal (99.8% of frames are <0.1 or >0.9), so the exact threshold is irrelevant.
7. **Per-session tongue percentiles** computed over the visible bins of that session's extracted windows, exactly as the spec states ('per-session discretization').
8. **Brain regions are hemisphere-specific** ('left ALM', 'right Medulla', ...), matching the reference code's `left_/right_` area split.
9. **All 173 sessions with >=1 good unit are kept** (the 174th has no good units). No behavioral-performance session filter is applied: the DANDI release already contains the sessions the authors published, and dropping low-performance sessions would remove error trials needed for the outcome output.

### Planned Sanity Checks
- [x] Number of good units per coarse region vs data-paper Fig. 1J numbers.
- [x] 173 sessions / 28 mice / 659 insertions vs paper (173 / 28 / 655-660).
- [x] Behavioral performance ~84%, trials/session ~476-546.
- [ ] Independent re-computation of binned rates for a random (session, trial, neuron) directly from the NWB file (`np.allclose`).
- [ ] Independent re-computation of the photostim input and the time-from-tone input for random trials.
- [ ] Independent re-computation of the tongue class for random trials, and check the class frequencies are ~40/20/40 among visible bins.
- [ ] Choice/outcome/early-lick distributions in the converted data match the raw trial-table distributions.
- [ ] Per-trial neural array shape (n_neurons, 80) and constant n_neurons within a session.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the plan of Step 5:

1. `process_session(path)` opens one NWB file with `h5py` and
   * selects **good units**: `units/classification == 'good'` and a non-empty `units/anno_name`
     (reference `qc_mode='classifier'` + histology requirement),
   * assigns each unit a region `'<hemisphere> <coarse CCF group>'`; hemisphere from the CCF ML
     coordinate of the unit's electrode with the reference midline `ML = 5700 um`,
   * selects trials: drops `auto_water`/`free_water` trials (reference `get_regular_trial_mask`),
     trials outside the units' `obs_intervals`, and trials with no recorded spikes at all,
   * bins spikes into **80 non-overlapping 50 ms bins** covering [go - 2.5 s, go + 1.5 s] with
     `np.searchsorted` on the absolute spike times, dividing by the bin width to obtain **Hz**
     (reference `sliding_histogram(..., rate=True)`),
   * builds the inputs (time from instruction-tone onset; photostimulation on/off per bin),
   * builds the outputs (choice, outcome, early lick, discretized tongue y).
2. `main()` runs the sessions in a `multiprocessing.Pool` (16 workers) and assembles the final dict.
3. `--show-processing` writes `processing_<session_id>.png` with: raw spike raster vs binned rates for
   the same trial, population PSTH split by choice, the two inputs, the tongue-y histogram with the
   40th/60th percentile cuts, the tongue class raster and the per-trial outputs.

Code inefficiencies identified:
- Reading `units/spike_times` per unit from HDF5 is slow -> the whole ragged array is read once and
  sliced with the index vector.
- Binning with nested Python loops (as in the reference `sliding_histogram`) is O(n_bins x n_trials)
  per unit -> replaced by one `np.searchsorted` per unit over all bin edges (vectorised over trials).
- Averaging the 3.4 ms video frames inside each 50 ms bin -> done with cumulative sums, no loop.

Code speedups added: single-pass HDF5 reads, `searchsorted` binning, cumsum-based video binning and
16-way multiprocessing over sessions. Result: **~0.5 s per session**.

### Issues found while developing (and fixed)
1. **All-zero trials**: in 8 of 173 sessions the ephys recording covers only a contiguous subset of the
   behavioural trials. The units' `obs_intervals` reveal this; trials outside them are now dropped
   (`observed_trial_mask`).
2. **Last trial of a truncated recording**: one further trial can be listed in `obs_intervals` while the
   probe was already off. Trials whose extracted window contains no spikes at all are dropped.
3. Brain-region names were `np.str_`; cast to plain `str`.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` ->
`/app/conversion_sample_out.txt`, `/app/sample_data.pkl`, two `processing_*.png`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (SC015_20190207_120657_s1, SC015_20190208_133600_s2) |
| Neurons (total) | 834 (459 + 375) |
| Neurons / session | 417 mean |
| Subjects | 1 (440956) |
| Trials (total) | 513 (354 + 159) |
| Trials / session | 354, 159 (the second session's ephys covers only the first 160 trials) |
| T per trial | 80 bins of 50 ms |
| time_from_tone_onset range | [-0.6, 5.7] s |
| photostim_on range | [0, 1] |
| choice distribution | no lick 0.136, left 0.456, right 0.407 |
| outcome distribution | ignore 0.136, miss 0.298, hit 0.565 |
| early_lick distribution | no 0.942, yes 0.058 |
| tongue_y distribution | 0.094 / 0.047 / 0.094 / 0.766 (i.e. 40/20/40 of the visible bins) |

### Processing Plots Review
- Raw spike raster and the binned rate image line up in time; the go cue is at 0 in both.
- The population PSTH shows the expected go-cue-locked response, differing between lick-left and
  lick-right trials.
- Input 0 is a straight line through the trial, crossing zero ~1.85 s before the go cue (the tone onset).
- Input 1 is a 0.5 s block ending before the go cue, present on ~20% of trials -> matches the protocol
  (photoinhibition of the last 0.5 s of the delay epoch on ~25% of trials).
- The tongue-y histogram is unimodal with the 40th/60th percentile cuts inside the bulk; the tongue class
  raster shows licking bouts only after the go cue (or before it on early-lick trials).
- No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| single read of `units/spike_times` + slicing | ~20x vs per-unit HDF5 reads |
| `searchsorted` binning instead of nested loops | ~100x vs reference `sliding_histogram` |
| cumsum video binning | ~50x vs per-bin loops |
| 16-process pool | ~10x wall clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| load + bin neural | ~0.35 s | ~60 s serial |
| inputs + outputs (incl. video) | ~0.1 s | ~20 s serial |
| whole session | 0.5 s (largest sessions ~3 s) | **< 2 min with 16 workers**, plus ~2-4 min to write the ~12 GB pickle |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (`Data format is valid, no errors or warnings.`)

### Decoder Results (Sample, 2 sessions / 513 trials)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.738 | 0.620 | 0.333 |
| outcome | 0.765 | 0.645 | 0.333 |
| early_lick | 0.832 | 0.733 | 0.500 |
| tongue_y_position | 0.691 | 0.601 | 0.250 |

Training loss decreased monotonically from 7.5 to 0.57 over 200 epochs (test loss 1.05). All four outputs
are well above chance on validation data. (Numbers regenerated with the final version of the script.)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.89 GB (173 sessions, 89,544 trials, 69,453 neurons, float32 rates)
- `conversion_full_out.txt`: created (conversion took 24 s of compute + 21 s to write the pickle)
- `verification_full_out.txt`: created, **`Data format is valid, no errors or warnings.`**

### Converted dataset statistics
| Statistic | Value |
|---|---|
| Sessions | 173 |
| Subjects | 28 (4-10 sessions each) |
| Neurons | 69,453 (mean 401.5/session, range 90-923) |
| Trials | 89,544 (mean 517.6/session, range 159-796) |
| Timepoints / trial | 80 (50 ms bins, -2.5 s to +1.5 s around the go cue) |
| Brain regions | 28 (14 coarse groups x 2 hemispheres) |
| choice | no lick 0.148, left 0.429, right 0.422 |
| outcome | ignore 0.148, miss 0.167, hit 0.685 |
| early lick | no 0.884, yes 0.116 |
| tongue class (all timepoints) | 0.098 / 0.049 / 0.098 / 0.755 (= 40/20/40 of the visible bins) |
| time from tone onset | [-1.52, 11.89] s |
| photostim trials | 17,930 / 89,544 = 20.0% |

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (NWB) | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total good neurons | 69,943 | classifier QC list | 69,453 (`classification=='good'`) | 69,453 | yes (0.7% diff) |
| Fraction of KS2 clusters | 25.9% | - | 25.5% | 25.5% | yes |
| Mean neurons/session | - | - | 399 (of 174 files) | 401.5 | yes |
| Subjects | 28 | - | 28 | 28 | yes |
| Sessions | 173 | - | 174 files, 173 with good units | 173 | yes |
| Probe insertions | 655-660 | - | 659 | 659 (implicit) | yes |
| Trials (total) | - | - | 94,990 | 89,544 (auto/free water, unobserved trials removed) | yes |
| Trials/session (mean) | 476 (range 130-785) | - | 546 all / 484 non-early | 517.6 (range 159-796) | yes |
| ALM neurons | 8,717 | - | 7,885 | 7,885 | close (paper's ALM is a stereotaxic volume, ours is MOp/MOs) |
| Striatum neurons | 7,664 | - | 7,839 | 7,839 | yes (2%) |
| Thalamus neurons | 12,808 | - | 12,922 | 12,922 | yes (0.9%) |
| Midbrain neurons | 7,495 | - | 7,495 | 7,495 | exact |
| Medulla neurons | 2,928 | - | 2,918 | 2,918 | yes (0.3%) |
| Photostim trials | ~25% of trials | excluded from analyses | 19.6% | 20.0% | yes |
| Outcome distribution | 84% correct (control, non-early) | - | hit 68.7 / miss 16.5 / ignore 14.8 (all trials); 81.0% correct on control non-early trials | hit 68.5 / miss 16.7 / ignore 14.8 | yes |
| Early lick | "early lick trials excluded" | excluded | 11.4% | 11.6% | yes |
| Bin size | 40 ms window / 3.4 ms stride | same | - | 50 ms non-overlapping (task spec) | deviation required by the spec |
| Alignment | go cue | go cue | - | go cue | yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports **`Data format is valid, no errors or warnings.`** No errors and
no warnings remain. Two classes of warning appeared in earlier iterations and were **fixed, not ignored**:
1. *"all neural data is zero"* for whole blocks of trials -> caused by sessions in which the ephys
   recording covers only part of the behavioural session. Fixed with `observed_trial_mask()`
   (the units' `obs_intervals`), which removes 8 sessions' worth of unobserved trials.
2. A single remaining all-zero trial at the end of a truncated recording -> fixed by dropping trials whose
   extracted window contains no spikes at all.

### Check 2: Sanity checks against the raw NWB files
`/app/cache/sanity_checks.py` reloads `converted_data.pkl` and recomputes everything **from the NWB files
without importing the conversion code**, for 5 randomly chosen sessions:
| Check | Method | Result |
|---|---|---|
| neural firing rates | brute-force `np.sum((sp>=e_b)&(sp<e_{b+1}))/0.05` for 8 random (trial, neuron) pairs per session | PASS (`np.allclose`) |
| number of trials / neurons | independent reconstruction of the curation rules | PASS |
| input 0 (time from tone) | bin centre minus the last `sample_start_time` before the go cue | PASS |
| input 1 (photostim) | overlap of each bin with `photostim_start/stop_times` | PASS |
| n photostim trials | vs `trials/photostim_power != 'N/A'` | PASS (exact) |
| outputs 0-2 | recomputed from `outcome`, `trial_instruction`, `early_lick` | PASS |
| outputs constant within a trial | | PASS |
| output 3 (tongue class) | full-session recompute from the DLC stream + session percentiles | PASS |
| tongue class balance | 40/20/40 among visible bins | PASS ([0.40, 0.20, 0.40]) |
| brain_region_idx length, subject id | | PASS |

**0 of 70 checks failed.**

### Check 3: Reference code comparison
| Step | Reference | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `preprocessing_utils.loadmat` on per-probe DataJoint `.mat`, concatenating probes | `h5py` on the NWB file, whose `units` table already concatenates all probes of the session | equivalent |
| (b) neuron filtering | units with ephys **and** histology, intersected with the classifier `goodunits` list, empty CCF labels dropped | `classification == 'good'` **and** non-empty `anno_name` (the NWB columns holding exactly those two pieces of information) | same |
| (b) region/hemisphere | coarse CCF group, `ccf_x >= 5700 -> left` | same 14 groups, same midline, from the electrode CCF x of each unit | same |
| (c) temporal alignment | spikes and video aligned to the go cue | spike times minus `go_start_times`; video binned on the same absolute grid | same |
| (d) binning | `sliding_histogram`, count/bin width -> Hz | identical formula, 50 ms non-overlapping bins (spec) | same up to the bin size |
| (e) input construction | reference has no decoder inputs; it stores `task_stimulation` (on/off times relative to the go cue) and the sample epoch times | photostim on/off per bin from the same events; time from tone onset from `sample_start_times` | consistent |
| (f) output construction | `behavior_report` x `task_trial_type` -> lick direction; `behavior_early_report`; markers aligned to the go cue | `outcome` x `trial_instruction`; `early_lick`; tongue y from the DLC marker stream | same variables |
| trial curation | `get_regular_trial_mask`: no early lick, no auto/free water, no ignore, no photostim | auto/free water removed; early-lick, ignore and photostim trials **kept** | deliberate deviation (they are decoder targets/inputs); documented in Step 5 |
| neuron FR filter | `< 2 Hz` excluded in the encoding analyses | not applied | deliberate; the decoder uses all QC-good units |

### Check 4: Key statistics comparison
See the table in Step 9: sessions (173 = 173), mice (28 = 28), good units (69,453 vs 69,943, 0.7%),
per-area unit counts (all within 2% except ALM, explained), trials/session, correct rate, photostim
fraction and early-lick fraction all agree with the papers.

### Check 5: Edge cases handled
- Sessions with 0 good units -> skipped (1 file, so 173 of 174 sessions are converted; matches the paper).
- Sessions whose ephys covers only part of the behaviour -> unobserved trials removed (8 sessions).
- Trials at the very end of a recording with no spikes -> removed.
- Trials whose [-2.5, +1.5] s window extends beyond the trial boundaries -> the window is cut on the
  **absolute** time base, so neighbouring-trial data is used where it exists and the ~0.5 s inter-trial
  gaps simply contribute no spikes / 'tongue not visible'. (3.1% of trials have < 2.5 s between trial
  start and the go cue, and error trials are truncated ~0.8 s after the go cue.)
- Early-lick trials replay the sample epoch, giving several `sample_start_times` per trial -> the **last**
  sample start before the go cue is used as the tone onset; if none exists the nominal 1.85 s is used
  (this fallback was never triggered).
- Sessions with a second camera (`Camera3_side_*`) or extra markers -> the side camera used by the paper
  (`Camera0_side_...`) is preferred, with `Camera3` as a fallback.
- `photostim_power`/`onset`/`duration` are strings with `'N/A'` -> the event timestamps are used, with the
  trial-table strings as a fallback.
- All 173 sessions have >= 2 trials, as required.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` ->
`/app/train_decoder_full_out.txt` (GPU, 173 sessions, 89,544 trials, 69,453 neurons).

### Training Progress
- Loss decreasing: **Yes** — 200 epochs, training loss falls monotonically
  (epoch 100: 0.871, 150: 0.709, 200: 0.658). Test loss 0.661, i.e. essentially the same as the
  training loss -> no overfitting.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance | Notes |
|--------|-------------|--------|--------|--------|-------|
| choice (no lick / left / right) | 0.7109 | **0.6817** | 0.3333 | 2.05x | |
| outcome (ignore / miss / hit) | 0.6928 | **0.6614** | 0.3333 | 1.98x | |
| early_lick (no / yes) | 0.8008 | **0.7503** | 0.5000 | 1.50x | binary, so 1.5x chance is a large effect |
| tongue_y_position (3 levels + not visible) | 0.6825 | **0.6549** | 0.2500 | 2.62x | |

The accuracies are averaged over **all 80 time bins**, including the 2.5 s before the go cue during which
the choice/tongue information is only weakly present; see Step 12 for the time-resolved analysis.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Validation balanced acc | Chance | Ratio |
|---|---|---|---|
| choice | 0.682 | 0.333 | 2.05x |
| outcome | 0.661 | 0.333 | 1.98x |
| early_lick | 0.750 | 0.500 | 1.50x |
| tongue_y_position | 0.655 | 0.250 | 2.62x |
No output is at or below chance and none is below 1.5x chance (early lick is binary, so 1.5x is the
maximum possible ratio range of 0.5-1.0; 0.75 is halfway to perfect).

### Check 2: Accuracy comparison to the papers
The papers do not decode outcome, early lick or tongue position from neural activity, and they do not
report a single time-averaged accuracy for choice. The comparable published numbers are AUCs for
**choice**, so I reproduced that analysis on the converted data
(`/app/cache/timecourse_check.py`: per-time-bin PCA(16) + logistic regression, 5-fold CV, lick-left vs
lick-right trials only — the same algorithm as the reference `nested_cross_validation`):

| Epoch (time from go cue) | Paper (video-based choice decoding) | Converted data (neural, 4 example sessions) |
|---|---|---|
| sample epoch (~-2.0 s) | AUC ~0.5 ("at chance") | 0.42-0.57 |
| delay epoch (-1.0 to -0.2 s) | AUC 0.66 +- 0.12 s.d. | 0.55-0.75 |
| response epoch (+0.5 to +1.0 s) | "choice easily decodable" (AUC ~0.9) | **0.87-0.99** |

The converted neural data therefore shows exactly the expected time course: no choice information before
the instruction, growing selectivity through the delay, and near-perfect decoding after the go cue.
This is strong evidence that the temporal alignment (go cue at bin 50) and the trial labels are correct.
The time-averaged balanced accuracy reported by `train_decoder.py` (0.68) is necessarily lower than the
response-epoch value because it averages over the 50 pre-go-cue bins.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.711 | 0.682 | 1.04 |
| outcome | 0.693 | 0.661 | 1.05 |
| early_lick | 0.801 | 0.750 | 1.07 |
| tongue_y_position | 0.683 | 0.655 | 1.04 |
All ratios are < 1.1, far below the 1.5 threshold: no overfitting and no sign of data leakage
(test loss 0.661 vs training loss 0.658).

### Additional verification performed in this step
1. **Output variation**: no output is dominated by a single class
   (choice 15/43/42%, outcome 15/17/68%, early lick 88/12%, tongue 10/5/10/76%).
2. **Temporal alignment plot**: the processing plots show the raw spike raster and the binned rates on the
   same time axis with the go cue at 0; the population PSTH rises exactly at t = 0 and the photostim input
   ends before t = 0, as the protocol requires ('photoinhibition always ended before the Go cue').
3. **Neural filtering** re-checked against the reference: classifier-good units only (Step 10, Check 3).
4. Sample-level results (2 sessions) and full-dataset results are consistent (0.61->0.68 for choice as more
   data is added), as expected.

### Issues Found and Resolved (whole project)
- *Unobserved trials produce all-zero neural data* (8 sessions) -> `observed_trial_mask()` using
  `units/obs_intervals`. Re-ran conversion, verification, sanity checks and training.
- *One trial at the end of a truncated recording* -> zero-spike trials dropped.
- *`Zona incerta` assigned to the thalamus* -> moved to the hypothalamus per the Allen CCF, which brings the
  thalamic count from 13,386 to 12,922 (paper: 12,808). Re-ran conversion and verification.
- *`np.str_` region names* -> cast to `str`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format specification, key
      statistics, curation rules, decoder performance)
- [x] `cache/` folder created with `README_CACHE.md` describing every investigation script
- [x] All files organized

### Final file inventory
| File | Description |
|---|---|
| `CONVERSION_NOTES.md` | this document |
| `README.md` | user-facing documentation |
| `convert_data.py` | the conversion script (`--full`, `--sample`, `--show-processing`) |
| `converted_data.pkl` | full converted dataset, 11.89 GB, 173 sessions |
| `sample_data.pkl` | 2-session sample, 73 MB |
| `conversion_full_out.txt`, `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt`, `verification_sample_out.txt` | `train_decoder.py --verify-only` logs (both: no errors, no warnings) |
| `train_decoder_full_out.txt`, `train_decoder_sample_out.txt` | decoder training logs |
| `processing_SC015_*.png` | per-step processing plots for the two sample sessions |
| `sample_trials.png`, `predictions.png` | plots produced by `train_decoder.py --plot-samples` |
| `cache/` | investigation scripts, extracted paper text, sanity checks |

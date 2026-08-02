# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide map / decoder benchmark dataset from local `data/` plus reference papers/code
- **Date started**: 2026-03-24
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `dataarchitecture.pdf`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `methodpaper.pdf`
- `methods.txt`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Merges probes within a session, loads spikes, trials, and behavior streams, and packages metadata. |
| `load_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads spike sorting outputs and cluster metadata for a probe insertion via `SpikeSortingLoader`; can QC-filter if `qc` is passed, but reference caching calls it with `qc=None`. |
| `merge_probes` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Reindexes cluster IDs and concatenates/sorts spikes across probes to make a session-wide spike train. |
| `load_trials_and_mask` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Builds a boolean trial inclusion mask based on reaction time, trial length, missing key events, and no-choice trials. |
| `list_brain_regions` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Maps cluster acronyms to Beryl atlas acronyms and enumerates regions used for decoding. |
| `select_brain_regions` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Selects cluster IDs belonging to the requested region set. |
| `bin_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Converts session-wide spikes to trial-aligned binned spike-count tensors using `stimOn_times` and the configured time window. |
| `get_spike_data_per_interval` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Performs multi-bin spike counting per interval; uses `ceil(interval_len / binsize)` bins. |
| `load_target_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads continuous behavior streams such as wheel speed and whisker motion energy from session objects. |
| `get_behavior_per_interval` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Trial-aligns and linearly interpolates continuous behavior signals to the same interval and nominal bin count as neural data. |
| `bin_behaviors` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Constructs per-trial categorical variables (`choice`, `block`, `reward`, `contrast`) and time-varying behavior arrays (`wheel-speed`, `whisker-motion-energy`, etc.). |
| `align_spike_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Drops trials with missing behavior / excluded mask and reshapes behavior arrays so trial counts match spike arrays. |
| `create_dataset` | `code/code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Serializes per-trial binned spikes plus behavior and cluster metadata into the cached dataset structure. |
| `SingleSessionDataset` | `code/code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Loads cached trials, reconstructs sparse spikes, standardizes spikes, one-hot encodes classification targets, and standardizes regression targets. |
| `0_data_caching.py` main loop | `code/code_zhang2025/src/0_data_caching.py` | LOADING | Reference end-to-end preprocessing entrypoint for selected sessions. |

### Notes
- Relevant reference repository is `code/code_zhang2025`; `code/ibllib` is the library dependency used to access IBL data.
- The reference pipeline is electrophysiology, not imaging. No delta-F/F computation is involved.
- Reference caching parameters in `0_data_caching.py`:
  - `align_time='stimOn_times'`
  - `time_window=(-0.5, 1.5)` seconds
  - `interval_len=2`
  - `binsize=0.02` seconds
- Therefore the nominal neural/behavior trial window is 2 s around stimulus onset, with 20 ms bins, yielding 100 bins.
- Trial curation in `load_trials_and_mask` uses:
  - reaction time between 0.08 s and 2.0 s
  - trial duration (`feedback_times - goCue_times`) <= 10 s
  - non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
  - exclude no-choice trials (`choice == 0`)
  - do not exclude unbiased block by default
- Spike processing:
  - probes are merged within session
  - all clusters are loaded during caching (`qc=None`)
  - cluster QC is still recorded in metadata as `good_clusters = (label >= 1)`
  - no firing-rate smoothing; data are spike counts per bin
- Behavior processing:
  - `choice`, `block` (`probabilityLeft`), `reward`, and signed `contrast` are per-trial values
  - `wheel-speed` is `abs(wheel velocity)`
  - `whisker-motion-energy` prefers left camera and falls back to right camera if needed
  - continuous streams are linearly interpolated within each trial-aligned window
  - interpolation sample points are `linspace(interval_beg + binsize, interval_end, n_bins)`
- Data orientation in the cached dataset is `(trial, time, neuron)` for spikes; decoder target format will need per-trial matrices transposed to `(neuron, time)`.
- Potential discrepancy to revisit later: `align_spike_behavior` uses Python list `and` semantics and overwrites `beh_mask` within the loop, so the effective trial filtering logic may only reflect the last behavior plus `trials_mask`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Main raw data cache is under `data/one_cache/`.
- Three registry snapshots are present:
  - `data/one_cache/2022_Q4_IBL_et_al_BWM/`
  - `data/one_cache/2025_Q3_IBL_et_al_BWM/`
  - `data/one_cache/Brainwidemap/`
- Each registry snapshot contains `sessions.pqt`, `datasets.pqt`, `cache_info.json`, and `QC.json`.
- Raw session folders are organized as:
  - `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/`
  - each session contains `alf/` with trial, wheel, camera, and probe-specific files
- Representative `alf/` contents observed:
  - trials: `_ibl_trials.table.pqt`, `_ibl_trials.goCueTrigger_times.npy`, `_ibl_trials.intervals_bpod.npy`
  - wheel: `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`
  - camera timing: `_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy`
  - whisker motion energy: `leftCamera.ROIMotionEnergy.npy`, `rightCamera.ROIMotionEnergy.npy`
  - spikes/clusters per probe in `alf/probeXX/pykilosort/...`:
    - `spikes.times.npy`, `spikes.clusters.npy`, `spikes.depths.npy`
    - `clusters.metrics.pqt`, `clusters.channels.npy`, `clusters.depths.npy`, `clusters.uuids.csv`
    - `channels.*`, `electrodeSites.*`
- The trial table schema was consistent across sampled sessions:
  - `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `probabilityLeft`, `feedback_times`, `feedbackType`, `rewardVolume`, `firstMovement_times`, `intervals_0`, `intervals_1`
- Example raw array shapes from one session (`angelakilab/NYU-11/2020-02-18/001`):
  - wheel timestamps / position: `(740931,)`
  - left camera times / motion energy: `(307375,)`
  - right camera times / motion energy: `(769676,)`
  - spike times / clusters: `(20667745,)`
  - clusters on one probe: `898`
- Local cache coverage summary from session folders:
  - 461 session folders found
  - all 461 have trial tables and wheel files
  - 441 have at least one whisker motion-energy stream (left or right)
  - 240 sessions have 2 probes, 221 have 1 probe
- Registry summary from snapshot metadata:
  - `2022_Q4_IBL_et_al_BWM`: 354 sessions in `sessions.pqt`
  - `2025_Q3_IBL_et_al_BWM`: 459 sessions in `sessions.pqt`
  - `Brainwidemap`: 480 sessions in `sessions.pqt`
- Important discrepancy to investigate later:
  - raw session folders on disk: 461 sessions / 141 subjects
  - `Brainwidemap/sessions.pqt`: 480 sessions / 143 subjects

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 622,377 raw clusters across session folders (sum of `clusters.metrics.pqt` row counts) |
| Neurons / session | mean 1350.06, median 1295, min 135, max 3140 |
| Subjects | 141 subjects present in raw session folders; 143 subjects in `Brainwidemap/sessions.pqt` |
| Sessions / subject | mean 3.27 using 461 raw session folders / 141 subjects |
| Trials (total) | 297,505 rows across `_ibl_trials.table.pqt` files in raw session folders |
| Trials / session | mean 645.35, median 601, min 401, max 1525 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 units in public release | “After applying these criteria, a total of 459 sessions, 699 insertions and 621,733 neurons remained...” |
| Neurons / session | ~1355 raw units/session in release (621,733 / 459) | derived from paper totals above |
| Subjects | 139 mice | “We trained 139 mice...” / “The data from n = 139 adult mice...” |
| Sessions / subject | ~3.30 in public release (459 / 139) | derived from paper totals above |
| Trials (total) | not explicitly stated in papers | sessions retained for analyses had at least 400 trials; release sessions had at least 250 |
| Trials / session | at least 250 for release; at least 400 for further analyses in main paper | “Sessions were included in the data release if the mice performed at least 250 trials...” and “Only sessions with at least 400 trials were retained for further analyses.” |
| Neural data time bin | 20 ms for choice / dynamic variables in methods paper; 50 ms for prior | “For each trial... 20-ms bins, producing T = 100”; “For prior... 50-ms non-overlapping time bins”; “dynamic behaviors... non-overlapping 20 ms bins” |
| Behavior data time bin | raw wheel/whisker sampled at camera / wheel rates; method paper describes wheel/whisker as time-varying and binned with 20 ms neural bins | “wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz” and dynamic behaviors decoded with 20 ms bins |
| Reward rate | not explicitly stated | not directly reported in scanned text |
| Unbiased block length | 90 trials | “For the first 90 trials, the stimulus appears randomly...” |
| Biased block probability | `probabilityLeft` 0.2 / 0.8 after unbiased block | “20:80% (right block) or 80:20% (left block)” |
| Block length | 20–100 trials, empirical mean 51 | “Blocks lasted for between 20 and 100 trials... empirical mean of 51 trials” |
| Brain regions in method paper | 270 brain regions | “433 IBL sessions... covering 270 brain regions” |
| Sessions in method paper | 433 IBL sessions | “We apply our models to 433 IBL sessions...” |
| Public release sessions in data paper | 459 sessions | “After applying these criteria, a total of 459 sessions...” |
| Well-isolated neurons | 75,708 | “Out of the 621,733 units collected, 75,708 were considered well-isolated neurons.” |


### Processing Details
- Task structure from paper:
  - visual stimulus left/right; mouse reports by wheel turn
  - 90 unbiased trials at 50:50, then alternating biased blocks with `probabilityLeft` 0.2 or 0.8
  - 0% contrast trials follow block prior and still require a left/right response
- Temporal alignment from method paper:
  - choice: align to stimulus onset, use neural activity from -0.5 s to +1.5 s
  - prior: align to stimulus onset, use neural activity from -0.6 s to -0.1 s
  - wheel speed / whisker motion energy: align to first movement onset, decode from alignment event to +1.0 s
- Temporal binning from method paper:
  - 20 ms bins for choice and dynamic behaviors in the main formulation
  - 50 ms bins specifically stated for prior
- Additional decoding details from data paper:
  - wheel speed / velocity decoded in non-overlapping 20 ms bins from -0.2 s to +1.0 s around first movement, with a causal window of W = 10 bins for regression
  - binary variables evaluated with balanced accuracy
  - continuous variables evaluated with R2
- Motion energy definition from data paper:
  - mean absolute pixel difference between adjacent frames in a whisker-pad bounding box anchored between nose tip and eye
- Data architecture white paper:
  - raw data are stored in ALF / ONE-style dataset types
  - session-relative files and revisions are expected; this matches the versioned `alf/#...#/` files observed locally

### Curation Steps

**Neuron curation rules**:
- Public release kept 621,733 units including multineuron activity.
- Main data-paper analyses further defined “well-isolated neurons” using RIGOR single-unit QC:
  - amplitude > 50 uV
  - noise cut-off < 20 uV
  - refractory-period violation criterion
- Final main analyses further restricted to grey-matter regions with at least 5 well-isolated neurons per session and at least 2 such sessions.

**Trial curation rules**:
- Release / analysis trial exclusions in data paper:
  - exclude trials missing `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, or `firstMovement_times`
  - exclude trials with reaction time (`firstMovement_times - stimOn_times`) outside 0.08–2.00 s
- Sessions in public release had to satisfy:
  - at least 250 trials
  - at least 90% correct on 100% contrast trials for both left and right blocks
  - at least 3 incorrect-choice trials after exclusions
  - hardware QC threshold
- Main paper also notes only sessions with at least 400 trials were retained for further analyses.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | multi-session behavioral-state model improves over baseline; example figure shows choice AUC improving from about 0.66 baseline to about 0.79 on an example session |
| Prior | LG-AR1 improves over baseline; example figure shows prior correlation improving from about 0.05 baseline to about 0.34 multi-session and about 0.65 oracle on an example session |
| Wheel speed | paper evaluates with R2; figures indicate multi-session RRR outperforms baseline / single-session models |
| Whisker motion energy | paper evaluates with R2; figures indicate multi-session RRR outperforms baseline / single-session models |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session universe | `code/code_zhang2025/data/bwm_release.csv` contains 699 probe insertions / 459 unique sessions / 139 subjects; `repro_ephys_release.txt` is only a 40-session example subset | local raw session folders: 461 sessions / 141 subjects; `Brainwidemap/sessions.pqt`: 480 sessions / 143 subjects; `2025_Q3_IBL_et_al_BWM/sessions.pqt`: 459 sessions / 139 subjects | data paper public release: 459 sessions / 699 insertions / 139 mice; method paper experiments on 433 sessions | Treat the 459-session / 699-insertion BWM release as authoritative for conversion because it matches both `bwm_release.csv` and the data paper release counts. Ignore extra/newer sessions in local registry snapshots unless needed only for file lookup. |
| Probe counts | code freeze implies 219 one-probe and 240 two-probe sessions | local folders show 221 one-probe and 240 two-probe session directories | data paper: “most recordings using 2 simultaneous probe insertions” and total 699 insertions | Use `bwm_release.csv` / probe metadata to define valid insertions for the target release; local folders appear to include extra sessions outside the frozen release. |
| Method-paper session count | code examples do not mention 433 explicitly | local cache has enough sessions to support a large subset but not an exact 433 count from folders alone | method paper: 433 IBL sessions across 270 brain regions | Interpret 433 as a downstream analysis subset of the 459-session release after additional modality / analysis availability constraints. Do not use 433 as the raw starting session count. |
| Neuron QC policy | `load_spiking_data(..., qc=None)` loads all clusters; metadata stores `good_clusters = label >= 1` but caching does not filter on it | cluster metrics files contain `label`, `firing_rate`, `noise_cutoff`, etc.; aggregating the frozen 699 insertions gives 621,733 raw units and exactly 75,708 units with `label >= 1` | data paper main analyses emphasize 75,708 well-isolated neurons; method paper text says “using all neurons, sorted by Kilosort 2.5, from each session” | Use `label >= 1` units for the converted decoder dataset. This exactly reproduces the paper’s well-isolated-neuron count and keeps the dense per-trial format computationally feasible, while still deriving the mask from the same cluster metrics bundled with the reference release. Preserve raw QC metadata in `metadata`. |
| Trial filtering | code excludes missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; RT outside 0.08–2.0 s; no-choice trials; plus `max_trial_len=10.0` | trial tables contain all required columns and some no-choice / NaN rows | data paper excludes the same missing events and RT 0.08–2.00 s; release sessions need >=250 trials; main analyses often >=400 trials | Mandatory filters are the shared missing-event and RT filters. Keep `exclude_nochoice=True` and `max_trial_len=10.0` to remain consistent with the provided code path, while noting these are stricter than the minimal text summary. |
| Temporal alignment for `choice` | code cache aligns to `stimOn_times` with window `(-0.5, 1.5)` and 20 ms bins | raw data contain `stimOn_times` and `firstMovement_times` | method paper also aligns choice to stimulus onset | No discrepancy. Use stimulus-onset alignment for choice. |
| Temporal alignment for `prior` | provided code cache does not special-case prior and would align to `stimOn_times` with the same 2 s / 20 ms window as other targets | raw data contain `probabilityLeft` per trial | method paper says prior uses stimulus onset but a different window (`-0.6` to `-0.1` s) and 50 ms bins | For this task, decoder inputs/outputs must share a common stimulus-onset-aligned trial grid. Use a common 20 ms stimulus-onset grid for the converted dataset, and document that this is a task-imposed simplification relative to the method paper’s prior-specific window. |
| Temporal alignment for `wheel-speed` and `whisker-motion-energy` | provided code cache aligns all behaviors to `stimOn_times` | raw data provide continuous wheel and whisker streams plus `stimOn_times` and `firstMovement_times` | method paper says dynamic behaviors should align to `firstMovement_times` and run to +1 s | User explicitly requires “Temporally align based on stimulus onset.” Therefore, use stimulus-onset alignment for dynamic outputs as a deliberate task override, while keeping the reference code’s binning, interpolation style, and trial masking wherever possible. |
| Binning of dynamic behavior | code uses 20 ms bins and linear interpolation inside each interval | raw wheel / motion-energy streams have irregular session-wide sampling (wheel) or camera-rate sampling (60 Hz / 150 Hz) | method paper dynamic behavior bins are 20 ms | No substantive discrepancy. Use 20 ms bins with trial-wise interpolation. |
| Binning of prior | code examples / cache infrastructure assume 20 ms grids; prior notebooks treat prior as scalar over trials | `probabilityLeft` is available per trial; no continuous prior trace in raw data | method paper prior bins neural data at 50 ms and decodes a scalar per trial | Represent prior as per-trial categorical output on the common trial grid in the converted dataset, duplicating across time if needed for shape consistency only if required later. |
| Motion-energy source | code prefers left whisker motion energy and falls back to right camera if missing | 434 local sessions have left motion energy; 421 have right; 441 have at least one side | data paper computes whisker-pad motion energy from left and right videos | Prefer left camera when available, else right, matching the provided code and the local coverage pattern. |
| Data organization | code assumes ONE / ALF loading with revisions and dataset types | local files are indeed stored under revisioned `alf/#...#/` paths and `probeXX/pykilosort/...` | data architecture paper describes ONE / ALF dataset-type conventions | Follow ALF / ONE naming and revision handling already present in local cache. |

Final understanding:
- Start from the frozen 459-session / 699-insertion BWM release, not from all locally visible session folders.
- Apply the shared trial-quality filters from the papers plus the provided code’s `exclude_nochoice=True` and `max_trial_len=10.0`.
- Use `label >= 1` well-isolated units because this reproduces the paper’s 75,708-neuron statistic exactly and is the release-consistent dense representation that remains computationally tractable for the provided decoder.
- Use a common 20 ms stimulus-onset-aligned grid for the converted dataset because the user explicitly requires stimulus-onset alignment and the target decoder needs common trial shapes across neural, input, and output streams.
- Treat method-paper target-specific alignment differences (especially dynamic behaviors and prior) as important reference context, but as overridden where they conflict with the user’s explicit decoder specification.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters`, `clusters.metrics.label` from all probes in a frozen-release session | `neural` | merge probes by session after filtering to clusters with `label >= 1`, bin spike counts on a common stimulus-onset window `[-0.5, 1.5]` s with 20 ms bins, transpose to `(n_neurons, 100)` per trial | `prepare_data`, `merge_probes`, `bin_spiking_data`, `get_spike_data_per_interval` | `label >= 1` reproduces the paper’s 75,708 well-isolated units exactly |
| common trial grid relative to `stimOn_times` | `input[0]` = `time_since_stimulus_onset_s` | 100-length vector of relative times on the same grid as aligned behavior/neural bins; use a common vector for every trial | reference code uses `time_window` and `binsize`; behavior interpolation grid from `get_behavior_per_interval` | represent as time-varying continuous input |
| `probabilityLeft` block structure in raw trial table | `input[1]` = `trial_number_in_block` | compute block index from original unfiltered session trial sequence; reset to 1 whenever `probabilityLeft` changes; then repeat the scalar across all 100 bins for each kept trial | custom logic consistent with raw trial table semantics | unbiased 0.5 block counts as its own block |
| raw `choice` in trial table (`-1`, `1`) after filtering | `output[0]` = `choice` | map left `1 -> 0`, right `-1 -> 1`; repeat across all 100 bins for a common 2D output shape | `load_trials_and_mask`, `bin_behaviors` | verified directly from high-contrast correct trials; no-choice trials are removed before mapping |
| raw `probabilityLeft` in trial table (`0.2`, `0.5`, `0.8`) | `output[1]` = `prior_probability_of_left` | map `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`; repeat across all 100 bins | `bin_behaviors` for `block` source; task-specific remapping | categorical per trial |
| wheel trace from `_ibl_wheel.*` via ONE/SessionLoader-equivalent logic | `output[2]` = `wheel_speed_bin` | compute wheel speed as absolute velocity; align/interpolate to the common stimulus-onset grid; discretize using global 3-bin thresholds; store category index `0/1/2` per time bin | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | global thresholds shared across all sessions/trials for consistent categories |
| whisker motion energy from left camera, else right camera | `output[3]` = `whisker_motion_energy_bin` | load left whisker motion energy if available, else right; align/interpolate to common stimulus-onset grid; discretize with global 3-bin thresholds; store `0/1/2` per time bin | `load_target_behavior`, `bin_behaviors`, `get_behavior_per_interval` | do not mix left/right within a trial; session uses preferred available side |
| subject IDs from frozen release metadata | `subjects`, `subject_idx` | unique sorted subject list and per-session index | `bwm_release.csv` session metadata | order matches session order in converted dataset |
| cluster acronyms from probe cluster metadata | `brain_regions`, `brain_region_idx` | map each neuron acronym to Beryl acronym, build unique region list, store integer region index per neuron | `list_brain_regions` / `BrainRegions().acronym2acronym(..., mapping='Beryl')` | Beryl mapping matches reference region aggregation used in code |
| release/session metadata | `metadata` | include task description, 20 ms bin size, stimulus-onset alignment, offsets `-0.5/+1.5`, release source, discretization thresholds, modality notes | custom | keep enough detail for reproducibility |

### Key Decisions
1. **Use the 459-session frozen BWM release as the source session set**: This matches both the data paper public release and `code/code_zhang2025/data/bwm_release.csv`.
1. **Require local existence of trials, wheel, and at least one whisker motion-energy stream**: All 459 frozen sessions exist locally; 439 have at least one whisker motion-energy stream, so sessions lacking whisker motion energy will be excluded because that output is mandatory for this decoder task.
1. **Use a single stimulus-onset-aligned 20 ms grid for every variable**: This satisfies the user’s explicit task requirement and simplifies the validator/decoder interface.
1. **Represent both inputs as 2D arrays of shape `(2, 100)`**: `time_since_stimulus_onset_s` varies over time; `trial_number_in_block` is repeated across time for each trial.
1. **Represent all outputs as 2D arrays of shape `(4, 100)`**: static outputs (`choice`, `prior`) are repeated across time so all outputs share a common structure with the dynamic outputs.
1. **Filter neurons to `label >= 1`, while preserving per-neuron QC metadata and Beryl region labels**: This matches the public-release well-isolated-neuron count of 75,708 exactly and keeps the dense pickle size compatible with the provided decoder.
1. **Compute `trial_number_in_block` on the original trial table before filtering**: This preserves the actual behavioural position within a block rather than renumbering after trial exclusion.
1. **Use left whisker motion energy when present, else right**: This matches the provided code and maximizes session retention.
1. **Discretize wheel speed and whisker motion energy with global 3-bin thresholds**: session-specific thresholds would make class labels inconsistent across sessions; global thresholds keep categories comparable.
1. **Drop any trial lacking full valid coverage of neural, wheel, or whisker data on the common window**: no padding or fabricated values.

### Planned Sanity Checks
- [ ] Neural spot-check: for a chosen session/trial/neuron, recompute binned spike counts directly from raw `spikes.times` / `spikes.clusters` and verify equality with `np.allclose()`.
- [ ] Input spot-check: verify the converted `time_since_stimulus_onset_s` row equals the planned 100-bin relative-time vector and that `trial_number_in_block` matches direct computation from the raw trial table.
- [ ] Output spot-check (static): verify converted `choice` and `prior_probability_of_left` categories match raw `choice` and `probabilityLeft` on selected trials.
- [ ] Output spot-check (dynamic): verify a selected trial’s wheel-speed and whisker-motion-energy category sequences match direct raw interpolation followed by bin assignment using the saved global thresholds.
- [ ] Session-count sanity: compare number of included sessions and trials after mandatory modality/trial filtering against the 459-session release and the 433-session method-paper analysis subset.
- [ ] Neuron-count sanity: compare converted neuron counts both before and after optional QC summaries against the raw cluster counts and the paper’s well-isolated-neuron count.

---

## Step 6: Script Development
**Status**: COMPLETE

- Installed `iblatlas` successfully so Beryl region mapping can match the reference code.
- Direct reuse of `brainbox.io.one.SessionLoader` is not viable in this environment because `brainbox.io.one` imports the unavailable `neuropixel` package, and `ONE.load_object(...)` against the local cache hits `.rest` permission issues.
- Conversion will therefore use direct ALF file loading plus the bundled `brainbox.behavior.wheel.interpolate_position` / `velocity_filtered` helpers and the bundled `brainbox.population.decode` logic where possible.
- Verified that filtering clusters by `label >= 1` across the 699 frozen-release insertions yields exactly 75,708 units, matching the data paper.
- Implemented `/app/convert_data.py` as a two-pass converter:
  - pass 1 loads trials + wheel + whisker, applies the reference trial mask plus behavior-coverage mask, and computes global dynamic-output tertile thresholds
  - pass 2 reloads good units (`label >= 1`), bins stimulus-aligned spikes into 20 ms bins, constructs decoder inputs/outputs, and writes the pickle
- Implemented direct Beryl region mapping from `channels.brainLocationIds_ccf_2017.npy` via `iblatlas.regions.BrainRegions`.
- Implemented reference-style video timestamp repair where camera timestamps longer than motion-energy arrays are trimmed from the front.
- Smoke test completed successfully:
  - command: `python3 -u /app/convert_data.py /app/_step6_smoke.pkl --sample`
  - result: 2 sessions, 651 trials, 340 well-isolated neurons, no runtime errors

Code inefficiencies identified:
- Repeated recursive path discovery across revisioned ALF files would be slow if done inside inner trial loops.
- Dense float32 storage would be too large for the full release.
- Initial full-run profiling showed the real bottleneck was spike-stream reload and dtype copying in pass 2, not the trial-binning code.

Code speedups added:
- Vectorized per-trial spike binning via `np.searchsorted` + flattened `np.bincount`.
- Low-precision storage for neural arrays (`float16`) to keep the dense pickle tractable while remaining compatible with the decoder validator.
- Behavior interpolation performed only once per session in pass 1; neural binning deferred until after global discretization thresholds are known.
- Session-level parallelism for full runs via `ThreadPoolExecutor`; sample / plotting runs stay single-worker for simpler debugging.
- BLAS/OpenMP thread counts are capped at 1 inside the converter so session-level parallelism is not drowned out by nested threading.
- Pass-2 spike loading now uses ALF-native cluster indexing, memory-mapped spike arrays, and no-copy dtype handling where possible; this removed the largest full-array copies.
- Merged spike times no longer use stable sort because within-timestamp probe ordering is irrelevant for binned counts.
- Full-run worker cap increased to 32 after confirming the machine has 64 CPUs and ample RAM.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 340 well-isolated neurons (`label >= 1`) |
| Neurons / session | [76, 264] |
| Subjects | 1 |
| Sessions / subject | [2] |
| Trials (total) | 651 |
| Trials / session | [407, 244] |
| `time_since_stimulus_onset_s` range | [-0.48, 1.50] |
| `trial_number_in_block` range | [1, 90] |
| `choice` distribution | [0.517665, 0.482335] |
| `prior_probability_of_left` distribution | [0.477727, 0.161290, 0.360983] |
| `wheel_speed_bin` distribution | [0.333333, 0.333333, 0.333333] |
| `whisker_motion_energy_bin` distribution | [0.333333, 0.333333, 0.333333] |

### Processing Plots Review
- Generated:
  - `processing_6713a4a7-faed-4df2-acab-ee4e63326f8d.png`
  - `processing_56956777-dca5-468c-87cb-78150432cc57.png`
- No obvious anomalies in the sample plots:
  - neural activity is stimulus-aligned with the expected 100-bin window
  - wheel speed and whisker motion energy interpolate smoothly onto the common trial grid
  - discretized wheel / whisker traces visually track the continuous traces
  - static outputs (`choice`, `prior`) and `trial_number_in_block` are consistent with the plotted trial annotations
- Verification script reported no format errors or warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Pass-2 spike loading switched to memory-mapped / no-copy reads and native cluster indexing | heavy-session payload build dropped from ~28.8 s to ~12.4 s; light-session payload build dropped from ~3.1 s to ~0.3 s |
| Session-level full-run parallelism (`ThreadPoolExecutor`) plus BLAS thread caps | avoids nested-thread oversubscription and lets build throughput scale with available CPUs |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample pass 1 (2 sessions) | ~0.42 s / session | N/A |
| Sample pass 2 build before spike-load optimization (2 sessions) | work-scaled; 6.18 s for 95,348 trial-neuron units | N/A |
| Sample pass 2 build after spike-load optimization (2 sessions) | 1.60 s total for the same 95,348 trial-neuron units | N/A |
| First full-run pass 1 measurement (16 workers) | N/A | 104.33 s for all 459 release sessions |
| First full-run build measurement before final optimization (16 workers) | N/A | projected ~20 min, so the run was stopped per instructions and optimized further |
| Revised expectation after spike-load optimization + 32 workers | N/A | expected to be comfortably below the previous ~20 min build projection; exact full-run time to be measured in Step 9 |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Note: initial GPU run failed with `torch.OutOfMemoryError` on a 1.64 GiB GPU, so the required sample training was re-run successfully with `--cpu` as allowed by the task instructions.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `choice` | 0.6226 | 0.5424 |
| `prior_probability_of_left` | 0.7109 | 0.6718 |
| `wheel_speed_bin` | 0.5895 | 0.5834 |
| `whisker_motion_energy_bin` | 0.6018 | 0.6035 |

- Loss decreased monotonically over the logged epochs:
  - epoch 1: 1.718008
  - epoch 50: 0.885866
  - epoch 100: 0.769632
  - epoch 150: 0.745233
  - epoch 200: 0.735164
- All validation balanced accuracies exceeded uniform chance:
  - `choice`: 0.5424 vs 0.5000
  - `prior_probability_of_left`: 0.6718 vs 0.3333
  - `wheel_speed_bin`: 0.5834 vs 0.3333
  - `whisker_motion_energy_bin`: 0.6035 vs 0.3333

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.2G
- `verification_full_out.txt`: created

Notes:
- Full conversion command completed successfully:
  - `python3 -u convert_data.py converted_data.pkl --full 2>&1 | tee conversion_full_out.txt`
- Full verification command completed successfully:
  - `python3 -u train_decoder.py converted_data.pkl --verify-only 2>&1 | tee verification_full_out.txt`
- Measured wall times from the final optimized run:
  - pass 1: 144.12 s
  - pass 2 build: 455.65 s
  - total conversion (before pickle write overhead): ~9.997 min
- `train_decoder.py --verify-only` reported no structural errors. It reported 16 warnings for all-zero neural windows; these are investigated in Step 10.

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Well-isolated neurons (total) | 75,708 | `label >= 1` in BWM release cache | 75,708 across 459-session release after exact aggregation | 72,757 | Expected subset: missing whisker streams remove 21 release sessions |
| Mean neurons/session | ~165.0 if 75,708 / 459 | reference cache uses same `label >= 1` QC | 165.0 over full 459-session release; 166.11 over 438 whisker-available sessions | 166.11 | Yes for whisker-available subset |
| Subjects | 139 mice in public release | 139 in `bwm_release.csv` | 139 in frozen release CSV | 135 | Expected subset: 4 subjects lost because all their eligible sessions are among the 21 excluded |
| Sessions | 459 public-release sessions | 459 rows grouped by `eid` in release CSV | 459 in frozen release CSV | 438 | Expected subset: 20 missing whisker/motion-energy streams + 1 session with no good units or <2 valid trials |
| Trials (total) | not explicitly stated; sessions retained for analyses had >=400 trials | trial mask + behavior coverage determines usable trials | 186,261 kept trials across the 438 included sessions after applying the final mask | 186,261 | Yes |
| Trials/session (mean) | not explicitly stated | determined by trial mask | 425.25 over included sessions | 425.25 | Yes |
| `time_since_stimulus_onset_s` range | 20 ms bins over [-0.5, 1.5] s around stimulus onset | `time_window=(-0.5, 1.5)`, `binsize=0.02`, 100 bins | derived from aligned-bin construction | [-0.48, 1.50] | Yes |
| `trial_number_in_block` range | first block 90 trials; later biased blocks 20-100 trials | block counter derived from `probabilityLeft` changes | [1, 99] over included sessions | [1, 99] | Yes |
| `choice` distribution | not explicitly stated | binary left/right after no-choice exclusion | [0.508609, 0.491391] | [0.508609, 0.491391] | Yes |
| `prior_probability_of_left` distribution | first 90 unbiased, then 0.2 / 0.8 biased blocks | mapped from `probabilityLeft` to {0.2,0.5,0.8} | [0.418751, 0.140604, 0.440645] | [0.418751, 0.140604, 0.440645] | Yes |
| `wheel_speed_bin` distribution | task-specific discretization required | tertile binning over kept timepoints | [0.333333, 0.333333, 0.333333] | [0.333333, 0.333333, 0.333333] | Yes |
| `whisker_motion_energy_bin` distribution | task-specific discretization required | tertile binning over kept timepoints | [0.333333, 0.333333, 0.333333] | [0.333333, 0.333333, 0.333333] | Yes |

Additional notes:
- Excluded-session reasons from converted metadata:
  - `missing_required_stream`: 20 sessions
  - `no_good_units_or_too_few_valid_trials`: 1 session
- The subset mismatch versus the 459-session / 139-mouse release is required by the decoder task because whisker motion energy is a mandatory output and the local cache lacks that stream for 20 release sessions.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification:
   - `verification_full_out.txt` reported 16 warnings of the form “all neural data is zero”.
   - Direct review showed these warnings are restricted to 3 sessions:
     - session index 37 / eid `b182b754-3c3e-4942-8144-6ee790926b58`: 1 trial
     - session index 334 / eid `195443eb-08e9-4a18-a7e1-d105b2ce1429`: 12 trials
     - session index 364 / eid `8c2f7f4d-7346-42a4-a715-4d37a5208535`: 3 trials
   - Checked every flagged trial against the raw good-unit spike times in its stimulus-aligned window; all 16 have exactly 0 spikes in `[-0.5, 1.5]` s and the converted trial matrices are correspondingly all zero.
   - Resolution: preserve these trials. Dropping them would invent an extra curation rule not present in the reference processing, and the warnings reflect true zero-spike windows rather than a conversion bug.
2. Construct sanity checks:
   - Added standalone raw-data reconstruction script `/app/_raw_sanity_check.py` (independent of `convert_data.py`; it loads ALF files directly and uses the bundled wheel helper).
   - For session `6713a4a7-faed-4df2-acab-ee4e63326f8d`, trial 0:
     - neural matrix reconstructed from raw `spikes.times.npy` / `spikes.clusters.npy` with `label >= 1` filtering: `np.allclose(...) == True`
     - input matrix reconstructed from raw trial table + raw wheel / whisker interpolation coverage: `np.allclose(...) == True`
     - output matrix reconstructed from raw choice / prior plus raw wheel / whisker traces and stored global tertile edges: `np.allclose(...) == True`
   - Spot values from the independent script:
     - `neural[5, 10]`: `0.0` vs converted `0.0`
     - `input[1, 10]` (trial number in block): `10.0` vs converted `10.0`
     - `output[0, 10]` (choice): `0` vs converted `0`
3. Reference code comparison:
   - Data loading:
     - reference: `query_and_load_data`, `load_data_from_pid`, and `preprocess_ephys` / `preprocess_widefield`
     - conversion: direct ALF readers in `load_trials_table`, `load_wheel_speed`, `load_whisker_motion_energy`, `load_good_spikes_and_regions`
     - comparison result: same source streams are used; direct ALF loading replaces `ONE` / `SessionLoader` only because those paths are broken in this environment
   - Neuron / trial filtering:
     - reference: `trials_mask` logic and behavioral coverage masks; ephys QC based on `label >= 1`
     - conversion: `compute_trial_mask`, interpolation coverage masks, and `clusters.metrics.label >= 1`
     - comparison result: matched; exact release-wide count of 75,708 well-isolated units was reproduced before subsetting to whisker-available sessions
   - Temporal alignment and binning:
     - reference: stimulus-onset alignment with `time_window=(-0.5, 1.5)` and `binsize=0.02` for 100 bins
     - conversion: `ALIGN_EVENT='stimOn_times'`, `TIME_WINDOW=(-0.5, 1.5)`, `BINSIZE=0.02`, `NBINS=100`
     - comparison result: matched
   - Input / output construction:
     - reference code decodes task and behavior variables on the aligned neural grid
     - conversion constructs decoder-specific inputs (`time_since_stimulus_onset_s`, `trial_number_in_block`) and outputs (`choice`, `prior_probability_of_left`, `wheel_speed_bin`, `whisker_motion_energy_bin`) on that same grid
     - comparison result: consistent, with the only intentional differences being those required by the Decoder Task specification
4. Key statistics comparison:
   - Converted dataset matches the raw post-mask counts exactly for the included sessions:
     - 438 sessions
     - 135 subjects
     - 186,261 trials
     - 72,757 well-isolated neurons
   - Mismatch versus the 459-session / 139-mouse release is fully explained by the mandatory whisker-motion-energy output:
     - 20 release sessions are missing the required whisker stream in the local cache
     - 1 additional session has no good units or fewer than 2 valid trials after the full mask
   - Output ranges and distributions are sensible:
     - `choice`: balanced near 50:50
     - `prior_probability_of_left`: non-uniform because many sessions spend more time in biased blocks than in the initial 90-trial unbiased block
     - dynamic outputs are exactly tertiled by construction
5. Edge-case review:
   - Confirmed camera timestamps longer than motion-energy arrays are trimmed from the front before interpolation, matching the raw-data irregularity noted in the data architecture.
   - Confirmed `trial_number_in_block` can legitimately reach 99 because biased blocks last 20-100 trials; the sample-only max of 90 was not representative of the full dataset.
   - Confirmed zero-spike trials are legitimate raw-data edge cases rather than off-by-one errors at trial boundaries.

### Issues Found and Resolved
- Initial full-run implementation was too slow because pass 2 copied large spike arrays and undershot available CPU parallelism.
  Resolution: switched to no-copy / memory-mapped spike loading, used native ALF cluster indexing, capped nested BLAS threads, and raised full-run worker count to 32; final full conversion completed in about 10 minutes.
- Full verifier warnings about all-zero neural trials.
  Resolution: investigated directly against raw spike times; all flagged trials have exactly zero spikes in the aligned window, so the warnings were documented rather than “fixed”.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

Notes:
- Full training completed with:
  - `python3 -u train_decoder.py converted_data.pkl --plot-samples --cpu 2>&1 | tee train_decoder_full_out.txt`
- CPU mode was required because the available GPU previously failed even on the sample run with `torch.OutOfMemoryError` on a 1.64 GiB GPU.
- Loss decreased monotonically across the logged epochs:
  - epoch 1: `1.477615`
  - epoch 20: `0.968977`
  - epoch 70: `0.733570`
  - epoch 120: `0.684017`
  - epoch 160: `0.671367`
  - epoch 200: `0.664907`
- Final test loss: `0.693096`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `choice` | 0.6449 | 0.6222 | Above binary chance (0.5000); weakest decoded variable but still clearly above chance |
| `prior_probability_of_left` | 0.6910 | 0.6689 | Above 3-class chance (0.3333) |
| `wheel_speed_bin` | 0.6509 | 0.6445 | Above 3-class chance (0.3333) |
| `whisker_motion_energy_bin` | 0.7479 | 0.7424 | Strongest decoded variable; above 3-class chance (0.3333) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
|----------|-------------------|-------------------------|
| `choice` | Validation balanced accuracy `0.6222` (chance `0.5000`) | Method paper reports example-session choice decoding improvements in AUC, from about `0.66` single-session to about `0.79` multi-session / `0.72` oracle-assisted variants; not directly comparable to this validator’s balanced accuracy, but consistent with clearly above-chance decoding |
| `prior_probability_of_left` | Validation balanced accuracy `0.6689` (chance `0.3333`) | Method paper reports example-session prior correlation improving from about `0.05` single-session to about `0.34` multi-session and about `0.65` oracle; metric differs, but the converted data likewise support strong above-chance decoding |
| `wheel_speed_bin` | Validation balanced accuracy `0.6445` (chance `0.3333`) | Reference papers evaluate wheel speed as a continuous target with `R2` or visual example traces, not the 3-bin balanced-accuracy task used here; qualitative expectation is that wheel should decode well, which it does |
| `whisker_motion_energy_bin` | Validation balanced accuracy `0.7424` (chance `0.3333`) | Reference papers evaluate whisker motion energy as a continuous target with `R2` or visual example traces; qualitative expectation is strong decodability, which is matched here |

- Accuracy-vs-chance check:
  - `choice`: `0.6222 / 0.5000 = 1.244x` chance
  - `prior_probability_of_left`: `0.6689 / 0.3333 = 2.007x` chance
  - `wheel_speed_bin`: `0.6445 / 0.3333 = 1.933x` chance
  - `whisker_motion_energy_bin`: `0.7424 / 0.3333 = 2.227x` chance
- All outputs are above chance, and all 3-class outputs exceed the requested `1.5x`-chance threshold.
- `choice` is the weakest output at `1.244x` chance, so it was re-checked carefully:
  - raw choice sign convention was re-verified directly from the trial table (`1 -> left`, `-1 -> right`)
  - converted mapping (`left = 0`, `right = 1`) was verified against raw high-contrast trials
  - stimulus-onset alignment, trial masking, and the independent raw-data sanity-check script all matched
  - train/validation gap is small (`0.6449 / 0.6222 = 1.036x`), so there is no sign of leakage or severe overfitting
- Paper-comparison caveat:
  - the local papers do not provide a release-wide table of balanced accuracies for this exact four-output decoder
  - the method paper mixes AUC (choice), correlation (prior), and continuous-target metrics / visual examples (wheel, whisker)
  - because this task forces a common stimulus-onset grid and discretizes wheel / whisker into 3 bins, the paper comparison is necessarily qualitative rather than numeric one-to-one

### Issues Found and Resolved
- `choice` had the lowest validation margin over chance.
  Resolution: re-checked raw choice mapping, trial filtering, and stimulus-onset alignment against the original data and the independent sanity-check script; no conversion bug was found, so the result was documented rather than “fixed.”
- Paper metrics are not directly commensurate with the provided decoder metrics.
  Resolution: compared only at the level that is actually defensible from the local references, and explicitly documented the metric/alignment mismatch.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
- Moved temporary smoke-test pickles and the standalone raw-data sanity-check script into `cache/`.
- Added `cache/README_CACHE.md` documenting those cached artifacts.

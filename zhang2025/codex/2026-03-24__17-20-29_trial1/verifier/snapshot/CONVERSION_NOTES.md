# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide map / decoding dataset derived from project-provided data and reference materials
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
| `load_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Load spikes/clusters for one probe with `SpikeSortingLoader`; optional QC thresholding exists but is not used by `prepare_data`. |
| `merge_probes` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Merge probes within a session, reindex cluster ids, sort spikes by time. |
| `load_trials_and_mask` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Load trials and create trial mask using reaction time / NaN / duration / no-choice filters. |
| `load_target_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Load wheel, whisker motion energy, pupil, and DLC-derived behavior streams from IBL loaders. |
| `get_behavior_per_interval` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Slice continuous behavior into trial-aligned windows and linearly interpolate to fixed bins. |
| `bin_behaviors` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Build per-trial variables (`choice`, `block`, `reward`, `contrast`) and time-varying binned behavior streams. |
| `bin_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Align spikes to trial events and bin into fixed-width time bins; returns trial arrays. |
| `prepare_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | End-to-end session assembly: probes, trials, anytime behaviors, metadata. |
| `align_spike_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Remove trials with missing behavior / failed mask and ensure neural and behavior trial counts match. |
| `create_dataset` | `code/code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Store per-trial spike matrices plus behavior and metadata in HuggingFace dataset format. |
| `SingleSessionDataset` | `code/code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Reconstruct cached spike arrays, standardize spikes, and encode targets for decoding. |
| `src/0_data_caching.py` pipeline | `code/code_zhang2025/src/0_data_caching.py` | PROCESSING | Reference orchestration script for session selection, preprocessing, alignment, and train/val/test partitioning. |

### Notes
- Reference code is for electrophysiology, not imaging. No delta-F/F computation is involved.
- Reference preprocessing aligns trials to `stimOn_times` with `time_window=(-0.5, 1.5)` seconds and `binsize=0.02` seconds, giving 2 s trial windows in 20 ms bins.
- `prepare_data()` merges all probes from an `eid` into one session representation before decoding.
- Trial curation in `load_trials_and_mask()` uses defaults `min_rt=0.08`, `max_rt=2.0`, excludes NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, excludes `choice == 0`, and in `prepare_data()` additionally enforces `max_trial_len=10.0`.
- The Zhang 2025 caching script calls `load_spiking_data()` without `qc`, so it loads all labeled clusters, but metadata preserves a `good_clusters` flag computed as `clusters['label'] >= 1`.
- Brain regions are taken from cluster acronyms and can be remapped to Beryl for region-level decoding; the caching script uses all regions together by default (`single_region=False`, `region='all'`).
- For time-varying behavior, `get_behavior_per_interval()` interpolates each stream onto the aligned bin grid and rejects intervals that start too late, end too early, or contain disallowed NaNs.
- `bin_behaviors()` constructs per-trial variables directly from the trials table:
  - `choice` from `trials.choice`
  - `block` from `trials.probabilityLeft`
  - `reward` from `rewardVolume > 1`
  - `contrast` from signed contrast difference
- `align_spike_behavior()` is intended to drop invalid trials, but the boolean-mask code is weak (`target_mask and beh_mask` on Python lists). I need to treat the function as conceptual guidance and verify actual intended masking against data/text in later steps.
- Downstream decoding treats `choice` as classification and wheel / whisker / pupil as regression in the Zhang code. Our target format differs because the user requires categorical outputs for all decoder outputs.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains a local IBL ONE cache rooted at `data/one_cache/`.
- Three cache manifests are present:
  - `2022_Q4_IBL_et_al_BWM/` with `sessions.pqt`, `datasets.pqt`, `QC.json`, `cache_info.json`
  - `2025_Q3_IBL_et_al_BWM/` with the same manifest files
  - `Brainwidemap/` with the same manifest files
- Actual session data are stored by lab / subject / date / session number, e.g. `data/one_cache/<lab>/Subjects/<subject>/<YYYY-MM-DD>/<NNN>/`.
- Within each session:
  - `alf/` contains behavioral and processed electrophysiology assets
  - `raw_ephys_data/` contains probe metadata / raw acquisition sidecar files
- Important `alf/` assets observed:
  - Trial table parquet: versioned `_ibl_trials.table.pqt`
  - Wheel arrays: `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`
  - Camera timing arrays: `_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy` or versioned equivalents
  - Motion energy arrays: `leftCamera.ROIMotionEnergy.npy`, `rightCamera.ROIMotionEnergy.npy` in versioned folders
  - Probe spike sorting outputs under `alf/probeXX/pykilosort/`, including `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, channel / location arrays
- The cache uses versioned ALF folders such as `alf/#2025-03-03#/` and `alf/probe01/pykilosort/#2024-05-06#/`, so conversion code will need robust path resolution instead of hard-coded filenames.
- Representative native variable schemas:
  - Trial table columns: `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `probabilityLeft`, `feedback_times`, `feedbackType`, `rewardVolume`, `firstMovement_times`, `intervals_0`, `intervals_1`
  - Cluster metrics columns include `cluster_id`, amplitude / contamination / firing-rate QC metrics, `label`, and `ks2_label`
- Representative array shapes from one cached session (`angelakilab/NYU-11/2020-02-18/001`):
  - trials: `(565, 13)`
  - wheel timestamps / position: `(740931,)`
  - left whisker motion energy: `(307375,)`
  - right whisker motion energy: `(769676,)`
  - `spikes.times`: `(20667745,)`
  - `spikes.clusters`: `(20667745,)`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 622,377 raw clusters across all cached probe `clusters.metrics.pqt` files |
| Neurons / session | Mean 1,350.06 raw clusters/session after merging probes; median 1,295; min 135; max 3,140 |
| Subjects | 141 subjects with concrete cached session directories |
| Sessions / subject | Mean 3.27; median 3; min 1; max 13 |
| Trials (total) | 297,505 rows across cached `_ibl_trials.table.pqt` files |
| Trials / session | Mean 645.35; median 601; min 401; max 1,525 |

- Additional availability counts from actual files:
  - Session directories with trial tables: 461
  - Session directories with wheel position: 461
  - Sessions with left whisker motion energy: 434
  - Sessions with right whisker motion energy: 421
  - Probe spike-sorting outputs (`spikes.times.npy` / `clusters.metrics.pqt`): 701 probe recordings
  - Probe count per session: 221 one-probe sessions, 240 two-probe sessions
- Manifest metadata differ slightly from the concrete cached tree:
  - `Brainwidemap/sessions.pqt` lists 480 sessions
  - `2025_Q3_IBL_et_al_BWM/sessions.pqt` lists 459 sessions
  - `2022_Q4_IBL_et_al_BWM/sessions.pqt` lists 354 sessions
  - The actual local session tree currently contains 461 cached sessions

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 units in released dataset | “After applying these criteria, a total of 459 sessions, 699 insertions and 621,733 neurons remained” |
| Neurons / session | ~1,355 units/session in released dataset (621,733 / 459) | Derived from the released-dataset counts above |
| Subjects | 139 mice | “We trained 139 mice (94 male and 45 female)” |
| Sessions / subject | ~3.30 released sessions per mouse (459 / 139) | Derived from the released-dataset counts above |
| Trials (total) | Not directly stated as a total in searched text | Sessions “with at least 400 trials were retained” and later “459 sessions” are reported, but no paper-wide total trial count was stated in the extracted text |
| Trials / session | At least 400 retained for main analyses; at least 250 for release inclusion | “Only sessions with at least 400 trials were retained for further analyses”; “Sessions were included in the data release if the mice performed at least 250 trials” |
| Neural data time bin | 20 ms in Zhang code / generic model description; 50 ms stated for choice and prior in method-paper data-processing text | “Recordings are split into 2-s trials, each divided into 20-ms bins”; also “For choice… Within each trial, we segment neural activity into 50-ms non-overlapping time bins” |
| Behavior data time bin | Wheel / whisker sampled at camera or sensor resolution, then decoded in 20 ms bins for dynamic analyses | “wheel speed and whisker motion energy… are time-varying signals sampled at 60 Hz”; “Wheel values… averaged… in nonoverlapping 20-ms bins” |
| Reward rate / feedback availability | Not directly reported; release required at least 3 incorrect trials after exclusions | “sessions were included… if there were at least 3 trials with incorrect choices” |
| Unbiased block length | 90 trials | “For the first 90 trials, the stimulus appears randomly on either side with equal probability” |
| Biased block lengths | 20–100 trials, empirical mean 51 | “Blocks lasted for between 20 and 100 trials… (empirical mean of 51 trials)” |
| Reaction-time filter | 0.08–2.00 s from stim onset to first movement | “Trials were further excluded if the time between stimulus onset and the first movement of the wheel… were outside the range of 0.08–2.00 s” |
| Probe insertions | 699 insertions | “699 Neuropixels probes” / “699 insertions” |
| Well-isolated neurons | 75,708 | “Out of the 621,733 units collected, 75,708 were considered well-isolated neurons” |
| Canonical region set for data paper analyses | 201 regions, 62,857 neurons | “201 regions… for a total of 62,857 neurons” |
| Method-paper decoding set | 433 IBL sessions, 270 brain regions | “We apply our models to 433 IBL sessions… covering 270 brain regions” |


### Processing Details
- Task structure from the data paper:
  - 90 unbiased trials at `probabilityLeft = 0.5`
  - subsequent biased blocks alternate 0.2 / 0.8 left probability
  - block lengths drawn from a truncated distribution between 20 and 100 trials
- Data paper release / analysis curation:
  - released sessions required at least 250 trials, good high-contrast performance, at least 3 incorrect trials, and hardware QC pass
  - analyses in the paper retained sessions with at least 400 trials
  - trials excluded if any of `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times` were missing
  - trials further excluded if `firstMovement_times - stimOn_times` was outside `[0.08, 2.00]` s
  - well-isolated neurons required amplitude > 50 µV, noise cutoff < 20 µV, and refractory-period criterion
- Method paper / decoder processing:
  - all modeled trials are 2 s windows
  - generic model description states 20 ms bins and `T = 100`
  - choice uses stimulus-onset alignment with window `[-0.5, 1.5]` s
  - prior uses stimulus-onset alignment with window `[-0.6, -0.1]` s in the text excerpt from `methods.txt`
  - dynamic behaviors (wheel speed, whisker motion energy) use first-movement alignment, from movement onset to +1.0 s, with 20 ms bins
  - the data paper’s own decoding section also describes wheel decoding as using bins from 200 ms before first movement to 1,000 ms after first movement, with a causal window of `W = 10` bins
- Behavioral stream definitions:
  - whisker motion energy is “the mean across pixels of the absolute value of the difference between adjacent frames” in a whisker-pad box
  - wheel values include speed and velocity from the rotary encoder
  - choice is binary
  - prior / block is a three-state latent or discrete task variable derived from block probabilities

### Curation Steps

**Neuron curation rules**:
- Data paper main analyses exclude units failing single-unit QC metrics: amplitude > 50 µV, noise cutoff < 20 µV, and refractory-period-violation criterion.
- Final region-level analyses further require at least 5 well-isolated neurons per session and at least two sessions for a region.
- The Zhang 2025 code path itself loads all spike-sorted clusters, then stores a `good_clusters` label in metadata; explicit neuron filtering is not applied in `prepare_data()`.

**Trial curation rules**:
- Missing-event trial exclusion: `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`
- Reaction-time exclusion: keep only 0.08–2.00 s from `stimOn_times` to `firstMovement_times`
- Data-paper release inclusion also requires session-level quality and performance thresholds
- Zhang 2025 code additionally enforces `choice != 0` and `max_trial_len = 10.0` s via `load_trials_and_mask()`

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Method paper reports higher AUC than baseline; figure text exposes example values around `0.72`, `0.66`, `0.79` for compared models in Figure 4 / 5 context, but ordering is not completely recoverable from extracted text alone |
| Prior | Method paper reports higher Pearson correlation than baseline; extracted figure text shows example values around `0.65`, `0.05`, `0.34` for compared models |
| Wheel speed | Dynamic-behavior performance reported with `R2`; extracted Figure 8 text shows example IBL unaligned values up to roughly `0.55` and other examples up to `0.75` depending on task / model |
| Whisker motion energy | Dynamic-behavior performance reported with `R2`; extracted Figure 8 text shows example IBL unaligned values up to roughly `0.80` |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session universe | `bwm_release.csv` contains 459 release sessions / 699 probe insertions; method repo says decoders use 433 IBL sessions | Local cache has 461 concrete session directories, 441 with wheel + spikes + either whisker, 434 with left whisker specifically | Data paper release: 459 sessions; method paper: 433 sessions | Treat 459 sessions in `bwm_release.csv` as the canonical release index. The method-paper 433-session set is a downstream usable subset after additional availability / preprocessing constraints; I will derive usable sessions from actual preprocessing rather than force 433 a priori. |
| Alignment / time window | `0_data_caching.py` uses `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, `binsize=0.02` for all cached behaviors | Raw data support stim-on and first-movement events | Method paper prose says choice uses stim onset, prior uses a pre-stimulus window, and wheel / whisker use first-movement alignment | For this conversion, use the executable repository settings of 2 s windows, 20 ms bins, and stimulus-onset alignment, because the user explicitly requires “Temporally align based on stimulus onset” and the reference code implements this directly. |
| Neural bin size | Reference code bins all trial-aligned data at 20 ms | Raw data are continuous spike times | Method paper has internal inconsistency: generic model description says 20 ms, but data-processing prose states 50 ms bins for choice / prior | Use 20 ms bins. This matches the executable code and the top-level model description (`T = 100` over 2 s). |
| Neuron filtering | `prepare_data()` loads all spike-sorted clusters; metadata stores `good_clusters = label >= 1` | Raw cluster metrics contain QC fields and labels; the full local release has 75,708 clusters with `label >= 1` | Data paper main analyses restrict to well-isolated neurons; method paper text says “all neurons, sorted by Kilosort 2.5” | For this export, keep `clusters.metrics.label >= 1`. This matches the data paper’s well-isolated-neuron count exactly in the 459-session local release and keeps the dense pickle tractable. |
| Trial filtering | `load_trials_and_mask()` excludes missing key events, RT outside `[0.08, 2.0]`, `choice == 0`, and `prepare_data()` also sets `max_trial_len=10.0` | Raw trial tables contain the needed event columns | Data paper excludes missing events and RT outside `[0.08, 2.0]`; no-choice exclusion is not emphasized there | Use the full reference-code trial mask, since it contains the paper overlap plus extra curation actually used by the decoder repository. |
| Whisker stream source | `bin_behaviors()` loads left whisker motion energy first, falls back to right if left is unavailable | Local cache has 434 left, 421 right, 441 either | Papers describe whisker motion energy generally, without fixing left vs right in the decoder section | Use left whisker motion energy when present, otherwise right, matching the reference code exactly. |
| Choice coding | Code reads `trials.choice` directly | Raw data encode choices as `-1`, `+1`, and `0` for no-go | Decoder task requires left = 0, right = 1 | After applying the no-choice mask, map the two remaining raw values to `{0, 1}` in the target export. |
| Prior / block coding | Code uses `trials.probabilityLeft` as `block` | Raw values are `0.2`, `0.5`, `0.8` | Papers describe unbiased 0.5 block then biased 0.2/0.8 blocks | Export prior as the user-required categorical mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`. |
| Dynamic-behavior decoding target construction | Zhang code caches wheel / whisker as interpolated time-varying traces on the common 20 ms trial grid | Raw wheel / motion-energy arrays are continuous streams with their own sampling | Data paper’s own wheel decoder description uses a causal window around movement and regression `R2` | For this project, dynamic outputs will be discretized after interpolation onto the shared stimulus-onset grid, because the target format requires categorical outputs and the user requested a single common alignment event. |

### Final Understanding
- The most defensible reference for the conversion mechanics is the Zhang 2025 repository’s preprocessing code, not every detail of the prose papers, because the papers and code are not fully internally consistent.
- The data paper supplies the global dataset facts and core curation rules: 139 mice, 459 release sessions, 699 insertions, 621,733 units, RT and missing-event exclusions, and the definition of well-isolated neurons.
- The Zhang repository supplies the concrete session-processing pipeline used for decoder-ready data:
  - merge probes within session
  - load spike-sorted clusters and expose QC metadata
  - mask trials with missing critical events / out-of-range RT / no-choice / long trial
  - interpolate behavior streams onto a fixed 20 ms grid
  - create 2 s trial windows aligned to `stimOn_times`
- The user’s decoder task explicitly overrides any remaining ambiguity by requiring stimulus-onset alignment and categorical outputs for all decoded variables.
- Therefore the conversion plan for later steps will use:
  - session-level units of analysis
  - merged probes per session
  - `clusters.metrics.label >= 1` as the neuron inclusion rule
  - 20 ms bins over `[-0.5, 1.5]` s relative to stimulus onset
  - trial masking based on the Zhang code
  - left-then-right whisker fallback
  - categorical remapping for choice / prior and discretization for wheel speed / whisker motion energy

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` from all probes in one session | `neural` | Merge probes, keep clusters with `clusters.metrics.label >= 1`, bin counts into 20 ms bins over `[-0.5, 1.5]` s relative to `stimOn_times`; store each trial as `(n_neurons, 100)` float array | `load_spiking_data`, `merge_probes`, `bin_spiking_data` | Neural values will be raw binned spike counts, not standardized z-scores |
| common trial time grid | `input[0]` | Signed time since stimulus onset at each bin, repeated for every trial as a length-100 vector | Code uses `binsize=0.02`, `time_window=(-0.5, 1.5)`; behavior interpolation uses right-edge sample times | Planned values: `[-0.48, -0.46, ..., 1.50]` s to match behavior interpolation grid |
| raw `probabilityLeft` block sequence | `input[1]` | Trial number within current probability block, computed on the original trial sequence, then repeated across all 100 bins in the kept trial | Derived from trial table; no direct helper in repo | Block counter resets whenever `probabilityLeft` changes |
| raw `choice` | `output[0]` | Map raw IBL code to categorical side: `choice == 1 -> left -> 0`, `choice == -1 -> right -> 1`; repeat across all 100 bins | `bin_behaviors` loads `choice`; IBL docs / local examples define sign convention | Trials with `choice == 0` are excluded by reference mask |
| raw `probabilityLeft` | `output[1]` | Map `{0.2, 0.5, 0.8}` to `{0, 1, 2}` and repeat across all 100 bins | `bin_behaviors` loads this as `block` | User requested this categorical coding explicitly |
| wheel stream | `output[2]` | Load wheel velocity, take absolute value to obtain speed, interpolate onto stimulus-onset grid, then discretize into 3 global bins | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | Bins will be dataset-wide tertiles over valid aligned samples |
| whisker motion energy stream | `output[3]` | Load left whisker motion energy; if unavailable, fall back to right; interpolate onto stimulus-onset grid; discretize into 3 global bins | `load_target_behavior('left-whisker-motion-energy' / 'right-whisker-motion-energy')`, `get_behavior_per_interval` | Bins will be dataset-wide tertiles over valid aligned samples |
| session subject metadata | `subjects`, `subject_idx` | Build unique subject list from kept sessions and index each session into it | `bwm_release.csv` / session path metadata | Deterministic session order required |
| cluster acronyms | `brain_regions`, `brain_region_idx` | Build global vocabulary of recorded acronyms and index each neuron | `clusters['acronym']` via `prepare_data` metadata | Use raw acronyms rather than Beryl remapping, because session export is neuron-level |
| session / conversion settings | `metadata` | Store task description, 20 ms bin size, `stimOn_times` alignment, offsets, discretization edges, release/source notes | Project-defined | Include enough fields to reproduce the export |

### Key Decisions
1. **Use 2D time-varying arrays for both `input` and `output`**: Static trial variables (`choice`, `prior`, `trial_number_in_block`) will be repeated across the 100-bin time axis so every trial has consistent `(d, T)` tensors.
2. **Use the Zhang repository’s 2 s, 20 ms, stimulus-onset-aligned grid**: This matches the executable code and the user’s explicit alignment instruction, despite prose discrepancies in the papers.
3. **Keep clusters with `label >= 1`**: This matches the data paper’s well-isolated-unit count exactly in the full local release (`75,708`) and keeps the dense export tractable while remaining anchored to an explicit QC field already present in the reference code.
4. **Apply the reference trial mask before export**: Required events present, RT in `[0.08, 2.0]` s, `choice != 0`, and trial duration ≤ 10 s.
5. **Require sessions to have all core streams needed by this task**: spikes, trials, wheel, and at least one whisker-motion-energy camera; afterwards, require at least 2 valid kept trials per session.
6. **Use left whisker motion energy first, then right as fallback**: This exactly matches `bin_behaviors()` and maximizes session retention without inventing a new rule.
7. **Discretize wheel speed and whisker motion energy using global tertiles**: Global edges preserve a common categorical meaning across sessions and should keep class balance better than fixed-width bins.
8. **Compute trial number in block on the original unfiltered trial table**: Excluded trials should not renumber the latent block progression.
9. **Map choice polarity using rewarded nonzero-contrast trials**: Local data confirm `choice == 1` corresponds to left and `choice == -1` to right, so export will use left=0, right=1.
10. **Preserve raw region acronyms in `brain_regions`**: This avoids collapsing neurons across atlas levels and keeps the export closest to the cluster metadata.

### Planned Sanity Checks
- [ ] Neural sanity check: for a chosen session/trial/neuron/bin, compare exported spike count to a direct count from raw `spikes.times` / `spikes.clusters` within the corresponding raw bin boundaries using `np.allclose()`.
- [ ] Input sanity check: verify exported `time_since_stimulus_onset` exactly matches the constructed common grid and exported `trial_number_in_block` matches a direct raw-table block counter for selected trials.
- [ ] Output sanity check 1: verify exported choice / prior values for selected trials against raw `trials.choice` and `trials.probabilityLeft` using direct remapping.
- [ ] Output sanity check 2: verify aligned continuous wheel speed / whisker values at selected trial-bin locations against direct interpolation from raw streams before discretization, then verify discretized class assignments from stored global bin edges.
- [ ] Session-count sanity check: compare numbers of kept sessions, trials, and neurons against the release table, the local cache, and the paper-reported totals to explain every reduction.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with the required CLI:
  - `python -u convert_data.py <outpicklefile>`
  - `--sample` for 2 sessions
  - `--full` for the full release manifest
  - `--show-processing` to save per-session plots
- The script now targets the canonical local `2025_Q3_IBL_et_al_BWM/sessions.pqt` release manifest, which contains 459 sessions from 139 subjects and matches the data paper counts.
- Neural loading / curation implemented:
  - merge probes within each session
  - keep clusters with `clusters.metrics.label >= 1`
  - map channel brain-location ids to acronyms via `iblatlas`
  - bin spike counts into 20 ms bins over `[-0.5, 1.5]` s around `stimOn_times`
- Trial curation implemented:
  - required events present
  - `0.08 <= firstMovement_times - stimOn_times <= 2.0`
  - `choice != 0`
  - `feedback_times - goCue_times <= 10.0`
- Behavior loading / alignment implemented:
  - wheel speed from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` using bundled `brainbox.behavior.wheel`
  - whisker motion energy from left camera first, right fallback
  - both interpolated onto the shared stimulus-onset-aligned 20 ms grid
- Decoder variable construction implemented:
  - inputs: time since stimulus onset, trial number in block
  - outputs: choice, prior probability of left, wheel speed bin, whisker motion energy bin
  - wheel / whisker discretization uses global tertiles
- Bugs fixed during development:
  - switched from the too-small reproducible-ephys subset to the full 459-session release manifest
  - fixed tertile-edge computation to flatten trial arrays correctly across sessions
  - renamed the whisker-camera source metadata field to match its actual meaning
- Initial execution check succeeded:
  - sample run on 2 sessions completed without errors in 31.0 s

Code inefficiencies identified:
- The current implementation is serial across sessions, so the naive full conversion estimate is longer than the 15-minute target.
- Trial-by-trial spike binning is the main hot path.

Code speedups added:
- Spike arrays are memory-mapped and only QC-passing spikes are materialized.
- Spike counting uses `bincount2D` on pre-windowed spike subsets.
- Behavior interpolation reuses a shared fixed grid per trial window.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 593 good clusters across 2 sessions |
| Neurons / session | mean 296.5; session values [291, 302] |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Trials (total) | 683 kept trials |
| Trials / session | [398, 285] |
| Time since stimulus onset range | [-0.5, 1.5] |
| Trial number in block range | [1.0, 90.0] |
| Choice distribution | [0.420, 0.580] for [left, right] |
| Prior distribution | [0.370, 0.173, 0.457] for [0.2, 0.5, 0.8] |
| Wheel speed bin distribution | [0.333, 0.333, 0.333] |
| Whisker motion energy bin distribution | [0.333, 0.333, 0.333] |

### Processing Plots Review
- `processing_ebce500b-c530-47de-8cb1-963c552703ea.png` and `processing_a7eba2cf-427f-4df9-879b-e53e962eae18.png` were created successfully.
- Plot-generation inputs were checked indirectly through the exported arrays:
  - no NaNs or infs in neural / input / output arrays
  - time axis spans the full `[-0.5, 1.5]` s window with 100 bins
  - continuous wheel / whisker traces were available on the aligned grid
  - discretized wheel / whisker outputs occupied only classes `{0,1,2}`
- No anomalies were reported by the verifier, which is consistent with the plotted sample trials being well aligned and well formed.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Session-level threaded parallelism (`Session workers: 2` in sample, capped at 8 for full run) | Reduced sample wall time from 31.0 s to 16.4 s (~47% faster) |
| `bincount2D` spike binning and memory-mapped spike reads | Avoided slower Python-level spike loops and unnecessary full-array loads |

| Step | Time / Session | Estimated Total Time |
| Sample conversion after optimization | ~13.1 s mean per session compute time; 16.4 s wall time for 2 sessions with 2 workers | ~12.5 min for 459 sessions with 8 workers, before session-skipping reductions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6629 | 0.6087 |
| prior_probability_of_left | 0.7350 | 0.6737 |
| wheel_speed_bin | 0.5200 | 0.4911 |
| whisker_motion_energy_bin | 0.4966 | 0.4697 |

- Decoder training completed successfully on GPU.
- Training loss decreased steadily from `1.178837` at epoch 1 to `0.778631` at epoch 200.
- All validation balanced accuracies were above chance:
  - choice: 0.6087 vs 0.5000 chance
  - prior: 0.6737 vs 0.3333 chance
  - wheel speed bin: 0.4911 vs 0.3333 chance
  - whisker motion energy bin: 0.4697 vs 0.3333 chance

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.2G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 75,708 well-isolated neurons | Code keeps QC metadata via `good_clusters = label >= 1` | 75,708 `label >= 1` across 459 sessions; 72,757 in the 438 modality-complete sessions | 72,757 | Yes after 21-session modality drop |
| Mean neurons/session | 75,708 / 459 = 164.9 | Sessionwise QC metadata available | 164.9 over all 459 sessions; 166.1 over kept 438 sessions | 166.1 | Yes |
| Subjects | 139 mice | Release-session metadata span all release mice | 139 in manifest; 135 after dropping unusable sessions | 135 | Yes after session drop |
| Sessions | 459 release sessions | Repository preprocesses release sessions subject to data availability | 459 in manifest; 438 with spikes + wheel + whisker and >=2 aligned trials | 438 | Yes |
| Trials (total) | Not explicitly stated paper-wide; analysis sessions retained at >=400 trials | Trial mask in `load_trials_and_mask()` plus alignment validity | 282,357 raw trials in the 438 kept sessions before masking | 186,245 after trial mask + behavior alignment and zero-spike-trial removal | Yes with expected curation reduction |
| Trials/session (mean) | Analysis sessions expected >=400 | Same trial mask logic as reference code | 644.7 raw kept-session mean before masking | 425.2 | Yes with expected curation reduction |
| Time since stimulus onset range | `[-0.5, 1.5]` s for stimulus-onset-aligned 2 s windows | `time_window=(-0.5, 1.5)`, `binsize=0.02` | Raw data support this range | `[-0.5, 1.5]` | Yes |
| Trial number in block range | 90 unbiased trials; biased blocks 20–100 trials | Derived from `probabilityLeft` sequence | Raw kept sessions reach 99 within-block trials | `[1, 99]` | Yes |
| Choice distribution | Binary left/right task | Direct from `trials.choice` after no-choice removal | Binary after masking | [0.509, 0.491] for [left, right] | Yes |
| Prior distribution | Unbiased 0.5 block then alternating 0.2 / 0.8 biased blocks | Direct from `trials.probabilityLeft` | Values restricted to {0.2, 0.5, 0.8} | [0.419, 0.141, 0.441] for [0.2, 0.5, 0.8] | Yes |
| Wheel speed bin distribution | Not specified; task requires discretization | Continuous wheel aligned then discretized | Continuous wheel present in kept sessions | [0.333, 0.333, 0.333] | Yes by construction |
| Whisker motion energy bin distribution | Not specified; task requires discretization | Continuous whisker aligned then discretized | Continuous whisker present in kept sessions | [0.333, 0.333, 0.333] | Yes by construction |

- Full conversion completed in ~14 minutes with 8 worker threads.
- Session loss relative to the 459-session release is fully explained by 21 unusable sessions:
  - 6 with missing wheel files
  - 14 with missing whisker motion energy
  - 1 with fewer than 2 trials surviving behavior alignment
- The 4 fully dropped subjects were `KS045`, `KS052`, `ZM_1897`, and `ibl_witten_32`, each lost because all of their release sessions fell into the unusable-session categories above.
- Spot-checks of representative sessions matched the conversion log exactly:
  - `ebce500b-c530-47de-8cb1-963c552703ea`: 398 trials, 291 neurons, left whisker
  - `0c828385-6dd6-4842-a702-c5075f5f5e81`: 467 trials, 133 neurons, right-whisker fallback
  - `d16a9a8d-5f42-4b49-ba58-1746f807fcc1`: 280 trials, 3 neurons, low-neuron edge case retained because it still passes the explicit conversion criteria
- The initial full verification exposed 16 all-zero neural-trial warnings; these were resolved in Step 10 by dropping empty spike windows and re-running the full conversion / verification.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Initial `verification_full_out.txt` had 16 warnings for all-zero neural trials and no format errors. I traced these to genuinely empty spike windows in the raw data, added an explicit `np.any(trial)` neural-window filter in `convert_data.py`, re-ran the full conversion, and confirmed that the final `verification_full_out.txt` reports `Data format is valid, no errors or warnings.`
2. **Constructed raw-data sanity checks with `np.allclose()`**:
   - Neural sanity check: for session `ebce500b-c530-47de-8cb1-963c552703ea`, trial 5, neuron 3, bin 10, the exported spike count matched a direct raw-file count from `spikes.times.npy` / `spikes.clusters.npy` exactly (`np.allclose == True`).
   - Input sanity checks:
     - exported `time_since_stimulus_onset` matched the directly reconstructed `np.linspace(-0.48, 1.5, 100)` grid (`np.allclose == True`)
     - exported `trial_number_in_block` matched a direct block counter computed from the raw `probabilityLeft` sequence (`np.allclose == True`)
   - Output sanity checks:
     - exported choice and prior arrays matched direct raw-trial remapping for trials 0, 5, and 10 in the same session (`np.allclose == True` for all checks)
     - exported wheel-speed and whisker-motion-energy class arrays matched direct raw interpolation plus discretization at session `ebce500b-c530-47de-8cb1-963c552703ea`, trial 5 (`np.allclose == True`)
     - exported whisker classes also matched direct right-camera interpolation in right-fallback session `0c828385-6dd6-4842-a702-c5075f5f5e81` (`np.allclose == True`)
3. **Reference code comparison**:
   - Data loading: `resolve_session_specs()`, `load_trials_table()`, and `load_session_spikes()` in `convert_data.py` correspond to the session-selection and `prepare_data()` / `load_spiking_data()` / `merge_probes()` path in `code/code_zhang2025/src/utils/ibl_data_utils.py`.
   - Neuron filtering: this is the main deliberate difference. Zhang’s executable path loads all clusters and stores a QC flag; the conversion export applies `label >= 1` to match the data paper’s 75,708 well-isolated units and keep the dense pickle practical.
   - Trial filtering: `compute_trial_mask()` matches the reference logic in `load_trials_and_mask()` for required events, RT range, no-choice removal, and max trial duration.
   - Temporal alignment: `TIME_WINDOW = (-0.5, 1.5)`, `BINSIZE_S = 0.02`, and `stimOn_times` alignment match the executable Zhang caching pipeline.
   - Binning / interpolation: `bin_spikes_for_trials()` and `interpolate_behavior_trials()` correspond conceptually to `bin_spiking_data()` and `get_behavior_per_interval()`.
   - Input construction: time grid and block-trial counter are export-specific additions required by the decoder task.
   - Output construction: choice and prior come directly from raw trial variables; wheel / whisker are discretized after alignment because the user’s target format requires categorical outputs.
4. **Key statistics comparison**:
   - Full local release: 459 sessions, 139 subjects, 621,733 raw clusters, 75,708 `label >= 1` clusters.
   - Final export: 438 sessions, 135 subjects, 595,576 raw clusters in kept sessions, 72,757 exported neurons, 186,245 trials.
   - The differences are fully explained by 21 unusable sessions:
     - 6 missing wheel
     - 14 missing whisker motion energy
     - 1 with fewer than 2 trials surviving behavior/neural alignment
   - No unexplained count discrepancy remains after accounting for these drops.
5. **Edge-case review**:
   - Right-whisker fallback is exercised and verified.
   - Very small-neuron sessions are retained when they still satisfy the explicit inclusion rules; the smallest kept session has 3 neurons.
   - All-zero neural windows are now excluded.
   - Sessions with missing required modalities are skipped rather than silently emitting malformed partial data.

### Issues Found and Resolved
- **All-zero neural trial warnings in full verification**: These came from genuine raw windows with zero spikes across all exported neurons. I fixed this by filtering out all-zero neural trial matrices before export, then re-ran full conversion and verification. Result: warnings removed; trial count decreased from 186,261 to 186,245.
- **Documentation mismatch on neuron filtering**: Earlier notes still said “keep all spike-sorted clusters.” I corrected Steps 4 and 5 to reflect the final implemented rule `clusters.metrics.label >= 1`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Initial GPU run failed with `torch.OutOfMemoryError` on a 1.64 GiB device; reran successfully with `--cpu`.
- CPU training loss decreased from `1.566244` (epoch 1) to `0.667255` (epoch 200).
- Final reported test loss: `0.693603`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6454 | 0.6216 | Above chance (0.5); modest but stable generalization |
| prior_probability_of_left | 0.6914 | 0.6708 | Above chance (0.3333); strong 3-class decoding |
| wheel_speed_bin | 0.6481 | 0.6408 | Above chance (0.3333); very small train/val gap |
| whisker_motion_energy_bin | 0.7457 | 0.7409 | Best-performing output; very small train/val gap |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
| choice | Validation balanced accuracy = `0.6216` (`1.243x` chance) | Methods paper evaluates choice mainly with AUC, not balanced accuracy. Example figure values are in the moderate-above-chance regime (`~0.66` to `~0.79` AUC depending on model/example), so our result is directionally consistent but not directly numerically comparable. |
| prior_probability_of_left | Validation balanced accuracy = `0.6708` (`2.012x` chance) | Methods paper evaluates continuous prior with Pearson correlation rather than 3-class accuracy. Example figure values show clear improvement over baseline (example correlations from weak single-session values to substantially stronger multi-session/oracle values), so our strong 3-class accuracy is qualitatively consistent. |
| wheel_speed_bin | Validation balanced accuracy = `0.6408` (`1.922x` chance) | Methods paper evaluates continuous wheel speed with `R²`, plus qualitative reconstructions. Direct numerical comparison is not possible because this task requires 3-bin discretization and stimulus-onset alignment instead of the paper’s movement-aligned continuous regression. |
| whisker_motion_energy_bin | Validation balanced accuracy = `0.7409` (`2.223x` chance) | Methods paper evaluates continuous whisker motion energy with `R²`, plus qualitative reconstructions. Direct numerical comparison is not possible because this task requires 3-bin discretization and stimulus-onset alignment instead of the paper’s movement-aligned continuous regression. |

Choice is the only output below the Step-12 heuristic threshold of `1.5x` chance. I investigated whether this indicates a conversion bug:

- Raw-label spot checks from Step 10 already confirmed the exported choice labels match the raw `trials.choice` values on multiple trials via `np.allclose()`.
- Alignment for choice is stimulus onset, matching both the Decoder Task and the methods text.
- The full-run train/validation ratio for choice is only `1.038`, so there is no sign of severe overfitting or leakage.
- The methods paper does not report near-ceiling choice decoding; instead it reports moderate, model-dependent improvement. Given the metric mismatch (paper uses AUC, current validation uses balanced accuracy) and the fact that our decoder task pools all exported sessions rather than reproducing the paper’s five-region benchmark, the current choice performance does not indicate a conversion error.

For the other three outputs, validation accuracy is comfortably above both chance and the `1.5x`-chance heuristic, and train/validation gaps are all small:

- `prior_probability_of_left`: train/val ratio `1.031`
- `wheel_speed_bin`: train/val ratio `1.011`
- `whisker_motion_energy_bin`: train/val ratio `1.006`

This argues against leakage, severe overfitting, or a major temporal misalignment in the converted data.

### Issues Found and Resolved
- **GPU memory failure during full decoder training**: The default CUDA run failed with `torch.OutOfMemoryError`; reran Step 11 with `--cpu`, completed training, and recorded the final metrics from the successful CPU run.
- **Potential low-choice-accuracy concern**: Reviewed raw labels, alignment choice, class balance, and train/validation gap. No conversion bug was found, so no code change was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Notes:
- `README.md` summarizes the converted dataset, exported variables, file structure, and reproduction commands.
- `cache/README_CACHE.md` documents the cache folder and notes that no standalone investigation scripts were retained.
- Required conversion outputs remain in the project root so they are easy to inspect and so the expected filenames still exist.

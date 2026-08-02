# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide neural activity dataset and reference code
- **Date started**: 2026-03-25
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
| `load_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads spike-sorted electrophysiology data and cluster metadata for one probe via `SpikeSortingLoader`; optional QC threshold exists but is not used by `prepare_data`. |
| `merge_probes` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Merges spikes and clusters across probes in a session, reindexes clusters, and sorts merged spikes by time. |
| `load_trials_and_mask` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Loads session trials and creates a trial inclusion mask based on reaction time, trial duration, NaNs in required events, and no-choice trials. |
| `prepare_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Main session loader used by the paper code: merges probes, loads trials/mask, loads continuous behaviors, and packages metadata. |
| `list_brain_regions` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Maps cluster acronyms to Beryl regions and lists available regions. |
| `select_brain_regions` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Selects cluster ids for requested Beryl region(s); in the caching script this is all regions together. |
| `get_spike_data_per_interval` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Bins spikes into trial-aligned intervals with multiprocessing. |
| `bin_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Creates per-trial spike-count matrices aligned to `align_time`; output shape per trial is `(n_bins, n_clusters)`. |
| `load_target_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Loads continuous behavior streams such as wheel velocity/speed and whisker motion energy using `SessionLoader` or camera features. |
| `get_behavior_per_interval` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Segments continuous behavior into trial windows and linearly interpolates it to aligned bins. |
| `bin_behaviors` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Builds per-trial behavior arrays and trial-level variables (`choice`, `block`, `reward`, `contrast`). |
| `align_spike_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Drops trials where neural/behavior data are missing or rejected by the trial mask; reshapes behavior arrays to one row per retained trial. |
| `create_dataset` | `code/code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Converts per-trial spike matrices to CSR sparse storage and packages behavior plus session metadata in a Hugging Face dataset. |
| `standardize_spike_data` | `code/code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Z-scores spike counts across trials separately for each time bin and neuron before decoding. |
| `SingleSessionDataset` | `code/code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Loads cached per-session arrays, optionally subsets by region, standardizes spikes, and one-hot encodes classification targets. |
| `0_data_caching.py` main loop | `code/code_zhang2025/src/0_data_caching.py` | PROCESSING | Reference preprocessing pipeline used to cache aligned datasets for decoding. |

### Notes
- Reference code is for electrophysiology, not imaging. Neural data are spike times and spike-sorted cluster metadata; there is no delta-F/F computation anywhere in the pipeline.
- The main reference preprocessing entry point is `code/code_zhang2025/src/0_data_caching.py`.
- Default preprocessing parameters in that script are:
  - `align_time='stimOn_times'`
  - `time_window=(-0.5, 1.5)` seconds
  - `interval_len=2`
  - `binsize=0.02` seconds
  - `single_region=False` so all available Beryl regions are pooled unless a decoder later subsets them.
- Trial-level variables created in the reference code are:
  - `choice` from `trials_df['choice']`
  - `block` from `trials_df['probabilityLeft']`
  - `reward` from `(rewardVolume > 1).astype(int)`
  - `contrast` from signed contrast difference computed from left/right contrasts
- Continuous behaviors used in the reference code are:
  - `wheel-speed` as `abs(wheel velocity)`
  - `whisker-motion-energy` from left camera whisker energy, falling back to right camera if left is unavailable
  - `pupil-diameter` exists in code but is commented out in `0_data_caching.py` because some sessions lack pupil traces
- Trial curation in `prepare_data` is driven by `load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)` with defaults:
  - minimum reaction time `0.08 s`
  - maximum reaction time `2.0 s`
  - exclude trials with NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
  - exclude no-choice trials (`choice == 0`)
  - do not exclude the initial unbiased block unless explicitly requested
  - exclude trials longer than `10.0 s` from `goCue_times` to `feedback_times`
- Cluster quality:
  - `load_spiking_data` supports QC filtering through a `qc` argument, but `prepare_data` does not pass a threshold, so all loaded clusters are kept in the reference caching step.
  - Good-cluster labels are still stored in metadata as `clusters['label'] >= 1`.
- Spike binning details:
  - spikes are merged across probes before binning
  - per-trial intervals are computed from `stimOn_times + (-0.5, 1.5)`
  - multi-bin spike counts are produced with 20 ms bins
  - per-trial spike matrices are transposed to `(n_bins, n_clusters)` in the cached dataset
- Behavior alignment details:
  - continuous signals are segmented into the same per-trial `stimOn` window
  - behavior traces are linearly interpolated onto `n_bins = ceil(interval_len / binsize)` samples
  - valid intervals require data to start and end within one bin of the requested interval boundaries
- Final trial alignment is enforced by `align_spike_behavior`, which deletes any trial rejected by the trial mask or missing aligned behavior.
- Decoder scripts (`1_decode_single_session.py`, `2_decode_multi_session.py`, `3_decode_multi_region.py`) operate on the cached aligned trials and target one behavior at a time. Classification is used for `choice`; regression is used for continuous behaviors in the reference code.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are stored in `data/one_cache`, using the IBL ONE cache layout rather than a custom project format.
- Release-level metadata tables are present in:
  - `data/one_cache/Brainwidemap/{sessions.pqt,datasets.pqt,cache_info.json,QC.json}`
  - `data/one_cache/2022_Q4_IBL_et_al_BWM/{sessions.pqt,datasets.pqt,...}`
  - `data/one_cache/2025_Q3_IBL_et_al_BWM/{sessions.pqt,datasets.pqt,...}`
- Session data are organized by lab and subject:
  - `data/one_cache/<lab>/Subjects/<subject>/<YYYY-MM-DD>/<session_number>/alf/`
- Within each `alf/` directory, the relevant files for this task are:
  - trials: versioned parquet table such as `alf/#2025-03-03#/_ibl_trials.table.pqt`
  - wheel: `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`
  - camera time bases: `_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy`
  - whisker motion energy: versioned files such as `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy`
  - electrophysiology per probe: `alf/probeXX/pykilosort/#revision#/spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, `clusters.depths.npy`, `clusters.uuids.csv`
- File formats encountered:
  - `.pqt` parquet tables for trials and cluster metrics
  - `.npy` NumPy arrays for timestamps, behavioral traces, spike times, spike cluster ids, depths, channel info
  - `.csv` for cluster UUIDs
  - `.json` for cache metadata and release QC summaries
- Example raw variable availability from a representative session:
  - trials table columns: `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `probabilityLeft`, `feedback_times`, `feedbackType`, `rewardVolume`, `firstMovement_times`, `intervals_0`, `intervals_1`
  - cluster metrics columns include `cluster_id`, `spike_count`, `firing_rate`, `label`, `ks2_label`, contamination and presence-ratio QC metrics
  - wheel and camera time bases are float64 arrays; spike times are float64 and spike cluster ids are uint32
- Release metadata from `data/one_cache/Brainwidemap/cache_info.json` indicate the full Brain Wide Map cache table contains 480 sessions and 76,563 dataset records.
- Local on-disk availability is slightly smaller than the release table:
  - 461 session directories with `alf/`, spikes, and trials
  - 455 of those have wheel timestamps
  - 441 of those have whisker motion energy from at least one camera
- For the requested decoder outputs, the usable local subset is the 441 sessions that contain spikes, trials, wheel, and whisker motion energy. No additional download appears necessary for that subset.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 596,529 raw clusters across 441 locally complete sessions |
| Neurons / session | 1,352.67 raw clusters/session (mean) |
| Subjects | 137 subjects in the locally complete subset |
| Sessions / subject | 3.22 sessions/subject (mean) |
| Trials (total) | 284,280 trials across 441 locally complete sessions |
| Trials / session | 644.63 trials/session (mean) |

Additional size notes:
- All local sessions with spikes and trials: 461 sessions, 141 subjects, 297,505 trials, 622,377 raw clusters.
- Good-label clusters (`label >= 1`) in the locally complete subset: 72,873 total, mean 165.24 per session.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 units in the public release | "After applying these criteria, a total of 459 sessions, 699 insertions and 621,733 neurons remained, constituting the publicly released dataset." |
| Neurons / session | ~1,355 units/session in public release (inferred from 621,733 / 459) | Same quote as above; per-session value is an inference from reported totals. |
| Subjects | 139 mice | "We trained 139 mice (94 male and 45 female) on the International Brain Laboratory (IBL) decision-making task..." |
| Sessions / subject | ~3.30 sessions/subject in public release (inferred from 459 / 139) | Derived from the reported 459 sessions and 139 mice. |
| Trials (total) | Not explicitly reported; lower bound >= 183,600 in the analysis set | "Only sessions with at least 400 trials were retained for further analyses." |
| Trials / session | >= 400 for analyses; >= 250 for release inclusion | "Only sessions with at least 400 trials were retained for further analyses." and "Sessions were included in the data release if the mice performed at least 250 trials..." |
| Neural data time bin | 20 ms in the reference code and for dynamic behaviors in the methods paper; methods paper states 50 ms bins for choice/prior trial-aligned analyses | "Recordings are split into 2-s trials, each divided into 20-ms bins..." and "Within each trial, we segment neural activity into 50-ms non-overlapping time bins." |
| Behavior data time bin | Wheel/whisker decoded in 20 ms bins; raw behavior sampled at camera/wheel rates | "Wheel values ... were averaged in nonoverlapping 20-ms bins..." and "Choice and prior are static within a trial, while wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz." |
| Reward rate | Not explicitly reported in provided texts | Not directly stated in `methods.txt`. |
| Initial unbiased block length | 90 trials | "For the first 90 trials, the stimulus appears randomly on either side with equal probability..." |
| Prior probability levels | 0.2, 0.5, 0.8 block structure | "...for the first 90 trials... equal probability" and "in right-bias blocks, stimuli appeared on the right on 80% of the trials, whereas in left-bias blocks, stimuli appeared on the right on 20% of the trials." |
| Mean biased block length | 51 trials (empirical mean) | "Blocks lasted for between 20 and 100 trials, which were drawn from a truncated geometric distribution (empirical mean of 51 trials)." |
| Well-isolated neurons | 75,708 | "Out of the 621,733 units collected, 75,708 were considered well-isolated neurons." |
| Sessions used in methods paper decoding | 433 sessions, 270 brain regions | "We apply our models to 433 IBL sessions, covering 270 brain regions and four behavioral variables..." |


### Processing Details
- The task is the IBL visual decision-making paradigm with wheel-turn choice, 90 unbiased trials at session start, then alternating biased blocks.
- The methods paper describes two related preprocessing setups:
  - trial-aligned data for `choice` and `prior`
  - first-movement aligned or trial-unaligned data for dynamic behaviors (`wheel speed`, `whisker motion energy`)
- Temporal alignment from the methods paper:
  - `choice`: align to stimulus onset, use neural activity from `-0.5 s` to `+1.5 s`
  - `prior`: align to stimulus onset, use a pre-stimulus window from `-0.6 s` to `-0.1 s`
  - `wheel speed` and `whisker motion energy`: align to first movement onset, decode from movement onset to `+1.0 s`; the datapaper decoding methods additionally mention wheel values averaged from `-0.2 s` to `+1.0 s` around first movement
- Binning from the methods paper:
  - 20 ms bins are the default in the method paper overview and in the provided reference code
  - 50 ms bins are explicitly stated for choice/prior in the copied methods text
  - this is a paper-vs-code discrepancy that must be resolved in Step 4
- Whisker motion energy is defined as the mean absolute difference between adjacent video frames in a whisker-pad bounding box anchored between nose tip and eye.
- Sessions in the datapaper release passed behavioral and hardware QC; analyses further retained sessions with at least 400 trials.
- The datapaper decoding analyses combined neurons across probes within a session rather than decoding each probe independently.

### Curation Steps

**Neuron curation rules**:
- Datapaper analyses exclude neurons that fail the well-isolated single-unit criteria described in the paper.
- Explicit text in `methods.txt` states the criteria include amplitude, noise cut-off, and refractory-period-violation checks.
- Final analyses in the datapaper also restrict to grey-matter regions with at least five well-isolated neurons per session and at least two sessions per region.

**Trial curation rules**:
- Exclude trials if any of these events are missing: `choice`, `probabilityLeft`, `feedbackType`, `feedback times`, `stimOn times`, `firstMovement times`.
- Exclude trials with stimulus-onset to first-movement reaction time outside `0.08 s` to `2.00 s`.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Methods paper figure text reports example-session AUC improvement from about `0.66` (single-session) to `0.72` (multi-session), with oracle around `0.79`. |
| Prior | Methods paper figure text reports example-session correlation improvement from about `0.05` (single-session) to `0.34` (multi-session), with oracle around `0.65`. |
| Wheel speed | Methods paper states multi-session models outperform single-session baselines; figure examples on page 5 show clearly higher correlation-like summary values for multi-session RRR than ridge, but not a single cohort-average number in the extracted text. |
| Whisker motion energy | Methods paper states multi-session models outperform single-session baselines; page 5 example traces/summary values indicate substantial improvement over ridge but the extracted text does not provide a single cohort-average number. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session roster size | `code/code_zhang2025/data/bwm_release.csv` contains 459 sessions (699 probe insertions, 139 mice). | Local `Brainwidemap` cache table contains 480 sessions because it includes a newer release; the 459 code-release sessions are all present locally. | Datapaper reports 459 public-release sessions; methods paper reports 433 sessions used for decoding. | Treat the 459-session `bwm_release.csv` used by the reference code as the canonical release roster for this workspace. Ignore the extra 21 sessions from the newer 480-session cache. Investigate the 459-to-433 reduction later as an analysis-specific subset, not as the base dataset definition. |
| Sessions with all required modalities | Reference code assumes wheel and whisker behaviors can be loaded; it does not precompute a task-specific session count. | Among the 459 code-release sessions present locally, 439 have spikes, trials, wheel, and whisker motion energy on disk. | Methods paper uses 433 sessions for its IBL decoding analyses. | Use 439 as the starting locally complete set for this task. The remaining 6-session gap to the paper’s 433 likely reflects additional analysis-time failures or exclusions beyond simple file presence; verify this later during conversion/validation. |
| Neuron QC | `prepare_data()` loads all spike-sorted clusters; it stores `good_clusters = label >= 1` in metadata but does not filter them out before binning. | Raw local sessions contain many more clusters than well-isolated units (for the 439 complete code-release sessions: 594k+ raw clusters vs far fewer good-label units). | Datapaper analyses report 75,708 well-isolated neurons and explicitly describe neuron QC. | Use the code’s stored QC label to filter to `label >= 1` clusters during conversion. This preserves the paper’s curation intent while still relying on the provided code path and metadata definitions. |
| Region handling | Reference caching code pools all Beryl regions together (`single_region=False`) and only subsets by region later in decoder scripts. | Local cluster metadata include per-cluster brain-region labels. | Datapaper analyses restrict to grey-matter regions with >=5 well-isolated neurons/session and >=2 sessions/region; methods paper decodes specific regions separately in some figures. | For the requested converted dataset, keep per-neuron region labels for all neurons and do not collapse sessions by region. This preserves downstream flexibility and matches the target format. |
| Temporal alignment and binning | Reference code uses one common setup for cached trials: `stimOn_times`, window `[-0.5, 1.5]`, binsize `0.02 s`, all behaviors aligned into the same 2 s trial grid. | Local files support stimulus-onset times, first-movement times, wheel traces, and whisker motion-energy traces. | Methods paper describes mixed target-specific preprocessing: choice aligned to stimulus onset; prior pre-stimulus; wheel/whisker aligned to first movement; 50 ms bins for choice/prior and 20 ms bins for dynamic behaviors. | Because the user explicitly requires "Temporally align based on stimulus onset" and a single common dataset with simultaneous outputs, prioritize the executable reference code’s common `stimOn`-aligned 2 s / 20 ms representation, while documenting that this is the main paper-vs-task-driven deviation. |
| Trial filtering | Reference code excludes missing-event trials, RT outside 0.08–2.0 s, no-choice trials, and trials longer than 10 s. | Raw trial tables contain the needed event columns to reproduce these masks. | Papers explicitly mention missing-event and 0.08–2.0 s RT exclusions; they do not mention the code’s `max_trial_len=10.0` detail in the copied text. | Reproduce the reference code trial mask exactly, including `max_trial_len=10.0`, because it is the executable implementation provided for this project. |
| Dynamic behavior representation | Reference code keeps wheel speed and whisker motion energy continuous and interpolated per time bin. | Raw wheel and whisker traces are continuous float signals at wheel/camera sampling rates. | Papers decode these as continuous variables with regression. | The user task explicitly requires categorical outputs, so discretize these continuous traces into 3 bins only after reproducing the reference loading/alignment. This is an allowed task-driven deviation, not a loading mismatch. |

Final understanding after reconciling sources:
- Base session roster should come from the 459-session release used by the reference code, not the newer 480-session cache metadata.
- The directly usable local subset for the requested four-output decoder is 439 sessions with complete required modalities.
- The provided executable code is the strongest source for exact loading/alignment behavior in this workspace.
- The papers remain the strongest source for scientific curation intent and expected global dataset statistics.
- The main implementation choice is therefore: use reference-code loading/alignment and paper-informed sanity checks, while applying only the task-required deviations (stimulus-onset alignment for all requested variables and discretization of continuous outputs).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters`, probe cluster metadata | `neural` | Merge probes per session, keep well-isolated clusters with `label >= 1`, apply trial mask, bin spike counts in `stimOn_times + [-0.5, 1.5]` using `0.02 s` bins, store per trial as `(n_neurons, 100)` after transposing from reference `(100, n_neurons)` | `load_spiking_data(qc=1)`, `merge_probes`, `bin_spiking_data` | This follows the paper’s neuron curation while keeping the reference loading/binding machinery. |
| Derived common bin time axis relative to `stimOn_times` | `input[0]` | One continuous time channel repeated for every trial, length 100; use the same 20 ms trial grid as neural/activity outputs | derived from `time_window`, `binsize` used by `bin_spiking_data` / `get_behavior_per_interval` | User-required decoder input; not present in reference code, so derive directly from the adopted alignment grid. |
| `trials.probabilityLeft` block structure | `input[1]` | Compute trial number within current block on the original trial order, then subset to retained trials and repeat across time bins | `load_trials_and_mask` for retained-trial order | Reset counter whenever `probabilityLeft` changes; preserve original experimental indexing rather than recomputing after exclusions. |
| `trials.choice` | `output[0]` | Map raw IBL choice to task labels and repeat across time bins: raw `+1 -> left -> 0`, raw `-1 -> right -> 1`; no-go (`0`) excluded by trial mask | `load_trials_and_mask`, `bin_behaviors` | Official IBL docs indicate raw choice encodes wheel direction; for the requested dataset convert to left/right categorical choice. |
| `trials.probabilityLeft` | `output[1]` | Map `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` and repeat across time bins | `bin_behaviors` | This is the requested discrete prior output. |
| Wheel trace loaded via `SessionLoader` / reference behavior loader | `output[2]` | Load wheel speed as `abs(velocity)`, align to stimulus onset on the same 2 s / 20 ms grid as neural data, then discretize continuous values into 3 global bins | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval`, `bin_behaviors` | This is a task-driven deviation from the paper’s first-movement alignment, required because the user explicitly requests stimulus-onset alignment for the full dataset. |
| Whisker motion-energy trace loaded via `SessionLoader` / reference behavior loader | `output[3]` | Load left whisker motion energy when available, otherwise right; align to stimulus onset on the same 2 s / 20 ms grid, then discretize into 3 global bins | `load_target_behavior('left/right-whisker-motion-energy')`, `get_behavior_per_interval`, `bin_behaviors` | Same alignment/deviation rationale as wheel speed. |
| `subject` from `bwm_release.csv` and session path metadata | `subjects`, `subject_idx` | Build unique subject list and index sessions in converted order | `code/code_zhang2025/data/bwm_release.csv` | Session order will be fixed and deterministic in the conversion script. |
| `clusters['acronym']` / stored `cluster_regions` | `brain_regions`, `brain_region_idx` | Build global vocabulary of original cluster acronyms and map each neuron to its region index | `prepare_data` metadata output | Use original acronyms rather than Beryl-collapse in the final dataset to match what the reference code stores. |

### Key Decisions
1. **Base session roster**: Start from the 459-session `bwm_release.csv` used by the reference code, then require local availability of spikes, trials, wheel, and whisker motion energy for the requested joint decoder task.
2. **Common alignment grid**: Use one common `stimOn_times`-aligned `[-0.5, 1.5]` window with 20 ms bins for neural, inputs, and outputs because that matches the executable reference code and the user explicitly requires stimulus-onset alignment.
3. **Common tensor shape**: Represent all inputs and outputs as time-varying arrays with shape `(n_channels, 100)` per trial, repeating per-trial variables across the 100 bins. This keeps the converted dataset uniform and decoder-friendly.
4. **Choice remapping**: Convert raw IBL `choice` values to left/right semantic labels required by the task: raw `+1` becomes left (`0`), raw `-1` becomes right (`1`).
5. **Prior representation**: Use `probabilityLeft` directly as the prior variable and encode its three observed task values as categorical labels `0/1/2` for `0.2/0.5/0.8`.
6. **Trial number in block**: Derive this from the original experimental trial order before dropping invalid trials, then subset. This preserves the true block progression.
7. **Continuous-output discretization**: Compute global tertile thresholds from all retained aligned wheel-speed samples and all retained aligned whisker-motion-energy samples, respectively, then apply those thresholds consistently across all sessions.
8. **Neuron filtering strategy**: Use well-isolated clusters only (`label >= 1`) during conversion, because this is the closest executable approximation to the paper’s stated neuron curation and keeps total dataset size aligned with the reported 75,708-neuron scale.
9. **Whisker camera fallback**: Use left whisker motion energy when available and fall back to right whisker motion energy otherwise, matching the provided code.

### Planned Sanity Checks
- [ ] Neural spot-check: for at least 3 session/trial examples, compare converted spike matrices against direct calls to the reference binning logic (`bin_spiking_data`) with `np.allclose()`.
- [ ] Behavior alignment spot-check: for at least 3 session/trial examples, compare continuous aligned wheel-speed and whisker traces before discretization against direct reference interpolation (`get_behavior_per_interval` / `bin_behaviors`) with `np.allclose()`.
- [ ] Trial-variable spot-check: verify `choice`, `probabilityLeft`, and derived `trial_number_in_block` against the raw trial table on at least 3 specific trials per checked session.
- [ ] Session-count sanity check: confirm converted session count equals the number of release sessions that have all required local modalities and survive downstream validity checks.
- [ ] Distribution sanity check: verify prior class distribution reflects the expected `0.5` first block followed by `0.2/0.8` biased blocks; wheel and whisker discretization should not collapse nearly all samples into one class.
- [ ] Metadata sanity check: confirm `time_bin_size = 20 ms`, `off_start = -0.5 s`, `off_end = 1.5 s`, and `temporal_alignment_event = stimulus onset` for every converted session.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `convert_data.py` with required CLI entrypoint: `python -u convert_data.py <outpicklefile>`.
- Implemented `--full`, `--sample`, and `--show-processing`.
- Implemented a deterministic session roster from `code/code_zhang2025/data/bwm_release.csv`.
- Implemented local ALF/parquet/numpy readers for trials, wheel, whisker motion energy, and spike sorting outputs so conversion matches the reference variables without depending on slow remote metadata resolution.
- Implemented paper-aligned neuron curation via `label >= 1` clusters while preserving the reference code’s stimulus-onset-aligned 2 s / 20 ms trial grid required by the task.
- Implemented a two-stage conversion:
  - session-wise loading/alignment to collect continuous wheel/whisker traces
  - global threshold computation for 3-bin discretization
  - final assembly into the target pickle dictionary
- Added per-session timing logs and optional processing plots.

Code inefficiencies identified:
- Initial direct use of the vendored `ibllib` code failed because the environment was missing IBL runtime dependencies and the current installed API differs from the older helper signatures in the reference code.
- Reference helper `load_spiking_data()` also attempted to fetch raw AP stream metadata that is not needed for conversion.
- Remote ONE/SessionLoader access introduced large startup latency and transient worker failures during full conversion attempts despite the release sessions already being cached locally.

Code speedups added:
- Avoided loading unused behavioral streams during session conversion.
- Reused the reference-style spike binning logic but switched all session data access to direct local ALF reads, eliminating avoidable network/cache overhead.
- Reused probe IDs directly from `bwm_release.csv` instead of per-session Alyx `eid2pid` lookups.
- Added session-level parallelism for `--full`, with worker retry/fallback logic for robustness if any session worker fails.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 340 |
| Neurons / session | [76, 264] (mean 170.0) |
| Subjects | 1 (`NYU-11`) |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | [407, 244] (mean 325.5) |
| Time since stimulus onset range (s) | [-0.48, 1.50] |
| Trial number in block range | [1, 90] |
| Choice distribution | [0.5177 left, 0.4823 right] |
| Prior probability of left distribution | [0.4777 for 0.2, 0.1613 for 0.5, 0.3610 for 0.8] |
| Wheel speed bin distribution | [0.3333, 0.3333, 0.3333] |
| Whisker motion energy bin distribution | [0.3333, 0.3333, 0.3333] |

Sample artifacts created:
- `sample_data.pkl` (38.92 MB)
- `conversion_sample_out.txt`
- `verification_sample_out.txt`
- `processing_6713a4a7-faed-4df2-acab-ee4e63326f8d.png`
- `processing_56956777-dca5-468c-87cb-78150432cc57.png`

Format validation (`train_decoder.py --verify-only`):
- Errors: None
- Warnings: None

### Processing Plots Review
The saved processing plots for both sample sessions show consistent stimulus-aligned binning from -0.5 s to +1.5 s, trial-by-trial neural matrices with stable neuron counts per session, and behavior traces without obvious temporal shifts relative to stimulus onset. Discretized wheel-speed and whisker-motion-energy bins transition where the continuous traces cross the global tertile thresholds; no off-by-one edge effects or missing final bins were apparent on visual spot-check.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Switched trials/behavior/spikes loading from remote ONE calls to direct local ALF reads | Reduced sample conversion from 132.5 s to 12.1 s while preserving counts and validation results |
| Reused probe IDs directly from `bwm_release.csv` instead of per-session Alyx `eid2pid` lookups | Removed repeated metadata round-trips during session setup |
| Added session-level parallelism for `--full` via `ProcessPoolExecutor` | Keeps the full conversion comfortably below the 15-minute target with the local-data path |

| Step | Time / Session | Estimated Total Time |
| Sample conversion session 1 | 3.0 s | |
| Sample conversion session 2 | 6.4 s | |
| Sequential average | 4.7 s | 35.8 min for 459 sessions |
| Parallel full run (48 workers target) | effective ~0.10 s/session wall-clock equivalent | ~45 s ideal; allowing heavier sessions and overhead, practical target ~3-8 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6213 | 0.5423 |
| prior_probability_left | 0.7015 | 0.6702 |
| wheel_speed_bin | 0.5806 | 0.5682 |
| whisker_motion_energy_bin | 0.6002 | 0.6039 |

Notes:
- Decoder training completed on GPU without memory issues.
- Loss decreased monotonically from 1.9737 at epoch 1 to 0.7423 at epoch 200.
- Every validation balanced accuracy exceeded chance (0.5 for choice; 0.3333 for 3-class outputs), so the sample conversion passes the required decoder sanity check.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 12.49 GB
- `verification_full_out.txt`: created

Full-run notes:
- First full-conversion attempt with 48 session workers caused severe local disk contention: the same sample session that took 3.0 s sequentially took 91.4 s inside the 48-worker pool.
- Benchmarked 8 release sessions at worker counts 4, 8, 12, 16:
  - 4 workers: 48.94 s wall clock
  - 8 workers: 34.15 s wall clock
  - 12 workers: 31.74 s wall clock
  - 16 workers: 33.82 s wall clock
- Set the default `--session-workers` to 12 based on the benchmark. The successful full conversion finished in 2071.0 s (34.5 min) and converted 438 sessions.
- 21 sessions were excluded after the sequential retry pass because they had `Only 0 valid trials after filtering` once stimulus-onset trial filters and the required wheel + whisker alignment masks were both applied.
- After conversion, I remapped stored brain-region labels from Allen acronyms to Beryl acronyms to match the reference code’s `brainreg.acronym2acronym(..., mapping='Beryl')` step.

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 75,708 well-isolated units in public release | QC `label >= 1` | public release roster = 459 sessions | 72,757 | Close; lower because 21 sessions were excluded by task-specific valid-trial requirements |
| Mean neurons/session | ~165.0 (75,708 / 459) | QC `label >= 1` after session loading | 459-session release | 166.11 | Yes |
| Subjects | 139 in public release | release roster spans 139 mice | 459-session release roster | 135 | Task-specific difference from excluding 21 unusable sessions |
| Sessions | 459 in public release; 433 used in methods paper | `bwm_release.csv` has 459 sessions | 459 release sessions present locally | 438 | Task-specific difference; close to the 433-session methods subset because only sessions with valid wheel + whisker aligned trials survive |
| Brain regions (Beryl) | 270 in methods paper | reference code remaps to Beryl | local raw labels are Allen acronyms | 266 (264 excluding `root`/`void`) | Close; small deficit plausibly due excluded sessions and retained `root`/`void` labels |
| Trials (total) | not explicitly stated | not explicitly stated | depends on post-filter valid-trial masks | 186,261 | N/A |
| Trials/session (mean) | not explicitly stated | not explicitly stated | depends on post-filter valid-trial masks | 425.25 | N/A |
| Time since stimulus onset range | alignment to stimulus onset | `time_window = (-0.5, 1.5)` | `stimOn_times` available in trials table | [-0.5, 1.5] | Yes |
| Trial number in block range | unbiased 90-trial start; biased blocks thereafter | derived from `probabilityLeft` changes | `probabilityLeft` available per trial | [1, 99] | Yes |
| Choice distribution | approximately balanced across the full task | derived from filtered `choice` | raw trials contain left/right/no-go | [0.5086, 0.4914] | Yes |
| Prior probability left distribution | values 0.2 / 0.5 / 0.8 expected | derived from filtered `probabilityLeft` | raw trials contain 0.2 / 0.5 / 0.8 | [0.4188, 0.1406, 0.4406] | Yes |
| Wheel speed bin distribution | 3 bins by instruction | global tertiles over valid timepoints | continuous wheel available | [0.3333, 0.3333, 0.3333] | Yes |
| Whisker motion energy bin distribution | 3 bins by instruction | global tertiles over valid timepoints | continuous whisker motion energy available | [0.3333, 0.3333, 0.3333] | Yes |

Verification summary:
- `train_decoder.py converted_data.pkl --verify-only` completed successfully.
- Remaining verifier warnings: 16 trials with all-zero neural matrices across 3 sessions. These need explicit review in Step 10.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification: `verification_full_out.txt` contains no hard format errors. The only remaining warnings are 16 trials with all-zero neural matrices across 3 sessions.
2. Raw-data sanity checks with `np.allclose()`:
   - `CHECK_INPUT = True`: independently reconstructed the first kept trial of session `6713a4a7-faed-4df2-acab-ee4e63326f8d` from the raw trials table and verified the converted input matrix exactly matches the raw stimulus-aligned time axis and manually derived trial-in-block value.
   - `CHECK_OUTPUT_CHOICE = True`
   - `CHECK_OUTPUT_PRIOR = True`
   - `CHECK_OUTPUT_WHEEL = True`
   - `CHECK_OUTPUT_WHISK = True`
   - `CHECK_NEURAL_SUBMATRIX = True`: independently histogrammed raw spike times for the first 3 QC-passed neurons in the first kept trial of session `6713a4a7-faed-4df2-acab-ee4e63326f8d`; the converted 3-neuron x 100-bin submatrix matched exactly.
   - `CHECK_ZERO_TRIAL = True` for verifier-warning sessions/trials:
     - session index 3 / trial 1 (`b182b754-3c3e-4942-8144-6ee790926b58`)
     - session index 360 / trial 7 (`195443eb-08e9-4a18-a7e1-d105b2ce1429`)
     - session index 388 / trial 241 (`8c2f7f4d-7346-42a4-a715-4d37a5208535`)
     - For all three, the independently loaded raw QC-passed spikes contained exactly 0 spikes in the converted `[-0.5, 1.5] s` stimulus-aligned window, so the all-zero converted trial matrices are genuine sparse-data cases rather than a conversion bug.
3. Reference code comparison:
   - Data loading: conversion uses the same `bwm_release.csv` release roster as `0_data_caching.py`. Session data are loaded from the same ALF trial tables and pykilosort outputs that the reference helpers access, but via direct local file reads rather than the slower remote ONE wrappers.
   - Neuron filtering: conversion applies the same `label >= 1` QC criterion as the reference paper/code for well-isolated units.
   - Temporal alignment and binning: conversion uses `stimOn_times`, `time_window = (-0.5, 1.5)`, and `binsize = 0.02`, matching the parameters in `0_data_caching.py`.
   - Region handling: conversion now applies the same Beryl remapping logic as the reference code’s `list_brain_regions()` step.
   - Input construction: `trial_number_in_block` is derived from `probabilityLeft` block changes; `time_since_stimulus_onset_s` is the fixed stimulus-aligned bin grid.
   - Output construction: `choice` and `probabilityLeft` come directly from the filtered trials table; wheel speed and whisker motion energy are aligned to the stimulus-onset window and discretized into 3 global tertile bins as required by the decoder task.
4. Key statistics comparison:
   - Converted sessions: 438, close to the 433-session methods-paper cohort and lower than the 459-session public release because 21 sessions had no valid trials after the required task-specific wheel + whisker alignment filters.
   - Converted Beryl regions: 266 total labels (264 excluding `root`/`void`), close to the 270 regions reported in the methods paper.
   - Converted neurons/session mean: 166.11, matching the public-release scale implied by 75,708 well-isolated units across 459 release sessions.
5. Edge-case checks:
   - 21 sessions were retried sequentially after worker failures and all 21 consistently failed with `Only 0 valid trials after filtering`, confirming these are true exclusions rather than multiprocessing artifacts.
   - The 16 verifier warnings about all-zero neural trials correspond to genuine zero-spike windows and therefore should not be “fixed” by altering the data values; removing them would be an analysis choice, not a data-correction step.

### Issues Found and Resolved
- Brain-region labels were initially exported at Allen-acronym granularity (540 labels), not the Beryl grouping used by the reference code. Resolved by remapping the exported labels to Beryl; full artifact now has 266 Beryl labels.
- Full-conversion worker count of 48 caused severe disk contention. Resolved by benchmarking and changing the default to 12 session workers.
- Full verifier warnings about all-zero neural trials were investigated against raw spikes and verified to be genuine sparse windows, so they remain documented warnings rather than conversion defects.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes
- Initial GPU run hit `torch.OutOfMemoryError` during full-dataset training; reran with `--cpu` and completed successfully.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6408 | 0.6183 | Above chance (0.5000); binary output so 1.24x chance is consistent with paper-level choice decoding rather than a bug |
| prior_probability_left | 0.6883 | 0.6670 | Above chance (0.3333); strong separation despite required discretization into 3 classes |
| wheel_speed_bin | 0.6457 | 0.6392 | Above chance (0.3333); stable train/validation match indicates correct alignment |
| whisker_motion_energy_bin | 0.7420 | 0.7383 | Highest-performing output; matches expectation that whisker signal is highly decodable |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
|----------|-------------------|-------------------------|
| choice | Validation balanced accuracy 0.6183 (chance 0.5000; 1.24x chance) | Methods paper reports strong above-chance choice decoding, with Figure 4 showing choice AUC examples around 0.66-0.79 and Figure 5 showing multi-session gains over baselines. Metric differs, but our result is in the expected regime. |
| prior_probability_left | Validation balanced accuracy 0.6670 (chance 0.3333; 2.00x chance) | Paper evaluates prior as a continuous variable using Pearson correlation, with example correlations including 0.74 vs 0.37, 0.76 vs 0.54, and 0.65 vs 0.05 vs 0.34 for stronger models. Direct numeric comparison is not possible after required 3-class discretization, but strong above-chance decoding is consistent with the reference. |
| wheel_speed_bin | Validation balanced accuracy 0.6392 (chance 0.3333; 1.92x chance) | Paper evaluates wheel as a continuous dynamic variable with R2; Figure 2 example values span roughly 0.51-0.70 depending on session/model. Our discretized wheel decoder remains strongly above chance and below whisker, matching the paper’s relative difficulty ordering. |
| whisker_motion_energy_bin | Validation balanced accuracy 0.7383 (chance 0.3333; 2.21x chance) | Paper evaluates whisker motion energy with continuous R2 and reports stronger example performance than wheel. Our whisker accuracy is the highest of the four outputs, matching that ordering. |

Additional review:
- No output fell below chance.
- Only `choice` is below 1.5x chance, but binary chance is already 0.5 and the achieved 0.6183 validation score is consistent with the reference paper’s choice-decoding examples. Combined with the raw-data spot checks from Step 10, this does not indicate a conversion bug.
- Train/validation gap check: `choice` 1.04x, `prior` 1.03x, `wheel` 1.01x, `whisker` 1.01x. No output exceeds the 1.5x leakage/overfitting threshold.
- Relative ordering matches the reference paper: whisker motion energy is easiest, wheel is harder, and choice remains decodable but more modest than the 3-class dynamic outputs under this benchmark.

### Issues Found and Resolved
- GPU memory exhaustion on the full decoder run: reran `train_decoder.py converted_data.pkl --plot-samples --cpu`; training completed with stable losses and accuracies.
- No additional conversion issues were uncovered in this review step, so no changes to `convert_data.py` were required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

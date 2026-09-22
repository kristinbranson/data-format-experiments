# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide map of neural activity during complex behaviour
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest` (423079 bytes)
- `code/`
- `CONVERSION_NOTES.md` (5085 bytes)
- `data/`
- `dataarchitecture.pdf` (1104210 bytes)
- `datapaper.pdf` (19801094 bytes)
- `decoder.py` (89127 bytes)
- `docker-compose.yaml` (1897 bytes)
- `Dockerfile` (3830 bytes)
- `methodpaper.pdf` (25020189 bytes)
- `methods.txt` (17557 bytes)
- `stage_cache.sh` (4416 bytes)
- `train_decoder.py` (7641 bytes)

Environment verification:
- Python 3 runs successfully.
- numpy and torch import successfully (versions checked in terminal).
- Required notes-file checkpoint passed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Load spike times/clusters and cluster metadata for an IBL probe insertion. |
| `merge_probes` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Concatenate probes while remapping cluster IDs to a session-unique index. |
| `load_trials_and_mask` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Load trials and construct validity masks, including finite event times and a maximum trial-length rule. |
| `bin_spiking_data` / `bin_spikes` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Histogram spike times into fixed event-aligned intervals and retain a consistent cluster ordering. |
| `load_target_behavior` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Load wheel velocity/speed, camera motion energy, pupil and DLC-derived behavioral streams. |
| `get_behavior_per_interval` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING/CURATION | Slice behavior around trial events, verify endpoint coverage, and linearly interpolate to common bins. |
| `load_anytime_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Parallel loading of session-wide wheel, whisker, pupil, paw and nose streams. |
| `bin_behaviors` | `code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Build per-trial categorical variables and aligned continuous behavioral arrays plus masks. |
| `prepare_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Orchestrate all probe, trial, behavior, region, cluster-QC, and metadata loading for one session. |
| caching loop | `code_zhang2025/src/0_data_caching.py` | PROCESSING | Apply preparation and spike binning session by session using decoder configuration. |

### Notes
- Repository: `code_zhang2025`, code associated with the correlation-based neural decoding methods paper. The main conversion-relevant implementation is `src/utils/ibl_data_utils.py`; `src/0_data_caching.py` is its pipeline call site.
- Neural source is electrophysiology, not imaging; delta-F/F is therefore not applicable.
- Spike data are loaded per probe and merged. Cluster metadata include Allen acronym, channel, depth, UUID and QC fields. The code records `good_clusters = (clusters['label'] >= 1)`. This QC criterion must be reconciled with the data-paper/BWM inclusion table before final mapping.
- `prepare_data` calls `load_trials_and_mask(..., max_trial_len=10.0)`. Trial validity is tied to event availability/order and maximum duration; exact applicable mask will be retained when mapping the local files.
- Spike binning uses histogram counts in fixed intervals around an alignment event, with one common ordered cluster axis. The reference code warns that the requested behavior bin size is exact only when it evenly divides the alignment interval.
- Behavior alignment computes interval starts/ends from the selected trial event (supported examples include `stimOn_times`, `firstMovement_times`, and `feedback_times`). It checks stream coverage at both endpoints and interpolates onto a common grid. Missing streams/intervals are represented by masks rather than silently imputed.
- Relevant loaded streams for this task are `wheel-speed`, left/right whisker motion energy, trial `choice`, and trial `probabilityLeft` (called `block` in `bin_behaviors`). The generic code prefers left whisker motion energy and falls back to right if left is unavailable.
- `bin_behaviors` also exposes reward and contrast, but these are not requested decoder outputs. All available variables will still be inventoried in Step 2.
- The helper `align_spike_behavior` appears unused by the caching call site and uses Python list `and` instead of elementwise conjunction; it will not be copied. Alignment will instead use explicit NumPy boolean masks and assertions.
- Reference processing to preserve where applicable: event-relative fixed windows, spike counts (not smoothed rates unless later text specifies otherwise), consistent neuron ordering, endpoint-validated linear interpolation of continuous behavior, and explicit trial/behavior masks.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/one_cache` is an IBL ONE cache. It contains three database snapshots: `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and rolling `Brainwidemap`.
- Each snapshot contains `sessions.pqt` (session index), `datasets.pqt` (dataset UUID, size/hash/QC/existence and ALF relative path), `QC.json`, and `cache_info.json`. `/app/data/one_cache/.rest` contains 6,483 cached Alyx/ONE REST responses, including full session and probe-insertion records.
- No ALF signal payloads are currently downloaded: outside `.rest`, the only 12 files are six JSON and six parquet metadata files (14,596,120 bytes). Required arrays must therefore be fetched through ONE after selecting the reference-consistent release/subset.
- Native organization after download is subject/date/session-number followed by ALF collections. Trial and wheel files are session-level; spikes/clusters are probe-level (`alf/probeXX/pykilosort`, sometimes revisioned); camera timestamps and ROI motion energy are session-level ALF camera objects.
- Available required variables include `_ibl_trials.table.pqt` (trial event times, choice, contrasts, feedback/reward, `probabilityLeft`), `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`, `{left,right}Camera.ROIMotionEnergy.npy`, `_ibl_{left,right}Camera.times.npy`, `spikes.times.npy`, `spikes.clusters.npy`, and cluster anatomy/QC metadata (`clusters.brainLocationAcronyms_ccf_2017.npy`, `clusters.metrics.pqt`, channels/depths/UUIDs).
- The 2025 Q3 `datasets.pqt` is a small partial/update cache (5,339 rows) and omits core spike/trial/wheel entries; the rolling `Brainwidemap` table contains the complete file inventory. Release selection must be resolved against paper/code statistics in Steps 3–4.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | Not directly available without payload/QC filtering. Metadata-inferred raw clusters in rolling table: 996,929 across 1,396 default cluster-channel arrays; this is explicitly **not** the curated neuron count. |
| Neurons / session | Not available until payloads are loaded and reference QC/region filtering is applied. |
| Subjects | 115 (2022 Q4); 139 (2025 Q3); 143 (rolling Brainwidemap) |
| Sessions / subject | 2022: mean 3.08, range 1–9; 2025 Q3: mean 3.30, range 1–13; rolling: mean 3.36, range 1–13 |
| Sessions | 354 (2022 Q4); 459 (2025 Q3); 480 (rolling Brainwidemap) |
| Probe insertions | 576 for 354-session release; 700 for 459-session release (from cached Alyx insertion records) |
| Trials (total) | Raw Alyx `n_trials`: 229,015 for all 354 sessions; 294,297 for all 459 sessions. Rolling cache REST coverage is incomplete for 21 newer sessions. |
| Trials / session | 2022 mean 646.94, range 45–1,525; 2025 Q3 mean 641.17, range 45–1,525 |
| Correct trials | 187,544/229,015 (81.89%) in 2022 metadata; 241,107/294,297 (81.93%) in 2025 metadata |

### File and Variable Details
- 2022 complete inventory: 38,814 dataset rows, 547 probe spike sets, trial tables for 354 sessions, wheel for 354, left ROI motion energy for 330, right for 328.
- Rolling complete inventory: 1,398 probe spike sets (8,388 rows / six spike array types), trial table entries for 459 sessions in the visible cache, wheel entries for up to 490 revisions/session records, left ROI motion energy for 432 and right for 433 sessions. Counts can exceed sessions due to revisions.
- Spike arrays are numeric NPY; trial tables/cluster metrics/camera features are parquet; cluster UUIDs are CSV. Signal dimensions/dtypes cannot be inspected until download.
- Valid-data indicators available in metadata include dataset QC, default revision, session extended QC, cluster metrics/labels, and missing-stream presence. These will be combined only according to reference curation, not by blanket session QC exclusion.
- Sanity note: metadata-derived cluster counts use NPY file size/header assumptions and are only an inventory diagnostic; they are not suitable as converted neuron counts.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Units (total, spike sorted) | 621,733 | Data paper Methods: “Out of the 621,733 units collected…” |
| Well-isolated neurons | 75,708 | “…75,708 were considered well-isolated neurons.” |
| Subjects | 139 in the 2025 Q3 BWM snapshot used by the methods-paper-era cache | Cache/paper release metadata; matches the 459-session release available locally. |
| Sessions | 459 in 2025 Q3 curated release metadata | Local public release metadata; methods paper scales IBL models up to 73 training sessions plus fixed test sessions rather than using every BWM session. |
| Probe insertions | 700 for the 459 sessions | Cached Alyx insertion records for the paper-era release. |
| Trials (raw metadata total) | 294,297; mean 641.17/session | Cached Alyx records; paper applies trial exclusions below, so converted valid total must be lower. |
| Trials / session | Raw range 45–1,525 | Cached Alyx records. |
| Neural data time bin | Methods-paper decoder config uses fixed event-aligned spike-count bins; data paper wheel decoder uses 20 ms; 12.5 ms is only for continuous Granger analysis. | “We averaged wheel values in nonoverlapping 20-ms bins… Spike counts were similarly binned.” |
| Behavior data time bin | 20 ms for data-paper wheel decoding; native whisker camera resolution is 60 Hz left or 150 Hz right. | Data paper Methods/Video analysis. |
| Correct rate (metadata) | 81.93% | 241,107 correct of 294,297 raw trials in cached Alyx records. |
| Well-isolated fraction | 12.18% | 75,708 / 621,733 units. |
| Reportable region requirement | ≥5 well-isolated neurons per session and ≥2 sessions | Data paper Methods. |
| Wheel-speed regional decodability | Significant in 81% of reportable areas (163 areas) | Data paper Results: “Wheel speed was decodable from 81% … (163 …).” |

### Processing Details
- Experiment: mice perform a visual two-alternative forced-choice task. `probabilityLeft` is the blockwise prior probability that the stimulus is on the left (native levels 0.2, 0.5, 0.8). Choice is wheel response direction.
- Required alignment is stimulus onset. Native `stimOn_times` is in session seconds. Data-paper stimulus decoding uses neural activity from 0 to 150 ms after stimulus onset; its wheel decoder instead aligns −0.2 to +1.0 s around first movement. Because the requested outputs include time-varying wheel/whisker and explicitly demand stimulus alignment, a longer stimulus-aligned window is required and will be chosen in Step 5 while retaining the reference fixed-bin/interpolation logic.
- Neural activity is spike counts in non-overlapping bins, with neurons in the same session/region combined across probes. Delta-F/F is not applicable.
- Reference behavior alignment checks both interval endpoints and linearly interpolates continuous streams to a common grid. Missing streams/intervals are masked, not filled silently.
- Wheel speed in the reference utility is derived from wheel position/timestamps through IBL `SessionLoader`; the data-paper wheel decoder averages wheel and spike values in 20 ms bins and uses a causal W=10-bin history. Our categorical output requirement supersedes continuous Lasso/R2 output but not stream derivation/alignment.
- Whisker motion energy is “the mean across pixels of the absolute value of the difference between adjacent frames” within DLC-defined whisker-pad ROIs. Left camera is 60 Hz, right 150 Hz. Reference code prefers left and falls back to right; this will be retained and resampled to neural bins.
- Methods-paper tensors are trials × time × neurons; decoder loading standardizes neural values using training-data neuron/time statistics. Its targets include choice, prior, wheel speed, and whisker motion energy. It benchmarks 5 regions across 10 IBL sessions and scaling runs with 5–73 training sessions plus five fixed tests.
- Binary data-paper decoding uses inverse-frequency weighting and balanced accuracy. Wheel is evaluated with R²; methods-paper continuous plots also report correlations. Our provided validator dictates final metrics and categorical three-bin behavior outputs.

### Curation Steps

**Neuron curation rules**:
- Apply the data paper’s RIGOR single-unit criteria: amplitude >50 µV, noise cutoff <20 µV, and refractory-period-violation criterion. The resulting reference count is 75,708 well-isolated neurons from 621,733 units.
- Use grey-matter Allen CCF regions. For region-level reportability the paper requires at least five well-isolated neurons per session and recordings in at least two sessions; target format can preserve all qualifying grey-matter neurons while recording region labels, unless exact BWM inclusion tables already enforce this.
- Merge probes within a session and remap cluster IDs to a unique neuron axis.

**Trial curation rules**:
- Exclude trials if required events are absent/nonfinite or occur in an invalid order (paper and `load_trials_and_mask`).
- Exclude trials where stimulus-onset to first-movement latency is outside 0.08–2.00 s.
- Reference utility additionally enforces maximum trial length 10 s.
- Require complete neural, wheel, and whisker coverage over the selected stimulus-aligned window because both time-varying outputs are mandatory; do not impute missing streams.
- Preserve zero-contrast trials for this requested choice/prior/behavior task (the paper excludes them only for decoding stimulus side, which is not an output here).

### Decoders Trained
| Decoded variable | Accuracy / expectation |
|------------------|------------------------|
| Choice | Methods paper reports multi-session decoding improvement and above-chance accuracy; figure text reports about 2% relative average gain over single-session RRR across 10 sessions. |
| Prior | Example methods-paper Pearson correlations approximately 0.74–0.76 versus ridge examples 0.37–0.54; requested categorical prior should be clearly above 1/3 chance. |
| Wheel speed | Methods-paper example correlations approximately 0.80–0.86 versus ridge 0.52–0.66; data paper reports significant regional decodability in 81% of reportable areas. |
| Whisker motion energy | Decodable but harder at high frequencies; methods paper notes little captured information above 5 Hz and scaling benefits with more sessions. |

### Applicability Notes
- Data-paper binary stimulus/choice windows and first-movement-aligned continuous wheel setup cannot be copied literally because this task mandates stimulus alignment and time-varying categorical wheel/whisker outputs. Reference unit/trial QC, native stream derivation, endpoint coverage, fixed bins, and probe merging remain applicable.
- Prior is continuous in the methods paper but explicitly categorical here: 0.2→0, 0.5→1, 0.8→2.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Stable release | Generic ONE/BWM queries; methods code can consume pre-aligned per-EID datasets. | 2025 Q3 has 459 sessions/139 subjects and is a strict superset of 2022; rolling has 21 later sessions. | Methods paper is 2025 and benchmarks subsets/scaling up to 73 sessions; data paper headline counts match the stable paper-era release. | Use the frozen `2025_Q3_IBL_et_al_BWM` session set, resolving file records through the complete rolling dataset table. Do not include 21 post-freeze sessions. |
| Session scope | Generic caching loop can process queried sessions; benchmark figures use selected 10-session/5-region examples and larger scaling subsets. | All 459 frozen sessions have trial, wheel, spike, and metric records; 445 have at least one complete whisker camera stream. | Data paper describes the full BWM; task asks full converted dataset. | Start from all frozen sessions, excluding only those lacking mandatory streams and those failing reference QC/minimum-trial requirements. Do not mistake a benchmark subset for the full source dataset. |
| Unit QC | Loader supports `qc` and keeps `label >= qc`; preparation stores `good_clusters=(label>=1)` but can cache all units. | Cluster metrics/labels are available per probe. | Paper defines 75,708 well-isolated neurons via amplitude >50 µV, noise cutoff <20 µV, and refractory violation criteria. | Filter to merged IBL `label >= 1`, the reference pipeline’s operational encoding of passing unit QC; verify metric fields/spot checks and compare total to 75,708. Do not merely store the indicator while retaining noise units. |
| Region curation | Code carries Allen acronyms and can subset regions at decoder load time. | Anatomy arrays exist for all spike sessions. | Grey matter only; reportable regions need ≥5 neurons/session and ≥2 sessions. | Use Beryl/Allen mapping as in IBL utilities, exclude void/root/non-grey labels, retain good neurons; apply ≥5/session-region and ≥2-session-region rule if reproducing reportable-region population, and document counts. |
| Trial validity | `load_trials_and_mask` checks finite/order constraints and max length 10 s. | Native trial tables provide all required events and variables. | Paper also requires stimulus-to-first-movement latency 0.08–2.00 s. | Use the conjunction of code mask and paper latency criterion; preserve zero-contrast trials because stimulus side is not decoded. |
| Temporal setup | Generic code uses event-relative fixed bins and endpoint-validated interpolation; methods decoder uses trial × time × neuron arrays. | Wheel/camera streams have distinct native rates. | Data paper uses 20 ms wheel bins around first movement and 0–150 ms for stimulus decoding. | Task requirement overrides alignment/target representation: use stimulus onset for every stream, fixed common bins, and categorical wheel/whisker outputs. Preserve reference interpolation, spike counts, masks, and no extrapolation. |
| Whisker camera | Code prefers left and falls back to right. | Frozen set: 432 complete left, 433 complete right, 445 either, 420 both; 14 have neither. | Motion energy is mean absolute adjacent-frame difference in whisker ROI at camera rate. | Retain left-first/right-fallback behavior and exclude the 14 sessions lacking both; record camera used per session. |
| Continuous behavior targets | Methods paper predicts continuous prior/wheel/whisker and evaluates correlation/R². | Native values are continuous except three-level prior. | Decoder task explicitly requires categorical prior and 3-bin wheel/whisker. | Map prior exactly and discretize each continuous stream using training-independent, documented thresholds (planned in Step 5). |

### Investigation Correction
- Initial metadata-only planning considered all 459 frozen sessions. Subsequent inspection of the actual reference caching call site showed that BWM processing samples one EID per subject (`SEED=42`) and uses exact parameters `binsize=0.02`, `align_time=stimOn_times`, `time_window=(-0.5,1.5)`. Steps 4–5 were corrected before implementation.

### Final Consistent Understanding
- Source population is defined by frozen `bwm_release.csv` (459 sessions, 139 subjects, 699 insertions). Matching reference `0_data_caching.py`, conversion selects one first-listed EID per subject; subjects are ordered by `np.random.seed(42)` random selection. Full mode uses all 139 subjects (before mandatory-stream/QC exclusions), rather than treating repeated sessions from the same animal as independent.
- Session is the target-format session unit. All probes in an EID are merged because they share trials/behavior and are not independent.
- Neural matrices contain non-overlapping event-aligned spike counts, good units only, with stable neuron order and Allen-derived region labels.
- Trials satisfy both reference-code event/max-duration checks and data-paper reaction-time QC, plus complete stream coverage for the requested fixed window.
- All modalities share a stimulus-onset time grid. Wheel speed and whisker motion energy are interpolated only when raw coverage brackets the grid; no extrapolation/imputation.
- Any selected representative session without either whisker stream is excluded. Further session loss is allowed only for explicit QC, insufficient trials/units, or incomplete interval coverage.
- Expected final statistics must be compared against 75,708 well-isolated neurons and raw release totals, while explaining reductions caused by mandatory whisker availability and valid-window filtering.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters`; merged cluster QC/anatomy | `neural` | Merge probes, keep `label>=1` good grey-matter units, histogram counts into 20 ms bins from −0.5 to +1.5 s relative to `stimOn_times`; transpose to neuron × time; `float32` | `load_spiking_data`, `merge_probes`, `bin_spiking_data` | Counts, not smoothed rates; fixed stable UUID/order per session. |
| Bin centers relative to stimulus onset | `input[0]` | 100-element continuous row: −0.49, −0.47, …, 1.49 s | event interval logic in `bin_spiking_data` | Decoder input name `time_since_stimulus_onset_s`. A continuous time input is explicitly requested, so it is not a binary event indicator. |
| Native trial index within session | `input[1]` | Zero-based original trial number, repeated for all 110 bins | trial table index | Preserves gaps caused by trial QC and thus true trial position/block progression; continuous `float32`. |
| `trials.choice` | `output[0]` | Per-trial scalar: native −1 (left wheel/left choice) → 0; native +1 (right) → 1; reject 0/no-go | `load_trials_and_mask`, `bin_behaviors` | Shape `(4,)` is not possible for mixed temporal outputs, so all outputs are represented as 4×T; choice is repeated over time. This keeps a single valid target tensor per trial. |
| `trials.probabilityLeft` | `output[1]` | Exact tolerance mapping 0.2→0, 0.5→1, 0.8→2; repeat over time | `bin_behaviors` (`block`) | Unexpected levels excluded and reported. |
| Wheel position/timestamps | `output[2]` | Derive velocity with IBL `SessionLoader`/reference wheel interpolation, take absolute value for speed, interpolate/bin-average on common grid, discretize globally into low/medium/high | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | Time-varying categorical integer row. |
| Left camera ROI motion energy/times, else right | `output[3]` | Left-first/right-fallback; interpolate/bin-average to grid; discretize globally into low/medium/high | `load_target_behavior`, `bin_behaviors` | Time-varying categorical integer row; no cross-camera normalization within a session beyond quantile transformation. |
| Alyx session subject | `subjects`, `subject_idx` | Unique sorted subject IDs and integer session lookup | ONE session cache | Only subjects with retained sessions. |
| Cluster Allen acronym | `brain_regions`, `brain_region_idx` | Remap to Beryl acronyms; exclude root/void/unmapped; global sorted region vocabulary | `BrainRegions.acronym2acronym(..., mapping='Beryl')` | Stable per-neuron index. |

### Reference Parameters Confirmed
- `SEED=42`; frozen `data/bwm_release.csv`; one first-listed EID per selected subject.
- `interval_len=2`, `binsize=0.02`, `align_time=stimOn_times`, `time_window=(-0.5, 1.5)`.
- Reference behavior list includes choice, block/prior, wheel speed, and whisker motion energy.

### Key Decisions
1. **Release and session scope**: Use frozen `bwm_release.csv`. Exactly match reference `0_data_caching.py`: `np.random.seed(42)`, unique subjects, random subject order, then the first CSV-listed EID for each selected subject. Full mode selects all 139 subjects (one session/subject); sample mode takes the first two eligible sessions from that same deterministic ordering. Use the rolling table only to resolve full file records.
2. **Window and bin size**: Exactly use reference caching parameters: `interval_len=2`, `binsize=0.02`, `align_time=stimOn_times`, `time_window=(-0.5, 1.5)`. This gives 100 bins over `[−0.5, +1.5)` s relative to stimulus onset.
3. **Trial mask**: Require finite `stimOn_times`, `firstMovement_times`, `feedback_times`, `choice`, and `probabilityLeft`; valid event order; trial duration ≤10 s; first movement latency 0.08–2.00 s; choice ±1; prior in {0.2,0.5,0.8}; and complete neural/wheel/whisker temporal coverage. Keep zero-contrast trials.
4. **Unit mask**: Explicitly keep merged cluster QC `label>=1`, Beryl grey-matter labels only. Session-region reportability counts will be computed; because target format supports arbitrary brain regions, neurons are retained if good/grey-matter, while regions failing paper reportability are flagged and can be removed if required to match the 75,708 reference count.
5. **Choice sign**: IBL native `choice=-1` denotes a left wheel turn/left choice and maps to required left=0; `choice=+1` maps to right=1. This will be checked against raw trials on named examples.
6. **Continuous discretization**: Use pooled empirical tertiles computed from all valid finite samples after alignment, separately for wheel speed and whisker motion energy. Thresholds are fixed once computed and stored in metadata. Quantile bins create approximately balanced classes and are deterministic (`np.searchsorted` with documented right-edge handling). To avoid domination by high-frame-rate sessions, each aligned time bin contributes once; because every retained trial has 100 bins, sessions contribute according to valid trial count as in decoder training.
7. **Whisker camera**: Prefer left camera to match reference code, use right only when left is unavailable/unusable. Store camera side per session. Quantile discretization is pooled after alignment; metadata records side counts and thresholds.
8. **Output representation**: Use shape `(4, 100)` integer output arrays for every trial. Per-trial choice/prior are repeated across time, while wheel/whisker vary. This is required because a single NumPy array cannot mix scalar and temporal rows and is supported by the validator.
9. **Input representation**: Use shape `(2, 100)` `float32`; repeat original trial number over bins. Original rather than compressed retained-trial order preserves trial position and block structure.
10. **Memory and I/O**: Download only required ALF attributes, process one session at a time, cast counts/inputs to `float32` and outputs/region indices to compact integers, release raw spikes immediately, and pickle once. Cache downloads under `/app/data/one_cache` for reproducibility.
11. **Missing data**: Never extrapolate behavior. Drop individual trials without full window coverage; drop sessions with fewer than two valid trials or no good neurons. Do not synthesize values.
12. **Metadata**: Record release, EIDs/subjects, camera side, original/retained counts, unit counts, exclusion reasons, bin edges/centers, discretization thresholds, choice/prior mappings, QC rules, and source/reference versions.

### Planned Sanity Checks
- [ ] Raw neural check: independently load one session’s raw `spikes.times/clusters`, apply retained UUID mapping, run `np.histogram` for trial 5/neuron 3, and `np.allclose` against converted neural row.
- [ ] Raw input check: independently load raw trial table, compute selected original index and bin-center vector, and use `np.allclose` against both input rows.
- [ ] Raw output check: independently map raw choice/prior and interpolate wheel/whisker for three named trials; compare continuous intermediates and final categorical arrays with `np.allclose`.
- [ ] Verify exactly 100 bins for every neural/input/output trial and exact alignment edges `[stimOn−0.5, stimOn+1.5)`.
- [ ] Compare raw candidate/retained subjects, sessions, trials, probes, and good units to release/paper totals; account for every exclusion reason.
- [ ] Confirm class distributions: choice near balanced enough for decoding, prior only classes 0/1/2, and pooled behavior classes near one-third each.
- [ ] Confirm wheel values are nonnegative before discretization and motion energy finite/nonnegative.
- [ ] Check no duplicated neuron UUIDs after probe merging and every brain-region index is in range.
- [ ] Plot raw streams, interpolated values, bin edges, categories, and population spike counts for up to two sessions in `--show-processing` mode.
- [ ] Verify sample conversion uses the first two eligible sessions under the same deterministic full-session ordering and identical processing.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required positional output path, `--full` (default), `--sample`, and `--show-processing` modes. The script:
- Reads the staged read-only ONE cache directly from `/mnt/dataset/one_cache` without network access or copying raw data.
- Reproduces frozen BWM/seeded one-session-per-subject selection from reference `0_data_caching.py`.
- Selects coherent newest pykilosort revisions, filters `label>=1`, maps anatomy to Beryl, and merges probes.
- Applies reference trial QC and exact stimulus-aligned 20 ms grid over −0.5 to +1.5 s.
- Uses IBL wheel interpolation/filtered velocity and left-first/right-fallback motion energy.
- Computes pooled tertiles, constructs target-format arrays, validates every shape/value, records detailed metadata, and creates per-session processing plots.
- Passed `python3 -m py_compile /app/convert_data.py`.

Code inefficiencies identified:
- Spike arrays can exceed 50 million rows/probe; loading full arrays or looping over every spike/trial would be prohibitive.
- Copying staged session directories would waste hundreds of GB and network access is unavailable.

Code speedups added:
- Memory-map spike arrays and use `searchsorted` to read only each trial window.
- Vectorize cluster lookup, time-bin assignment, and count accumulation with `np.bincount`.
- Process/release one session at a time; load only required ALF arrays.
- Compute aligned behavior in compact `float32`; pickle once at the end.


---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (summed session populations) | 467 |
| Neurons / session | 218, 249 |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Sessions | 2 |
| Trials (total) | 443 |
| Trials / session | 262, 181 |
| Time input range | exact right edges −0.48 to +1.50 s, 20 ms spacing |
| Trial-number input range | 0–421; 0–502 (original indices retain QC gaps) |
| Choice distribution (trial-level) | left 46.3%, right 53.7% |
| Prior distribution (trial-level) | 0.2: 42.7%, 0.5: 19.6%, 0.8: 37.7% |
| Wheel bins (pooled timepoints) | approximately 33.33% each by construction |
| Whisker bins (pooled timepoints) | approximately 33.33% each by construction |
| Sample pickle size | 41.5 MB |

### Processing Plots Review
- Two final plots (`processing_ff96...png`, `processing_51e...png`) are 1950×1560 and nonblank (~23% nonwhite pixels).
- Population spikes, wheel speed, whisker motion energy, thresholds, and spike rasters all share the stimulus-relative x-axis; stimulus time zero is marked. No visual temporal offset or category-threshold anomaly was found.
- A stale plot from the pre-fix run was removed.

### Format Validation
- Required command completed and created `/app/verification_sample_out.txt`.
- `train_decoder.py --verify-only` reported **Data verification complete** with no errors or warnings.
- Every neural/input/output trial has T=100; input dimension 2; output dimension 4; region-vector lengths equal neuron counts.
- Output ranges are exactly choice 0–1 and the other outputs 0–2.

### Edge Case Found and Fixed
- The first seeded session initially failed because its wheel timestamps contain one exact duplicate at index 80. Whole-session exclusion was unjustified.
- Fixed by retaining the first finite sample at each duplicate timestamp before reference interpolation; applied equivalent finite/deduplication robustness to camera streams.
- Re-ran sample conversion and all validation from scratch: first two reference sessions now pass, with no skipped sessions.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Memory maps + per-trial `searchsorted` windows + `bincount` | Session processing is sub-second for sample sessions despite tens of millions of source spikes. |
| Direct mounted-cache reads | Avoids staging/copying gigabytes per session. |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion core | ~0.4–0.6 s for sample sessions; allow several seconds for larger probes | Well under 15 minutes for ≤139 candidate sessions; pickle I/O may dominate. |
| Sample total | ~seconds plus plot/verification startup | Completed successfully. |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- `/app/train_decoder_sample_out.txt` created; script finished successfully.

### Training Progress
- Loss decreased monotonically from 8.098 at epoch 5 to 0.657 at epoch 200.
- Test loss: 0.693.
- Train/validation gaps are small for every output (largest ratio about 1.09), with no concerning overfit or leakage signature.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Choice | 0.6536 | 0.5993 | 0.5000 |
| Prior probability left | 0.7643 | 0.7440 | 0.3333 |
| Wheel speed bin | 0.6610 | 0.6498 | 0.3333 |
| Whisker motion-energy bin | 0.6821 | 0.6661 | 0.3333 |

All validation accuracies exceed chance. Prior, wheel, and whisker exceed 1.5× chance; choice is 1.20× chance on only two sessions but remains clearly above chance and has correct sign/representation.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3.3 GB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; “Data format is valid, no errors or warnings” and “Data verification complete.”

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Raw units | 621,733 all BWM; 75,708 well-isolated | `label>=1` means all 3 RIGOR tests pass | 199,417 raw units in selected representative sessions | 21,396 good grey-matter units | Plausible selected-session fraction (10.73% vs paper-wide 12.18%); scope differs intentionally by one-session/subject reference code. |
| Mean neurons/session | Not given for one-session/subject cohort | Good clusters merged across listed probes | — | 159.67 (median 135, range 7–516) | Plausible. |
| Subjects | 139 release subjects | One selected session per subject | 139 candidates | 134 retained | Five subjects excluded solely for unavailable mandatory whisker output. |
| Sessions | 459 release; methods benchmarks subsets | Seed 42, first listed EID per selected subject | 139 candidates | 134 retained | Exact reference selection followed; five mandatory-stream exclusions accounted. |
| Trials (total) | 294,297 raw across all 459 | Event/max-duration mask | 85,688 raw in 134 retained sessions | 55,475 valid | Reduction fully explained by paper/code QC and complete-window requirement. |
| Trials/session | Raw release mean 641 | Variable | Selected raw 639.46 mean | Retained mean 413.99, median 395, range 114–923 | Plausible after reaction-time/event/window QC. |
| Time range/bin | Reference `(-0.5,1.5)`, 20 ms | Exact caching parameters | N/A | 100 bins; behavior right edges −0.48…+1.50 | Match. |
| Choice distribution | Binary | Native −1/+1 | — | left 26,955 (48.59%), right 28,520 (51.41%) | Plausible/near balanced. |
| Prior distribution | 0.2/0.5/0.8 | `block=probabilityLeft` | — | 22,613 / 7,847 / 25,015 trials | Correct values; unbiased 0.5 blocks less frequent as expected. |
| Wheel bins | Required 3 bins | Continuous reference stream | — | 1,849,167 / 1,849,166 / 1,849,167 timepoints | Exact pooled tertiles. |
| Whisker bins | Required 3 bins | Continuous reference stream | — | 1,849,167 / 1,849,166 / 1,849,167 timepoints | Exact pooled tertiles. |
| Camera side | Left preferred, right fallback | Explicit fallback | — | 131 left, 3 right | Match. |

### Spot Checks
- Sessions 0, 67, and 133 have finite arrays, correct `(neurons,100)` shape, and valid output classes.
- All five skipped session EIDs and reasons are stored in metadata; each lacks both complete left and right motion-energy streams.
- `x` region was investigated: it is Allen “Nucleus x” (atlas ID 765, level 6), not an unknown/fiber label, so its 11 neurons are correctly retained.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` explicitly reports “Data format is valid, no errors or warnings” and “Data verification complete.” No warning needed a waiver.
2. **Independent raw neural `np.allclose` check**: `/app/cache/raw_sanity_checks.py` directly loaded raw spike time/cluster arrays (without importing conversion code), independently histogrammed session 0, converted trial 5, neuron 3 (raw cluster 58), and matched the converted 100-bin vector exactly.
3. **Independent raw input `np.allclose` check**: Directly loaded the trial parquet and independently constructed the time grid and original trial-number rows for converted trials 0, 5, and 261; all matched exactly.
4. **Independent raw output `np.allclose` check**: Directly mapped raw choice/prior and independently rebuilt IBL wheel interpolation/filtered speed and camera motion-energy interpolation for three trials. Both categorical time series matched exactly.
5. **Dimensions/finite values**: All neural/input/output trials have T=100; every inspected and globally asserted value is finite; region-vector lengths match neuron axes.
6. **Identity/order checks**: Trial indices are strictly increasing after QC; all neuron UUIDs are unique within each merged session; all time grids are identical; expected output classes are exactly `{0,1}`, `{0,1,2}`, `{0,1,2}`, `{0,1,2}`.
7. **Region checks**: 202 Beryl labels are present. 119 satisfy the paper’s reportability condition (≥5 neurons in each of ≥2 sessions), containing 19,763 neurons in reportable session-region populations. Reference all-region caching (`single_region=False`) retains all good Beryl neurons, so the converted all-region dataset correctly retains 21,396 and records labels for downstream filtering. Allen acronym `x` was verified to be real “Nucleus x” (ID 765), not unknown anatomy.
8. **Key statistics**: 134 sessions/subjects, 55,475 trials, 21,396 good neurons, 131 left-camera and 3 right-camera sessions, five fully accounted missing-whisker exclusions. Good-unit fraction in selected sessions is 10.73%, close to the paper-wide 12.18% despite different session scope.
9. **Class checks**: Choice is 48.59/51.41%; prior trial counts are 22,613/7,847/25,015; each pooled wheel/whisker class differs by at most one timepoint from one-third.
10. **Beginning/end edge cases**: Spike bins are left-closed/right-open over `[stim−0.5, stim+1.5)`; behavior is evaluated at exact right edges as in reference interpolation. Complete endpoint coverage is required, preventing extrapolation. First/last retained trial and first/middle/last session checks pass.

### Reference Code Comparison
| Processing stage | Reference implementation | Converter implementation | Result |
|------------------|--------------------------|--------------------------|--------|
| Data loading | ONE ALF loaders; frozen `bwm_release.csv`; merge listed probes | Direct staged ALF reads using the same frozen CSV and listed probes | Equivalent offline implementation; newest coherent revisions selected. |
| Neuron/trial filtering | `label>=1`; Beryl mapping; event/order/max-10-s mask | Same, plus paper 0.08–2.00-s movement latency and mandatory-stream coverage | Match plus justified paper/task constraints. |
| Temporal alignment | `align_time=stimOn_times`, `time_window=(-.5,1.5)` | Exact same | Match. |
| Binning | 20 ms spike histograms; common cluster axis | 20 ms histograms with memory-mapped windows and stable UUID axis | Match; independent histogram check passed. |
| Input construction | Reference code stores alignment metadata but task specifies inputs | Right-edge time since stimulus and original trial number repeated over time | Required task-specific construction. |
| Output construction | Choice/block scalars; continuous wheel/whisker interpolation | Same raw variables/alignment, then exact requested categorical mappings/tertiles | Required task-specific difference; raw checks passed. |

### Issues Found and Resolved
- **Duplicate wheel timestamp**: one exact duplicate initially caused a sample-session skip. Fixed by deterministic finite filtering/deduplication before interpolation; reran sample/full conversion and every check. No sessions are now lost for this benign edge case.
- **Network unavailable**: avoided downloads by reading the provided read-only staged ONE cache directly. This does not alter data content.
- **Initial scope/window assumption corrected before implementation**: inspecting the call site revealed exact seed-42 one-session-per-subject selection and −0.5/+1.5-s window. Notes and implementation were corrected before final conversions.
- **No unresolved conversion errors remain.**

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, smoothly from 13.1056 (epoch 1) to 0.6929 (epoch 200).
- Test loss: 0.7055.
- Device: CUDA; all 200 epochs and requested sample plotting completed.
- `/app/train_decoder_full_out.txt` created; script finished successfully.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| Choice | 0.6233 | 0.6072 | 0.5000 | Above chance; small 0.0161 absolute gap. |
| Prior probability left | 0.7053 | 0.6954 | 0.3333 | 2.09× chance. |
| Wheel speed bin | 0.6172 | 0.6144 | 0.3333 | 1.84× chance. |
| Whisker motion-energy bin | 0.7057 | 0.7038 | 0.3333 | 2.11× chance. |

All outputs exceed chance and all train/validation ratios are near 1, with no concerning overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Balanced Accuracy | Chance | Ratio to chance | Expectation from papers |
|----------|------------------------------|--------|-----------------|-------------------------|
| Choice | 0.6072 | 0.5000 | 1.21× | Methods paper reports above-chance choice decoding and ~2% relative multi-session gain across 10 sessions, but does not provide a directly comparable whole-cohort value for this validator/temporal target. |
| Prior | 0.6954 | 0.3333 | 2.09× | Methods-paper example continuous-prior correlations are about 0.74–0.76 (ridge examples 0.37–0.54); categorical balanced accuracy here is not the same metric but is strongly above chance. |
| Wheel speed bin | 0.6144 | 0.3333 | 1.84× | Methods-paper example continuous correlations are about 0.80–0.86 (ridge 0.52–0.66); data paper finds significant wheel-speed decodability in 81% of reportable areas. Strong categorical performance is consistent. |
| Whisker motion-energy bin | 0.7038 | 0.3333 | 2.11× | Methods paper reports decodability, weaker capture above 5 Hz, and benefit from more sessions. Strong low/medium/high decoding is consistent because bins emphasize coarse amplitude. |

The papers' prior/wheel/whisker values are Pearson correlation or R² on continuous targets, while the task and validator use balanced accuracy on categorical targets. They are therefore qualitative expectations rather than numerically interchangeable benchmarks.

### Accuracy-vs-Chance Investigation
- Every output is above chance. Prior, wheel, and whisker exceed 1.5× chance.
- Choice is 1.21× chance and was investigated under the required low-accuracy protocol:
  1. Direct raw checks on three named converted trials confirmed native −1→left/0 and +1→right/1 exactly.
  2. Choice has substantial variation and near balance: 26,955 left (48.59%) and 28,520 right (51.41%), not a dominant-class artifact.
  3. Processing/sample plots show spike bins and all targets on the same stimulus-onset grid, with no temporal shift. The −0.5 to +1.5-s window is exactly the reference caching window.
  4. Unit filtering is exactly `label>=1`, independently confirmed to mean all three RIGOR criteria pass.
  5. Input/output and raw spike histogram `np.allclose` checks all pass.
- Choice is expected to be harder here because all 100 stimulus-aligned timepoints—including 25 pre-stimulus bins and early pre-movement bins—carry the repeated choice label and are scored. The data paper's choice-specific decoder instead aligns a short post-movement window. Changing to movement alignment or scoring only late bins would violate this task's explicit stimulus-alignment requirement. Thus no conversion change is justified.

### Train-vs-Validation Gap
| Variable | Train | Validation | Train/validation ratio | Assessment |
|----------|-------|------------|------------------------|------------|
| Choice | 0.6233 | 0.6072 | 1.027 | No overfitting concern. |
| Prior | 0.7053 | 0.6954 | 1.014 | No overfitting concern. |
| Wheel | 0.6172 | 0.6144 | 1.005 | No overfitting concern. |
| Whisker | 0.7057 | 0.7038 | 1.003 | No overfitting concern. |

All ratios are far below the 1.5× concern threshold. There is no evidence of leakage: validation remains lower than training for every output and gaps are modest.

### Plot and Training Review
- Full training completed all 200 epochs and ended normally.
- Loss decreased smoothly from 13.1056 to 0.6929; test loss was 0.7055.
- Generated sample/prediction plots are nonblank and show aligned neural/input/output traces. No plotting or category artifact was observed.

### Issues Found and Resolved
- No new conversion issue was found in Critical Review 2.
- The low-choice diagnostic was fully investigated; retaining exact reference stimulus alignment/window is better justified than altering data to inflate accuracy.
- All checks were re-read after full training; no reconversion is necessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `/app/README.md` created with dataset description, loading instructions, schema, statistics, reproduction commands, and decoder results.
- [x] `/app/cache/` contains investigation scripts rather than cluttering the project root.
- [x] `/app/cache/README_CACHE.md` documents cached/investigation files.
- [x] All required conversion, validation, training, notes, and log files exist.
- [x] Final documentation includes decisions, corrections, raw sanity checks, paper/code comparisons, edge cases, and accuracy reviews.


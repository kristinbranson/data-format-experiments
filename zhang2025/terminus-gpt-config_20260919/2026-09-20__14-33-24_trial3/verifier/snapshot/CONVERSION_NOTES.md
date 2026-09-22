# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide map of neural activity during complex behaviour
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.3.5 imported successfully
- PyTorch 2.6.0+cu124 imported successfully
- Checkpoint confirmed `/app/CONVERSION_NOTES.md` exists.

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
- `stage_cache.sh`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Resolve all probe insertions for an EID, load and merge spikes/clusters, load trials and continuous behavior, and assemble cluster metadata. |
| `load_spiking_data` | same | LOADING/CURATION | Load spike sorting with `SpikeSortingLoader`; merge cluster metrics. Optional `label >= qc` filtering exists, but the reference caller passes `qc=None`, retaining all clusters. |
| `merge_probes` | same | PROCESSING | Reindex clusters across probes, concatenate them, and stable-sort merged spikes by time. |
| `load_trials_and_mask` | same | CURATION | Build a trial-validity mask from reaction time, duration, missing fields, unbiased-block, and no-choice criteria. |
| `list_brain_regions` / `select_brain_regions` | same | CURATION | Map native acronyms to the Beryl atlas and select cluster IDs; reference uses all available Beryl regions. |
| `get_spike_data_per_interval` / `bin_spiking_data` | same | PROCESSING | Count spikes in fixed-width bins in event-aligned trial intervals and return `(trial, neuron, time)` arrays. |
| `load_target_behavior` | same | LOADING | Load wheel and camera motion-energy streams with `SessionLoader`. |
| `get_behavior_per_interval` / `bin_behaviors` | same | PROCESSING | Slice behavior by trial interval, verify temporal coverage, and linearly interpolate to the neural bin grid. |
| `align_spike_behavior` | same | CURATION | Remove invalid trials and reshape aligned behavioral arrays. |
| `create_dataset` | `code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Store spike-count matrices sparsely plus behavior and cluster/session metadata. |
| `SingleSessionDataset` | `code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Reconstruct sparse spikes, standardize neural data, encode choice, standardize regression targets, and mean-fill residual target NaNs for model fitting. |

### Notes
- Primary workflow: `prepare_data` → Beryl region mapping/selection → `bin_spiking_data` → `bin_behaviors` → `align_spike_behavior` → cached dataset creation.
- This is electrophysiology, not imaging; delta-F/F is not applicable.
- Reference trial parameters are `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` seconds, and `binsize=0.02` seconds, producing 100 bins per trial.
- Spike bins use half-open trial intervals (`times >= start` and `times < end`) and integer spike counts. Returned matrices are neuron by time after transposition.
- `prepare_data` merges every probe in a session and calls `load_spiking_data` without a QC threshold. Thus all sorted clusters are retained; `clusters.label >= 1` is saved only as `good_clusters` metadata. This differs from pipelines that load only good units and must be checked against the papers/data release.
- Actual trial mask defaults plus `max_trial_len=10`: reaction time (`firstMovement_times-stimOn_times`) must be 0.08–2.0 s; `feedback_times-goCue_times` must be ≤10 s; required fields (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`) must be non-NaN; choice 0 is excluded. Probability-left 0.5 trials are retained (`exclude_unbiased=False`).
- Wheel speed is `abs(wheel.velocity)`. Whisker motion energy uses left-camera `whiskerMotionEnergy`, falling back to right camera only if left loading is skipped.
- Behavior samples are sliced with `searchsorted(..., side='left')`. Interpolation points are `linspace(interval_start + binsize, interval_end, n_bins)`, i.e. bin-end times from -0.48 through +1.50 s relative to stimulus onset. Linear interpolation permits extrapolation after checking first/last samples are within one bin of interval boundaries.
- The methods-paper decoder treats wheel speed and whisker motion energy as continuous regression outputs and choice as categorical. Required 3-class discretization in this task is therefore an intentional downstream-format divergence to be planned later.
- Reference caching randomly partitions trials 70%/10%/20% after alignment. The requested target format stores complete sessions, so conversion should not pre-partition trials.
- Potential reference-code issue noted for later consistency review: `align_spike_behavior` combines Python lists with `and`, so its final mask does not robustly combine every behavior validity mask. Conversion should explicitly combine validity across required streams and document the justified robustness difference.
- Spike/behavior standardization and one-hot encoding occur in the reference model data loader, not during raw caching; converted data should preserve counts/classes and let the supplied decoder perform its own model preprocessing.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/one_cache` is a staged IBL ONE cache. Writable release tables and REST metadata are local; 12 laboratory directories are symlinks into a read-only native dataset mount. Native files are therefore accessed through `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/...` while remaining immutable.
- **Issue found and corrected**: the initial `find`/`Path.rglob` inventory did not follow directory symlinks and incorrectly suggested only metadata was present. Inspection of `stage_cache.sh` revealed the intended symlink architecture. Re-running exploration with symlink-following paths exposed all native arrays. Step 2 was reopened and all statistics below were recomputed directly from native files.
- Release tables are in `2022_Q4_IBL_et_al_BWM/`, `2025_Q3_IBL_et_al_BWM/`, and `Brainwidemap/`. The Zhang repository's `bwm_release.csv` provides the authoritative paper cohort and maps 699 PIDs to 459 EIDs, 139 subjects, labs, dates, session numbers, and probe names.
- Native organization is `<lab>/Subjects/<subject>/<date>/<number>/alf/`. Trial/wheel/camera files are session-level; spikes and cluster metadata are under `alf/<probe>/pykilosort/`. Revised ALF datasets occur in `#YYYY-MM-DD#` subdirectories.
- Revision selection for exploration chose, within each paper-cohort session/probe, the candidate full table with the greatest row count and then latest lexical revision as a tie-break. This selected all 459 trial tables and all 699 cluster tables with no missing files. Exact totals match the paper, validating this rule for these objects.
- Relevant formats: Parquet trial/cluster tables, NumPy arrays for spikes/wheel/camera streams, and CSV cluster UUIDs.
- Representative EID `02fbb6da-3034-47d6-a61b-7d06c796a830` has 646 trials, 781 clusters, and 24,771,586 spikes on probe00. Native spike times are float64 and spike cluster IDs uint32. Its full trial table has 13 columns: `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `probabilityLeft`, `feedback_times`, `feedbackType`, `rewardVolume`, `firstMovement_times`, `intervals_0`, and `intervals_1`.
- Representative left motion energy/timestamps both have 174,360 float64 samples (~60 Hz); right both have 436,588 samples (~150 Hz). The reference preference for left therefore gives the stated 60 Hz stream. Native wheel timestamps and positions are float64; SessionLoader derives velocity/acceleration.
- Across the 459 paper sessions, left camera time+motion-energy pairs exist for 437 sessions, right pairs for 420, and at least one side for 445. Fourteen sessions lack either required whisker stream and cannot support the specified output.

### Dataset Size (from native data files; paper cohort)
| Statistic | Value |
|-----------|-------|
| Neurons/clusters (total) | 621,733 across 699 insertions |
| Well-isolated (`label == 1`) | 75,708 |
| Clusters / insertion | min 135, median 864, max 2,311; paper mean 889 |
| Subjects | 139 |
| Sessions / subject | computed release cohort; 459 sessions total |
| Sessions | 459 |
| Insertions | 699 |
| Trials (raw total) | 296,090 |
| Trials / session | min 401, median 601, max 1,525 |
| Trials after Zhang discrete-event mask | 195,781 before continuous-stream coverage filtering |
| Valid-trial choice distribution | IBL -1: 96,189; +1: 99,592 |
| Valid-trial probabilityLeft distribution | 0.2: 81,747; 0.5: 27,566; 0.8: 86,468 |

### Available Variables and Native Types
| Source object | Meaning | Native format/type |
|---------------|---------|--------------------|
| `_ibl_trials.table.pqt` | choice, probabilityLeft, event times, contrasts, feedback, reward | Parquet DataFrame; one row/trial |
| `spikes.times.npy` / `spikes.clusters.npy` | spike timestamps and integer cluster assignments per probe | float64 / unsigned integer NumPy arrays |
| `clusters.metrics.pqt` and cluster/channel arrays | sorting QC label, depths/channels, UUID/anatomical metadata | Parquet/NumPy |
| `_ibl_wheel.timestamps.npy` / `_ibl_wheel.position.npy` | native wheel trajectory; velocity derived by reference SessionLoader | float64 NumPy arrays |
| `left/rightCamera.times.npy` and `Camera.ROIMotionEnergy.npy` | sample times and whisker-region motion-energy traces | matched float64 NumPy arrays |

### Native Sanity Results
- Summing selected cluster rows gives exactly 621,733, matching the paper. QC-label counts are 112,878 at 0; 255,535 at 1/3; 177,612 at 2/3; and 75,708 at 1. The label-1 count exactly matches the paper's well-isolated total.
- All 459 cohort sessions have full trial tables; all 699 listed probes have cluster metrics.
- Raw trial minimum is 401, consistent with the paper Results subset statement that retained sessions have at least 400 trials.
- Trial filtering reproduces the expected balanced choice distribution and the three protocol prior values only.
- Detailed selected-file maps and statistics were saved to `/app/cache/selected_trials.json`, `/app/cache/selected_metrics.json`, and `/app/cache/native_stats.json` for later independent checks.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote / Basis |
|-----------|-------|----------------------|
| Neurons (all sorted units) | 621,733 | Data paper: spike sorting produced 621,733 units including multi-unit activity. |
| Well-isolated neurons | 75,708 | Data paper: 75,708 passed stringent RIGOR single-unit QC. |
| Neurons / insertion | 889 all units; 108 well-isolated (means) | Data paper recording summary. |
| Subjects | 139 (94 male, 45 female) | Data paper and `bwm_release.csv`. |
| Sessions | 459 public release; 433 used by methods paper | Data paper release criteria; Zhang methods extract states 433 IBL sessions. |
| Insertions | 699 | Data paper and `bwm_release.csv`. |
| Laboratories | 12 | Data paper recording summary. |
| Trials (total) | Not stated in supplied extract | Per-session inclusion thresholds are stated, but no total trial count. |
| Trials / session | Release minimum 250; Results analyses say at least 400 | Data paper Methods vs Results wording; resolved as release vs analysis subset in Step 4. |
| Neural data time bin | 20 ms for common 2 s Zhang representation; 50 ms for Zhang per-trial choice/prior analysis | Methods-paper overview and variable-specific processing. Provided caching code uses 20 ms for all requested streams. |
| Behavior data time bin | 20 ms after interpolation for reference cache | Zhang caching code; paper dynamic behavior also uses 20 ms bins. |
| Reward rate | Not stated | No fraction supplied in available methods extract. |
| Initial unbiased trials | 90 | Task description. |
| Biased block probabilities | probabilityLeft 0.2 or 0.8; blocks 20–100 trials, empirical mean 51 | Data-paper task description. |
| Method-paper variables | choice, prior, wheel speed, whisker motion energy | Zhang methods extract. |
| Method-paper sessions | 433, covering 270 brain regions | Zhang methods extract. |

### Processing Details
- Method-paper common representation: spike-sorted, temporally binned counts with each trial represented as `N × T`; 2 s at 20 ms gives `T=100`.
- Choice is aligned to stimulus onset using neural activity from -0.5 to +1.5 s. Prior is stimulus-aligned but its paper-specific window is -0.6 to -0.1 s. Dynamic wheel/whisker analyses in the prose align to first movement and use 0 to +1 s. The supplied caching code instead uses the task-required common stimulus alignment of -0.5 to +1.5 s and 20 ms for all streams; this is the applicable reference for this conversion.
- Data-paper wheel analysis averages wheel values and spike counts in non-overlapping 20 ms bins. Its causal decoder uses the current plus ten preceding bins and reports R².
- Data-paper binary stimulus/choice/feedback decoding uses class-weighted L1 logistic regression and balanced accuracy. Neurons from probes in the same session and region are combined because simultaneous probes are not independent.
- Whisker motion energy is the mean absolute pixel difference between adjacent frames in a whisker-pad bounding box anchored between DLC nose-tip and eye estimates. Left camera is 60 Hz, right is 150 Hz, and body is 30 Hz. The Zhang overview describes the used whisker stream as sampled at 60 Hz, consistent with preferring left camera.
- Trial task: 90 initial 0.5-prior trials followed by alternating 0.2/0.8 probability-left blocks; choice is made by turning the wheel. Block changes are uncued.

### Curation Steps

**Neuron curation rules**:
- Data-paper analyses define well-isolated neurons using RIGOR metrics: amplitude >50 μV, noise cutoff <20 μV, and refractory-period-violation criterion.
- Final paper analyses additionally restrict to Allen CCF gray matter, at least five well-isolated neurons per region/session, and regions recorded in at least two qualifying sessions.
- In contrast, the Zhang caching code deliberately bins all Kilosort clusters (`qc=None`) and records good-unit labels only as metadata. Its prose also says all neurons sorted by Kilosort 2.5. This conversion is for reproducing the Zhang neural-decoding representation, so the all-cluster behavior is a candidate mapping choice to resolve explicitly in Steps 4–5.

**Trial curation rules**:
- Data-paper analyses exclude trials missing choice, probabilityLeft, feedbackType, feedback time, stimOn time, or firstMovement time.
- Reaction time must be 0.08–2.00 s from stimulus onset to first wheel movement.
- Zhang code additionally excludes no-choice trials and feedback-goCue durations >10 s, retains initial probabilityLeft=0.5 trials, and requires the same core fields.
- Release sessions require ≥250 trials, ≥90% correct at 100% contrast in each biased block side, ≥3 incorrect trials after exclusion, and task hardware QC. The Results section's “at least 400 trials” refers to the stricter subset retained for further paper analyses rather than public-release eligibility.
- Insertions require acceptable whole-recording RIGOR, no major artifacts, recovered histology, and resolved alignment.

### Decoders Trained
| Decoded variable | Accuracy / metric reported in available extract |
|------------------|-----------------------------------------------|
| Choice | Balanced accuracy metric described; no numerical value in `methods.txt` |
| Prior | Included by method paper; no numerical value in available extract |
| Wheel speed | R² in source papers (continuous regression); no numerical value in available extract |
| Whisker motion energy | Continuous regression/correlation in method paper; no numerical value in available extract |

### Source-access note
- Installed environment has no `pdftotext`, Ghostscript, or Python PDF parser. The PDFs are compressed such that `strings` exposes metadata but not article body text. The supplied `/app/methods.txt`, reference code, and release CSV therefore provide the usable reference details. No numerical decoder-accuracy table was available in those sources; this limitation is recorded rather than inventing values.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release size | Zhang `bwm_release.csv`: 459 EIDs, 699 PIDs | Exact native totals: 459 sessions, 699 probes, 621,733 clusters | Data paper: 459/699/621,733 | Use `bwm_release.csv` as authoritative cohort; exact agreement validates loading/revision selection. |
| Snapshot tables | Code CSV uses current cohort | 2022 table has 354 sessions; 2025 has exactly all 459; combined has 480 | Current paper release has 459 | Do not use older 354 snapshot or 21 combined extras. |
| Method-paper session count | Caching loop can process arbitrary selected EIDs | 445 sessions have either camera; 444 have ≥2 jointly covered valid trials | Method paper states 433 sessions | Basic stream presence does not explain 433; likely additional paper video/session QC. For required outputs, use explicit stream validity and retain sessions with ≥2 trials unless an authoritative 433-EID list is found. Document 444 vs 433. |
| Unit filtering | Zhang `prepare_data` passes `qc=None`, bins all clusters, records label metadata | 621,733 total; QC label 1 count exactly 75,708 | Zhang prose says all Kilosort 2.5 neurons; data-paper analyses use 75,708 well-isolated neurons | Match the methods-paper code for decoder data: retain all clusters. Preserve QC metadata/statistics; do not silently substitute data-paper analysis filtering. |
| Brain-region filtering | Zhang maps every cluster to Beryl and uses all regions | Anatomical arrays available per probe | Data-paper final analyses require gray matter, ≥5 good units/session, ≥2 sessions | Those restrictions apply region-level data-paper analyses, not Zhang whole-session decoder caching. Use all mapped Beryl regions. |
| Trial threshold | Loader filters trial rows but not sessions by count | Cohort raw minimum is 401 trials | Release Methods says ≥250; Results says analyses retained ≥400 | The provided 459 cohort is the stricter analysis subset (observed minimum 401); no extra threshold needed. |
| Trial filtering | RT 0.08–2 s, required non-NaNs, no-choice removal, ≤10 s feedback-goCue; retain pLeft=.5 | 296,090 raw → 195,781 discrete-valid | Paper specifies required fields and RT bounds | Use code's superset, including no-choice and 10 s duration checks. |
| Temporal alignment | Supplied cache script uses stimOn, -0.5 to +1.5 s, 20 ms for all variables | Streams cover 188,964 valid trials in 444 sessions | Zhang prose uses choice stimOn but dynamic targets first-movement; task explicitly requires stimulus alignment | Task and executable reference cache take precedence: all variables use stimulus onset and common 100-bin grid. |
| Behavior timestamps | Reference interpolates at bin ends (-0.48…+1.50 s) | Native camera rates differ (left ~60 Hz, right ~150 Hz) | Whisker metric is camera-frame based | Reproduce reference bin-end linear interpolation, preferring left and falling back right. |
| Missing behavior | Reference calls `allow_nans=True`; `align_spike_behavior` combines masks incorrectly via Python `and` | 14 no-camera sessions plus one uncovered session; finite coverage leaves 444 sessions | Required output must be defined and sessions need ≥2 trials | Explicitly intersect all trial/stream validity masks and exclude undefined trials/sessions; this fixes a clear implementation bug without changing intended processing. |
| ALF revisions | Generic SessionLoader can select partial legacy objects in this mixed cache | Full revised tables reproduce exact paper totals | Papers assume curated current objects | Resolve full trial/cluster objects explicitly using cohort paths and validated revision selection; avoid partial legacy objects. |
| Choice coding | Native IBL values are -1/0/+1 | Valid counts: -1=96,189; +1=99,592 | Task requires left=0, right=1 | Map -1→0 and +1→1; exclude 0. |
| Prior coding | Native `probabilityLeft` is 0.2/0.5/0.8 | Counts are only those three values | Task prescribes 0/1/2 mapping | Map exactly 0.2→0, 0.5→1, 0.8→2. |

### Final Consistent Understanding
- The source cohort is the exact 459-session/139-subject/699-insertion current data-paper release in `bwm_release.csv`; native totals reproduce every headline statistic.
- The applicable methods-paper representation is whole-session, merged-probe spike counts with all Kilosort clusters, Beryl anatomical mapping, 20 ms bins, and a common 2 s trial window.
- The task-required stimulus alignment overrides variable-specific first-movement windows in prose and agrees with the supplied caching script's executable parameters.
- Continuous behavior is linearly interpolated to neural bin-end times. Required categorical outputs will be discretized only after interpolation using globally defined training-independent thresholds planned in Step 5.
- Session inclusion for this task must require all four outputs and at least two jointly valid trials. Native coverage gives at most 444 sessions/136 subjects before any later interpolation or spike/anatomy failures. The paper's 433-session number is recorded as an unresolved additional-QC difference, not forced by arbitrary session removal.
- Corrected native exploration and coverage scripts are in `/app/cache/`; all subsequent conversion checks will compare directly against their selected raw files.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Per-probe `spikes.times`, `spikes.clusters` | `neural[session][trial]` | Merge listed probes, reindex clusters, count spikes in 100 half-open 20 ms bins from stimOn-0.5 to stimOn+1.5; shape `(neurons,100)` | `load_spiking_data`, `merge_probes`, `get_spike_data_per_interval`, `bin_spiking_data` | Store integer counts compactly; validate no count exceeds uint8 before casting. |
| Relative bin-end times | `input[...][0,:]` | Fixed float32 vector `[-0.48,-0.46,...,1.50]` s | `get_behavior_per_interval` interpolation grid | Continuous, time-varying input. Metadata alignment remains stimulus onset at t=0. |
| `trials.probabilityLeft` transitions | `input[...][1,:]` | Number trials consecutively elapsed in current block, first trial=1; broadcast over 100 bins | Task-specific derived variable | Count on the original full session before trial filtering so exclusions do not renumber the experimental block. |
| `trials.choice` | `output[...][0,:]` | Native -1→0 (left), +1→1 (right); broadcast over 100 bins | `bin_behaviors` plus task coding | Native choice 0 is excluded. |
| `trials.probabilityLeft` | `output[...][1,:]` | 0.2→0, 0.5→1, 0.8→2; broadcast over 100 bins | `bin_behaviors` plus task coding | Exact protocol values only; unexpected values fail validation. |
| Absolute filtered wheel velocity | `output[...][2,:]` | Native position → 1 kHz linear interpolation → 20 Hz low-pass filtered velocity → absolute value → linear interpolation at trial bin ends → global tertile classes 0/1/2 | `SessionLoader.load_wheel`, `wheel.interpolate_position`, `wheel.velocity_filtered`, `load_target_behavior`, `get_behavior_per_interval` | Uses reference wheel-speed definition and temporal grid. |
| Left/right camera ROI motion energy | `output[...][3,:]` | Prefer left, fallback right; linear interpolation at trial bin ends; global tertile classes 0/1/2 | `load_target_behavior`, `get_behavior_per_interval`, `bin_behaviors` | Require matched timestamp/value arrays and finite interpolated values. |
| Cluster channel → `channels.brainLocationIds_ccf_2017` | `brain_region_idx[session]` | Atlas ID→native acronym→Beryl acronym, then global categorical index | `list_brain_regions`, `select_brain_regions` | Preserve `root` if assigned; one region per retained cluster. |
| `bwm_release.csv.subject` | `subjects`, `subject_idx` | Stable first-appearance unique subject list and per-session index | cohort CSV | Only subjects with retained sessions appear. |

### Key Decisions
1. **Cohort**: Start from all 459 current-release sessions in `bwm_release.csv`, not the older 354-session snapshot or 21 combined extras. Retain only sessions with all required streams and at least two fully valid trials.
2. **Neuron curation**: Retain all Kilosort clusters to match the Zhang methods-paper executable pipeline (`qc=None`) and prose (“all neurons”). This yields the paper's 621,733 source clusters before behavior-based session exclusion. Data-paper well-isolated QC statistics are retained as checks but not imposed because that would change the methods-paper decoder dataset.
3. **Probe handling**: Merge every listed probe from the same EID, because simultaneous probes share behavior and the reference explicitly combines them. Cluster order is CSV probe order, then native cluster row order; spike cluster IDs are remapped accordingly.
4. **Trial curation**: Apply reaction time 0.08–2.0 s inclusive, required non-NaN events, choice !=0, and feedback-goCue duration ≤10 s. Retain probabilityLeft=0.5 trials. Also require complete finite wheel and selected camera coverage over the whole [-0.5,+1.5] window.
5. **Temporal grid**: Match executable code: 2 s around `stimOn_times`, 20 ms bins, 100 spike-count bins. Behavior and time input use bin ends -0.48 through +1.50 s. Spike bins are `[start,start+0.02)`, etc.; this deliberate 20 ms offset matches reference behavior interpolation.
6. **Mixed outputs**: Store every output as `(4,100)`. Per-trial choice and prior are broadcast over time, because a single trial output array cannot mix scalar and time-varying row shapes and the supplied decoder accepts categorical values at every timepoint.
7. **Discretization**: Use global 1/3 and 2/3 empirical quantiles across all retained finite wheel-speed and whisker values. Apply `np.searchsorted([q1,q2], value, side='right')`, yielding classes 0/1/2. Global thresholds preserve a common physical class meaning across sessions. If tied quantiles collapse (e.g. wheel q1=0), detect this and use equal-frequency rank assignment with deterministic stable tie handling; record the final thresholds/method in metadata and logs.
8. **Trial number in block**: Compute before filtering, reset whenever `probabilityLeft` changes, and number from 1. This preserves the actual experimental trial position despite excluded trials.
9. **ALF revisions**: Resolve full current objects explicitly. For tables, choose candidates containing required columns, then greatest row count and latest revision tie-break. For NumPy objects, use matched revision pairs and prefer newest validated pair. Never accept the partial legacy one-column trial object selected by generic SessionLoader in this mixed cache.
10. **Anatomy**: Use `clusters.channels` to index `channels.brainLocationIds_ccf_2017`, then Beryl mapping. Assert cluster-array lengths equal the selected cluster metrics/spike cluster universe.
11. **Storage and performance**: Neural trial matrices use uint8 after checking maximum count; inputs float32; outputs uint8; region indices int32. Estimated all-cluster neural payload is ~26.5 GB for 444 preliminary sessions. Process one session at a time, memory-map spike arrays, vectorize histogramming, and avoid loading all spikes globally.
12. **No data split**: Save complete retained sessions. The supplied decoder performs its own train/validation split; reference caching's 70/10/20 split is not represented in the required format.

### Planned Sanity Checks
- [ ] Recompute raw cohort totals directly: 459 sessions, 139 subjects, 699 probes, 621,733 clusters, 75,708 label-1 units, 296,090 trials.
- [ ] For at least three raw trials, compare conversion spike counts to independent `np.histogram2d`/`bincount` calculations using `np.allclose`.
- [ ] Compare converted time input to `np.arange(-0.48,1.5001,0.02)` with `np.allclose`.
- [ ] Compare trial-in-block values for trials around index 89/90 and later probability transitions to an independent raw-table calculation with `np.allclose`.
- [ ] Compare choice/prior conversion for three named raw trials with exact expected mappings and `np.allclose`.
- [ ] Independently derive wheel velocity from raw wheel position with Brainbox functions and compare interpolated converted values/classes.
- [ ] Independently interpolate raw camera motion energy for three trials and compare pre-discretization values/classes.
- [ ] Verify every trial matrix has 100 timepoints, every session has ≥2 trials, all arrays are finite, and category values are contiguous and within declared ranges.
- [ ] Verify merged neuron counts equal sums of selected cluster rows and every neuron has one valid Beryl index.
- [ ] Report retained/excluded sessions and trials by reason; investigate the expected preliminary ceiling of 444 stream-covered sessions and the method-paper value of 433.
- [ ] Verify output class distributions, especially that all three behavior bins are populated and not pathologically imbalanced.
- [ ] Estimate conversion time from two-session sample and optimize if projected full runtime exceeds 15 minutes.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required invocation and mutually exclusive `--full`/`--sample` modes plus `--show-processing`.
- Implemented explicit full ALF trial-table and matched camera revision resolution, authoritative cohort loading, reference trial mask, left-camera preference/right fallback, exact Brainbox 1 kHz wheel interpolation and filtered velocity, reference bin-end interpolation, global tertiles, merged-probe spike binning, Beryl mapping, metadata, structural assertions, timing logs, and plots.
- Uses a two-pass design: behavior is aligned and cached per session first; global discretization thresholds are then computed; spike arrays are memory-mapped and binned one session/probe at a time.
- Neural counts are stored as uint8 when maximum bin count ≤255 and uint16 otherwise. Inputs are float32, outputs uint8, and index arrays int32.
- Syntax and CLI help checks passed.
- Smoke test on EID `6713a4a7-faed-4df2-acab-ee4e63326f8d`: 565 raw trials, 407 jointly valid trials, finite wheel/whisker arrays of `(407,100)`. Two spike trials binned successfully for 898 neurons, max count 7, with 7,439 total spikes in the first tested matrix pair.
- Code-review fixes before sample conversion: replaced endpoint-clamping `np.interp` with reference-matching linear extrapolation via `interp1d`; corrected spike time/cluster mask propagation for unexpected cluster IDs.

Code inefficiencies identified:
- Full output is intrinsically large (~26.5 billion count elements before behavior-stage exclusions) because the applicable methods pipeline retains all clusters.
- Python trial loops remain for variable event windows but expensive spike arrays are memory-mapped and each trial uses binary search plus vectorized `bincount`.

Code speedups added:
- Two-pass behavior caches avoid recomputing wheel filtering/interpolation during spike processing.
- Vectorized per-trial spike binning with flattened `np.bincount`.
- Compact integer output dtypes and pickle protocol 5.
- Session/probe streaming avoids holding all raw spikes in RAM.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (summed across sessions) | 2,626 |
| Neurons / session | 898, 1,728 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | 407, 244 |
| Time since stimulus range | [-0.48, 1.50] s (verifier rounds to [-0.5,1.5]) |
| Trial number in block range | [1, 90] |
| Choice distribution (per trial) | class 0: 314 (48.2%); class 1: 337 (51.8%) |
| Prior distribution (per trial) | 0.2/class0: 311; 0.5/class1: 105; 0.8/class2: 235 |
| Wheel output distribution (timepoints) | exactly 21,700 each class |
| Whisker output distribution (timepoints) | exactly 21,700 each class |
| Neural count range | 0–17 spikes/20 ms |
| Brain regions | 19 Beryl categories |
| Sample pickle size | 76 MB |

### Processing Plots Review
- Two plots were created at 1680×1400 RGBA: one per sample session.
- Each plot overlays a stimulus-onset marker, stimulus-aligned spike raster/count image, raw aligned wheel speed and whisker motion energy with global thresholds, and resulting categorical traces. Shapes and time axes are consistent; no temporal discontinuities or missing ranges were observed.
- Initial mandated run found a plotting-only `NameError` because `out` was not passed into `make_plot`. The function signature/call were fixed, syntax rechecked, and the entire required sample conversion rerun successfully. Final log contains only the successful rerun.

### Format Validation
- `/app/sample_data.pkl`, `/app/conversion_sample_out.txt`, and `/app/verification_sample_out.txt` exist.
- Supplied verifier completed with `Data verification complete.` No structural errors, NaNs, invalid ranges, or missing classes were reported.
- Warning addressed: verifier reports each neural trial is uint8 rather than preferred float32 and explicitly says it will be converted during training. This is intentional lossless compact storage of integer spike counts. Full float32 storage would expand the estimated neural payload from ~26.5 GB to ~105.9 GB without adding information; decoder `SessionData` already converts each session to float32. The warning cannot reasonably be eliminated without severe unnecessary storage/I/O cost.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Compact uint8 counts | ~4× lower pickle size and serialization I/O than float32 |
| Memory-mapped spikes + vectorized bincount | Avoids loading/copying whole recordings and per-spike Python loops |
| Behavior cache | Avoids repeated 1 kHz wheel filtering/interpolation |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Behavior alignment | ~0.27 s | ~2.1 min for 459 |
| Spike binning + plotting in sample | ~0.55 s | ~4.2 min for 459 |
| Overall sample conversion | 0.85 s/session | ~6.5 min plus large-pickle serialization; expected under 15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: uint8 neural-count storage is converted to float32 by the decoder; justified in Step 7.

### Training Progress
- Completed all 200 epochs without errors.
- Loss decreased from the initial high value to 0.601694; test loss was 0.642176.
- No divergence or late-epoch loss increase was observed.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform chance |
|--------|-----------------------|-------------------------|----------------|
| Choice | 0.7059 | 0.6164 | 0.5000 |
| Prior probability of left | 0.8320 | 0.7938 | 0.3333 |
| Wheel speed | 0.6779 | 0.6419 | 0.3333 |
| Whisker motion energy | 0.7014 | 0.6708 | 0.3333 |

All outputs exceed chance on held-out trials. Dynamic outputs reach about 1.93× and 2.01× chance, supporting correct temporal alignment. Prior reaches 2.38× chance. Choice is above chance (1.23×); the smaller two-session sample and held-out trial variability plausibly limit this output, to be reassessed on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 26.71 GB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; ends with `Data verification complete.`

### Full Conversion Results
- Runtime: 318.69 s (5.31 min), below the 15-minute threshold and faster than the ~6.5 min estimate.
- Candidate paper sessions: 459; retained: 444; excluded: 15 (14 missing matched camera streams, one without ≥2 jointly valid interpolable trials).
- Retained subjects: 136; retained trials: 188,925; summed session neuron count: 599,865; Beryl categories: 281.
- Choice counts: class 0=92,973; class 1=95,952.
- Prior counts: class 0=78,864; class 1=26,561; class 2=83,500.
- Wheel classes: 6,297,500 timepoints each.
- Whisker classes: 6,297,499 / 6,297,501 / 6,297,500 timepoints.
- Global wheel thresholds: [0.0151168453, 0.405079236].
- Global whisker thresholds: [2.73344064, 7.84546248].

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total source clusters | 621,733 | all clusters | 621,733 | 599,865 after stream-based session exclusion | Yes, difference fully accounted by excluded sessions |
| Well-isolated units | 75,708 | metadata only, not filtered | 75,708 | not imposed | Intentional methods-paper match |
| Subjects | 139 source | BWM cohort | 139 | 136 after stream exclusion | Yes, three subjects have no retained session |
| Sessions | 459 release; 433 method paper | stream-dependent | 459 | 444 | Basic availability match; 433 discrepancy retained for critical review |
| Trials (raw) | not stated | event filtered | 296,090 | 188,925 | Every exclusion has explicit trial/stream criterion |
| Trials/session | ≥400 raw | mask after loading | raw 401–1,525 | retained 125–1,445 after filtering | Yes |
| Choice distribution | binary | native -1/+1 | 96,189/99,592 before stream filtering | 92,973/95,952 | Yes |
| Prior values | 0.2/0.5/0.8 | same | only these values | 0/1/2 mapping | Yes |
| Time input | -0.5 to +1.5 window | bin ends -0.48…1.50 | N/A | -0.48…1.50 | Yes |
| Dynamic classes | task requires 3 bins | source is continuous | continuous | global tertiles, nearly exact thirds | Task-mandated transform |

### Validation and Spot Checks
- Supplied full verifier completed without structural errors, invalid values, or missing output classes.
- Expected uint8→float32 warnings occur once per trial (188,925) and are justified by compact lossless spike-count storage; decoder conversion is explicit.
- Full pickle loaded successfully in 12.35 s. First, middle, and last sessions and first/middle/last trials were checked: all neural/input/output matrices have 100 columns, finite inputs, legal categorical outputs, and region-vector lengths equal neuron counts.
- Metadata session records and global thresholds are present and consistent with conversion logs.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `verification_full_out.txt`; verifier reaches `Data verification complete.` with no structural errors, NaNs, invalid categories, or shape mismatches. The only warning type is neural uint8 rather than preferred float32, repeated per trial. This is intentional lossless spike-count storage; the supplied decoder converts per session to float32. Eliminating it would add ~79 GB without changing data.
2. **Independent raw-file sanity checks (`np.allclose`)**: `/app/cache/raw_sanity_checks.py` loads original ALF files directly rather than using converted arrays. For the first, middle, and last converted sessions it checked first/middle/last trials for time input, trial-in-block input, choice, and prior; trial 5/neuron 3 for direct raw spike histograms; and trial 5 for independently derived Brainbox wheel speed and direct camera motion-energy interpolation/classes. All 45 comparisons passed with maximum absolute difference 0. Output is in `raw_sanity_checks_out.txt`.
3. **Reference-code comparison**:
   - Loading: conversion resolves full ALF revisions explicitly; reference uses ONE/SessionLoader. This differs only to avoid the demonstrated mixed-cache partial-object ambiguity; exact paper totals validate selection.
   - Neuron/trial filtering: all Kilosort clusters (`qc=None`) matches Zhang. Trial required fields, RT 0.08–2 s, choice nonzero, ≤10 s duration, and retained unbiased trials match reference.
   - Alignment: both use `stimOn_times` plus [-0.5,+1.5].
   - Binning: both use half-open 20 ms spike bins and return neuron×100 matrices. Direct histogram comparisons pass.
   - Inputs: time uses reference behavior bin-end grid; trial number is task-required and computed on unfiltered raw block sequence.
   - Outputs: choice/prior source fields match reference. Wheel uses reference `interpolate_position` + `velocity_filtered` + absolute value. `SessionLoader.load_motion_energy` explicitly loads `Camera.ROIMotionEnergy` and renames left/right values `whiskerMotionEnergy`, confirming the selected source. Required tertile discretization is the intentional task divergence.
   - Robustness difference: conversion explicitly intersects every stream mask instead of reproducing the reference Python-list `and` bug.
4. **Key statistics comparison**: Source-native sums exactly reproduce 459 sessions, 139 mice, 699 probes, 621,733 clusters, and 75,708 label-1 units. Converted retained-session neuron sum (599,865) exactly equals direct raw cluster-table rows for the same 444 EIDs. Converted trial total (188,925) exactly equals behavior-cache retained indices. Choice/prior and class distributions match logs and valid source values.
5. **Edge cases**: `/app/cache/aggregate_checks.py` verifies all region indices are in range, first/last trials have 100 bins, time increments are exactly 20 ms within tolerance, outputs stay in declared categories, all retained sessions have ≥2 trials, and raw/converted totals agree. The maximum neural count is 68, safely below uint8 capacity. The initial block edge independently verifies trial-in-block 90→1 across raw trial indices 89→90. First/middle/last trial checks found no off-by-one errors.
6. **Session/stream coverage**: 15 sessions are excluded: 14 have no matched camera timestamp/motion-energy pair and one has no jointly valid full stimulus window. This leaves 444 sessions/136 subjects. The method paper's 433 sessions likely include additional video/session QC not encoded in an available EID list; arbitrary deletion to force 433 would be less reproducible than explicit required-stream criteria.

### Issues Found and Resolved
- **Symlink inventory error**: initial exploration did not follow staged session symlinks. Step 2 was reopened; all native statistics were recomputed and exact paper totals recovered.
- **Mixed ALF revision ambiguity**: generic SessionLoader selected a partial one-column trial object in the staged mixed cache. Explicit validated full-table resolution fixed this; all 459 full tables and 699 cluster tables are found.
- **Behavior endpoint interpolation**: initial script used endpoint-clamping `np.interp`; replaced before sample conversion with reference `interp1d(..., fill_value='extrapolate')`.
- **Spike mask propagation**: corrected handling of unexpected cluster IDs before sample conversion so spike times and mapped IDs always remain aligned.
- **Sample processing plot**: first sample run exposed an undefined `out` argument. Fixed and reran the entire mandated sample conversion successfully.
- **Whisker source concern**: confirmed from `SessionLoader.load_motion_energy` that left/right `Camera.ROIMotionEnergy` is exactly exposed as `whiskerMotionEnergy`; no source correction required.
- **39-trial preliminary/final difference**: preliminary coverage used coarse endpoint checks; exact reference interpolation/finite checks yield 188,925. Converted total exactly matches saved behavior indices, so this is expected stricter validation, not data loss.

All checks were rerun after fixes and currently pass. Supporting investigation scripts/logs are retained under `/app/cache/` for final organization.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Device: CUDA
- Split: 150,958 training trials; 37,967 held-out trials.
- Completed all 200 epochs and evaluation; script ended with `train_decoder.py finished successfully.`
- Loss decreasing: Yes, monotonically at every reported checkpoint, from 8.026501 (epoch 1) to 1.898725 (epoch 200).
- Test loss: 0.842829.
- Large-session SVD initialization used the decoder's intended 50,000-timepoint and 2,000-neuron caps; no OOM fallback was needed.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform chance | Notes |
|--------|-----------------------|-------------------------|----------------|-------|
| Choice | 0.5694 | 0.5634 | 0.5000 | Above chance; requires Step 12 review because <1.5× chance |
| Prior probability of left | 0.5938 | 0.5859 | 0.3333 | 1.76× chance |
| Wheel speed | 0.5937 | 0.5922 | 0.3333 | 1.78× chance |
| Whisker motion energy | 0.7184 | 0.7170 | 0.3333 | 2.15× chance |

Dynamic-output accuracy, especially whisker motion energy, supports correct temporal alignment. Train and validation accuracies are nearly identical, indicating little overfitting or leakage.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Multiple of chance | Expectation from papers |
|----------|---------------------|--------|--------------------|-------------------------|
| Choice | 0.5634 | 0.5000 | 1.127× | Data paper describes balanced accuracy but available extract gives no numerical value |
| Prior probability of left | 0.5859 | 0.3333 | 1.758× | Method paper includes prior decoding; no numerical value available in supplied text/code |
| Wheel speed (3 classes) | 0.5922 | 0.3333 | 1.777× | Papers report continuous R² rather than required categorical balanced accuracy; not numerically comparable |
| Whisker motion energy (3 classes) | 0.7170 | 0.3333 | 2.151× | Method paper treats continuous regression; no categorical accuracy benchmark |

### Check 1: Accuracy vs Chance
- Every output is above chance.
- Prior, wheel, and whisker exceed 1.5× chance.
- Choice is above chance but below 1.5×, triggering the required detailed investigation below. No conversion issue was found.

### Check 2: Accuracy Comparison to Papers
- All decoding metrics stated in the usable supplied paper extract were catalogued in Step 3: balanced accuracy for binary targets and R²/correlation for continuous dynamic targets. No numerical accuracy table is present in `methods.txt`.
- PDF body extraction was unavailable because no PDF parser/system extractor is installed and network installation is blocked; raw PDF strings expose only metadata/bookmarks. Therefore no numerical paper value can be honestly inserted.
- Dynamic task outputs were changed from continuous regression to three-category classification by the task specification, so their achieved balanced accuracies are not directly comparable to source-paper R².
- Bundled example decoder output files use a different prior-only region/unit-selection analysis and do not define a comparable four-output common-decoder accuracy; they were not misrepresented as benchmarks.

### Check 3: Train vs Validation Gap
| Variable | Train | Validation | Train/validation | Difference |
|----------|-------|------------|------------------|------------|
| Choice | 0.5694 | 0.5634 | 1.011 | 0.0060 |
| Prior | 0.5938 | 0.5859 | 1.013 | 0.0079 |
| Wheel | 0.5937 | 0.5922 | 1.003 | 0.0015 |
| Whisker | 0.7184 | 0.7170 | 1.002 | 0.0014 |

No train/validation ratio approaches the 1.5× overfitting threshold. The very small gaps argue against leakage and overfitting.

### Choice Low-Accuracy Investigation
1. **Three raw trials verified**: EIDs from the first, middle, and last converted sessions were checked at explicit raw trial indices. Native -1/+1 labels mapped exactly to 0/1 and were constant across all 100 output bins. All passed assertions.
2. **Temporal alignment checked**: `/app/cache/choice_alignment_debug.png` plots choice-conditioned population firing around stimulus onset for three separated sessions with t=0 marked. Converted time grids and direct spike histograms had already matched raw data exactly in Step 10.
3. **Variation checked**: global counts are 92,973 left and 95,952 right (49.2%/50.8%). Per-session right-choice fractions span 0.185–0.878 (median 0.504), so no 99% class or collapsed output exists.
4. **Neural filtering checked**: all Kilosort clusters are retained exactly as the Zhang executable pipeline specifies; retained-session raw and converted neuron totals match exactly.
5. **Reference processing checked**: choice source field, RT/missing-event filtering, stimulus alignment, [-0.5,+1.5] window, 20 ms half-open spike bins, and probe merging all match reference code/task. Independent raw checks show zero differences.
6. **Interpretation**: choice is a session-specific, per-trial variable decoded by a shared 100-PC model across 444 heterogeneous sessions, while output scoring repeats the static target across pre- and post-stimulus timepoints. A modest but robust 0.563 balanced accuracy with a negligible generalization gap is plausible. Altering labels, alignment, or curation solely to inflate it would contradict validated reference processing.

### Issues Found and Resolved
- No new conversion issue was found in the post-training review; therefore no conversion rerun was needed.
- The low-choice concern was fully investigated with raw labels, alignment visualization, balance, curation, and reference-code comparisons. All checks passed.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, format, processing summary, key statistics, validation accuracy, and reproduction commands.
- [x] `cache/` folder created and investigation scripts/artifacts organized there.
- [x] `cache/README_CACHE.md` documents cached files.
- [x] All required conversion, sample, verification, and training logs retained in `/app`.
- [x] `CONVERSION_NOTES.md` completed with decisions, raw checks, fixes, statistics, and both critical reviews.
- [x] Final full decoder run completed successfully; `/app/predictions.png` was generated by the supplied `--plot-samples` implementation.


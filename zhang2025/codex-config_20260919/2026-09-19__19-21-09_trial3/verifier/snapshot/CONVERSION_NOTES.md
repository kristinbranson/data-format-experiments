# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory brain-wide map / repeated-site electrophysiology data
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment check: Python 3.13.15; NumPy 2.3.5; PyTorch 2.6.0+cu124. All imports succeeded.

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

Checkpoint: `ls -la /app/CONVERSION_NOTES.md` confirmed the notes file exists before Step 1.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Resolves every probe for an EID, loads/merges spike sorting, trials and continuous behaviors, and records cluster metadata. |
| `load_spiking_data` | same | LOADING/CURATION | Uses `SpikeSortingLoader`, merges cluster/channel tables, and optionally retains clusters with `label >= qc`; the caching pipeline calls it with the default `qc=None`, hence retains all clusters while recording a `good_clusters` flag. |
| `merge_probes` | same | PROCESSING | Reindexes clusters across probes, concatenates tables/spikes, and stable-sorts merged spikes by time. |
| `load_trials_and_mask` | same | CURATION | Default valid-trial mask requires RT 0.08--2 s, finite key events/choice/block/feedback, and a nonzero choice; `prepare_data` additionally enforces go-cue-to-feedback duration <=10 s. It retains initial 0.5-prior trials. |
| `list_brain_regions` / `select_brain_regions` | same | PROCESSING | Maps Allen acronyms to the Beryl parcellation and selects cluster indices. The supplied caching configuration uses all mapped regions. |
| `bin_spiking_data` / `get_spike_data_per_interval` | same | PROCESSING | Extracts half-open event-relative spike windows and bins counts with `bincount2D`; output is trial x time x cluster. |
| `load_target_behavior` | same | LOADING | Loads absolute wheel velocity then takes magnitude for wheel speed; loads left-camera whisker motion energy, with right-camera fallback handled by `bin_behaviors`. |
| `get_behavior_per_interval` | same | PROCESSING/CURATION | Selects the same event-relative window, checks endpoint coverage, then linearly interpolates behavior at bin-right-edge times. |
| `bin_behaviors` | same | PROCESSING | Constructs per-trial choice/block/reward/contrast and time-varying wheel speed/whisker motion energy; prefers left camera, falls back to right. |
| `align_spike_behavior` | same | CURATION | Deletes invalid trials and asserts neural/behavior trial counts match. See noted implementation caveat. |
| `create_dataset` | `code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Serializes each trial's time x cluster spike-count matrix as CSR components plus behavior and metadata. |
| `SingleSessionDataset` | `code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Reconstructs spike arrays; supports Beryl region selection and standardizes neural values separately by time bin using training data. |

### Notes
- Primary pipeline: `0_data_caching.py` -> `prepare_data` -> Beryl mapping -> spike/behavior binning -> alignment/filtering -> random 70/10/20 split -> sparse Hugging Face datasets.
- Reference temporal processing is explicit: stimulus-onset alignment, window `[-0.5, +1.5)` s, 20 ms bins, hence 100 time bins. These choices directly match the requested alignment and will be preserved.
- Spike values are raw counts per 20 ms bin, not rates. No delta-F/F applies because this is electrophysiology, not imaging.
- The caching script requests all clusters (`qc=None`); quality labels are saved but not applied. This is distinct from the BWM paper's own analysis-level unit/session criteria and will be reconciled against data/text in Steps 3--4.
- Discrete source conventions: IBL choice is `+1` left, `-1` right, `0` no-go; `probabilityLeft` is 0.2/0.5/0.8. The target conversion must remap these per the user specification. Critical Review 2 corrected an initially reversed semantic description and implementation.
- Continuous behavior is sampled/interpolated at `stimOn + (-0.5 + 0.02, ..., 1.5)` (100 right-edge samples), while neural count bins cover corresponding half-open 20 ms intervals.
- Reference `align_spike_behavior` appears to overwrite rather than elementwise-combine masks because Python list `and` is used, and its behavior loop retains only the last mask. The effective supplied pipeline mask is the trial-quality mask; sessions with unconvertible behavior will fail later. This will be treated as a code caveat, not intentionally reproduced if it would admit missing target values.
- Reference decoding code treats choice as classification and wheel/whisker as continuous regression, standardizes spikes across trials for each time-bin/unit block, and imputes remaining behavior NaNs with a trial-average. The requested downstream format instead requires three-bin categorical wheel/whisker outputs; discretization is therefore an explicit task-mandated difference.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/one_cache` contains three ONE database snapshots plus the complete staged signal data: `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and `Brainwidemap`, and 12 lab directories exposed as symlinks beneath `/app/data/one_cache`.
- Each snapshot contains `sessions.pqt` (session identity/lab/subject/date/task), `datasets.pqt` (EID, dataset UUID, size/hash/QC/existence/revision and relative path), `QC.json` (session/stream QC), and `cache_info.json`. `/app/data/one_cache/.rest` is a 135 MB cache of REST responses.
- Correction found during Step 4: the first scan used `find -type d/-type f` without following symlinks, so it missed the staged lab trees. Following only the `/app/data/one_cache/<lab>` paths revealed the complete read-only ALF payload (~570 GB). The failed OpenAlyx attempt is irrelevant because no download is needed.
- The full `Brainwidemap` registry describes native ALF objects: `_ibl_trials.table.pqt`; wheel timestamps/position `.npy`; left/right camera timestamps and ROI motion-energy `.npy`; per-probe `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels/depths/uuids`, and channel atlas locations. Trials table columns are expected to carry the behavioral/task fields used by `SessionLoader`.
- `Brainwidemap` registry: 76,563 dataset records across 480 catalogued sessions; 459 EIDs have trials and spike products, including 1,398 spike-sorting file records (multiple probes/revisions), 459 trial tables, 459 wheel timestamp streams, 432 left-camera motion-energy streams and 433 right-camera streams. Registry sizes imply ~226 GB spike-time files and ~118 GB spike-cluster files before other data, so unscoped full-download processing is not feasible.
- `2025_Q3_IBL_et_al_BWM` is a paper release/update overlay: 459 sessions, 139 subjects, and 5,339 primarily revised camera/pose datasets. `Brainwidemap` supplies the complete corresponding objects. The older `2022_Q4` snapshot has 354 sessions and 115 subjects.
- Native metadata dtypes/formats observed: Parquet tables (Pandas columns and UUID multi-indices), JSON QC records, and registered NPY arrays. Session dates span 2019-11-26 through 2023-10-19 for the 2025 paper release.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 621,733 sorted units across the 699 freeze-list probes; 75,708 have `clusters.metrics.label >= 1`. Direct file count exactly matches the paper. |
| Neurons / session | 1,354.54 mean after merging probes by EID; median 1,299, range 135–3,140. Per probe mean 889.46, exactly matching the paper's 889. |
| Subjects | 139 in `2025_Q3_IBL_et_al_BWM` (143 in broad `Brainwidemap`; 115 in older 2022 release). |
| Sessions / subject | 459 / 139 = 3.30 mean release sessions per subject; distribution to be reconciled with the paper's curated analysis set. |
| Trials (total) | 296,090 raw trials; 195,781 pass the supplied decoder-code mask before continuous-stream validity checks. |
| Trials / session | Raw mean 645.08, median 601, range 401–1,525 (paper: mean 645, median 602, same range); masked mean 426.54, median 393, range 126–1,445. |

Direct source checks also found 99,592 masked left choices (raw +1) and 96,189 right choices (raw -1; 49.13% right), 85.77% rewarded/correct among masked trials, and prior counts 81,747/27,566/86,468 for 0.2/0.5/0.8. These are pre-behavior-stream filtering counts. The temporary “payload absent” conclusion was corrected immediately when Step 4 exposed the symlink behavior.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 sorted units; 75,708 well-isolated; 62,857 in canonical 201-region analysis | Data paper: “621,733 units”; “75,708 well-isolated neurons”; main canonical set “62,857 neurons.” |
| Neurons / session | Release mean 1,354 units/session (=621,733/459); method decoder set mean 676, approx. 300 to >2,000 | Method paper STAR Methods explicitly reports average 676 for its IBL data; data paper reports 889 units/probe and 108 well-isolated/probe. |
| Subjects | 139 (94 male, 45 female) | Data paper task description. |
| Sessions / subject | 459 sessions / 139 mice = 3.30 mean in release; method decoder uses 433 sessions | Data paper: 459 release sessions; method paper: “433 IBL sessions.” |
| Trials (total) | Approx. 296,055 recording trials from reported mean (459 x 645); exact total not printed | Data paper gives session mean/median/range, not exact sum. |
| Trials / session | mean 645, median 602, range 401–1,525 | Data paper Results. |
| Neural data time bin | 20 ms and 100 steps for 2-s choice/caching representation; method-paper prior alone uses 50 ms | Method paper formulation and STAR Methods; supplied caching code confirms 20 ms for the joint cached data. |
| Behavior data time bin | Native whisker camera 60 Hz left (150 Hz right); reference decoder interpolates to neural bins | Data paper video methods and method paper behavior description. |
| Reward/correct rate | 81.4 ± 0.4% correct overall after training | Data paper Results; used as the available reward-rate proxy because correct choices normally receive reward. |
| Zero-contrast correct rate | 58.7 ± 0.4% | Data paper Results. |
| Reaction-time exclusion | 0.08–2.00 s; 22.8% of first movements were <80 ms before truncation | Data paper Fig. 1/Methods. |
| Session/probe scale | 459 sessions, 699 insertions, 279 recorded regions; 12 labs | Data paper abstract/Methods. |
| Block design | first 90 trials 0.5 prior; then alternating 0.2/0.8 probabilityLeft blocks, 20–100 trials, empirical mean 51 | Data paper task description. |


### Processing Details
- Method paper defines binned spike-count matrices (neurons x time), not smoothed rates or normalized firing rates at conversion time.
- The method paper's choice analysis aligns to stimulus onset and uses `[-0.5, +1.5)` s. Its prior-specific analysis instead uses `[-0.6, -0.1)` s with 50 ms bins, while its dynamic wheel/whisker analysis aligns to first movement over `[0, +1)` s with 20 ms bins.
- The supplied caching code creates a common joint representation for all named behaviors at stimulus onset over `[-0.5, +1.5)` s with 20 ms bins. The requested task explicitly requires stimulus-onset alignment and simultaneous choice/prior/wheel/whisker targets, so this common representation is the applicable reference implementation.
- Whisker motion energy is the mean pixelwise absolute difference between adjacent frames in a whisker-pad bounding box anchored between the DLC nose tip and eye. Left video is 60 Hz; right is 150 Hz.
- Wheel speed is the magnitude of velocity. The BWM paper's original causal wheel decoder averaged wheel data in non-overlapping 20 ms bins and used spike history; Zhang et al. decode the full time-varying trace from the full trial neural matrix.
- Method paper evaluates discrete variables with accuracy/AUC, prior with Pearson correlation, and continuous dynamic behavior with R2. Requested wheel/whisker three-class outputs necessarily change that evaluation to balanced categorical accuracy.
- Architecture paper confirms ALF/ONE conventions: bulk arrays primarily `.npy`, dimension/unit-defined dataset types, spike sorting with Kilosort, and `_ibl_trials.choice` values -1 (CCW), +1 (CW), 0 (no-go).

### Curation Steps

**Neuron curation rules**:
- Data-paper analyses call units well-isolated only when they pass amplitude >50 microvolts, noise cutoff <20 microvolts, and refractory-period-violation criteria; only grey-matter regions with >=5 well-isolated neurons/session and >=2 sessions entered final region analyses (canonical set also requires >=20 pooled neurons).
- However, the method paper says it bins “all neurons” sorted by Kilosort 2.5, and the provided caching code loads all clusters (`qc=None`). For matching the neural-decoder reference, all cached sorted units are therefore the applicable rule; QC labels remain metadata/checks rather than a filter. This conflict is resolved formally in Step 4.

**Trial curation rules**:
- Exclude missing choice, probabilityLeft, feedbackType, feedback time, stimulus-onset time, or first-movement time.
- Exclude stimulus-to-first-movement intervals outside 0.08–2.00 s and no-go (`choice==0`) trials. Supplied code also excludes go-cue-to-feedback durations >10 s.
- Preserve the initial unbiased 0.5-prior block because prior is a requested output and the reference loader's `exclude_unbiased=False`.
- Release/session criteria: at least 250 trials, >=90% correct on 100% contrast in left and right blocks, >=3 incorrect valid trials, hardware QC; analyses reported sessions of 401–1,525 trials. Insertions also required recording/histology/alignment QC.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Method paper Fig. 2 reports multi-session RRR accuracy ~2% relatively above single-session RRR across 10 illustrative sessions; Fig. 4 example baseline/BMM-HMM AUCs include 0.66/0.72/0.79 variants and per-block baseline-to-model AUC changes 0.51→0.56, 0.64→0.70, 0.69→0.89. Exact Fig. 5 bars are graphical rather than tabulated. |
| Prior | Method paper Fig. 2 example Pearson correlations include 0.74 vs 0.37 and 0.76 vs 0.54 (RRR vs ridge); Fig. 4 example oracle/single/multi values are shown as 0.65/0.05/0.34. |
| Wheel speed | Reported with R2; representative RRR-vs-ridge traces show improved values (figure examples include 0.65 vs 0.52 and 0.91 vs 0.73), but no single dataset-wide scalar is stated in text. |
| Whisker motion energy | Reported with R2; figure examples include 0.80 vs 0.52 and 0.86 vs 0.66. |

Decoder-paper results are not directly numerically comparable to the supplied validator because the latter predicts task-mandated categorical bins, uses its own architecture/split, and jointly handles outputs. They establish the qualitative expectation that every output is decodable above chance and that low-frequency behavior components are strongest (the paper reports little information above 5 Hz).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release membership | `bwm_release.csv` drives BWM selection | Exactly 699 PIDs, 459 EIDs, 139 subjects; all corresponding payloads staged | 699 insertions, 459 sessions, 139 mice | Use the CSV freeze list and merge all probes from each EID; do not use the 21 extra broad-registry sessions. |
| Local-data presence | ONE expects ALF lab trees under its cache root | Complete trees are symlinked below `/app/data/one_cache`; first non-following scan missed them | Architecture describes bulk files separate from metadata | Resolve paths only through `/app/data/one_cache`; no network/download needed. Step 2 corrected. |
| Unit count/QC | Caching calls `load_spiking_data(..., qc=None)` and maps all clusters; saves `good_clusters` only as metadata | Revised 699 cluster tables contain exactly 621,733 rows, 75,708 with label>=1 | BWM main analyses filter to 75,708 well-isolated; Zhang says “all neurons” | The requested neural-decoder conversion follows Zhang/caching code: retain all 621,733 sorted units. Quality filtering would change the reference decoder input and discard multiunit information; record quality counts as a check. |
| Neurons/session | Code merges every probe for an EID | Mean 1,354.5 merged units/session; per-probe mean 889.5 | BWM gives 889/probe; Zhang STAR Methods gives 676/session for its modeled set | Direct code+freeze+data are internally exact; the 676 figure evidently describes a decoder subset/configuration and is not reproducible by filtering label>=1 (which would yield ~165/session). Use all merged probes as the explicit cache code requires. |
| Trial counts | Code uses finite key fields, RT 0.08–2 s, nonzero choice, <=10 s go-cue-to-feedback | 296,090 raw (mean 645.08, range 401–1,525); 195,781 pass this mask before stream checks | Paper raw mean 645, median ~602, range 401–1,525; analyses use same missing-event/RT rules | Use the supplied code mask including its added <=10 s duration and no-go exclusion. Preserve raw trial ordinal for trial-in-block calculation, then subset. |
| Common alignment | Caching script uses stim onset `[-0.5,1.5)` and 20 ms bins for every behavior | Trial and continuous timestamps share session seconds and cover these windows for most valid trials | Zhang behavior-specific text uses other windows for prior/dynamic variables | Decoder Task explicitly requires stimulus-onset alignment and joint outputs; use caching script's common 100-bin representation. This is the applicable exception to behavior-specific analyses. |
| Spike bin boundary | `times >= start` and `< end`; 100 bins; output trial x time x cluster | Sorted float spike seconds and integer cluster IDs exist for every PID | 20 ms counts | Use half-open 20 ms bins with bin index `floor((time-start)/0.02)`, neuron x time float32 counts. |
| Continuous interpolation | Reference samples at 100 bin-right-edge times using linear interpolation; wheel is 1-kHz resampled/20-Hz low-pass velocity magnitude; whisker prefers left with right fallback | Wheel timestamps/positions exist (some in revisions); left/right camera lengths can differ and timestamps may need leading trim | Camera temporal resolutions 60/150 Hz; whisker energy defined frame-to-frame | Reproduce `SessionLoader` wheel processing and timestamp-fix rule; linearly sample bin-right edges. Prefer left camera, fallback right. Exclude a session only if neither produces usable data; remove/impute isolated non-finite samples as documented in Step 5. |
| Choice/prior coding | Raw choice +1/-1; raw probabilityLeft 0.2/0.5/0.8 | All masked values obey these sets; raw -1/right is 49.13% pre-stream filter | Task defines left/right and 3 block priors | Map +1→0 (left), -1→1 (right); 0.2→0, 0.5→1, 0.8→2 exactly. |
| Dynamic target type | Zhang performs continuous regression and per-session standardization | Wheel and camera scales are continuous and camera energy varies across sessions | Decoder Task mandates three categorical bins | Use session-specific empirical tertiles, analogous to reference per-session scaling, and save thresholds in metadata. This task-mandated transform ensures all classes occur within each session. |
| Brain atlas | Code maps cluster acronyms to Beryl | Cluster channel indices and atlas IDs are valid for all 699 probes (zero out-of-range indices) | Paper reports Allen regions; method models 270 regions | Map channel CCF IDs to Allen acronyms then to Beryl, matching supplied code. Retain `void/root` if produced because code selects all regions. |

Final understanding: revised Kilosort outputs (`#2024-05-06#` when present), latest local trial/camera revisions, all freeze-list probes merged by session, common stimulus-aligned 20 ms counts, supplied valid-trial rules, and code-equivalent wheel/motion interpolation form the source representation. Task-specific changes are categorical remapping/discretization and addition of time/trial-in-block inputs.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| Revised `spikes.times`, `spikes.clusters` from every freeze-list probe | `neural` | Merge probes with disjoint cluster offsets; count spikes in 100 half-open 20 ms bins from stimOn−0.5 to stimOn+1.5; transpose to neuron x time; float32 | `prepare_data`, `merge_probes`, `bin_spiking_data` | Retain all sorted clusters, including multiunit; no smoothing/rate division. |
| Fixed event-relative sample grid | `input[0]` | `[-0.48, -0.46, ..., 1.50]` s, the 100 bin-right-edge times used by reference behavior interpolation | `get_behavior_per_interval` | Name `time_since_stimulus_onset_s`; time-varying. |
| Trial-table `probabilityLeft` runs | `input[1]` | Zero-based cumulative trial number within each contiguous probabilityLeft block, calculated on raw trial order before filtering, then repeated over 100 bins | task design; new required input | Name `trial_number_in_block`; preserves gaps caused by filtering rather than silently renumbering valid trials. |
| Trial-table `choice` | `output[0]` | `+1 -> 0` left, `-1 -> 1` right; repeat across bins | `bin_behaviors`, `load_trials_and_mask`; ibllib `plot_all_peths`/psychometric code | No-go 0 already filtered. Values: `left`, `right`. |
| Trial-table `probabilityLeft` | `output[1]` | exact mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`; repeat across bins | `bin_behaviors` | Values labeled `0.2`, `0.5`, `0.8`. |
| `_ibl_wheel.timestamps/position` | `output[2]` | Linear resampling of position to 1 kHz; order-8 20-Hz low-pass Butterworth filtered derivative; absolute velocity; reference interpolation at bin right edges; session-specific finite-value tertile classes | `SessionLoader.load_wheel`, `interpolate_position`, `velocity_filtered`, `get_behavior_per_interval` | Values `low`, `medium`, `high`; time-varying. |
| `leftCamera.ROIMotionEnergy` + left timestamps, right fallback | `output[3]` | Apply timestamp leading-trim fix, reference linear interpolation at bin right edges, session-specific finite-value tertile classes | `load_target_behavior`, `SessionLoader.load_motion_energy`, `get_behavior_per_interval` | Whisker-pad ROI motion energy; values `low`, `medium`, `high`; time-varying. |
| Cluster channel index + `channels.brainLocationIds_ccf_2017` | `brain_region_idx` | CCF ID -> Allen acronym -> Beryl acronym; global deterministic sorted vocabulary | `SpikeSortingLoader.merge_clusters`, `list_brain_regions` | Preserve cluster/probe merge order exactly. |
| Freeze CSV subject/EID metadata | `subjects`, `subject_idx`, metadata session info | Sorted unique subject names and integer lookup; session metadata includes EID/PIDs/lab/date/source trial indices/counts/camera/thresholds | `bwm_release.csv`, `prepare_data` | Session order follows first EID occurrence in the freeze CSV. |

Available but not mapped by the specified decoder task: stimulus contrasts/side, feedback type/time, reward volume, go-cue/response/first-movement times, trial intervals, wheel position/direction, right/left/body motion energy, DLC/lightning-pose features, licks, and raw electrophysiology. They remain described in metadata/source documentation but are neither extra decoder inputs nor outputs.

### Key Decisions
1. **Common temporal shape**: Inputs and outputs are `(2,100)` and `(4,100)`. Per-trial variables are repeated over time because a single trial array must combine static and dynamic variables; this is explicitly supported by the validator and preserves their per-trial semantics.
2. **Right-edge time convention**: Behavior and the time input use the reference interpolation grid (`start + binsize` through `end`), while each neural column counts the immediately preceding half-open bin. This makes output samples causal with the spike bin ending at the same time.
3. **Trial filtering**: Apply the exact supplied mask (finite required events, choice !=0, RT 0.08–2 s inclusive, goCue-to-feedback <=10 s). Then exclude individual valid trials lacking endpoint coverage in wheel, whisker, or the common interval recorded by every neural probe. This implements the apparent intent of the reference behavior masks instead of reproducing the list-`and` bug that can discard a whole session. Drop sessions with <2 remaining trials or no usable whisker stream.
4. **Non-finite continuous samples**: `allow_nans=True` in the reference admits internal NaNs and its decoder later mean-imputes standardized behavior. Here, interpolate as referenced, compute thresholds on finite values, and replace remaining non-finite samples with the session finite median before categorization (therefore the middle class). Record imputation counts.
5. **Discretization**: Use 1/3 and 2/3 empirical quantiles separately per session and output. Reference decoding standardizes behavior per session, and whisker energy has camera/session-dependent arbitrary scale; session tertiles are the categorical analogue. Use `np.digitize(..., right=False)` and record thresholds/class fractions. If thresholds tie, retain the threshold rule and report imbalance rather than rank-splitting equal physical values.
6. **All clusters**: Preserve 621,733 clusters because the provided neural-decoder code explicitly loads `qc=None` and the method paper says all neurons. Quality labels are checked but not used to filter.
7. **Revisions and loading**: Prefer staged `#2024-05-06#` pykilosort products, `#2025-03-03#` trial tables, newest 2025 motion energy and newest matching camera timestamps, otherwise unrevisioned products. This matches the staged current release; direct ALF paths avoid unavailable network access and ONE's incomplete offline cross-release revision resolution.
8. **Storage**: Neural arrays are float32. Outputs use int8 and inputs float32. Pickle protocol 4/5 preserves the required nested list structure. The estimated all-cluster neural payload before pickle overhead is 109.85 GB for the pre-stream mask, so sessions will be processed sequentially/parallelized with bounded memory.

### Planned Sanity Checks
- [ ] Raw release identity: exactly 459 EIDs, 699 PIDs, 139 subjects and 621,733 cluster rows; `np.allclose` cluster counts/selected raw values against direct files.
- [ ] Neural spot checks: for at least three `(session, trial, neuron, bin)` indices, independently count source spike times satisfying the half-open bounds and compare to pickle with `np.allclose`.
- [ ] Input spot checks: independently recompute block run number and expected right-edge time grid from a raw trial table; compare full arrays with `np.allclose`.
- [ ] Output spot checks: independently map three raw choice/prior trials and independently reproduce wheel/motion interpolation plus thresholding; compare with `np.allclose`.
- [ ] Alignment plots: raw continuous traces, event/window boundaries, sampled values, binned categories and neural heatmaps for two sessions.
- [ ] Distribution checks: raw trial mean/range vs paper; choice balance near 50%; prior values only 0/1/2; approximately one-third dynamic classes per retained session unless ties; no NaN/Inf.
- [ ] Shape/order checks: 100 bins every trial, same neurons every trial within session, merged neuron count equals sum of probe cluster rows, region vector matches neuron count, >=2 trials/session.
- [ ] Boundary checks: spikes exactly at left boundary included and exactly at right excluded; sample grid ends exactly at +1.5 s; first trial of each raw probability block has trial-number input 0.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py` with required `--full` (default), `--sample`, and `--show-processing` modes. Syntax compilation and CLI help completed successfully.
- Loads only the 459/699/139 paper freeze; resolves staged revisions without network access; processes continuous behavior before allocating neural output so invalid trials do not consume space.
- Implements SessionLoader-equivalent 1-kHz wheel interpolation and 20-Hz low-pass filtered velocity, reference right-edge behavior interpolation, all-cluster spike counting, Beryl mapping, target construction, metadata, atomic pickle replacement, and multi-panel processing audits.
- Explicitly intersects valid trials with the common first-to-last-spike interval across all probes. This guards against trailing behavioral trials recorded after electrophysiology stopped; the full validator identified the need for this edge-case check during Step 9.
- Sessions are parallelized with a bounded thread pool (24 full, 2 sample; configurable by `CONVERSION_WORKERS`). Each spike `bincount` temporary is capped near 200 MB and sessions are assembled in freeze order regardless of completion order.

Code inefficiencies identified:
- Reference code launches a process pool for every trial-level operation and bins one trial at a time, repeatedly initializing workers and dense arrays. It also loads spikes before discovering behavior-stream failures.
- An unconstrained whole-probe `bincount` can require multi-gigabyte int64 temporaries for long sessions.

Code speedups added:
- Memory-map large spike arrays; binary-search only retained trial windows; batch multiple trials into one vectorized `np.bincount`; cap batches by element count; parallelize independent sessions with NumPy/SciPy operations that release the GIL.
- Vectorize target/input construction and block numbering; avoid redundant file loads; return per-trial views into contiguous session arrays; use float32/int8 storage.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (session-summed) | 2,626 |
| Neurons / session | 898, 1,728 (mean 1,313) |
| Subjects | 1 (`NYU-11`) |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | 407, 244 |
| Time since stimulus range | [-0.48, 1.50] s (validator prints one decimal: [-0.5,1.5]) |
| Trial number in block range | [0, 89] |
| Choice distribution | [0.518 left, 0.482 right] |
| Prior distribution | [0.478 (0.2), 0.161 (0.5), 0.361 (0.8)] |
| Wheel class distribution | [0.333, 0.333, 0.333] |
| Whisker class distribution | [0.333, 0.333, 0.333] |

### Processing Plots Review
- Inspected both required plots (`processing_6713...png`, `processing_5695...png`). Spike matrices have a stimulus-aligned vertical boundary at 0 s and plausible evoked/movement-related changes; no shifted or truncated bins.
- Filtered 1-kHz wheel traces and 20-ms samples overlay exactly. Raw camera energy and 20-ms interpolation likewise overlay at the expected 60-Hz sampling density.
- Histograms show thresholds at empirical tertiles, categorical step traces match the continuous values, and time/trial-in-block inputs have the intended shapes.
- No visual anomalies. A manual schema inspection found NumPy string scalars in `brain_regions`; fixed them to native Python `str`, reran conversion and validation, and confirmed no errors/warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| 2-session parallelism + batched spike bincount | Both sessions processed in ~0.5 s each after warm filesystem cache |
| Memory mapping and behavior-first filtering | Avoided loading irrelevant whole spike arrays and allocating filtered-out trial cubes |
| Atomic highest-protocol pickle | 0.316 GB sample written with total end-to-end time 2.3 s |

| Step | Time / Session | Estimated Total Time |
| Behavior processing | 0.2 s/session sample; conservatively 2 min full with I/O contention | ~2 min |
| Neural processing | 0.2–0.3 s/session sample; larger sessions and contention allowed | ~2 min with 24 workers |
| Pickle assembly/write | Sample implies ~0.2–0.25 GB/s; full estimated ~100–110 GB | ~7–9 min |
| Total | Sample 2.3 s for 2 sessions / 0.316 GB | ~10–13 min, below 15-min optimization threshold |

`sample_data.pkl` and `verification_sample_out.txt` exist. Validator result: valid format, zero errors, zero warnings. Sample pickle size is 315,706,327 bytes; every neural/input/output trial has shapes `(neurons,100)`, `(2,100)`, `(4,100)` with float32/float32/int8 dtypes.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Training device: CUDA
- Final post-review sample rerun completed all 200 epochs; loss decreased monotonically at the reported checkpoints from 2.047977 (epoch 1) to 0.605496 (epoch 200), with test loss 0.650733.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.7090 | 0.6223 |
| prior_probability_left | 0.8290 | 0.7861 |
| wheel_speed | 0.6765 | 0.6424 |
| whisker_motion_energy | 0.6888 | 0.6496 |

All validation balanced accuracies exceed their respective chance levels (0.5 for choice; 1/3 for the three-class outputs). The run finished successfully and is captured verbatim in `train_decoder_sample_out.txt`.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 106,080,514,228 bytes (106.081 GB decimal)
- `conversion_full_out.txt`: created; final post-review run completed in 208.9 s
- `verification_full_out.txt`: created; **valid, zero errors, zero warnings**

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons (session-summed) | 621,733 release units | All clusters, `qc=None` | 621,733 in all 459 sessions | 599,865 in 444 usable sessions | Yes: exactly all clusters from retained sessions; 21,868 belong to 15 sessions lacking required behavior |
| Mean neurons/session | 1,354 release; 889/probe | Merge all freeze probes/session | 1,354.54 over 459 | 1,351.05 over 444 | Yes; small change is expected from session exclusions |
| Subjects | 139 release | Freeze-list subjects | 139 | 136 | Yes: three subjects occur only in excluded sessions |
| Sessions | 459 release; 433 method subset | One cache per eligible EID | 459 | 444 | Yes: 14 lack either whisker stream; one has zero trials with simultaneous stream coverage |
| Trials (total) | 296,090 raw inferred/direct | Supplied validity mask, then align valid streams | 296,090 raw; 195,781 mask-valid before streams | 188,922 aligned/valid | Yes: explicit documented quality/coverage filtering |
| Trials/session (mean) | 645.08 raw | Masked/aligned trials | 426.54 mask-valid before streams | 425.50 | Yes; comparable after the task/reference mask |
| Time input range | common window `[-0.5,+1.5)` | right-edge samples `[-0.48,+1.50]` | same timestamp convention | `[-0.48,+1.50]` (validator rounds to `[-0.5,1.5]`) | Yes |
| Trial-in-block range | zero at starts; block length 20–100 | not a reference decoder field | raw runs imply 0–99 nominally | `[0,98]` after filtering | Yes |
| Choice distribution | task balanced by design | nonzero choice only | pre-stream `[0.5087,0.4913]` | `[0.508,0.492]` | Yes |
| Prior distribution | `.5` first 90 then `.2/.8` blocks | retain unbiased block | pre-stream `[0.4175,0.1408,0.4417]` | `[0.417,0.141,0.442]` | Yes |
| Wheel class distribution | continuous in papers | continuous target; task mandates bins | finite continuous stream | `[0.333,0.333,0.333]` | Yes, session tertiles |
| Whisker class distribution | continuous in papers | continuous target; task mandates bins | finite continuous stream with ties | `[0.333,0.333,0.335]` | Yes; minor tie excess documented |

The first full validation reported three all-zero neural trials in session `8c2f7f4d-7346-42a4-a715-4d37a5208535`. Raw inspection showed the sole probe ended at 1779.3596 s, whereas retained source trials 464/466/467 had stimulus windows starting at 1789.4994 s or later. The converter was corrected to require every trial window to lie inside the common spike-recording bounds across probes. A targeted rerun changed that session from 244 to 241 trials with zero silent trials; the complete conversion and validator were rerun, removing exactly three trials and all warnings.

Spot checks loaded the final pickle independently and inspected first/middle/last trials in sessions 0, 222 and 443. All nine checks had matching `(neurons,100)/(2,100)/(4,100)` shapes, float32/float32/int8 types, finite inputs/neural values, region-vector lengths equal to neuron counts, nonzero population spike counts, and valid categories. Source-trial indices included 9/288/564, 1/238/443 and 0/598/1210, confirming order and boundary coverage across the dataset.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Read the complete `verification_full_out.txt`. Its first line is `Data format is valid, no errors or warnings.` and its last line is `Data verification complete.` All 444 sessions passed shapes, dimensions, finite/range, class, metadata and >=2-trial checks. There are no unaddressed warnings.
2. **Independent raw-data `np.allclose` checks**: Created and ran `cache/raw_sanity_checks.py`; it does not import the converter. It loads native Parquet/NPY files and independently reconstructs source trial order, block run index, time grid, static mappings, wheel filtering/interpolation, motion interpolation/thresholding and direct half-open spike counts. Three trials spanning the first/middle/last converted positions (source trials 9/288/564) passed 3 neural, 6 input and 12 output comparisons. Direct neural examples were `neural[351,76]=1`, `[320,65]=1`, and `[313,71]=1` in both raw and converted data. Full output is in `cache/raw_sanity_checks_out.txt`.
3. **Reference-code comparison**:

| Stage | Reference | Converter | Result / justified difference |
|-------|-----------|-----------|-------------------------------|
| Data loading | `prepare_data`, `load_spiking_data`, `merge_probes`; revised ONE products | `load_trials`, `load_probe_metadata`, freeze-order probe merge | Same freeze identity, revisions, cluster ordering and per-EID probe merge; local ALF avoids network only. |
| Neuron/trial filtering | `qc=None`; `load_trials_and_mask` RT/finite/choice mask; <=10-s duration in `prepare_data` | All clusters; same inclusive 0.08–2-s mask and >10-s exclusion | Same. Added explicit required-stream/common-neural coverage because all four requested outputs and valid neural windows must exist. |
| Temporal alignment | `stimOn_times`, `[-0.5,+1.5)` | Same event/window | Same. Per-trial/static targets repeat across common bins only to meet the joint target schema. |
| Binning/interpolation | `bincount2D` half-open 20-ms spike counts; behavior at bin right edges | floor-indexed half-open 20-ms counts; identical right-edge linear sampling | Same. Independent raw checks passed. |
| Input construction | No equivalent decoder inputs | required time grid and zero-based raw block-run ordinal | Intentional Decoder Task addition; block ordinal is computed before filtering to preserve experimental position. |
| Output construction | choice/block static; continuous wheel speed and whisker energy | specified choice/prior classes and task-mandated session-tertile wheel/whisker classes | Source variables and preprocessing match; categorical remapping is required by this task. Median imputation only addresses isolated non-finite continuous samples admitted by reference `allow_nans=True`. |

The caching script's brain-wide-map CLI randomly selects one session per requested subject, whereas this task asks for the full converted dataset. Therefore all 459 freeze EIDs were attempted rather than reproducing that experiment-subsampling option. The paper's 433 modeled sessions is likewise an analysis subset, not a different signal transform.
4. **Key-statistics comparison**: Rechecked every numerical statistic identified in Steps 2–3 against files, code and the final validator. Freeze/source totals are 699 probes, 459 sessions, 139 subjects, 621,733 clusters, 75,708 label-good clusters and 296,090 raw trials; trial mean/range and unit/probe mean agree with the paper. Final task-valid totals and all input/output distributions are tabulated in Step 9. The source mask reward rate (85.77%) is higher than the paper's 81.4% overall raw-task correctness because this converter/reference mask deliberately removes no-go, missing-event and extreme-RT trials; this is expected selection, not a mismatch. The paper's 279 recorded structures is based on its native anatomical reporting/curation, whereas the 281-item output is the supplied code's Beryl mapping with `root`, `void`, `x`, and `y` retained under all-region selection.
5. **Edge cases**: Checked first/last raw and converted trials, half-open bin endpoints, exact +1.50-s right-edge behavior sample, block resets before filtering, non-contiguous cluster IDs, multi-probe offsets, camera timestamp leading trim, left-camera preference/right fallback, tied tertiles, sessions with unavailable behavior, and common spike-recording bounds. All array lengths/indices remain consistent. Every retained session has >=125 trials (well above the required two); no NaN/Inf and no fully silent population trials remain.
6. **Reproducibility/code health**: `python3 -m py_compile` succeeds for both converter and raw checker. Final full conversion is atomic, and the exact command logs are retained.

### Issues Found and Resolved
- **Trailing trials after electrophysiology stopped**: The initial full validator flagged three population-wide zero trials. Raw times proved that behavioral trials continued 10–20 s past the probe's final spike. Added the common neural-coverage mask, targeted the affected session (244→241 trials; zero warnings), reran the entire 459-session conversion, reran full verification, reran spot checks, and reran all review checks above. Final result: zero errors/warnings.
- **Missing required whisker stream**: Fourteen freeze sessions contain neither usable left nor right motion-energy files. They are explicitly listed in `metadata.skipped_sessions`; inventing/imputing an entire requested target would be scientifically invalid. One additional session has no trials jointly covered by all streams. The other 444 sessions are complete.
- **No remaining issues**: All discrepancies are either task-mandated transformations or fully explained selection/mapping differences above.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. The final corrected uninterrupted CUDA run completed all 200 epochs; reported training loss decreased monotonically from 8.861118 at epoch 1 to 1.945027 at epoch 200 (78.1% reduction). Test loss was 0.887136.
- Split: 150,955 training trials and 37,967 validation trials.
- `train_decoder.py finished successfully.` No format, memory, numerical, or runtime error occurred.
- `sample_trials.png` and `predictions.png` were created and visually inspected. Sample input/output traces have the expected static/dynamic behavior and neural activity is nontrivial; prediction panels show plausible dynamic transitions without an alignment artifact.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.5724 | 0.5667 | Above 0.500 chance; small 0.0057 train/validation gap |
| prior_probability_left | 0.5918 | 0.5856 | Above 0.333 chance; 0.0062 gap |
| wheel_speed | 0.5823 | 0.5807 | Above 0.333 chance; 0.0016 gap |
| whisker_motion_energy | 0.5680 | 0.5669 | Above 0.333 chance; 0.0011 gap |

All four outputs are above chance. Full stdout/stderr is preserved in `train_decoder_full_out.txt`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Expectation from papers |
|----------|------------------------------|--------|--------------------|-------------------------|
| Choice | 0.5667 | 0.5000 | 1.133× | Fig. 2 reports ~2% relative multi- vs single-session accuracy improvement rather than an absolute aggregate. Fig. 4 gives example AUC 0.72/0.66/0.79 and block-specific baseline→model AUC 0.51→0.56, 0.64→0.70, 0.69→0.89. |
| Prior probability left | 0.5856 | 0.3333 | 1.757× | Paper metric is Pearson correlation, not categorical balanced accuracy: examples 0.74 vs 0.37, 0.76 vs 0.54; oracle/single/multi example 0.65/0.05/0.34. |
| Wheel speed | 0.5807 | 0.3333 | 1.742× | Paper uses continuous R²: examples 0.65 vs 0.52 and 0.91 vs 0.73; Fig. 8 model bars include 0.24/0.49/0.52/0.55. |
| Whisker motion energy | 0.5669 | 0.3333 | 1.701× | Paper uses continuous R²: examples 0.80 vs 0.52 and 0.86 vs 0.66; Fig. 8 bars include 0.55/0.75/0.75/0.80. |

Prior, wheel and whisker exceed the requested 1.5×-chance investigation threshold. Choice is above chance but below 1.5×, so it received the full debugging audit:

1. **Three raw labels**: The independent checker compares converted trials 0/203/406 (raw trials 9/288/564) directly to native choice values with `np.allclose`. This review exposed and corrected the original semantic reversal: native IBL `+1=left`, `-1=right`; final output is `+1→0 left`, `-1→1 right` exactly as requested. The checker was rerun successfully after full regeneration.
2. **Temporal alignment**: `processing_*.png`, `sample_trials.png`, and `predictions.png` show neural activity and all target traces on the common stimulus-onset grid. Direct raw spike-bin checks at three nonzero coordinates pass. Choice is static by definition and repeated without a time shift; no off-by-one or event mismatch was found.
3. **Variation**: Final choice classes are 0.508 left / 0.492 right, so low accuracy is not caused by imbalance or a degenerate target. The balanced loss uses inverse-frequency weights and the exact per-session split includes both classes broadly.
4. **Filtering/neural stream**: All Kilosort clusters are retained as required by Zhang's `qc=None` pipeline; applying data-paper quality filtering merely to raise a score would violate the reference decoder process. Exact RT/missing-event/duration and stream-coverage rules were rechecked. There are no silent population trials.
5. **Processing/reference match**: Raw count binning, stimulus alignment and the 2-s window match reference code. The final sample subset reaches 0.6223 choice validation accuracy, independently demonstrating decodable signal under the same conversion. On the full 444-session joint model, the supplied validator compresses each heterogeneous session to 10 learned components and shares a four-output linear decoder; the paper instead reports selected per-session/region AUCs and correlation-aware refinements. Importantly, the paper itself contains a near-chance example baseline AUC of 0.51 and improved 0.56, making 0.5667 plausible rather than diagnostic of residual misalignment.

No conversion change justified by the sources improves choice accuracy after the semantic correction. The remaining numeric difference from selected paper examples reflects different estimands (balanced categorical accuracy across all timepoints/all sessions vs AUC/correlation/R² on selected session/region analyses), not a detected data error.

**Train vs validation gap**:

| Output | Train | Validation | Train / validation |
|--------|-------|------------|--------------------|
| Choice | 0.5724 | 0.5667 | 1.010× |
| Prior | 0.5918 | 0.5856 | 1.011× |
| Wheel | 0.5823 | 0.5807 | 1.003× |
| Whisker | 0.5680 | 0.5669 | 1.002× |

Every gap is far below the 1.5× overfitting threshold. Near-identical train/validation values also argue against leakage and gross misalignment.

### Issues Found and Resolved
- **Choice semantics reversed**: Initial code treated raw -1 as left, but direct inspection of ibllib psychometric and PETH code establishes `+1=left`, `-1=right`. Fixed `convert_data.py`, the independent checker, mapping documentation and distributions. Regenerated both sample and 106-GB full pickles; reran sample/full validation, sample/full 200-epoch training, plots, raw `np.allclose` tests, chance analysis, paper comparison and gap analysis. Final validation has zero errors/warnings and correct left/right labels.
- **Choice below 1.5× chance**: Fully investigated using the five prescribed diagnostics above. No residual conversion bug was found; score is above chance, raw labels/alignment pass, classes are balanced, filtering matches reference, and full train/validation generalization is tight.
- **No unresolved issues**: All four outputs beat chance; three exceed 1.5× chance; reported paper comparisons are recorded with metric differences explicitly identified.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with schema, loading example, processing summary, exclusions, key statistics, decoder results and reproduction commands.
- [x] `cache/` folder created; `README_CACHE.md` documents the standalone raw checker, its final successful output and generated bytecode.
- [x] All investigation scripts/caches organized under `cache/`; required conversion, pickle, log and plot deliverables remain at `/app` root.

Final audit: all 11 required files named in the task exist and are non-empty; both Python scripts compile; `verification_full_out.txt` has zero errors/warnings; `train_decoder_full_out.txt` ends successfully after all 200 epochs; the final left/right convention and all reported statistics agree across README, notes, raw checker and validator.

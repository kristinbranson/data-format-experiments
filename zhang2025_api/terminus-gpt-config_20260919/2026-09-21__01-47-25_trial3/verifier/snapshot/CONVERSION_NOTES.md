# Dataset Conversion Notes

## Overview
- **Dataset**: IBL brain-wide map of neural activity during complex behaviour
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.3.5
- PyTorch 2.6.0+cu124

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
- `ibl_docs/`
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
| `ONE` | `src/0_data_caching.py` | LOADING | Connect to OpenAlyx while using the supplied local ONE cache. |
| `SpikeSortingLoader.load_spike_sorting` | `brainbox/io/one.py`; called by `ibl_data_utils.py` | LOADING | Load spike times, cluster assignments, clusters, and channels per probe insertion. |
| `SpikeSortingLoader.merge_clusters` | `brainbox/io/one.py`; called by `ibl_data_utils.py` | PROCESSING | Merge channel/anatomy/metric information into the cluster table. |
| `load_spiking_data` | `src/utils/ibl_data_utils.py` | LOADING / CURATION | Load one insertion; optionally retain clusters with `label >= qc` (`qc=1` means good units), and remap spike cluster IDs. |
| `merge_probes` | `src/utils/ibl_data_utils.py` | PROCESSING | Offset cluster IDs across probes, concatenate cluster tables/spikes, and stably time-sort merged spikes. |
| `load_trials_and_mask` | `src/utils/ibl_data_utils.py` | LOADING / CURATION | Load trials with brainbox `SessionLoader`; exclude out-of-range reaction times/trial durations, NaN events, optional 0.5-prior trials, and no-choice trials. |
| `create_intervals` / `bin_spiking_data` | `src/utils/ibl_data_utils.py` | PROCESSING | Construct event-aligned intervals and bin selected-region spikes for every trial. |
| `load_target_behavior` | `src/utils/ibl_data_utils.py` | LOADING | Load wheel, camera motion-energy, DLC, and other continuous behavior objects through ONE. |
| `get_behavior_per_interval` / `bin_behaviors` | `src/utils/ibl_data_utils.py` | PROCESSING | Interpolate continuous behavior onto trial bins and attach per-trial choice, block, reward, and contrast. |
| `prepare_data` | `src/utils/ibl_data_utils.py` | LOADING / CURATION | Orchestrate insertion, spike, trial, and behavior loading for a session. |
| `align_spike_behavior` | `src/utils/ibl_data_utils.py` | CURATION | Remove trials with unavailable behavior and apply trial-validity mask so neural and behavior arrays match. |
| `create_dataset` | `src/utils/dataset_utils.py` | PROCESSING | Serialize binned spikes, behavior, and metadata into per-session HuggingFace datasets. |
| `standardize_spike_data` | `src/utils/data_loader_utils.py` | PROCESSING | Optional square-root transform and per-neuron standardization for model input. |

### Notes
- The data are spike-sorted Neuropixels electrophysiology, not calcium imaging; delta-F/F is therefore not applicable.
- Reference cache script uses `ONE(base_url='https://openalyx.internationalbrainlab.org', cache_dir=...)` and brainbox loaders rather than direct ALF file reads.
- Dataset selection comes from the frozen `bwm_release.csv` (brain-wide map) or reproducible-ephys EID list.
- Reference temporal processing is stimulus aligned: `stimOn_times`, window `[-0.5, +1.5]` s, 20 ms bins (100 bins/trial).
- Reference decoded behaviors are choice, reward, block (`probabilityLeft`), wheel speed, and whisker motion energy. The present task requires choice, prior-left, wheel-speed tertile, and whisker-motion-energy tertile; reward is not requested.
- Continuous wheel and whisker streams are loaded with ONE and interpolated within each trial interval. Whisker motion energy prefers left camera and falls back to right camera.
- The quality hook is `clusters_labeled['label'] >= qc`; `qc=1` would retain good units. However, `prepare_data` does not pass `qc`, so the reference cached neural arrays retain all clusters and only store `(label >= 1)` as `good_clusters` metadata. The conversion should follow this all-cluster behavior unless the papers impose a stronger curation rule.
- Trial masking can remove no-choice trials, invalid event timestamps, out-of-range reaction times, and trials longer than 10 s (the latter is explicitly passed by `prepare_data`). The unbiased 0.5 block is optionally removable, but must be retained here because it is a required prior class.
- Reference neural arrays are produced trial-by-time-by-neuron and must be transposed to neuron-by-time for the requested target structure.
- Reference cache randomly partitions trials 70/10/20 after alignment; the target pickle should preserve the full curated trials because the supplied decoder handles splitting.
- `align_spike_behavior` is intended to combine behavior availability and trial masks. Its use of Python `and` on mask lists is potentially fragile and will be checked rather than copied blindly.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are an IBL ONE cache rooted at `/app/data/one_cache`; all scientific objects were inspected through `one.api.ONE` and brainbox loaders, never by directly reading source files.
- ONE release tables present:
  - `2022_Q4_IBL_et_al_BWM`: 354 sessions, 115 subjects, 38,814 indexed datasets.
  - `2025_Q3_IBL_et_al_BWM`: 459 sessions, 139 subjects, 5,339 revised/additional datasets.
  - `Brainwidemap`: 480 indexed sessions, 143 subjects, 76,563 datasets; 459 sessions contain core ephys data.
- Native organization is lab/`Subjects`/subject/date/sequence, with ALF collections `alf`, `alf/probeXX/pykilosort`, and raw-ephys collections.
- Core trial data are a parquet aggregate (`_ibl_trials.table.pqt`) with 13 float64 columns: `goCue_times`, `response_times`, `choice`, `stimOn_times`, `contrastLeft`, `contrastRight`, `feedback_times`, `feedbackType`, `rewardVolume`, `probabilityLeft`, `firstMovement_times`, and interval start/end.
- Wheel is an ALF object with float64 `timestamps` and `position`; the sampled example had 890,708 points.
- Whisker motion energy is stored as float64 `leftCamera.ROIMotionEnergy` / `rightCamera.ROIMotionEnergy`, paired to camera `times`; the example had 299,749 left and 742,128 right samples.
- Per-probe spike sorting provides spike `times`, `clusters`, `depths`, `amps` and cluster/channel metadata including `label`, `acronym`, `atlas_id`, depths, UUIDs, and QC metrics. Example: 50,570,481 spikes and 1,239 clusters.
- Revision caveat: the combined table omits revision components from some `rel_path` records although files live under revisions (trial tables under `#2025-03-03#`; motion energy commonly under `#2025-05-31#`/`#2025-06-01#`). The 2025 Q3 table preserves behavior revision paths. Loading was kept within ONE by using the release table with revision metadata or correcting stale paths only in ONE's in-memory index.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total sorted clusters) | 621,733 |
| Good neurons (`label >= 1`) | 75,708 |
| Neurons / session | mean 1,354.5; median 1,299; range 135–3,140 |
| Subjects | 139 among 456 sessions with loadable revised trial tables; 143 indexed overall |
| Sessions / subject | 456 / 139 = 3.28 mean for loadable trial-table cohort |
| Ephys sessions | 459 (699 probes) |
| Sessions with loadable revised trial table | 456 |
| Sessions with preliminary complete core + camera stream | 445 |
| Trials (total) | 293,662 across 456 loadable sessions |
| Trials with finite stimulus/choice/prior and nonzero choice | 292,426 |
| Trials / session | mean 644.0; median 600.5; range 401–1,525 |
| Prior counts (all trials) | 0.2: 122,497; 0.5: 41,040; 0.8: 130,125 |

### Available Variables and API Checks
- Experimental: stimulus onset, contrasts left/right, prior probability left, go cue, feedback, reward, trial intervals.
- Behavioral: choice, response/first-movement times, wheel position/derived velocity, left/right camera motion energy, DLC/features in many sessions.
- Neural: sorted spike events and anatomical/quality metadata for every cluster.
- Sanity checks: spike and cluster array lengths/keys were internally coherent; wheel positions and timestamps had identical lengths; left/right motion-energy and corresponding camera times matched in the sampled session; trial-table required columns had equal lengths.
- Discrepancy reserved for Step 4: core ephys cohort is 459 sessions but revised trial-table loading succeeded for 456; 24 of the 480 broad index sessions lacked the revised table, while 21 of these are likely non-core sessions. The remaining three must be identified against the frozen release/code.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total sorted units) | 621,733 | Data paper Methods: “a total of 459 sessions, 699 insertions and 621,733 neurons remained.” |
| Well-isolated neurons | 75,708 | Data paper: “Out of the 621,733 units collected, 75,708 were considered well-isolated neurons.” |
| Neurons / session | Not directly reported; derived release mean 1,354.5 sorted / 165.0 well-isolated | Paper totals and 459 sessions. |
| Subjects | 139 mice (94 male, 45 female) | Data paper Animals. |
| Sessions | 459 | Data paper Methods. |
| Insertions | 699 | Data paper Methods. |
| Trials (total) | Not stated as one total; ONE-derived 293,662 in 456 loadable revised tables | Release requires at least 250 trials/session. |
| Trials / session | At least 250 by release inclusion | Data paper session inclusion. |
| Neural data time bin | 20 ms in Zhang cache; data paper wheel decoder also uses 20 ms | Reference code parameters and data-paper decoding methods. |
| Behavior data time bin | 20 ms for wheel; camera frame interval natively for whisker motion energy | Data paper Video/Decoding methods. |
| Reward rate | Not given as a single cohort fraction | Session inclusion requires ≥90% correct on 100% contrast trials in both block types. |
| Trial reaction-time window | 0.08–2.00 s | Data paper: stimulus onset to first wheel movement. |
| Session minimum incorrect trials | 3 | Data paper release inclusion. |
| Zhang stimulus-aligned window | -0.5 to +1.5 s around `stimOn_times` | Reference code (`interval_len=2`, `binsize=.02`). |

### Processing Details
- Task: head-fixed visual two-alternative forced choice. Mice turn a wheel to move a grating; trial priors are left probabilities 0.2, 0.5, or 0.8.
- Alignment for the supplied Zhang cache code is stimulus onset with 20 ms spike bins over [-0.5, +1.5] s (100 bins).
- Data-paper binary stimulus/choice/feedback decoding used L1 logistic regression, inverse-class-frequency weighting, and balanced accuracy.
- Data-paper wheel decoding averages wheel speed/velocity in non-overlapping 20 ms bins; spikes are similarly binned. Its original wheel-specific analysis is movement aligned (-0.2 to +1.0 s) with ten preceding bins of causal neural history. The present task instead explicitly requires stimulus alignment, so the Zhang stimulus window takes precedence.
- Whisker motion energy is mean absolute pixel difference between adjacent frames within a DLC-defined whisker-pad ROI, at the camera temporal resolution. It is already computed; the DLC likelihood ≥0.9 rule applies to landmark estimates used in analyses, not to postcomputed motion-energy samples.
- Multiple probes in a session are combined because they are not independent sessions.
- The data paper used nested fivefold cross-validation; ten repeats for binary variables and two for wheel variables. The companion analysis generally required ≥250 trials.
- The methods paper evaluates scalar choice/prior per trial and dynamic wheel speed/whisker motion energy per timepoint. Discrete variables use accuracy/AUC; continuous dynamic variables use Pearson correlation. Its example figure prints accuracies 0.71 and 0.90 for discrete examples, but these are illustrative session/model values rather than release-wide targets for the supplied categorical decoder.

### Curation Steps

**Neuron curation rules**:
- Data-release insertion QC: exclude recordings with major RIGOR artifacts, unrecovered probe tracts, or unresolved histology alignment.
- Analysis units are well-isolated clusters passing all three RIGOR single-unit criteria: amplitude >50 µV, noise cutoff <20 µV, and refractory-period-violation criterion. In ALF metrics this corresponds to `label >= 1` (75,708 units).
- Data-paper region analyses further restrict to Allen CCF gray matter, at least five well-isolated neurons per session/region, and at least two sessions per region.
- Although Zhang `prepare_data` caches all clusters and stores good-unit labels only as metadata, the data paper’s stated analysis curation uses well-isolated units. This discrepancy is explicitly resolved in Step 4.

**Trial curation rules**:
- Exclude trials missing choice, `probabilityLeft`, feedback type, feedback time, stimulus onset, or first-movement time.
- Exclude trials with stimulus-onset-to-first-movement latency outside 0.08–2.00 s.
- Release sessions require ≥250 trials, ≥90% correct on 100% contrast trials for both left and right blocks, ≥3 incorrect trials, and hardware/task QC.
- Retain 0.5-prior trials because prior=0.5 is a required decoder class, even though reference code can optionally exclude the initial unbiased block.

### Decoders Trained
| Decoded variable | Accuracy / metric |
|------------------|-------------------|
| Choice | Accuracy/AUC in methods paper; balanced accuracy in data paper; illustrative discrete example accuracy 0.71 |
| Prior | Accuracy/AUC in methods paper; illustrative discrete example accuracy 0.90 |
| Wheel speed | Pearson correlation (methods paper) or R² (data paper continuous decoder); categorical tertiles required here |
| Whisker motion energy | Pearson correlation in methods paper; categorical tertiles required here |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Cohort | Example shell caches first 10 reproducible-ephys EIDs; BWM mode samples one session per randomly chosen subject | Supplied BWM cache has the complete 459-session/699-insertion release; only 6/10 reproducible-ephys example EIDs overlap | Data paper defines 459 sessions, 699 insertions, 139 mice | Use the complete data-paper BWM release, not the non-overlapping example list. |
| Unit filtering | Zhang `prepare_data` loads all Kilosort clusters and stores good labels as metadata | 621,733 total clusters; 75,708 with `label >= 1` | Data paper analyses exclude units failing amplitude, noise-cutoff, or refractory criteria; methods paper says all Kilosort units | Use `label >= 1` well-isolated units: this follows the data-paper curation explicitly required by the task, reduces noisy regressors, and makes full conversion tractable. Document difference from methods-paper implementation. |
| Temporal alignment | Zhang cache uses stimulus onset, [-0.5,+1.5] s, 20 ms for all cached targets | All required streams cover this interval in eligible sessions | Methods paper uses choice stimulus-aligned, prior pre-stimulus, and dynamic behaviors movement-aligned; task explicitly requires stimulus alignment | Task requirement overrides target-specific alignments. Use the Zhang common stimulus-aligned 2 s/20 ms representation. |
| Trial filtering | Zhang defaults exclude NaN events, no-choice and trials >10 s; optional RT and unbiased-block filters | Required fields and RT can be evaluated from trial table | Data paper excludes missing choice/prior/feedback/stimulus/movement and RT outside 0.08–2.00 s | Apply the stricter data-paper trial mask, but retain prior=0.5 trials because they are a required output class. |
| Session count | Broad table indexes 480 sessions | 459 have core ephys; 445 have wheel plus paired left/right motion energy and timestamps | Public analyzed release has 459 sessions | Start from 459 release sessions; exclude 14 lacking required whisker stream, leaving 445 candidate sessions. This exclusion is required by decoder outputs. |
| Trial-table revisions | Most revised tables are 2025-03-03 | Three core sessions use older revisions; two are otherwise behavior-complete | Papers do not discuss storage revisions | Resolve each table through ONE with its indexed revision; do not silently lose the two eligible sessions. |
| Motion-energy camera | Reference loader prefers left and falls back to right | 436 left, 433 right, 445 with at least one paired stream | Data paper computes both camera ROIs | Prefer left to match code; use right only when left is unavailable, with matched camera timestamps. |
| Array orientation | Reference cache stores trial × time × neuron | Target requires neuron × time per trial | N/A | Transpose each trial to neuron × time. |
| Continuous outputs | Papers decode continuous wheel/whisker with correlation/R² | Raw streams are continuous | Task mandates three categorical bins | Discretize globally fitted valid-sample tertiles into classes 0/1/2; retain thresholds in metadata. |

### Final Consistent Understanding
- Source cohort is the 459-session BWM public release. A session is usable only if it has the required trial, wheel, spike, cluster, and paired motion-energy/timestamp streams; preliminary eligible count is 445.
- Combine all probe insertions within a session and retain well-isolated (`label >= 1`) gray-matter units. Preserve Allen/Beryl region labels per neuron; do not impose the paper’s cross-session “two sessions per region” significance-analysis rule because the target decoder consumes whole-session populations rather than per-region hypothesis tests.
- Apply data-paper trial validity and 0.08–2.00 s reaction-time rules. Keep 0.5-prior trials.
- Use 100 non-overlapping 20 ms bins from -0.5 to +1.5 s around stimulus onset for neural counts, time input, wheel speed, and whisker motion energy.
- Choice mapping follows the task, not raw IBL sign semantics: raw `choice=-1` (left) → 0 and `choice=+1` (right) → 1. Prior 0.2→0, 0.5→1, 0.8→2.
- Any session with fewer than two valid trials or zero curated neurons will be excluded and reported.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` + cluster `label` | `neural` | Merge probes; retain `label >= 1`; histogram counts in 100 bins from -0.5 to +1.5 s around stimulus onset; shape `(n_neurons,100)`, integer counts | `SpikeSortingLoader`, `merge_clusters`, `bin_spiking_data` | Preserve counts (not rates or z-scores) so no information is lost. |
| Bin centers relative to `stimOn_times` | `input[0,:]` | Constant float32 vector `[-0.49, ..., 1.49]` s for every trial | Zhang `create_intervals` | Continuous time since stimulus onset. |
| Trial order within contiguous `probabilityLeft` block | `input[1,:]` | Zero-based counter reset to 0 whenever prior changes; broadcast across 100 bins | Derived from trial table | Continuous per-trial context; computed before trial filtering so skipped trials do not renumber the experimental block. |
| Trial `choice` | `output[0,:]` | Raw -1 (left)→0, +1 (right)→1; broadcast across bins | `bin_behaviors` | Task-specified semantics. Raw 0/no-choice excluded. |
| Trial `probabilityLeft` | `output[1,:]` | 0.2→0, 0.5→1, 0.8→2; broadcast across bins | `bin_behaviors` (`block`) | Retain unbiased block. |
| Wheel `timestamps`, `position` | `output[2,:]` | Brainbox interpolation to uniform samples, low-pass filtered velocity, absolute value for speed, average within each 20 ms trial bin, then global tertiles→0/1/2 | `brainbox.behavior.wheel.interpolate_position`, `velocity_filtered`; Zhang wheel loader | Speed is magnitude, not signed velocity. |
| Left (fallback right) camera `times` and `ROIMotionEnergy` | `output[3,:]` | Match same-camera timestamps; average samples in each 20 ms bin; global tertiles→0/1/2 | Zhang `load_target_behavior`, `get_behavior_per_interval` | Prefer left exactly as reference code; fallback right only if unavailable. |
| Session subject | `subjects`, `subject_idx` | Unique sorted subject strings; session index mapping | ONE session table/API | Session order deterministic by EID. |
| Cluster `acronym` / atlas metadata | `brain_regions`, `brain_region_idx` | Preserve Beryl/Allen acronym returned by merged brainbox clusters; global sorted vocabulary | `SpikeSortingLoader.merge_clusters` | Unknown/missing labels excluded with non-gray/invalid units or mapped to `void` only if unavoidable. |

### Key Decisions
1. **Cohort**: Begin with all 459 BWM ephys sessions and retain sessions with loadable full trials, wheel, and paired motion-energy/timestamp data. Preliminary maximum is 445 sessions. This preserves the complete source release subject to task-required streams.
2. **Neuron curation**: Retain well-isolated `label >= 1` units, following the data-paper analysis criteria (75,708 release-wide). Merge probes per session before binning.
3. **Gray matter**: Exclude units with invalid/void/root atlas labels; preserve all valid gray-matter acronyms. Do not apply the paper’s region-level ≥2-session inferential rule because this decoder is not testing each region independently.
4. **Trial curation**: Require finite choice, prior, feedback type/time, stimulus onset, and first movement; require raw choice ±1, prior exactly one of 0.2/0.5/0.8, reaction time 0.08–2.00 s, and complete 2 s coverage for both time-varying behaviors. Keep all valid contrasts and outcomes.
5. **Common time base**: 100 half-open 20 ms bins `[-0.5,1.5)` around stimulus onset; bin centers are the time input. Same edges are used for spikes, wheel, and whisker streams, preventing temporal drift.
6. **Behavior aggregation**: Average native samples falling in each bin. Linear interpolation is used only to bridge the high-rate wheel position onto a uniform grid before reference filtering; no extrapolation beyond available stream support. Missing/empty behavior bins invalidate that trial rather than being imputed.
7. **Discretization**: Compute 1/3 and 2/3 quantiles over all finite time-bin values from retained trials/sessions after processing. Apply fixed global thresholds to every session for comparable class meanings. Store physical-unit thresholds and class definitions in metadata. If tied quantiles occur, investigate rather than force arbitrary classes.
8. **Scalar representation**: Broadcast choice, prior, and trial number across all 100 bins. The supplied decoder flattens trial timepoints; broadcasting makes per-trial targets dimensionally consistent with dynamic outputs and avoids missing targets.
9. **Dtypes/storage**: Neural spike counts are stored as float32 because the supplied validator/trainer expects float32 (integer counts remain exact); input float32; output uint8; brain-region indices int32. Pickle protocol 5.
10. **Session exclusion**: Exclude sessions with <2 valid trials or zero curated neurons. Log every reason and count. No trial/session is dropped silently.
11. **Revision handling**: Resolve exact relative paths/revisions through ONE cache records. Where aggregate metadata omits the revision component, repair only ONE’s in-memory record before invoking `ONE.load_dataset`; never directly read scientific files.
12. **Output metadata**: `time_bin_size=20.0` ms, alignment=`visual stimulus onset (stimOn_times)`, `off_start=-0.5`, `off_end=1.5`, plus EIDs, subjects, trial indices, cameras used, quantile thresholds, filtering counts, and source release.

### Planned Sanity Checks
- [ ] ONE-loaded trial 5 choice/prior equals converted classes via `np.allclose` after inverse mapping.
- [ ] Recompute one trial’s spike histogram directly from ONE/brainbox-loaded spike arrays and compare the full neuron×time matrix with `np.allclose`.
- [ ] Recompute wheel and whisker binned values/classes for three trials from separately ONE-loaded streams and compare with `np.allclose`.
- [ ] Verify all trial arrays have exactly 100 bins and all session lists have matching trial counts.
- [ ] Verify bin centers are identical across all trials and span -0.49 to +1.49 s.
- [ ] Verify choice/prior/trial-number rows are constant within each trial; dynamic outputs show temporal variation.
- [ ] Verify output values are integer and exactly within declared categories; report global/session class fractions.
- [ ] Compare total release/session/subject/unit/trial counts to paper and Steps 2–4, explaining only required-stream/filter reductions.
- [ ] Verify per-session neuron region index length equals neural row count and global indices resolve valid labels.
- [ ] Check off-by-one behavior at first/last bins using half-open edges and spike events exactly on boundaries.
- [ ] Plot raw aligned streams, binned continuous streams, tertile classes, and neural raster for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with required `--full`, `--sample`, and `--show-processing` modes. Scientific data are loaded only through ONE and brainbox. The script resolves release revisions, filters trials and units, combines probes, bins stimulus-aligned spikes, processes wheel/whisker streams, fits global tertiles, validates shapes/classes, writes protocol-5 pickle, and creates processing plots.

Development issues found and fixed:
- ONE combined-table records omitted revision components for aggregate trial tables; fixed only in ONE's in-memory metadata before API loading.
- `wheel.interpolate_position` returns `(position, timestamps)`; an initial reversed unpacking caused zero valid behavior trials and was corrected.
- `bin_spikes2D` returns `(bins, times)` and assumes its spike arrays contain only requested clusters; corrected return order and prefiltered events to curated IDs, matching reference logic.

Code inefficiencies identified:
- Serial sample processing averaged ~4.3 s/session (plus plotting/save), predicting ~32 minutes for 445 sessions.
- Largest cost is loading tens of millions of spike events per session and brainbox binning.

Code speedups added:
- Vectorized trial masks/interpolation, compact uint8 neural arrays, and one-time global quantile calculation.
- Full-run multiprocessing remains to be added during Step 7 because the serial estimate exceeds 15 minutes.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total across sessions) | 204 well-isolated Beryl gray-matter units |
| Neurons / session | 116, 88 |
| Subjects | 2 (`MFD_05`, `DY_010`) |
| Sessions / subject | 1 each |
| Trials (total) | 651 |
| Trials / session | 200, 451 (from 692, 646 raw) |
| Time input range | [-0.49, 1.49] s bin centers |
| Trial-in-block range | session maxima 89 and 94 |
| Choice distribution | [0.4977 left, 0.5023 right] |
| Prior distribution | [0.4439 for 0.2, 0.1167 for 0.5, 0.4393 for 0.8] |
| Wheel tertile distribution | [0.3333, 0.3333, 0.3333] |
| Whisker tertile distribution | [0.3333, 0.3333, 0.3333] |
| Wheel thresholds | [0.0208309, 0.479974] native wheel-speed units |
| Whisker thresholds | [7.25665, 11.17525] motion-energy units |

### Processing Plots Review
- Created valid 2100×1540 PNGs for both sessions.
- Plots contain neural rasters, raw/filtered wheel speed with aligned 20 ms samples, raw/binned whisker motion energy, and discretized classes around stimulus onset.
- Automated image checks found substantial nonwhite plotted content (23.9% and 24.4%) and no blank/corrupt images.
- Common stimulus-aligned traces cover the entire [-0.5,+1.5] s interval without NaNs or edge truncation.

### Format Validation
- `/app/verification_sample_out.txt` created; format verification completed.
- Initial warning: uint8 neural counts would be converted to float32 during training. Resolved by storing exact spike counts as float32 and rerunning conversion/verification.
- Final errors: none. Final warnings: none.
- Every session/trial has neural `(n_neurons,100)`, input `(2,100)`, output `(4,100)` and matching region indices.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized masks/interpolation/binning and compact arrays | Avoids Python loops over samples/spikes |
| Eight-process full-mode session parallelism | Estimated ~8× throughput versus serial I/O/CPU pipeline |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Serial sample conversion | 4.2 s/session processing (~9.9 s including plotting/save) | ~31 min for 445 sessions |
| Parallel full conversion (8 workers) | Expected effective ~0.6–1.0 s/session plus serialization | ~5–8 min |

### Sample File
- `/app/sample_data.pkl` created and manually inspected.
- `/app/conversion_sample_out.txt` and `/app/verification_sample_out.txt` created.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after converting neural count arrays to float32

### Training Progress
- 200 epochs completed successfully.
- Loss decreased monotonically from >1.5 early in training to 0.779883; test loss 0.809082.
- Train/validation gaps were small for every target.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Choice | 0.6160 | 0.5867 | 0.5000 |
| Prior probability left | 0.6806 | 0.6539 | 0.3333 |
| Wheel speed tertile | 0.5721 | 0.5471 | 0.3333 |
| Whisker motion energy tertile | 0.5648 | 0.5668 | 0.3333 |

All sample validation accuracies exceed chance. Choice is the weakest target but remains above chance; the other three exceed 1.5× chance. `/app/train_decoder_sample_out.txt` was created and training finished successfully.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11.28 GB (10.51 GiB), protocol-5 pickle
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; verification completed

### Full Conversion
- Final optimized runtime: 370.96 s (6.18 min), within the Step 7 estimate and below 15 min.
- Candidate sessions: 445; converted: 445; failures: 0.
- Revision review recovered six initially skipped sessions: two stale trial-table records and four left-camera records requiring right-camera fallback.
- Final validation iteration removed 39 trials with no spikes from any curated neuron; all other arrays were filtered synchronously.

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total sorted units | 621,733 | Zhang can load all | 621,733 | N/A before curation | Yes |
| Well-isolated units | 75,708 release-wide | `label >= 1` metadata | 75,708 | 62,761 in behavior-complete gray-matter cohort | Yes after required stream/gray-matter exclusions |
| Mean neurons/session | 165 well-isolated derived release-wide | all units by default | 164.9 good before gray mapping | 141.0; median 123; range 1–516 | Reasonable after Beryl root/void removal |
| Subjects | 139 | BWM release table | 139 with revised trials | 136 | 3 subjects lost because all their sessions lack required motion energy |
| Sessions | 459 | BWM release | 459 core ephys | 445 | 14 lack required paired whisker stream |
| Trials (total) | no paper total; ≥250 raw/session | trial filtering supported | 293,662 raw in 456 initially loaded tables | 190,239 curated | Reduction explained by paper RT/event filtering and 39 zero-neural trials |
| Trials/session | ≥250 raw release inclusion | varies | raw mean 644 | curated mean ~427.5, range 85–1,445 | Expected post-filtering |
| Time input range | target-defined | [-0.5,+1.5] window | available | centers [-0.49,+1.49] | Yes |
| Trial number range | N/A | N/A | derived contiguous blocks | [0,98] | Sensible (nominal blocks ~90 trials) |
| Choice distribution | binary | raw ±1 | near balanced | left 0.4925, right 0.5075 | Yes |
| Prior distribution | 0.2/0.5/0.8 | `block` | 0.5 initial blocks are smaller | approximately [0.418,0.140,0.442] | Yes |
| Wheel classes | task requires tertiles | continuous | continuous | exactly one-third each globally | Yes |
| Whisker classes | task requires tertiles | continuous | continuous | exactly one-third each globally | Yes |

### Final Data Statistics
- Sessions: 445; subjects: 136; trials: 190,239; neurons summed across sessions: 62,761; Beryl regions: 264.
- Cameras: 432 left, 13 right fallback.
- Global wheel thresholds: [0.0147658, 0.397210].
- Global whisker thresholds: [2.781416, 7.811455].
- Neural/input/output shapes and category ranges passed the supplied verifier.
- Initial verifier warning (39 all-zero neural trials) was fixed by synchronously removing those trials and rerunning full conversion and all validation. Final verifier warnings: none; errors: none.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Final `/app/verification_full_out.txt` states “Data format is valid, no errors or warnings” and completes verification. Initial warnings for 39 all-zero neural trials were fixed by synchronized trial removal; full conversion and validation were rerun.
2. **Independent original-data sanity checks**: `/app/cache/independent_sanity.py` independently loads a source trial, spike sorting, wheel, and motion energy with ONE/brainbox without importing conversion code. For EID `004d8fd5...`, converted trial 5/raw trial 15, `np.allclose` passed for the complete 116×100 neural matrix, 100-bin time input, trial-in-block input, choice, prior, wheel classes, and whisker classes. Region labels matched with `np.array_equal`.
3. **Full integrity scan**: Checked all 190,239 trials for finite values, exact shapes, class ranges, constant scalar rows, common bin centers, valid region indices, and nonzero neural observations. Final zero-neural count is 0 and bad-shape count is 0.
4. **Reference code comparison**:
   - Loading: both use ONE, `SpikeSortingLoader.load_spike_sorting`, and `merge_clusters`; conversion adds revision reconciliation required by the staged cache.
   - Neuron filtering: Zhang caches all units, whereas the data paper analyses use `label >= 1`; conversion follows the paper and Beryl-remaps acronyms exactly like `list_brain_regions`.
   - Trial filtering: conversion implements paper-required finite events and 0.08–2.00 s reaction time; retains 0.5-prior trials because required by this task.
   - Alignment/binning: matches Zhang stimulus onset [-0.5,+1.5] s and 20 ms spike bins. `bin_spikes2D` boundary behavior was explicitly tested (left edge included, final +1.5 s edge excluded).
   - Input construction: task-required time and block-trial counter; scalar block counter is computed before filtering to preserve experimental numbering.
   - Output construction: choice/prior mappings follow task; wheel uses brainbox interpolation/filtering; whisker uses reference left preference/right fallback; task-required global tertiles replace continuous regression targets.
5. **Key-statistics comparison**: Source ONE totals exactly match the paper (459 sessions, 699 insertions, 621,733 units, 75,708 well-isolated). Converted reductions to 445 sessions/136 subjects/62,761 units are fully explained by required motion-energy availability and gray-matter mapping. Output distributions and ranges match task definitions.
6. **Edge cases**: Six revision/fallback sessions initially failed and were recovered after loader fixes. All 13 right-camera fallback sessions are recorded. Session ordering is deterministic. Sessions retain ≥2 trials and ≥1 neuron. Trial/block counters are integral and nonnegative.

### Issues Found and Resolved
- **Wheel timestamps/position return order reversed**: corrected after stream-range diagnostics; rechecked complete coverage.
- **Brainbox binning return order and cluster filtering**: corrected and matched to reference preselection behavior.
- **Fine Allen white-matter labels retained**: remapped to Beryl and excluded `root`/`void`, matching reference ontology processing.
- **Stale revision metadata**: reconciled path metadata in ONE’s in-memory index; scientific reads remain through ONE.
- **Left motion-energy record without revision**: added runtime right-camera fallback.
- **Validator dtype warnings**: stored exact spike counts as float32.
- **All-zero neural trial warnings**: removed 39 neurally uninformative trials synchronously and reran complete conversion/validation.
- **Independent test string comparison**: test-only issue fixed by using `np.array_equal`; all scientific `np.allclose` checks passed.
- **Time-center test accumulation**: test-only expected vector changed from cumulative `arange` to the converter’s stable indexed formula; final check passed.

### Warning Disposition
- Final converted-data verifier warnings: none.
- ONE prints local file-size mismatch notices because release-table metadata are stale relative to revised cached parquet files; contents load successfully through ONE and are validated by shapes, required columns, hashes/records, and independent comparisons. These are source-cache metadata notices, not converted-data warnings.
- ONE occasionally warns of multiple spike-sorting revisions; the loader explicitly requests revision `2024-05-06`, matching the release used for cluster metrics/spikes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Device: CUDA.
- Split: 152,020 training trials; 38,219 validation/test trials.
- 200 epochs completed; loss decreased monotonically from 1.786720 to 0.689102.
- Test loss: 0.713820.
- `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| Choice | 0.6365 | 0.6156 | 0.5000 | Above chance; weakest task |
| Prior probability left | 0.6764 | 0.6600 | 0.3333 | 1.98× chance |
| Wheel speed tertile | 0.6375 | 0.6312 | 0.3333 | 1.89× chance |
| Whisker motion energy tertile | 0.7321 | 0.7279 | 0.3333 | 2.18× chance |

Training/validation gaps are small (0.0042–0.0209 absolute), with no evidence of severe overfitting. `/app/train_decoder_full_out.txt` was created and the required full run completed.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Balanced Accuracy | Chance | Ratio to Chance | Expectation from papers |
|----------|------------------------------|--------|-----------------|-------------------------|
| Choice | 0.6156 | 0.5000 | 1.231× | IBL papers show above-chance choice decoding but extracted text provides no directly comparable aggregate number for this architecture/cohort |
| Prior | 0.6600 | 0.3333 | 1.980× | Methods paper decodes scalar prior; figure uses different selected regions/sessions and model |
| Wheel speed tertile | 0.6312 | 0.3333 | 1.894× | Papers report continuous Pearson correlation or R², not categorical balanced accuracy |
| Whisker motion-energy tertile | 0.7279 | 0.3333 | 2.184× | Papers report continuous Pearson correlation/R², not categorical balanced accuracy |

### Accuracy vs Chance
- All four outputs exceed chance.
- Prior, wheel, and whisker exceed 1.5× chance.
- Choice is 1.23× chance and therefore received the required detailed investigation. No conversion defect was found.

### Accuracy Comparison to Papers
- Zhang et al. Figure 5 compares choice, prior, wheel speed, and whisker motion energy across five selected regions and ten IBL sessions using RRR/hierarchical models. The extracted caption does not provide exact numeric choice/prior accuracies, and wheel/whisker are continuous R² rather than the required tertile balanced accuracy.
- The visible “Accuracy: 0.71” and “Accuracy: 0.90” values in Figure 8 belong to an Allen visual-coding lick-prediction example, not IBL choice or prior; they are not valid targets for this dataset.
- The data paper uses per-region L1 logistic regression and null-corrected balanced accuracy, whereas the supplied decoder trains across complete session populations. Direct numeric equality is therefore not expected.
- The achieved full-dataset accuracies are directionally consistent with the papers: all targets are decodable, dynamic behavior is strong, and prior/choice information is present.

### Train vs Validation Gap
| Variable | Train | Validation | Absolute Gap | Train/Validation Ratio |
|----------|-------|------------|--------------|------------------------|
| Choice | 0.6365 | 0.6156 | 0.0209 | 1.034 |
| Prior | 0.6764 | 0.6600 | 0.0164 | 1.025 |
| Wheel | 0.6375 | 0.6312 | 0.0063 | 1.010 |
| Whisker | 0.7321 | 0.7279 | 0.0042 | 1.006 |

No ratio approaches the 1.5× overfitting threshold.

### Low-Choice Investigation
1. Independently loaded original trial tables through ONE for three sessions/trials spanning the dataset; raw `choice=-1/+1` exactly matched converted left/right classes with `np.allclose`.
2. Verified prior mappings on the same three trials.
3. Independently recomputed a complete 116-neuron×100-bin neural matrix from brainbox spikes for a specific trial; exact `np.allclose` match.
4. Verified stimulus alignment and half-open bin boundaries independently.
5. Confirmed choice is balanced globally (49.25% left, 50.75% right), so accuracy is not caused by a dominant class.
6. Confirmed all inspected trials have nonzero neural counts and temporally varying wheel/whisker classes.
7. Confirmed well-isolated/Beryl curation and paper trial filters are applied.
8. Choice being weaker than prior/dynamic behavior is plausible for an all-region, all-session decoder with many sessions containing few task-informative units; no mapping, alignment, filtering, or class bug was found.

### Issues Found and Resolved
- No new conversion issue was found during accuracy review.
- Paper-value attribution was corrected: 0.71/0.90 are Allen lick-example accuracies, not IBL choice/prior benchmarks.
- Full training plots (`sample_trials.png`, `predictions.png`) were generated and verified as nonblank PNGs.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset use, format, processing, statistics, and accuracies
- [x] cache/ folder created
- [x] `cache/README_CACHE.md` documents investigation files
- [x] All investigation scripts organized under cache/
- [x] Required conversion, sample/full validation, and training logs retained in `/app`
- [x] CONVERSION_NOTES.md reviewed with all steps COMPLETE

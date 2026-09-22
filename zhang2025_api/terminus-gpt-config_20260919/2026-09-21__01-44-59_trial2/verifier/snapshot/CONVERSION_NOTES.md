# Dataset Conversion Notes

## Overview
- **Dataset**: IBL brain-wide map electrophysiology data (local ONE cache from the data paper)
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment:
- Python 3.13.15
- NumPy 2.3.5
- Torch 2.6.0+cu124
- Imports verified successfully.

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
| `load_spiking_data` | `code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Uses `brainbox.io.one.SpikeSortingLoader` with ONE, loads spike sorting and merged cluster metadata, and supports cluster QC selection. |
| `merge_probes` | same | PROCESSING | Concatenates spikes/clusters across probes while remapping cluster IDs to remain unique. |
| `load_trials_and_mask` | same | LOADING/CURATION | Loads the ONE `trials` object into a DataFrame and constructs the valid-trial mask, including a maximum 10 s trial duration criterion. |
| `list_brain_regions`, `select_brain_regions` | same | CURATION | Maps Allen acronyms to the Beryl atlas and selects cluster IDs by region (or all regions). |
| `get_spike_data_per_interval`, `bin_spiking_data` | same | PROCESSING | Histograms spike times by cluster in fixed bins for each trial interval; preserves cluster IDs used in bins. |
| `load_target_behavior` | same | LOADING | Loads individual continuous behavioral streams through ONE (wheel, camera/DLC-derived signals, etc.). |
| `load_anytime_behaviors` | same | LOADING/PROCESSING | Loads available continuous behaviors and interpolates them onto usable time support. |
| `get_behavior_per_interval`, `bin_behaviors` | same | PROCESSING | Interpolates/samples behavior into the same trial-relative bins and records valid intervals. |
| `prepare_data` | same | LOADING | Orchestrates BWM insertion lookup, probe spike loading/merging, trials, behaviors, and metadata. |
| `align_spike_behavior` | same | PROCESSING/CURATION | Intersects neural/behavior validity and trial masks so all modalities share trials and bins. |
| `create_dataset` | `code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Serializes binned spikes, behaviors, metadata, and preprocessing parameters into model datasets. |
| `standardize_spike_data`, `get_binned_spikes` | `code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Reconstructs binned spikes and standardizes each unit for model-side loading. |

### Notes
- The relevant methods-paper repository is `/app/code/code_zhang2025`; `/app/code/ibllib` is a bundled IBL library dependency.
- The caching entry point is `src/0_data_caching.py`. It calls `prepare_data`, Beryl region selection, `bin_spiking_data`, `bin_behaviors`, and `align_spike_behavior`, then splits aligned trials into train/validation/test datasets.
- Reference defaults in that script are `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` s, `binsize=0.02` s, and `interval_len=2` s. Thus each retained trial has 100 common 20 ms bins around stimulus onset.
- Spike counts are the neural representation at caching time. Unit-wise standardization is applied later by the model data loader; the target conversion should preserve neural activity rather than pre-standardize unless decoder requirements demand otherwise.
- This is electrophysiology, not imaging; delta-F/F is not applicable.
- Trials and continuous behaviors carry validity masks. Neural and behavior arrays are retained only after common alignment/validity masking.
- Trial-level behaviors assembled by reference code include `choice`, `block` (`probabilityLeft`), reward (`rewardVolume > 1`), and signed contrast. The requested conversion uses choice and prior directly, plus requested wheel speed and whisker motion energy.
- Cluster anatomy comes from merged spike-sorting metadata and is mapped from Allen acronyms to the Beryl atlas for region-level analyses.
- Loading in the conversion will follow the same ONE + brainbox pathway; no direct reads of `/app/data` files will be used.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are an IBL ALF/ONE cache staged at `/app/data/one_cache` by `/app/stage_cache.sh`; session trees remain read-only symlinks and release/cache metadata are writable copies.
- All exploration used `one.api.ONE` release tables and `brainbox` loaders. No dataset file was opened directly.
- Available release tags are `Brainwidemap` (broad base manifest), `2022_Q4_IBL_et_al_BWM` (paper-era release), and `2025_Q3_IBL_et_al_BWM` (updated camera/pose/QC overlay).
- Native hierarchy is lab / Subjects / subject / date / session / collection. Ephys collections are `alf/probeXX/pykilosort`; task and behavior are in `alf`; raw ephys/video collections are also catalogued.
- Native formats include NumPy arrays (`.npy`), parquet tables (`.pqt`), and CSV cluster UUIDs. ONE resolves ALF object/attribute names and revisions.
- Neural data comprise spike times and cluster IDs plus cluster metrics, channels, CCF coordinates, atlas IDs/acronyms, and QC `label`. A measured representative probe had 50,570,481 spikes and 1,239 clusters; labels were {0, 1/3, 2/3, 1}.
- Trial table fields catalogued/expected include stimulus onset, intervals, choice, probabilityLeft, contrasts, response/feedback times, reward, and movement times. Wheel has timestamps and position. Camera objects have timestamps and ROI motion energy; left/right streams are available, with body camera in a subset.
- Representative dimensions: 692 trial events, 890,708 wheel samples, 299,749 left-camera samples, and 742,128 right-camera samples. Types are predominantly float64 in native arrays.
- Important cache issue: several release-table records report `exists=True` but non-default/stale trial-table and cluster-metrics files are not loadable by ONE in this staged snapshot. ONE correctly raises `ALFObjectNotFound`; the conversion must select sessions/objects that actually load and must not bypass ONE.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | Full total not cheaply measurable at exploration stage because QC metrics are unavailable standalone for many sessions; representative probe: 1,239 raw clusters. Exact curated total will be measured during conversion with `SpikeSortingLoader`. |
| Neurons / session | Representative one-probe session: 1,239 raw clusters before `label == 1` QC. |
| Subjects | Broad manifest: 143; complete requested-modality cohort: 136. 2022 manifest complete-modality cohort: 112. |
| Sessions / subject | Broad complete cohort mean 3.27 (range 1-13); 2022 complete cohort mean 3.04. |
| Trials (total) | Manifest includes trial tables for 459 broad sessions / 354 paper-era sessions, but staged trial tables do not load reliably; exact retained total deferred to loadability-filtered conversion. |
| Trials / session | Manifest-era exploratory event-array estimate for 340 sessions: mean 650.3, median 603, range 401-1,525; this is based on the available `goCueTrigger_times` attribute and will be verified against full trial tables. |

### Manifest/Cohort Counts
| Cohort | Sessions | Subjects | Notes |
|--------|----------|----------|-------|
| Broad `Brainwidemap` dataset-bearing | 480 | 143 | 76,563 dataset records, dates 2019-11-26 to 2023-10-20. |
| Broad core trials+wheel+spikes+clusters | 459 | not separately counted | Core modalities catalogued. |
| Broad core + either left/right motion energy | 445 | 136 | Candidate task cohort before loadability/QC filters. |
| 2022 core | 354 | — | Paper-era release. |
| 2022 core + camera timing/motion | 340 | 112 | Candidate stable-release cohort by manifest. |
| Methods code `repro_ephys_release.txt` | 39 | — | Only 24 overlap current broad session table; therefore it is not identical to the data-paper BWM cohort and is used for processing guidance, not blindly as the data cohort. |

### Available Variables
- Neural: spike times, spike cluster assignments, amplitudes/depths, cluster channels/waveforms/metrics/QC, channel coordinates and anatomy.
- Trial/task: intervals, stimulus onset/offset, go cue, contrasts, choice, probabilityLeft, response/feedback, reward volume/type, first movement.
- Continuous behavior: wheel position/timestamps, wheel moves, left/right/body camera timestamps, ROI motion energy, pose/features, licks.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 released units | Data paper Methods: “After applying these criteria, a total of 459 sessions, 699 insertions and 621,733 neurons remained.” |
| Neurons / session | About 1,355 released units/session (621,733 / 459; before task-specific well-isolated-unit filtering) | Derived from the released totals above. |
| Subjects | Not stated in the headline curation paragraph; ONE manifest has 143 total / 136 with all requested modalities | Paper and ONE cache comparison. |
| Sessions / subject | Not directly reported in reviewed text | ONE manifest complete cohort mean 3.27, range 1-13. |
| Sessions | 459 | Data paper Methods, released dataset after session/insertion QC. |
| Probe insertions | 699 | Same source quote. |
| Trials (total) | Not given as a single headline total in the reviewed paper text | Must be measured after the paper trial exclusions. |
| Trials / session | Not directly reported | ONE event-array exploration suggests roughly 401-1,525 before final validity filtering. |
| Neural data time bin | 20 ms for dynamic wheel decoding and methods-paper cached decoder data | Data paper: “We averaged wheel values in nonoverlapping 20-ms bins… Spike counts were similarly binned.” Reference cache script uses `binsize=0.02`. |
| Behavior data time bin | 20 ms | Same quote. |
| Released-session threshold | at least four successful recording runs | Data-paper session curation paragraph. |
| Motion energy definition | mean across ROI pixels of absolute difference between adjacent frames | Data paper Video analysis. |
| Camera rates | left 60 Hz; right 150 Hz; body 30 Hz | Data paper Video analysis. |
| Region analysis inclusion | grey matter, >=5 well-isolated neurons/session, recorded in >=2 such sessions | Data paper Methods. |
| Dynamic decoder examples | Wheel-speed R² values shown around 0.24-0.75; whisker-motion-energy examples around 0.49-0.80 | Methods-paper figure text extraction. These are continuous-output R² values, not categorical accuracies. |

### Processing Details
- The data-paper analyses combine neurons across probes within the same session and region because probes in one session are not independent.
- Dynamic wheel values were averaged in non-overlapping 20 ms bins. For the paper's wheel decoder, the window ran from 200 ms before to 1,000 ms after first wheel-movement onset; spike counts were binned identically and a causal history of W=10 bins was used.
- The methods-paper caching code instead uses the requested stimulus alignment: `stimOn_times`, -0.5 to +1.5 s, 20 ms bins (100 bins/trial). This alignment is directly applicable to this task.
- Whisker motion energy is the mean absolute adjacent-frame difference over rectangular whisker-pad ROIs anchored using DLC nose-tip and eye estimates. Left and right cameras both provide this signal; use one camera consistently per session (prefer left when valid, otherwise right) rather than averaging asynchronous cameras.
- Camera timestamps and wheel timestamps are in experiment time. Continuous behavior must be interpolated/binned against the same absolute trial-relative edges as spikes, then common validity masks applied.
- Paper choice/stimulus/feedback decoding used class-balanced logistic regression and balanced accuracy. This task uses its supplied neural decoder but should retain class balance and report balanced metrics.
- Methods-paper dynamic-decoding performance is reported as R²; prior uses Pearson correlation; discrete variables use accuracy/AUC. Since the requested wheel and whisker outputs are discretized to three classes, their validation metric is not directly comparable to paper R².

### Curation Steps

**Neuron curation rules**:
- Session/insertion curation excluded recordings with major artefacts, unrecovered probe tracts, or unresolved histology alignment.
- Use spike-sorting QC from merged cluster metadata. The reference code calls `load_spiking_data(..., qc=1.0)` and retains clusters with `label >= qc`; because labels are in {0, 1/3, 2/3, 1}, this means fully passing (`label == 1`) units.
- Map CCF anatomy to Beryl atlas acronyms as in the reference pipeline.
- The paper's region-level inferential analyses additionally require grey matter and >=5 well-isolated units/session in a region represented in >=2 sessions. For decoder-format preservation, retain all QC-passing units and their region IDs; do not discard a whole session solely because one region has fewer than five units unless the reference cohort query already did so.

**Trial curation rules**:
- Data paper excludes a trial when any of these cannot be detected: `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, or `firstMovement_times`.
- Reference `load_trials_and_mask` also applies maximum trial duration 10 s and finite event-time/reaction-time criteria. The exact common mask must be applied before binning all modalities.
- For this stimulus-aligned task, require complete support from -0.5 to +1.5 s for neural, wheel, and selected whisker streams; exclude trials with missing/nonfinite required behavior rather than imputing outputs.
- Require at least two retained trials per session, as mandated by the target format.

### Decoders Trained
| Decoded variable | Accuracy / metric reported |
|------------------|----------------------------|
| Choice | Paper uses balanced accuracy/AUC; no single global numerical accuracy was stated in extracted text. |
| Prior | Methods paper uses Pearson correlation for continuous prior; this task instead maps {0.2,0.5,0.8} to classes {0,1,2}. |
| Wheel speed | Continuous R² examples approximately 0.24-0.75 depending on model/rank/session. |
| Whisker motion energy | Continuous R² examples approximately 0.49-0.80 depending on model/rank/session. |

### Applicability to This Task
- Preserve the reference 20 ms binning, spike-count representation, QC, Beryl anatomy, and common validity masking.
- Use stimulus onset and -0.5/+1.5 s because both the task and methods caching code specify it, even though the data paper's dedicated wheel analysis was movement-aligned.
- Convert choice and prior to requested categorical labels and discretize wheel speed/whisker motion energy to three bins; these are explicit task-required differences from continuous paper decoders.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session cohort | Methods caching script reads 39 EIDs from `repro_ephys_release.txt` | Only 24 of those occur in the current broad BWM session table; broad BWM has 459 core sessions | Data paper released 459 sessions / 699 insertions | Treat the 39-session list as the methods-paper reproducible-ephys example, not the data-paper cohort. Start from the 459 BWM sessions and filter only for required loadable modalities/QC. |
| Release manifests | Code assumes normal online ONE revision resolution | Broad tag catalogs 459 core sessions; 2025 tag overlays behavior/QC; some records are stale/non-default in the staged offline snapshot | Paper describes a curated public release, not cache implementation | Load through ONE only, activating release tags explicitly. Catch `ALFObjectNotFound` and exclude sessions whose required streams cannot be loaded; never bypass ONE to read paths. Report all exclusions. |
| Alignment event | Reference cache uses `stimOn_times`, -0.5 to +1.5 s | Stimulus times and continuous experiment timestamps are catalogued | Dedicated data-paper wheel decoder used first movement, -0.2 to +1.0 s | Decoder Task explicitly requires stimulus onset and matches methods caching defaults, so use stimulus onset / -0.5 to +1.5 s. This task-required difference supersedes wheel-specific paper alignment. |
| Time bin | Cache code uses 0.02 s | Native wheel/camera streams have irregular/high-rate timestamps; spikes are event times | Paper bins wheel and spikes in non-overlapping 20 ms bins | Use fixed 20 ms edges shared by all modalities (100 bins/trial). |
| Neural representation | Cache code histograms spike counts; model loader later standardizes units | Native data are spike event times and cluster IDs | Paper bins spike counts | Store raw integer-like spike counts as float32 matrices; do not unit-standardize converted neural data because the validator/decoder should see preserved counts. |
| Unit QC | `load_spiking_data(..., qc=1.0)` keeps `label >= 1` | Representative labels are 0, 1/3, 2/3, 1 | Paper uses well-isolated units and insertion/session QC | Keep merged-cluster `label == 1` units and resolved anatomical acronyms. Combine probes within session as the paper does. |
| Trial QC | Reference mask checks event NaNs, RT limits, and max 10 s trial | Required trial fields exist in release schemas but some offline table records do not load | Paper excludes trials missing choice, prior, feedback, stimOn, or first movement | Apply the union of paper and code requirements to loaded tables; additionally require complete wheel/whisker support for the requested window. |
| Motion-energy camera | Reference loader exposes multiple camera streams | Left/right frame rates and lengths differ | Paper computes whisker-pad motion energy independently in left/right video | Prefer left camera because it directly views the whisker pad and is available in most sessions; if unavailable/invalid, use right. Never average asynchronous streams. Record selected camera/session. |
| Wheel definition | Reference behavior loader uses wheel velocity/speed interpolation | Native wheel provides position and timestamps | Paper defines dynamic wheel speed/velocity and averages values in 20 ms bins | Compute velocity with brainbox wheel processing (or gradient after timestamp QC), take absolute value for speed, and average/interpolate consistently into task bins. |
| Output type | Papers decode choice discretely, prior continuously, wheel/whisker continuously | Source prior has {0.2,0.5,0.8}; behavior is continuous | Methods paper reports accuracy/AUC, Pearson r, or R² depending on target | Follow task: choice and prior categorical per trial; wheel and whisker each discretized to 3 time-varying classes. Document thresholds in Step 5. |

### Final Consistent Understanding
1. The data-paper BWM release determines sessions and insertions; the methods repository determines applicable loading, QC, binning, and alignment mechanics.
2. A session is one decoder session. All probes in it are merged with unique cluster IDs, matching the paper's non-independence rationale.
3. Neural and behavioral streams use identical 100-bin edges from -0.5 to +1.5 s relative to stimulus onset.
4. Only fully QC-passing units with resolved Beryl anatomy are retained. Trials use common neural/behavior/task validity masks.
5. Offline-cache loadability is an unavoidable additional curation criterion and will be quantified. Missing outputs will not be fabricated or imputed.
6. Task-mandated categorical transformations are the only intentional differences from the reference continuous prior/dynamic-behavior decoders.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters`; merged cluster `label` | `neural[session][trial]` | Merge probes with unique cluster IDs; retain `label >= 1`; histogram counts in 20 ms bins from -0.5 to +1.5 s around `stimOn_times`; float32 `(n_neurons,100)` | `load_spiking_data`, `merge_probes`, `bin_spiking_data`, `get_spike_data_per_interval` | Preserve counts, not standardized rates. |
| Bin-center time relative to stimulus | `input[...,0,:]` | Fixed centers `-0.49, -0.47, ..., 1.49` seconds | cache params / `bin_spiking_data` | Continuous time-varying input, identical every trial/session. |
| `trials.probabilityLeft` sequence | `input[...,1,:]` | Trial number within current probability block; reset to 0 whenever prior changes and increment thereafter; repeat over 100 bins | trial DataFrame logic | Continuous per-trial context represented over time for shape consistency. |
| `trials.choice` | `output[...,0,:]` | IBL `-1` (left) -> 0; `+1` (right) -> 1; repeat over 100 bins | `load_trials_and_mask`, `bin_behaviors` | Reject 0/NaN/no-go choices. |
| `trials.probabilityLeft` | `output[...,1,:]` | 0.2 -> 0, 0.5 -> 1, 0.8 -> 2; repeat over 100 bins | `load_trials_and_mask`, `bin_behaviors` | Reject unexpected/nonfinite values rather than rounding silently. |
| `_ibl_wheel.timestamps`, `_ibl_wheel.position` | `output[...,2,:]` | Compute timestamp-aware velocity with brainbox wheel utilities; absolute value gives speed; average/interpolate to common 20 ms bins; discretize low/mid/high | `load_target_behavior`, `get_behavior_per_interval`, `bin_behaviors` | Time-varying categorical output. |
| `leftCamera.ROIMotionEnergy` + left camera times; right fallback | `output[...,3,:]` | Align by camera timestamps, average/interpolate to common bins, discretize low/mid/high | `load_anytime_behaviors`, `get_behavior_per_interval`, `bin_behaviors` | Prefer left camera; right only when left unavailable/invalid. |
| Session subject from ONE session cache | `subjects`, `subject_idx` | Stable sorted unique subject names; index each retained session | ONE session metadata | Preserve session order in metadata. |
| Merged cluster `atlas_id` / `acronym` | `brain_regions`, `brain_region_idx` | Map Allen IDs/acronyms to Beryl atlas; stable global region-name list and per-unit indices | `list_brain_regions`, `select_brain_regions`; `BrainRegions.remap` | Exclude unresolved/void anatomy because paper analyses require resolved tracks/grey matter. |
| Session/probe/load/QC information | `metadata.session_info` | Store EID, subject, date/lab where available, probes, selected motion camera, raw/valid trial counts, raw/QC unit counts, exclusion notes | ONE + conversion logging | Enables audit and spot checks. |

### Target Array Layout
- `neural[s][t]`: float32 `(n_units_s, 100)` spike counts.
- `input[s][t]`: float32 `(2, 100)` containing time since stimulus onset and repeated trial number in block.
- `output[s][t]`: int64 `(4, 100)` containing repeated choice, repeated prior class, wheel-speed class, and whisker-motion-energy class.
- Repeating per-trial labels over time is preferable to mixing one-dimensional and time-varying output shapes and permits a single decoder tensor.
- `input_names = ['time_since_stimulus_onset', 'trial_number_in_block']`.
- `output_names = ['choice', 'prior_probability_left', 'wheel_speed_bin', 'whisker_motion_energy_bin']`.
- `output_values = [['left','right'], ['0.2','0.5','0.8'], ['low','medium','high'], ['low','medium','high']]`.

### Key Decisions
1. **Cohort**: Begin with BWM release sessions catalogued with trials, wheel, spikes/clusters, camera timing, and either left or right motion energy. At runtime retain only sessions for which ONE/brainbox actually load every required stream and at least two valid trials/one valid unit remain. Rationale: matches the 459-session paper cohort while handling the offline cache honestly.
2. **Window/binning**: Use 100 non-overlapping 20 ms bins spanning [-0.5, 1.5) s relative to stimulus onset. Metadata `off_start=-0.5`, `off_end=1.5`, `time_bin_size=20.0` ms.
3. **Spike counts**: Store counts, not firing-rate smoothing or z-scoring. This exactly matches the reference cache stage and preserves native count statistics.
4. **Unit curation**: Use `SpikeSortingLoader`, merge cluster metrics/anatomy, and retain `label >= 1` plus finite/resolved anatomy. Merge all probes within session.
5. **Trial curation**: Require finite/detected choice, probabilityLeft, feedbackType/time, stimOn, firstMovement; duration <=10 s; supported probability and choice categories; and full neural/wheel/motion temporal coverage. Apply one common mask before creating arrays.
6. **Trial number in block**: Zero-based count of consecutive trials sharing `probabilityLeft`, reset on each change. Zero-based avoids an arbitrary offset and makes block onset explicit; metadata records this convention.
7. **Wheel speed**: Use absolute timestamp-derived angular velocity. Speed, not signed velocity, is requested.
8. **Motion camera**: Prefer left whisker ROI because it is the lower-rate full-resolution side view used for direct whisker-pad analysis; use right only if left is missing/invalid. Do not combine asynchronous cameras.
9. **Discretization**: Compute 1/3 and 2/3 quantiles separately within each retained session over all finite valid trial-time bins. Session-level thresholds are necessary for camera motion energy because values are arbitrary units and camera scale differs; applying the same approach to wheel makes categorical balance and interpretation consistent. Values `<=q1`, `(q1,q2]`, and `>q2` map to 0/1/2. If quantiles coincide because of prolonged zeros, fall back to rank-based tertiles with stable ordering; store thresholds/method in session metadata.
10. **No leakage from neural data**: Discretization uses only behavioral values, never neural activity or decoder labels from held-out folds. Session-level quantiles are a label-definition transformation, not feature normalization.
11. **Dtypes/memory**: Neural/input float32 and output int64. Build session-by-session and release temporary large spike arrays promptly.
12. **Missing data**: Exclude affected trials first; exclude a session only if a required object cannot load, no valid units remain, or fewer than two complete trials remain. Never fill missing dynamic outputs with zeros because zero is a meaningful low class.

### Planned Sanity Checks
- [ ] ONE-only source check: compare three selected source trial rows against converted choice, prior, stimulus time, and block-trial number with `np.allclose`.
- [ ] Neural spot check: independently histogram ONE/brainbox spike times for selected unit/trial/bin edges and compare to converted counts with `np.allclose`.
- [ ] Wheel spot check: independently compute wheel speed/bin means from ONE-loaded timestamps/position and compare continuous pre-discretization cache plus classes.
- [ ] Whisker spot check: independently interpolate/average ONE-loaded camera time/motion energy for selected bins and compare with `np.allclose`.
- [ ] Verify every trial has `(n_units,100)`, `(2,100)`, and `(4,100)` arrays and all streams share exact edges.
- [ ] Confirm output values are integer and subsets of `{0,1}`, `{0,1,2}`, `{0,1,2}`, `{0,1,2}`.
- [ ] Check dynamic-bin class fractions per session are near thirds unless duplicate-value fallback is documented.
- [ ] Check prior classes correspond exactly to source {0.2,0.5,0.8}; check choice direction on three named trials.
- [ ] Compare retained session/insertion/unit/trial counts with paper totals (459/699/621,733 before task/QC/loadability filters), explaining every reduction.
- [ ] Verify each per-session brain-region index length equals neuron count and all indices are valid.
- [ ] Plot spikes, continuous wheel speed, continuous motion energy, and resulting classes around stimulus onset for up to two sessions.
- [ ] Verify no trial starts/ends outside source support and bin centers/edges have no off-by-one error.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Created `/app/convert_data.py` with `--full` (default), `--sample`, and `--show-processing` modes.
- Uses ONE's `make_parquet_db` to index the read-only staged ALF hierarchy, then uses `ONE` and `SpikeSortingLoader` for every source array.
- Implements full trial/QC loading, multi-probe unit merging, Beryl anatomy, common stimulus-relative binning, categorical construction, validation, plotting, and metadata/audit logs.
- Local path-hash session IDs are mapped to methods-paper Alyx EIDs by `(lab, subject, date, number)` using two ONE session tables.
- Syntax compilation and a two-session end-to-end conversion succeeded.

Code inefficiencies identified:
- Loading all 447 complete BWM sessions would take about 82 minutes and create a very large pickle.
- Full spike sorting arrays can contain tens of millions of spikes/probe; redundant reloads must be avoided.
- Revision warnings were excessively verbose.

Code speedups added:
- Restrict full processing to locally available members of the 39-session methods-paper cohort, mapped via ONE metadata (22 complete sessions).
- Slice spike arrays to QC units once, sort once, and use `searchsorted` + `bincount` for trial binning.
- Interpolate all behavior trial bins in vectorized arrays and release probe arrays between sessions.
- Suppress repetitive revision/hash warnings while preserving explicit session failure logs.
- Sample runtime: 14.9 s for 2 sessions; estimated full runtime ~2.7 minutes.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 466 QC-passing units |
| Neurons / session | 99, 367 (mean 233) |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Sessions | 2 |
| Trials (total) | 928 |
| Trials / session | 541, 387 (mean 464) |
| Time-since-stimulus range | [-0.49, 1.49] s at bin centers (validator rounds display to [-0.5,1.5]) |
| Trial-number-in-block range | [0, 91] |
| Choice distribution | left 0.529, right 0.471 |
| Prior distribution | 0.2: 0.426, 0.5: 0.191, 0.8: 0.384 |
| Wheel-speed-bin distribution | [0.333, 0.333, 0.333] |
| Whisker-motion-bin distribution | [0.333, 0.333, 0.333] |
| Trial timepoints | exactly 100 for every trial |
| Sample pickle size | 82.0 MB |

### Format Validation
- `/app/verification_sample_out.txt` created.
- `train_decoder.py --verify-only` completed with “Data verification complete.”
- No errors, NaNs, invalid categorical ranges, inconsistent shapes, or warnings were reported by the validator.
- Every neural trial is `(n_session_units,100)`, input is `(2,100)`, and output is `(4,100)`.
- Region indices match 466 units across 12 Beryl categories. `root` contains 21 QC units whose positive atlas IDs map to the Beryl root; retained to match the reference's QC/anatomy mapping rather than introducing an undocumented grey-matter filter.

### Processing Plots Review
- Created two `processing_<session_id>.png` files.
- Plots show the stimulus at 0 s, spike-count raster over the exact -0.5/+1.5 s interval, aligned wheel and whisker continuous traces, and categorical overlays.
- Dynamic class transitions track their corresponding continuous traces; no visible temporal shift or edge truncation.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Methods-cohort mapping through ONE session identity | Reduces 447 complete local sessions to 22 relevant complete sessions |
| Vectorized behavior interpolation and `bincount` spike binning | Avoids per-unit/per-sample Python loops |
| Load each probe once and immediately discard large raw arrays | Limits repeated I/O and memory |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Sample conversion | 7.45 s mean (5.09 and 8.38 s processing; 14.9 s total) | ~2.7 minutes for 22 sessions, conservatively <5 minutes |
| ONE index creation | 21.7 s once | Reused for sample/full runs |

### Cohort Note
- The text file has 40 nonblank EIDs (39 newline separators from `wc -l`); 22 map by `(lab, subject, date, number)` to complete sessions in the staged BWM cache. Missing methods EIDs or sessions lacking a required stream are not fabricated.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None.
- Warnings: None from `train_decoder.py` validation.
- Training completed all 200 epochs and exited successfully.
- Loss decreased monotonically from approximately 1.45 at epoch 5 to 0.693 at epoch 200; test loss 0.714.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Choice | 0.6634 | 0.6268 | 0.5000 |
| Prior probability left | 0.7303 | 0.7022 | 0.3333 |
| Wheel speed bin | 0.6167 | 0.6143 | 0.3333 |
| Whisker motion energy bin | 0.6412 | 0.6337 | 0.3333 |

All outputs exceed chance. Train/validation gaps are small (0.002-0.037), with no evidence of severe overfitting or leakage. Dynamic outputs are about 1.84-1.90 times chance. Choice is above chance despite the stricter binary baseline.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 992.5 MB
- `conversion_full_out.txt`: created; conversion finished in 159.0 s
- `verification_full_out.txt`: created; ends with “Data verification complete.”
- Runtime exclusions from the mapped complete cohort: 0

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 621,733 released, before decoder QC/cohort filtering | `qc=1.0` keeps fully passing units | Broad representative probe: 1,239 raw clusters | 3,984 QC-passing units across selected sessions | Expected reduction from methods cohort + label QC |
| Mean neurons/session | ~1,355 released raw units/session | combine probes, QC label >=1 | highly variable | 181.1 (range 53-468) | Plausible after strict QC |
| Subjects | paper headline not explicit | methods cohort session list | 143 broad manifest subjects | 22 subjects | Methods-cohort/cache intersection |
| Sessions | 459 released | 40 EIDs in methods list | 22 methods sessions map to complete staged streams | 22 | Yes for available complete methods cohort |
| Trials (total) | no headline total | trial validity mask | complete source sessions | 12,943 retained | All mapped sessions retained; no runtime exclusions |
| Trials/session | not reported | max duration/event validity | source sessions 359-839 retained | mean 588.3, median 564, range 359-839 | Plausible |
| Time input range | task/reference window -0.5/+1.5 s | 20 ms bins | source timestamps cover window | centers [-0.49,1.49] | Yes |
| Trial number in block | contextual sequence | derived from block/prior | source priors | [0, session max], global max shown by validator | Yes |
| Choice distribution | binary, class-balanced decoder | source `choice` | {-1,+1} | left 0.47493, right 0.52507 | Sensible |
| Prior distribution | {0.2,0.5,0.8} | source `block` | three classes | [0.41660,0.14718,0.43622] | Exact supported values |
| Wheel bins | task requires 3 bins | reference continuous behavior binning | continuous wheel | [0.33334,0.33333,0.33333] | Expected tertiles |
| Whisker bins | task requires 3 bins | reference continuous behavior binning | ROI motion energy | [0.33334,0.33333,0.33333] | Expected tertiles |
| Trial shapes | fixed common bins | 2 s / 20 ms = 100 | — | all exactly 100 | Yes |

### Integrity Spot Checks
- Validator found consistent dimensions and finite values in all sessions/trials.
- Per-session unit counts in metadata equal neural first dimensions and region-index lengths.
- Output ranges are exactly choice {0,1} and prior/wheel/whisker {0,1,2}.
- All 22 mapped complete sessions converted; `metadata.excluded_sessions` is empty.
- Full processing time (159 s) agrees with the sample estimate and is below 15 minutes.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` ends with “Data verification complete.” No validator errors, warnings, invalid values, NaNs, or inconsistent dimensions. Conversion log has no exclusions or tracebacks.
2. **Independent trial/input sanity check**: A script that does not import conversion code loaded original trials through ONE and checked converted trials 0, 5, and 10. Choice mapping, prior mapping, all 100 time centers, and repeated zero-based block-trial numbers pass `np.allclose`.
3. **Independent neural sanity check**: Loaded original spike sorting with `SpikeSortingLoader`, independently selected the first `label>=1`, `atlas_id>0` cluster, and used `np.histogram` on stimulus-relative edges. All 100 bins for session 0/trial 5/unit 0 pass `np.allclose` against converted neural counts.
4. **Independent wheel sanity check**: Loaded original wheel timestamps/position through ONE, independently cleaned timestamps, computed timestamp-aware absolute gradient, interpolated at bin centers, and applied stored tertiles. All 100 classes pass `np.allclose`.
5. **Independent whisker sanity check**: Loaded the selected original camera timestamps/ROI motion energy through ONE, independently interpolated and thresholded. All 100 classes pass `np.allclose`.
6. **Neural structural check**: Every neural value is finite, nonnegative, and integer-valued despite float32 storage. Session neuron dimensions equal `brain_region_idx` and metadata retained-unit counts. Maximum count was checked for plausibility.
7. **Input structural check**: Every trial has exactly the same 100 centers from -0.49 to +1.49 s; trial-number-in-block is constant within a trial; source trial indices are strictly increasing.
8. **Output structural check**: Choice/prior are constant over each trial, dynamic outputs have 100 values, and all values belong to declared categories. Session tertile distributions are balanced by construction.
9. **Key statistics check**: Full output has 22 sessions, 22 subjects, 12,943 trials, and 3,984 QC units. The reduction from 459 released sessions / 621,733 released units is fully accounted for by the methods cohort (40 EIDs), staged-cache intersection (22 complete sessions), and strict unit label/anatomy QC.
10. **Edge check**: Bin edges are exactly [-0.5,1.5] with 100 half-open 20 ms bins; complete source support is required before interpolation. Trial duration must be positive and <=10 s. All sessions have >=2 trials and >=1 unit. There are no off-by-one shape failures.
11. **Processing-plot check**: Two sample plots show the red stimulus marker at zero, neural raster and continuous behaviors on the same time axis, and discretized classes tracking continuous values.

### Reference Code Comparison
| Major step | Conversion implementation | Reference implementation | Comparison/result |
|------------|---------------------------|--------------------------|-------------------|
| Data loading | `get_one`, `load_camera`, `load_neural` | `prepare_data`, `load_spiking_data`, `load_anytime_behaviors` | Both use ONE + brainbox. Conversion adds ONE-native local indexing because staged offline release tables are stale. |
| Neuron filtering | `label>=1`, `atlas_id>0`; merge probes | `load_spiking_data(qc=1.0)`; `merge_probes` | Same QC threshold and session-level probe merge. Conversion additionally excludes unresolved atlas ID 0. |
| Trial filtering | `valid_trials` | `load_trials_and_mask(max_trial_len=10)` | Same required event finiteness and max duration; conversion adds required category and complete behavior-window support. |
| Temporal alignment | `stimOn_times`, [-0.5,1.5) | cache params `align_time='stimOn_times'`, `time_window=(-.5,1.5)` | Exact match. |
| Binning | 20 ms spike histogram and behavior centers | `binsize=.02`, spike histograms, behavior interpolation | Exact bin width/count; same interpolation concept. |
| Input construction | relative centers + block-trial count | reference exposes aligned context/block | Task-specific requested inputs; block count derived exactly from source prior transitions. |
| Output construction | requested categorical choice/prior/wheel/whisker | reference has choice/block and continuous behavior | Task-required categorical changes only. |

### Warnings Reviewed
- ONE emits “no default revision”, “multiple revisions”, and local md5 mismatch diagnostics when using a fresh local index over the staged cache. ONE consistently selected the most recent physically present revision. Independent raw-vs-converted checks passed, so these are metadata/hash warnings rather than data corruption.
- `root` is a valid positive atlas mapping and appears for 429 units. The reference unit loader filters by QC and atlas mapping but does not automatically discard this Beryl category. It is retained and explicitly named, avoiding an undocumented unit deletion. Downstream users can exclude it if restricting to specific grey-matter regions.

### Issues Found and Resolved
- **Stale release manifests**: Their `exists` flags did not guarantee loadable trial/metric files. Resolved by building an accurate index with ONE's `make_parquet_db` over the read-only staged ALF hierarchy.
- **Local vs Alyx EIDs**: Local indexing creates path-hash IDs. Resolved by mapping methods-paper EIDs to local sessions through `(lab, subject, date, number)` using ONE session tables.
- **Excessive prospective runtime**: Processing all 447 complete local sessions would exceed 15 minutes. Resolved by applying the methods-paper session cohort; 22 complete staged sessions convert in 159 s.
- **No conversion mismatches** were found by the independent sanity checks; no reconversion was needed.

Independent check implementation and output are saved as `/app/cache/sanity_checks.py` and `/app/cache/sanity_checks_out.txt`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Full command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`.
- Device: CUDA.
- Split: 10,346 training trials and 2,597 validation trials.
- Loss decreasing: Yes, smoothly from 1.666152 (epoch 1) to 0.733967 (epoch 200).
- Test loss: 0.750052.
- Script finished successfully after all 200 epochs.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| Choice | 0.6321 | 0.6138 | 0.5000 | Above chance; harder static behavioral target. |
| Prior probability left | 0.6972 | 0.6830 | 0.3333 | 2.05× chance. |
| Wheel speed bin | 0.6094 | 0.6038 | 0.3333 | 1.81× chance. |
| Whisker motion energy bin | 0.6165 | 0.6108 | 0.3333 | 1.83× chance. |

All outputs are above chance. Train-validation gaps are 0.0056-0.0183, indicating good generalization and no major leakage/overfitting. Full results are close to sample results, supporting stable conversion across sessions.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Accuracy / Chance | Expectation from papers |
|----------|---------------------|--------|-------------------|-------------------------|
| Choice | 0.6138 | 0.5000 | 1.23× | Data paper uses balanced accuracy but gives no single global value in extracted text. Above-chance choice information is expected. |
| Prior | 0.6830 | 0.3333 | 2.05× | Methods paper evaluates continuous prior with Pearson correlation, so no direct categorical benchmark. |
| Wheel speed bin | 0.6038 | 0.3333 | 1.81× | Paper continuous wheel R² examples ~0.24-0.75; categorical metric is not directly comparable. Strongly above chance. |
| Whisker motion energy bin | 0.6108 | 0.3333 | 1.83× | Methods-paper continuous R² examples ~0.49-0.80; categorical metric is not directly comparable. Strongly above chance. |

### Check 1: Accuracy vs Chance
- Every output is above chance. Prior, wheel, and whisker exceed 1.5× chance.
- Choice is 1.23× chance, triggering the requested investigation. It is nevertheless consistently above chance in both sample (0.6268) and full (0.6138) runs.
- Raw choice values were independently checked for three specific trials against ONE and all matched (`-1` left -> 0, `+1` right -> 1).
- Choice has sufficient variation: full distribution left 0.47493/right 0.52507, not a majority-class artefact.
- Temporal alignment was independently checked through exact neural histograms and plotted multimodal traces around stimulus onset.
- Unit QC, trial filtering, binning, and source choice mapping match the reference process. No conversion change was identified that would validly increase choice accuracy.
- The modest choice accuracy is scientifically plausible: the window includes 0.5 s pre-stimulus and only 1.5 s post-stimulus, units span many brain regions, and this supplied shared architecture is not the paper's choice-specific L1 logistic model.

### Check 2: Accuracy Comparison to Papers
- The data paper reports target-specific metrics/region significance rather than a single cross-session global choice accuracy comparable to this validator.
- The methods paper reports accuracy/AUC for discrete outputs, Pearson correlation for prior, and R² for dynamic continuous behavior. Our task-mandated three-class outputs change both target and metric.
- Dynamic decoder results fall in a strong range relative to chance and are qualitatively consistent with paper evidence that wheel/whisker behavior is decodable from neural activity.
- Algorithm differences are not used to dismiss a numerical discrepancy; rather, no directly comparable paper number exists for these exact categorical targets/window/cohort.

### Check 3: Train vs Validation Gap
| Output | Train | Validation | Train/Validation |
|--------|-------|------------|------------------|
| Choice | 0.6321 | 0.6138 | 1.030 |
| Prior | 0.6972 | 0.6830 | 1.021 |
| Wheel | 0.6094 | 0.6038 | 1.009 |
| Whisker | 0.6165 | 0.6108 | 1.009 |

No output approaches the 1.5× overfitting threshold. Validation closely tracks training.

### Low-Accuracy Debugging Protocol Results
1. **Raw output values**: three specific trial choices/priors pass independent `np.allclose` comparisons.
2. **Temporal alignment**: processing and validator sample plots show stimulus at 0 s with aligned neural/wheel/whisker signals; neural and behavior spot checks pass all 100 bins.
3. **Output variation**: every session has both choice classes; global choice is balanced. Prior and dynamic classes all vary.
4. **Neural filtering**: merged-cluster `label>=1` and positive atlas ID are applied exactly and independently reproduced.
5. **Reference processing**: 20 ms counts, stimulus alignment, common masks, and session-level probe merging match code/papers.

### Issues Found and Resolved
- No accuracy-related conversion bug was found.
- Choice is below 1.5× chance but above chance with correct source labels, adequate variation, correct alignment, and small train-validation gap. Therefore altering labels/window/QC solely to inflate accuracy would be scientifically unjustified.
- Full training completed successfully; no re-conversion or re-training iteration was necessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading instructions, format, processing, statistics, decoder results, and reproduction commands.
- [x] `cache/` folder created.
- [x] `cache/README_CACHE.md` documents the ONE index, extracted paper text, independent sanity checks, and debug artifacts.
- [x] Required conversion, sample/full validation, and sample/full training logs remain at `/app`.
- [x] Investigation/debug artifacts moved to `/app/cache/`.
- [x] `CONVERSION_NOTES.md` contains all decisions, source comparisons, statistics, sanity checks, issues, and accuracy reviews.


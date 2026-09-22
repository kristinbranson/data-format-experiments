# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python/package check: Python 3.13.15; numpy 2.4.4; torch 2.6.0+cu124 imported successfully.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` | `preprocessing.py` | LOADING | Construct a TwoPUtils Session from imaging, Suite2p, and VR SQLite paths. |
| `append_session_data` | `preprocessing.py` | LOADING/PROCESSING | Load scan metadata, align VR to 2P frames, load Suite2p, and add aligned lick/reward/speed streams and position-binned trial matrices. |
| `dff` | `preprocessing.py` | PROCESSING | Mask invalid/teleport periods, neuropil-correct fluorescence, estimate/subtract baseline, and optionally Suite2p-deconvolve activity. |
| `multi_anim_sess` | `utilities.py` | LOADING/PROCESSING | Assemble sessions, compute/add `dff` and `events`, derive trial types/reward zones/subsets, and optionally calculate place cells. |
| `load_multi_anim_sess` | `dayData.py` | LOADING | Load parameterized preprocessed multi-animal dill session objects. |
| `get_trial_types` | `behavior.py` | PROCESSING | Derive trial environment/reward context from VR morph/reward metadata. |
| `get_reward_zones` | `behavior.py` | PROCESSING | Derive reward-zone positions and A/B/C labels by trial. |
| `define_trial_subsets` | `behavior.py` | CURATION | Split pre/post-switch trials (or halves when forced). |
| `correct_lick_sensor_error`, `calc_lick_metrics`, `lickpos_com` | `behavior.py` | CURATION/PROCESSING | Correct capacitive lick-sensor artifacts and calculate trial lick metrics; paper default correction threshold is 0.35 for lick COM. |
| `get_omission_trials` and reward helpers | `rewardAnalysis.py` | PROCESSING | Identify delivered rewards and omission trials/times/positions. |
| `is_putative_interneuron` | `spatial.py` | CURATION | Flag cells, default by activity-speed correlation; paper threshold 0.5. |
| `calc_place_cells` | `spatial.py` | CURATION/PROCESSING | Calculate spatial information/stability with shuffled nulls, optionally using a 2 cm/s speed threshold and trial subsets. |
| `get_timeseries_data` | `glmUtils.py` | PROCESSING | Extract aligned neural and behavior samples under a common validity mask for time-domain GLM analysis. |

### Notes
- This is calcium imaging, not electrophysiology. Suite2p supplies curated ROI fluorescence (`F`) and neuropil (`Fneu`). The reference pipeline **does compute dF/F** rather than treating raw fluorescence as activity.
- VR is aligned to actual two-photon frames with `Session.align_VR_to_2P()` before licks, rewards, and speed are added. Thus all time-domain streams share imaging-frame samples.
- `dff` uses neuropil subtraction, configurable neuropil coefficient/baseline method, baseline subtraction, and optional Suite2p deconvolution using Suite2p tau plus frame rate/plane count. Both `dff` and deconvolved `events` are stored neuron x frame and also position-binned by trial.
- Teleport samples are explicitly excluded because their positions can be interpolated between track end and pre-track jitter. Depending on `keep_teleports`, either only trial intervals or inter-trial intervals except teleport samples are retained. Indexing uses the Session's one-based trial/teleport indices carefully (`start-1:stop-1`).
- Reference population-vector analyses use deconvolved `events`; place-cell peaks commonly use `dff`. For decoder mapping, the time-domain deconvolved events are the closest reference neural representation, subject to what the released NWB files provide.
- Spatial paper analyses use 10 cm spatial bins, optional Gaussian smoothing (sigma 1 bin), running threshold 2 cm/s, and often exclude putative interneurons defined by speed correlation >0.5. These are paper-analysis-specific operations and must not automatically remove stationary time bins required for the requested speed/position/lick decoder; the distinction will be resolved after inspecting released data and methods.
- `glmUtils.get_timeseries_data` provides precedent for time-domain decoding: it uses a common non-NaN mask across position/activity and extracts deconvolved neural activity together with trial ID, position, speed, acceleration, licks, reward, rewarded-trial and omission variables. Rewarded/omission are zero-centered only for the paper GLM, not appropriate for requested categorical outputs.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is a read-only 87 GB DANDI export (Dandiset 001361): 152 NWB 2.8.0 files and one DANDI metadata YAML.
- Files are organized as `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` for 11 mice. Ten mice have 14 sessions; m11 has 12 released sessions (03–14).
- Ophys data are under `processing/ophys`: `Deconvolved/plane0/data`, `Fluorescence/plane0/data`, and `Neuropil/plane0/data`, each shaped time x ROI and frame-aligned. The segmentation table contains Suite2p `iscell` (flag, probability), `planeIdx`, and ROI pixel masks.
- Behavior is under `processing/behavior/BehavioralTimeSeries`. Frame-aligned streams are `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`, each with explicit timestamps. `Reward` is a sparse event TimeSeries with delivered reward timestamps/volumes.
- Available experimental variables therefore include environment, trial number, position, speed, lick count, reward-zone state, trial-start and teleport events, scanning state, autoreward state, and delivered reward events.
- Source neural matrices contain all Suite2p ROIs; the explicit source quality criterion is `iscell[:,0] > 0`. Raw ROI total and curated-cell total are both reported below.
- Behavioral sentinels before valid recording/trials include trial/environment/scanning = -1 and position = -500. Within nonnegative trial IDs, position spans about -50 to 452.5 cm; small negative speeds occur from interpolation/noise.
- Trial IDs are contiguous in every session and all sessions have at least 41 trials. Importantly, samples assigned a trial ID can precede its `trial_start` pulse (pre-track/jitter), while `teleport` marks its end. Requested temporal alignment must therefore use the onset pulse, not merely the first sample with that trial ID.
- `reward_zone` contains time-varying integer codes 0–8 and can change within one trial; it is not directly the requested per-trial A/B/C category. This encoding will be resolved against reference code/text before mapping.
- Detailed per-session metadata were recorded in `/app/dataset_inventory.csv`; no large neural arrays were loaded for this inventory.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 312,110 ROIs raw; 138,678 Suite2p `iscell` ROIs |
| Neurons / session | raw 315–5,085 (mean 2,053.4); `iscell` 155–2,341 (mean 912.4) |
| Subjects | 11 (m3, m4, m7, m11–m15, m17–m19) |
| Sessions / subject | 14 for 10 subjects; 12 for m11 |
| Sessions total | 152 |
| Trials (total) | 12,217 source trial IDs |
| Trials / session | 41–100 (mean 80.38) |
| Imaging frames | 3,610,867 |
| Frame rate | 15.5078125 Hz (124 sessions); 62.03125 Hz (28 sessions) |
| Delivered reward events | 10,345 |
| Position observed in trials | -50 to 452.5 cm |
| Speed observed in trials | -6.447 to 144.5 cm/s |
| Lick count/sample | 0 to 8 (must be binarized for requested output) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote / interpretation |
|-----------|-------|------------------------------|
| Subjects | 11 switch mice | Methods explicitly reports “11 switch mice”. |
| Trials (paper analysis) | 12,376 imaged trials | Methods lick-curation denominator. |
| Corrupt lick trials | 81/12,376 (about 0.65%) | Trials with sensor damage were set to NaN for licking analyses. |
| Neural/behavior time bin | about 0.0645 s (about 15.5 Hz) | “All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.” |
| Spatial bin | 10 cm | Neural activity, licking, and speed were spatially binned at 10 cm for relevant analyses. |
| Track length | 450 cm | GLM position bases span 0–450 cm. |
| Running threshold | 2 cm/s | Place-cell and selected neural analyses excluded activity below 2 cm/s. |
| Lick representation | binary per imaging frame | Remaining cumulative lick counts were converted to a binary vector. |
| Reward-relative sessions passing k=2 criterion | 50/77 | Reported for trial-by-trial RR population-map analysis; not global dataset curation. |
| Released-data reward rate (derived) | about 76.3% of source trial IDs | Preliminary direct NWB reward-timestamp assignment; to be finalized after boundary edge cases. |
| Released-data corrupt lick trials (derived) | about 70/12,217 (0.57%) | Close to paper’s 0.65%; preliminary crop logic found boundary edge cases. |

### Processing Details
- Suite2p performed motion correction, ROI segmentation and cell classification. The reference pipeline uses neuropil-corrected fluorescence to compute dF/F and Suite2p deconvolution to obtain calcium events.
- Time-domain GLM analysis uses deconvolved calcium event time series at the imaging frame rate (~15.5 Hz), with behavior aligned on the same samples. Trial identity groups train/test data.
- The paper GLM predicts neural activity from task/movement variables, opposite to this task’s neural-to-behavior decoder. It therefore provides fraction deviance explained rather than directly comparable categorical decoder accuracies.
- Position and reward-relative position are represented by 45 bases (10 cm spacing for the linear track). Movement predictors include speed, acceleration and licking.
- Spatial place-cell/population analyses use 10 cm bins, sometimes smooth by a 10 cm s.d. Gaussian, and omit samples below 2 cm/s. These choices are not automatically applicable to this requested decoder because stationary periods and framewise outputs must be preserved.
- Lick sensor corruption criterion: more than 30% of 0.0645 s samples in a trial have cumulative lick count >2; affected trials are NaN only for licking analyses. Valid counts are binarized.

### Curation Steps

**Neuron curation rules**:
- Use Suite2p cell classification (`iscell`) as the primary ROI quality filter.
- Paper analyses may additionally exclude putative interneurons/activity-speed-correlated cells (>0.5) and select place-cell/RR subpopulations, but those are analysis-specific and not general imaging-quality criteria.

**Trial curation rules**:
- Trial-set analyses split before/after reward switch; omission trials are retained and explicitly modeled.
- Lick-corrupt trials are excluded/set NaN for licking metrics only, not necessarily discarded from all neural/behavior analyses.
- Teleport samples are excluded from valid track activity.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Requested six outputs | Not reported in paper |
| Paper GLM (behavior/task predicts neural events) | Fraction deviance explained, not categorical accuracy; direction and metric are not directly comparable |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial count | Trials are defined by session start/teleport indices | 12,217 nonnegative IDs but 12,216 start pulses; one edge ID is incomplete | 12,376 imaged trials in original analysis | Use the 12,216 complete released trials. The 160-trial difference reflects released-file/session scope plus one incomplete edge ID, not silent conversion loss. |
| Trial boundary | Reference dF/F excludes teleport samples and uses explicit start indices | Trial-ID assignment crosses teleport boundaries: 2,562 IDs have no teleport and 2,562 have two | Teleport is not valid track activity | Ignore trial-ID boundaries for slicing. Match each global start to the first subsequent teleport before the next start; include start and exclude teleport. All 12,216 starts match. |
| Sampling rate | Time-domain GLM uses about 15.5 Hz | Acquisition attribute is 15.5078 Hz for single-plane and 62.03125 Hz for two-plane sessions, but behavior timestamps in both have 0.0644836 s spacing and each ophys plane has one row per behavior sample | About 15.5 Hz / 0.0645 s | The 62.03 Hz value is aggregate scanner rate. Data are already synchronized at 15.5078 samples/s per plane; concatenate planes, do not downsample. |
| Ophys planes/ROI count | Multi-plane cells are combined and retain plane labels | Separate `plane0`/`plane1` matrices share time rows; segmentation table concatenates both | Multi-plane imaging used | Concatenate deconvolved matrices along neurons in plane order, matching `planeIdx` and ROI-region rows. |
| Neural quality | Suite2p cell classification is loaded; analysis-specific filters may follow | 312,110 ROIs, 138,678 with `iscell` flag | Suite2p segmentation/classification | Apply `iscell[:,0] > 0` as imaging-quality curation. Do not impose place-cell, reward-relative, speed, or interneuron analysis selections on a general decoder. |
| Neural representation | Population and GLM analyses use deconvolved events; dF/F is also computed | NWB directly stores `Deconvolved`, `Fluorescence`, and `Neuropil` | GLM uses deconvolved event time series | Use released `Deconvolved`; no recomputation or spatial binning. |
| Reward-zone code | `get_reward_zones` derives A/B/C and coordinates from scene, switching at trial 30 | Framewise `reward_zone` values 0–8 are brief event states near reward, not identity | Zones are 50 cm fixed intervals | Parse one/two A/B/C labels from NWB identifier and switch after first 30 complete trials, matching reference code. Use fixed coordinates from `reward_zone_dict`. |
| Environment | `Env1` maps to 0 and `Env2` to 1 | Framewise environment is 0/1 and can switch within cross-environment sessions | Two distinct environments | Use framewise stream sampled at trial start/as constant within each trial; it is authoritative even for identifier naming-order anomalies. |
| Lick corruption | Set affected trial licking to NaN; criterion >30% frames with cumulative count >2 | About 70 released trials meet criterion in preliminary complete-trial scan | 81/12,376 (0.65%) | Since categorical target cannot represent NaN, exclude corrupt trials entirely; document exact final count. Convert remaining lick counts >0 to binary. |
| Paper decoder comparison | Paper predicts neural events from behavior | Requested task predicts six variables from neural events | Paper reports FDE, not requested accuracies | No direct accuracy target exists; validate against chance, alignment, distributions and paper processing instead. |

### Final Understanding
The released NWBs are already temporally aligned, reference-processed data. Conversion should operate at the native per-plane imaging sample grid (median 64.4836 ms), concatenate planes, retain Suite2p-classified cells, and preserve full time-domain deconvolved events. Complete track trials are aligned to the explicit start pulse and stop immediately before teleport. Spatial speed thresholds and position binning are downstream paper analyses and are inappropriate for this framewise decoder because they would remove requested low-speed classes and temporal detail.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/<plane>/data` + segmentation `iscell` | `neural` | Concatenate planes as ROI x time, then retain `iscell[:,0] > 0`; slice start-inclusive/teleport-exclusive; float32 | `multi_anim_sess`, `dff`, `get_timeseries_data` | Released deconvolution is already aligned; no spatial binning/smoothing or speed mask. |
| behavior timestamps | `input[0]` time from trial start | `timestamp - timestamp[start]`, seconds, repeated samplewise | VR-to-2P alignment | Continuous time-varying. |
| `environment/data` | `input[1]` environment type | 0=ENV1, 1=ENV2; use value at start and repeat | `get_trial_types`; `env_morph_dict` | Binary per trial. |
| source trial order/ID | `input[2]` trial number | Original nonnegative trial ID, repeated across samples | session trial identity | Continuous per trial; preserve zero-based source value. |
| sparse `Reward/timestamps` on previous source trial | `input[3]` previous trial outcome | 0 for first trial/omitted previous trial, 1 if previous complete source trial contains reward | reward helpers | Based on chronological source trial even if that previous trial is excluded for bad licking. |
| position + current zone interval | `output[0]` distance to reward zone | Signed distance to nearest point in interval: position-start below, 0 inside, position-stop above; discretize 7 requested classes | `get_reward_zones` | Class 3 exactly covers all positions inside the 50 cm zone (“distance to any location in zone” = 0). |
| `position/data` | `output[1]` absolute position | `np.digitize(position,[90,180,270,360])`, classes 0–4 | time-domain position | Track samples from start to pre-teleport only. |
| `speed/data` | `output[2]` speed | Thresholds [2,10,20,40] into classes 0–4; small negative values remain class 0 | time-domain speed | Preserve stationary samples; no 2 cm/s exclusion. |
| `lick/data` | `output[3]` lick | `(lick > 0)` after paper corruption filtering | lick methods | Binary framewise target. |
| NWB identifier scene + chronological trial index | `output[4]` reward zone location | Parse A/B/C; for switch sessions use first label for trials 0–29 and second from trial 30; map A=0,B=1,C=2; repeat | `get_reward_zones(change_trial=30)` | Fixed coordinates X/A=80–130 cm, Y/B=200–250 cm, Z/C=320–370 cm from code. |
| sparse `Reward/timestamps` | `output[5]` reward outcome | Any delivered reward timestamp in [start, teleport); repeat 0/1 | reward helpers | Per-trial categorical outcome; autoreward remains an available but unused source variable. |
| subject ID | `subjects`, `subject_idx` | Unique mouse IDs in natural order; per-session index | NWB subject table | 11 subjects. |
| all curated ROIs | `brain_regions`, `brain_region_idx` | Region `dorsal CA1`; all neuron indices 0 | paper methods | Record plane indices additionally in metadata session info. |

### Output Discretization Details
- Distance classes: 0 `<-50`; 1 `[-50,-10)`; 2 `[-10,0)`; 3 `==0`; 4 `(0,10]`; 5 `(10,50]`; 6 `>50` cm. Boundary handling follows the wording exactly.
- Position classes use edges 90, 180, 270 and 360 cm; values exactly on an edge enter the higher conventional half-open bin.
- Speed uses conventional edges 2, 10, 20 and 40 cm/s (`np.digitize`); exact-edge occurrences are negligible and enter the higher bin.
- All six outputs are stored as int64 rows over time. Per-trial outputs are repeated to produce a uniform `(6, n_timepoints)` trial array.
- All four inputs are float32 rows over time, including repeated per-trial context.

### Key Decisions
1. **Native temporal grid**: Use explicit timestamps and the already common median 64.4836 ms sample spacing; scanner-rate attributes do not define behavior/neural row rate in multi-plane sessions.
2. **Complete trials only**: Pair every start with first teleport before next start; exclude the single incomplete released trial ID and exclude teleport itself, matching reference processing.
3. **Lick-invalid trial exclusion**: Exclude a complete trial if >30% of its samples have lick count >2. A NaN categorical lick target is impossible, and retaining it would train on known sensor artifact.
4. **Quality-filtered cells, not analysis-selected cells**: Keep Suite2p `iscell` only. Place-cell, reward-relative, speed-correlation and k-means criteria answer paper-specific questions and would bias a general decoder.
5. **No speed threshold**: Required speed class 0 and lick behavior occur during stopping; applying the paper’s spatial-analysis threshold would contradict decoder outputs.
6. **Zone identity from scene**: Framewise `reward_zone` is an event-state code, whereas scene schedule and trial-30 switch exactly match reference `get_reward_zones`.
7. **Reward from sparse timestamps**: Delivered reward events are direct evidence of outcome; do not infer outcome from autoreward or zone entry.
8. **Variable trial lengths retained**: Trials naturally last different durations. Time bins are equal-sized, while trial matrices may differ in number of columns as permitted by the format.

### Planned Sanity Checks
- [ ] For at least three sessions (single-plane and multi-plane), load raw NWB independently and verify converted neural slices against concatenated raw deconvolved/`iscell` data using `np.allclose`.
- [ ] Verify converted time, environment, trial number and previous outcome against independently read timestamps/behavior/reward events using `np.allclose`.
- [ ] Verify all six converted outputs for three named trials against independent raw calculations using `np.allclose`.
- [ ] Assert every selected start precedes its matched teleport and no trial crosses the next start.
- [ ] Assert all plane matrices have equal row counts and concatenated columns match segmentation rows before filtering.
- [ ] Assert timestamp spacing is consistent near 64.4836 ms across all sessions.
- [ ] Compare final subjects/sessions/trials/cells, reward fraction and lick-corruption fraction to raw inventory and paper values.
- [ ] Check all output values are within declared categories and every session retains at least two trials.
- [ ] Plot raw continuous position/speed/lick, class transitions, zone distance and neural activity for sample sessions to inspect temporal alignment.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements `--full` (default), `--sample` (first two naturally sorted sessions), and `--show-processing`. It reads one NWB at a time, concatenates synchronized ophys planes, filters Suite2p `iscell`, constructs start-aligned trials, validates all dimensions/ranges, emits timing, and saves processing plots for up to two sessions.

A pre-sample one-session test produced 155 curated cells x 80 trials with neural/input/output dtypes float32/float32/int64 and all expected classes for a B→A session. Review caught and fixed an initial A-zone coordinate transcription error: reference A maps to X=80–130 cm (not dictionary key A, which belongs to another task convention); B and C are 200–250 and 320–370 cm.

Code inefficiencies identified:
Large neural matrices dominate I/O and pickle size. Recomputing dF/F/deconvolution or reading unused fluorescence would be costly.

Code speedups added:
Read only released Deconvolved arrays; vectorize class construction; process sessions serially to bound memory; use float32; concatenate planes once per session; slice trials from the in-memory session array. The metadata-only all-session scan took about 7 s; the one-session full neural test completed within the 10 s command window.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (m3 sessions 01–02) |
| Neurons (total across sessions) | 1,969 curated cells |
| Neurons / session | 1,052; 917 |
| Subjects | 1 (m3) |
| Trials (total) | 160 |
| Trials / session | 80; 80 |
| Trial samples | min 162, overall mean about 315, max 1,546 |
| Time from start | 0 to 99.6 s |
| Environment | ENV1 only in this sample |
| Trial number | 0–79 |
| Previous outcome | 0/1 |
| Distance class fractions | [0.222, 0.055, 0.169, 0.394, 0.037, 0.066, 0.057] |
| Position class fractions | [0.075, 0.066, 0.081, 0.601, 0.177] |
| Speed class fractions | [0.433, 0.089, 0.081, 0.143, 0.254] |
| Lick fractions | [0.767, 0.233] |
| Reward-zone class fractions | [0, 0, 1.000] (C-only sample) |
| Reward outcome fractions | [0.113, 0.887] |

### Processing Plots Review
Two 1960x1680 PNG plots were created (`processing_sub-m3_ses-01.png`, `processing_sub-m3_ses-02.png`). They show start-aligned neural activity, monotonic time-from-start, position/distance and speed/lick class traces, and constant per-trial environment/zone context. No shape/alignment anomaly was detected. The long maximum trial is retained because it is bounded by valid start/teleport events and timestamps remain uniformly spaced.

### Format Validation
`verification_sample_out.txt` reports “Data verification complete” with consistent neural/input/output dimensions, declared output values and brain-region indices. No errors, NaNs or infinities were reported. Zone/environment constancy is sample-selection-specific, not a conversion warning.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Read Deconvolved only, vectorized transforms, float32, one session at a time | Avoids dF/F/deconvolution and controls memory |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Sample conversion including plots/pickle | about 1.15 s/session overall |
| Full conversion | conservatively 3–8 minutes including larger multi-plane matrices and pickle I/O; below 15 minutes |

Sample output size is 0.203 GB. Scaling by all-session curated-neuron/frame burden suggests a multi-GB full pickle, so disk capacity will be checked before the full run.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None.
- Warnings: sklearn reported predicted classes absent from `y_true` for reward-zone location because both sample sessions contain only zone C while the declared/full task has A/B/C. This cannot be fixed without violating the prescribed first-two-session sample; full data contain all classes.

### Training Progress
Loss decreased from 291.6739 (epoch 1) to 4.6342 (epoch 200); test loss was 8.9680. Training completed successfully on GPU.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Distance to reward zone | 0.7521 | 0.5679 | 0.1429 |
| Absolute position | 0.8142 | 0.6846 | 0.2000 |
| Speed | 0.6413 | 0.5468 | 0.2000 |
| Lick | 0.7637 | 0.7113 | 0.5000 |
| Reward zone location | 0.9955 | 0.9962 | 0.3333 (C-only true sample) |
| Reward outcome | 0.6854 | 0.5330 | 0.5000 |

All validation scores exceed nominal chance. Strong position/distance/speed performance supports correct temporal alignment. Reward outcome is only modestly above chance on two sessions and will be reassessed on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Iteration 1 edge case
The first full run stopped at session 114 (`m17` session 04): both neural planes had 22,791 rows versus 22,790 behavior samples. An all-file audit found exactly 10 two-plane sessions with the same +1 neural-row pattern and no other mismatch. In each case the extra row is after the final behavior sample; valid trials end before teleport. The script now permits only a 0 or +1 neural difference, trims that trailing row, records it, and rejects all other differences. A targeted failing-session test passed and the complete rerun succeeded.

### Output Files
- `converted_data.pkl`: 9.0 GiB (9.661 GB), created in 54.8 s
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; “Data format is valid, no errors or warnings” and “Data verification complete”

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total curated neurons | Suite2p-classified cells | `iscell` curation | 138,678 `iscell`; 312,110 raw ROIs | 138,678 | Exact |
| Mean neurons/session | not global | all curated cells | 912.36 (155–2,341) | 912.36 (155–2,341) | Exact |
| Subjects | 11 switch mice | configured animal lists | 11 | 11 | Exact |
| Sessions | analysis subsets vary | all released sessions | 152 | 152 | Exact |
| Trials | 12,376 paper-analysis trials | explicit start/teleport | 12,216 complete released | 12,135 kept + 81 corrupt-lick excluded | Fully accounted |
| Trials/session | varies | explicit events | 41–100 complete | 40–100 kept, mean 79.84 | Expected after curation |
| Corrupt lick trials | 81 (0.65%) | >30% samples with count >2 | 81 using complete boundaries | 81 excluded | Exact count |
| Time bin | about 64.5 ms | imaging-frame aligned | median 64.4836 ms | 64.4836 ms | Exact |
| Environment range | ENV1/ENV2 | 0/1 | 0/1 | 0/1 | Exact |
| Zone distribution | A/B/C schedules | scene and switch trial 30 | all A/B/C | [0.33159, 0.33588, 0.33253] frame-weighted | Plausible/balanced |
| Reward outcome | rewarded/omitted | sparse reward events | both | [0.15771, 0.84229] frame-weighted | Plausible |
| Lick output | binary | binarize valid cumulative counts | 0/positive | [0.77742, 0.22258] | Exact transform |
| Distance classes | requested 7 | zone coordinates X/Y/Z | all classes | [0.25127,0.10196,0.07326,0.23846,0.02071,0.07178,0.24255] | Valid |
| Position classes | requested 5 | frame position | all classes | [0.21201,0.17674,0.23121,0.22641,0.15363] | Valid |
| Speed classes | requested 5 | frame speed | all classes | [0.11658,0.08735,0.13412,0.31919,0.34276] | Valid |

Spot checks of sessions 0, 76 and 151 confirmed neural/input/output time dimensions match. All 152 sessions retain at least 40 trials. No data are unaccounted for: 12,217 nonnegative source IDs = 12,216 complete starts + 1 incomplete edge ID; 12,216 = 12,135 retained + 81 known lick-corrupt trials.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` begins “Data format is valid, no errors or warnings” and ends “Data verification complete.” No NaN/Inf, dimension, dtype, range or metadata warnings require action.
2. **Independent raw-data sanity checks**: `/app/sanity_checks.py` loads original NWBs directly without importing conversion code. For m3 session 01 trial 5, m17 session 04 trial 5 (two-plane and +1 trailing-frame case), and m19 session 14 trial 5, it independently reconstructs neural, all inputs and all outputs. `np.allclose` passed for every array. Tested neural shapes were (1052,382), (506,206), and (589,215).
3. **Reference code comparison**:
   - Loading: converter reads released NWB Deconvolved streams; reference `multi_anim_sess` creates that stream through `dff(..., deconvolve=True)`. Same signal, avoiding redundant recomputation.
   - Neuron filtering: converter uses Suite2p `iscell`; reference loads Suite2p curation. It intentionally does not apply place-cell/RR/interneuron selections because those are analysis-specific rather than recording-quality filters.
   - Trial filtering: converter uses explicit start/teleport and excludes teleport, matching reference dF/F masking. It additionally excludes exactly the lick-corrupt trials identified by the paper criterion because categorical lick cannot represent NaN.
   - Temporal alignment: behavior and ophys rows are already VR-to-2P aligned; converter retains the shared ~15.5 Hz frame grid and subtracts start timestamp.
   - Binning: no temporal or spatial rebinning is applied. This matches paper time-domain GLM and preserves requested framewise low-speed behavior. Requested categorical thresholds alone are applied.
   - Inputs: direct frame timestamps/environment/trial identity and sparse previous reward outcome; no GLM zero-centering, because requested semantics are binary/continuous decoder inputs.
   - Outputs: direct frame position/speed/lick, reference scene-derived A/B/C schedule, and sparse delivered rewards. Zone coordinates/switch trial exactly follow `get_reward_zones`.
4. **Key-statistics comparison**: 11 subjects, 152 sessions, 138,678 curated cells and 12,216 complete starts exactly match independent raw scans. The final 12,135 trials plus 81 exclusions account for every complete trial; 81 is exactly the paper’s corrupt-lick count. All sessions retain >=40 trials and all output classes occur.
5. **Edge/off-by-one checks**: One incomplete source trial ID without start is excluded. Every one of 12,216 starts has exactly one teleport before the next start. Start is included and teleport excluded. Ten two-plane sessions have one neural-only trailing frame; it lies after behavior and is trimmed under a strict +1-only assertion. Trial-29/30 zone transitions use source chronological index, unaffected by lick exclusions. Previous outcome also uses the immediately preceding source trial, not previous retained trial.

### Issues Found and Resolved
- **A-zone transcription**: Initial draft used 50–100 cm; inspection showed reference labels A/B/C map to X/Y/Z, so A is 80–130 cm. Fixed before sample/full outputs.
- **Neural/behavior trailing edge**: Ten sessions had one extra neural row. Added strict, documented trailing trim; all raw comparisons pass.
- **Trial-ID boundary ambiguity**: Teleport markers straddle source trial IDs. Replaced ID-based slicing with global start-to-next-teleport matching, resolving empty/reversed trial crops.
- **Paper/release trial-count discrepancy**: Fully reconciled as 12,376 paper-analysis denominator versus 12,216 complete trials in released files. No released complete trial is unaccounted for.

No further issues were found after rerunning full conversion, verification, aggregate checks and independent raw comparisons.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, from 188.8574 (epoch 1) to 1.2541 (epoch 200); test loss 1.1466.
- Device: CUDA. Four sessions with >2,000 neurons used the decoder’s random projection to 2,000 dimensions only for SVD initialization; source data remained intact.
- Full execution completed successfully, including sample plots.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Distance to reward zone | 0.3996 | 0.3538 | chance 0.1429 |
| Absolute position | 0.5321 | 0.5037 | chance 0.2000 |
| Speed | 0.4313 | 0.4044 | chance 0.2000 |
| Lick | 0.6076 | 0.5957 | chance 0.5000 |
| Reward zone location | 0.8429 | 0.8153 | chance 0.3333 |
| Reward outcome | 0.5538 | 0.5044 | chance 0.5000 |

All validation accuracies are above nominal chance. Position-related variables and zone context are strongly decodable. Reward outcome is marginal and is investigated in Step 12.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Accuracy / chance | Expectation from Paper |
|----------|---------------------|--------|-------------------|------------------------|
| Distance to reward zone | 0.3538 | 0.1429 | 2.477x | Paper RR-position decoder is related but not directly comparable |
| Absolute position | 0.5037 | 0.2000 | 2.518x | No matching categorical accuracy reported |
| Speed | 0.4044 | 0.2000 | 2.022x | No matching decoder reported |
| Lick | 0.5957 | 0.5000 | 1.191x | No matching decoder reported |
| Reward-zone location | 0.8153 | 0.3333 | 2.446x | No matching categorical accuracy reported |
| Reward outcome | 0.5044 | 0.5000 | 1.009x | No matching decoder reported |

**Accuracy versus chance**: Every output is above chance. Distance, position, speed and zone exceed 2x chance. Lick and reward were investigated because they are below 1.5x chance.

**Paper comparison**: The paper’s “Decoding of RR position” uses circular-linear regression on selected RR/TR/non-RR remapping neurons, restricts to speeds >2 cm/s, occupancy-matches 45 circular bins, and reports mean cosine decode score and shuffle z-scores. The requested decoder instead uses all curated cells, includes stationary samples, and predicts seven distance categories. The paper reports no categorical balanced accuracies for the six requested variables, so there is no like-for-like numeric table. The strong requested position/distance performance is qualitatively consistent with the paper’s successful RR-position decoding.

**Train/validation gap**: Train/validation ratios are distance 1.129, position 1.056, speed 1.067, lick 1.020, zone 1.034 and reward 1.098. All are far below the 1.5x concern threshold; no evidence of severe overfitting or leakage.

**Low-accuracy debugging**:
1. `/app/critical_accuracy_checks.py` independently loaded three specific raw trials: m3 session 01 trial 6 (omitted), m3 session 01 trial 0 (reward timestamp 28.3728 s), and m13 session 09 trial 0 (omitted). Raw sparse reward timestamps exactly matched converted outcomes; raw positive lick counts exactly matched converted binary licks using `np.allclose`.
2. `/app/critical_alignment_check.png` overlays converted neural activity and outputs on the same start-aligned frame grid. No temporal shift is visible; this complements the exact array checks in Step 10.
3. Output variation is adequate: trial-weighted outcomes are 15.36% omitted / 84.64% rewarded over 12,135 trials; framewise licks are 77.74% no-lick / 22.26% lick over 2,576,026 samples. Neither is a 99% class.
4. Neural filtering follows Suite2p `iscell`, and deconvolved events match raw NWB arrays exactly. No incorrect place-cell/speed filtering was applied.
5. Processing matches the paper’s time-domain frame alignment and lick binarization. The requested reward output is repeated over the full trial. Much of each trial occurs before reward delivery, when neural activity cannot predict the later stochastic omission; therefore full-trial reward balanced accuracy near chance is scientifically expected. Changing labels only after delivery would violate the requested per-trial output specification.

### Issues Found and Resolved
- No conversion bug was found in Step 12. Raw labels, temporal alignment, class variation, filtering and processing all passed independent checks.
- Low lick/reward scores are retained rather than artificially improved by leakage, post-outcome cropping, speed filtering, or trial-level label changes.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with loading instructions, variable definitions, processing rationale, statistics and decoder results.
- [x] `cache/` created; investigation scripts and metadata inventory moved there.
- [x] `cache/README_CACHE.md` documents cached files.
- [x] All required conversion, verification and training logs remain at `/app`.
- [x] Required `converted_data.pkl`, `sample_data.pkl`, `convert_data.py`, and documentation exist and are nonempty.
- [x] Final full decoder log confirms successful completion.

### Final Deliverables
- `/app/converted_data.pkl`: complete 152-session decoder dataset (9.0 GiB).
- `/app/convert_data.py`: reproducible converter with full/sample/plot modes.
- `/app/CONVERSION_NOTES.md`: ordered workflow, decisions, discrepancies, checks and results.
- `/app/README.md`: user-facing usage guide.
- `/app/train_decoder_full_out.txt`: completed 200-epoch full decoder run.
- `/app/verification_full_out.txt`: error/warning-free full format validation.
- Processing and critical-alignment PNGs provide visual checks; required sample/full text logs are retained.


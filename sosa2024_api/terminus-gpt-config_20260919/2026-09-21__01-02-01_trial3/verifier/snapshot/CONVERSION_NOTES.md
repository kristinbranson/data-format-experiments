# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

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
- `pynwb_docs/`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_sess_pickle` | `src/reward_relative/utilities.py` | LOADING | Loads an author-preprocessed session pickle using the session path resolved from animal/day metadata. |
| `get_sess_pkl_path` | `src/reward_relative/utilities.py` | LOADING | Resolves animal, date, scene, and session to a preprocessed pickle; if a day contains multiple sessions, the original utility explicitly selects the first. |
| `dayData` methods | `src/reward_relative/dayData.py` | PROCESSING | Aggregate sessions across mice/days, define trial subsets, construct spatial/circular trial matrices, and classify track- and reward-relative cells. |
| `TwoPUtils.spatial_analyses.trial_matrix` | external dependency called by `dayData.py` | PROCESSING | Bins neural events or behavior by position between `trial_start_inds` and `teleport_inds`. |
| `behav.lickrate_PETH` | behavior helper called by `dayData.py` | PROCESSING | Computes trial-aligned/binned licking around the circularized track. |

### Notes
- The repository is analysis code for two-photon CA1 imaging. It primarily consumes already-preprocessed session pickle objects rather than reading raw imaging files in this repository.
- Session objects expose `timeseries`, `vr_data`, `trial_start_inds`, and `teleport_inds`. Conversion-relevant behavior fields include position (`pos`), speed, and lick; trial start and teleport delimit each traversal.
- The main neural activity key used for place-cell detection is `events` (deconvolved calcium-event activity), not raw fluorescence. Thus no new delta-F/F calculation is indicated for this conversion when the NWB already provides processed event/deconvolved traces.
- Reference place-cell analyses use position-binned trial matrices, typically apply a running-speed threshold of 2 cm/s, split trials into subsets around reward-location switches, and use shuffle-based classification (commonly 100 permutations and maximin baseline settings). These are downstream place/reward-cell classification steps, not general quality filters to apply to decoder time series unless the source NWB marks invalid units explicitly.
- The decoder task requires temporal trial alignment and all time points, including low-speed periods needed for the specified speed classes; therefore the analysis-only 2 cm/s running filter should not be applied to converted decoder trials.
- Original analysis utilities sometimes select the first session on a multi-session day. For conversion, each NWB file/session will be treated according to its native session identity rather than silently dropping sessions, unless data exploration shows files are duplicate representations.
- The conversion itself must replace author pickle loading with `pynwb.NWBHDF5IO`; no `h5py` loading will be used.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 152 `*_behavior+ophys.nwb` files, one file per recording session, organized in subject directories.
- All files were opened and inspected with `pynwb.NWBHDF5IO(..., mode="r", load_namespaces=True)`; no `h5py` was used.
- Neural data are under `processing/ophys`. `Deconvolved/plane0` is a time-by-ROI float32 `RoiResponseSeries`; `Fluorescence` and `Neuropil` are also available. Sampling rates vary by acquisition configuration (approximately 15.5 or 30 Hz; exact per-session rate will be honored).
- ROI metadata are in `ImageSegmentation/PlaneSegmentation`, with `iscell` (Suite2p cell flag and probability), `planeIdx`, and pixel or voxel masks. Neural conversion will retain rows with `iscell[:,0] > 0`.
- Behavior is under `processing/behavior/BehavioralTimeSeries`. Frame-sampled streams include explicit `trial_start`, `teleport`, `trial number`, position, speed, lick, environment, reward-zone/distance variables, and omission information. Each stream has timestamps; behavior and imaging have differing sample counts/rates and therefore must be aligned by time, not index.
- `Reward` is a sparse event `TimeSeries` (0.004 mL per event) with its own timestamps. Reward outcome is determined by whether a Reward timestamp lies from trial start through teleport.
- NWB `trials`, `intervals`, and `units` tables are absent. Trials are reconstructed from each `trial_start` pulse to the following `teleport` pulse; all scanned starts had a valid following teleport before the next start.
- Native trials include inter-trial periods between teleport and the next trial start; these are excluded because temporal alignment is to explicit trial start.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated cell recordings across sessions (`iscell[:,0] > 0`); 312,110 segmented ROIs before curation |
| Neurons / session | 155-2341, mean 912.4, median 922 |
| Subjects | 11 |
| Sessions / subject | 3-15 depending on subject; 152 sessions total |
| Trials (total) | 12,216 |
| Trials / session | 41-100, mean 80.4, median 80; predominantly 80 or 100 |

### Available Variables and Data Types
- Neural: deconvolved event activity, fluorescence, neuropil fluorescence (`float32`, time x ROI).
- Behavioral continuous: absolute corridor position (cm), speed (cm/s), and distance relative to reward zone (cm).
- Behavioral binary/event: lick, trial start, teleport, omission, and sparse reward delivery events.
- Trial context: environment identifier, trial number, and reward-zone identity/location.
- Session/subject metadata: NWB subject ID, session ID, start time, and imaging-plane metadata.

### Exploration Checks
- All 152 NWB files opened successfully through pynwb.
- Trial starts and teleports were paired chronologically and checked for one valid end per retained start.
- Behavioral timestamp arrays within a session were checked for equality; neural and behavior streams use independent sampling grids.
- Both pixel-mask (124 sessions) and voxel-mask (28 sessions) PlaneSegmentation layouts carry the same `iscell` and plane metadata.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 11 in released NWB cohort | Data scan; paper analyses use subsets depending on longitudinal inclusion criteria |
| Sessions | 152 released NWB sessions | Data scan; paper reports analysis-specific subsets (for example 77 sessions with an RR population vector) |
| Neural data time bin | Imaging frames, approximately 15.5 Hz for the reference GLM | “All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.” |
| Track length | 450 cm | Methods design matrix: “linear track position (from 0 to 450 cm)” |
| Spatial analysis bin | 10 cm | Position expanded into 45 bases; trial-by-trial matrices use 10 cm bins |
| Running threshold | 2 cm/s for place-cell analyses | Reference code default `speed_thr=2` |
| Place-cell shuffles | 100, p=0.05 | Reference code defaults |
| Neural signal | Deconvolved calcium event time series | Methods GLM and remapping analyses |
| GLM FDE | all place cells 0.10±0.19; TR 0.32±0.13; RR 0.29±0.11; non-RR remapping 0.29±0.11 | Methods/model-results text |

### Processing Details
- The task is a 450 cm virtual linear corridor with reward-zone locations A, B, and C at fixed track positions; reward location changes across trial blocks.
- Reference analyses use Suite2p segmentation and deconvolved calcium events. Neural and behavior are synchronized by time/imaging frames.
- Place/reward-relative analyses spatially bin deconvolved activity, commonly into 10 cm bins. Trial-by-trial remapping matrices are smoothed with a 10 cm s.d. Gaussian and normalized per neuron.
- The GLM uses unsmoothed deconvolved event time series at imaging rate. Position, reward-relative position, rewarded state, speed, acceleration, and lick count are predictors. Licks are binary per frame before smoothing in that model.
- Rewarded and omission trials are explicitly analyzed separately. Paper train/test splits preserve trial groups and outcome categories.

### Curation Steps

**Neuron curation rules**:
Use Suite2p `iscell` classification. Place-cell significance and reward-relative classification are downstream analysis labels and must not restrict a general neural decoder.

**Trial curation rules**:
Use complete traversals from explicit trial start through teleport. Exclude inter-trial/sentinel periods. Preserve both rewarded and omission trials.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| No directly comparable categorical neural-to-behavior decoder | Paper reports encoding-model FDE rather than classification accuracy |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Loader | Preprocessed session pickles | Released data are NWB | Analyses use preprocessed sessions | Use pynwb as required, mapping equivalent processed fields. |
| Neural signal | `timeseries['events']` | `processing/ophys/Deconvolved/plane0` | Deconvolved calcium events | Use NWB Deconvolved series. |
| Cell curation | Preprocessed session cells | All ROIs plus two-column `iscell` | Suite2p processing | Retain `iscell[:,0] > 0`. |
| Alignment | trial start/teleport indices | Explicit timestamped pulse streams | Imaging-frame synchronized streams | Pair pulses and align all data by timestamps. |
| Speed filter | 2 cm/s for place-cell maps | Full speed including stopping | Paper filter is analysis-specific | Preserve all speeds because speed <2 cm/s is a required decoder class. |
| Spatial vs temporal binning | Spatial trial matrices for maps | Native temporal streams | Decoder requires time from trial start | Use 100 ms temporal bins; do not apply spatial smoothing. |
| `reward_zone` stream | Not used as identity in reference code | 0–6 transient values near reward delivery | Fixed A/B/C zones | Derive zone identity from task block/location; do not interpret this stream as A/B/C. |
| Session count | Some utilities select first session/day | One NWB per released session | Analyses use varying subsets | Retain every nonduplicate NWB session. |

Final understanding: released NWBs contain the author-processed deconvolved signal, Suite2p curation, and timestamped behavioral streams needed for direct temporal decoding. Analysis-specific place-cell filters are intentionally not applied.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `Deconvolved/plane0` + `iscell` | `neural` | retain `iscell[:,0]>0`; average event amplitudes in 100 ms bins; transpose to neuron×time | `dayData`, `trial_matrix` | Timestamp-aligned, no spatial smoothing |
| trial-relative bin centers | `input[0]` | seconds from explicit trial start | trial boundaries | continuous time-varying |
| `environment` | `input[1]` | map native 0/1 to ENV1/ENV2; constant per trial | session behavior | binary per trial, repeated over time |
| `trial number` | `input[2]` | native continuous trial number | session behavior | constant per trial, repeated |
| previous Reward event | `input[3]` | previous retained trial rewarded=1 else 0; first trial=0 | reward/omission trial split | repeated over time |
| position + active fixed zone | `output[0]` | signed distance to nearest point in zone; bins `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50` | reward-relative position analyses | zero throughout zone |
| `position` | `output[1]` | bins `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360` | linear track position | clip only tiny interpolation excursions for binning |
| `speed` | `output[2]` | `<2`, `2–10`, `10–20`, `20–40`, `>40` cm/s | movement variables | preserve low/negative estimates in class 0 |
| `lick` | `output[3]` | native count >0 | binary licks/frame in GLM | binary time-varying |
| active zone center/start | `output[4]` | A=0, B=1, C=2 | reward-location blocks | per-trial value repeated over time |
| sparse `Reward` timestamps | `output[5]` | any event in start-through-teleport window | rewarded/omission split | per-trial binary repeated over time |

### Key Decisions
1. **Temporal bins**: 100 ms for every session. This is fine enough relative to ~15.5/30 Hz imaging, ensures a common bin size, and avoids pretending unequal native frame durations are identical.
2. **Trial extent**: explicit `trial_start` timestamp through the following `teleport`, excluding inter-trial -500 cm sentinel periods.
3. **Neural curation**: Suite2p `iscell` only; no place-cell, speed, or reward-relative selection, because the decoder should receive the full curated CA1 population.
4. **Reward zones**: fixed 20 cm zones A=80–100, B=200–220, C=320–340 cm. Determine active zone from the reward block/location evident in the trial behavior; distance is to the closest point in that zone, making all in-zone samples exactly zero.
5. **Interpolation**: linear interpolation for position/speed; nearest prior/sample for discrete streams. Sparse rewards are evaluated directly by timestamps.
6. **Trial labels as matrices**: repeat per-trial inputs/outputs across time so every trial has `(n_variables, n_timepoints)` and validator semantics are unambiguous.
7. **Brain region**: all curated neurons are dorsal CA1/hippocampal CA1, indexed to one region label `CA1`.

### Planned Sanity Checks
- [ ] Compare raw and converted deconvolved values for specified neuron/time bins with `np.allclose`.
- [ ] Compare raw trial-relative time, environment, trial number, and previous outcome with converted input using `np.allclose`.
- [ ] Compare raw/interpolated position, speed, lick, zone, and reward-derived outputs with converted output using `np.allclose`.
- [ ] Verify every start pairs to one following teleport and all trial arrays share time length.
- [ ] Check output classes stay within declared ranges and all classes have plausible frequencies.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- `/app/convert_data.py` uses `pynwb.NWBHDF5IO` exclusively for NWB access.
- Implements `--full`, `--sample`, and `--show-processing` modes.
- Uses explicit timestamped trial starts/teleports, 100 ms bins, Suite2p `iscell` curation, deconvolved events, requested inputs/outputs, assertions, and per-session timing.
- Smoke test on two sessions completed successfully: 160 trials, 323 neuron-session recordings, 0.5 s total.

Code inefficiencies identified:
- Full deconvolved matrices are read once per session; repeated per-trial file reads are avoided. The small per-bin mean loop is bounded by trial duration.

Code speedups added:
- Vectorized timestamp searches/interpolation and one-time curated-column extraction. Estimated full conversion is under two minutes based on the smoke test.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total session-sum) | 323 |
| Neurons / session | 155, 168 |
| Subjects | 1 (m11) |
| Sessions / subject | 2 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| Time from trial start range | 0.05–30.55 s |
| Trial lengths | 88–306 100-ms bins |
| Environment | ENV1 only in these first two deterministic sample sessions |
| Distance-class distribution | [0.086, 0.093, 0.057, 0.208, 0.029, 0.090, 0.437] |
| Position-class distribution | [0.310, 0.207, 0.204, 0.146, 0.133] |
| Speed-class distribution | [0.122, 0.062, 0.066, 0.233, 0.517] |
| Lick distribution | [0.821, 0.179] |
| Reward-zone distribution | [0.792, 0.208, 0] by time point |
| Reward outcome distribution | [0.161, 0.839] by time point |

### Processing Plots Review
- `processing_m11_03.png` and `processing_m11_04.png` created.
- Trial-relative progression of neural activity, position, reward distance, speed, and licking is temporally coherent; no boundary or alignment anomalies were detected.
- Zone C/ENV2 absence is expected for the first two sorted sessions and is not a full-dataset deficiency.

### Format Validation
- `train_decoder.py --verify-only`: valid format, no errors or warnings.
- Neural/input/output time dimensions agree for every trial; all values are finite and categorical outputs are in declared ranges.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| One neural read/session; vectorized interpolation/search | Sample conversion only 0.5 s without plots, 1.6 s with plots |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion (without plots) | ~0.25 s for sample sessions | <2 minutes allowing for larger files and serialization |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Training Progress
- Loss decreased from 34.33 at epoch 9 to 0.884 at epoch 200; test loss 0.939.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Distance to reward zone | 0.4398 | 0.3460 | 0.1429 |
| Absolute position | 0.5488 | 0.4455 | 0.2000 |
| Speed | 0.3945 | 0.3326 | 0.2000 |
| Lick | 0.7500 | 0.6877 | 0.5000 |
| Reward zone location | 0.9653 | 0.9591 | 0.3333 |
| Reward outcome | 0.6467 | 0.6793 | 0.5000 |

All sample validation accuracies are above chance. Training and validation are close, with no concerning overfit gap.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Iteration 1 issue and fix
- The first full run stopped at the first multi-plane session (`sub-m17_ses-01`) after 68 single-plane sessions. Cause: the initial implementation selected only `Deconvolved/plane0` but applied the `iscell` vector for the full two-plane segmentation table (936 response columns versus 2,162 table rows).
- Inspection through pynwb showed plane0 references segmentation rows 0–935 and plane1 rows 936–2161. Fixed the converter to iterate every deconvolved RoiResponseSeries, use its `rois` DynamicTableRegion to subset `iscell`, bin each plane on its own timestamps, and concatenate curated cells.
- The failed run did not write `converted_data.pkl`; sample and full conversions will be rerun after regression tests.

### Iteration 2 issue and fix
- Final audit found one all-zero warning caused by behavior beyond imaging coverage. Added complete neural-coverage filtering.
- Correctly interpreted m17/m18 interleaved two-plane nominal 31.015625 Hz as 15.5078125 Hz per plane, exactly matching behavior duration and Methods. All 28 multi-plane sessions retain full trials.
- Excluded one genuinely uncovered terminal trial in m14_12; corrected full conversion has 12,215 trials and no validator warnings.

### Output Files
- `converted_data.pkl`: 5.89 GiB
- `conversion_full_out.txt`: created; 124.5 s corrected full run
- `verification_full_out.txt`: created; format valid with no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total curated neuron recordings | analysis subsets vary | Suite2p cells/events | 138,678 | 138,678 | Exact |
| Mean neurons/session | not a single reported cohort value | all curated session cells | 912.4 | 912.4 | Exact |
| Subjects | cohort/subsets vary by analysis | named animal sessions | 11 | 11 | Exact |
| Sessions | subsets vary (for example 77 RR-vector sessions) | available sessions | 152 | 152 | Exact |
| Trials (total) | not reported for complete release | trial start to teleport | 12,216 behavioral / 12,215 fully imaged | 12,215 | Exact |
| Trials/session (mean) | predominantly 80-trial sessions | native session trials | 80.37 | 80.37 | Exact |
| Environment range | two environments | native 0/1 | [0,1] | [0,1] | Exact |
| Trial number range | session-local | native 0–99 | [0,99] | [0,99] | Exact |
| Rewarded trials | rewarded and omissions retained | sparse Reward events | 10,342/12,216 behavioral; 10,341/12,215 retained | 10,341/12,215 | Exact |
| Zone trial counts | three fixed locations | blockwise zones | A=4,184; B=4,008; C=4,024 behavioral | A=4,183; B=4,008; C=4,024 retained | Exact/inferred from native location activation |
| Distance output fractions | task-specific conversion | N/A | derived from raw position/zone | [0.2518,0.1019,0.0739,0.1676,0.0269,0.0832,0.2946] | Consistent |
| Position output fractions | 0–450 cm track | linear position | derived from raw position | [0.2092,0.1784,0.2318,0.2273,0.1533] | Consistent |
| Speed output fractions | full speed retained | 2 cm/s map threshold only | derived from raw speed | [0.1224,0.0888,0.1343,0.3179,0.3366] | Consistent |
| Lick output fractions | binary lick/frame | binary before GLM smoothing | raw count >0 | [0.7689,0.2311] | Consistent |

All per-session trial counts and neuron counts match the independently generated pynwb scan exactly. Multi-plane sessions include all referenced planes.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` says “Data format is valid, no errors or warnings” and completes normally. No warning required a waiver.
2. **Independent neural sanity checks**: A separate script loaded raw NWBs directly with pynwb (without importing conversion code). For session/trial (0,4), multi-plane (68,5), and final-session (151,20), it independently applied each RoiResponseSeries DynamicTableRegion, `iscell`, timestamps, and 100 ms means. All neural arrays passed `np.allclose(rtol=1e-6, atol=1e-7)`.
3. **Independent input sanity checks**: The same raw loads reconstructed time, environment, trial number, and previous raw Reward outcome. All three trials passed `np.allclose`.
4. **Independent output sanity checks**: Raw timestamp-interpolated position/speed, nearest-sample lick, zone distance/classes, and sparse Reward outcome matched converted outputs exactly with `np.allclose(rtol=0, atol=0)`.
5. **Reference processing comparison**:
   - Loading: author code loads processed session pickles; conversion loads equivalent released NWB fields via pynwb.
   - Filtering: both use Suite2p `iscell`; conversion intentionally does not restrict to place/RR cells.
   - Alignment: both use trial start and teleport; conversion uses their timestamps to avoid cross-stream index assumptions.
   - Binning: paper spatial analyses use 10 cm bins, while required decoder alignment necessitates common 100 ms temporal bins; neural events are mean-binned without spatial smoothing.
   - Inputs: directly use raw time/context/trial/outcome fields required by this task.
   - Outputs: derive requested categorical behavior from raw position, speed, lick, fixed reward location, and Reward events. Binarizing lick matches paper GLM preprocessing before its analysis-specific smoothing.
6. **Key statistics**: Independent raw and converted per-session arrays match exactly: 11 subjects, 152 sessions, 12,215 fully imaged trials, and 138,678 curated neuron recordings. Retained rewarded count is 10,341; retained zone trial counts are A=4,183, B=4,008, C=4,024.
7. **Edge cases**: Every session has at least two trials; all arrays are finite; all 100 ms time vectors are strictly increasing; all categorical values are in range; per-trial variables are constant. Explicit starts pair one-to-one with following teleports.
8. **Longest trial review**: m4_04 trial index 39 is 216.66 s in the raw timestamps, remains `scanning=1`, has the expected explicit start/teleport, finite neural data, and traverses all position classes after a long pause near track start. It is valid and was retained rather than applying an unsupported duration filter.

### Issues Found and Resolved
- **Incomplete neural coverage warning**: Final audit exposed validator warning “session 76, trial 42: all neural data is zero.” Raw pynwb inspection showed behavior continued after imaging ended in m17_09; nearest-frame extrapolation was invalid. A full-coverage rule was added, requiring trial start and teleport to lie within every plane's neural time range.
- **Multi-plane effective-rate metadata**: An initial application of the coverage rule appeared to remove half of all m17/m18 trials. Investigation showed their NWB series store the ~31.0 Hz interleaved aggregate scan rate on each plane. Frame count and behavior duration—and the Methods statement that m17/m18 were sampled at ~15.5 Hz per plane—prove the effective rate is nominal rate divided by two planes. Corrected timestamps retain every m17/m18 trial and pass independent raw-neural `np.allclose` checks.
- **True incomplete trial**: Only the terminal trial of m14_12 lacks complete imaging coverage; it was excluded. Final data retain 12,215 of 12,216 behavioral trials. Validator warnings are now absent and no retained neural trial is all-zero.
- **Multi-plane ROI mismatch**: First full attempt selected only plane0 while applying the full segmentation mask. Fixed by using each RoiResponseSeries `rois` DynamicTableRegion, binning every plane, and concatenating curated neurons. Sample regression, focused multi-plane test, corrected full conversion, full validation, and all critical checks passed afterward.
- **Sparse Reward series edge case**: Reward is event-based and shorter than frame-sampled behavior. It is aligned directly by timestamps, never indexed as a behavioral frame array.
- **Transient `reward_zone` semantics**: It is not an A/B/C label. Active fixed-zone identity is inferred from its position-localized activation and nearest trial in contiguous blocks for omissions; global A/B/C counts are balanced and all three classes are present.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Corrected final run used 9,771 training and 2,444 held-out trials on CUDA and finished successfully.
- Loss decreased from 170.5288 at epoch 1 to 1.1863 at epoch 200; test loss was 1.0621.
- Four sessions with more than 2,000 neurons used the validator's random projection to 2,000 dimensions for SVD initialization.

### Decoder Results (Full, Final Corrected Dataset)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Validation / Chance |
|--------|-----------------------|-------------------------|--------|---------------------|
| Distance to reward zone | 0.4577 | 0.4123 | 0.1429 | 2.89× |
| Absolute position | 0.5653 | 0.5307 | 0.2000 | 2.65× |
| Speed | 0.4971 | 0.4608 | 0.2000 | 2.30× |
| Lick | 0.6586 | 0.6466 | 0.5000 | 1.29× |
| Reward zone location | 0.8551 | 0.8110 | 0.3333 | 2.43× |
| Reward outcome | 0.6167 | 0.5162 | 0.5000 | 1.03× |

Sample plots were requested with `--plot-samples`; the script finished successfully.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Expectation from Paper |
|----------|---------------------|--------|-----------------|------------------------|
| Distance to reward zone | 0.4123 | 0.1429 | 2.89× | Reward-relative neural coding expected |
| Absolute position | 0.5307 | 0.2000 | 2.65× | Strong track-relative place coding expected |
| Speed | 0.4608 | 0.2000 | 2.30× | Movement modulation expected |
| Lick | 0.6466 | 0.5000 | 1.29× | Lick predictor used in paper GLM |
| Reward zone location | 0.8110 | 0.3333 | 2.43× | Strong context/reward-location coding expected |
| Reward outcome | 0.5162 | 0.5000 | 1.03× | Paper reports encoding FDE, not comparable outcome-decoding accuracy |

### Checks and Interpretation
1. **Accuracy versus chance**: Every final validation output is above chance. Distance, position, speed, and reward-zone location exceed 1.5× chance. Lick and outcome were investigated directly.
2. **Accuracy comparison to paper**: The paper reports Poisson encoding-model fraction deviance explained rather than categorical balanced accuracy: all place cells 0.10±0.19, TR 0.32±0.13, RR 0.29±0.11, and non-RR remapping 0.29±0.11. There is no like-for-like paper decoder accuracy. Strong position and reward-location decoding is qualitatively consistent with place and reward-relative coding.
3. **Train-validation gap**: Final train/validation ratios are 1.11, 1.07, 1.08, 1.02, 1.05, and 1.19. None exceeds 1.5; there is no concerning overfit gap.
4. **Outcome raw-value verification**: Specific rewarded, omission, and multi-plane trials were independently loaded through pynwb. Sparse Reward timestamps within start-to-teleport windows exactly matched converted labels.
5. **Outcome variation**: Retained data contain 1,874 omissions and 10,341 rewarded trials, so both classes have substantial support despite imbalance.
6. **Temporal alignment**: Outcome is explicitly per trial and repeated from trial start. Reward timestamps, complete imaging coverage, and exact boundaries were independently checked.
7. **Neural processing/filtering**: Deconvolved events and Suite2p `iscell` match the reference. Multi-plane effective timing follows the Methods' ~15.5 Hz per plane. Applying target-selective cell filtering would introduce leakage.
8. **Lick review**: Native lick count is binarized, matching the paper's binary licks per frame before GLM-specific smoothing. The positive class occupies 23.1% of retained time points.
9. **Post-fix revalidation**: After resolving incomplete neural coverage and interleaved multi-plane timing, full conversion, verify-only validation, independent `np.allclose` checks, edge checks, and all 200 decoder epochs were rerun. No validator warnings remain and no retained trial has all-zero neural data.

### Issues Found and Resolved
- The final audit initially exposed an all-zero neural warning. Raw inspection identified behavior after imaging ended; complete neural-coverage filtering removed one genuinely invalid terminal trial.
- The m17/m18 NWB nominal rate is the ~31 Hz interleaved aggregate rate. Methods, frame counts, and behavior duration establish ~15.5 Hz effective sampling per plane. Correcting this preserved all valid multi-plane trials and substantially improved final neural decoding.
- Lower reward-outcome performance is not due to incorrect labels, missing variation, alignment, curation, or incomplete neural data. It remains slightly above balanced chance after all checks.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with format, processing, usage, statistics, and decoder results
- [x] cache/ folder created
- [x] `cache/README_CACHE.md` documents investigation artifacts
- [x] All required output files retained and analysis artifacts organized

Final audit:
- Complete and sample pickle files pass the provided validator.
- Full decoder training finished successfully.
- Conversion script compiles and is reproducible with pynwb-only NWB access.
- All workflow steps are COMPLETE.

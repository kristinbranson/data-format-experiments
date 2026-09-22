# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- numpy 2.4.4
- torch 2.6.0+cu124

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
| `create_sess` (TwoPUtils, called by notebook) | `notebooks/make_session_pkl.md` | LOADING | Creates a synchronized session from scan metadata, VR, Suite2p, and behavior streams. |
| `dff` | `src/reward_relative/preprocessing.py` | PROCESSING | Computes fluorescence delta-F/F; published multi-session files use the maximin baseline method. |
| `multi_anim_sess` | `src/reward_relative/utilities.py` | PROCESSING/CURATION | Computes dF/F, place-cell shuffles, spatial trial matrices, and trial-set/session metadata. |
| `get_timeseries_data` | `src/reward_relative/glmUtils.py` | PROCESSING | Extracts synchronized trial, relative-position, absolute-position, speed, reward, and neural event timeseries. |
| `calc_lick_metrics` | `src/reward_relative/behavior.py` | PROCESSING | Computes trial-wise lick matrices and lick metrics. |
| decoder helpers | `src/reward_relative/decode.py` | PROCESSING | Cross-validated decoding across pre/post reward-location blocks. |

### Notes
- The repository represents calcium-imaging data. Raw `sess` objects synchronize fluorescence with VR and do not initially contain dF/F. Session creation calls `create_sess(..., load_scaninfo=True, load_VR=True, load_suite2p=True, load_behavior=True)` and writes a pickle.
- The analysis-level `multi_anim_sess` computes dF/F and place-cell statistics. Documentation states that published files use a **maximin** dF/F baseline. Thus dF/F is required if starting from raw fluorescence; if provided files already contain processed neural streams, those streams should not be recomputed blindly.
- Stored per-animal fields include `sess`, place-cell masks/information/shuffles for sets 0/1, reward metadata, reward-zone label/location, and `trial dict`. Set 0 is the 30 trials before a reward switch; set 1 is post-switch.
- Fig. 3 calls `glmUtils.get_timeseries_data` for `trials`, reward-relative position, absolute position, speed, and rewards, using a 2 cm/s speed threshold. Its design matrix is deconvolved neural events (timepoints x neurons); NaNs are replaced with zero for model fitting. Relative-position NaNs are interpolated.
- The paper decoder occupancy-matches absolute-position samples between trial sets and compares reward-relative, track-relative, and remapping cell groups. Those downsampling/cell-group choices answer the paper's cross-map scientific question and are not general trial curation rules for this task, which explicitly requires neural activity to predict all listed variables over complete trials.
- No electrophysiology quality filtering applies. Suite2p cell classification and any existing session curation must be respected; place-cell-only filtering is analysis-specific rather than a general recording-quality filter.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is approximately 87 GB and contains 152 NWB 2.5/HDF5 files organized as `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`; no separate README is present.
- Every file has `processing/behavior/BehavioralTimeSeries` with framewise `autoreward`, `environment`, cumulative `lick`, `position`, cumulative `reward_zone` entry, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`. `Reward` is instead a sparse timestamped event series whose values are 0.004 mL.
- Trial numbers are contiguous 0..N-1; -1 denotes intertrial/invalid periods. `trial_start` and `teleport` identify track entry and trial end. Position is in cm and speed in cm/s. Environment uses 0/1 in trials and -1 outside trials.
- `processing/ophys` contains `Deconvolved/plane0`, `Fluorescence/plane0`, `Neuropil/plane0`, and Suite2p `PlaneSegmentation`. Neural arrays are timepoints x ROIs. `iscell[:,0]` is the accepted-cell mask and `iscell[:,1]` the classifier probability.
- All recordings are from `hippocampus, CA1`. Ophys rates are 15.5078125 Hz (124 sessions) or 31.015625 Hz (28 sessions). Behavior has explicit synchronized timestamps and no NaNs in core continuous/categorical fields.
- Ten sessions have one extra neural sample relative to every framewise behavior stream; conversion must use the common stream length. Reward must be aligned from its sparse timestamps rather than treated as framewise data.
- `lick` and `reward_zone` are cumulative counters; lick/zone-entry events are positive increments, not positive levels. The `reward_zone` stream does not encode A/B/C identity; location must be obtained from trial reward-zone entry position/reference definitions.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 Suite2p-accepted cells; 260,091 all segmented ROIs |
| Neurons / session | accepted: 155–2,341; all ROIs: 315–3,934 (all-ROI mean 1,711.1) |
| Subjects | 11: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19 |
| Sessions / subject | 14 each, except m11 has 12; 152 total |
| Trials (raw/retained) | 12,217 numbered IDs / 12,216 complete trials retained |
| Trials / session | raw 41–100; retained 40–100, mean 80.368 |

Additional checks:
- Trial numbering is contiguous and has no detected gaps in any session.
- 73 sessions contain ENV1 trials only, 68 ENV2 only, and 11 contain both environment codes.
- There are 12,216 trial-start and 12,216 teleport markers versus 12,217 numbered trials, consistent with a single recording-boundary omission; numbered trial samples are the robust segmentation source.
- All 152 sessions expose identical behavior variable names and all have at least two trials.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote / interpretation |
|-----------|-------|-------------------------------|
| Neurons (total) | Not stated as one unfiltered total in supplied methods; data contain 138,678 accepted ROIs | Paper reports analysis-specific subpopulations rather than the complete NWB export. |
| Neurons / session | Not specified globally | Simultaneously imaged CA1 populations. |
| Subjects | Paper analyses include an initial cohort of 7 and later animals; complete export has 11 | “In an initial cohort of n=7 mice…” is used to set the RR threshold; it is not the full-data count. |
| Sessions / subject | Not specified globally | Complete NWB export provides 152 sessions. |
| Trials (total) | Not specified globally | Trials are laps of the track, split around reward switches for relevant analyses. |
| Trials / session | Set 0 is often the 30 trials before switch; post-switch set uses remaining trials | Code documentation; this is an analysis grouping, not acquisition duration. |
| Neural data time bin | Imaging frames, approximately 15.5 Hz (~64.48 ms) | “All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.” |
| Behavior data time bin | Synchronized to imaging frames, approximately 15.5 Hz | Same source statement. |
| Reward rate | Not reported as one global fraction | Rewarded and omission trials are both included; sparse reward delivery timestamps are present in NWB. |
| Track length | 450 cm | Methods/task description. |
| Spatial analysis bin | 10 cm, often smoothed with 10 cm s.d. Gaussian | Used for spatial maps/k-means, not temporal decoder binning. |
| Movement threshold | Exclude <2 cm/s in selected spatial/decoder analyses | Analysis-specific; requested outputs explicitly include a <2 cm/s class. |

### Processing Details
- The experiment uses a virtual 450 cm linear corridor, with reward delivered at one of three locations (A/B/C), reward-location switches, omission trials, lick sensing, running speed, and teleport/intertrial periods.
- Suite2p performs motion correction, ROI extraction and cell classification. The NWB export includes the Suite2p `iscell` result, fluorescence, neuropil, and deconvolved activity.
- Paper population decoding uses **deconvolved calcium events**. Behavioral and neural streams are synchronized at the imaging frame rate (~15.5 Hz).
- Trial-based spatial analyses bin activity into 10 cm position bins and may Gaussian smooth by 10 cm; this spatial averaging is inappropriate for the requested time-varying temporal decoder.
- Fig. 3 predicts circular reward-relative position from deconvolved events, occupancy-matches position between maps, and uses cross-validation. It does not report categorical accuracies for the six outputs requested here, so only qualitative above-chance expectations are transferable.

### Curation Steps

**Neuron curation rules**:
Use Suite2p-accepted cells (`iscell[:,0] == 1`) as the recording-quality criterion. Place-cell, reward-relative-cell, and track-relative-cell masks are scientific subpopulation definitions and are not general quality filters.

**Trial curation rules**:
Use numbered track trials with a `trial_start` marker and exclude intertrial samples (`trial number == -1`) plus the unique incomplete recording-edge fragment. Paper-specific selections (30 trials pre-switch, rewarded/omission stratification, k-means significance, reward-zone subsets, or speed >2 cm/s) apply only to individual analyses and should not remove trials here. The requested speed output requires retaining low-speed frames.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Circular reward-relative position (paper Fig. 3) | Reported with circular-regression scores/shuffle comparisons, not directly comparable categorical accuracy |
| Requested six outputs | No matching accuracies reported in paper |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural rate | VR is aligned one-to-one to imaging frames | 124 series report 15.5078125 Hz; 28 report 31.015625 Hz, but all neural rows match behavior rows and behavior median spacing is ~0.06448 s even in nominal 31 Hz files | All behavior and neural series sampled at ~15.5 Hz | Preserve rowwise alignment and use common 64.4836 ms bins; do **not** factor-2 downsample nominal-31-Hz files because that would misalign and discard half the behavior-aligned neural samples. |
| Neural signal | Paper decoder uses deconvolved events; dF/F used for spatial-field analyses | NWB provides Deconvolved, raw Fluorescence, Neuropil | Decoder methods use deconvolved calcium events | Use provided Deconvolved activity, avoiding redundant dF/F recomputation. |
| Cell filtering | Suite2p cell masks are available; place-cell masks are analysis-specific | `iscell` includes accepted flag/probability | Suite2p preprocessing followed by scientific cell classification | Retain `iscell[:,0] == 1`; do not impose place-cell/RR/TR filters. |
| Trial definition | Session code uses synchronized VR trials | `trial number` is contiguous 0..N-1; -1 is intertrial; one boundary marker is missing globally | Trials are corridor laps separated by teleport | Segment by nonnegative trial number, which is more robust than event markers. |
| Reward stream | Reward is event-based | `Reward` has sparse timestamps and 0.004 mL values | Rewarded and omission trials are analyzed | A trial is rewarded if any sparse reward timestamp falls within its frame timestamp interval. Multiple deliveries remain binary 1. |
| Lick stream | Licks are accumulated into imaging frames | Values 0–6 fluctuate within trials rather than being globally cumulative | Lick counts/rates are analyzed | Binary lick output is `lick > 0` per frame, not a difference of adjacent frames. |
| Reward-zone identity | Code uses reward-zone start and stable pre/post-switch sets | First positive `reward_zone` frame occurs near 80, 200, or 320 cm; omission trials may lack an event | Three zones A/B/C and reward switches | Label observed trials by nearest start; nearest-known trial propagation gives one block in 75 sessions and exactly two blocks in 77, with no extra switches. Counts A/B/C = 4,185/4,008/4,024. |
| Reward-zone distance | Reference aligns position to reward-zone start for circular analyses | Track is 0–450 cm and starts cluster at 80/200/320 cm | Zones are 50 cm regions | For requested “distance to any location in reward zone,” use signed Euclidean distance to interval [start,start+50]: negative before, exactly 0 inside, positive after. This differs intentionally from circular distance-to-start because the requested zero class denotes being anywhere in-zone. |
| Speed threshold | Some paper analyses exclude <2 cm/s | Full speed stream includes low and negative near-zero estimates | Decoder task explicitly defines a <2 cm/s output class | Retain all in-trial frames; values <2, including tiny negative estimates, map to class 0. |

Final understanding: NWB files are already synchronized, processed exports. Use accepted deconvolved CA1 cells and all numbered trial frames. Preserve every valid trial and behavior-aligned sample, except truncate the one unmatched trailing neural sample in ten sessions implicitly by indexing neural rows with behavior-derived trial indices.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data[:, iscell]` | `neural` | Select `iscell[:,0]==1`, transpose each trial to neurons x time, float32 | `glmUtils.get_timeseries_data` | Paper decoder uses deconvolved events. |
| behavior timestamps + trial sample indices | `input[0]` | Seconds since first frame of numbered trial | synchronized `sess` timeseries | Continuous time-varying. |
| `environment` | `input[1]` | Modal in-trial 0/1 value, broadcast over trial | `get_timeseries_data` | ENV1=0, ENV2=1. |
| `trial number` | `input[2]` | Native zero-based continuous trial index, broadcast | trial dictionary/session timeseries | Preserves source numbering. |
| previous trial sparse reward outcome | `input[3]` | Trial 0=0; otherwise prior trial binary outcome, broadcast | reward timeseries loading | Omitted=0, rewarded=1. |
| position + inferred reward-zone interval | `output[0]` | Signed distance to [start,start+50], then classes: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50` | reward-aligned position logic in spatial analyses | Exactly zero throughout zone. |
| `position` | `output[1]` | Classes `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360` | synchronized behavior | Five equal 90 cm bins. |
| `speed` | `output[2]` | Classes `<2`, `[2,10)`, `[10,20)`, `[20,40]`, `>40` | synchronized behavior | Retain low-speed frames. |
| `lick` | `output[3]` | `(lick > 0).astype(int)` per frame | behavior lick processing | Frame-bin lick presence. |
| first reward-zone entry position, block-filled | `output[4]` | nearest 80/200/320 cm -> A/B/C = 0/1/2, broadcast | reward-zone metadata/relative position code | Nearest-known trial fill handles omissions. |
| sparse `Reward/timestamps` | `output[5]` | Any event in trial frame-time interval -> 1, else 0, broadcast | event alignment | Multiple deliveries remain 1. |

### Key Decisions
1. **Temporal bins**: preserve one sample per behavior-aligned imaging row (nominal 15.5078125 Hz, 64.4836 ms). Reported 31 Hz metadata conflicts with explicit behavior timestamps and rowwise synchronization, so it is not used to resample.
2. **Trials**: include every complete numbered trial (12,216 total); exclude intertrial samples and the unique 3-frame edge fragment lacking `trial_start`. Variable trial lengths are preserved.
3. **Neurons**: use all Suite2p accepted cells, not only place/RR/TR cells. This is the reference recording-quality filter and avoids scientific-selection leakage.
4. **Reward location**: identify first positive reward-zone frame per trial by nearest canonical start (80/200/320 cm), then nearest-known trial propagation. Stable one/two-block structure validates inference.
5. **Distance**: signed distance to the closed 50 cm zone interval makes the required class 3 exactly represent “in reward zone.” Boundary implementation follows the requested inequalities.
6. **Position outliers**: only numbered-trial samples are used; intertrial sentinel positions (-50/-500) are excluded naturally.
7. **Memory**: float32 neural/input and compact integer outputs; read only selected neural columns and trial rows from HDF5.

### Planned Sanity Checks
- [x] Confirm all behavior streams have common length and trial numbers are contiguous.
- [x] Confirm accepted-cell counts equal `iscell[:,0]` sums and all sessions have >=2 trials.
- [x] Confirm zone labels form at most one switch/session and starts cluster near canonical positions.
- [ ] Raw-vs-converted `np.allclose` neural spot-check for selected session/trial/cell/timepoint.
- [ ] Raw-vs-converted `np.allclose` input time/environment/trial/previous-outcome check.
- [ ] Raw-vs-converted `np.allclose` output position/speed/lick/reward check.
- [ ] Validate categorical ranges/distributions and total session/trial/neuron counts.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements `--full` (default), `--sample`, and `--show-processing`. It derives trial metadata from small behavior arrays, infers reward-zone blocks, aligns sparse reward events by timestamps, reads deconvolved trial slices, applies the Suite2p accepted-cell mask, constructs 4 inputs and 6 categorical outputs, validates every temporal shape, and writes the required dictionary.

Boundary tests cover every specified discretization edge and nearest-trial reward-zone filling. The script passes `py_compile` and CLI help tests.

Code inefficiencies identified:
- Loading all 260,091 segmented ROI streams would waste memory because only 138,678 accepted cells are used.
- Reopening files or reading each neural timepoint separately would cause excess HDF5 I/O.

Code speedups added:
- Each session is opened once for conversion; each trial is read as one contiguous HDF5 slice and then cell-selected.
- Behavior arrays are read once per session and vectorized discretization is used.
- Neural arrays use float32 and outputs use int8 to reduce pickle size and memory.
- Timing is printed per session and globally.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 323 session-cells |
| Neurons / session | 155, 168 |
| Subjects | 1 (m11) |
| Sessions / subject | 2 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| Trial length | 171–562 frames; mean 253.21 |
| Time from trial start range | [0, 36.18] s |
| Environment range | [0, 0] in this two-session sample |
| Trial number range | [0, 80] |
| Previous outcome range | [0, 1] |
| Distance-class fractions | [0.3463, 0.0659, 0.0411, 0.2037, 0.0155, 0.0512, 0.2763] |
| Position-class fractions | [0.5056, 0.1478, 0.1454, 0.1046, 0.0966] |
| Speed-class fractions | [0.0885, 0.0449, 0.0484, 0.1828, 0.6354] |
| Lick fractions | [0.8672, 0.1328] |
| Zone fractions | [0.8015, 0.1985, 0] |
| Reward-outcome fractions | [0.1590, 0.8410] |

### Processing Plots Review
Two 203–204 KB plots were created, one per sample session. Each overlays raw position/speed/lick with derived signed distance and categorical outputs on the same trial-relative time axis, plus deconvolved neural activity. Trial-relative time begins at zero; output transitions follow their continuous source traces; per-trial zone/outcome remain constant. No alignment or discretization anomaly was found.

The provided validator reported: `Data format is valid, no errors or warnings.` All neural/input/output temporal dimensions match, all requested category ranges are represented except categories absent naturally from this two-session subset, and every trial begins at relative time zero.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Contiguous per-trial HDF5 reads, vectorized transforms, float32/int8 storage | Sample conversion completed in 1.27 s |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion computation | 0.45–0.79 s (0.64 s mean) | ~97 s for 152 sessions, plus pickle write |
| Full conversion total | projected | <3 minutes, well below 15 minutes |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

Loss decreased steadily from approximately 46 at initialization to 0.955 at epoch 200; test loss was 0.995. Training finished successfully.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Distance to reward zone | 0.3993 | 0.3230 |
| Absolute position | 0.5123 | 0.4434 |
| Speed | 0.3757 | 0.3220 |
| Lick | 0.7173 | 0.6759 |
| Reward zone location | 0.9070 | 0.9301 |
| Reward outcome | 0.5969 | 0.6666 |

Every validation accuracy is above uniform chance (respectively 0.1429, 0.2, 0.2, 0.5, 0.3333, 0.5). Train-validation gaps are modest and provide no evidence of conversion misalignment.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Iteration 1: the initial full run stopped at session 69 because 28 m17/m18 sessions contain two imaging planes. Their combined PlaneSegmentation table did not match plane0 alone. The converter was corrected to use `planeIdx`, apply each plane-specific accepted-cell mask, and concatenate all `Deconvolved/plane*` activity. A two-plane session spot-test matched the combined raw `iscell` count exactly. Sample conversion and verification were rerun with no regression or warnings.

Iteration 2 completed all sessions in 52.57 s, substantially faster than the <3 minute estimate. The provided full verifier reported valid format with no errors or warnings and completed successfully.

### Output Files
- `converted_data.pkl`: 13.330 GB (13 GB displayed by `du`)
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | analysis-specific subsets | Suite2p accepted cells | 138,678 accepted | 138,678 | Yes |
| Mean neurons/session | not global | simultaneous session population | 912.36 | 912.36 | Yes |
| Subjects | complete export cohort | session dictionaries by animal | 11 | 11 | Yes |
| Sessions | not stated globally | one synchronized `sess` per recording | 152 | 152 | Yes |
| Trials (total) | laps of corridor | complete trial starts | 12,217 numbered IDs; 12,216 complete | 12,216 | Yes (one edge fragment excluded) |
| Trials/session (mean) | task blocks vary | complete numbered trials | 80.368 (40–100) | 80.368 (40–100) | Yes |
| Time bin | ~15.5 Hz | behavior aligned to imaging rows | median 64.48 ms | 64.4836 ms metadata | Yes |
| Environment | ENV1/ENV2 | codes 0/1 | [0,1] | [0,1] | Yes |
| Reward zone | A/B/C | stable pre/post switch blocks | [0,2] | [0,2] | Yes |
| Reward outcome | omission/reward | sparse reward events | [0,1], trial reward rate 0.8465 | [0,1] | Yes |
| Distance output | requested seven bins | N/A | classes 0–6 derivable | classes 0–6 | Yes |
| Position output | requested five bins | N/A | classes 0–4 derivable | classes 0–4 | Yes |
| Speed output | requested five bins | N/A | classes 0–4 derivable | classes 0–4 | Yes |
| Lick output | framewise detection | lick stream | classes 0/1 | classes 0/1 | Yes |

No sessions or accepted cells were lost; one demonstrably incomplete numbered recording-edge fragment was intentionally excluded. Ten one-sample neural/behavior length discrepancies are safely handled by behavior trial indexing; all converted frames have matching neural/input/output dimensions. Spot checks include early one-plane sessions and an m17 two-plane session.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` begins “Data format is valid, no errors or warnings” and ends “Data verification complete.” It reports 152 sessions, 12,216 trials, 11 subjects, 4 inputs, 6 outputs, 138,678 CA1 cells, and all expected categorical ranges.
2. **Independent raw neural checks (`np.allclose`)**: `/app/sanity_checks.py` directly loads NWB files without importing conversion code. Complete accepted-cell neural matrices match for trial 5 in an early one-plane session (m11), a two-plane session (m17), and a late one-plane session (m7). Multi-plane raw matrices are independently selected by `planeIdx` and concatenated.
3. **Independent raw input checks (`np.allclose`)**: For the same three trials, independently reconstructed relative time, environment, native trial number, and prior native-trial reward outcome all match exactly.
4. **Independent raw output checks (`np.allclose`)**: Independently reconstructed signed zone distance classes, position classes, speed classes, binary lick, inferred A/B/C zone, and sparse-timestamp reward outcome all match exactly.
5. **Data loading comparison**: Reference `create_sess` loads scan/VR/Suite2p/behavior into synchronized sessions. Conversion reads the equivalent synchronized NWB processing modules. Difference: NWB is the supplied finalized format, so no raw Scanbox/VR reconstruction is needed.
6. **Neuron/trial filtering comparison**: Reference preprocessing uses Suite2p and scientific analyses subsequently select place/RR/TR cells. Conversion uses Suite2p `iscell[:,0]` only, retaining all quality-accepted cells because scientific subpopulation selection would bias this general decoder. Complete trial starts are retained regardless of speed/outcome.
7. **Temporal alignment comparison**: Reference `get_timeseries_data` aligns deconvolved events and behavior at imaging frames. Conversion preserves one-to-one NWB rows and aligns trials at their first numbered sample/start. Explicit timestamps yield 64.4836 ms spacing.
8. **Binning comparison**: Paper 10 cm position bins/smoothing are specific to spatial-map analyses. Conversion does no spatial averaging because requested outputs are time-varying; it applies only the mandated categorical boundaries. Low-speed frames are retained because `<2 cm/s` is an explicit target class.
9. **Input construction comparison**: Native synchronized timestamps/trial/environment and sparse reward events are used. Previous outcome is shifted by one native trial, with first trial set to omission=0.
10. **Output construction comparison**: Position/speed/lick come directly from synchronized behavior. Sparse rewards are aligned by timestamps. Zone starts inferred at 80/200/320 cm form exactly one stable block in 75 sessions or one switch in 77 sessions before edge-fragment curation.
11. **Key statistics**: 11 subjects, 152 sessions, 138,678 cells, and 12,216 retained complete trials match independent raw counts. Global converted output counts/fractions are: distance `[0.452443,0.074304,0.053825,0.173454,0.015041,0.052407,0.178525]`; position `[0.421159,0.130103,0.168970,0.165706,0.114062]`; speed `[0.113280,0.069066,0.107752,0.273356,0.436546]`; lick `[0.817574,0.182426]`; zone `[0.330746,0.330562,0.338691]`; outcome `[0.156006,0.843994]`.
12. **Global edge/invariant checks**: 3,590,698 frames; trial length 123–10,968 frames; all trial-relative times start at zero and are monotonic; all per-trial zone/outcome labels are constant; every category is in range; every neural/input/output temporal dimension agrees.

### Issues Found and Resolved
- **Two-plane sessions**: Initial full run exposed 28 m17/m18 sessions where PlaneSegmentation combines both planes. Fixed by selecting accepted cells separately using `planeIdx` and concatenating every `Deconvolved/plane*`. Direct raw count and matrix checks pass. Sample and full conversion/validation were rerun.
- **Incomplete final fragment**: Raw m11 session 03 native trial 80 was uniquely only 3 frames, had no `trial_start` or `teleport`, moved backward over a partial corridor, and was the only numbered ID lacking a start. It was excluded. All other 12,216 trials have a start marker; missing teleport markers are common and therefore not a filter. Sample conversion, verification, decoder training, full conversion, full verification, and all checks were rerun.
- **Nominal 31 Hz metadata**: Two-plane series report aggregate scanner rate 31.015625 Hz, but each plane row is aligned one-to-one with behavior timestamps at 15.5078125 Hz. No downsampling is applied; raw-vs-converted alignment checks pass.
- **No remaining warnings, mismatches, or unresolved issues.**

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Loss fell from 158.8767 at epoch 1 to 1.2565 at epoch 200; test loss was 1.1227.
- Training completed successfully on CUDA using 9,772 training and 2,444 validation trials.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Distance to reward zone | 0.4445 | 0.3950 | Chance 0.1429; 2.77x chance validation |
| Absolute position | 0.5220 | 0.4986 | Chance 0.2; 2.49x chance |
| Speed | 0.4597 | 0.4303 | Chance 0.2; 2.15x chance |
| Lick | 0.6143 | 0.6025 | Chance 0.5; 1.21x chance |
| Reward zone location | 0.7978 | 0.7537 | Chance 0.3333; 2.26x chance |
| Reward outcome | 0.5691 | 0.5163 | Chance 0.5; 1.03x chance; investigated in Step 12 |

All outputs are above chance. Train-validation gaps are small (absolute gaps 0.012–0.053), with no >1.5x train/validation ratio.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Accuracy / Chance | Expectation from Paper |
|----------|---------------------|--------|-------------------|------------------------|
| Distance to reward zone | 0.3950 | 0.1429 | 2.765 | Paper shows reward-relative position is decodable, but reports circular-regression scores rather than matching categorical accuracy |
| Absolute position | 0.4986 | 0.2000 | 2.493 | Track-relative spatial coding predicts strong above-chance decoding; no matching categorical value |
| Speed | 0.4303 | 0.2000 | 2.151 | Speed is represented in CA1/task GLMs; no matching categorical value |
| Lick | 0.6025 | 0.5000 | 1.205 | Not decoded categorically in paper |
| Reward zone location | 0.7537 | 0.3333 | 2.261 | Reward-relative remapping predicts strong contextual information; no matching categorical value |
| Reward outcome | 0.5163 | 0.5000 | 1.033 | Paper compares rewarded/omission activity but does not report an outcome decoder accuracy |

**Accuracy versus paper:** The paper's Fig. 3 decoder predicts circular reward-relative position with circular-regression scores and occupancy-matched scientific cell classes. None of the six requested categorical tasks has a directly comparable reported accuracy. It would be misleading to equate circular-regression scores with balanced categorical accuracy. Qualitatively, the strong distance/position/zone results agree with the paper's spatial and reward-relative coding conclusions.

**Low-accuracy investigations:**
- Reward outcome and previous outcome were checked on three specific raw trials (rewarded, omitted, and a transition) by direct sparse timestamp alignment. Converted labels and previous-outcome inputs matched exactly.
- Independent whole-trial raw-vs-converted output checks passed for one-plane and two-plane sessions. Outcome class variation is sufficient: 15.60% omitted and 84.40% rewarded by frames (trial-level reward rate in raw data ~84.65%).
- Outcome is only revealed when the animal reaches the reward zone, but the required per-trial label is broadcast over the entire trial. Most pre-zone neural samples cannot causally encode a future stochastic omission, limiting balanced accuracy despite correct alignment. Above-chance 0.5163 is therefore plausible.
- Licks are sparse (18.24% positive frames) and brief within-frame events; raw `lick > 0` comparisons passed. A modest 0.6025 balanced accuracy is plausible and still above chance.
- Neural filtering matches Suite2p `iscell`; deconvolved events and behavior are synchronized rowwise. No processing change consistent with the task/reference would improve these labels without leakage (for example, restricting to post-reward frames would violate full-trial output requirements).

**Train-validation gap:** Train/validation ratios are distance 1.125, position 1.047, speed 1.068, lick 1.020, zone 1.059, and outcome 1.102. All are far below the 1.5 overfitting threshold; absolute gaps are 0.0118–0.0528.

### Issues Found and Resolved
- No Step 12 conversion error was found. Neural/output alignment, category variation, quality filtering, and reference processing were rechecked.
- Sample plots (`sample_trials.png`, `predictions.png`, and processing plots) were produced successfully.
- Full training finished successfully, all outputs exceeded chance, and no further conversion iteration is justified.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading instructions, format, statistics, and validation accuracies
- [x] cache/ folder created; independent sanity script documented in `cache/README_CACHE.md`
- [x] All required output pickles, logs, plots, conversion code, and documentation organized

Final audit: full format verification has no errors/warnings; full training completed successfully; all six outputs exceed chance; all independent raw comparisons pass.

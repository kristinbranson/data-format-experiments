# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification: python 3.13.15; numpy 2.4.4; torch 2.6.0+cu124. Imports succeeded.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `app`
- `code`
- `data`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `pynwb_docs`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `pp.create_sess` | external dependency `TwoPUtils` (called by preprocessing notebooks) | LOADING | Constructs a session from scan metadata, VR, Suite2p, and behavior; notebooks call it with `load_scaninfo`, `load_VR`, `load_suite2p`, and `load_behavior` enabled. |
| GLM data preparation functions | `src/reward_relative/glmUtils.py` | PROCESSING | Segment samples from trial start through teleport, retain deconvolved event activity, position, speed and licks, and construct reward-relative/task predictors. |
| place-cell/spatial-information functions | `src/reward_relative/placeCellAnalysis.py` | CURATION | Compute position-binned activity and shuffled spatial-information significance for analysis-level place-cell labels. |
| reward-relative classification/decoding logic | `notebooks/Fig2_*`, `notebooks/Fig3_Decoder.md` and package utilities | PROCESSING | Compare maps across reward locations and decode reward-relative position using trial subsets/cross-validation. |
| session metadata dictionaries (`single_plane`, `multi_plane`) | `src/reward_relative/sessions_dict.py` | LOADING | Map animals and experimental days to date, scene, session, scan, and experimental-day metadata. |

### Notes
- The experiment is hippocampal two-photon calcium imaging, not electrophysiology. Local setup describes “reward-relative activity in hippocampal 2P imaging data.” Neural analyses use Suite2p-derived deconvolved event/activity arrays (`sess.timeseries['events']`); therefore no spike sorting quality filter is applicable and delta-F/F should not be recomputed when the released NWB already supplies processed neural activity.
- Early loading/session construction is delegated to the external `TwoPUtils` dependency. Repository preprocessing notebooks call `pp.create_sess(...)`, align scan/VR/Suite2p/behavior streams, and pickle session objects. For this conversion the released NWB must instead be read with `pynwb`, while reproducing the downstream semantics visible here.
- Trials are represented from a trial-start index to a teleport/end index. GLM code iterates `zip(trial_starts, teleports)` and uses deconvolved activity over the corresponding interval. This is consistent with trial-start temporal alignment and variable trial duration.
- Available local streams include forward/absolute position, speed, licks, deconvolved events, reward-zone identity/location, omission status, and scene/environment metadata. Scene names explicitly encode `Env1`/`Env2` and reward locations A/B/C, including within-session transitions.
- Lick preprocessing in `glmUtils.py` corrects likely sensor failures: trials with an implausibly high fraction of cumulative lick values are set to NaN; remaining lick values above one are clipped to one. GLM-specific smoothing and zero-centering are analysis transforms and should not be imposed on the decoder’s required binary lick output unless the released representation requires it.
- GLM preparation masks samples with invalid speed or lick values and can optionally apply a speed threshold. That mask is model-specific; conversion should inspect NWB validity and preserve valid trial samples rather than blindly apply a GLM-only speed threshold.
- Omission trials are explicitly tracked. Reward-relative position is calculated relative to the active reward zone, while the requested decoder output requires distance to any location in the reward zone; exact boundaries and sign convention must be resolved from data and methods in later ordered steps.
- Place-cell and reward-relative-cell labels are downstream analysis labels based on shuffled significance (README notes 100 shuffles per cell for spatial information and resulting slight stochasticity). They are not evidence for excluding otherwise valid recorded neurons from the general neural decoder.
- No files under `/app/data` were inspected during this step, in accordance with the ordered workflow.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 152 NWB files arranged as `sub-<id>/sub-<id>_ses-<nn>_behavior+ophys.nwb` (approximately one file per imaging session). All inspection used `pynwb.NWBHDF5IO`; `h5py` was not used.
- Each file has an NWB subject, a `TwoPhotonSeries` acquisition, a `behavior` processing module containing one `BehavioralTimeSeries`, and an `ophys` module containing `Deconvolved`, `Fluorescence`, `Neuropil`, `ImageSegmentation`, and background images. There is no NWB trials table and no units table.
- Frame-aligned behavioral series present in every file are: `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`. `Reward` is event-based and shorter than the frame streams. Trial identities must therefore be reconstructed from nonnegative values of `trial number`; the initial `-1` samples are outside numbered trials. Trial boundaries are cross-checkable against `trial_start` and `teleport` pulses.
- Neural activity is stored as an ophys `RoiResponseSeries` under `Deconvolved`, with time as the first dimension and ROI as the second. Fluorescence and neuropil traces are also available, but the reference analysis uses deconvolved events.
- `ImageSegmentation` has `pixel_mask`, `iscell`, and `planeIdx`. `iscell` is the Suite2p two-column output: column 0 is binary cell classification and column 1 is probability. All 312,110 segmented ROIs are represented in storage; 138,678 pass Suite2p `iscell[:,0] == 1` and constitute the quality-curated neural population.
- All frame-aligned streams use a 0.064483627204 s interval (~15.508 Hz). Ten sessions have exactly one extra neural sample relative to behavior; later conversion must align safely by truncating all frame streams to their common length and document/verify this edge handling.

### Available Variables and Native Types
| Variable | Native representation | Meaning/use |
|----------|-----------------------|-------------|
| Deconvolved | time × ROI float array | Suite2p-derived deconvolved calcium event activity |
| Fluorescence / Neuropil | time × ROI float arrays | Raw/processed optical traces; available but not the reference decoder signal |
| position | frame-aligned numeric | Absolute forward position on the 450 cm corridor |
| speed | frame-aligned numeric | Running speed |
| lick | frame-aligned numeric | Lick sensor signal |
| environment | frame-aligned categorical numeric | Environment identity (ENV1/ENV2 coding to verify from methods/code) |
| reward_zone | frame-aligned categorical numeric | Reward-zone/location state; native values include transition/state codes beyond three final A/B/C labels |
| Reward | timestamped event series | Delivered reward events/outcomes |
| autoreward | frame-aligned numeric | Automatic reward indicator |
| trial number | frame-aligned integer-like numeric | Trial ID; `-1` is pretrial/outside numbered trials |
| trial_start / teleport | frame-aligned binary pulses | Start and end/teleport boundary markers |
| scanning | frame-aligned numeric | Imaging/scanning validity/state indicator |

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 Suite2p-classified cells (312,110 all segmented ROIs) |
| Neurons / session | 155–2,341 curated; mean 912.36, median 921.5 |
| Subjects | 11 (`m3`, `m4`, `m7`, `m11`–`m15`, `m17`–`m19`) |
| Sessions / subject | 12 for m11; 14 for each other subject; 152 total |
| Trials (total) | 12,217 nonnegative native trial IDs |
| Trials / session | 41–100; mean 80.38; 139/152 sessions have 80 trials |

### Data-derived Checks
- Every session contains all 11 named behavioral series (`Reward` plus ten frame-aligned streams).
- Deconvolved time length matches the trial-number stream exactly in 142/152 sessions; the remaining ten differ by one terminal neural frame.
- Trial-count distribution is: 80 trials in 139 sessions; 100 in 7; and one session each with 41, 50, 60, 75, 81, and 90 trials.
- Native session timepoint counts span 14,164–51,520 frames; the fixed frame interval implies variable session/trial durations.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote / interpretation |
|-----------|-------|-------------------------------|
| Subjects | 11 mice in released data | Paper/methods describe longitudinal two-photon imaging across mice; native NWB census gives 11. |
| Sessions | 152 released NWB sessions | Native release census; paper analyses use task-specific subsets (for example, 77 sessions entered one RR population-vector analysis and 50/77 passed its k=2-vs-shuffle criterion). |
| Neurons (total) | 138,678 Suite2p-classified cells in release | Native `iscell[:,0]` census. Paper analyses subsequently restrict to place cells or other analysis-specific subsets rather than treating all segmented ROIs as neurons. |
| Track length | 450 cm | Hidden reward-zone task uses a 450 cm virtual linear corridor. |
| Neural/behavior sample interval | 0.0644836 s (~15.5 Hz) | NWB timestamps; consistent with two-photon frame-rate methods. |
| Spatial analysis bins | commonly 10 cm; some analyses use other circular/bin representations | K-means uses trial × 10 cm linear-position-bin deconvolved activity, smoothed with 10 cm s.d.; decoder methods use spatially binned activity as specified in that section. |
| Trials / session | usually 80; some 100 and incomplete/other sessions | NWB census: 139/152 have 80, seven have 100, others 41/50/60/75/81/90. Task design includes stable and switch sessions. |
| Reward rate | not stated as one global percentage | Rewarded and omission trials are explicitly compared; analyses of omission require at least three omission trials within each pre/post-switch trial set. Outcome should be derived directly from reward events per trial. |
| RR k-means session subset | 50 of 77 sessions | “n = 50 out of 77 sessions for the RR population vector” passed real-vs-shuffle k=2 criterion. |

### Processing Details
- Mice navigate a 450 cm virtual linear track with a hidden reward zone. Reward location can switch among A/B/C, and visual environments ENV1/ENV2 can remain or switch independently, enabling track-relative versus reward-relative analyses.
- Two-photon data were processed with Suite2p. The released NWB provides fluorescence, neuropil, deconvolved activity, segmentation, and Suite2p `iscell`; the reference analyses use deconvolved calcium activity. Recomputing delta-F/F is unnecessary for the requested neural decoder.
- Reference analyses align behavior and calcium samples on the imaging-frame timeline. Trials run from trial start through teleport; the new task specifically requires temporal alignment to trial start.
- Position-binned reference analyses commonly use 10 cm bins. This conversion should retain native temporal bins (~64.48 ms) because the requested decoder has time-varying outputs (time, position, speed, lick), rather than collapse samples into spatial bins.
- Licking analyses account for sensor errors and quantify spatial lick behavior. The requested output is binary lick, so valid native frame-level lick values should be thresholded/clipped to 0/1 rather than spatially smoothed.
- Reward-relative position in the paper is continuous and analyzed with ridge/linear decoding and cross-validation. The requested distance output instead has seven prescribed categorical bins, so exact paper decoder scores are not directly comparable.
- Trial-by-trial remapping analyses normalize deconvolved activity per neuron and use trial × 10 cm position bins, but this normalization is specific to those analyses. The downstream supplied neural decoder expects trial time series; conversion should preserve deconvolved event amplitudes and let decoder preprocessing handle scaling.

### Curation Steps

**Neuron curation rules**:
- Apply Suite2p's binary `iscell[:,0] == 1` classification to exclude non-cell ROIs.
- Do not restrict the general decoder to place cells, RR cells, TR cells, or cells passing shuffled spatial-information tests. Those are downstream scientific subpopulation labels and would discard neural information relevant to other requested outputs.
- Place-cell identification in the paper uses spatial information relative to shuffled null distributions (100 shuffles per cell according to repository README); resulting labels have small stochastic variation.

**Trial curation rules**:
- Include numbered trials (`trial number >= 0`) with valid aligned neural and required behavior samples.
- Use `trial_start`, trial-number transitions, and `teleport` to cross-check boundaries; exclude the initial `trial number == -1` period.
- Preserve rewarded and omitted trials because reward outcome and previous outcome are required variables. Do not apply the paper's “at least three omissions per trial set” restriction, which is specific to omission-comparison analyses.
- Respect scanning/validity and finite-data indicators; handle ten one-frame stream mismatches by common-length truncation at the terminal edge.

### Decoders Trained
| Decoded variable | Accuracy / metric |
|------------------|-------------------|
| Continuous reward-relative position | Paper uses cross-validated ridge/linear decoding and compares real scores with shuffled/control models; it does not report classification accuracy matching the task's seven bins. |
| Newly required distance, position, speed, lick, zone, and outcome categories | No directly comparable paper accuracy; validation must compare against chance and supplied decoder results. |

### Reference-text conclusion
The reference paper's scientific curation (place/RR/TR labels, omission-analysis subsets, k-means session acceptance) is analysis-specific. For a general neural decoder, the defensible release-level curation is Suite2p cell classification plus valid trials/samples, while preserving all task conditions and outcomes.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | Reference analyses use `sess.timeseries['events']` / deconvolved activity | NWB `ophys/Deconvolved` is time × all segmented ROIs | Calcium data processed with Suite2p | Use Deconvolved and retain only `iscell[:,0] == 1`; do not recompute dF/F. |
| Trial table | Code segments from trial starts to teleports | No NWB trials table; frame-level trial number/start/teleport exist | Trials are traversals of the VR track | Group contiguous samples by nonnegative trial number, require a start pulse and complete traversal, and align at its first sample/start. |
| Trial count | Typical analyses expect complete trial sets | 12,217 IDs, but one terminal ID has only 3 frames, no start/teleport, and max position 154 cm | Analyses use completed trials | Exclude only this malformed terminal trial, leaving 12,216 valid trials. All others have one start pulse and reach >=440 cm. |
| Teleport marker | Reference code uses teleport/end indices | Resampling yields 0, 1, or 2 positive teleport samples per otherwise complete trial | Teleport includes tunnel/jitter period | Do not require exactly one pulse. Trial-number grouping plus start pulse and track completion is robust; include the whole numbered trial as the native start-to-next-boundary interval. |
| Scanning validity | Imaging/behavior are aligned during scanning | `scanning==1` at every numbered-trial sample and `-1` only outside trials | Analyze imaging periods | Selecting numbered trials exactly enforces scanning validity. |
| Environment | Scene metadata names ENV1/ENV2 | Frame stream is 0 in ENV1 and 1 in ENV2, including environment-switch sessions | Two visual environments can switch independently of reward | Use per-trial modal frame value, 0=ENV1 and 1=ENV2. |
| Reward-zone source | Code uses active reward-zone location and reward-relative position | Native `reward_zone` values 0–7 are transient state/event combinations, not A/B/C labels | Locations A/B/C are fixed along track and switch at task-defined trial | Derive per-trial A/B/C from scene/task schedule: scene's first location for trials <40 and second for trials >=40 in reward-only `LocationX_to_Y`; stable scenes retain X. Combined `EnvN_X_to_EnvM_Y` sessions switch at trial 30 and use the frame environment to select the corresponding location. Cross-check: native nonzero-state positions center near A≈82, B≈202, C≈322 cm. |
| Reward outcome | Omission logic is event/trial based | `Reward` has timestamped 0.004-volume events; all map cleanly to numbered trials. `autoreward` is always zero. | Rewarded and omission trials are compared | Outcome=1 if at least one Reward timestamp maps within that trial, else 0. Do not use `autoreward`. |
| Previous outcome | Not a primary reference predictor | First trial has no preceding outcome | Task requires binary previous outcome | Encode first valid trial as 0 (no prior rewarded trial available); subsequent trials copy prior valid trial outcome. |
| Lick | GLM code clips values >1 after sensor-error handling | Native lick ranges 0–8 | Lick behavior is event/count-like | Required output is binary: 0 if value <=0, 1 if >0. Do not smooth. |
| Stream lengths | Reference session construction aligns modalities | 142 sessions match exactly; ten have one extra terminal neural frame | Frame-aligned imaging/behavior | Truncate to the common time length before trial indexing; mismatch is terminal and one frame only. |
| Position domain | Reference analyses may include teleport/tunnel position down to -50 cm | Numbered trials usually span about -50 to 450 cm; first trial can start near -5 cm | Main corridor is 0–450 cm; teleport is separately discussed | Preserve complete numbered-trial samples for temporal fidelity, but absolute-position category follows required thresholds (<90, 90–180, ..., >360), naturally assigning negative teleport samples to bin 0. Distance-to-zone is computed against the active zone interval. |

### Final Consistent Understanding
- The release is already temporally aligned at a fixed 64.4836 ms imaging-frame interval; no interpolation is needed.
- Neural conversion is a time × curated-cell slice of NWB Deconvolved activity, transposed to neuron × time per trial.
- Trial context comes from frame-level environment plus scene-defined reward-location schedule. Behavioral outputs remain frame-level; outcome and zone location are repeated across trial time so all output rows have a common `(6, time)` shape.
- Reference place/RR/TR filters are scientific subset definitions, not recording-quality filters. Suite2p `iscell` is the appropriate general-neuron curation rule.
- Sanity checks must explicitly verify scene/code mapping against native reward-state positions, event-to-trial outcome mapping, trial boundaries, and exact source-array slices through `pynwb`.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Deconvolved` + `ImageSegmentation.iscell[:,0]` | `neural` | Select binary cell ROIs, truncate to common aligned length, slice each valid trial, transpose time × cells to cells × time, cast float32 | `glmUtils` uses deconvolved activity | Preserve native event amplitudes; no dF/F recomputation or spatial binning. |
| frame timestamps / fixed interval | `input[0]` time from trial start | `arange(n_time) * 0.064483627204`, float32 | trial start-to-teleport segmentation | Continuous and time-varying, starting at exactly 0. |
| `environment` | `input[1]` environment type | Modal value over trial, repeat over time; 0=ENV1, 1=ENV2 | session scene/task handling | Native coding agrees with scene names. |
| `trial number` | `input[2]` trial number | Native nonnegative trial ID, repeat over time, float32 | trial dictionaries | Preserve zero-based native numbering; continuous per-trial predictor. |
| previous derived reward outcome | `input[3]` previous trial outcome | Prior valid trial's event-derived outcome, repeated; first valid trial=0 | omission/reward logic | 0 omitted/no prior, 1 rewarded. |
| `position` + active scene-derived zone interval | `output[0]` distance to reward zone | Signed distance to nearest point in interval: negative before zone, 0 inside, positive after; discretize prescribed 7 bins | reward-relative position analyses | Zone intervals from task: A=80–100 cm, B=200–220 cm, C=320–340 cm. Exact 0 represents every location inside zone (“distance to any location in zone”). |
| `position` | `output[1]` absolute position | Prescribed bins: `<90`, `[90,180)`, `[180,270)`, `[270,360]` convention via `np.digitize(...,[90,180,270,360], right=False)`, `>=360` class 4 | absolute-position maps | Values outside corridor naturally enter edge classes. |
| `speed` | `output[2]` speed | Prescribed thresholds 2,10,20,40 cm/s with exact 40 retained in class 3 because class 4 is strictly >40; negative values, if any, class 0 | behavior/GLM speed | No smoothing or speed-threshold sample deletion. |
| `lick` | `output[3]` lick | `(lick > 0).astype(int)` | `glmUtils` clips >1 | Binary frame output, no smoothing. |
| scene reward schedule | `output[4]` reward zone location | A=0, B=1, C=2, repeated over time | scene dictionaries and task methods | Stable `LocationX`: X all trials. Reward-only `LocationX_to_Y`: X for IDs <40, Y for IDs >=40. Combined `EnvN_X_to_EnvM_Y` sessions switch both variables at native trial 30; use the aligned environment stream to select the matching X/Y side. |
| timestamped `Reward` events | `output[5]` reward outcome | Map each event timestamp to nearest frame/trial; any event in trial=1 else 0; repeat over time | omission analyses | `autoreward` is all zero and is not used. |
| NWB subject ID | `subjects`, `subject_idx` | Unique sorted IDs and per-session index | session dictionaries | 11 subjects. |
| hippocampal CA1 recording | `brain_regions`, `brain_region_idx` | `brain_regions=['CA1']`; zeros for every curated cell | paper anatomy / hippocampal imaging | One region label per cell. |

### Output Categories
| Output | `output_values` labels in integer order |
|--------|-----------------------------------------|
| distance to reward zone | `< -50 cm`, `-50 to -10 cm`, `-10 to <0 cm`, `inside reward zone (0 cm)`, `>0 to +10 cm`, `+10 to +50 cm`, `> +50 cm` |
| absolute position | `<90 cm`, `90-180 cm`, `180-270 cm`, `270-360 cm`, `>360 cm` |
| speed | `<2 cm/s`, `2-10 cm/s`, `10-20 cm/s`, `20-40 cm/s`, `>40 cm/s` |
| lick | `no`, `yes` |
| reward zone location | `A`, `B`, `C` |
| reward outcome | `no`, `yes` |

### Key Decisions
1. **Native temporal sampling**: Keep every aligned imaging frame at 64.483627 ms. This preserves all time-varying requested targets and uses a common bin size across sessions.
2. **Variable trial lengths**: Retain native complete trial durations rather than time-warping. The target permits a per-trial number of timepoints; all arrays within each trial match exactly.
3. **Cell curation**: Use Suite2p `iscell[:,0] == 1` only. Place/RR/TR labels are downstream scientific filters and inappropriate for a general decoder.
4. **Trial curation**: Exclude pretrial `-1` periods and the sole malformed 3-frame terminal trial; preserve all 12,216 complete trials, including omissions.
5. **Reward zone geometry**: Use methods-defined 20 cm zones A=80–100, B=200–220, C=320–340 cm. Signed distance is zero anywhere in the active interval, satisfying “distance to any location in the reward zone,” unlike distance to a center/start.
6. **Switch conventions**: Reward-only sessions switch after 40 trials (IDs 0–39 versus >=40). All combined environment+reward sessions switch at native trial 30; the aligned environment stream robustly selects the corresponding scene side. Native reward-state medians validate both conventions.
7. **Outcome assignment**: Timestamped Reward events are authoritative and map cleanly to trials. This is superior to `autoreward`, which is invariant zero.
8. **Common output shape**: Repeat per-trial zone/outcome across time, yielding integer `(6, n_time)` outputs. Repeat contextual inputs similarly, yielding float32 `(4, n_time)` inputs.
9. **Memory/I/O**: Process NWBs sequentially, materialize only deconvolved activity and required behavior for one session, immediately split/cast into compact trial arrays, then close the file. Use float32 neural/input and compact integer outputs.
10. **Metadata offsets**: Alignment is start of each numbered trial; `off_start=0.0`. Trial durations vary, so `off_end=None`; explain this explicitly in metadata.

### Planned Sanity Checks
- [ ] With a fresh independent `pynwb` load, compare a selected converted neural trial against the original Deconvolved `[trial_indices, iscell]` slice using `np.allclose` after transpose.
- [ ] Compare converted time, environment, trial number, and previous outcome for at least three trials (including a switch-boundary trial) against independently read timestamps/streams/events using `np.allclose`.
- [ ] Compare converted position/speed/lick classes and per-trial zone/outcome to independently computed raw-NWB values with `np.allclose` for at least three trials.
- [ ] Verify every session has >=2 trials, all within-trial neural/input/output time dimensions match, and curated-cell counts match `iscell[:,0]`.
- [ ] Verify exactly 152 sessions, 11 subjects, 12,216 trials, and 138,678 curated cells after conversion.
- [ ] Verify zone native-state medians cluster near A≈82, B≈202, C≈322 cm and every scene parser result agrees with frame environment codes.
- [ ] Verify all categorical outputs are integer-valued and within declared class ranges; report distributions and reward rate.
- [ ] Check edge thresholds explicitly (`-50,-10,0,+10,+50`; `90,180,270,360`; `2,10,20,40`) with unit tests to prevent off-by-one class errors.
- [ ] Verify the ten one-frame mismatches are resolved only by terminal common-length truncation and all retained trial indices remain in range.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the documented conversion with `pynwb.NWBHDF5IO` only. It supports `--full` (default), `--sample`, and `--show-processing`, validates every trial/session, reports per-session timing, and saves processing plots for up to two sessions.

Implementation includes scene parsing for stable, reward-switch, environment-switch, and combined-switch names; event-to-trial reward mapping; Suite2p cell curation; malformed-trial exclusion; categorical boundary tests; and rich session metadata.

Code inefficiencies identified:
- The 87 GB source requires avoiding repeated reads and retaining full-session float64 arrays.
- Plotting and diagnostic extraction can add overhead if performed for all sessions.

Code speedups added:
- Each NWB is opened once and closed after one session.
- Only required behavior and deconvolved activity are loaded; neural data are immediately cell-filtered and cast to float32.
- Outputs use int8 and session processing is sequential to control memory.
- Processing plots are restricted to two sessions and four representative trials per session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 323 curated cells |
| Neurons / session | 155, 168 |
| Subjects | 1 (`m11`) |
| Sessions / subject | 2 |
| Trials (total) | 160 valid trials |
| Trials / session | 80, 80 |
| Timepoints / trial | 171–562; session means 246.0 and 260.4 |
| Time input range | 0–36.2 s |
| Environment range | 0 only (both sample sessions ENV1) |
| Trial number range | 0–79 |
| Previous outcome range | 0–1 |
| Distance output fractions | [0.365, 0.066, 0.040, 0.141, 0.021, 0.063, 0.304] |
| Position output fractions | [0.506, 0.148, 0.145, 0.105, 0.097] |
| Speed output fractions | [0.088, 0.045, 0.048, 0.183, 0.635] |
| Lick output fractions | [0.867, 0.133] |
| Reward-zone output fractions | [0.741, 0.259, 0.000] (C absent from these two sessions) |
| Reward outcome, frame fractions | [0.159, 0.841] |
| Reward outcome, trial counts | 24 omitted, 136 rewarded (85.0% rewarded) |

### Processing Plots Review
Two processing PNGs were generated, one per sample session, each showing raw position/signed zone distance, speed/lick, and all categorical transforms for four trials. Images are valid 2080×1664 RGB files. Curves and category rasters share the same trial-relative time axis; no dimension or alignment anomaly was detected. The reward-switch sample shows the expected B-to-A category change across trial 40.

### Format Validation
- `/app/train_decoder.py --verify-only` completed with “Data verification complete.”
- Errors: none.
- Warnings: none.
- Every neural/input/output trial has matching time length and declared dimensions.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Single NWB open per session, float32 neural, int8 output, restricted plots | Sample conversion completed in 2.5 s including two plots and pickle I/O |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion (sample observed) | ~1.25 s including plots | ~190 s for 152 sessions; conservatively <5 min including larger files and full pickle I/O |

The estimate is far below 15 minutes, so no parallel I/O (which could increase memory pressure on 87 GB of NWB sources) is needed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Training Progress
- Loss decreased from approximately 45 at initialization to 0.9638 at epoch 200; test loss was 1.0271.
- Training completed successfully.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| distance to reward zone | 0.4028 | 0.3259 | 0.1429 |
| absolute position | 0.5046 | 0.4478 | 0.2000 |
| speed | 0.3505 | 0.2974 | 0.2000 |
| lick | 0.7190 | 0.6789 | 0.5000 |
| reward zone location | 0.9153 | 0.8932 | 0.3333 |
| reward outcome | 0.6024 | 0.6608 | 0.5000 |

Every validation output exceeds uniform chance. Distance, position, lick, and zone are substantially above chance. Speed is just under 1.5× chance in this one-subject/two-session sample and will be reassessed on the full data. Outcome validation exceeds both chance and its training estimate, providing no evidence of overfitting.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Iterations
- **Iteration 1 issue**: Initial full conversion stopped at session 6 because combined `Env1_B_to_Env2_C` changed environment/reward at trial 30, while the first parser incorrectly assumed trial 40 for every switch.
- **Fix**: Inspected all 11 combined-switch NWBs independently with `pynwb`; every environment transition occurred at trial 30 and native reward-state positions changed to the matching second location. Parser now distinguishes combined switches and uses aligned environment to choose the corresponding scene side. Reward-only switches retain the validated trial-40 rule.
- **Iteration 2 issue**: The second run stopped at the first multi-plane session because the initial implementation applied the complete PlaneSegmentation `iscell` mask to only `plane0` response columns.
- **Fix**: Inspected all RoiResponseSeries through pynwb. There are 124 single-series and 28 two-series sessions. Each response series links its columns to segmentation rows through `rs.rois.data`; conversion now applies `iscell` after this mapping and concatenates curated columns across every plane. All 152 mappings had valid, unique, in-range links and response-column counts equal to link counts.

### Output Files
- `converted_data.pkl`: 13.330 GB (13G displayed by `ls`)
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; ended with “Data verification complete.”

### Conversion Performance
- 152 sessions processed in 112.0 s including 13.33 GB pickle serialization.
- This was faster than the <5 min estimate and did not require parallel I/O.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total curated neurons | analysis-dependent subsets | Suite2p cells/deconvolved events | 138,678 `iscell` cells | 138,678 | Yes |
| Mean neurons/session | not stated globally | session-specific | 912.36 | 912.36 | Yes |
| Subjects | 11 in reported analyses | named session dictionaries | 11 | 11 | Yes |
| Sessions | task-specific subsets (for example 77) | release dictionaries | 152 NWBs | 152 | Yes for full release |
| Trials (total) | task-specific | complete numbered traversals | 12,217 IDs; one malformed terminal fragment | 12,216 valid complete trials | Yes after justified exclusion |
| Trials/session (mean) | usually 80, some 100 | session-specific | 80.375 raw IDs | 80.368 valid trials | Yes |
| Neuron range/session | not globally stated | session-specific | 155–2,341 curated | 155–2,341 | Yes |
| Time bin | imaging frame | frame-aligned streams | 64.483627 ms | 64.483627 ms | Yes |
| Input ranges | task design | aligned behavior | env 0–1, native trial IDs 0–99, previous outcome 0–1 | same | Yes |
| Output class ranges | decoder specification | N/A | derivable from behavior | distance 0–6, position/speed 0–4, lick 0–1, zone 0–2, outcome 0–1 | Yes |
| Brain region | hippocampal CA1 | hippocampal 2P | CA1 | CA1 | Yes |

### Spot Checks
- Full verifier found matching neural/input/output dimensions for every retained trial and valid class ranges.
- Conversion logs cover all 152 source paths and report the expected curated-cell and valid-trial counts.
- Both 124 single-series and 28 two-plane sessions converted successfully after applying each RoiResponseSeries DynamicTableRegion mapping.
- All stable, reward-only switch, and combined environment+reward switch scene patterns completed without parser inconsistencies.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output Log Verification
- Read `/app/verification_full_out.txt` in full and searched case-insensitively for errors, warnings, tracebacks, failures, and invalid values.
- Result: no errors or warnings; verifier ended with “Data verification complete.”
- All required keys, dimensions, class ranges, subjects, region indices, and metadata were accepted.

### Check 2: Independent Raw-NWB Sanity Checks
`/app/cache/audit_conversion.py` loads original NWBs directly with `pynwb` and does not call conversion functions. It loaded the finished pickle and used `np.allclose()` for:
1. **Neural**: exact Deconvolved slices after independently resolving each RoiResponseSeries DynamicTableRegion and `iscell`, transposed to cell × time.
2. **Inputs**: independently reconstructed time, environment, trial number, and prior reward outcome.
3. **Outputs**: independently reconstructed signed zone distance classes, absolute-position classes, speed classes, binary lick, zone A/B/C, and event-derived outcome.

Specific passing trials:
| Session case | Trial(s) | Purpose | Result |
|--------------|----------|---------|--------|
| `Env1_LocationB_to_A` | 39, 40 | reward-only switch boundary | exact `np.allclose` pass |
| `Env1_B_to_Env2_C` | 29, 30 | combined environment/reward switch boundary | exact `np.allclose` pass |
| m17 `Env2_LocationB` | 7 | two-plane DynamicTableRegion concatenation | exact `np.allclose` pass, 591×242 neural matrix |

The exhaustive portion checked every converted trial's dimensions, finite values, time vector, and class bounds. It confirmed the malformed terminal fragment is absent and exactly ten sessions use terminal common-length truncation. Final message: `ALL INDEPENDENT AUDITS PASSED`.

### Check 3: Reference Code Comparison
| Processing step | Reference implementation | Conversion implementation | Comparison / rationale |
|-----------------|--------------------------|---------------------------|------------------------|
| (a) Data loading | Preprocessing notebooks call TwoPUtils `create_sess` for scan/VR/Suite2p/behavior | `NWBHDF5IO(...).read()` accesses released aligned processing modules | Same released signals; pynwb is required for NWB and used exclusively. |
| (b) Neuron/trial filtering | Suite2p cells; scientific analyses later select place/RR/TR subsets; trial loops use starts→teleports | `iscell[:,0]`; complete numbered traversals with start, scanning validity, and end-track reach | Matches recording-quality curation while intentionally retaining all scientific cell types and outcomes for general decoding. |
| (c) Temporal alignment | `glmUtils` loops over `zip(trial_starts, teleports)` and slices aligned timeseries/events | contiguous numbered-trial samples aligned to first/start frame | Equivalent native trial segmentation; robust to resampled teleport marker having 0/2 pulses. |
| (d) Binning | Scientific maps use 10 cm spatial bins/smoothing | native fixed 64.4836 ms temporal frames | Required difference: decoder outputs are time-varying speed/lick/position, so spatial aggregation would destroy targets. |
| (e) Input construction | Reference GLM includes position/reward-relative/task variables | exact four task-specified inputs only | Deliberately follows Decoder Task; time is native-frame relative, contextual values repeat across trial. |
| (f) Output construction | Reference analyzes/decodes continuous position and behavior | exact six requested categorical outputs with prescribed boundaries | Required transformation; geometry and outcome sources match reference task semantics. |

### Check 4: Key Statistics Comparison
- Subjects: 11 raw = 11 converted.
- Sessions: 152 NWBs = 152 converted.
- Curated cells: 138,678 raw Suite2p cells = 138,678 converted; 155–2,341/session, mean 912.36.
- Trials: 12,217 raw IDs minus one objectively incomplete 3-frame fragment = 12,216 converted.
- Sampling interval: 64.483627 ms raw timestamps = metadata/value construction.
- Brain region: paper/code hippocampal CA1 = converted `CA1`.
- Input/output ranges and every declared class were verified; full release includes both environments and all three reward zones.
- Paper subset values such as 50/77 accepted k-means sessions are analysis-specific and correctly not applied to the full general decoder.

### Check 5: Edge Cases
- Equality boundaries unit-tested: distance −50/−10/0/+10/+50; position 90/180/270/360 (exact 360 remains class 3); speed 2/10/20/40 (exact 40 remains class 3 because class 4 is strictly >40).
- One malformed m11 terminal trial ID 80 excluded; all 12,216 retained trials have one start pulse, scanning=1, and reach >=440 cm.
- Ten one-extra-neural-frame sessions truncate only to common terminal length.
- 28 two-plane sessions concatenate all curated neurons using each series' linked ROI rows; 124 single-plane sessions follow the same generic path.
- Reward-only switches use trial 40; combined environment+reward switches use the aligned environment transition at trial 30.
- First valid trial previous outcome is 0; later values derive from the preceding retained trial.

### Issues Found and Resolved
1. **Combined switch timing**: Initial parser assumed trial 40 universally. Full conversion assertion exposed trial-30 environment changes. All 11 combined sessions were inspected; parser fixed and revalidated.
2. **Multi-plane ROI mapping**: Initial code applied full-table `iscell` to plane0. Full conversion exposed the mismatch. All 152 RoiResponseSeries layouts were inspected; linked-row filtering plus cross-plane concatenation fixed the issue.
3. After each fix, affected conversion was restarted from scratch, full verification rerun, and all independent checks above passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, from 165.5802 at epoch 1 to 1.2760 at epoch 200.
- Test loss: 1.1214.
- Device: CUDA; full execution completed successfully.
- Train/test split: 9,772 / 2,444 trials.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| distance to reward zone | 0.4213 | 0.3772 | 0.1429 | 2.64× chance validation |
| absolute position | 0.5237 | 0.5000 | 0.2000 | 2.50× chance validation |
| speed | 0.4369 | 0.4084 | 0.2000 | 2.04× chance validation |
| lick | 0.6427 | 0.6318 | 0.5000 | above chance |
| reward zone location | 0.7826 | 0.7375 | 0.3333 | 2.21× chance validation |
| reward outcome | 0.5509 | 0.5076 | 0.5000 | slightly above chance; investigated in Step 12 |

All validation outputs are above uniform chance. Train/validation values are close, with no >1.5× gap. Sample and prediction plots were generated by `--plot-samples`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Train/Validation Ratio | Expectation from Paper |
|----------|---------------------|--------|-----------------|------------------------|------------------------|
| distance to reward zone | 0.3772 | 0.1429 | 2.64× | 1.12× | Paper decodes continuous reward-relative position with ridge scores/shuffle controls; no matching categorical accuracy. |
| absolute position | 0.5000 | 0.2000 | 2.50× | 1.05× | Place coding predicts strong above-chance performance; no matching categorical accuracy reported. |
| speed | 0.4084 | 0.2000 | 2.04× | 1.07× | Speed is represented in GLM/behavior analyses; no matching classification accuracy. |
| lick | 0.6318 | 0.5000 | 1.26× | 1.02× | Instantaneous lick is sparse/noisy; paper analyzes lick maps/ratios rather than binary decoding accuracy. |
| reward zone location | 0.7375 | 0.3333 | 2.21× | 1.06× | Strong reward-relative coding is expected; no matching three-class accuracy. |
| reward outcome | 0.5076 | 0.5000 | 1.02× | 1.09× | Reward omissions are externally imposed and paper compares rewarded/omission activity rather than decoding trial outcome. |

### Check 1: Accuracy vs Chance
- Every output exceeds uniform chance.
- Distance, position, speed, and zone exceed 1.5× chance.
- Lick (1.26×) and outcome (1.02×) were investigated in detail rather than dismissed.
- Lick is a sparse instantaneous event (sample data ~13% positive frames) and native values are sensor counts 0–8. Independent raw checks verify exact `(raw lick > 0)` conversion and temporal alignment. Above-chance 0.6318 is plausible for noisy frame-level licking.
- Outcome is per-trial and highly imbalanced: 1,874 omitted and 10,342 rewarded trials (15.34%/84.66%). Because omission is externally imposed and the output repeats across the entire trial—including frames before reward delivery—much neural activity cannot causally identify the later outcome. Near-chance balanced accuracy is scientifically plausible.

### Check 2: Accuracy Comparison to Paper
- Searched all paper/methods occurrences of decoder, decoding, accuracy, ridge regression, cross-validation, R², and coefficient of determination.
- The paper's principal decoder is continuous reward-relative position, evaluated with regression/cross-validation and shuffle/control comparisons. It does not provide categorical balanced accuracies for the six variables in this task.
- Therefore no like-for-like numerical paper accuracy can be tabulated without misrepresenting metrics. Qualitatively, strong distance/position/zone decoding agrees with the paper's hippocampal spatial and reward-relative coding results.

### Check 3: Train vs Validation Gap
- No output has a train/validation ratio above 1.5; ratios are 1.02–1.12.
- There is no evidence of major overfitting or leakage. Validation outcome being only slightly above chance is not accompanied by inflated training accuracy (0.5509), supporting intrinsic task difficulty rather than a split bug.

### Low-Accuracy Debugging Performed
1. **Raw labels on specific trials**: Independently loaded three sessions with pynwb and checked rewarded and omitted examples (six trials total). Every raw Reward-event assignment exactly equaled converted outcome; outputs were constant per trial.
2. **Temporal alignment**: Created `cache/outcome_alignment_review.png` with population activity and position classes on the same trial-relative axis. Neural/input/output lengths matched exactly for all checked trials.
3. **Variation**: Outcome has 1,874 omitted trials—far from a 99% single-class failure. Lick also has meaningful variation.
4. **Filtering**: Confirmed Suite2p `iscell` filtering, all-plane concatenation, complete-trial selection, and native-frame alignment.
5. **Reference match**: Reward events and omission definitions match reference analyses; no alternative raw outcome variable is appropriate (`autoreward` is invariant zero).

### Issues Found and Resolved
- No new conversion issue was found in Step 12. Low lick/outcome scores are supported by exact raw-data checks, intrinsic sparsity/timing, and absence of a train-validation gap.
- Earlier combined-switch and multi-plane issues remained fixed under full training and independent review.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading instructions, format, processing, statistics, and decoder results
- [x] cache/ folder created
- [x] Investigation scripts and review artifacts moved to cache and documented in `cache/README_CACHE.md`
- [x] Production outputs, scripts, logs, notes, and processing plots retained at top level

# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

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
| `create_sess` | `src/reward_relative/preprocessing.py` | LOADING | Constructs a synchronized session by loading scan metadata, VR, Suite2p output, and behavior through TwoPUtils. |
| session pickle utilities | `src/reward_relative/utilities.py` | LOADING | Read/write processed `sess` objects used by analyses. |
| behavioral trial/reward helpers | `src/reward_relative/behavior.py` | PROCESSING | Operate on synchronized VR fields including `pos`, `lick`, `dz`, trials, outcomes, and reward zones. |
| place-cell / spatial-information routines | `src/reward_relative/spatial.py` | CURATION | Build spatially binned trial matrices and identify place cells relative to shuffled spatial information. |
| `multi_anim_sess` workflow | README and notebooks | PROCESSING | Adds dF/F, spatial trial matrices, place-cell results, and trial-set dictionaries to session data. |

### Notes
- The repository is the Sosa et al. 2024 `reward_relative` package for hippocampal two-photon imaging.
- README processing hierarchy: raw `sess` stores fluorescence synchronized to VR and does **not** inherently contain dF/F; dF/F and spatial trial matrices can be added. `multi_anim_sess` computes dF/F, place cells, and trial sets. `dayData` performs later population analyses and retains copies of trial matrices.
- Session construction loads scan information, VR, Suite2p, and behavior; multiplane frame count is based on scanner frames divided by number of planes.
- Native synchronized VR variables used by code include position (`sess.vr_data['pos']`), lick (`lick`), and movement variable `dz`; code also handles trial indices, reward delivery/outcome, environment/scene, and reward-zone boundaries.
- Fluorescence requires dF/F computation when starting from raw `sess`; whether supplied data already include processed dF/F must be determined in Step 2. Suite2p-derived cell masks and fluorescence are the relevant neural sources.
- Place-cell and reward-relative classifications rely on random shuffles (100 per cell noted by README) and can vary slightly. These are downstream scientific classifications, not evidence for excluding every non-place cell from a general neural decoder; exact curation will be reconciled with supplied data and methods.
- Spatial trial matrices in the paper code are position-binned. The requested decoder instead requires temporal alignment to trial start, so synchronized frame-level streams must be used while retaining reference signal processing where applicable.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 152 NWB 2.8/HDF5 files plus Dandiset YAML metadata (Dandiset 001361), arranged as one directory per subject and one `*_behavior+ophys.nwb` file per session.
- There are 11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, and m19. m11 has 12 supplied sessions (03-14); every other subject has 14 (01-14).
- `processing/behavior/BehavioralTimeSeries` contains synchronized frame-level position, speed, lick, autoreward, environment, scanning, teleport, trial number, and trial_start. `Reward` differs: it is a sparse event TimeSeries with event timestamps, not a frame-length vector.
- No `intervals/trials` table is present. Complete trials are identified by `trial_start == 1` (entry to track) and paired `teleport == 1`; the trial-number stream has an initial `-1` sentinel and valid trial labels thereafter.
- `processing/ophys` provides Suite2p `Fluorescence`, `Neuropil`, and `Deconvolved` matrices, each stored as time x ROI. ImageSegmentation/PlaneSegmentation provides ROI metadata, `iscell`, and `planeIdx`.
- 28 sessions have two imaging planes; 124 have one. Neural metadata rates are 15.5078125 Hz for single-plane data and 31.015625 Hz on plane series in multiplane files, while synchronized behavior rows correspond to volume-level samples. Row matching is therefore the reliable native synchronization. Ten files have neural matrices exactly one terminal row longer than behavioral streams; this edge case requires trimming to the shared length.
- Native `reward_zone` is constant 0 in all frame streams despite session identifiers encoding LocationA/B/C and transitions. This is treated as a source-field limitation to reconcile with identifier, reward positions, code, and methods in Steps 3-5.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 Suite2p `iscell` ROIs summed across sessions |
| Neurons / session | 155-2,341 accepted cells; mean 912.36; median 921.5 |
| Subjects | 11 |
| Sessions / subject | 12 for m11; 14 for each other subject; 152 total |
| Trials (total) | 12,216 complete `trial_start`/`teleport` trials |
| Trials / session | 41-100; mean 80.37; median 80 |
| All segmented ROIs | 312,110 total; 315-5,085/session |
| Frame/volume samples | 3,610,867 total; 14,164-51,520/session |
| Imaging planes | 180 session-planes (124 one-plane, 28 two-plane sessions) |

### Available Variables and Native Statistics
- Position and speed are float time series; lick, environment, autoreward, scanning, teleport, trial number, and trial_start are synchronized behavioral series.
- Environment across complete trials: ENV1/code 0 = 6,226; ENV2/code 1 = 5,990.
- Sparse reward events assigned by timestamps to complete trials give 10,342 rewarded and 1,874 omitted trials (84.66% rewarded).
- Every session has at least 41 complete trials, satisfying the downstream minimum of two.
- Session identifiers include date and scene/reward-location labels (for example `Env1_LocationB_to_A`), useful for resolving per-trial zone identity where the NWB reward-zone stream is uninformative.
- Neural and behavior streams are already synchronized in exported NWB row order. Ten one-row terminal mismatches were explicitly identified and will be handled by truncation to common length, not interpolation.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Source text reports analysis-specific cell counts rather than one universal raw total | Counts vary by place-cell/RR/TR analysis and are not a raw-data curation target. |
| Neurons / session | Not specified as a single expected value | Paper reports simultaneously imaged population analyses. |
| Subjects | 11 switch-task mice (plus 3 fixed-condition mice not represented in supplied switch dataset) | “switch task (n = 11 mice) versus ... fixed-condition ... (n = 3).” |
| Sessions / subject | Experimental schedule spans repeated environment/reward-location sessions | Supplied data determine exact available count. |
| Trials (total) | Not stated as one universal total | Analyses use trial identities and rewarded/omission trial strata. |
| Trials / session | Not stated as one universal number | Sessions contain trials before and after reward switches. |
| Neural data time bin | ~15.5 Hz (~64.5 ms) | “All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.” |
| Behavior data time bin | ~15.5 Hz (~64.5 ms) | Same quote. |
| Reward rate | Predominantly rewarded with explicit omission trials | GLM split stratified rewarded trials before/after switches and all omission trials. |
| Track length | 450 cm | Linear track position is “from 0 to 450 cm.” |
| Spatial analysis bin | 10 cm | Position expanded into 45 bases; clustering used 10 cm linear-position bins. |
| Switch-task subjects | 11 | Methods sample-size statement. |

### Processing Details
- Imaging ROIs and traces were produced with Suite2p. Reference dF/F processing subtracts neuropil (default coefficient 0.7), uses a maximin baseline, computes dF/F as `(F - baseline) / abs(baseline)`, and smooths with a two-sample (~0.129 s SD) Gaussian.
- dF/F is computed separately within individual trials to account for photobleaching and avoid teleport periods during which laser power could be reduced.
- Activity is deconvolved from dF/F with a canonical calcium kernel using OASIS/Suite2p. The paper explicitly says this is not interpreted as spike rate, but removes asymmetric calcium-indicator smoothing.
- The paper's encoding model used deconvolved activity as samples x neurons and behavioral/neural streams at ~15.5 Hz. Predictors included absolute position, reward-relative position, rewarded state, speed, acceleration, and smoothed binary lick counts.
- Position-binned analyses used 10 cm bins and, for factorized k-means, a 10 cm SD Gaussian plus per-neuron min-max scaling. Those transforms are analysis-specific and not appropriate for the requested time-aligned decoder.
- Trial identities grouped train/test data (85% training, 15% held-out test, with rewarded pre/post-switch and omissions stratified).

### Curation Steps

**Neuron curation rules**:
- Suite2p `iscell` defines accepted cellular ROIs in the supplied NWB files.
- Place-cell, track-relative, and reward-relative labels are downstream functional classifications based partly on shuffles; they are not general imaging-quality filters and should not restrict a decoder intended to use population activity.
- FDE > 0.15 was only used for the paper's model-ablation interpretation, not initial neural data inclusion.

**Trial curation rules**:
- Use complete track traversals bounded by entry (`trial_start`) and teleport, excluding pre-track sentinel and teleport/intertrial samples.
- Preserve rewarded and omission trials; both are scientifically central and explicitly stratified in the reference GLM.
- Analysis-specific session filters (for example 50/77 sessions whose k=2 clustering exceeded shuffle) apply only to remapping analyses and should not remove sessions for this general decoder.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| No directly comparable neural-to-behavior categorical decoder | Not reported |
| Paper Poisson GLM predicts neural events from task/movement variables | FDE 0.10±0.19 all place cells; 0.32±0.13 TR; 0.29±0.11 RR; 0.29±0.11 non-RR remapping cells |

The paper GLM is an encoding model in the opposite direction from this task, so its FDE values are contextual rather than target decoder accuracies.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | Workflow computes dF/F and OASIS events; analyses commonly use events | NWB supplies Fluorescence, Neuropil, and `Deconvolved` | Encoding/clustering response is deconvolved activity | Use supplied `Deconvolved` activity, avoiding an irreproducible second deconvolution; select `iscell`. |
| Sampling | Frame rate adjusted for imaging planes | 15.5078125 metadata for single-plane; 31.015625 per-plane metadata in multiplane files, but rows are volume-synchronized to behavior | All behavior and neural streams sampled at ~15.5 Hz | Treat each synchronized row as one 64.4836 ms volume bin; concatenate accepted ROIs across planes by row. |
| Trial definition | `trial_start_inds` and `teleport_inds`; trial-specific dF/F | Frame pulse streams exist; trial-number has initial -1 sentinel | Trials are laps/traversals; teleport excluded | Slice from each `trial_start` pulse through sample before paired teleport. Exclude sentinel/pre-track and teleport samples. |
| Reward events | Trial types derive from reward-delivery events | `Reward` is sparse timestamped events, unlike frame streams | Rewarded and omission trials are both retained | Assign sparse reward timestamps to trial time intervals; any event means rewarded. |
| Reward-zone identity | `get_reward_zones` derives zones from reward-location values/positions and labels | Exported `reward_zone` frame field is constant 0; identifier encodes A/B/C and switches | Three reward locations and switch sequences are central | Infer per-trial zone from session identifier plus reward-position transitions, validate against event positions; never treat constant field as all-A. |
| Cell filtering | Functional place/RR/TR labels used for particular analyses | `iscell` is available for all ROIs | Paper reports functional subsets for specific questions | Apply `iscell` only; do not restrict to functional subsets. |
| Session filtering | Some remapping analyses retain 50/77 significant k=2 sessions | All 152 supplied sessions have >=41 complete trials | Filter is analysis-specific | Retain all sessions for requested decoder. |
| Terminal lengths | Assumes synchronized streams | Ten sessions have neural arrays one sample longer | No conflicting statement | Trim only terminal excess neural row to common behavior length; trial windows are unaffected. |
| Position analysis | Reference spatial analyses often use 10 cm bins | Continuous synchronized position is available | GLM used 0-450 cm and temporal samples | Keep native temporal samples; discretize outputs exactly per task rather than spatially averaging. |

### Final Understanding
- Supplied NWBs are already processed and synchronized exports of the reference workflow. The expert-consistent neural representation is accepted-cell Suite2p/OASIS deconvolved calcium activity at ~15.5 Hz.
- Trial-start pulse alignment and native temporal rows satisfy the new decoder requirement; paper-specific position-binned transforms are intentionally not applied.
- Behavioral outputs come from synchronized native streams, except sparse rewards are timestamp-assigned and reward-zone identity must be reconstructed from identifiers/position evidence because the exported field is uninformative.
- All 11 switch-task subjects, 152 sessions, complete trials, and accepted cells are retained unless a source edge case prevents a valid mapping.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane*/data` + `PlaneSegmentation/iscell`, `planeIdx` | `neural` | Select accepted cells per plane, transpose time×cell to cell×time, concatenate planes, slice complete trials | `dff`, multi-session workflow | Supplied OASIS deconvolved calcium events; float32. |
| behavior timestamps relative to `trial_start` pulse | `input[0]` | `(timestamp - timestamp[start])` in seconds | trial-index logic | Time-varying continuous. |
| behavior `environment` | `input[1]` | Trial mode, map native 0/1 to ENV1/ENV2, repeat across trial | behavioral helpers | Time-varying matrix row but constant per trial. |
| native trial label | `input[2]` | Valid trial number as continuous float, repeat across trial | trial logic | Preserve source 0-based number, excluding -1 sentinel. |
| preceding complete trial reward event | `input[3]` | Previous outcome; first trial = 0 because no previous within-session trial | `get_trial_types` | Binary and repeated across trial. |
| position + per-trial reward-zone interval | `output[0]` distance to reward zone | Signed distance: `pos-start` before zone, 0 inside inclusive interval, `pos-stop` after; discretize with task thresholds | `get_reward_zones` | Zone interval/label follows scene and switch trial; validate from reward positions. |
| behavior `position` | `output[1]` absolute position | bins `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360` | native synchronized position | Clip no values; apply exact inequalities. |
| behavior `speed` | `output[2]` speed | bins `<2`, `[2,10)`, `[10,20)`, `[20,40]`, `>40` cm/s | native synchronized speed | Keep native speed, no paper GLM smoothing because requested categorical instantaneous speed. |
| behavior `lick` | `output[3]` lick | `lick > 0` -> 1, else 0 | native lick series | Binary per frame. |
| scene/zone sequence + reward positions | `output[4]` reward zone location | A=0, B=1, C=2, scalar per trial | `get_reward_zones` | Exported `reward_zone` stream is uninformative; use reference scene logic and validate nearest centers ~80/200/320 cm. |
| sparse `Reward` timestamps | `output[5]` reward outcome | 1 iff any reward event timestamp falls from trial start through paired teleport, else 0 | `get_trial_types` | Scalar per trial. |
| NWB subject ID | `subjects`, `subject_idx` | Unique m3...m19 IDs and session index mapping | NWB metadata | 11 switch-task mice. |
| hippocampal imaging metadata/paper | `brain_regions`, `brain_region_idx` | `['CA1']`; all accepted cells index 0 | paper experiment | Dorsal CA1 two-photon recordings. |

### Exact Output Categories
- Distance: 0 `< -50`; 1 `[-50,-10)`; 2 `[-10,0)`; 3 exactly `0`; 4 `(0,10]`; 5 `(10,50]`; 6 `>50` cm.
- Position: 0 `<90`; 1 `[90,180)`; 2 `[180,270)`; 3 `[270,360]`; 4 `>360` cm.
- Speed: 0 `<2`; 1 `[2,10)`; 2 `[10,20)`; 3 `[20,40]`; 4 `>40` cm/s.
- Lick: 0 no lick; 1 lick. Zone: 0 A, 1 B, 2 C. Outcome: 0 omitted, 1 rewarded.

### Key Decisions
1. **Temporal bins**: Preserve every native synchronized ~15.5 Hz row (64.4836 ms) rather than position-bin averaging. This matches the paper's temporal GLM streams and requested trial-start temporal alignment.
2. **Trial window**: Include the `trial_start` sample and stop before the paired teleport sample. This yields track traversal only and excludes teleport/intertrial laser artifacts as in reference dF/F processing.
3. **Neural representation**: Use supplied deconvolved events, not raw F or a second dF/F/deconvolution pass. The NWB is explicitly processed Suite2p output and paper models use deconvolved activity.
4. **Neuron curation**: Retain `iscell==1` only; concatenate accepted ROIs across planes. Do not impose place/RR/TR labels or analysis-specific session filters.
5. **Reward zones**: Follow reference `get_reward_zones` scene logic and switch point. Cross-check labels against sparse reward-event positions nearest A/B/C clusters. For omission trials, inherit the scene-defined pre/post-switch zone rather than infer from absent events.
6. **Outcomes**: Assign sparse Reward events by timestamps. Previous outcome is within-session only; first retained trial is 0 (omitted/no known prior), as required binary without an unknown category.
7. **Length edge case**: Truncate the ten one-row-long neural streams at the terminal end to behavior length before trial slicing; never shift/interpolate.
8. **Storage**: neural float32; input float32; output int64. Per-trial scalars (zone/outcome) are stored as a 6-row output matrix repeated through time for uniform validation/training, while semantically per-trial.
9. **Sessions/trials**: Retain all 152 sessions and 12,216 complete pulse-defined trials; all sessions exceed two trials.

### Planned Sanity Checks
- [ ] Directly load raw NWB and use `np.allclose` to compare one converted neural trial against selected/transposed raw Deconvolved rows.
- [ ] Directly compare converted time, environment, trial number, and previous outcome for at least three trials to raw timestamps/series/events with `np.allclose`.
- [ ] Directly compare converted position, speed, lick, zone, and outcome categories for at least three trials to independently computed raw NWB values with `np.allclose`.
- [ ] Confirm every trial starts at time zero, all row lengths match, and no teleport sample is included.
- [ ] Confirm total subjects/sessions/trials/cells equal 11/152/12,216/138,678 before per-session accepted-cell counting and all sessions have >=2 trials.
- [ ] Confirm reward distribution is 10,342/1,874 rewarded/omitted and environment counts are 6,226/5,990.
- [ ] Confirm zone labels agree with reward-event position cluster on every rewarded trial except documented boundary/noise cases.
- [ ] Plot raw continuous variables with categorical overlays and trial boundaries for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required positional output path, mutually exclusive `--full`/`--sample`, and `--show-processing` modes.
- Implemented reference scene-to-zone mapping with exact intervals A=[80,130], B=[200,250], C=[320,370] cm and switch trial 30.
- Implemented sparse reward-event assignment, complete trial pairing, accepted-cell extraction across planes, exact categorical discretizers, per-trial validation, metadata, timing logs, and diagnostic plots.
- Script compiles and help output is valid. A two-session smoke run converted 160 trials/323 cells in 0.24 s and saved a 20.5 MB pickle without errors.

Code inefficiencies identified:
- Loading full raw fluorescence and neuropil would waste memory; only selected columns of Deconvolved are read.
- Retaining session-scale time×cell matrices after trial slicing would duplicate memory unnecessarily.

Code speedups added:
- Read monotonic accepted-cell columns directly from HDF5 and cast to float32.
- Process one session at a time and use vectorized discretization and trial slicing.
- Avoid recomputing dF/F/OASIS because processed deconvolved activity is supplied.

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
| Trial length | 137-476 bins; session means 178.5 and 182.7 bins |
| Time input range | 0-30.6 s |
| Environment range | 0 only in selected sample |
| Trial number range | 0-79 |
| Previous outcome range | 0-1 |
| Distance-bin fractions | [0.088, 0.092, 0.058, 0.285, 0.022, 0.071, 0.384] |
| Position-bin fractions | [0.312, 0.207, 0.203, 0.146, 0.133] |
| Speed-bin fractions | [0.121, 0.062, 0.066, 0.233, 0.518] |
| Lick fractions | [0.820, 0.180] |
| Zone fractions (time-weighted) | A 0.792, B 0.208 |
| Outcome fractions (time-weighted) | omitted 0.161, rewarded 0.839 |

### Processing Plots Review
- Created one diagnostic PNG for each of the two sample sessions. Files are valid nonempty 1820x1400 RGB images.
- Plots overlay continuous position, signed zone distance, speed, categorical distance bins, and lick samples across the first six trial-start-aligned trials.
- Trial boundaries begin at track entry; signed distance is exactly zero over zone intervals and category transitions follow the requested thresholds. No temporal discontinuity beyond expected trial concatenation was detected.

### Format Validation
- `/app/train_decoder.py --verify-only` completed with “Data verification complete.”
- Errors: none. Warnings: none.
- All six output dimensions have valid categorical ranges and matching `output_values`; all trial neural/input/output time lengths match.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| Selected-column HDF5 reads, float32, vectorized bins, no redundant deconvolution | Smoke/formal sample processing is sub-second for two sessions |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| Conversion computation | ~0.1-0.2 s in sample | <1 minute nominal; conservatively <5 minutes including large sessions and pickle I/O |
| Sample serialization | ~0.03 s for 20.5 MB | Expected several seconds to a few minutes depending on full size |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Distance to reward zone | 0.4216 | 0.3372 |
| Absolute position | 0.5594 | 0.4757 |
| Speed | 0.3536 | 0.3058 |
| Lick | 0.7298 | 0.6728 |
| Reward zone location | 0.9644 | 0.9588 |
| Reward outcome | 0.6484 | 0.6411 |

- Loss decreased from approximately 51 at initialization to 0.8973 at epoch 200; test loss was 0.9962.
- Every validation balanced accuracy is above uniform chance (respectively 0.1429, 0.2, 0.2, 0.5, 0.3333, 0.5).
- Distance, position, and zone are especially strong indicators that neural and behavioral streams are temporally and trial-wise aligned. Speed exceeds chance; lick and outcome also exceed chance despite their noisier behavioral/trial-level nature.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9.842 GB (9.2 GiB)
- `verification_full_out.txt`: created; ended with “Data verification complete.”
- Conversion runtime: 67.90 s computation + 7.15 s serialization, well below the 15-minute threshold.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total accepted cells | Analysis-specific subsets | Suite2p `iscell` | 138,678 | 138,678 | Yes |
| Mean cells/session | Not universal | `iscell` | 912.36 | 912.36 | Yes |
| Subjects | 11 switch-task mice | session dictionaries | 11 | 11 | Yes |
| Sessions | Repeated sessions | supplied session set | 152 | 152 | Yes |
| Trials (total) | Complete traversals | start-to-teleport | 12,216 | 12,216 | Yes |
| Trials/session | Not fixed | pulse-defined | 41-100, mean 80.37 | 41-100, mean 80.37 | Yes |
| Outcome distribution | Rewarded + omission retained | any reward in trial | [0.153405, 0.846595] | [0.153405, 0.846595] | Yes |
| Environment trials | ENV1/ENV2 | morph/environment | [6,226, 5,990] | same source values | Yes |
| Zone trial distribution | Three locations | A/B/C scene logic | [4,186, 4,010, 4,020] | [4,186, 4,010, 4,020] | Yes |
| Distance distribution | task-specific new output | reference intervals | [0.253186,0.101537,0.073673,0.237271,0.020540,0.071536,0.242257] | same | Yes |
| Position distribution | 0-450 cm | native position | [0.210748,0.177737,0.230974,0.226522,0.154019] | same | Yes |
| Speed distribution | native cm/s | native speed | [0.122129,0.088412,0.133794,0.317502,0.338162] | same | Yes |
| Lick distribution | binary frame licks | native lick | [0.769560,0.230440] | same | Yes |

### Spot Checks
- Inspected sessions 0, 1, 75, 150, and 151, spanning subjects, fixed/switch scenes, environment switches, and dataset endpoints.
- Every checked trial had neural `(cells,time)`, input `(4,time)`, and output `(6,time)` with identical time length.
- Subject/session distribution exactly matches raw data (m11=12 sessions; each other mouse=14).
- Full verifier reported no errors or warnings. All classes are represented globally and metadata/categories are valid.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` begins “Data format is valid, no errors or warnings” and ends “Data verification complete.” No warning requires exemption.
2. **Independent raw-data `np.allclose` checks**: `/app/review_raw_vs_converted.py` loads original NWBs directly and independently reconstructs neural, input, and output data. Exact/near-exact comparisons passed for six trials: session 0 trial 0; switch trials 29 and 30; a multiplane environment-switch session; a source session with a one-row neural excess; and the final trial of the final session. Neural and categorical outputs passed with zero tolerance; timestamp-derived input passed at `atol=1e-6`.
3. **Reference-code comparison**:
   - Loading: reference `create_sess` loads synchronized Suite2p+VR; converter reads the corresponding NWB ophys and BehavioralTimeSeries exports.
   - Neuron filtering: reference/Suite2p uses accepted cells; converter applies `iscell`, preserving plane order via `planeIdx`. Functional place/RR/TR subsets are not quality filters.
   - Trial filtering/alignment: reference dF/F uses `trial_start_inds` to `teleport_inds` and avoids teleport periods; converter includes start and excludes teleport, aligned to time zero.
   - Neural processing: paper analyses use OASIS deconvolved calcium activity; converter uses supplied `Deconvolved` without duplicate processing.
   - Binning: reference temporal GLM uses ~15.5 Hz synchronized rows; converter preserves these rows. Requested categorical thresholds replace paper-specific 10 cm spatial averaging.
   - Input/output construction: native timestamps/environment/trial labels/behavior streams and sparse reward timestamps are used. Reward zones exactly reproduce `get_reward_zones`: A=[80,130], B=[200,250], C=[320,370], switch at trial 30.
4. **Key-statistics comparison**: Raw and converted counts match exactly for 11 animals, 152 sessions, 12,216 trials, 138,678 accepted cells, environment counts 6,226/5,990, outcomes 1,874/10,342, and zones 4,186/4,010/4,020. All output distributions are documented in Step 9.
5. **Edge cases**: Independent threshold tests passed at every distance/position/speed boundary. Global audit found zero empty trials, shape mismatches, nonzero trial starts, nonconstant per-trial fields, invalid first-trial previous outcomes, or out-of-range categories. Ten one-row neural excesses are safely terminal-trimmed with an exact raw match in a tested affected session.

### Issues Found and Resolved
- **Sparse Reward schema**: Reward is event-timestamped rather than frame-aligned. Fixed by assigning events to start-through-teleport intervals; exact outcome totals match raw events.
- **Constant exported reward-zone stream**: It cannot encode A/B/C. Fixed using exact reference scene logic and intervals. Reward positions validated the assigned class on 10,337/10,342 rewarded trials (99.9517%). Five late-delivery samples (positions ~140, 260-273 cm) were nearest the next center; scene-defined zone remains authoritative and these represent delayed delivery/frame sampling, not label errors.
- **Multiplane rate metadata**: Per-plane metadata reports 31.015625 Hz while rows and behavior are volume-synchronized at ~15.5 Hz. Resolved by native row alignment, consistent with the paper's statement that all neural/behavioral streams are sampled at ~15.5 Hz.
- **Terminal one-row mismatch**: Ten sessions have one extra terminal neural row. Truncating only the terminal excess preserves all trial windows and passed exact comparison.
- **No conversion defects remained after re-check**: all independent tests and global audits passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Loss fell from 182.6331 at epoch 1 to 1.2408 at epoch 200; test loss was 1.1054.
- Full execution completed on CUDA with 9,772 training and 2,444 validation trials.
- Four sessions with >2,000 cells used the decoder's built-in random projection for SVD initialization; training completed normally.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Distance to reward zone | 0.3825 | 0.3473 | 2.43x uniform chance |
| Absolute position | 0.5543 | 0.5293 | 2.65x uniform chance |
| Speed | 0.4140 | 0.3916 | 1.96x uniform chance |
| Lick | 0.6621 | 0.6514 | 1.30x uniform chance |
| Reward zone location | 0.8440 | 0.7991 | 2.40x uniform chance |
| Reward outcome | 0.6048 | 0.5153 | 1.03x uniform chance |

- Every validation balanced accuracy exceeded chance.
- Train/validation ratios are 1.10, 1.05, 1.06, 1.02, 1.06, and 1.17 respectively; none approaches the >1.5 overfitting criterion.
- `--plot-samples` completed and the script ended with “train_decoder.py finished successfully.”

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Uniform Chance | Accuracy / Chance | Expectation from Paper |
|---|---:|---:|---:|---|
| Distance to reward zone | 0.3473 | 0.1429 | 2.431x | No comparable categorical decoder reported |
| Absolute position | 0.5293 | 0.2000 | 2.646x | No comparable categorical decoder reported |
| Speed | 0.3916 | 0.2000 | 1.958x | No comparable categorical decoder reported |
| Lick | 0.6514 | 0.5000 | 1.303x | No comparable categorical decoder reported |
| Reward zone location | 0.7991 | 0.3333 | 2.397x | No comparable categorical decoder reported |
| Reward outcome | 0.5153 | 0.5000 | 1.031x | No comparable categorical decoder reported |

### Accuracy vs Paper
- The paper does not report neural-to-behavior categorical decoding accuracy for these six requested variables.
- The only related reported predictive model is in the opposite direction: a Poisson GLM predicts deconvolved neural activity from task/movement variables. Reported FDE is 0.10±0.19 for all place cells, 0.32±0.13 for TR cells, 0.29±0.11 for RR cells, and 0.29±0.11 for non-RR remapping cells. FDE is not classification accuracy and cannot be numerically equated to balanced accuracy.
- Qualitatively, strong position/reward-zone decoding agrees with the paper's central finding that CA1 activity carries track-relative and reward-relative information.

### Train vs Validation Gap
- Train/validation ratios: distance 1.101, position 1.047, speed 1.057, lick 1.016, zone 1.056, outcome 1.174.
- No output exceeds the specified 1.5x overfitting threshold. Validation performance tracks training performance closely.

### Low-Accuracy Investigation
- Lick (1.303x chance) and reward outcome (1.031x chance) were investigated explicitly.
- Original NWBs were loaded directly for three trials covering an omitted/lick-poor trial and two rewarded/lick-rich trials in early and final sessions. `np.allclose` passed for complete lick vectors and outcome labels; trial lengths and zero-time alignment also matched.
- Outcome has sufficient variation: 1,874 omitted and 10,342 rewarded trials. Lick fraction ranges 0-1 across trials, median 0.211; 99.77% of trials contain at least one lick.
- Reward outcome is a per-trial future event not known at trial start, and omission trials share zone/context with rewarded trials; near-chance decoding is scientifically plausible. The exact raw-label checks, high reward-zone accuracy, and strong time-varying decoding rule out a global alignment error.
- Licking is sparse and behaviorally variable at frame scale; nevertheless its balanced accuracy is robustly above chance and has virtually no train-validation gap.

### Issues Found and Resolved
- No conversion defect was found in Step 12. All outputs exceed chance; labels, temporal alignment, variation, filtering, and reference processing were rechecked.
- No rerun was required because the investigation confirmed the conversion rather than revealing a mismatch.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading instructions, format, processing, statistics, and reproduction commands.
- [x] cache/ folder created.
- [x] Investigation script moved to `cache/` and documented in `cache/README_CACHE.md`.
- [x] All required conversion, sample/full validation, and training outputs retained in `/app`.
- [x] CONVERSION_NOTES.md completed with all decisions, checks, statistics, and decoder results.


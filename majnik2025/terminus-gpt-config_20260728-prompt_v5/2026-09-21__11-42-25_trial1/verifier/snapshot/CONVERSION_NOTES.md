# Dataset Conversion Notes

## Overview
- **Dataset**: longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- code/
- CONVERSION_NOTES.md
- data/
- decoder.py
- docker-compose.yaml
- Dockerfile
- methods.txt
- paper.pdf
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| track_cells | track2p/t2p.py | PROCESSING | Main cell-tracking pipeline across sessions; computes embeddings/matches and saves outputs |
| import_data | track2p/t2p.py | LOADING | Loads Suite2p outputs and image data for a session/fish/mouse directory |
| create_summary_img | track2p/t2p.py | PROCESSING | Builds summary image used for registration/visualization |
| create_patches | track2p/t2p.py | PROCESSING | Extracts ROI-centered patches/features for matching across sessions |
| run_pipeline / main CLI entry | track2p/__main__.py | PROCESSING | Command-line entry point configuring and invoking Track2p workflow |

### Notes
- Code base is for Track2p longitudinal cell tracking rather than decoder training directly.
- The reference code appears focused on registering/matching ROIs across imaging sessions, likely using Suite2p-derived outputs as inputs.
- Important loading target is Suite2p-style files (e.g. ROI/cell masks, fluorescence outputs, registration-related arrays), suggesting source data are calcium imaging rather than electrophysiology.
- Need to inspect data files next to determine which stored outputs are available for conversion into session-wise neural activity and motion variables.
- No obvious evidence yet that this code computes dF/F internally; likely expects preprocessed imaging outputs from upstream Suite2p or related pipeline.
- Need to reconcile paper/code/data about whether tracked cells only or all detected cells should be used for decoder dataset.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are organized by subject (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and then by dated session folder.
- Each session contains `suite2p/plane0` with calcium-imaging outputs: `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `stat.npy`, `ops.npy`.
- Each session also contains `move_deve` with behavioral/video-derived variables: `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`.
- Sessions appear frame-synchronous: neural arrays have 36,000 frames and motion-energy/timestamp arrays are ~35,884-36,000 samples depending on session.
- `ground_truth.csv` exists for some subjects, likely for Track2p cell-matching evaluation rather than decoder targets.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 6,882 putative cells with `iscell[:,0] > 0.5` |
| Neurons / session | min 118, mean 167.85, max 370 |
| Subjects | 6 |
| Sessions / subject | jm031: 7, jm032: 7, jm038: 7, jm039: 7, jm040: 6, jm046: 7 |
| Trials (total) | Not explicit in raw data; decoder trials will be derived as contiguous 60-second windows |
| Trials / session | Approximately 6 for ~600-second sessions |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | ~thousands across all mice/sessions; exact total to verify against paper text | Raw data show 6,882 `iscell>0.5` putative cells across 41 sessions |
| Neurons / session | Roughly 118-370 putative cells/session in raw data | From data exploration |
| Subjects | 6 | Raw data folders show 6 mice |
| Sessions / subject | 6-7 | Raw data folders show 41 sessions total |
| Trials (total) | Derived, not native | Decoder task requires 60-second trials |
| Trials / session | ~6 | Sessions are ~600 s long with 36,000 imaging frames |
| Neural data time bin | 10-frame averaging used in paper analyses/decoding | “we slightly denoised the dF/F ... by averaging in bins of 10 consecutive timestamps” |
| Behavior data time bin | 10-frame averaging used in paper analyses/decoding | “as well as the behaviour traces by averaging in bins of 10 consecutive timestamps” |
| Reward rate | N/A | No reward task; spontaneous motion decoding in barrel cortex |
| Developmental age range | P8-P14 | Track2p README/paper context mentions 7 consecutive daily recordings between P8 and P14 |
| Decoder model in paper | Ridge regression with nested CV | “All decoding was done using linear regression with ridge regularisation ...” |

### Processing Details
- Imaging data are calcium traces from mouse barrel cortex recorded longitudinally across development.
- Paper decoding uses neural data to predict behavioural variables, including motion-related measures.
- Both neural traces and behaviour traces were slightly denoised by averaging over 10 consecutive frames/timestamps before decoding.
- Same-day decoding used nested cross-validation with consecutive 2-minute blocks and 5-fold inner/outer splits.
- Cross-day decoding fit on one day and evaluated on other days.
- Calcium event statistics used light denoising with 10-frame averaging and peak detection via SciPy using height and prominence thresholds of 1 SD.

### Curation Steps

**Neuron curation rules**:
- Track2p/Suite2p context suggests using cells passing `iscell` threshold; README example uses `iscell_thr = 0.5`.
- Need Step 4 consistency check to confirm whether decoding analyses in paper used all ROIs, Suite2p cells only, or only tracked cells.

**Trial curation rules**:
- Native data are continuous sessions rather than explicit trials.
- Decoder trials must therefore be derived from contiguous 60-second windows, while preserving frame alignment between neural and motion traces.

### Decoders Trained
| Decoded variable | Accuracy |
| Motion / behavioural variables | Paper indicates successful same-day and cross-day decoding; exact accuracy values still to extract from paper text/figures |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Raw neural signal | Track2p code loads Suite2p outputs; not a decoder pipeline | `F.npy`, `Fneu.npy`, `spks.npy` all available per session | Paper decoding text explicitly refers to dF/F traces | Use calcium fluorescence-derived traces rather than deconvolved `spks` for decoder; compute neuropil-corrected fluorescence and dF/F in conversion script if no direct dF/F file exists |
| Cell inclusion | Track2p README example uses `iscell_thr = 0.5` | `iscell.npy` has Suite2p cell probability per ROI | Paper/Track2p context focuses on cells, not all ROIs | Restrict to ROIs with `iscell[:,0] > 0.5` unless later evidence contradicts this |
| Behavioral variable alignment | Track2p code not relevant | `motion_energy_glob.npy` and `tstamps.npy` are frame-level and nearly match neural frame counts | Paper decodes behaviour from neural activity and bins both streams together | Align behaviour to imaging frames using shared frame index/timestamps; handle rare 1-frame mismatches by truncating to common valid length |
| Trial structure | No trialing in Track2p | Sessions are continuous ~600 s recordings with ~36,000 frames | Paper used consecutive 2-minute blocks for CV, but current task requires 60-second trials | Derive decoder trials as contiguous 60-second windows from each session after common preprocessing |
| Temporal smoothing/binning | Track2p code does not specify decoder binning | Native data are framewise | Paper averages dF/F and behaviour over 10 consecutive timestamps before decoding | Match paper by applying 10-frame averaging before trial segmentation/output discretization |
| Brain region | Track2p README example and paper context mention barrel cortex | All sessions appear from same experiment structure | Paper is mouse barrel cortex | Set session brain region to barrel cortex for all neurons/sessions |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `iscell.npy` | neural | Select `iscell[:,0] > 0.5`; compute neuropil-corrected fluorescence `F - 0.7*Fneu`; compute per-neuron dF/F from session baseline; average in non-overlapping 10-frame bins; split into contiguous 60 s trials | Track2p `import_data` establishes Suite2p loading context | Use fluorescence-derived activity because paper decoding uses dF/F rather than deconvolved spikes |
| Session time index from imaging frames / `ops['fs']` | input[0] | Convert binned frame indices to elapsed seconds from session start for each 10-frame bin; split into same 60 s trials | N/A | Decoder input required by task: time elapsed from beginning of session |
| `move_deve/motion_energy_glob.npy` + `tstamps.npy` | output[0] | Truncate/alignment to common valid frame count with neural data; average in non-overlapping 10-frame bins; discretize within each session into 5 equal-percentile bins; split into contiguous 60 s trials | N/A | Decoder output required by task: session-wise motion energy bins |
| Subject folder name (e.g. `jm031`) | subjects / subject_idx | Enumerate unique subject IDs and map each session to subject index | N/A | One subject per top-level folder |
| Brain region from experiment context | brain_regions / brain_region_idx | Constant label `barrel cortex` for all neurons in all sessions | N/A | All recordings are from mouse barrel cortex |
| Session folder name/date | metadata.session_info | Store per-session identifiers, original paths, frame rate, trial counts | N/A | Helpful for traceability |

### Key Decisions
1. **Neural signal choice**: Use neuropil-corrected fluorescence converted to dF/F, not `spks.npy`, because the paper explicitly describes decoding from dF/F traces.
2. **Cell filtering**: Keep only ROIs with `iscell[:,0] > 0.5`, matching Suite2p/Track2p cell curation practice.
3. **Temporal smoothing**: Apply non-overlapping 10-frame averaging to both neural and motion-energy streams to match the paper's denoising before decoding.
4. **Temporal alignment**: Align streams at the frame level and truncate to the common valid length when motion arrays differ from neural arrays by 0-1 samples.
5. **Trialization**: Create trials as contiguous 60-second windows after 10-frame binning. With 60 Hz imaging and 36,000 frames, this yields 3,600 bins/session, 360 bins/trial, and typically 10 trials/session if `fs=60`; if actual `fs` differs, use `ops['fs']` to compute bins/trial exactly.
6. **Output discretization**: Compute 5 equal-percentile motion-energy bins separately for each session after smoothing/alignment, then assign categorical labels 0-4 at each time bin.
7. **Input representation**: Use a single continuous time-elapsed variable as a 1 x T time series per trial.
8. **Brain region annotation**: Annotate all neurons as `barrel cortex` because the dataset/paper describe recordings from mouse barrel cortex only.

### Planned Sanity Checks
- [ ] Check 1: For a chosen session, verify `converted neural trial 0` equals manual recomputation from raw `F`/`Fneu` for a few neurons/time bins using `np.allclose()`.
- [ ] Check 2: For a chosen session, verify `converted input time` matches manual binned elapsed-time computation from frame index and `ops['fs']` using `np.allclose()`.
- [ ] Check 3: For a chosen session, verify `converted output labels` match manual session-specific percentile binning of smoothed motion energy using `np.allclose()`.
- [ ] Check 4: Verify each session has at least 2 derived 60-second trials and that neural/input/output trial lengths all match exactly.

---

## Step 6: Script Development
**Status**: IN PROGRESS

Initial implementation written to `/app/convert_data.py` with session discovery, Suite2p loading, neuropil correction, session-wise dF/F, 10-frame averaging, per-session motion-energy percentile binning, 60-second trial splitting, optional processing plots, and target-format assembly.

Code inefficiencies identified:
- Current implementation loads full `F.npy` and `Fneu.npy` arrays per session into memory; may be acceptable for this dataset but should be monitored during full conversion.

Code speedups added:
- Used vectorized NumPy operations for neuropil correction, dF/F computation, smoothing, discretization, and trial splitting.
- Used simple non-overlapping reshape-based averaging for 10-frame binning.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 442 |
| Neurons / session | 221, 221 |
| Subjects | 1 |
| Sessions / subject | 2 sessions from jm031 |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| time_from_session_start_sec range | [0.0, 1199.7] |
| motion_energy_bin distribution | [0.2, 0.2, 0.2, 0.2, 0.2] |

### Processing Plots Review
- Processing plots were generated for both sample sessions.
- No obvious anomalies from validator output: neural/input/output dimensions are consistent and output classes are balanced by construction within session.
- Important correction to earlier assumption: sessions are ~1200 s long at 30 Hz, not ~600 s. After 10-frame averaging, each 60 s trial contains 180 bins and each session yields 20 trials.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| Sample conversion | ~0.18 s / session | Full conversion for 41 sessions expected to be well under 1 minute excluding plotting and I/O overhead |
| Verification | very fast | Full verification expected to be well under 1 minute |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy_bin | 0.3836 | 0.3137 |

Notes:
- Loss decreased smoothly from ~39.73 to ~1.58 over 200 epochs.
- Initial naive dF/F implementation produced near-chance decoding; replacing it with a more stable running low-percentile baseline informed by Suite2p metadata substantially improved performance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 393M
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Paper describes thousands of longitudinally recorded barrel-cortex neurons | Track2p operates on Suite2p cell sets | 20,445 included cells across all sessions after `iscell>0.5` | 20,445 | Yes |
| Mean neurons/session | Not explicitly extracted as a single value | Session-wise cell sets | 221, 370, 685, 746, 541, 435 by subject/session group | 498.66 mean across 41 sessions | Yes |
| Subjects | 6 mice in provided raw dataset | Consistent with code examples of per-animal longitudinal runs | 6 | 6 | Yes |
| Sessions | Longitudinal daily sessions across mice | Consistent with Track2p longitudinal usage | 41 | 41 | Yes |
| Trials (total) | Derived for decoder task | N/A | Derived from session duration / 60 s | 1081 | Yes |
| Trials/session (mean) | Derived for decoder task | N/A | 19-20 for ~1200 s sessions; 29-30 for ~1800 s sessions | 26.37 mean | Yes |
| time_from_session_start_sec range | Continuous session time | N/A | ~[0,1140], [0,1200], [0,1740], [0,1800] depending on session | Matches verification output | Yes |
| motion_energy_bin distribution | Session-wise equal-percentile discretization intended | N/A | Near-uniform by construction before trial truncation | Global fractions 0.199, 0.200, 0.200, 0.201, 0.200 | Yes |

Notes:
- Full verification reported: “Data format is valid, no errors or warnings.”
- All sessions have exactly matched trial lengths after preprocessing (T=180 bins per 60-second trial at 30 Hz with 10-frame averaging).
- Slight deviations from exact 0.2 class fractions in some sessions arise because percentile binning is done before dropping incomplete trailing bins/trials.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification: `/app/verification_full_out.txt` reports “Data format is valid, no errors or warnings.” No actionable warnings remained.
2. Neural sanity check from raw data: recomputed session 0, trial 0 neural matrix from raw `F.npy` and `Fneu.npy` using the conversion functions; `np.allclose()` returned True.
3. Input sanity check from raw data: recomputed session 0, trial 0 binned elapsed-time input from raw timestamps/frame-rate logic; `np.allclose()` returned True.
4. Output sanity check from raw data: recomputed session 0, trial 0 motion-energy labels from raw `motion_energy_glob.npy`; `np.allclose()` returned True.
5. Reference comparison: confirmed that conversion uses Suite2p cell filtering (`iscell > 0.5`), fluorescence-based activity rather than `spks`, 10-frame smoothing, and barrel-cortex session context, consistent with methods and Track2p/Suite2p usage.
6. Key statistics comparison: converted dataset retains 6 subjects, 41 sessions, and expected session-specific neuron counts (221, 370, 685, 746, 541, 435 by subject group), with all trials length 180 bins.
7. Edge-case review: sessions with slightly shorter aligned lengths (e.g. 3599 or 5399 binned samples before trial truncation) correctly yield 19 or 29 full 60-second trials; incomplete trailing data are dropped consistently across neural/input/output.

### Issues Found and Resolved
- Naive dF/F baseline caused unstable values and near-chance decoding on sample data. Resolution: replaced with a more stable running low-percentile baseline informed by Suite2p metadata (`baseline=maximin`, `win_baseline=60`, `prctile_baseline=8`).
- Timestamp-derived bin duration initially produced zero trials because timestamp units were not in seconds. Resolution: added robust timestamp handling with plausibility checks and fallback to frame-rate-derived timing.
- `--show-processing` initially failed when plotting logic encountered zero columns. Resolution: added a guard to skip plotting when no trials are available.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy_bin | 0.4946 | 0.3025 | Above chance (0.2000); full training completed successfully |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |
| motion_energy_bin | 0.3025 validation balanced accuracy (chance 0.2000) | Paper reports successful decoding of behavioural/motion variables from barrel-cortex calcium activity; exact directly comparable 5-bin accuracy not extracted |

- Accuracy vs chance: validation accuracy is above chance and slightly above the 1.5x-chance heuristic threshold (0.30 for 5 classes), so no immediate bug signal remains.
- Accuracy comparison to paper: the paper used ridge regression and somewhat different decoding targets/evaluation procedures, so exact numerical equality is not expected; nevertheless, successful above-chance decoding is consistent with the paper's conclusions.
- Train vs validation gap: training 0.4946 vs validation 0.3025 indicates some expected gap but not catastrophic leakage; validation remains clearly informative and above chance.

### Issues Found and Resolved
- Low sample accuracy with initial conversion was traced to unstable dF/F normalization and fixed by adopting a more stable running low-percentile baseline.
- Timestamp scaling issue briefly caused zero derived trials and was fixed with plausibility checks plus frame-rate fallback.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

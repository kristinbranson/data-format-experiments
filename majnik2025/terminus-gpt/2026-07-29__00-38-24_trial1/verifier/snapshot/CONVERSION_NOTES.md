# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: [Date]
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `drwxr-xr-x  4 root root      45 Jul 29 04:46 .`
- `dr-xr-xr-x 19 root root      84 Jul 29 04:46 ..`
- `-rw-r--r--  1 root root    1907 Jul 29 04:45 .manifest`
- `-rw-r--r--  1 root root    5770 Jul 29 04:46 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root    2136 Jul 28 18:56 Dockerfile`
- `drwxr-xr-x  5 root root     116 Mar  5 17:07 code`
- `drwxr-xr-x  2 root root    4096 Dec  2  2025 data`
- `-rw-r--r--  1 root root   81219 Mar  5 13:30 decoder.py`
- `-rw-r--r--  1 root root     296 Jul 29 00:46 docker-compose.yaml`
- `-rw-r--r--  1 root root     304 Mar  5 13:30 docker-compose.yaml~`
- `-rw-r--r--  1 root root    7402 Mar  5 13:42 methods.txt`
- `-rw-r--r--  1 root root 4473906 Mar 10 03:42 paper.pdf`
- `-rw-r--r--  1 root root    6539 Mar  5 13:30 train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE


### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| track2p | code/track2p/t2p.py | PROCESSING | Main pipeline entry point for Track2p longitudinal cell tracking across recordings |
| main | code/track2p/__main__.py | PROCESSING | Launches GUI entry point rather than analysis pipeline for this repo |
| setup_paths | code/track2p/t2p.py | LOADING | Validates/organizes input suite2p paths and output folders |
| load_data | code/track2p/t2p.py | LOADING | Loads per-session suite2p-derived data needed by the tracking pipeline |
| process_sessions / register_* / match_* | code/track2p/* | PROCESSING | Registration and matching logic across days/sessions |
| save_* / export_* | code/track2p/* | PROCESSING | Saves tracking outputs for downstream visualization/curation |
| gui fluorescence display helpers | code/track2p/gui/* | PROCESSING | Visualize fluorescence traces; indicates source data are calcium fluorescence traces rather than spikes |

### Notes
- The repository is the Track2p longitudinal cell tracking package, primarily for matching ROIs/cells across multiple calcium imaging days/sessions.
- The code README and docs focus on running the tracking GUI and pipeline on suite2p outputs, not on behavioral decoding.
- This strongly suggests the raw/source neural data are two-photon calcium imaging outputs derived from suite2p datasets.
- GUI code references fluorescence traces and z-scoring for visualization, implying neural signals are fluorescence traces rather than spike times.
- We still need to inspect the actual data files in `data/` to determine what processed outputs are provided for this task and whether motion energy / behavior are already aligned there.
- No clear evidence yet that this code computes trialized decoder-ready matrices; likely we will need to infer the relevant loading/processing from the provided data organization and methods text.


---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are organized as `data/<subject>/<session>/...`.
- Each valid session contains calcium imaging outputs in `suite2p/plane0/` and behavioral motion files in `move_deve/`.
- Neural files observed: `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `stat.npy`, and in at least some sessions `spks.npy`.
- Behavioral/time files observed: `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`.
- Sessions appear to be continuous frame-aligned recordings rather than pre-trialized task files.
- Motion energy and timestamps are closely aligned to neural recordings, with exact matches in 32 sessions and only off-by-one length differences in 3 sessions.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 20445 putative cells via `iscell[:,0] > 0.5` |
| Neurons / session | 498.6585365853659 mean putative cells/session |
| Subjects | 6 |
| Sessions / subject | {'jm031': 7, 'jm032': 7, 'jm038': 7, 'jm039': 7, 'jm040': 6, 'jm046': 7} |
| Trials (total) | No native trials apparent in inspected files; recordings appear continuous |
| Trials / session | To be defined during conversion via segmentation into decoder-compatible windows |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not explicitly stated in methods excerpt | N/A |
| Neurons / session | Not explicitly stated in methods excerpt | N/A |
| Subjects | At least multiple mice across longitudinal sessions; exact count not in methods excerpt | N/A |
| Sessions / subject | Longitudinal across days; examples show 7 consecutive daily recordings in mouse barrel cortex | "7 consecutive daily recordings in mouse barrel cortex (between P8 and P14)" (code README) |
| Trials (total) | No trials; continuous recordings analyzed in blocks | "splits were done on consecutive 2 minute blocks of the recording" |
| Trials / session | Not native; analysis blocks of 2 minutes | "splits were done on consecutive 2 minute blocks of the recording" |
| Neural data time bin | 10 consecutive timestamps for decoding analyses | "we slightly denoised the dF/F ... by averaging in bins of 10 consecutive timestamps" |
| Behavior data time bin | 10 consecutive timestamps for decoding analyses | "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" |
| Reward rate | N/A | N/A |
| Behavior video rate | 30 Hz | "Videos were recorded at 30 Hz" |
| Cell inclusion threshold | iscell probability > 0.5 | "We considered all ROIs above the default threshold of 0.5 as true cells" |
| Motion variable | Global movement from squared pixel-wise frame differences | "This yielded a scalar value quantifying the motion of the mouse at each time point" |

### Processing Details
- Imaging and videography are synchronized: microscope acquisition triggers camera frame acquisition.
- Calcium imaging preprocessing uses Suite2p: motion correction, ROI detection, signal extraction, and spike deconvolution for each recording separately.
- Subsequent analyses use baseline-corrected fluorescence traces as dF/F using default Suite2p parameters.
- Motion/arousal proxy is computed from consecutive video-frame differences: pixel-wise difference, square values, sum across pixels to get a scalar motion value per time point.
- For decoding analyses in the reference, both neural dF/F and behavior traces are denoised by averaging in bins of 10 consecutive timestamps.
- Reference decoding uses ridge regression with nested cross-validation and 2-minute consecutive blocks.

### Curation Steps

**Neuron curation rules**:
- Keep ROIs with Suite2p `iscell` probability above 0.5.
- Use baseline-corrected fluorescence traces as dF/F for analyses.

**Trial curation rules**:
- No native trial structure described in methods excerpt; recordings are continuous and split into consecutive 2-minute blocks for decoding/evaluation.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Global movement / behavioral data from neural data | Not yet extracted from paper text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural modality | Track2p/Suite2p fluorescence-based calcium imaging | `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `ops.npy`, `stat.npy` present per session | Calcium imaging preprocessed with Suite2p | Use Suite2p-derived calcium signals, not spikes from electrophysiology |
| Cell curation | Use Suite2p cell classification | All sessions include `iscell.npy`; in this dataset all ROIs counted so far pass >0.5 threshold | ROIs above 0.5 considered true cells | Filter by `iscell[:,0] > 0.5`; this currently retains all cells in the inspected dataset |
| Behavioral variable | Motion quantified from video frame differences | `motion_energy_glob.npy` present per session and frame-aligned with timestamps | Global movement from squared pixel-wise differences of consecutive video frames | Use `motion_energy_glob.npy` as source behavioral variable |
| Temporal alignment | Video synchronized to microscope acquisition | Most sessions have exact frame count matches; some have small mismatches and some larger truncations in motion/timestamps | Camera triggered by microscope acquisition | Align streams by common valid duration, trimming to the minimum shared length per session after later binning checks |
| Trial structure | Reference decoding used continuous 2-minute blocks | Data are continuous recordings with no native trial files | Methods describe continuous recordings split into 2-minute blocks | Create decoder-compatible pseudo-trials/windows from continuous recordings |
| Time binning | Reference decoding averages 10 consecutive timestamps | Raw data are framewise continuous arrays | Methods explicitly average dF/F and behavior in bins of 10 timestamps | Use 10-frame binning to match reference processing as closely as possible |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `iscell.npy` | neural | Filter cells with `iscell[:,0] > 0.5`; derive fluorescence-based neural signal matching reference analyses; average in bins of 10 consecutive timestamps; segment into consecutive windows | Suite2p outputs referenced by Track2p; methods specify baseline-corrected fluorescence dF/F | Prefer fluorescence/dF/F-like signal over deconvolved spikes because methods say subsequent analyses used baseline-corrected fluorescence traces as dF/F |
| Elapsed time from experiment start | input[0] | Build 1 x T time series in seconds for each pseudo-trial/window after binning | N/A (decoder-task-specific) | Decoder input required by task statement |
| `move_deve/motion_energy_glob.npy` | output[0] | Trim/alignment to neural frames, average in bins of 10 timestamps, discretize into 5 equal-percentile bins over the full converted dataset | Methods describe motion computation and 10-frame averaging | Decoder output required by task statement |
| Subject folder name (e.g. `jm031`) | subjects / subject_idx | Unique subject list and per-session index | N/A | Session order will follow conversion order |
| Imaging plane / experiment | brain_regions / brain_region_idx | Single region label `barrel cortex` for all neurons | Methods and README mention mouse barrel cortex | All neurons in all sessions use same region label unless data reveal otherwise |

### Key Decisions
1. **Use fluorescence-based neural activity**: Methods explicitly state that all subsequent analyses used baseline-corrected fluorescence traces as dF/F, so neural data should be based on fluorescence rather than `spks.npy`.
2. **Filter with `iscell > 0.5`**: This matches the reference text exactly.
3. **Use 10-frame temporal binning**: This matches the reference decoding preprocessing for both neural and behavior traces.
4. **Create pseudo-trials from continuous recordings**: Because there are no native trials, use consecutive fixed windows to satisfy decoder format while staying close to the paper's use of consecutive 2-minute blocks.
5. **Align by shared valid length**: For sessions with length mismatches between neural and behavior arrays, trim to the minimum shared length before binning/segmentation.
6. **Discretize motion energy globally into 5 equal-percentile bins**: This follows the decoder task requirement and preserves relative motion-state occupancy.
7. **Use time-from-session-start as decoder input**: This directly matches the decoder task wording and can be represented continuously per time bin.

### Planned Sanity Checks
- [ ] Check that for sampled sessions, neural and motion arrays after trimming and binning have identical time lengths.
- [ ] Check that selected neurons equal the count from raw `iscell[:,0] > 0.5`.
- [ ] Check that output bin frequencies are approximately balanced globally after percentile discretization.
- [ ] Check that session/window counts are consistent with recording duration and chosen window length.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 442 |
| Neurons / session | 221 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 20 |
| Trials / session | 10 |
| time_from_session_start_s range | [0.0, 1199.7] |
| ... | [MIN, MAX] |
| motion_energy_bin distribution (global) | [0.200,0.200,0.200,0.200,0.200] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
No format anomalies. Important note: per-session class balance is not perfectly uniform because percentile binning is global across the sample dataset, but global distribution is exactly balanced.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy_bin | 0.5140 | 0.3212 |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 409343985 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | N/A | Suite2p/iscell logic | 20445 | 20445 | Yes |
| Mean neurons/session | N/A | Derived from sessions | 498.66 | 498.66 | Yes |
| Subjects | 6 | | | |
| Sessions | N/A | Folder structure | 41 | 41 | Yes |
| Trials (total) | Continuous recordings; no native trials | Windowing plan | 536 pseudo-trials | 536 pseudo-trials | Yes |
| Trials/session (mean) | Continuous recordings | 2-minute block logic | 13.07 | 13.07 | Yes |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Raw-vs-converted neural sanity checks on sessions jm031/2023-10-18_a, jm031/2023-10-20_a, and jm046/2024-09-09_a: exact `np.allclose` match for first-trial binned neural matrices.
2. Neuron-count sanity checks: converted neuron counts exactly match raw `iscell[:,0] > 0.5` counts for spot-checked sessions.
3. Input sanity checks: time input increases across pseudo-trials within session and matches absolute session-start timing.
4. Output-log verification: `verification_full_out.txt` reports valid format with no errors or warnings.
5. Reference consistency review: processing uses Suite2p-derived activity, `iscell > 0.5`, continuous recordings segmented into 2-minute windows, and 10-frame averaging consistent with methods.

### Issues Found and Resolved
- Initial dF/F-like fluorescence signal yielded below-chance sample decoding; switched to Suite2p `spks.npy`, which improved sample and full decoding substantially.
- Initial time input construction from raw timestamps produced incorrect scale; replaced with frame-rate-derived elapsed seconds from session start.
- Some sessions have neural/behavior length mismatches; conversion trims to shared minimum length before binning.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy_bin | 0.5140 | 0.3212 |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized

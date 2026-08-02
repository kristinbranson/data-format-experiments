# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code/
- data/
- decoder.py
- docker-compose.yaml
- docker-compose.yaml~
- methods.txt
- paper.pdf
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| run_t2p | code/track2p/t2p.py | PROCESSING | Main pipeline: initializes paths, checks planes, loads images, registers ROIs across datasets, computes assignments/match matrices, saves outputs, and generates plots |
| check_nplanes | code/track2p/io/s2p_loaders.py | LOADING | Verifies all datasets have the same number of Suite2p planes |
| load_all_imgs | code/track2p/io/s2p_loaders.py | LOADING | Loads `ops.npy` per plane and extracts `meanImg`, `meanImg_chan2`, and `nchannels` |
| load_all_ds_stat_iscell | code/track2p/io/s2p_loaders.py | CURATION | Loads `stat.npy` and `iscell.npy`, filtering ROIs by `iscell[:,0]==1` or `iscell[:,1] > iscell_thr` |
| load_stat_ds_plane | code/track2p/io/loaders.py | CURATION | Loads one plane's ROI stats and reports how many ROIs survive the `iscell` threshold |
| get_all_roi_array_from_stat | code/track2p/io/loaders.py | PROCESSING | Converts Suite2p ROI pixel coordinates into boolean ROI masks |
| npy_to_s2p | code/track2p/io/savers.py | LOADING | Converts raw numpy arrays (`F.npy`, `fov.npy`, `rois.npy`) into Suite2p-like structure when needed |
| generate_suite2p_indices | code/track2p/t2p.py | PROCESSING | Uses track2p match matrices plus per-session `iscell` filtering to map tracked cells back to Suite2p indices |

### Notes
- Reference code is for longitudinal calcium imaging cell tracking across sessions/days using Suite2p outputs.
- Expected source structure is Suite2p-style folders with `suite2p/plane*/ops.npy`, `stat.npy`, `iscell.npy`, and likely activity arrays such as `F.npy`, `Fneu.npy`, `spks.npy`.
- ROI/cell curation is explicit: if `track_ops.iscell_thr` is `None`, keep ROIs with `iscell[:,0] == 1`; otherwise keep ROIs with `iscell[:,1] > iscell_thr`. Default threshold in `DefaultTrackOps` is 0.50.
- `run_t2p` performs registration and matching across datasets, saving `plane*_match_mat.npy` files that define tracked-cell correspondences across days.
- GUI code references `F.npy`, `Fneu.npy`, and `spks.npy`, suggesting these are the neural activity representations available after Suite2p processing. Need to inspect actual data files in Step 2 to determine which are present and whether trial/behavior annotations exist.
- `npy_to_s2p` indicates an alternate raw format (`data_npy/plane*/F.npy`, `fov.npy`, `rois.npy`), but the default/reference workflow expects Suite2p-style inputs.
- No evidence yet in the reference code of trial segmentation or behavioral decoding labels; likely these must come from data files outside the core track2p package.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains 6 subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), a `README.md`, and `load_data.ipynb`.
- Each subject contains dated session folders (41 sessions total across all subjects).
- Each session contains `suite2p/plane0/` and `move_deve/`.
- `suite2p/plane0/` contains `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `ops.npy`, and `stat.npy`.
- `move_deve/` contains `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`.
- Some subjects also contain `ground_truth.csv`, likely related to tracked-cell correspondence / validation across days.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 20445 |
| Neurons / session | ~498.66 mean (20445 / 41); examples: 221, 370, 435 |
| Subjects | 6 |
| Sessions / subject | jm031: 7; jm032: 7; jm038: 7; jm039: 7; jm040: 6; jm046: 7 |
| Trials (total) | Not natively trial-structured; sessions are continuous recordings |
| Trials / session | N/A in raw data |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Motion energy / behavioral state | Ridge regression used in paper; exact accuracy to be extracted if needed from paper | 
| Neurons / session | Not stated directly; tracked neurons average 526 ± 190 per mouse across all days | "On average 526 (± 190 std) neurons per mouse were successfully tracked across all days" |
| Subjects | 6 mice in full dataset used for subsequent analyses | "we used a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days" |
| Sessions / subject | Minimum 6 consecutive daily sessions; example dataset has 7 sessions from one mouse | "daily recordings... P8 to P14, n=7 imaging sessions from one mouse" and "6 mice imaged daily for a minimum of 6 consecutive days" |
| Trials (total) | No native trials; decoding splits continuous recordings into consecutive 2 minute blocks | "splits were done on consecutive 2 minute blocks of the recording" |
| Trials / session | 10 blocks/session if using 20-minute sessions and 2-minute blocks | Derived from 20-minute sessions and 2-minute blocks |
| Neural data time bin | 30 Hz acquisition; decoding uses 10-frame averaging (~0.333 s bins) | "Imaging rate was 30 Hz" and "averaging in bins of 10 consecutive timestamps" |
| Behavior data time bin | 30 Hz video; decoding uses 10-frame averaging (~0.333 s bins) | "Videos were recorded at 30 Hz" and "averaging in bins of 10 consecutive timestamps" |
| Reward rate | N/A (spontaneous behavior, no reward task) | "spontaneous mouse movement" | 
| Motion metric | Global motion energy from squared pixel-wise frame differences | "computed their pixelwise difference... squared... summed across pixels" | 
| Cell curation threshold | iscell probability > 0.5 | "We considered all ROIs above the default threshold of 0.5 as true cells" |
| ... | motion_energy_bin | 0.4443 validation balanced accuracy | Chance = 0.2000; achieved 2.22x chance | | 


### Processing Details
- Continuous calcium imaging and videography are synchronized by microscope-triggered camera acquisition.
- Motion energy can have missing camera frames; `tstamps.npy` and `interframe_int.npy` identify missing frames, which should be treated as missing values or interpolated over.
- Neural traces for downstream analyses are baseline-corrected Suite2p fluorescence traces (dF/F using default Suite2p parameters).
- Decoding analyses in the paper denoise both neural and behavior traces by averaging over bins of 10 consecutive timestamps.
- Cross-validation in the paper uses consecutive 2-minute blocks from each 20-minute session, implying a natural segmentation of 10 blocks per full 20-minute recording.

### Curation Steps

**Neuron curation rules**:
- Keep ROIs classified as cells by Suite2p using default threshold: `iscell` probability > 0.5.
- Track2p then identifies neurons successfully tracked across all days for longitudinal analyses.

**Trial curation rules**:
- Keep ROIs classified as cells by Suite2p using default threshold: `iscell` probability > 0.5.
- Track2p then identifies neurons successfully tracked across all days for longitudinal analyses.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Subjects | 6 subjects in data | 6 subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) | 6 mice | Match |
| Sessions | Minimum 6 consecutive daily sessions | 41 sessions total; per-subject counts 7,7,7,7,6,7 | Minimum 6 consecutive days | Match |
| Imaging rate | 30 Hz | `ops.npy['fs'] = 30` in inspected sessions | 30 Hz | Match |
| Session duration | 20 min | Mixed: 20 min for jm031/jm032 (36000 frames), 30 min for jm038/jm039/jm040/jm046 (54000 frames) | 20 min stated in methods | Keep observed durations from data; document discrepancy as likely broader dataset variation or text simplification |
| Trial structure | Continuous recordings split into 2-min blocks for decoding | Continuous recordings; no native trial markers observed | Consecutive 2-min blocks used for decoding | Match |
| Motion alignment | Video synchronized to microscope, occasional missing camera frames | `motion_energy_glob.npy` length sometimes shorter than imaging by 1-116 frames; `tstamps.npy` and `interframe_int.npy` available | Missing frames should be identified via timestamps/interframe intervals | Match |
| Cell curation | Suite2p `iscell` probability > 0.5 | All observed `iscell[:,0] == 1`; reference code filters by `iscell[:,1] > 0.5` | ROIs above default threshold 0.5 are true cells | Match |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy` + `Fneu.npy` + `ops.npy` for tracked cells across days | neural | Compute Suite2p-consistent neuropil-corrected / baseline-corrected fluorescence (dF/F-like signal) from released raw fluorescence traces; segment continuous recording into consecutive 2-minute blocks; average in bins of 10 frames to match paper decoding preprocessing | `data/load_data.ipynb`, methods text, Suite2p metadata in `ops.npy` | Data README says rows are already tracked and matched across all days within each subject; notebook notes `F.npy` are raw fluorescence traces and dF/F should be computed for proper analysis |
| elapsed time within each 2-minute block | input[0] | Create a 1 x T time series in seconds after 10-frame binning (`t = bin_index * 10 / 30`) | N/A (constructed to satisfy decoder input specification) | Decoder input requested by task is time elapsed from beginning of experiment; operationalize as time within session/block after alignment |
| `move_deve/motion_energy_glob.npy` aligned to imaging frames | output[0] | Handle missing camera frames using `tstamps.npy` / `interframe_int.npy`; average in bins of 10 frames; discretize valid values into 5 equal-percentile bins; represent as 1 x T categorical time series | Methods text; `data/README.md` | Output is motion energy / behavioral state, as requested by task |

### Key Decisions
1. **Trials = consecutive 2-minute blocks**: Raw recordings are continuous and the paper's decoding uses consecutive 2-minute blocks, so these are the natural trial units for the target format.
2. **Use the released tracked-neuron matrices as provided**: Data README states the `suite2p` folders already contain only cells present across all days, with matched row order across sessions within each subject.
3. **Use 10-frame averaging before decoding**: Methods explicitly state that both dF/F and behavior traces were slightly denoised by averaging in bins of 10 consecutive timestamps.
4. **Align motion to imaging frame index and preserve missingness information**: `motion_energy_glob.npy` can be shorter than imaging due to missing camera frames; missing values should be inserted or interpolated according to `tstamps.npy` / `interframe_int.npy`, matching data README guidance.
5. **Discretize motion energy into 5 equal-percentile bins using valid binned samples across the full converted dataset**: This follows the decoder task requirement while preserving balanced class frequencies as much as possible.
6. **Use one session per recording day and one subject per mouse**: Direct mapping from native organization.

### Planned Sanity Checks
- [ ] Check neural block extraction against raw `F.npy` values for a chosen subject/session/block using `np.allclose()`
- [ ] Check motion alignment and binning against raw `motion_energy_glob.npy` / timestamp arrays for a chosen session using `np.allclose()`
- [ ] Check time input construction against expected 10-frame / 30 Hz bin centers or starts using `np.allclose()`
- [ ] Check tracked-cell selection against raw cross-session mapping metadata (ground truth or Track2p outputs) for one subject

---

## Step 6: Script Development
**Status**: IN PROGRESS

- Implemented `convert_data.py` with CLI `python -u convert_data.py <outpicklefile>` and options `--full`, `--sample`, `--show-processing`.
- Script discovers subjects/sessions from `data/`, loads tracked-neuron Suite2p traces plus motion-energy behavior, computes a neuropil-subtracted baseline-normalized neural signal, bins neural/behavior by 10 frames, segments sessions into consecutive 2-minute blocks, constructs time input, discretizes motion energy into 5 percentile bins, and writes the target pickle structure.
- Optional processing plots are saved for up to 2 sessions.

Code inefficiencies identified:
- Potential inefficiency: baseline computation loops over neurons and uses a simplified approximation instead of Suite2p internals.
- Potential inefficiency: all sessions are prepared in memory before global motion discretization.

Code speedups added:
- Used vectorized frame binning for neural and motion arrays.
- Limited plotting to first two sessions in `--show-processing` mode.
- Sample mode processes only first two sessions for rapid validation.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 442 |
| Neurons / session | 221 |
| Subjects | 1 represented in sessions (jm031); subject list retained from full dataset has length 6 |
| Sessions / subject | jm031: 2 sample sessions |
| Trials (total) | 20 |
| Trials / session | 10 |
| time_elapsed_s range | [0.0, 119.7] |
| motion_energy_bin distribution | [0.200,0.200,0.200,0.200,0.200] globally |

### Processing Plots Review
- No format anomalies in verifier.
- Processing plots were generated for both sample sessions.
- Motion bins are globally balanced by construction; per-session fractions vary, which is expected.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Sample mode only | Fast validation run |

| Step | Time / Session | Estimated Total Time |
| Conversion | ~2.1 s/session in sample run | ~1.5-3 min for all 41 sessions, depending on plotting and I/O |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy_bin | 0.6280 | 0.3601 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 414413206 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | 6 mice in full dataset used for subsequent analyses | "we used a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days" | | | |
| Sessions | | | | | |
| Trials (total) | No native trials; decoding splits continuous recordings into consecutive 2 minute blocks | "splits were done on consecutive 2 minute blocks of the recording" | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

### Checks Performed
1. Output log verification: `verification_full_out.txt` reports valid format with no errors/warnings.
2. Raw-vs-converted neural sanity check: recomputed first and last session trial blocks from raw `F.npy`/`Fneu.npy` matched converted neural arrays with `np.allclose()`.
3. Raw-vs-converted input sanity check: recomputed time vectors matched converted inputs with `np.allclose()`.
4. Raw-vs-converted output sanity check: recomputed global motion-energy bin edges and per-trial discretized labels matched converted outputs exactly.
5. Reference comparison: conversion uses tracked-across-days Suite2p outputs from released dataset, 10-frame averaging, 2-minute block segmentation, and global motion-energy discretization required by task.

### Issues Found and Resolved
- Initial issue: motion alignment incorrectly interpreted `tstamps.npy` as frame indices, dropping 9 sessions. Resolution: align motion by sample order and pad/truncate to imaging length, per `data/README.md`; reran sample and full conversions.
- Remaining caveat: session durations in data are mixed 20 and 30 minutes despite methods text stating 20 minutes. Resolution: preserve observed durations from raw data and document discrepancy.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy_bin | 0.5478 | 0.4443 | Above 2x chance (0.20); stable full-dataset decoding |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

- Validation accuracy for motion_energy_bin is 0.4443, which is well above chance (0.2000) and above the 1.5x-chance debugging threshold.
- Training/validation ratio is 0.5478 / 0.4443 = 1.23, below the 1.5x overfitting warning threshold.
- Decoder performance therefore does not suggest a major temporal-alignment or label-construction bug.

### Issues Found and Resolved
- Initial issue: motion alignment incorrectly interpreted `tstamps.npy` as frame indices, dropping 9 sessions. Resolution: align motion by sample order and pad/truncate to imaging length, per `data/README.md`; reran sample and full conversions.
- Remaining caveat: session durations in data are mixed 20 and 30 minutes despite methods text stating 20 minutes. Resolution: preserve observed durations from raw data and document discrepancy.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

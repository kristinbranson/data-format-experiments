# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `total 21568`
- `drwxr-xr-x  4 root root       45 Sep 21 16:23 .`
- `dr-xr-xr-x 19 root root      116 Sep 21 16:22 ..`
- `-rw-r--r--  1 root root      370 Sep 21 16:22 .manifest`
- `-rw-r--r--  1 root root     5010 Sep 21 16:23 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root     2673 Sep 21 02:57 Dockerfile`
- `drwxr-xr-x  2 root root     4096 Sep 21 02:52 code`
- `drwxr-xr-x  2 root root     4096 Dec  3  2025 data`
- `-rw-r--r--  1 root root    89127 Sep 21 02:52 decoder.py`
- `-rw-r--r--  1 root root      650 Sep 21 02:52 docker-compose.yaml`
- `-rw-r--r--  1 root root     9381 Sep 21 02:52 methods.txt`
- `-rw-r--r--  1 root root 21948605 Sep 21 02:52 paper.pdf`
- `-rw-r--r--  1 root root     7641 Sep 21 02:52 train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_S6_dat | S6.py | LOADING | Load processed supplementary behavior/neural summary dictionaries from .npy files |
| load_fig5_dat | fig5.py | LOADING | Load processed data for figure 5 analyses |
| lickCount | utils.py | PROCESSING | Count licks in defined temporal ranges; used to classify licking trials |
| fmt / plotting helpers | utils.py | PROCESSING | Plot formatting, useful for inferring variable meanings and axes |
| notebook cells in data_process_script.ipynb | data_process_script.ipynb | PROCESSING | Primary reference for data loading, trial processing, and saving intermediate results |

### Notes
- README indicates `data_process_script.ipynb` is the key reference for processing and saving intermediate results.
- Figure scripts mostly consume already processed `.npy` dictionaries from `process_data`, so they help infer variable names and alignment conventions.
- Initial grep suggests behavior variables include lick timing, reward-related trial labels, cue alignment, first-lick alignment, and example sessions named like `VR2_2021_04_11_1`.
- fig4 comments explicitly state recording frame rate is 3 Hz for lick-rate conversion, which is likely the imaging/behavior sample rate used in processed traces.
- Need to inspect notebook code carefully to determine exact raw-data loading, curation, alignment to corridor entry, and construction of trial-by-time neural/behavior matrices.
- Notebook code loads behavior dictionaries `Beh_<exp_type>.npy`, processed d-prime files, and calls utility functions such as `utils.Get_dprime_rewPred_neuron(...)`, implying `utils.py` contains core trial-aligned processing logic.


---

## Step 2: Dataset Exploration
**Status**: IN PROGRESS

### Data Structure
- Top-level directories under `/app/data`: `spk`, `beh`, `process_data`, `retinotopy`.
- `spk` contains 89 per-session `*_neural_data.npy` files. Each loads as a dict with key `spks`, a list of trial arrays.
- Example neural session `DR10_2022_07_12_1` contains 3 trial arrays, each float32 with identical shape `(19408, 31707)`.
- Another example session `TX123_2024_01_02_1` has trial shape `(14367, 29251)`, so dimensions vary by session.
- `beh` contains 25 aggregate behavior `.npy` dictionaries, e.g. `Beh_sup_test2.npy`, keyed by session ID.
- `retinotopy` contains 90 date-specific `*_trans.npz` files with keys `A`, `xpos`, `ypos`, `xy_t`, `iarea`, likely for anatomical/area assignment metadata.
- `process_data` contains author-generated intermediate processed outputs used by figure scripts; these may help replicate reference processing but are not raw trial data.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | pending orientation/session inspection |
| Neurons / session | pending orientation/session inspection |
| Subjects | 19 |
| Sessions / subject | min=1, max=8 |
| Trials (total) | pending behavior/session inspection |
| Trials / session | example session DR10_2022_07_12_1 has 3 |

---

## Step 3: Reference Text Reading
**Status**: IN PROGRESS

### Notes in progress
- Collecting quantitative dataset statistics and any reported decoding results from methods.txt and paper.pdf.

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 20,547 to 89,577 per recording (raw Suite2p traces) | "we ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording" | 
| Neurons / session | | |
| Subjects | 19 mice | "We performed 89 recordings in 19 mice..." |

| Sessions / subject | 1 to 8 in provided data | data files; methods reports 89 recordings total |
| Trials (total) | not yet extracted | pending dataset exploration |
| Behavior data time bin | likely imaging/behavior frame samples; figure code comments mention 3 Hz for processed traces | fig4.py comment |
| Running threshold | 6 cm/s | "mice moved forward ... by running faster than a threshold of 6 cm s−1" |
| VR corridor speed | 60 cm/s while running | "virtual corridors always moved at a constant speed (60 cm s−1)" |
| Sound cue position | uniform 0.5–3.5 m | methods.txt |
| Trials (total) | | |
| Trials / session | cue/reward trials in 4 m VR corridors; exact count not stated here | methods.txt task description |
| Neural data time bin | based on imaging frame samples; exact frame rate not stated in methods excerpt | methods.txt + code comments |
| Behavior data time bin | | |
| Reward rate | rewarded corridor only; unsupervised has no rewards | methods.txt task description | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| ... | | | | 


### Processing Details
- Calcium imaging data were processed with Suite2p.
- Analyses are based on deconvolved fluorescence traces rather than raw fluorescence.
- Non-negative deconvolution used a decay timescale of 0.75 s.
- For imaging mice, sound cue was presented in all trial types.
- Sound cue timing/location was randomized uniformly between corridor positions 0.5 m and 3.5 m.
- In rewarded corridors for task mice, the sound cue indicated the beginning of the reward zone.
- Reward was delivered upon licking after the sound cue in rewarded corridors; delivery locations were approximately uniformly distributed between 0.5 m and 3.5 m because animals often licked soon after cue.
- Anticipatory licking before the sound cue inside the corridor is behaviorally important and should be represented.


### Curation Steps

**Neuron curation rules**:
- Suite2p ROI detection/cell classification and neuropil correction were applied before deconvolution.
- Exact downstream inclusion/exclusion criteria still need confirmation from reference code/notebook.

**Trial curation rules**:
- Reference text states only running timepoints were considered for analysis, removing periods when task mice stopped to collect water rewards.
- Exact trial exclusion criteria still need confirmation from reference code/notebook.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: IN PROGRESS

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Data source granularity | Figure scripts largely consume processed `.npy` summaries and utility functions | Raw-ish neural data are in `spk`, behavior in aggregate `beh` dicts, retinotopy in `.npz` | Methods describe Suite2p/deconvolved traces and running-timepoint analysis | Use raw-ish `spk` + `beh` + `retinotopy` for conversion; use `process_data` and utility code only to infer processing/alignment conventions |
| Session counts | Reference code implicitly operates across many experiment groups | `spk` contains 89 session files across 19 subjects | Methods report 89 recordings in 19 mice | Counts are consistent |
| Neuron counts | Methods mention 20,547–89,577 traces per recording after Suite2p | Example `spk` arrays have shapes like `(25510,20271)` with second dimension closely matching behavior frame count (`ft_len=20272`), implying orientation is approximately `(n_neurons, n_timepoints)` | Same methods statement | Orientation is neuron-by-time. Reference utility code concatenates `np.load(...)["spks"]` along axis 0, so the 3 list items are separate neuron groups that together form the full-session neural population, not separate trials. Trial segmentation must be reconstructed from behavior frame indices such as `StartFr`, `EndFr`, and `ft_trInd` |
| Retinotopy linkage | Utility functions use retinotopy for area-based analyses | Matching retinotopy file lengths do not equal example `spk` dimensions | Paper describes simultaneous recordings across visual areas | Retinotopy files may refer to broader ROI sets or pre-filtered cell lists; inspect utility code for mapping logic |
| Running-period analysis | Methods say only running timepoints used | Raw session arrays appear full-length; running mask not yet identified | Same | Need behavior-variable inspection and utility-code review to reproduce running-only selection when applicable |
| Temporal alignment | Utility functions include cue- and first-lick-aligned helpers | Raw trial arrays are already trial-segmented in `spks` lists | Methods emphasize cue timing and corridor structure | For decoder, primary alignment must be corridor entry; derive cue-relative variables from behavior/session metadata |
| Reward/task structure | Behavior files are grouped by experiment names like `Beh_sup_test2.npy` | Session IDs appear inside these dicts | Methods describe rewarded vs unsupervised cohorts and multiple stimulus sets | Use behavior dict membership/session metadata to assign day/training condition and reward availability |


---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spk/*_neural_data.npy` -> dict[`spks`] | neural | Concatenate the 3 neuron-group arrays along axis 0 to form full-session `(n_neurons, n_timepoints)` data, then segment trials using behavior frame indices | raw data + `utils.py` concat logic | `spks` entries are neuron groups, not trials |
| Behavior field `SoundFr` / `SoundTime` | input[0] time to sound cue | For each trial, compute continuous time-until-cue at each frame/time bin | `spk_2_cue`, behavior dict fields | Decoder input requires continuous time-varying variable |
| Session training day / experiment order | input[1] day of training | Continuous per-trial scalar, broadcast across timepoints if needed | notebook `exp_info`, behavior file grouping | Need mapping from experiment/session to day index |
| Trial time index | input[2] time since trial start | Continuous time-varying variable from corridor entry alignment | raw trial segmentation | Alignment event is trial start / corridor entry |
| Behavior fields `isRew`, `WallType`, `WallName`, `TrialStim` | input[3] reward availability | Use per-trial rewarded-corridor identity / reward availability, broadcast across timepoints | behavior dict | 1 in rewarded corridor, 0 otherwise |
| Behavior field `TrialStim` (plus `WallName` / `WallType` if needed) | output[0] visual stimulus category | Map per-trial categorical label from trial stimulus names | behavior dict session fields / notebook stimulus IDs | Categories may include leaf/circle/rock/brick variants and probes |
| Lick fields `LickFr`, `LickTrind` | output[1] licking | Binary time series per trial | `spk_2_cue`, `spk_2_firstLick` | Build framewise lick vector |
| Behavior fields `ft_Pos` or `VRpos` | output[2] corridor position bin | Discretize corridor position into 4 equal 1 m bins over 4 m corridor | methods + behavior variables | Prefer frame-aligned `ft_Pos`; exclude gray-space frames as needed |
| Behavior field `ft_RunSpeed` | output[3] running speed bin | Discretize running speed into quartiles across dataset | behavior dict | Use frame-aligned speed values |

### Key Decisions
1. **Use deconvolved traces, not raw fluorescence**: Matches methods text and reference analyses.
2. **Use raw-ish `spk` + `beh` + `retinotopy` for conversion**: `process_data` is mainly for derived analyses and figure generation.
3. **Primary temporal alignment is corridor entry / trial start**: Required by decoder task; cue-relative variables will be derived from behavior frame indices.
4. **Represent per-trial variables as time-varying broadcasts when needed**: Keeps input/output arrays shape-consistent for decoder.
5. **Use behavior frame masks to define valid analysis periods**: Methods explicitly state only running timepoints were analyzed; candidate masks include `ft_isMoving` and `ft_CorrSpc`.

### Planned Sanity Checks
- [ ] Verify a few `SoundFr` alignments by comparing raw lick/neural windows to `spk_2_cue` outputs.
- [ ] Verify lick raster reconstruction from `LickFr` and `LickTrind` for several trials.
- [ ] Verify trial counts match between `spks` and behavior `ntrials` for matched sessions.
- [ ] Verify stimulus labels and reward availability against behavior file grouping and raw session metadata.

-----------------|--------------|-----------|----------------------------|-------|
| `spk/*_neural_data.npy` -> dict[`spks`] | neural | Concatenate the 3 neuron-group arrays along axis 0 to form full-session `(n_neurons, n_timepoints)` data, then segment trials using behavior frame indices | raw data + `utils.py` concat logic | `spks` entries are neuron groups, not trials |
| Behavior field `SoundFr` / `SoundTime` | input[0] time to sound cue | For each trial, compute continuous time-until-cue at each frame/time bin; likely `(SoundFr - frame_idx) * dt` or using times directly | `spk_2_cue`, behavior dict fields | Decoder input requires continuous time-varying variable |
| Session training day / experiment order | input[1] day of training | Continuous per-trial scalar, broadcast across timepoints if needed | notebook `exp_info`, behavior file grouping | Need mapping from experiment/session to day index |
| Trial time index | input[2] time since trial start | Continuous time-varying variable from corridor entry alignment | raw trial segmentation | Alignment event is trial start / corridor entry |
| Behavior fields `isRew`, `WallType`, `WallName`, `TrialStim` | input[3] reward availability | Use per-trial rewarded-corridor identity / reward availability, broadcast across timepoints | behavior dict | 1 in rewarded corridor, 0 otherwise |
| Behavior field `TrialStim` (plus `WallName` / `WallType` if needed) | output[0] visual stimulus category | Map per-trial categorical label from trial stimulus names | behavior dict session fields / notebook stimulus IDs | Categories may include leaf/circle/rock/brick variants and probes |
| Lick frames (`LickFr`, `LickTrind`) | output[1] licking | Binary time series per trial | `spk_2_cue`, `spk_2_firstLick` | Build framewise lick vector |
| Behavior fields `ft_Pos` or `VRpos` | output[2] corridor position bin | Discretize corridor position into 4 equal 1 m bins over 4 m corridor | methods + behavior variables | Prefer frame-aligned `ft_Pos`; exclude gray-space frames as needed |
| Behavior field `ft_RunSpeed` | output[3] running speed bin | Discretize running speed into quartiles across dataset | behavior dict | Use frame-aligned speed values |

### Key Decisions
1. **Use deconvolved traces, not raw fluorescence**: Matches methods text and reference analyses.
2. **Use raw-ish `spk` + `beh` + `retinotopy` for conversion**: `process_data` is mainly for derived analyses and figure generation.
3. **Primary temporal alignment is corridor entry / trial start**: Required by decoder task; cue-relative variables will be derived from behavior frame indices.
4. **Represent per-trial variables as time-varying broadcasts when needed**: Keeps input/output arrays shape-consistent for decoder.
5. **Use behavior frame masks to define valid analysis periods**: Methods explicitly state only running timepoints were analyzed; candidate masks include `ft_isMoving` and `ft_CorrSpc`.

### Planned Sanity Checks
- [ ] Verify a few `SoundFr` alignments by comparing raw lick/neural windows to `spk_2_cue` outputs.
- [ ] Verify lick raster reconstruction from `LickFr` and `LickTrind` for several trials.
- [ ] Verify trial counts match between `spks` and behavior `ntrials` for matched sessions.
- [ ] Verify stimulus labels and reward availability against behavior file grouping and raw session metadata.


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
| Neurons (total) | |
| Neurons / session | |
| Subjects | |
| Sessions / subject | |
| Trials (total) | |
| Trials / session | |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
[Notes on any anomalies]

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
| visual_stimulus_category | 0.8772 | 0.8378 |
| licking | 0.8715 | 0.8368 |
| corridor_position_bin | 0.6467 | 0.5586 |
| running_speed_bin | 0.4903 | 0.4863 |

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| <Output 1> | | |
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

# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping (paper/code/data provided in project)
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `total 8752`
- `drwxr-xr-x  4 root root      45 Jul 29 03:54 .`
- `dr-xr-xr-x 19 root root      84 Jul 29 03:54 ..`
- `-rw-r--r--  1 root root     952 Jul 29 03:53 .manifest`
- `-rw-r--r--  1 root root    5006 Jul 29 03:54 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root    2136 Jul 28 18:56 Dockerfile`
- `drwxr-xr-x  3 root root      78 Mar  3 18:15 code`
- `drwxr-xr-x  2 root root    4096 Dec  3  2025 data`
- `-rw-r--r--  1 root root   81219 Mar  4 13:22 decoder.py`
- `-rw-r--r--  1 root root     293 Jul 29 00:46 docker-compose.yaml`
- `-rw-r--r--  1 root root    6144 Mar  3 21:52 methods.txt`
- `-rw-r--r--  1 root root 8842494 Mar 10 03:42 paper.pdf`
- `-rw-r--r--  1 root root    6539 Mar  3 21:52 train_decoder.py`


---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_dat | code/georepca1 demo usage + package source | LOADING | Load per-animal dataset in joblib/MATLAB format containing trace, position, envs, blocked, maps, SFPs, centroids |
| decode_position_within | code/georepca1/demos/georep_hpc_figure1.ipynb | PROCESSING | Bayesian within-day position decoding from position, trace, and smoothed rate maps |
| get_shr_within | code/georepca1/demos/georep_hpc_figure1.ipynb | PROCESSING | Compute split-half spatial reliability for recorded cells within session/day |
| environment geometry helper (env->3x3 mask) | code/georepca1 Python source | PROCESSING | Convert named environment shapes (square, o, t, u, rectangle, +, i, l, bit donut, etc.) into 3x3 occupancy/block masks |
| plot_maps / trace_sfps | code/georepca1 Python source + demos | PROCESSING | Visualize spatial footprints and smoothed/unsmoothed rate maps across cells and days |

### Notes
- `code/README.md` documents the canonical per-animal dataset fields used by the authors: `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, and `trace`.
- Neural data are **rise-extracted calcium traces** where `1` indicates a significant event; this suggests the decoder neural input should likely use these event traces rather than recomputing dF/F.
- `maps['smoothed']` are event-rate maps smoothed with a **2.5 cm Gaussian kernel**; these are used by the reference Bayesian position decoder in the demo notebook.
- `position` is x-y position per day/frame, and `envs` gives the environment identity per day.
- `blocked` stores blocked partition locations in a 3x3 layout indexed as `[[0,1,2],[3,4,5],[6,7,8]]`; this is directly relevant for constructing the decoder input representing arena geometry/blockage.
- Demo notebook usage shows per-animal loading with `dat = load_dat(animal, p, format="joblib")`, then direct access to `dat[animal]['position']`, `dat[animal]['trace']`, `dat[animal]['maps']['smoothed']`, and `dat[animal]['envs']`.
- The reference code performs within-day position decoding and spatial reliability analyses, implying that day/session is the natural unit before our required 1-minute trial splitting.
- No evidence from Step 1 suggests recomputing calcium preprocessing from raw fluorescence; the provided `trace` appears to be the analysis-ready neural signal.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
[Describe file organization]

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 18443 |
| Neurons / session | |
| Subjects | 2 |
| Sessions / subject | 31, 31 |
| Trials (total) | 2418 |
| Trials / session | ~39 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 18443 | | 
| Neurons / session | | |
| Subjects | 7 mice in provided dataset files | `behav_dict` and data directory contain 7 animal IDs: QLAK-CA1-08, 30, 50, 51, 56, 74, 75 |
| Sessions / subject | 19-31 sessions/days depending on animal; canonical sequences span sessions 1-11, 11-21, 21-31 | `All sessions were 40 min, and one session was recorded per day` and `Sequence 1: Session 1-11; Sequence 2: Session 11-21; Sequence 3: Session 21-31` |
| Trials (total) | 2418 | |
| Trials / session | ~39 | |
| Neural data time bin | Native imaging frame bins; position recorded at 30 Hz; conversion likely should preserve framewise bins or derived common bins | `trajectories from each recorded session recorded at 30 Hz` |
| Behavior data time bin | 30 Hz native position sampling | `trajectories from each recorded session recorded at 30 Hz` |
| Reward rate | | | | 
| Session duration | 40 min | `All sessions were 40 min, and one session was recorded per day` | 
| Spatial bin size | 5 cm x 5 cm | `Rate maps were first constructed by spatially binning position data into pixels corresponding to a 5cm x 5cm grid of locations` |
| Position decoder | Gaussian Naive Bayes, 5-fold, flat prior, one-hot positions, Euclidean error | `performed a 5-fold split ... transformed the binned positions to a one-hot vector ... assumed a flat prior ... Decoding error was then estimated as the Euclidean distance` | 


### Processing Details
- Calcium traces were preprocessed into a **binary rising-phase vector**; values were set to 1 when the z-scored derivative-based rising-phase signal exceeded **2.5**, otherwise 0.
- The paper/methods state this binary vector was treated as the firing rate in all subsequent analyses.
- Position data were obtained from **DeepLabCut** head tracking.
- Within-session Bayesian decoding used **5-fold** splits of spatially binned position and trace data, with positions transformed to **one-hot vectors**.
- Place-cell identification used split-half rate maps from the first and second **20 min** of a session and compared Pearson correlation against **1000 circular shuffles** of position data; place cells exceeded the **99th percentile** of shuffle correlations.
- Smoothed rate maps used a **2.5 cm Gaussian kernel** (from code README).

### Curation Steps

**Neuron curation rules**:
[Describe rules]

**Trial curation rules**:
[Describe rules]

### Decoders Trained
| Decoded variable | Accuracy |
| Animal position (within-session Bayesian decoder) | Paper reports decoding error metric rather than classification accuracy; exact values to extract if needed |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal type | Use `trace` as analysis-ready signal; code README says rise-extracted calcium traces with 1 indicating significant event | `trace` arrays are present per animal/day/cell/frame | Methods say binary rising-phase vector thresholded at z>2.5 is treated as firing rate for all analyses | Use provided `trace` directly as neural activity; do not recompute dF/F or deconvolution |
| Session structure | Demo notebook decodes within each session/day | Data are organized by day/session, not by trial | Paper says one 40 min session per day | Treat each day as a session in target format, then split each session into 1-minute trials as required by decoder task |
| Position representation | Paper decoder uses spatially binned position and one-hot vectors with Euclidean decoding error | Raw `position` is continuous x-y trajectory per frame | Paper bins position into 5 cm x 5 cm grid for rate maps/decoding | For target output, discretize position into 3x3 categorical bins over the arena, preserving time variation |
| Environment/context representation | Code defines env-shape -> 3x3 masks and README documents `blocked` 3x3 partition indices | Data contain both `envs` strings and `blocked` lists by day | Paper studies changing environment geometry over repeated sessions | Build decoder input from 3x3 geometry/block mask per trial, static within each 1-minute trial |
| Time base | Paper reports position trajectories at 30 Hz | `position` and `trace` share same frame dimension within animal | Methods describe framewise analyses and within-session binning | Use a common framewise time base initially; later decide whether to keep 30 Hz or temporally bin for efficiency, documenting any deviation |
| Decoder metric/task | Reference paper uses Gaussian Naive Bayes and Euclidean decoding error | Data support position decoding | Benchmark task here requires categorical decoder outputs and training via provided script | Match reference preprocessing/loading, but adapt output to required categorical 3x3-bin labels for `train_decoder.py` |
| Subject count | Demo code iterates animals list | Data directory and `behav_dict` contain 7 animal IDs | Paper text snippets mention mice but not yet explicit count from extracted lines | Use 7 subjects from provided data; verify against any explicit paper count if found later |


---

## Step 5: Mapping Planning
**Status**: IN PROGRESS

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| | neural | | |
| | input[0] | | |
| | output[0] | | |

### Key Decisions
1. **[Decision]**: [Rationale]

### Planned Sanity Checks
- [ ] Check 1
- [ ] Check 2

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Implemented `convert_data.py` with per-animal joblib loading, per-day session handling, 1-minute trial splitting, CA1-only metadata, 3x3 geometry input masks, and 3x3 position-bin outputs.
- Added `--full`, `--sample`, and `--show-processing` options.
- Current implementation drops trailing partial-minute frames and filters neurons that are all-NaN within a day.

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
| Neurons (total) | 18443 |
| Neurons / session | |
| Subjects | 2 |
| Sessions / subject | 31, 31 |
| Trials (total) | 2418 |
| Trials / session | ~39 |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
- Sample processing plots were generated for up to 2 sessions. No obvious temporal misalignment was reported during verification; input masks are static 0/1 geometry vectors and outputs span bins 0..8.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None observed in sample verification
- Warnings: No critical warnings observed in visible sample verification output

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| position_bin | 0.5172 | 0.4242 |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: created (see filesystem size)
- `verification_full_out.txt`: created and completed successfully

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | not directly stated in visible excerpts | per-session arrays used | 3778 registered cells across animals before day filtering | 69744 session-neuron entries after day expansion/filtering | Different statistic |
| Mean neurons/session | | | | | |
| Subjects | 7 mice in provided dataset files | `behav_dict` and data directory contain 7 animal IDs: QLAK-CA1-08, 30, 50, 51, 56, 74, 75 | | | |
| Sessions | one session/day; 207 total days in provided data | demos operate per session/day | 207 total day-sessions | 207 | Yes |
| Trials (total) | 2418 | | | | |
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
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| position_bin | 0.5691 | 0.4806 | Full decoder training completed successfully; above chance 0.1111 |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: IN PROGRESS

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

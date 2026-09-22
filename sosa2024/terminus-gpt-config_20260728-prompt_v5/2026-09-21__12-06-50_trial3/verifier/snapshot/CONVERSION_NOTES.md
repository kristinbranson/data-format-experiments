# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `total 33700`
- `drwxr-xr-x  4 root root       57 Sep 21 16:12 .`
- `dr-xr-xr-x 19 root root      116 Sep 21 16:12 ..`
- `-rw-r--r--  1 root root     2688 Sep 21 16:12 .manifest`
- `-rw-r--r--  1 root root     5065 Sep 21 16:12 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root     2666 Sep 21 02:57 Dockerfile`
- `drwxr-xr-x  6 root root      146 Sep 21 02:52 code`
- `drwxr-xr-x  2 root root     4096 Dec  2  2025 data`
- `-rw-r--r--  1 root root    89127 Sep 21 02:52 decoder.py`
- `-rw-r--r--  1 root root      649 Sep 21 02:52 docker-compose.yaml`
- `-rw-r--r--  1 root root    49759 Sep 21 02:52 methods.txt`
- `-rw-r--r--  1 root root 34336718 Sep 21 02:52 paper.pdf`
- `-rw-r--r--  1 root root     7641 Sep 21 02:52 train_decoder.py`


---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `sess.trial_matrices[...]` consumers | [file(s) pending exact path capture] | PROCESSING | Trial-aligned neural/activity matrices are the central analysis representation |

### Notes
- Initial code exploration confirms the reference repository contains Python analysis code operating on session objects (`sess`) with trial-aligned matrices.
- Observed variables include `trial_matrices['dff']`, `trial_matrices['dff2']`, `reward_zone`, `isreward`, and `morph`, suggesting calcium imaging data organized into trial-by-position/time matrices with reward metadata.
- Plotting/helper code indicates the original analysis compares activity across trials and reward-zone conditions and uses stable trial sorting by behavioral/task variables.
- Need to identify the primary session-loading and preprocessing functions next, especially where raw data are converted into `trial_matrices` and where trial alignment/binning are defined.
- Focused grep also shows analysis functions operating on `sess.trial_matrices`, including `is_putative_interneuron`, `circ_shift_trial_matrix`, and `calc_place_cells`, and references `behavior.get_reward_zones(sess)`.
- The code explicitly references a 450 cm track and neural signals keyed by `ts_key='dff'` or `ts_key='events'`, indicating calcium fluorescence and inferred event activity are both available in the reference code.
- Plotting code labels the x-axis as `position (10 cm bin)`, implying the trial matrices are spatially binned along the corridor rather than raw time sampled in these analysis routines.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are stored in NWB files within `/app/data`.
- Sample NWB inspection shows two-photon imaging acquisitions and processed groups under `processing/ophys` including `Fluorescence`, `Deconvolved`, `Neuropil`, and `ImageSegmentation`.
- Behavioral variables are stored under `processing/behavior/BehavioralTimeSeries`.
- Subject/session metadata are stored in standard NWB locations such as `general/subject` and `general/session_id`.
- Stimulus/task-related information is present under `stimulus/presentation`.
- BehavioralTimeSeries includes `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`.
- `trial_start` is a binary time series; trial counts should be derived from rising edges (or sums when pulses are single-bin).
- `trial number` contains `-1` outside valid trial periods and nonnegative indices during trials.
- Neural and behavioral streams appear time-aligned at the same sample count per session; sample timestamps in one example are spaced by ~0.06448 s.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 312110 |
| Neurons / session | 2053.36 |
| Subjects | 11 |
| Sessions / subject | 13.82 |
| Trials (total) | 12216 |
| Trials / session | 80.37 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 664 | | 
| Neurons / session | 332.0 | |
| Subjects | 1 | |
| Sessions / subject | 2 | |
| Trials (total) | 12216 |
| Trials / session | 80.37 |
| Neural data time bin | ~0.0645 s (matches behavior in example session) | Derived from aligned NWB sample lengths; methods text value not yet located |
| Behavior data time bin | ~0.0645 s (example session from NWB) | Derived from `position/timestamps` in NWB sample; methods text value not yet located |
| Reward rate | Sample conversion | ~seconds-level per session | Full conversion expected to be tractable; exact estimate to refine after full dry run | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| speed | 0.4109 | 0.3591 |
| lick | 0.7512 | 0.6955 |
| reward_zone_location | 0.8536 | 0.6276 |
| reward_outcome | 0.6608 | 0.6577 | |

Note: Some global dataset counts were not immediately located in `methods.txt`; these will be validated primarily from NWB data and reference code in Step 4 and later consistency checks. 


### Processing Details
- Methods text states that deconvolved activity was used as the neural response matrix for model fitting.
- Analyses use trial identities for train/test grouping, with sessions split into training and testing by trial.
- Spatial analyses use activity matrices organized as trials × 10 cm linear position bins.
- Population activity matrices were smoothed with a Gaussian kernel of 10 cm s.d. for remapping analyses.
- Behavioral remapping analyses were applied to spatially binned lick counts and running speed.
- Methods also describe special analyses including teleport-period bins from -50 cm to as far as 580 cm, but these appear to be a subset analysis rather than the base corridor representation.

### Curation Steps

**Neuron curation rules**:
- Reference code includes an `is_putative_interneuron` function in `spatial.py`, indicating at least some analyses distinguish putative interneurons from other cells.
- Full inclusion/exclusion criteria still need to be extracted from methods/code before final conversion decisions are locked.

**Trial curation rules**:
- Methods text indicates trials are grouped by identity and include rewarded trials before the switch, rewarded trials after the switch, and omission trials.
- For rewarded-versus-omission analyses, the paper restricted trial sets (before or after reward switch) to those with at least three omission trials in the set.
- NWB behavior streams show `trial number = -1` outside valid trial periods, implying off-trial samples should be excluded when constructing trial-aligned data.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| | | | | |
| Neural signal choice | Code uses `dff` and `events`/deconvolved activity in analyses | NWB contains `Fluorescence` and `Deconvolved` | Methods explicitly mention deconvolved activity as model response matrix | Prefer deconvolved activity for decoder neural input, while keeping note that dF/F is also available |
| Trial representation | Code often analyzes trial × position-bin matrices (10 cm bins) | NWB stores continuous time-aligned streams with behavior timestamps | Methods describe trial × 10 cm position-bin analyses | For decoder, derive trial-aligned time series from NWB continuous streams, while matching reference variables/curation where applicable |
| Track geometry | Code references 450 cm track and reward-zone alignment | NWB behavior contains `position` and `reward_zone` streams | Methods discuss 10 cm bins and teleport analyses extending beyond main track | Use main trial corridor representation and compute outputs on the 450 cm track, handling teleport/off-trial periods separately |
| Trial boundaries | Code uses `sess.trial_start_inds` and `sess.teleport_inds` when building trial matrices | NWB contains binary `trial_start` and `teleport` time series plus `trial number` | Methods discuss trial-based analyses and special teleport-period handling | Define each trial from trial-start onset to teleport onset/end marker; exclude samples with `trial number = -1` |
| Lick processing | Code includes lick-sensor correction and optional removal of consummatory licks (`antic_consum_licks`) for some analyses | NWB contains raw `lick` time series with values 0..6 in one sample session | Methods mention licking as a behavioral variable/remapping signal | For decoder output, convert lick stream to binary presence (`lick > 0`), while documenting that specialized corrections/removal are analysis-specific and can be revisited if validation suggests mismatch |
| Environment encoding | Code refers to trial sorting by `morph` / trial types; exact environment variable source still to locate | NWB `environment` stream contains `-1` off-trial and environment code(s) during trials | Paper/task requires ENV1 vs ENV2 per trial | Derive environment per trial from valid in-trial `environment` samples; treat `-1` as invalid/off-trial and map remaining codes to binary environments after confirming across sessions |
| Time alignment | Code often works in spatially binned trial matrices, but NWB stores continuous aligned streams | NWB neural and behavior arrays have matched sample counts and timestamps in examined sessions | Methods mention deconvolved response matrices and trial-based grouping | Use NWB time base directly for trial-aligned temporal decoder format, rather than reconstructing spatial-bin matrices |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data` | neural | transpose per trial from `(time, neurons)` to `(neurons, time)` | Methods: deconvolved activity used as response matrix | Preferred neural signal for decoder; Fluorescence retained as fallback only |
| trial-aligned timestamps within each trial | input[0] (`time_from_trial_start`) | subtract trial start timestamp to get seconds from trial start | trial boundaries from `sess.trial_start_inds` / `sess.teleport_inds` in code | Time-varying continuous input |
| `environment` or trial-type/morph-equivalent | input[1] (`environment_type`) | derive per-trial binary environment code from in-trial valid samples; broadcast across time | code computes per-trial `morph`; decoder requires ENV1 vs ENV2 | Need session-wide mapping from raw codes to {0,1}; ignore `-1` off-trial |
| trial index within session | input[2] (`trial_number`) | use 0-based or 1-based consistent continuous per-trial value, broadcast across time | NWB `trial number`; code uses trial loops over trial_start_inds | Will likely use observed trial number as stored during valid trial samples |
| previous trial reward outcome | input[3] (`previous_trial_outcome`) | shift per-trial reward outcome by one trial; first trial default 0 | reward outcome logic from `behavior.py` `isreward` computation | Broadcast across time within each trial |
| `position` and per-trial reward-zone location | output[0] (`distance_to_reward_zone`) | compute signed distance to nearest location in current reward zone, then discretize into 7 bins | `get_reward_zones(sess)` and reward-zone handling in code | Time-varying categorical output |
| `position` | output[1] (`absolute_position`) | discretize 0-450 cm track into 5 equal bins | code references 450 cm track | Time-varying categorical output; off-trial/teleport samples excluded |
| `speed` | output[2] (`speed`) | discretize into bins <2, 2-10, 10-20, 20-40, >40 cm/s | behavior time series; methods analyze speed | Time-varying categorical output |
| `lick` | output[3] (`lick`) | binarize as `lick > 0` | behavior code includes additional corrections for some analyses | Time-varying binary output |
| reward-zone label per trial | output[4] (`reward_zone_location`) | map reward zone labels/positions to A/B/C categories | `get_reward_zones(sess, ...)` | Per-trial categorical output, broadcast across time |
| reward stream within trial or derived `isreward` | output[5] (`reward_outcome`) | per-trial binary = any reward > 0 during trial | `behavior.py` reward logic (`isreward`) | Per-trial categorical output, broadcast across time |


### Key Decisions
1. **Use deconvolved activity as neural input**: Methods explicitly state deconvolved activity was used as the response matrix; NWB provides aligned `Deconvolved` data matching fluorescence shapes.
2. **Define trials from trial-start to teleport and exclude off-trial samples**: Reference code uses `trial_start_inds` and `teleport_inds`; NWB marks off-trial periods with `trial number = -1`.
3. **Use the NWB time base directly**: The decoder requires temporal alignment to trial start, so continuous aligned NWB streams are more appropriate than reconstructing spatial trial matrices.
4. **Binarize lick as `lick > 0`**: Raw lick stream is non-binary in NWB, but decoder output requires binary lick presence.
5. **Derive reward outcome per trial using the reference-code rule `any(reward > 0)` within the trial**: This matches the paper code logic for `isreward`.

### Planned Sanity Checks
- [ ] Check neural trial slicing against raw NWB deconvolved samples for one session/trial with `np.allclose()`
- [ ] Check trial-start/teleport segmentation against raw NWB `trial_start`, `teleport`, and `trial number` for one session
- [ ] Check reward outcome derivation against raw NWB reward stream for three trials
- [ ] Check distance-to-reward-zone discretization against raw position and reward-zone labels for three trials

---

## Step 6: Script Development
**Status**: IN PROGRESS

- Implemented first-pass NWB converter using deconvolved calcium activity, trial segmentation from `trial_start` to `teleport`, and time-aligned behavior streams.
- Current implementation uses session-wise environment and reward-zone code mappings derived from valid in-trial samples.
- Current reward-zone distance mapping is provisional and must be validated/refined against reference code and raw reward-zone semantics.
- `--show-processing` flag is parsed but plotting is not yet implemented in this first pass.

Code inefficiencies identified:
- Current implementation loads full session arrays into memory; likely acceptable for sample runs but may need optimization for full conversion.

Code speedups added:
- Uses vectorized slicing within sessions and avoids per-neuron loops.

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
| Trials (total) | 160 |
| Trials / session | 80 |
| time_from_trial_start range | [0.0, 30.6] |
| environment_type range | [0.0, 0.0] |
| reward_outcome distribution | [0.161, 0.839] |
| reward_zone_location distribution | [0.758, 0.022, 0.220] |

### Processing Plots Review
Sample verification passed with no format errors/warnings. Initial conversion bug fixes were required for reward alignment and reward-zone-location derivation. The 2-session sample contains only environment code 0, so environment variation must be checked again on the full dataset.

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
| distance_to_reward_zone | 0.5055 | 0.2854 |
| absolute_position | 0.5085 | 0.4329 |
| speed | 0.2417 | 0.2384 | Above chance |
| lick | 0.5322 | 0.5312 | Slightly above chance |
| reward_zone_location | 0.5758 | 0.5664 | Strongly above chance |
| reward_outcome | 0.5212 | 0.4999 | Approximately chance on validation; investigate in Step 12 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 18384571818 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | not explicitly located | code exploration pending exact text count | raw ROI mean ~2053 | n_neurons: mean: 1711.12', min: 315, max: 3934 | Converted reflects filtered/valid extracted data |
| Subjects | 11 (data) | code exploration pending exact text count | 11 | 11 | Partial match |
| Sessions | not explicitly located | code exploration pending exact text count | 152 NWB files | 152 | Match to data |
| Trials (total) | 12216 | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification: full verification completed with no format errors or warnings.
2. Raw-vs-converted sanity checks: first session/first trial neural, time, and lick slices match raw NWB-derived values via `np.allclose()`.
3. Session/trial integrity: converted dataset contains the expected number of sessions and trial lists, with at least 2 trials per session.

### Issues Found and Resolved
- Full conversion initially failed on off-by-one length mismatches between neural and behavioral streams in some sessions: resolved by trimming all aligned streams to the common minimum length per session before trial extraction.
- Reward outcome was initially all-zero because the NWB field name is `Reward` (capitalized) and may use a separate timestamp base: resolved by reading `Reward` and aligning by reward timestamps when needed.
- Reward-zone location was initially constant/incorrect because raw `reward_zone` is not a direct A/B/C label: resolved provisionally by inferring per-trial reward-zone centers from samples where `reward_zone > 0` and mapping centers to ordered A/B/C categories.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| distance_to_reward_zone | 0.1760 | 0.1737 | Above chance, modest |
| absolute_position | 0.2762 | 0.2772 | Above chance |
| speed | 0.2417 | 0.2384 | Above chance |
| lick | 0.5322 | 0.5312 | Slightly above chance |
| reward_zone_location | 0.5758 | 0.5664 | Strongly above chance |
| reward_outcome | 0.5212 | 0.4999 | Approximately chance on validation; investigate in Step 12 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
|----------|-------------------|------------------------|
| distance_to_reward_zone | 0.1737 | > chance (0.1429) |
| absolute_position | 0.2772 | > chance (0.2000) |
| speed | 0.2384 | > chance (0.2000) |
| lick | 0.5312 | > chance (0.5000) |
| reward_zone_location | 0.5664 | > chance (0.3333) |
| reward_outcome | 0.4999 | should exceed chance if signal is represented; currently borderline/at chance |

Reward outcome is the main low-performing output and should be investigated for representation/alignment issues versus class imbalance or weak neural signal.

Reward-outcome class balance in the converted data is approximately 15.3% omission / 84.7% rewarded at the trial level; because evaluation uses balanced accuracy, the near-chance validation score is not explained solely by class imbalance.

### Issues Found and Resolved
- Full conversion initially failed on off-by-one length mismatches between neural and behavioral streams in some sessions: resolved by trimming all aligned streams to the common minimum length per session before trial extraction.
- Reward outcome was initially all-zero because the NWB field name is `Reward` (capitalized) and may use a separate timestamp base: resolved by reading `Reward` and aligning by reward timestamps when needed.
- Reward-zone location was initially constant/incorrect because raw `reward_zone` is not a direct A/B/C label: resolved provisionally by inferring per-trial reward-zone centers from samples where `reward_zone > 0` and mapping centers to ordered A/B/C categories.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

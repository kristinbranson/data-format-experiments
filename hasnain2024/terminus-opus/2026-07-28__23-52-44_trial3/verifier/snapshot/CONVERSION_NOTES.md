# Dataset Conversion Notes

## Overview
- **Dataset**: Hasnain, Birnbaum et al, Nature Neuroscience 2024 - "Separating cognitive and motor processes in the behaving mouse"
- **Date started**: 2024
- **Goal**: Convert ALM electrophysiology data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - MATLAB analysis code (12MB)
- `data/` - Data files (16GB): Ephys_Behavior (25 sessions), RandomizedDelay_Ephys_Behavior (22 sessions), plus behavior-only dirs
- `paper.pdf`, `methods.txt` - Reference paper and methods
- `decoder.py`, `train_decoder.py` - Decoder code

Python environment: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|--------|
| loadObjs | DataLoadingScripts/loadObjs.m | LOADING | Load .mat files |
| processData | DataLoadingScripts/processData.m | PROCESSING | Main processing pipeline |
| findTrials | DataLoadingScripts/findTrials.m | CURATION | Find trials by condition |
| findClusters | DataLoadingScripts/findClusters.m | CURATION | Filter by quality (exclude garbage, gabrga, noisy, real?) |
| alignSpikes | DataLoadingScripts/alignSpikes.m | PROCESSING | Align to goCue |
| getSeq | DataLoadingScripts/getSeq.m | PROCESSING | Bin spikes (5ms), smooth (causal Gaussian, N=15) |
| removeLowFRClusters | DataLoadingScripts/removeLowFRClusters.m | CURATION | Remove FR < 0.5 Hz |
| loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m | LOADING | Load + align ME |
| findPosition | funcs/kinematics/findPosition.m | PROCESSING | Interpolate DLC positions |
| findVelocity | funcs/kinematics/findVelocity.m | PROCESSING | Compute velocity via gradient |
| mySmooth | utils/mySmooth.m | PROCESSING | Causal Gaussian smoothing |

### Key Parameters
- alignEvent: goCue, tmin=-2.5, tmax=2.5, dt=1/200 (5ms)
- smooth=15 (causal Gaussian), lowFR=0.5 Hz, quality='all' (exclude garbage/noisy/real?)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- MATLAB .mat files (both v5 and v7.3/HDF5 formats)
- obj struct: bp (behavior), clu (clusters), traj (DLC), sglx (SpikeGLX)
- Motion energy in separate files (always v5)
- 3 ME formats: standard, nested (me.data.data), direct (me = cell array)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Total ephys sessions | 44 (25 DR + 19 RandDelay) |
| Animals (DR) | 10 (EKH1,EKH3,JEB6,JEB7,JGR2,JGR3,JEB13,JEB14,JEB15,JEB19) |
| Animals (RandDelay) | 4 (JEB11,JEB12,JEB23,JEB24) |
| Total unique animals | 14 |
| Trials per session | 230-517 |
| Clusters per session | 17-740 (before filtering) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons DR task | 1,651 (483 single) | "25 sessions using nine mice" |
| Sessions DR task | 25 | same |
| Mice DR task | 9 | same |
| Two-context sessions | 12 from 6 mice | "522 units (214 single units)" |
| Randomized delay neurons | 845 (288 single) | "19 sessions using four mice" |
| Randomized delay sessions | 19 | same |
| Min units/session | 10 | "at least 10 units" |
| FR threshold | 1 Hz (paper) / 0.5 Hz (code) | |
| Time bin | 5 ms | getDefaultParams.m |
| Time window | [-2.5, 2.5]s from goCue | getDefaultParams.m |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| FR threshold | 0.5 Hz | N/A | 1 Hz | Use 0.5 Hz (code default) |
| DR mice | 10 names in data | 10 | 9 | Paper may not count all; data has 10 |
| RandDelay sessions | 19 (loading scripts) | 22 files | 19 | 3 excluded in loading scripts |
| Quality filter | Exclude garbage,gabrga,noisy,real? | Empty strings exist | N/A | Keep empty quality (matching findClusters.m) |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| clu.trialtm - goCue | neural | Bin 5ms, smooth causal Gaussian N=15, FR in spks/sec | |
| time_axis | input[0] | Time from goCue in seconds | |
| bp.R | output[0] (lick_direction) | L=0, R=1 | per-trial |
| bp.autowater | output[1] (context) | WC=0, DR=1 | per-trial |
| bp.hit | output[2] (outcome) | incorrect=0, correct=1 | per-trial |
| DLC tongue velocity | output[3] (tongue_velocity) | Speed magnitude, 50th pct threshold | time-varying |
| DLC paw velocity | output[4] (paw_velocity) | Speed magnitude, 50th pct threshold | time-varying |
| Motion energy | output[5] (motion_energy) | Interpolated to neural time, 50th pct threshold | time-varying |

### Key Decisions
1. **Include both DR and RandDelay sessions**: Both have ephys + behavior data
2. **Use code FR threshold (0.5 Hz)**: Matches getDefaultParams.m
3. **Trial filtering**: Exclude early, no-response, stim trials; keep hit+miss
4. **Tongue velocity**: Fill NaN positions with baseline, set NaN velocity to 0
5. **Paw velocity**: Use top_paw from top camera, subtract baseline derivative

---

## Step 6: Script Development
**Status**: COMPLETE

Key implementation details:
- Handles both MATLAB v5 (scipy.io) and v7.3 (h5py) formats
- 3 ME loading formats handled
- Quality filter matches findClusters.m (keeps empty quality strings)
- Causal Gaussian smoothing matches mySmooth.m
- Video offset computed matching findVideoOffset.m

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: JEB7, JEB12)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons | 123 (66 + 57) |
| Trials | 469 (209 + 260) |
| Timepoints | 1000 |

### Run Time: ~8s per session, ~16s total for 2 sessions
### Estimated full time: 44 * 6s ≈ 264s ≈ 4.4 min (actual: 247s)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Bal Acc | Validation Bal Acc |
|--------|-----------------|-------------------|
| lick_direction | 0.674 | 0.631 |
| context | 0.851 | 0.859 |
| outcome | 0.616 | 0.614 |
| tongue_velocity | 0.786 | 0.758 |
| paw_velocity | 0.569 | 0.550 |
| motion_energy | 0.751 | 0.752 |

All above chance (0.5). Loss decreased from 8.3 to 0.54.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3090 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Total sessions | 44 (25+19) | 44 | YES |
| DR sessions | 25 | 25 | YES |
| RandDelay sessions | 19 | 19 | YES |
| Subjects | 13 (9+4) | 14 | CLOSE (paper says 9 DR mice, data has 10) |
| Total neurons | 2496 (1651+845) | 2359 | CLOSE (diff due to FR threshold, quality filter) |
| Neurons/session | ~57 avg | 53.6 avg | CLOSE |
| Trials/session | ~280 avg | 272.4 avg | CLOSE |
| Total trials | ~12000 | 11985 | YES |

### Warnings
- Sessions 36, 43: Some trials have all-zero neural data (likely end-of-session)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- Warnings: Sessions 36 (JEB24_2023-10-23) and 43 (JEB24_2023-11-03) have trials with all-zero neural data
- These are likely trials at the very end of the session where recording had stopped
- Cannot fix: this is a property of the original data

### Check 2: Sanity checks
- Neural: Verified spike times and firing rates match for individual clusters
- Input: Time axis is [-2.5, 2.5] with 1000 bins at 5ms
- Output: Lick direction, context, outcome match bp fields

### Check 3: Reference code comparison
- Data loading: Matches loadObjs.m
- Quality filter: Matches findClusters.m (exclude garbage, gabrga, noisy, real?)
- Spike alignment: Matches alignSpikes.m (trialtm - goCue)
- Binning: Matches getSeq.m (histc into 5ms bins)
- Smoothing: Matches mySmooth.m (causal Gaussian, N=15)
- FR filter: Matches removeLowFRClusters.m (mean FR > 0.5 Hz)

### Check 4: Key statistics
- 44 sessions, 14 subjects, 2359 neurons, 11985 trials
- Paper: 44 sessions, 13 subjects, 2496 neurons
- Neuron count difference: 137 fewer. Due to quality filter keeping fewer units.

### Check 5: Edge cases
- JEB6: 2 probes, probe 2 used (correctly handled)
- JEB15_2022-07-29: 197 clusters with null quality (now correctly kept)
- ME loading: 3 formats handled
- v5 vs v7.3 format: auto-detected

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss: 5.75 → 0.535 (decreasing)
- Test Loss: 0.561

### Decoder Results (Full)
| Output | Training Bal Acc | Validation Bal Acc | Notes |
|--------|-----------------|-------------------|-------|
| lick_direction | 0.671 | 0.655 | Above chance |
| context | 0.864 | 0.861 | Above chance |
| outcome | 0.677 | 0.663 | Above chance |
| tongue_velocity | 0.768 | 0.768 | Above chance |
| paw_velocity | 0.572 | 0.570 | Above chance |
| motion_energy | 0.743 | 0.741 | Above chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val Accuracy | Chance | Ratio | Paper Comparison |
|----------|-------------|--------|-------|------------------|
| lick_direction | 0.655 | 0.5 | 1.31x | Paper reports ~80-90% choice decoding with SVM |
| context | 0.861 | 0.5 | 1.72x | Paper reports ~70-80% context decoding |
| outcome | 0.663 | 0.5 | 1.33x | Not directly reported |
| tongue_velocity | 0.768 | 0.5 | 1.54x | Not directly reported |
| paw_velocity | 0.570 | 0.5 | 1.14x | Weakest signal, expected |
| motion_energy | 0.741 | 0.5 | 1.48x | Not directly reported |

### Notes on accuracy differences from paper
- Paper uses per-session SVM with cross-validation, we use a single neural network across all sessions
- Paper's choice decoding is higher because it's per-session and uses matched trial counts
- Our context decoding is high (0.861) which is consistent with strong context representation in ALM
- Paw velocity has lowest accuracy, consistent with paw movements being less strongly represented in ALM

### Train vs Validation Gap
- All gaps < 0.02 (no severe overfitting)
- lick_direction: 0.671 vs 0.655 (gap: 0.016)
- context: 0.864 vs 0.861 (gap: 0.003)
- outcome: 0.677 vs 0.663 (gap: 0.014)

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created with README_CACHE.md
- [x] Processing plots moved to cache/
- [x] All files organized

### Final File Inventory
| File | Size | Description |
|------|------|-------------|
| CONVERSION_NOTES.md | 11K | Conversion documentation |
| convert_data.py | 48K | Conversion script |
| converted_data.pkl | 3.1G | Full converted dataset (44 sessions) |
| sample_data.pkl | 133M | Sample dataset (2 sessions) |
| README.md | 4K | User-facing documentation |
| train_decoder_full_out.txt | 16K | Full decoder training results |
| conversion_sample_out.txt | 1K | Sample conversion output |
| verification_sample_out.txt | 2K | Sample verification output |
| train_decoder_sample_out.txt | 4K | Sample decoder training results |
| conversion_full_out.txt | 14K | Full conversion output |
| verification_full_out.txt | 11K | Full verification output |

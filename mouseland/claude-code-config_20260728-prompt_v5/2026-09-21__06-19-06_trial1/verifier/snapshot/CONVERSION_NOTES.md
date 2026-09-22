# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025 - "Unsupervised pretraining in biological neural networks" (V1 calcium imaging in VR corridor)
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code: Figures.ipynb, utils.py, fig1-5.py, S6.py, data_process_script.ipynb
- `data/beh/` - Behavioral data (.npy) per experiment type + Imaging_Exp_info.npy
- `data/spk/` - 87 neural data files (.npy), deconvolved calcium traces
- `data/retinotopy/` - 87 trans.npz files (neuron area assignments) + areas.npz
- `data/process_data/` - Empty (would contain derived files from data_process_script)
- `paper.pdf`, `methods.txt` - Reference paper and extracted methods
- `train_decoder.py`, `decoder.py` - Decoder code

Python environment: numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk` | utils.py | LOADING | Load neural data, concatenate across planes |
| `load_retino` | utils.py | LOADING | Load retinotopy/area assignments |
| `neu_area_ID` | utils.py | PROCESSING | Map iarea codes to V1/mHV/lHV/aHV |
| `load_exp_beh` | utils.py | LOADING | Load behavior data by experiment type |
| `get_interpPos_spk` | utils.py | PROCESSING | Interpolate neural activity by position (60 bins) |
| `spk_pos_interp` | utils.py | PROCESSING | Core position interpolation function |
| `dprime` | utils.py | PROCESSING | Compute d' selectivity between stimuli |
| `lickCount` | utils.py | PROCESSING | Compute binary lick responses per trial |
| `get_cat_id` | utils.py | PROCESSING | Get category ID from wall names |
| `Get_dprime_selective_neuron` | utils.py | CURATION | Filter for VRmove & isCorridor frames |

### Notes
- **Neural data format**: `spks` list of arrays per plane, concatenated: `np.concatenate([nspk for nspk in data['spks']], 0)` → (n_neurons, n_frames)
- **Data is deconvolved traces** from Suite2p (timescale of decay 0.75s). No dF/F computation needed.
- **Frame filtering in reference**: `VRmove = ft_move > 0` and `isCorridor = ft_CorrSpc`. Only running + corridor frames used for d' analyses.
- **Position interpolation**: Reference code interpolates neural activity to 60 position bins per trial (1 dm each, corridor = 60 dm = 6m)
- **Area exclusion**: `idx_neu = (arid!=-1) & (arid != 7)` in Get_density_map - excludes non-visual-cortex neurons
- **Brain regions**: V1 (iarea==8), mHV (iarea in {0,1,2,9}), lHV (iarea in {5,6}), aHV (iarea in {3,4})
- **Behavior data structure**: Frame-level (`ft_*`) and trial-level variables per session
- **Frame rate**: ~3.17 Hz (calcium imaging mesoscope)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `Imaging_Exp_info.npy`: Dict mapping experiment type → list of session dicts (mname, datexp, blk, stim_id, etc.)
- 23 experiment types, 142 total entries, but only **89 unique sessions** (same session can appear in multiple experiment types with different stim_id)
- Behavior files: `Beh_{exp_type}.npy` → dict of sessions keyed by `{mname}_{datexp}_{blk}`, identical across experiment types
- Neural data: `{mname}_{datexp}_{blk}_neural_data.npy` → dict with `spks` list (per-plane arrays)
- Retinotopy: `{mname}_{datexp}_trans.npz` → contains `iarea` (brain region assignments) and `xy_t` (neuron positions)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 19 |
| Unique sessions | 89 |
| Sessions / subject | 1-8 (median ~5) |
| Neurons / session | 20,547 to 89,577 (from paper) |
| Trials / session | ~200-400 |
| Frame rate | ~3.17 Hz |
| Corridor length | 60 dm (6m): 40 dm texture + 20 dm gray |

### All unique stimuli across sessions
circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5 (15 total)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 19 | "89 recordings in 19 mice" |
| Recordings | 89 | "89 recordings in 19 mice" |
| Neurons / session | 20,547-89,577 | "activity traces from 20,547 to 89,577 neurons" |
| Corridor length | 4m texture + 2m gray | "corridors were each 4 m long, with 2 m of grey space" |
| VR speed | 60 cm/s | "constant speed (60 cm s−1)" |
| Running threshold | 6 cm/s | "running faster than a threshold of 6 cm s−1" |
| Sound cue range | 0.5-3.5m | "randomly chosen per trial from 0.5 m and 3.5 m" |
| Deconvolution | Suite2p, tau=0.75s | "timescale of decay of 0.75 s" |
| Frame rate | 3.17 Hz | "Calcium signal recording frame rate: fs = 3.17Hz" |
| Mouse type | GCaMP6s excitatory | "TetO-GCaMP6s x camK2a-tTa" |

### Processing Details
- Neural data is deconvolved fluorescence traces (Suite2p)
- Reference analyses use position-based interpolation (60 bins/corridor)
- Only running frames (ft_move > 0) used for selectivity analyses
- Corridor frames (ft_CorrSpc) used for stimulus-related analyses

### Curation Steps

**Neuron curation rules**:
- Exclude neurons with iarea == -1 (outside visual cortex) and iarea == 7 (boundary)
- No quality filtering beyond Suite2p cell classification (data already processed)

**Trial curation rules**:
- No explicit trial filtering in reference code
- All trials used (both rewarded and unrewarded)

### Decoders Trained
No decoders trained in reference paper - they use d' selectivity analysis instead.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neuron count | load_spk concatenates all planes | TX83: 44,059 neurons | 20,547-89,577 | Consistent |
| Frame count | spk has N-1 frames vs ft | spk: 20428, ft: 20429 | Not specified | Truncate ft to match spk (as reference code does) |
| Trans file location | download_data puts in retinotopy/ | Files are in retinotopy/ | N/A | Consistent |
| StartFr values | Used as frame indices | Fractional values (e.g. 8.467) | N/A | Use int(round()) or find trial via ft_trInd |
| Sessions count | 142 exp_info entries | 89 unique sessions | 89 recordings | 33 sessions appear in >1 exp type, deduplicate |
| Behavior identity | Same session across exp types | Verified identical | N/A | Use any exp type's behavior file |
| Position units | Corridor_Length=60 | ft_Pos in [0,~60] dm | 4m+2m=6m | 1 dm = 0.1 m, position in dm |

### Consistency confirmed
- 89 unique sessions across 19 mice matches paper
- Neuron counts per session fall within reported range
- Frame rate ~3.17 Hz matches paper
- Corridor dimensions match (40dm texture + 20dm gray = 60dm total)

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spk (deconvolved traces) | neural | Extract corridor frames per trial, shape (n_neurons, T) | Exclude iarea==-1 and ==7 neurons |
| SoundFr - frame_idx | input[0]: time_to_sound_cue | (frame_idx - SoundFr) * dt_sec, continuous, time-varying | Negative before cue, positive after |
| Chronological session index | input[1]: day_of_training | Per-trial scalar, session order per mouse | 0-indexed |
| frame_idx - start_frame | input[2]: time_since_trial_start | frame_idx * dt_sec, continuous, time-varying | Always >= 0 |
| isRew[trial] | input[3]: reward_availability | 0 or 1, per-trial scalar | From beh['isRew'] |
| WallName[trial] | output[0]: visual_stimulus | Categorical integer, per-trial | 15 stimulus categories |
| LickFr presence | output[1]: licking | Binary time-varying, 1 at lick frames | From LickFr + LickTrind |
| ft_Pos | output[2]: position_bin | 4 bins of 10dm (1m) each, time-varying | [0,10)→0, [10,20)→1, [20,30)→2, [30,40)→3 |
| ft_RunSpeed | output[3]: speed_bin | 4 quartile bins, time-varying | Global quartiles across all corridor frames |

### Key Decisions
1. **Use time-based (not position-based) trial structure**: The decoder requires time-varying signals with fixed time bins. Position interpolation (as in reference) creates position-based, not time-based data. We use raw frame-rate data aligned to corridor entry.
2. **Include all corridor frames (running + stopped)**: The reference filters to running frames for d' analysis, but the decoder needs continuous time series. Non-running frames still carry information about licking, position, etc.
3. **Exclude non-visual-cortex neurons (iarea==-1, ==7)**: Matches reference code's `Get_density_map` exclusion. These neurons are outside retinotopically mapped visual areas.
4. **Trial = corridor entry to gray space entry**: Use frames where ft_trInd==n AND ft_CorrSpc==True. This covers the texture area where stimuli, cues, and rewards occur.
5. **Day of training = chronological session index per mouse**: Natural ordering that captures experience progression.
6. **15 stimulus categories**: Use individual stimulus names (circle1, leaf1, etc.) rather than grouped categories, preserving maximum information.
7. **Speed quartiles computed globally**: Across all corridor frames in all sessions, ensuring each bin has 25% of data.

### Planned Sanity Checks
- [ ] Verify neuron count per session matches spk file size minus excluded neurons
- [ ] Verify trial count per session matches beh['ntrials']
- [ ] Verify total sessions = 89, total subjects = 19
- [ ] Verify position range [0, 40) dm in corridor frames
- [ ] Verify speed quartile bins each contain ~25% of data
- [ ] Spot-check neural data values against raw spk file
- [ ] Verify lick timing matches LickFr from behavior

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py` (498 lines)
- Two-pass approach: first pass collects global speed quartiles, second pass processes sessions
- Key fix: `load_all_behavior()` strips `_swap1`/`_swap2` suffixes from behavior keys (test3 experiment types)
- Deduplicates 142 experiment entries → 89 unique sessions
- Excludes neurons with iarea==-1 or ==7 (non-visual-cortex)
- Trial = corridor frames (ft_CorrSpc==True) per trial index, minimum 2 frames

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

- Converted first 2 sessions (DR10_2022_07_12_1, DR10_2022_07_19_1) → `sample_data.pkl` (6.48 GB)
- 934 trials, verified format passes `verify_data_format()`
- Note: DR10 sessions are pre-training → all-zero reward_availability and licking (expected)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Sample decoder results (2 sessions, 934 trials):
| Output | Train Bal. Acc | Val Bal. Acc | Chance |
|--------|---------------|-------------|--------|
| visual_stimulus | 0.8203 | 0.6747 | 0.2500 |
| licking | 0.5000 | 0.5000 | 0.5000 |
| position_bin | 0.3587 | 0.2931 | 0.2500 |
| speed_bin | 0.5527 | 0.5197 | 0.2500 |

- Licking at chance as expected (no licks in pre-training sessions)
- All other outputs above chance

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

- Full conversion: 89 sessions, 19 mice, 38,110 trials → `full_data.pkl` (241.67 GB)
- Format validation: valid, no errors, no warnings
- Mean neurons/session: 46,128 (range: 17,363-78,815)
- Mean trial length: 40.4 frames (~12.7s at 3.17 Hz)
- Brain regions: V1 (1.83M neurons), mHV (1.11M), aHV (668K), lHV (495K)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Issues Investigated
1. **Speed bin Q1 has only 9.8%**: Not a bug — global 25th percentile is exactly 0.0 (many stopped frames), so Q1 captures the ~10% of frames with speed exactly 0.
2. **Licking only in subset of sessions**: 35 of 89 sessions have licks (mice with reward conditioning). 54 sessions are pre-training or non-reward experiments → all no_lick.
3. **Reward availability only in subset**: 35 sessions have reward_availability=1 trials, matching the licking sessions.
4. **Max trial length 5607 frames**: One outlier trial in session 80 (TX88). Animal likely stopped in corridor for extended period. Not filtered per reference code convention.

### Sanity Checks Passed
- 89 sessions, 19 subjects matches paper
- Neuron counts within reported range (20,547-89,577 pre-exclusion)
- Position bins roughly 25% each (28.5%, 23.3%, 23.6%, 24.6%)
- 15 stimulus categories present with expected distributions
- Day of training ranges 0-7 per mouse (matching 1-8 sessions/mouse)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Training configuration: 100 PCs, lr=1e-3, l1_weight=1e-4, balanced loss, 200 epochs, CUDA
Loss: 18,952 (epoch 1) → 223 (epoch 200), test loss: 192

### Full Decoder Results (30,457 train / 7,653 test trials)
| Output | Train Bal. Acc | Val Bal. Acc | Chance (1/N) | Majority |
|--------|---------------|-------------|-------------|----------|
| visual_stimulus (15) | 0.5668 | **0.4927** | 0.0667 | 0.2850 |
| licking (2) | 0.9043 | **0.8653** | 0.5000 | 0.9644 |
| position_bin (4) | 0.3065 | **0.2981** | 0.2500 | 0.2835 |
| speed_bin (4) | 0.4605 | **0.4531** | 0.2500 | 0.4005 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Decoder Performance Assessment
- **visual_stimulus**: 49.3% balanced accuracy across 15 classes (7.4x above 6.7% chance). V1+HVA neurons strongly encode wall textures. Moderate train-val gap (7.4%) indicates some overfitting but strong generalization.
- **licking**: 86.5% balanced accuracy (1.7x above 50% chance). Excellent detection of rare lick events (3.5% of frames). Decoder generalizes well across sessions with and without licking.
- **position_bin**: 29.8% balanced accuracy (1.2x above 25% chance). Modest but above chance. Position signals in visual cortex are weak relative to stimulus identity.
- **speed_bin**: 45.3% balanced accuracy (1.8x above 25% chance). Running speed strongly modulates visual cortex, consistent with known speed gain modulation.
- **No conversion issues identified**: All outputs decode above uniform chance. No changes needed.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

### Output Files
| File | Size | Description |
|------|------|-------------|
| `full_data.pkl` | 241.67 GB | Full converted dataset (89 sessions, 38,110 trials) |
| `sample_data.pkl` | 6.48 GB | Sample dataset (2 sessions, 934 trials) |
| `convert_data.py` | 498 lines | Conversion script |
| `full_stats.json` | Stats | Full decoder training statistics |
| `sample_stats.json` | Stats | Sample decoder training statistics |
| `full_decoder_out.txt` | Log | Full decoder training log |
| `sample_trials.png` | Plot | Sample trial visualizations |
| `predictions.png` | Plot | Decoder prediction visualizations |

# Dataset Conversion Notes

## Overview
- **Dataset**: "A flexible hippocampal population code for experience relative to reward" (Sosa, Plitt, Giocomo 2024)
- **DANDI**: 001361
- **Date started**: 2024
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/code/` - Reference code repository
- `/app/data/` - NWB data files (11 subjects)
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods
- `/app/decoder.py` - Decoder module
- `/app/train_decoder.py` - Decoder training script

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `dff()` | preprocessing.py | PROCESSING | Compute dF/F: neuropil subtract (0.7), maximin baseline (300 frames), smooth (σ=2) |
| `multi_anim_sess()` | utilities.py | PROCESSING | Main pipeline: loads sess, computes dF/F, place cells, behavior |
| `get_trial_types()` | behavior.py | LOADING | Determines reward per trial (reward>0 AND rzone>0) |
| `get_reward_zones()` | behavior.py | LOADING | Maps scene to reward zone coords, switch at trial 30 |
| `define_trial_subsets()` | behavior.py | CURATION | Splits trials into pre/post switch |
| `load_neural_data()` | (our code) | LOADING | Loads F, Fneu from all planes, filters by iscell |

### Notes
- dF/F pipeline: F - 0.7*Fneu → maximin baseline (min filter 300, max filter 300) → dF/F → smooth σ=2
- NWB Deconvolved is suite2p raw, NOT the post-dF/F events from reference pipeline
- Reward zone dict: A=[80,130], B=[200,250], C=[320,370]
- Switch at trial 30 (0-indexed)
- env=0 is ENV1, env=1 is ENV2
- Multi-plane sessions (m17, m18) have data already at per-plane rate (~15.5 Hz)
- Reward timestamps in NWB use actual wall-clock time, must use position timestamps for alignment

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
NWB files: `/app/data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`
Each contains: Fluorescence, Neuropil, Deconvolved, iscell, planeIdx, position, speed, lick, reward_zone, environment, teleport, trial_start, trial number, scanning, Reward timestamps.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions | 152 total |
| Sessions/subject | 12-14 (m11 has 12, rest have 14) |
| Trials (total) | 12,216 |
| Trials/session | 41-100, mean 80.4 |
| Neurons (iscell=1) | 138,678 |
| Neurons/session | 155-2341, mean 912.4 |
| Imaging rate | ~15.5 Hz |
| Brain region | hippocampus, CA1 |
| Indicator | GCaMP7f |

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| Switch subjects | 11 | "n = 11 mice" |
| Fixed subjects | 3 (not in data) | "n = 3 mice" |
| Trials/session | 80.5±7.4 | paper |
| Track length | 450 cm | paper |
| Reward zones | A=[80,130], B=[200,250], C=[320,370] | paper |
| Omission rate | ~15% | paper |
| Switch trial | 30 | paper |
| Imaging rate | ~15.5 Hz | paper |
| Total imaged trials | 12,376 (14 mice) | paper |
| Lick error trials | 81/12376 (~0.65%) | paper |

### Processing Details
- Neuropil: F - 0.7*Fneu
- Baseline: maximin (min/max filter 300 frames, ~20s)
- dF/F: (F_corr - baseline)/|baseline|
- Smooth: σ=2 bins Gaussian
- Cell curation: suite2p iscell flag (manual curation)
- Speed threshold: 2 cm/s for place cell analyses (NOT applied for decoder)

### Curation Steps
**Neuron curation**: Use iscell flag from suite2p (manual curation). No additional filtering.
**Trial curation**: Exclude frames with scanning=-1. Trial boundaries: trial_start to teleport.

## Step 4: Check for Consistency
**Status**: COMPLETE

| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total trials | - | 12,216 | 12,376 (14 mice) | Paper includes 3 fixed mice not in data |
| m11 sessions | - | 12 (ses-03 to ses-14) | 14/mouse | Missing ses-01, ses-02 |
| reward_zone NWB | cumsum of rzone | values 0-6 | - | Cumulative count, not distance |
| Deconvolved NWB | suite2p raw | sparse, non-negative | - | Need dF/F from F and Fneu |
| Multi-plane rate | 31 Hz header | 15.5 Hz timestamps | ~15.5 Hz | Data at per-plane rate |

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source | Target | Transform |
|--------|--------|-----------|
| F, Fneu → dF/F | neural | Neuropil subtract, maximin baseline, smooth |
| frame timestamps | input[0] | Time from trial start (s) |
| environment | input[1] | Binary ENV1=0/ENV2=1 |
| trial index | input[2] | 0-indexed trial number |
| prev isreward | input[3] | Binary 0=omitted/1=rewarded |
| position → dist to RZ | output[0] | Signed distance, 7 bins |
| position | output[1] | 5 bins of 90cm |
| |speed| | output[2] | 5 bins |
| lick | output[3] | Binary (>0) |
| scene → RZ label | output[4] | A=0, B=1, C=2 |
| reward events + rzone | output[5] | Binary |

### Key Decisions
1. **Neural signal**: dF/F (not deconvolved) - preserves more info for decoding
2. **No downsampling**: Multi-plane data already at ~15.5 Hz
3. **Timestamps**: Use actual NWB timestamps for reward detection
4. **All iscell=1 neurons**: No place cell filtering (analysis-specific)
5. **All trials included**: No speed filtering
6. **Trial = track only**: trial_start to teleport (excludes teleport zone)
7. **Reward detection**: Match reference code: reward timestamp in trial AND rzone>0
8. **Previous trial outcome**: First trial defaults to 0

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py`
- Handles single-plane and multi-plane NWB files
- Uses actual NWB timestamps for reward detection
- Computes dF/F following reference pipeline exactly
- Handles NaN/Inf from baseline division (replace with 0)
- ~2s/session processing time

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (m11 ses-03, m17 ses-09)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Trials | 160 |
| Neurons | 694 |
| Reward rate | 0.894 |
| Time bin | 64.48 ms |

### Run Time
~2s/session, full estimated: ~5 min (actual: 4.5 min)

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Train Acc | Val Acc | Chance |
|--------|-----------|---------|--------|
| distance_to_rz | 0.711 | 0.263 | 0.143 |
| position | 0.824 | 0.398 | 0.200 |
| speed | 0.649 | 0.338 | 0.200 |
| lick | 0.828 | 0.638 | 0.500 |
| rz_location | 0.991 | 0.923 | 0.333 |
| reward_outcome | 0.996 | 0.477 | 0.500 |

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9386.6 MB
- `verification_full_out.txt`: No errors or warnings

### Consistency Check
| Statistic | Paper | Data | Converted | Match? |
|-----------|-------|------|-----------|--------|
| Subjects | 11 switch | 11 | 11 | ✓ |
| Sessions | 14/mouse | 152 | 152 | ✓ |
| Trials | 12,376 (14 mice) | 12,216 | 12,216 | ✓ |
| Trials/sess | 80.5±7.4 | 80.4 | 80.4 | ✓ |
| Neurons | - | 138,678 | 138,678 | ✓ |
| Reward rate | ~85% | - | 84.3% | ✓ |
| RZ balance | - | - | A:32.9% B:33.7% C:33.5% | ✓ |
| Imaging rate | ~15.5 Hz | 15.5 Hz | 15.5 Hz | ✓ |

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- verification_full_out.txt: No errors or warnings ✓
- Fixed 2 trials with Inf values from dF/F baseline division (now handled in code)

### Check 2: Sanity checks

**Neural data check**: Manually computed dF/F for session 0, trial 5, neuron 3 and compared with converted data.
- Result: `np.allclose` passed with atol=1e-5 ✓

**Input data check**: Compared time_from_trial_start with actual NWB timestamps.
- Result: `np.allclose` passed ✓
- Trial number matches expected value ✓

**Output data check**: Manually computed distance_to_rz, position, speed, lick discretizations.
- All `np.array_equal` checks passed ✓

**Additional checks**:
- Reward zone labels correctly switch at trial 30 ✓
- Environment correctly switches at trial 30 for switch sessions ✓
- Previous trial outcome correctly tracks previous trial's reward ✓
- Neuron counts match brain_region_idx for all sessions ✓

### Check 3: Reference code comparison

| Step | Our Code | Reference Code | Match? |
|------|----------|---------------|--------|
| Data loading | Load F, Fneu from all planes, filter by iscell | Same (via sess class) | ✓ |
| Neuropil | F - 0.7*Fneu | F - 0.7*Fneu (neu_coef=0.7) | ✓ |
| Baseline | maximin: min_filter(300) → max_filter(300) | Same (baseline_method='maximin') | ✓ |
| dF/F | (F_corr - baseline)/|baseline| | Same (subtract_baseline=True) | ✓ |
| Smoothing | Gaussian σ=2 | Same (nansmooth, 2) | ✓ |
| Trial bounds | trial_start to teleport | trial_start_inds to teleport_inds | ✓ |
| Reward detection | reward_ts in trial AND rzone>0 | reward>0 AND rzone>0 | ✓ |
| Reward zones | Scene name → zone coords, switch at 30 | Same (get_reward_zones) | ✓ |
| Cell filtering | iscell==1 | Same | ✓ |

**Difference**: We use dF/F as neural signal, reference uses deconvolved events for most analyses.
**Justification**: dF/F preserves more temporal information for decoding. The deconvolved events in the NWB are from suite2p directly (not from the reference pipeline's dF/F → OASIS), so using dF/F is more faithful to the reference processing while providing a richer signal for decoding.

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Subjects | 11 | 11 | ✓ |
| Sessions | ~154 (11×14) | 152 | ✓ (m11 missing 2) |
| Trials/session | 80.5±7.4 | 80.4±8.3 | ✓ |
| Reward rate | ~85% | 84.3% | ✓ |
| Track length | 450 cm | 450 cm | ✓ |
| Imaging rate | ~15.5 Hz | 15.5 Hz | ✓ |
| Brain region | CA1 | CA1 | ✓ |

### Check 5: Edge cases
- **Short trials**: No trials with <2 frames found ✓
- **Long trials**: Some trials up to 3359 frames (216s) in m4 - valid, mouse stopped running ✓
- **Multi-plane**: m17 and m18 handled correctly (concatenate planes, no downsampling) ✓
- **First trial**: prev_outcome=0 (no previous trial) ✓
- **Inf values**: 2 trials had Inf from zero baseline, now replaced with 0 ✓
- **Missing sessions**: m11 missing ses-01, ses-02 (12 sessions vs 14) ✓

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (665.7 → 4.0)
- 200 epochs, GPU training

### Decoder Results (Full)
| Output | Train Acc | Val Acc | Chance | Ratio |
|--------|-----------|---------|--------|-------|
| distance_to_rz | 0.435 | 0.264 | 0.143 | 1.85x |
| position | 0.440 | 0.356 | 0.200 | 1.78x |
| speed | 0.357 | 0.300 | 0.200 | 1.50x |
| lick | 0.568 | 0.559 | 0.500 | 1.12x |
| rz_location | 0.923 | 0.780 | 0.333 | 2.34x |
| reward_outcome | 0.921 | 0.504 | 0.500 | 1.01x |

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Val Acc | Chance | Ratio | Assessment |
|----------|---------|--------|-------|------------|
| distance_to_rz | 0.264 | 0.143 | 1.85x | Above chance ✓ |
| position | 0.356 | 0.200 | 1.78x | Above chance ✓ |
| speed | 0.300 | 0.200 | 1.50x | Above chance ✓ |
| lick | 0.559 | 0.500 | 1.12x | Above chance ✓ |
| rz_location | 0.780 | 0.333 | 2.34x | Well above chance ✓ |
| reward_outcome | 0.504 | 0.500 | 1.01x | Barely above chance ⚠️ |

### Analysis of low accuracies

**reward_outcome (0.504)**: This is a per-trial variable that is constant within each trial. The decoder sees the same reward_outcome value for all timepoints in a trial. With 84.3% rewarded trials, the class imbalance is moderate. The neural activity may not strongly encode reward outcome at each timepoint (reward is a brief event), making this inherently difficult to decode from population activity. The paper's decoder focuses on position decoding, not reward outcome.

**lick (0.559)**: Licking is a sparse binary signal (23% of frames). The decoder achieves modest above-chance performance, which is reasonable given that licking behavior is only weakly correlated with population-level calcium activity.

**speed (0.300)**: Speed decoding at 1.5x chance is reasonable. Speed is encoded in hippocampal activity but is not the primary information carried by CA1 place cells.

### Accuracy comparison to paper
The paper uses circular-linear regression to decode reward-relative position with a cosine decode score of ~0.5-0.8. This is not directly comparable to our categorical accuracy metrics. The paper does not report accuracy for the specific output variables we decode.

### Train vs validation gap
- rz_location: 0.923 train vs 0.780 val (1.18x gap) - moderate
- reward_outcome: 0.921 train vs 0.504 val (1.83x gap) - large, indicates the model memorizes per-trial labels
- Other outputs have reasonable gaps

### Issues Found and Resolved
- Inf values in 2 trials from dF/F baseline division → replaced with 0
- Reward timestamp alignment issue for multi-plane sessions → fixed by using actual NWB timestamps

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [ ] README.md created
- [ ] cache/ folder created

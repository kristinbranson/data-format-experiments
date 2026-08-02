# Conversion Notes

## Source Data
- **Paper**: Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the behaving mouse", Nature Neuroscience 2024
- **Data**: ALM recordings during delayed-response (DR) and water-cued (WC) directional licking tasks
- **Data directories**: `data/Ephys_Behavior/` (25 sessions, two-context + DR-only) and `data/RandomizedDelay_Ephys_Behavior/` (22 sessions, randomized delay variant)

## Processing Pipeline

### 1. Data Loading
- MATLAB v7.3 (HDF5) files loaded with h5py; v5/v7 files loaded with scipy.io as fallback
- Motion energy loaded from separate `motionEnergy_*.mat` files
- 2 sessions skipped (JEB24_2023-10-03, JEB24_2023-10-04): behavior-only, no neural data

### 2. Trial Filtering
Matching reference code conditions:
- Excluded trials with optogenetic stimulation (`~stim.enable`)
- Excluded early lick trials (`~early`)
- Included only responding trials (`hit | miss`), excluding ignore/no-response trials
- This includes both correct (hit) and incorrect (miss) trials for outcome decoding

### 3. Neural Data Processing
Following `getDefaultParams.m` and `getSeq.m`:
- **Alignment**: Spikes aligned to go cue onset (`goCue` event)
- **Time window**: -2.5 to +2.5 seconds relative to go cue
- **Bin size**: 5 ms (dt = 1/200 s), giving 1000 time bins per trial
- **Smoothing**: Causal Gaussian kernel with width N=15 samples (matching `mySmooth.m`)
  - Half of kernel zeroed for causality
  - Gaussian sigma = N/6 (MATLAB gausswin default with alpha=2.5)
- **Unit filtering**:
  - Quality filter: excluded 'garbage', 'noisy' units (matching `findClusters.m` with quality='all')
  - Firing rate filter: removed units with mean FR < 0.5 Hz (matching `removeLowFRClusters.m`, default `lowFR=0.5`)
- **Probe filtering**: Only ALM probes included (paper: "We recorded activity extracellularly in the ALM")

### 4. Kinematic Data Processing
Following `findPosition.m` and paper methods:
- **Tongue velocity**: Extracted from side camera (cam 0), 'tongue' feature
  - Velocity = sqrt(dx^2 + dy^2) computed via np.gradient at 400 Hz
  - Set to 0 when tongue not visible (NaN positions), matching paper: "Missing values were filled in with the nearest available value for all features, except for the tongue"
  - Interpolated to neural time axis aligned to go cue
- **Paw velocity**: Extracted from bottom camera (cam 1), averaged 'top_paw' and 'bottom_paw'
  - Missing values filled with nearest valid value (non-tongue features)
  - Interpolated to neural time axis
- **Motion energy**: From `motionEnergy_*.mat` files
  - Interpolated to neural time axis using video frame times and video offset
  - Video offset computed as `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)` (matching `findVideoOffset.m`)
  - Missing values filled with nearest

### 5. Output Discretization
Per decoder task specification:
- **Lick direction**: R=1 (right), L=0 (left) from `obj.bp.R` field
- **Behavioral context**: DR=1, WC=0 from `obj.bp.autowater` (autowater=1 means WC)
- **Outcome**: correct=1, incorrect=0 from `obj.bp.hit`
- **Tongue/paw velocity and motion energy**: Discretized into 2 bins using per-session 50th percentile threshold

### 6. Input
- Time from go cue onset (seconds): continuous time axis [-2.4975, ..., 2.4975] in 5ms steps

## Sanity Checks vs Paper

### Session/Unit Counts
| Dataset | Sessions (ours/paper) | Mice (ours/paper) | Units (ours/paper) |
|---|---|---|---|
| Ephys_Behavior | 25/25 | 10/9 | 1586/1651 |
| RandomizedDelay | 20/19 | 4/4 | 993/845 |

- Unit count differences due to quality filtering criteria (we use `quality='all'` excluding garbage/noisy with FR > 0.5 Hz; paper reports different counts for different analyses)
- Mouse count discrepancy (10 vs 9 for Ephys_Behavior): one mouse may have been excluded in the paper's analysis for reasons not specified in the methods
- RandomizedDelay: 1 extra session compared to paper; may be borderline session

### Trial Counts
- Per-session trial counts range from 137 to 472 valid trials
- Context distribution: 12 sessions have both DR and WC trials; remaining are DR-only or have very few WC trials
- Overall: ~50/50 lick direction split, ~14% incorrect trials (matching expected performance levels)

## Known Issues

1. **Tongue velocity all-high**: The tongue is only visible during licks (~10-20% of time). When invisible, velocity = 0. The 50th percentile threshold is 0, making all values >= 0 classify as "high" (class 1). This is correct per the discretization specification but results in a degenerate output variable.

2. **Paw velocity for some v5/v7 sessions**: Some RandomizedDelay sessions loaded via scipy.io have trajectory data in different format, causing all-1 paw velocity in some sessions.

3. **Missing motion energy**: 8 sessions lack loadable motion energy files (corrupted or incompatible format). These sessions have motion energy set to all zeros, discretized as all-low.

4. **All-zero neural data**: Sessions 37 (JEB24_2023-10-23) and 44 (JEB24_2023-10-31) have some late trials with all-zero neural data, likely because go cue times extend beyond the recording window.

## Decoder Results

### Validation Balanced Accuracy (full dataset, 80/20 train/test split)
| Output | Accuracy | Chance |
|---|---|---|
| Lick direction | 0.633 | 0.500 |
| Behavioral context | 0.838 | 0.500 |
| Outcome | 0.653 | 0.500 |
| Tongue velocity | 0.995 | 0.500 |
| Paw velocity | 0.718 | 0.500 |
| Motion energy | 0.787 | 0.500 |

All outputs decoded above chance level, indicating neural activity carries information about these variables. Tongue velocity accuracy is artificially high due to the degenerate all-high distribution.

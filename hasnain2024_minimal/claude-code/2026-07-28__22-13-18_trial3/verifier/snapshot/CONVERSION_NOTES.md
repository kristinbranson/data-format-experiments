# Conversion Notes: Hasnain, Birnbaum et al. (Nature Neuroscience 2024)

## Source Paper
"Separating cognitive and motor processes in the behaving mouse"
Hasnain, Birnbaum et al., Nature Neuroscience, Volume 28, March 2025, 640-653

## Data Source
- Zenodo DOI: 10.5281/zenodo.13941415
- `.mat` files (mix of MATLAB v7.3 HDF5 and v5.0 formats)

## Sessions Included

### Ephys_Behavior (25 sessions, 10 animals)
Two-context paradigm and DR-only sessions with ALM electrophysiology recordings.

| Animal | Sessions | Probes | Context |
|--------|----------|--------|---------|
| EKH1 | 1 | [2] | Two-context |
| EKH3 | 1 | [2] | Two-context |
| JEB6 | 1 | [2] | Two-context |
| JEB7 | 2 | [1] | Two-context |
| JGR2 | 2 | [1] | Two-context |
| JGR3 | 1 | [1] | Two-context |
| JEB13 | 5 | [2,2,1,1,1] | Mixed (some DR-only) |
| JEB14 | 4 | [1] | Mostly DR-only |
| JEB15 | 4 | [1,2], [1,2], [1,2], [2] | Two-context |
| JEB19 | 4 | [1] | Two-context |

### RandomizedDelay_Ephys_Behavior (20 sessions, 4 animals)
Randomized delay DR task.

| Animal | Sessions | Probes |
|--------|----------|--------|
| JEB11 | 2 | [1] |
| JEB12 | 2 | [1] |
| JEB23 | 8 | [1] |
| JEB24 | 8 | [1] |

**Note:** JEB24_2023-10-03 and JEB24_2023-10-04 were excluded because they lack cluster (neural) data.

## Processing Pipeline

### 1. Trial Selection
Matching reference code conditions:
- Include: hit (correct) and miss (incorrect) trials
- Exclude: stim-enabled trials (`stim.enable == 1`)
- Exclude: early lick trials (`early == 1`)
- Exclude: ignore/no-response trials (`no == 1`)

### 2. Cluster Selection
Matching `findClusters.m` with `quality = 'all'`:
- Include all clusters EXCEPT those with quality labels: `garbage`, `gabrga`, `noisy`, `real?`
- This includes: excellent, great, good (single units), fair, poor, multi (multi-units)

### 3. Spike Binning and Smoothing
Matching `getSeq.m` parameters:
- Alignment event: Go cue onset
- Time window: -2.5 to 2.5 seconds from go cue
- Bin size: 10 ms (dt = 1/100)
- Smoothing: Causal Gaussian kernel (window = 15 bins)
- Boundary condition: reflect (prepend first 15 bins before convolution)
- The Gaussian kernel uses MATLAB's `gausswin` default (alpha=2.5)
- Causal: first half of kernel zeroed out

### 4. Low Firing Rate Removal
Matching `removeLowFRClusters.m`:
- Compute mean firing rate across all valid trials and time bins
- Remove units with mean FR <= 1 Hz
- Sessions with fewer than 10 remaining units are excluded

### 5. Video Alignment
Matching `findVideoOffset.m` and `WorkingWithDataObjs.m`:
- Video offset = mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)
- Frame times aligned: frameTimes - vidshift - goCue[trial]

### 6. Behavioral Variables

#### Decoder Input
- **time_from_go_cue**: Time axis relative to go cue onset (continuous, -2.495 to 2.495 s)

#### Decoder Outputs
- **lick_direction**: left=0, right=1 (per-trial, from bp.R)
- **behavioral_context**: WC=0 (autowater=1), DR=1 (autowater=0) (per-trial)
- **outcome**: incorrect=0 (miss), correct=1 (hit) (per-trial)
- **tongue_velocity**: Discretized at per-session 50th percentile of velocity magnitude
  - Computed from bottom camera DLC tracking of `top_tongue` feature
  - Velocity = sqrt(dx^2 + dy^2) / dt at 400 Hz, interpolated to 10ms bins
  - NaN values (outside video coverage) assigned to class 0
- **paw_velocity**: Same as tongue but using `bottom_paw` feature
- **motion_energy**: Discretized at per-session 50th percentile
  - Loaded from separate `motionEnergy_*.mat` files
  - Interpolated from 400 Hz video frame rate to 10ms time bins

### 7. Brain Region
All recordings are from ALM (anterior lateral motor cortex).

## Statistics Comparison with Paper

### Paper Reports:
- Two-context: 12 sessions, 6 mice, 522 units (214 single units)
- DR task total: 25 sessions, 9 mice, 1,651 units (483 single units)
- Randomized delay: 19 sessions, 4 mice, 845 units (288 single units)

### Our Dataset:
- Total: 45 sessions, 14 subjects, 2,504 units
- Ephys_Behavior: 25 sessions from 10 animals
- RandomizedDelay: 20 sessions from 4 animals

**Differences explained:**
- We include ALL quality clusters (not just single units) matching `params.quality = {'all'}`
- Unit counts are higher because multi-units and fair/poor quality units are included (per the reference code's default settings)
- The paper's unit counts (e.g., 1,651) refer to specific quality subsets used for particular analyses
- Our processing uses all units above 1 Hz FR threshold, matching the code's default behavior

## Validation Results

### Data Format
- All 45 sessions pass format validation
- No errors
- Warnings: ~30 trials with all-zero neural data (late trials in sessions where recording quality degraded)

### Decoder Performance (Full Dataset, 45 sessions)
Training balanced accuracy / Validation balanced accuracy:
- Lick direction: 0.694 / 0.679
- Behavioral context: 0.886 / 0.879
- Outcome: 0.690 / 0.668
- Tongue velocity: 0.792 / 0.789
- Paw velocity: 0.610 / 0.606
- Motion energy: 0.780 / 0.778

All outputs decoded well above chance (0.50).

## Known Issues and Decisions

1. **Motion energy loading**: Some ME files have different formats (struct vs cell array). Both are handled in the loader.
2. **Tongue velocity skew**: ~95% of tongue velocity bins fall below threshold because the tongue is visible only during licking (~response epoch), making most time bins have near-zero velocity.
3. **DR-only sessions**: Behavioral context is always DR=1 for sessions without WC blocks (JEB13 sessions 11-12, JEB14, and all RandomizedDelay sessions).
4. **Paw velocity skew in some sessions**: JEB15 sessions show very high fraction of low paw velocity, likely due to camera positioning or animal behavior.
5. **Session 35 (JEB23_2023-10-20)**: Motion energy output is all class 0, indicating ME data was unavailable for this session.

## File Manifest

- `convert_data.py` - Main conversion script
- `converted_data.pkl` - Full converted dataset (45 sessions)
- `sample_data.pkl` - Sample dataset (first 3 sessions)
- `train_decoder.py` - Decoder training/validation script (provided)
- `decoder.py` - Decoder implementation (provided)
- `CONVERSION_NOTES.md` - This file
- `README.md` - Project overview
- `conversion_full_out.txt` - Full conversion log
- `conversion_sample_out.txt` - Sample verification output
- `verification_full_out.txt` - Full data verification output
- `verification_sample_out.txt` - Sample data verification output
- `train_decoder_full_out.txt` - Full decoder training output
- `train_decoder_sample_out.txt` - Sample decoder training output

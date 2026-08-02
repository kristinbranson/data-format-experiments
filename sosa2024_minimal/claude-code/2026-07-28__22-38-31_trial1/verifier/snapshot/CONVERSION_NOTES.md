# Conversion Notes

## Source Data

**Paper**: "A flexible hippocampal population code for experience relative to reward"
Sosa, Plitt, & Giocomo (2025)

**Data**: NWB files from 11 switch-condition mice (m3, m4, m7, m11-m15, m17-m19), 12-14 sessions each (152 total sessions, 12,216 trials).

**Imaging**: 2-photon calcium imaging (GCaMP7f) of hippocampal CA1 neurons at 15.5078125 Hz (~64.5 ms time bins). Most mice have single-plane recordings; m17 and m18 have 2-plane recordings (pooled across planes per paper methods).

## Processing Pipeline

### 1. Cell Selection
- Used `iscell[:, 0] == 1` from suite2p curation to select valid ROIs.

### 2. dF/F Computation (matching `preprocessing.py`)
1. Mask fluorescence to within-trial timepoints only (NaN outside trials)
2. Neuropil subtraction: `F_corrected = F - 0.7 * F_neuropil`
3. Per trial: add back neuropil mean so baseline is close to true fluorescence
4. Maximin baseline with 300-sample window (~20s): running minimum of running maximum
5. `dF/F = (F - baseline) / |baseline|`
6. Smooth with 2-sample Gaussian kernel

### 3. Deconvolution
- OASIS algorithm via suite2p (`dcnv.oasis` with `batch_size=F.shape[0]`, `tau=0.7`, `fs=15.5078125`)
- Applied to the smoothed dF/F signal

### 4. Interneuron Filtering
- Computed Pearson correlation between each neuron's deconvolved activity and running speed
- Removed neurons with correlation > 0.5 (matching paper's criterion)
- Observed filtering rate: ~3.8% of neurons removed (paper reports mean 0.42% +/- 0.85%)
- The discrepancy may be due to differences in the exact dF/F computation or speed signal processing, but the effect on decoder performance is minimal

### 5. Trial Segmentation
- Trials defined by `trial_start` signal in NWB behavioral timeseries
- Trial boundaries: from one trial_start to the next (or end of recording)
- Each trial corresponds to one traversal of the virtual linear track

### 6. Behavioral Variables
- **Position**: Extracted from `position` timeseries (0-450 cm linear track)
- **Speed**: From `speed` timeseries (cm/s)
- **Lick**: Converted from cumulative lick count to binary per-frame lick signal
- **Environment**: From `environment` timeseries (0 or 1, corresponding to ENV1/ENV2)
- **Reward zone**: Determined from position when `reward_zone` signal > 0, identifying which zone (A/B/C) the animal is in
- **Reward outcome**: Detected from `Reward` timestamps relative to trial boundaries

## Decoder Format

### Neural Data
- `data['neural']`: List of sessions, each a list of trials
- Each trial: `(n_neurons, T)` array of deconvolved calcium events

### Inputs (4 dimensions)
| Index | Name | Description |
|-------|------|-------------|
| 0 | time_from_trial_start | Seconds since trial start |
| 1 | environment_type | 0=ENV1, 1=ENV2 |
| 2 | trial_number | 0-indexed trial number within session |
| 3 | previous_trial_outcome | 0=no reward, 1=reward on previous trial |

### Outputs (6 dimensions, all integer-coded)
| Index | Name | Values | Description |
|-------|------|--------|-------------|
| 0 | distance_to_reward_zone | 0-6 | Distance to nearest reward zone edge: <-50, -50 to -10, -10 to 0, 0 (inside), 0 to +10, +10 to +50, >+50 cm |
| 1 | absolute_position | 0-4 | Position bin: 0-90, 90-180, 180-270, 270-360, 360-450 cm |
| 2 | speed | 0-4 | Speed bin: <2, 2-10, 10-20, 20-40, >40 cm/s |
| 3 | lick | 0-1 | Binary: no lick / lick |
| 4 | reward_zone_location | 0-2 | Current session's reward zone: A(0), B(1), C(2) |
| 5 | reward_outcome | 0-1 | Binary: no reward / reward on this trial |

### Metadata
- `time_bin_size`: 64.48 ms
- `brain_region`: CA1
- `frame_rate`: 15.5078125 Hz

## Key Decisions

1. **Multi-plane handling**: For m17 and m18, fluorescence data from both imaging planes are concatenated along the neuron axis, matching the paper's statement that "planes were pooled for all analyses."

2. **Shape mismatch handling**: Some multi-plane sessions had off-by-one differences between neural and behavioral timeseries lengths. Resolved by truncating to the minimum length.

3. **Reward zone determination**: The reward zone for each trial is determined from the `reward_zone` behavioral signal by finding positions where the signal is active. For omission trials (where the signal may not activate), the zone is inferred from neighboring trials within the same session.

4. **Lick signal**: The NWB lick timeseries stores cumulative lick counts. Converted to binary per-frame by computing the diff and marking frames with positive diff as lick=1.

5. **Distance to reward zone**: Computed as signed distance from current position to nearest edge of the active reward zone. Negative = before zone, 0 = inside zone, positive = past zone.

## Validation Results

### Data Format Verification
- All 152 sessions pass format verification (no errors or warnings)
- 12,216 total trials across 11 subjects

### Decoder Performance (Sample: 10 sessions, 800 trials)
| Output | Train Balanced Acc | Val Balanced Acc | Chance |
|--------|-------------------|-----------------|--------|
| distance_to_reward_zone | 0.6850 | 0.6056 | 0.1429 |
| absolute_position | 0.7677 | 0.7421 | 0.2000 |
| speed | 0.6554 | 0.6217 | 0.2000 |
| lick | 0.7909 | 0.7743 | 0.5000 |
| reward_zone_location | 0.9863 | 0.9846 | 0.3333 |
| reward_outcome | 0.7632 | 0.5789 | 0.5000 |

All output dimensions substantially exceed chance, confirming that the neural data contains decodable information about behavioral and task variables.

### Sanity Checks vs Paper
| Metric | Observed | Paper |
|--------|----------|-------|
| Subjects | 11 | 11 switch mice |
| Sessions/subject | 12-14 | 12-14 |
| Trials/session | 80.4 +/- 6.1 | 80.5 +/- 7.4 |
| Neurons/session | 154-2323 | 155-2172 |
| Frame rate | 15.51 Hz | ~15.5 Hz |
| ENV1/ENV2 split | 6226/5990 | ~50/50 |

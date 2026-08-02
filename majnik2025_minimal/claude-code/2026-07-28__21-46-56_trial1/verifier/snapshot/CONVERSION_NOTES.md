# Conversion Notes: Track2p Data to Decoder Format

## Source Data

**Paper**: Majnik et al. 2025, "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p", eLife 14:RP107540.

**Dataset**: 6 mice (jm031-jm046) imaged daily in barrel cortex (layer 2/3) during postnatal development (P7-P14). Each mouse has 6-7 sessions recorded at 30 Hz using 2-photon calcium imaging with GCaMP8m.

## Data Loading

### Neural Data
- **Source**: Suite2p outputs from Track2p's matched-cell pipeline
  - `F.npy`: Raw fluorescence traces (n_neurons x n_timepoints)
  - `Fneu.npy`: Neuropil fluorescence
- **Neurons are pre-matched**: Track2p already identified cells present across all days for each mouse. All neurons in `iscell.npy` are marked as cells (probability >= 0.5 threshold).
- **No additional neuron filtering** was applied since Track2p already curated the population.

### Behavioral Data
- **Source**: `move_deve/motion_energy_glob.npy` - global motion energy from videography
- **Missing video frames**: Detected via `interframe_int.npy` (gaps > 1.5x median interval). Missing frames interpolated linearly. Affects 9/41 sessions (1-148 frames missing).

## Processing Pipeline

### 1. dF/F Computation
Following the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)"

- **Neuropil correction**: Fc = F - 0.7 * Fneu (Suite2p default coefficient)
- **Baseline**: Suite2p default 'maximin' method:
  1. Gaussian smooth with sigma = window/6
  2. Minimum filter (60-second window = 1800 frames)
  3. Maximum filter (60-second window = 1800 frames)
- **dF/F** = (Fc - F0) / F0

### 2. Temporal Binning
Following the paper: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps"

- Bin size: 10 frames (30 Hz -> 3 Hz, ~333.33 ms per bin)
- Applied to both neural (dF/F) and behavioral (motion energy) data

### 3. Trial Segmentation
Following the paper: "splits were done on consecutive 2 minute blocks of the recording"

- Trial duration: 2 minutes = 360 bins (after 10-frame binning)
- 20-minute sessions: 10 trials per session
- 30-minute sessions: 15 trials per session
- Total: 545 trials across 41 sessions

### 4. Motion Energy Discretization
Following the task specification: "normalized and discretized into five equal-percentile bins"

- Computed global quintile bin edges across all sessions and mice
- Bin edges: [531661.5, 721126.0, 822566.6, 1015832.6, 1642195.4, 39962858.7]
- Each bin contains exactly 20% of the data globally
- Output values: 0-4 (Q1 through Q5)

### 5. Decoder Input
- **Time elapsed from start of session** (in seconds), time-varying
- For each trial, time runs from trial_start to trial_end (centers of bins)
- E.g., trial 0: [0.17, 0.50, ..., 119.83] seconds

## Data Structure

| Field | Value |
|-------|-------|
| Sessions | 41 |
| Subjects | 6 mice |
| Neurons per mouse | 221 (jm031), 370 (jm032), 685 (jm038), 746 (jm039), 541 (jm040), 435 (jm046) |
| Mean neurons | 500 +/- 198 (paper: 526 +/- 190) |
| Trials per session | 10 (20-min) or 15 (30-min) |
| Timepoints per trial | 360 |
| Time bin size | 333.33 ms |
| Brain region | barrel_cortex |
| Input dimension | 1 (time_elapsed_s) |
| Output dimension | 1 (motion_energy_bin, 5 classes) |

## Sanity Checks

1. **Neuron counts match across sessions**: All sessions for each mouse have the same number of neurons (verified).
2. **Mean neuron count**: 500 +/- 198, consistent with paper's 526 +/- 190.
3. **Session counts**: 6-7 sessions per mouse, matching paper's "minimum of 6 consecutive days".
4. **Total mice**: 6, matching paper's "full dataset of 6 mice".
5. **No NaN/Inf values**: All trials verified clean.
6. **Equal-percentile bins**: Each bin contains exactly 20% of global data.
7. **Frame rate**: 30 Hz confirmed from Suite2p ops.npy.
8. **Recording durations**: jm031/jm032 = 20 min (36000 frames), others = 30 min (54000 frames).

## Decoder Performance

- **Sample (2 sessions)**: Balanced accuracy = 0.41 (chance = 0.20)
- **Full dataset**: See train_decoder_full_out.txt

## Decisions and Justifications

1. **No additional cell filtering**: Track2p data already contains only matched, verified cells.
2. **Global percentile bins**: Motion energy discretized using global quintiles to ensure balanced classes across the full dataset.
3. **Maximin baseline**: Used Suite2p's default 'maximin' method for dF/F baseline, as the paper states "default Suite2p parameters."
4. **Linear interpolation for missing frames**: Simple and appropriate for small gaps (1-3 frames typical). Larger gaps (116-148 frames in 2 sessions) also interpolated, which may introduce minor artifacts but preserves temporal alignment.
5. **2-minute trial blocks**: Matches the paper's cross-validation blocking strategy for decoding.

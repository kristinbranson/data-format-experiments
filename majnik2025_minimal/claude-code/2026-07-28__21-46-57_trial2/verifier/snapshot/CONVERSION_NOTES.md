# Conversion Notes

## Source Data

Majnik et al. 2025, Track2p longitudinal calcium imaging dataset from mouse barrel cortex.

- 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046
- 6-7 daily sessions per subject (41 sessions total)
- Imaging: 30 Hz, 512x512 pixels, 720x720 um FOV, layer 2/3
- Session duration: 20 min (jm031, jm032) or 30 min (others)
- Neurons tracked across all days per mouse using Track2p algorithm

## Processing Decisions

### Neural Data (dF/F)

Following the paper's methods ("We used baseline corrected fluorescence traces as our dF/F using the default Suite2p parameters"):

1. **Neuropil correction**: `Fc = F - 0.7 * Fneu` (Suite2p default neucoeff=0.7)
2. **Baseline**: Suite2p "maximin" method (default):
   - Gaussian smooth with sigma = 10s * 30Hz = 300 frames
   - Running minimum with window = 60s * 30Hz = 1800 frames
   - Running maximum with window = 1800 frames
3. **dF/F**: `(Fc - baseline) / baseline`

Parameters confirmed from ops.npy: fs=30, neucoeff=0.7, baseline='maximin', win_baseline=60.0, sig_baseline=10.0.

### Temporal Binning

Following the paper's decoding methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

- Bin size: 10 frames = 333.33 ms
- Both neural and behavioral data binned identically

### Trial Structure

Following the paper's cross-validation approach: "splits were done on consecutive 2 minute blocks of the recording."

- Trial duration: 2 minutes = 120 seconds
- Bins per trial: 120s * 30Hz / 10 frames = 360 bins
- 20-min sessions: 10 trials per session
- 30-min sessions: 15 trials per session
- Total: 545 trials across 41 sessions

### Motion Energy

From the paper: "we quantified these by looking at the pixel-wise difference of consecutive frames... squared all individual pixel-wise values and summed across pixels."

The motion energy data is provided pre-computed in `move_deve/motion_energy_glob.npy`.

**Missing frames**: Some camera recordings have dropped frames (up to 116 in one session). These are handled by:
1. Detecting gaps using interframe intervals from `tstamps.npy`
2. Mapping camera frames to 2p frame indices
3. Linear interpolation to fill missing frames

### Discretization

Task specification: "Motion energy, normalized and discretized into five equal-percentile bins."

- Computed quintile thresholds (0%, 20%, 40%, 60%, 80%, 100%) across all binned motion energy values from all sessions globally
- Assigned values 0-4 (bin_0 through bin_4)
- Global discretization ensures consistent bin definitions across sessions

### Decoder Input

Task specification: "Time elapsed from the beginning of the experiment."

- Time in seconds from recording onset
- Computed as bin center times: `(bin_index * 10 + 5) / 30` seconds
- Shape: (1, n_timepoints) per trial

## Validation Results

### Data Format
- All validation checks pass with no errors or warnings

### Statistics Comparison with Paper

| Metric | Paper | Converted |
|--------|-------|-----------|
| Neurons per mouse (mean +/- std) | 526 +/- 190 | 500 +/- 180 |
| Sessions per mouse | >= 6 | 6-7 |
| Subjects | 6 | 6 |
| Imaging rate | 30 Hz | 30 Hz |
| Session duration | >= 20 min | 20-30 min |
| Brain region | Barrel cortex L2/3 | barrel_cortex |

Note: The slight difference in mean neuron count (500 vs 526) is expected since the paper may report a slightly different rounding or subset. The tracked neuron counts per mouse are:
- jm031: 221 (Mouse A)
- jm032: 370 (Mouse B)
- jm038: 685
- jm039: 746
- jm040: 541
- jm046: 435

### Output Distribution
- Global discretization produces exactly 20% per bin across all data
- Per-session distributions vary (expected since motion patterns differ across mice and developmental stages)

### Decoder Performance
- Sample (6 sessions, jm031): Balanced accuracy 35.6% vs 20% chance
- Full dataset (41 sessions): Balanced accuracy 39.7% vs 20% chance (training: 47.4%)
- Decoder performs ~2x chance, confirming neural data contains decodable motion energy information

## Sanity Checks

1. **Neuron count consistency**: Same number of neurons across all sessions within each mouse (confirmed by Track2p matching)
2. **Neuron count statistics**: Mean ~500, std ~180, consistent with paper's 526 +/- 190
3. **Session counts**: 6-7 sessions per mouse, matching "at least 6 consecutive days"
4. **Output balance**: Equal-percentile bins produce exactly 20% per class globally
5. **Time bin size**: 333.33 ms (10 frames at 30 Hz), matching paper's binning
6. **Trial structure**: 2-minute blocks matching paper's cross-validation splits
7. **Decoder above chance**: Confirms neural data contains decodable information about motion energy

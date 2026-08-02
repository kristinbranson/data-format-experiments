# Conversion Notes

## Overview

Converted NWB data from the brain-wide neural activity dataset to the standardized decoder format. The source data is from "Brain-wide neural activity underlying memory-guided movement" and analyzed in "Brain-wide analysis reveals movement encoding structured across and within brain areas".

## Data Loading

### Source Format
- 174 NWB files across 28 subjects in `data/sub-*/sub-*.nwb`
- Each NWB file represents one recording session
- Some sessions have multiple probe insertions covering different subsets of trials

### Spike Times
- Loaded from `/units/spike_times` (ragged array with index in `/units/spike_times_index`)
- Observation windows checked via `/units/obs_intervals` to determine which trials have valid neural data

### Trial Data
- Loaded from `/intervals/trials/` table
- Go cue times from `/acquisition/BehavioralEvents/go_start_times/timestamps`
- Sample start times from `/acquisition/BehavioralEvents/sample_start_times/timestamps`
- Trial start times from `/intervals/trials/start_time`

### Tongue Tracking
- Side-view camera tracking from `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`
- Shape (N, 3): [x, y, likelihood] from DeepLabCut
- All tracking data used (no confidence filtering) for percentile computation and discretization

## Processing Decisions

### Neuron Quality Control
- **Method**: Classifier-based QC (`classification='good'` in NWB units table)
- **Matches reference**: The reference code uses `qc_mode='classifier'` which applies trained logistic regression classifiers per brain area
- **Result**: 57,925 good units with valid brain regions across 144 sessions
- **Paper reports**: 69,943 good units across 173 sessions (our count is lower due to session filtering)

### Brain Region Mapping
- NWB `anno_name` field contains detailed Allen Brain Atlas annotation names (293 unique)
- Mapped to 14 high-level categories matching the reference code: ALM, OtherCortex, Orbital, Striatum, Pallidum, Thalamus, Hypothalamus, Hippocampus, Midbrain, Pons, Medulla, Cerebellum, Olfactory, CorticalSubplate
- ALM = "Secondary motor area" (all layers) - the primary target region for this study
- Units with empty or unmappable annotation names excluded (490 units, <1%)
- Pretectal nuclei and Fields of Forel mapped to Thalamus following Allen Brain Atlas hierarchy

### Session Filtering
- **Performance threshold**: >65% correct on control trials (no photostim, no early lick, no auto_water, no free_water)
- **Minimum correct trials**: >=50 correct left AND >=50 correct right trials
- **Matches reference**: Methods text states "overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each"
- **Result**: 144 sessions kept, 30 skipped
- **Skipped reasons**: 20 for low performance, 7 for insufficient correct trials per direction, 1 for no good units with valid regions, 2 for other reasons

### Trial Filtering
- Excluded `auto_water` and `free_water` trials (as in reference code)
- **Kept** early lick trials (needed for early_lick decoder output)
- **Kept** ignore/no-response trials (needed for outcome decoder output)
- **Kept** photostimulation trials (needed for photostim decoder input)
- Only included trials within the recording observation window (some NWB files contain trials beyond the neural recording period)

### Temporal Alignment
- Aligned to go cue onset (t=0)
- Window: -2.5s to +1.5s (as specified in decoder task)
- Reference code uses -3.0 to +3.0s or +3.5s, but decoder task specifies -2.5 to +1.5s

### Firing Rate Computation
- 50ms non-overlapping bins (as specified in decoder task)
- Spike counts divided by bin width (0.05s) to get rates in Hz
- 80 time bins per trial
- Reference code uses 100ms bandwidth with 50ms stride (Gaussian-weighted), but decoder task specifies 50ms bins

### Input Variables

#### Time from tone onset
- Computed as seconds since the first tone of the (successful) sample epoch
- For trials with early lick replays, uses the last sample_start before the go cue
- Typical value at go cue: ~1.85s (0.65s sample + 1.2s delay)
- Range varies across trials due to early lick replays that extend the sample/delay epochs

#### Photostimulation
- Binary time series: 1 when photostim is on, 0 otherwise
- `photostim_onset` in NWB is relative to trial start (converted to absolute then to go-cue-relative)
- Photostim typically occurs during late delay epoch (-1.2s to -0.7s relative to go cue)
- Duration: 0.5s (last 0.5s of delay including 100ms ramp-down)

### Output Variables

#### Choice (lick direction)
- 0 = left, 1 = right
- Based on `trial_instruction` field (which port the animal should lick)
- Per-trial, broadcast across time bins

#### Outcome
- 0 = ignore (no response), 1 = miss (wrong port), 2 = hit (correct)
- Per-trial, broadcast across time bins

#### Early Lick
- 0 = no early lick, 1 = early lick during sample/delay
- Per-trial, broadcast across time bins

#### Tongue Y-Position
- Time-varying, discretized per session
- 0 = below 40th percentile, 1 = 40th-60th percentile, 2 = above 60th percentile
- Percentiles computed over all tongue y-position values across all trials in the session
- All DLC tracking data used (no confidence filtering)

## Sanity Checks

### Neuron Counts
- Paper reports 69,943 good units across 173 sessions
- We have 57,925 good units across 144 sessions
- Difference due to session filtering (30 sessions excluded) and units without valid region annotation

### Regional Distribution (our data)
- Thalamus: 11,891 (20.5%) - paper reports 12,808
- Orbital: 8,962 (15.5%)
- OtherCortex: 7,369 (12.7%)
- ALM: 6,046 (10.4%) - paper reports 8,717
- Striatum: 5,851 (10.1%) - paper reports 7,664
- Midbrain: 5,555 (9.6%) - paper reports 7,495
- Olfactory: 3,767 (6.5%)
- Medulla: 2,628 (4.5%) - paper reports 2,928
- Hippocampus: 1,652 (2.9%)
- Cerebellum: 1,540 (2.7%)
- Pallidum: 1,019 (1.8%)
- Hypothalamus: 663 (1.1%)
- CorticalSubplate: 643 (1.1%)
- Pons: 339 (0.6%)

### Trial Statistics
- Mean trials per session: 519.2 (paper reports mean 476, range 130-785)
- Our range: 160-796, consistent with paper

### Performance
- Mean session performance: ~83% (paper reports 84% mean, range 65-99%)
- Consistent with reference

### Decoder Performance (sample data, 4 sessions)
- Choice: 74.5% balanced accuracy (chance 50%)
- Outcome: 60.8% balanced accuracy (chance 33%)
- Early lick: 68.9% balanced accuracy (chance 50%)
- Tongue y-position: 52.8% balanced accuracy (chance 33%)
- All outputs substantially above chance

## Discrepancies from Reference

1. **Time window**: -2.5 to +1.5s instead of -3.0 to +3.0s (per decoder task specification)
2. **Bin width**: 50ms non-overlapping instead of 100ms with 50ms stride (per decoder task specification)
3. **Trial inclusion**: Kept early lick, ignore, and photostim trials (per decoder task requiring these as outputs/inputs)
4. **Trial exclusion**: Only excluded auto_water and free_water trials (reference excludes more trial types)
5. **Session count**: 144 vs 173 sessions (due to stricter session filtering applied)

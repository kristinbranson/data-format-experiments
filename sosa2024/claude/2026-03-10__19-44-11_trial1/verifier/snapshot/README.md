# Neural Decoder Dataset: Sosa et al. 2025

Converted from NWB calcium imaging data (Sosa et al. 2025, Nature Neuroscience) into a decoder-compatible Python dictionary format.

## Source Data

- **Paper**: "A flexible hippocampal population code for experience relative to reward"
- **DOI**: 10.1038/s41593-025-01985-4
- **Subjects**: 11 mice (m3, m4, m7, m11-m15, m17-m19), reward zone switching paradigm
- **Sessions**: 152 total (14 per mouse, 12 for m11)
- **Brain region**: Hippocampal CA1
- **Recording**: Two-photon calcium imaging (GCaMP7f), ~15.5 Hz frame rate
- **Task**: Virtual linear track (450 cm), hidden reward zones (A/B/C), switch at trial 30

## Output Files

| File | Description | Size |
|------|-------------|------|
| `converted_data.pkl` | Full converted dataset (152 sessions, 12,216 trials) | ~9.4 GB |
| `convert_data.py` | Conversion script | - |
| `train_decoder.py` | Decoder training script | - |
| `CONVERSION_NOTES.md` | Detailed conversion notes and validation | - |
| `sample_trials.png` | Sample trial visualizations | - |
| `predictions.png` | Decoder prediction visualizations | - |

## Data Format

Load with: `data = pickle.load(open('converted_data.pkl', 'rb'))`

### Structure

```python
data['neural']       # list of 152 sessions, each list of trials, each (n_neurons, T) array
data['input']        # list of 152 sessions, each list of trials, each (4, T) array
data['output']       # list of 152 sessions, each list of trials, each (6, T) array
data['input_names']  # ['time_from_start', 'environment', 'trial_number', 'prev_trial_outcome']
data['output_names'] # ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick',
                     #  'reward_zone_location', 'reward_outcome']
data['output_values']  # list of possible discrete values per output dimension
data['bin_size']       # 0.064484 seconds (~15.5 Hz)
data['brain_region']   # 'CA1'
data['species']        # 'Mus musculus'
```

### Neural Data
- Deconvolved calcium events (OASIS, suite2p)
- Filtered: iscell=1, interneurons excluded (speed-dFF correlation > 0.5)
- 138,394 total neurons across sessions (155-2,327 per session, mean 910.5)

### Inputs (4 dimensions)
| Dim | Name | Type | Description |
|-----|------|------|-------------|
| 0 | time_from_start | continuous | Seconds from trial start |
| 1 | environment | per-trial | 0=Env1, 1=Env2 |
| 2 | trial_number | per-trial | 0-indexed within session |
| 3 | prev_trial_outcome | per-trial | 0=omitted/first, 1=rewarded |

### Outputs (6 dimensions, all discretized)
| Dim | Name | Classes | Bins |
|-----|------|---------|------|
| 0 | distance_to_reward_zone | 7 | <-50, [-50,-10], [-10,0), 0(inside), (0,10], [10,50], >50 cm |
| 1 | absolute_position | 5 | [0,90), [90,180), [180,270), [270,360), [360,450] cm |
| 2 | speed | 5 | <2, [2,10), [10,20), [20,40), >=40 cm/s |
| 3 | lick | 2 | 0=no, 1=yes |
| 4 | reward_zone_location | 3 | 0=A(80-130), 1=B(200-250), 2=C(320-370) |
| 5 | reward_outcome | 2 | 0=no, 1=yes |

## Decoder Results

Trained with 70/30 train/validation split, balanced loss, PCA to 100 components.

| Output | Val Balanced Acc | Chance | Ratio |
|--------|-----------------|--------|-------|
| distance_to_reward_zone | 0.342 | 0.143 | 2.39x |
| absolute_position | 0.478 | 0.200 | 2.39x |
| speed | 0.402 | 0.200 | 2.01x |
| lick | 0.644 | 0.500 | 1.29x |
| reward_zone_location | 0.796 | 0.333 | 2.39x |
| reward_outcome | 0.521 | 0.500 | 1.04x |

All spatial/kinematic outputs well above chance. Lick (sparse) and reward outcome (stochastic ~15% omission) are inherently harder to decode.

## Reproduction

```bash
# Convert NWB data (requires data/ directory with NWB files)
python3 convert_data.py

# Verify data format only
python3 train_decoder.py converted_data.pkl --verify-only --plot-samples

# Train decoder
python3 train_decoder.py converted_data.pkl --plot-samples --stats-json stats.json
```

## Processing Pipeline

1. Load NWB files (h5py)
2. Filter neurons: iscell=1, exclude interneurons (speed-dFF Pearson r > 0.5)
3. Parse scene name from NWB identifier to determine reward zones per trial
4. Extract trial-aligned data (trial_start to teleport flags)
5. Compute inputs: time, environment, trial number, previous reward outcome
6. Compute outputs: distance to reward zone, position, speed, lick, reward zone, reward outcome
7. Discretize continuous outputs per specification
8. Apply lick sensor error correction (>35% threshold)

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Total sessions | 152 |
| Total trials | 12,216 |
| Total neurons | 138,394 |
| Mean trials/session | 80.4 (paper: 80.5 +/- 7.4) |
| Mean neurons/session | 910.5 |
| Reward rate | 84.3% (paper: ~85%) |
| Time bin | 64.48 ms (~15.5 Hz) |

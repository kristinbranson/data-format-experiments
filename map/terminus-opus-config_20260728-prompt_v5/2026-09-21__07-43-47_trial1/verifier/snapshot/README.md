# MAP Dataset - Decoder-Compatible Format

## Dataset Description

This dataset contains brain-wide neural recordings from mice performing an audio delay task, converted from NWB format to a decoder-compatible Python dictionary structure.

**Source**: Mesoscale Activity Project (MAP) - DANDI archive 000363
**Papers**:
- Li et al. "Brain-wide neural activity underlying memory-guided movement" (data paper)
- Wang et al. "Brain-wide analysis reveals movement encoding structured across and within brain areas" (method paper)

## Task Description

Mice discriminate between two auditory tones and report their choice by directional licking (left or right) after a delay period. The trial structure is:
1. **Presample**: Variable duration before tone
2. **Sample**: Tone presentation (~0.7s, at -1.85s relative to go cue)
3. **Delay**: 1.2s delay period (no stimulus)
4. **Go cue**: Response window opens (time 0)
5. **Response**: Mouse licks left or right (or no response)

Photostimulation (optogenetic inactivation of ALM) is applied on ~25% of trials in a subset of mice.

## Data Format

The converted data is saved as a Python pickle file (`converted_data.pkl`).

### Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Data Structure

```python
data = {
    'neural': [  # list of 173 sessions
        [  # list of trials per session
            np.array,  # shape: (n_neurons, 80) - firing rates in Hz
        ],
    ],
    'input': [  # decoder inputs
        [  # list of trials per session
            np.array,  # shape: (2, 80)
            # input[0]: time from tone onset (seconds)
            # input[1]: photostimulation on/off (binary)
        ],
    ],
    'output': [  # decoder outputs
        [  # list of trials per session
            np.array,  # shape: (4, 80)
            # output[0]: lick choice (0=left, 1=right, 2=no_lick)
            # output[1]: outcome (0=hit, 1=miss, 2=ignore)
            # output[2]: early lick (0=no, 1=yes)
            # output[3]: tongue y-position (0=<p40, 1=p40-p60, 2=>p60, 3=not_visible)
        ],
    ],
    'subjects': list,  # 28 subject IDs
    'subject_idx': np.array,  # shape (173,) - index into subjects
    'brain_regions': list,  # 293 brain region names
    'brain_region_idx': [np.array],  # per-session neuron region indices
    'input_names': ['time_from_tone_onset', 'photostim_on'],
    'output_names': ['lick_choice', 'outcome', 'early_lick', 'tongue_y_position'],
    'output_values': [  # categorical labels
        ['left', 'right', 'no_lick'],
        ['hit', 'miss', 'ignore'],
        ['no_early', 'early'],
        ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible'],
    ],
    'metadata': dict,  # task description, time bin info, etc.
}
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Temporal alignment | Go cue onset |
| Time window | -2.5s to +1.5s |
| Bin width | 50 ms |
| Time bins per trial | 80 |
| Total sessions | 173 |
| Total subjects | 28 |
| Total neurons | 69,453 |
| Total trials | 89,532 |
| Brain regions | 293 |

### Neural Data

- Firing rates computed from spike times using 50ms non-overlapping bins
- Only "good" quality units included (QC classifier from spike sorting pipeline)
- Aligned to go cue onset

### Inputs

1. **time_from_tone_onset**: Continuous time (seconds) from the auditory tone onset
2. **photostim_on**: Binary indicator of optogenetic photostimulation

### Outputs

1. **lick_choice**: Per-trial lick direction (left/right/no_lick)
2. **outcome**: Per-trial behavioral outcome (hit/miss/ignore)
3. **early_lick**: Per-trial early lick indicator
4. **tongue_y_position**: Time-varying discretized tongue position
   - Discretized using session-specific 40th/60th percentiles
   - 3 = tongue not visible (likelihood < 0.9)

## Trial Filtering

- Auto-water trials removed (not genuine behavioral trials)
- Free-water trials removed
- Trials outside neural recording window removed
- All other trials retained (including photostim, early lick, ignore)

## Decoder Training

Run the decoder with:
```bash
python train_decoder.py converted_data.pkl
```

### Expected Decoder Performance

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| lick_choice | 0.681 | 0.333 |
| outcome | 0.661 | 0.333 |
| early_lick | 0.751 | 0.500 |
| tongue_y_position | 0.669 | 0.250 |

## Files

- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - 2-session sample for testing
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion notes

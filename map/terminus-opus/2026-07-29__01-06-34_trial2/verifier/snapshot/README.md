# MAP Dataset - Decoder-Compatible Format

## Dataset Description
This is the Mesoscale Activity Map (MAP) dataset (DANDI:000363), converted from NWB format to a decoder-compatible Python dictionary format.

**Task**: Auditory delayed response task. Mice hear tones (3kHz or 12kHz) during a sample epoch, maintain memory during a 1.2s delay, then lick left or right after a go cue.

**Source**: Chen, Nguyen, Li, Svoboda (2023). "Brain-wide neural activity underlying memory-guided movement."

## Quick Start
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
fr = data['neural'][0][0]  # shape: (n_neurons, 80)

# Access inputs for session 0, trial 0
inp = data['input'][0][0]  # shape: (2, 80)

# Access outputs for session 0, trial 0
out = data['output'][0][0]  # shape: (4, 80)
```

## Data Format

### Structure
```python
data = {
    'neural': list of sessions → list of trials → (n_neurons, 80) arrays,
    'input': list of sessions → list of trials → (2, 80) arrays,
    'output': list of sessions → list of trials → (4, 80) arrays,
    'subjects': list of subject IDs,
    'subject_idx': (n_sessions,) array,
    'brain_regions': ['ALM', 'BLA', 'ECT', 'Medulla', 'Midbrain', 'Striatum', 'Thalamus'],
    'brain_region_idx': list of (n_neurons,) arrays per session,
    'input_names': ['time_from_tone_onset', 'photostim_on'],
    'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
    'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
    'metadata': {...}
}
```

### Key Parameters
- **Temporal alignment**: Go cue onset
- **Time window**: -2.5s to +1.5s relative to go cue
- **Bin size**: 50ms (80 time bins)
- **Neuron filtering**: QC classifier-based ('good' units only)
- **Trial filtering**: Excludes early lick, auto water, free water, no-response, and photostimulation trials

### Inputs
1. `time_from_tone_onset`: Time in seconds since first tone onset (continuous, time-varying)
2. `photostim_on`: Binary indicator of photostimulation (always 0 after trial filtering)

### Outputs
1. `choice`: Lick direction (0=left, 1=right)
2. `outcome`: Trial outcome (0=ignore, 1=miss, 2=hit)
3. `early_lick`: Early lick indicator (0=no, 1=yes; always 0 after filtering)
4. `tongue_y`: Discretized tongue y-position (0=low, 1=mid, 2=high; per-session percentiles)

## Statistics
- **Sessions**: 173
- **Subjects**: 28
- **Total neurons**: 69,453
- **Total trials**: 52,990
- **Brain regions**: 7
- **Mean neurons/session**: 401
- **Mean trials/session**: 306

## Decoder Training
```bash
python train_decoder.py converted_data.pkl
```

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample (2 sessions) for testing
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion notes
- `train_decoder.py` - Decoder training script

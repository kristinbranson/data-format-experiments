# Neural Decoder Dataset: Separating Cognitive and Motor Processes

## Dataset Description

This dataset contains electrophysiology recordings from the anterior lateral motor cortex (ALM) of mice performing a two-context delayed response (DR) and water-cued (WC) licking task. The data has been converted from the original MATLAB format into a Python-compatible format suitable for neural decoding.

**Source Paper**: "Separating cognitive and motor processes in the behaving mouse" (Economo lab)

## Quick Start

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access outputs for session 0, trial 0
output = data['output'][0][0]  # shape: (6, n_timepoints)
```

## Data Format

### Structure
```python
data = {
    'neural': list of sessions, each containing list of trials (n_neurons, 500)
    'input': list of sessions, each containing list of trials (1, 500)
    'output': list of sessions, each containing list of trials (6, 500)
    'subjects': list of 10 animal IDs
    'subject_idx': array of session-to-subject mapping
    'brain_regions': ['ALM']
    'brain_region_idx': list of arrays mapping neurons to brain regions
    'input_names': ['time_from_goCue']
    'output_names': ['lick_direction', 'behavioral_context', 'outcome',
                     'tongue_velocity', 'paw_velocity', 'motion_energy']
    'output_values': [['left','right'], ['WC','DR'], ['incorrect','correct'],
                      ['low','high'], ['low','high'], ['low','high']]
    'metadata': {...}
}
```

### Key Parameters
- **Time bins**: 500 bins at 10ms each (5 seconds total)
- **Time window**: -2.5 to 2.5 seconds from go cue onset
- **Temporal alignment**: Go cue onset
- **Neural data**: Smoothed firing rates (spks/sec) with causal Gaussian kernel
- **Brain region**: ALM (anterior lateral motor cortex)

### Outputs
| Output | Values | Type |
|--------|--------|------|
| lick_direction | 0=left, 1=right | Per-trial |
| behavioral_context | 0=WC, 1=DR | Per-trial |
| outcome | 0=incorrect, 1=correct | Per-trial |
| tongue_velocity | 0=low, 1=high | Time-varying |
| paw_velocity | 0=low, 1=high | Time-varying |
| motion_energy | 0=low, 1=high | Time-varying |

## Dataset Statistics
- **Subjects**: 10 mice
- **Sessions**: 25
- **Total trials**: 6,150
- **Total neurons**: 1,453 (after quality + FR filtering)
- **Neurons per session**: 27-134 (mean 58)

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample (2 sessions) for testing
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder module

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl

# Train on sample data
python train_decoder.py sample_data.pkl
```

## Conversion

```bash
# Full conversion
python -u convert_data.py converted_data.pkl --full

# Sample conversion (2 sessions)
python -u convert_data.py sample_data.pkl --sample

# With processing visualizations
python -u convert_data.py sample_data.pkl --sample --show-processing
```

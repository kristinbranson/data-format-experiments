# ALM Electrophysiology Decoder Dataset

## Dataset Description

This dataset contains neural recordings from the anterior lateral motor cortex (ALM) of mice performing a two-context behavioral paradigm, converted from the paper "Separating cognitive and motor processes in the behaving mouse" (Bhagat et al., 2024).

### Task
Mice performed two directional licking tasks alternating block-wise:
- **Delayed Response (DR)**: Auditory stimulus → delay → go cue → lick to correct side
- **Water Cued (WC)**: Water presented at random port → mouse consumes water

### Neural Data
- **Brain region**: ALM (anterior lateral motor cortex)
- **Recording**: High-density silicon probes (Neuropixels 1.0)
- **Processing**: Spike-sorted, binned at 10ms, smoothed with causal gaussian kernel (15ms window)
- **Filtering**: Units with quality labels 'garbage'/'noisy' excluded; units with mean FR < 1 Hz removed

### Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 44 (25 DR + 19 randomized delay) |
| Subjects | 14 mice |
| Total neurons | 2,452 |
| Total trials | 14,972 |
| Time bins per trial | 500 (10ms bins, -2.5 to 2.5s from go cue) |

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access decoder inputs
input_data = data['input'][0][0]  # shape: (1, n_timepoints) - time from go cue

# Access decoder outputs
output_data = data['output'][0][0]  # shape: (6, n_timepoints)
```

## Data Format

### Inputs (1 dimension)
- `time_from_go_cue`: Time from go cue onset in seconds [-2.5, 2.5]

### Outputs (6 dimensions)
| Output | Values | Type |
|--------|--------|------|
| lick_direction | 0=left, 1=right, 2=none | Per-trial |
| context | 0=DR, 1=WC | Per-trial |
| outcome | 0=incorrect, 1=correct, 2=ignore | Per-trial |
| tongue_velocity | 0=low, 1=high, 2=not_visible | Time-varying |
| paw_velocity | 0=low, 1=high, 2=not_visible | Time-varying |
| motion_energy | 0=low, 1=high, 2=no_video | Time-varying |

### Metadata
- `subjects`: List of mouse IDs
- `subject_idx`: Session-to-subject mapping
- `brain_regions`: ['ALM']
- `brain_region_idx`: Neuron-to-region mapping per session
- `metadata`: Task description, temporal alignment info, processing parameters

## Files
- `converted_data.pkl`: Full converted dataset
- `sample_data.pkl`: 2-session sample for testing
- `convert_data.py`: Conversion script
- `train_decoder.py`: Decoder training script
- `CONVERSION_NOTES.md`: Detailed conversion documentation

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl

# Train on sample
python train_decoder.py sample_data.pkl
```

## Conversion

```bash
# Convert all sessions
python -u convert_data.py converted_data.pkl --full

# Convert sample (2 sessions)
python -u convert_data.py sample_data.pkl --sample
```

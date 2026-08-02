# Hasnain et al. 2024 - Neural Decoder Dataset

## Dataset Description

This dataset contains electrophysiology recordings from the anterior lateral motor cortex (ALM) of mice performing a two-context licking task, converted from the paper:

**"Separating cognitive and motor processes in the behaving mouse"**  
Hasnain, Birnbaum et al., Nature Neuroscience 2024

### Task
Head-fixed mice performed two directional licking tasks alternating block-wise:
- **Delayed-Response (DR)**: Auditory cue → delay → go cue → directional lick
- **Water-Cued (WC)**: Water presented at random port → consume

### Recording
- **Region**: Anterior Lateral Motor Cortex (ALM)
- **Method**: High-density silicon probes (Neuropixels)
- **Species**: Mouse (Mus musculus)

## Data Format

The converted data is stored in `converted_data.pkl` (Python pickle format).

### Loading the data
```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Data Structure
```python
data = {
    'neural': [sessions x [trials x (n_neurons, n_timepoints)]],
    'input': [sessions x [trials x (1, n_timepoints)]],
    'output': [sessions x [trials x (6, n_timepoints)]],
    'subjects': list of str,
    'subject_idx': array of int,
    'brain_regions': ['ALM'],
    'brain_region_idx': [array per session],
    'input_names': ['time_from_goCue'],
    'output_names': ['lick_direction', 'context', 'outcome', 
                     'tongue_velocity', 'paw_velocity', 'motion_energy'],
    'output_values': [['left','right'], ['WC','DR'], ['incorrect','correct'],
                      ['low','high'], ['low','high'], ['low','high']],
    'metadata': {...}
}
```

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 44 |
| Subjects | 14 |
| Total trials | 11,985 |
| Total neurons | 2,359 |
| Mean neurons/session | 53.6 |
| Mean trials/session | 272.4 |
| Time bins | 1000 (5ms bins, -2.5 to 2.5s) |
| Temporal alignment | Go cue onset |

### Decoder Inputs
- **time_from_goCue**: Time from go cue onset in seconds (continuous, time-varying)

### Decoder Outputs
- **lick_direction**: Left (0) or Right (1), per-trial
- **context**: Water-Cued (0) or Delayed-Response (1), per-trial
- **outcome**: Incorrect (0) or Correct (1), per-trial
- **tongue_velocity**: Low (0) or High (1), time-varying, 50th percentile threshold
- **paw_velocity**: Low (0) or High (1), time-varying, 50th percentile threshold
- **motion_energy**: Low (0) or High (1), time-varying, 50th percentile threshold

## Processing Pipeline

1. Load session data from MATLAB .mat files
2. Filter clusters by quality (exclude garbage, noisy, real?)
3. Align spikes to go cue onset
4. Bin spikes into 5ms bins
5. Smooth with causal Gaussian kernel (window=15)
6. Remove low firing rate clusters (< 0.5 Hz)
7. Extract behavioral variables from Bpod data
8. Compute tongue/paw velocity from DeepLabCut tracking
9. Load and align motion energy data
10. Discretize continuous outputs using 50th percentile threshold

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion notes

## Running the Decoder
```bash
python train_decoder.py converted_data.pkl
```

## Decoder Performance
| Output | Validation Balanced Accuracy |
|--------|----------------------------|
| lick_direction | 0.655 |
| context | 0.861 |
| outcome | 0.663 |
| tongue_velocity | 0.768 |
| paw_velocity | 0.570 |
| motion_energy | 0.741 |

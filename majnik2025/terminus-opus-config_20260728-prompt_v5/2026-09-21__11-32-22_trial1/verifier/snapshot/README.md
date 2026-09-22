# Track2p Neural Decoder Dataset

## Dataset Description

This dataset contains longitudinal calcium imaging recordings from mouse barrel cortex (layer 2/3) during spontaneous behavior. Data is from Majnik et al. 2025 ("Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p").

### Key Features
- **6 mice** (jm031-jm046) imaged daily for 6-7 consecutive days during postnatal development (P7-P14)
- **Calcium indicator**: GCaMP8m
- **Imaging**: 2-photon, 30 Hz, 720×720 μm FOV, 512×512 pixels
- **Brain region**: Barrel cortex, layer 2/3
- **Neurons tracked across days** using Track2p algorithm
- **Behavioral measure**: Motion energy from videography

### Decoder Task
Decode motion energy (discretized into 5 equal-percentile bins) from neural activity.

## How to Load the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access input (time in session)
input_data = data['input'][0][0]  # shape: (1, n_timepoints)

# Access output (motion energy bins)
output_data = data['output'][0][0]  # shape: (1, n_timepoints)

# Metadata
print(data['metadata'])
```

## Data Format

### Structure
```
data = {
    'neural': list of 41 sessions, each containing 20 or 30 trials
                Each trial: (n_neurons, 180) float32 array
    'input': list of sessions/trials, each (1, 180) float32 - time in seconds
    'output': list of sessions/trials, each (1, 180) int64 - motion energy bin (0-4)
    'subjects': ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
    'subject_idx': (41,) array mapping sessions to subjects
    'brain_regions': ['barrel_cortex']
    'brain_region_idx': list of (n_neurons,) arrays, all zeros
    'input_names': ['time_in_session']
    'output_names': ['motion_energy']
    'output_values': [['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']]
    'metadata': dict with processing parameters
}
```

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions | 41 |
| Total trials | 1090 |
| Neurons per mouse | 221-746 (mean 499) |
| Time bins per trial | 180 (60 seconds at 3 Hz) |
| Time bin size | 333.3 ms (10 frames at 30 Hz) |
| Output classes | 5 (equal-percentile bins) |

### Processing Pipeline
1. **Neuropil correction**: Fc = F - 0.7 × Fneu
2. **Baseline estimation**: Maximin filter (σ=10, window=60s)
3. **Baseline subtraction**: dF = Fc - Flow
4. **Temporal binning**: Average 10 consecutive frames (30 Hz → 3 Hz)
5. **Motion energy**: Interpolate missing camera frames, bin, discretize into 5 quintiles per session
6. **Trial splitting**: 60-second non-overlapping windows

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl

# Train with sample plots
python train_decoder.py converted_data.pkl --plot-samples
```

## Files
- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - Sample dataset (2 sessions, 6 MB)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder implementation

## Reference
Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel JC, Cossart R (2025). Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p. eLife 14:RP107540.

# Zhong et al. 2025 - Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from 19 mice performing a visual discrimination task in virtual reality corridors. Mice discriminated between naturalistic visual textures (leaf vs circle patterns) in rewarded and unrewarded corridors.

**Reference**: Zhong et al. 2025, "Unsupervised pretraining in biological neural networks"

## Data Format

The converted dataset is stored in `converted_data.pkl` (pickle format).

### Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Data Structure

```python
data = {
    'neural': list of 89 sessions, each containing list of trials
              Each trial: np.array shape (n_neurons, n_timepoints), float32
    
    'input': list of 89 sessions, each containing list of trials
             Each trial: np.array shape (4, n_timepoints), float32
             Inputs: time_to_sound_cue, day_of_training, time_since_trial_start, reward_availability
    
    'output': list of 89 sessions, each containing list of trials
              Each trial: np.array shape (4, n_timepoints), int64
              Outputs: visual_stimulus_category, licking, position_bin, running_speed_bin
    
    'subjects': list of 19 subject names
    'subject_idx': np.array shape (89,) - index into subjects for each session
    'brain_regions': ['V1', 'mHV', 'lHV', 'aHV']
    'brain_region_idx': list of 89 arrays, each shape (n_neurons,)
    'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
    'output_names': ['visual_stimulus_category', 'licking', 'position_bin', 'running_speed_bin']
    'output_values': list of lists with category names for each output
    'metadata': dict with task description, time bin size, etc.
}
```

### Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 89 |
| Subjects | 19 |
| Total trials | 38,110 |
| Neurons per session | ~2,000 (subsampled from 20k-90k) |
| Time bin size | ~315.5 ms (3.17 Hz) |
| Stimulus categories | 12 |
| Brain regions | V1, mHV, lHV, aHV |

### Decoder Inputs
1. **time_to_sound_cue**: Time until sound cue (seconds), time-varying
2. **day_of_training**: Day of training, per-trial
3. **time_since_trial_start**: Time since corridor entry (seconds), time-varying
4. **reward_availability**: 1 if rewarded corridor, 0 if not, per-trial

### Decoder Outputs
1. **visual_stimulus_category**: Stimulus identity (12 categories), per-trial
2. **licking**: Binary licking (0/1), time-varying
3. **position_bin**: Position in corridor (4 bins of 1m), time-varying
4. **running_speed_bin**: Running speed quartile (4 bins), time-varying

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl

# Train with CPU only
python train_decoder.py converted_data.pkl --cpu
```

## Files

- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion notes
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model

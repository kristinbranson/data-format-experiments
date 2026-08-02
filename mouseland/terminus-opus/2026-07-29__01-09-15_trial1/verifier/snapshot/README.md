# Zhong et al. 2025 - Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from the paper "Unsupervised pretraining in biological neural networks" (Zhong et al., 2025). The data comes from head-fixed mice running through virtual reality corridors with naturalistic texture patterns (leaf, circle, rock, wood/brick).

### Experimental Setup
- **Task**: Visual discrimination in virtual reality corridors
- **Species**: Mouse (19 subjects)
- **Recording**: Two-photon calcium imaging (GCaMP6s)
- **Brain regions**: V1, medial higher visual (mHV), lateral higher visual (lHV), anterior higher visual (aHV)
- **Frame rate**: 3.17 Hz
- **Neural data**: Suite2p deconvolved fluorescence traces

### Experiment Types
- **Supervised (sup)**: Mice trained with water rewards in one corridor type
- **Unsupervised (unsup)**: Mice exposed to corridors without rewards
- **Naive**: Mice with no prior corridor exposure
- **Grating**: Control mice exposed to grating stimuli

## Data Format

The converted dataset is saved as `converted_data.pkl` and can be loaded with:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure

```python
data = {
    'neural': list of sessions, each containing list of trials
              Each trial: numpy array (n_neurons, n_timepoints), float16
    
    'input': list of sessions, each containing list of trials
             Each trial: numpy array (4, n_timepoints), float32
             Inputs:
               [0] time_to_sound_cue - signed time to sound cue (seconds)
               [1] day_of_training - chronological day index for this mouse
               [2] time_since_trial_start - time from corridor entry (seconds)
               [3] reward_availability - 1 if rewarded corridor, 0 if not
    
    'output': list of sessions, each containing list of trials
              Each trial: numpy array (4, n_timepoints), int64
              Outputs:
                [0] visual_stimulus - stimulus category index (13 categories)
                [1] licking - binary (0=no lick, 1=lick)
                [2] position_bin - corridor position (4 bins of 1m each)
                [3] speed_bin - running speed quartile (4 bins)
    
    'subjects': list of 19 mouse IDs
    'subject_idx': array of subject index per session
    'brain_regions': ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
    'brain_region_idx': list of arrays, one per session
    'input_names': ['time_to_sound_cue', 'day_of_training', ...]
    'output_names': ['visual_stimulus', 'licking', 'position_bin', 'speed_bin']
    'output_values': list of category names for each output
    'metadata': dict with task description, time bin size, etc.
}
```

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 76 |
| Subjects | 19 |
| Total trials | 31,442 |
| Neurons/session | 24,383 - 89,577 |
| Time bin | 315.5 ms (3.17 Hz) |
| Stimulus categories | 13 |
| File size | ~117 GB |

### Temporal Alignment
Trials are aligned to corridor entry (trial start). Each trial spans from the start of the textured corridor to the grey space transition.

## Usage

### Verify data format
```bash
python train_decoder.py converted_data.pkl --verify-only
```

### Train decoder
```bash
python train_decoder.py converted_data.pkl --cpu
```

### Regenerate converted data
```bash
python -u convert_data.py converted_data.pkl --full
```

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model
- `CONVERSION_NOTES.md` - Detailed conversion notes
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods

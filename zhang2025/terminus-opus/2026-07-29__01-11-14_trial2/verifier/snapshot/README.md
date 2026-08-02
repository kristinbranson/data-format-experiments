# IBL Brain-wide Map - Neural Decoder Dataset

## Dataset Description

This dataset contains neural recordings from the International Brain Laboratory (IBL) Brain-wide Map project, converted to a decoder-compatible format. The data includes Neuropixels recordings from 378 sessions across 125 mice performing a visual detection task with biased blocks.

## Task Description

Mice perform a two-alternative forced choice task where they must detect a visual stimulus (Gabor patch) appearing on the left or right side of a screen and turn a wheel to indicate their choice. The task includes biased blocks where the probability of the stimulus appearing on the left side changes (0.2, 0.5, or 0.8).

## Data Format

The converted data is stored in `converted_data.pkl` as a Python dictionary with the following structure:

### Loading the data
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure
- **`neural`**: List of 378 sessions, each containing a list of trials. Each trial is a numpy array of shape `(n_neurons, 100)` representing spike counts in 20ms bins.
- **`input`**: List of sessions/trials. Each trial has shape `(2, 100)`:
  - `input[0]`: Time since stimulus onset (seconds), ranges from -0.48 to 1.5
  - `input[1]`: Trial number within the current block (constant across time bins)
- **`output`**: List of sessions/trials. Each trial has shape `(4, 100)`:
  - `output[0]`: Choice (0=left, 1=right)
  - `output[1]`: Prior probability of left (0=0.2, 1=0.5, 2=0.8)
  - `output[2]`: Wheel speed (0=low, 1=medium, 2=high)
  - `output[3]`: Whisker motion energy (0=low, 1=medium, 2=high)
- **`subjects`**: List of 125 unique subject names
- **`subject_idx`**: Array of shape (378,) mapping sessions to subjects
- **`brain_regions`**: List of 274 brain region names (Beryl mapping)
- **`brain_region_idx`**: List of arrays, one per session, mapping neurons to brain regions
- **`input_names`**: `['time_since_stim_onset', 'trial_number_in_block']`
- **`output_names`**: `['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy']`
- **`output_values`**: Names for each output category
- **`metadata`**: Processing parameters and dataset information

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 378 |
| Subjects | 125 |
| Total trials | 164,322 |
| Mean trials/session | 434.7 |
| Total neurons | 539,857 |
| Mean neurons/session | 1,428.2 |
| Brain regions | 274 |
| Time bins per trial | 100 |
| Time bin size | 20 ms |
| Time window | -0.5 to 1.5s relative to stimulus onset |

## Processing Pipeline

1. **Spike binning**: Spike times binned at 20ms resolution, aligned to stimulus onset
2. **Trial filtering**: Excluded trials with RT < 0.08s or > 2.0s, NaN events, no-choice trials
3. **Wheel processing**: Position interpolated to 1000Hz, Butterworth filtered (20Hz corner, order 8), velocity computed, speed = |velocity|
4. **Whisker motion energy**: Loaded from left camera (fallback to right), interpolated to match neural bins
5. **Discretization**: Wheel speed and whisker ME discretized into 3 equal-frequency bins using global quantiles
6. **Brain regions**: Mapped using IBL atlas Beryl mapping

## Decoder Performance

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| Choice | 0.558 | 0.500 |
| Prior probability | 0.565 | 0.333 |
| Wheel speed | 0.579 | 0.333 |
| Whisker motion energy | 0.711 | 0.333 |

## Files

- `converted_data.pkl` - Full converted dataset (90 GB)
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder implementation
- `CONVERSION_NOTES.md` - Detailed conversion notes

## References

- IBL et al. (2023). "A brain-wide map of neural activity during complex behaviour." 
- Zhang et al. (2025). "Exploiting correlations across trials and behavioral sessions to improve neural decoding."

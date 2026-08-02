# IBL Brain-Wide Map Neural Decoder Dataset

## Dataset Description

This dataset contains neural recordings and behavioral data from the International Brain Laboratory (IBL) Brain-Wide Map project, converted into a standardized format suitable for training neural decoders.

### Source
- **Data Paper**: "A brain-wide map of neural activity during complex behaviour" (IBL, 2023)
- **Methods Paper**: "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al., 2025)
- **Data**: IBL Neuropixels recordings from mice performing a visual decision-making task

### Task
Mice rotate a wheel to indicate the location of a visual stimulus (left or right). The task includes:
- Visual stimulus at varying contrasts (0%, 6.25%, 12.5%, 25%, 100%)
- Block structure with biased stimulus probabilities (20:80 or 80:20)
- Initial unbiased block (first 90 trials)

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 340 |
| Subjects (mice) | 119 |
| Total trials | 149,339 |
| Brain regions | 269 |
| Neurons per session | ~1,200 (mean) |
| Trials per session | ~440 (mean) |
| Time bin size | 20 ms |
| Time steps per trial | 100 |
| Temporal alignment | Stimulus onset |
| Time window | -0.5 to 1.5 s |

## Data Format

The data is stored in `converted_data.pkl` as a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure

- **`neural`**: List of sessions, each containing a list of trials. Each trial is a `(n_neurons, 100)` uint8 array of spike counts.
- **`input`**: Decoder inputs. Each trial is a `(2, 100)` float32 array:
  - `input[0]`: Time since stimulus onset (seconds)
  - `input[1]`: Trial number within current block
- **`output`**: Decoder outputs. Each trial is a `(4, 100)` int8 array:
  - `output[0]`: Choice (0=left, 1=right) - per-trial, constant across time
  - `output[1]`: Prior probability of left (0=0.2, 1=0.5, 2=0.8) - per-trial
  - `output[2]`: Wheel speed (0=low, 1=medium, 2=high) - time-varying
  - `output[3]`: Whisker motion energy (0=low, 1=medium, 2=high) - time-varying
- **`subjects`**: List of subject names
- **`subject_idx`**: Index into subjects for each session
- **`brain_regions`**: List of brain region names (Beryl mapping)
- **`brain_region_idx`**: List of arrays mapping neurons to brain regions
- **`input_names`**: `['time_since_stim_onset', 'trial_number_in_block']`
- **`output_names`**: `['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']`
- **`output_values`**: Names for each output class
- **`metadata`**: Task description, timing info, filtering criteria

## Processing Pipeline

1. **Neural data**: Spike counts binned at 20ms in a 2s window around stimulus onset
2. **Trial filtering**: Reaction time 0.08-2.0s, no-choice trials excluded, NaN trials excluded
3. **Wheel speed**: Position interpolated to 1000Hz, Butterworth filtered (20Hz, order 8), velocity computed, absolute value taken, interpolated to trial bins, discretized into 3 quantile-based bins
4. **Whisker motion energy**: Loaded from ROI motion energy files, interpolated to trial bins, discretized into 3 quantile-based bins
5. **Brain regions**: Mapped using IBL Atlas Beryl mapping

## How to Train a Decoder

```bash
python train_decoder.py converted_data.pkl
```

Options:
- `--verify-only`: Only verify data format
- `--plot-samples`: Plot sample trials
- `--cpu`: Force CPU training

## Files

- `converted_data.pkl`: Full converted dataset (20.7 GB)
- `sample_data.pkl`: Sample dataset (2 sessions, 240 MB)
- `convert_data.py`: Conversion script
- `train_decoder.py`: Decoder training script
- `CONVERSION_NOTES.md`: Detailed conversion documentation

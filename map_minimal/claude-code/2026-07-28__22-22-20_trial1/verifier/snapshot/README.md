# MAP Dataset: NWB to Neural Decoder Format

Converts Neuropixels electrophysiology data from the MAP (Mesoscale Activity Project) dataset into a Python dictionary format for training neural decoders.

## Dataset Overview

The MAP dataset (Li et al.) contains brain-wide Neuropixels recordings from mice performing an auditory delayed response task. Mice hear tones during a sample epoch, wait through a delay period, and lick left or right at a go cue.

**Converted data**: 105 sessions, 25 subjects, 48,356 trials, 41,197 neurons across 15 brain regions.

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script (NWB -> pickle) |
| `converted_data.pkl` | Full converted dataset (~5.95 GB) |
| `sample_data.pkl` | Sample dataset (5 sessions, 250 trials, ~29 MB) |
| `decoder.py` | Decoder implementation (verification + training) |
| `train_decoder.py` | Script to verify and train decoder |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and validation |
| `conversion_full_out.txt` | Full conversion log |
| `conversion_sample_out.txt` | Sample conversion log |
| `verification_full_out.txt` | Full dataset verification output |
| `verification_sample_out.txt` | Sample dataset verification output |
| `train_decoder_full_out.txt` | Full dataset decoder training output |
| `train_decoder_sample_out.txt` | Sample dataset decoder training output |

## Data Format

The pickle file contains a dictionary with keys:

```python
{
    'neural': list[list[np.ndarray]],      # [session][trial] -> (n_neurons, T) float64
    'input': list[list[np.ndarray]],       # [session][trial] -> (n_input, T) float64
    'output': list[list[np.ndarray]],      # [session][trial] -> (n_output, T) int64
    'subjects': list[str],                 # unique subject names
    'subject_idx': list[int],              # session -> subject index
    'brain_regions': list[str],            # unique region names
    'brain_region_idx': list[list[int]],   # [session] -> region index per neuron
    'input_names': ['time_from_tone_onset', 'photostimulation'],
    'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
    'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
    'metadata': dict,                      # conversion parameters and task description
}
```

## Usage

### Convert from NWB
```bash
# Full dataset
python convert_data.py

# Sample dataset (5 sessions, 50 trials each)
python convert_data.py --sample
```

### Verify and Train
```bash
# Verify format and train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples

# On sample data
python train_decoder.py sample_data.pkl --cpu --plot-samples
```

### Load in Python
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access session 0, trial 0 neural data
fr = data['neural'][0][0]  # shape: (n_neurons, 80)

# Access outputs
choice = data['output'][0][0][0, 0]  # 0=left, 1=right
```

## Conversion Parameters

- **Time window**: -2.5 to +1.5 s relative to go cue onset
- **Bin width**: 50 ms (80 time bins)
- **QC**: Classifier-based quality control ('good' units only)
- **Trial filter**: Early lick and no-response trials excluded
- **Session filter**: >65% correct, >= 50 correct trials per side

See `CONVERSION_NOTES.md` for detailed decisions and validation results.

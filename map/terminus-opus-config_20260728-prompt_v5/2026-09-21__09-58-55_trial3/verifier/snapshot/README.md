# MAP Dataset - Neural Decoder Format

## Dataset Description

This is the Mesoscale Activity Map (MAP) dataset (DANDI:000363) converted to a decoder-compatible format. The original data comes from the paper "Brain-wide neural activity underlying memory-guided movement" (Chen et al.) and was further analyzed in "Brain-wide analysis reveals movement encoding structured across and within brain areas" (Wang et al.).

### Task
Mice perform an audio delayed-response licking task:
1. **Presample period**: Baseline period before stimulus
2. **Sample period**: A tone is played (0.65s duration) indicating the correct lick direction
3. **Delay period**: Variable duration (typically 1.2s) where the mouse must wait
4. **Go cue**: Signals the mouse to respond by licking
5. **Response**: Mouse licks left or right (or doesn't lick)

On ~25% of trials in VGAT-ChR2-EYFP mice, bilateral ALM photoinhibition is applied during the late delay period.

### Neural Data
- Extracellular electrophysiology (Neuropixels probes)
- 69,453 quality-controlled units across 27 brain regions
- 173 sessions from 28 mice
- ~93,290 trials total

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 80)

# Access inputs for session 0, trial 0
inputs = data['input'][0][0]  # shape: (2, 80)
# inputs[0] = time from tone onset (seconds)
# inputs[1] = photostim on (binary)

# Access outputs for session 0, trial 0
outputs = data['output'][0][0]  # shape: (4, 80)
# outputs[0] = choice (0=left, 1=right, 2=no_lick)
# outputs[1] = outcome (0=ignore, 1=miss, 2=hit)
# outputs[2] = early_lick (0=no, 1=yes)
# outputs[3] = tongue_y_position (0=below_p40, 1=p40_to_p60, 2=above_p60, 3=not_visible)
```

## Data Format

### Structure
```
data = {
    'neural': list of sessions, each containing list of trials (n_neurons, 80)
    'input': list of sessions, each containing list of trials (2, 80)
    'output': list of sessions, each containing list of trials (4, 80)
    'subjects': list of subject IDs
    'subject_idx': array mapping sessions to subjects
    'brain_regions': list of brain region names
    'brain_region_idx': list of arrays mapping neurons to regions
    'input_names': ['time_from_tone_onset', 'photostim_on']
    'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position']
    'output_values': [['left', 'right', 'no_lick'], ['ignore', 'miss', 'hit'], ['no', 'yes'], ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible']]
    'metadata': dict with task description, timing info, etc.
}
```

### Temporal Alignment
- Aligned to **go cue onset**
- Window: **-2.5s to +1.5s** relative to go cue
- Bin size: **50ms** (80 time bins)

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 173 |
| Subjects | 28 |
| Total trials | ~93,290 |
| Total neurons | 69,453 |
| Mean neurons/session | ~400 |
| Mean trials/session | ~540 |
| Time bins/trial | 80 |
| Brain regions | 27 |

## Files

- `converted_data.pkl` - Full converted dataset (~12 GB)
- `sample_data.pkl` - Sample dataset (2 sessions, ~93 MB)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion documentation

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --plot-samples

# Use CPU if GPU memory is insufficient
python train_decoder.py converted_data.pkl --cpu
```

## Decoder Results

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| Choice | 0.664 | 0.333 |
| Outcome | 0.653 | 0.333 |
| Early Lick | 0.760 | 0.500 |
| Tongue Y Position | 0.658 | 0.250 |

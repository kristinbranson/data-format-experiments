# MAP Dataset - Decoder-Compatible Format

## Dataset Description

This dataset contains neural recordings from the **Mesoscale Activity Project (MAP)** dataset (DANDI:000363), converted to a decoder-compatible format.

**Task**: Auditory delayed response task. Mice hear tones during a sample epoch, maintain memory during a delay period, then lick left or right after a go cue.

**Neural data**: Neuropixels recordings from multiple brain regions (ALM, Striatum, Thalamus, Midbrain, Medulla, BLA, ECT) across both hemispheres.

**Reference papers**:
- Chen et al. "Brain-wide neural activity underlying memory-guided movement"
- Wang et al. "Brain-wide analysis reveals movement encoding structured across and within brain areas"

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 28 |
| Sessions | 143 |
| Total neurons | 57,560 |
| Total trials | 74,484 |
| Mean neurons/session | 402.5 |
| Mean trials/session | 520.9 |
| Brain regions | 14 |
| Time bins | 80 (50ms each) |
| Time window | -2.5s to 1.5s (go cue aligned) |

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 80)

# Access inputs
input_data = data['input'][0][0]  # shape: (2, 80)
# input_data[0]: time from tone onset (continuous)
# input_data[1]: photostim on/off (binary)

# Access outputs  
output_data = data['output'][0][0]  # shape: (4, 80)
# output_data[0]: choice (0=left, 1=right)
# output_data[1]: outcome (0=ignore, 1=miss, 2=hit)
# output_data[2]: early_lick (0=no, 1=yes)
# output_data[3]: tongue_y_position (0=below_p40, 1=p40_to_p60, 2=above_p60)
```

## Data Format

```python
data = {
    'neural': list of sessions, each a list of trials (n_neurons, 80)
    'input': list of sessions, each a list of trials (2, 80)
    'output': list of sessions, each a list of trials (4, 80)
    'subjects': list of subject IDs
    'subject_idx': array mapping sessions to subjects
    'brain_regions': list of brain region names
    'brain_region_idx': list of arrays mapping neurons to brain regions
    'input_names': ['time_from_tone_onset', 'photostim_on']
    'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position']
    'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['below_p40','p40_to_p60','above_p60']]
    'metadata': dict with task description and parameters
}
```

## Processing Details

- **Neural data**: Spike times from "good" units (QC classifier) binned into 50ms firing rate bins
- **Temporal alignment**: Go cue onset = time 0
- **Window**: -2.5s to +1.5s relative to go cue
- **Trial filtering**: Excluded auto_water and free_water trials; included early lick, ignore, and photostim trials
- **Session filtering**: >65% correct rate, >=50 correct trials per side
- **Tongue y discretization**: Per-session percentiles (40th, 60th)

## Decoder Training

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --plot-samples
```

## Decoder Results

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| choice | 0.717 | 0.500 |
| outcome | 0.653 | 0.333 |
| early_lick | 0.748 | 0.500 |
| tongue_y_position | 0.784 | 0.333 |

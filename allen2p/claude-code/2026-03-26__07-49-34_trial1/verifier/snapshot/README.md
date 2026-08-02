# Allen Brain Observatory Visual Behavior 2P - Decoder Dataset

## Dataset Description

Converted data from the Allen Brain Observatory Visual Behavior 2-photon (2P) calcium imaging dataset for neural decoding. Head-fixed mice performed a go/no-go change detection task: they viewed a continuous stream of flashed natural images (250ms on, 500ms off) and licked to report identity changes.

**Source**: Allen Institute for Brain Science, Visual Behavior Optical Physiology dataset v1.1.0

**References**:
- Whitepaper: "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper"
- Paper: Piet et al. (2024), "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex", Neuron

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects (mice) | 38 |
| Sessions | 202 |
| Total trials | 51,992 |
| Total neurons | 29,444 |
| Brain regions | VISp (29,282), VISl (162) |
| Time bin | ~32.3 ms (~31 Hz) |
| Neural signal | dF/F (calcium imaging) |

## Output Format

The converted data is saved as `converted_data.pkl` (Python pickle). Load with:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Data Structure

```python
data = {
    'neural': [sessions x [trials x (n_neurons, T)]],  # dF/F traces
    'input': [sessions x [trials x (0, T)]],            # no inputs
    'output': [sessions x [trials x (5, T)]],           # 5 output variables
    'subjects': list of str,           # 38 mouse IDs
    'subject_idx': np.array,           # session -> subject mapping
    'brain_regions': ['VISp', 'VISl'],
    'brain_region_idx': [np.array per session],  # neuron -> region mapping
    'input_names': [],
    'output_names': ['image_identity', 'image_change', 'running_speed',
                     'pupil_diameter', 'trial_outcome'],
    'output_values': [list of category names per output],
    'metadata': dict,
}
```

### Output Variables

| # | Name | Type | Categories | Description |
|---|------|------|------------|-------------|
| 0 | image_identity | Time-varying, 17 classes | gray + 16 images | Currently displayed image |
| 1 | image_change | Time-varying, binary | no_change, change | 1 at image change onset |
| 2 | running_speed | Time-varying, 5 bins | bin_0 to bin_4 | Percentile-binned running speed |
| 3 | pupil_diameter | Time-varying, 5 bins | bin_0 to bin_4 | Percentile-binned pupil diameter |
| 4 | trial_outcome | Static per trial | hit, miss, false_alarm, correct_reject | Behavioral outcome |

### Trial Selection
- Included: Go trials (image changes) and Catch trials (sham changes)
- Excluded: Aborted trials (premature licks) and Auto-rewarded trials

## How to Run

```bash
# Convert data (full dataset, ~9 min)
python3 -u convert_data.py converted_data.pkl --full

# Convert sample (2 sessions, ~8 sec)
python3 -u convert_data.py sample_data.pkl --sample

# Verify format
python3 -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (8.5 GB) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder module |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `cache/` | Intermediate files |

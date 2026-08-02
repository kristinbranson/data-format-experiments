# IBL Brain-Wide Map - Neural Decoder Dataset

## Dataset Description

This dataset contains neural recordings from the International Brain Laboratory (IBL) Brain-Wide Map project, converted into a format suitable for training neural decoders. The data comes from Neuropixels recordings across 444 sessions in 136 mice performing a visual decision-making task.

### Task
Mice rotate a wheel to indicate the location of a visual stimulus (left or right). The stimulus appears with varying contrast and with block-dependent prior probabilities (20%, 50%, or 80% left).

### Data Source
- Data paper: "A brain-wide map of neural activity during complex behaviour" (IBL et al.)
- Methods paper: "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al.)

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 444 |
| Subjects (mice) | 136 |
| Total trials | 188,985 |
| Total neurons | 599,865 |
| Brain regions (Beryl) | 281 |
| Mean neurons/session | 1,351 |
| Mean trials/session | 426 |
| Time bins per trial | 100 |
| Bin size | 20 ms |
| Trial duration | 2.0 s |
| Time window | -0.5 to 1.5 s (relative to stimulus onset) |

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]  # shape: (n_neurons, 100)
inputs = data['input'][0][0]   # shape: (2, 100)
outputs = data['output'][0][0] # shape: (4, 100)
```

## Data Format

### Neural Data
- Spike counts binned in 20ms non-overlapping bins
- Aligned to stimulus onset (stimOn_times)
- Window: -0.5s to 1.5s (100 time bins)
- All neurons from all probes merged per session
- No quality control filtering (all Kilosort 2.5 sorted units included)

### Decoder Inputs (2 variables)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time_since_stim_onset | Continuous, time-varying | Time in seconds relative to stimulus onset |
| 1 | trial_number_in_block | Continuous, per-trial | Trial number within current probability block |

### Decoder Outputs (4 variables)
| Index | Name | Type | Values | Description |
|-------|------|------|--------|-------------|
| 0 | choice | Binary, per-trial | 0=left, 1=right | Mouse choice |
| 1 | prior | 3-class, per-trial | 0=0.2, 1=0.5, 2=0.8 | Prior probability of left stimulus |
| 2 | wheel_speed | 3-class, time-varying | 0=low, 1=medium, 2=high | Discretized wheel speed |
| 3 | whisker_motion_energy | 3-class, time-varying | 0=low, 1=medium, 2=high | Discretized whisker motion energy |

### Other Fields
- subjects: List of subject names
- subject_idx: Subject index for each session
- brain_regions: List of brain region names (Beryl mapping)
- brain_region_idx: Brain region index for each neuron in each session
- metadata: Task description, timing parameters, bin edges

## Training the Decoder

```bash
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl --plot-samples --cpu
```

## Decoder Performance

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| Choice | 0.559 | 0.500 |
| Prior | 0.612 | 0.333 |
| Wheel speed | 0.588 | 0.333 |
| Whisker ME | 0.716 | 0.333 |

## Conversion

```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
python -u convert_data.py converted_data.pkl --full
```

## Files

| File | Description |
|------|-------------|
| converted_data.pkl | Full converted dataset (99 GB) |
| sample_data.pkl | Sample dataset (2 sessions) |
| convert_data.py | Conversion script |
| train_decoder.py | Decoder training script |
| decoder.py | Decoder implementation |
| CONVERSION_NOTES.md | Detailed conversion notes |
| README.md | This file |

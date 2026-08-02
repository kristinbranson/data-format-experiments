# IBL Brain-wide Map - Decoder-Compatible Dataset

Converted dataset from the International Brain Laboratory (IBL) Brain-wide Map project into a format suitable for neural decoder training.

## Dataset Summary

| Statistic | Value |
|-----------|-------|
| Sessions | 335 |
| Subjects | 118 |
| Total trials | 146,747 |
| Brain regions | 269 |
| Time bins per trial | 100 (20ms bins, 2s window) |
| Alignment | stimulus onset (stimOn_times) |
| Trial window | -0.5s to +1.5s |

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full dataset (21 GB, uint8 neural data) |
| `converted_data_reduced.pkl` | Neuron-subsampled version (7.4 GB, max 500 neurons/session) |
| `convert_data.py` | Conversion script |
| `reduce_data.py` | Neuron subsampling script |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder model code |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |

## Data Format

The pickle file contains a dictionary with:

```python
{
    'neural':           # list[session][trial] -> np.array(nneurons, 100), uint8
    'input':            # list[session][trial] -> np.array(2,) or (2, 100), float32
    'output':           # list[session][trial] -> np.array(4,) or (4, 100), int64
    'input_names':      # ['time_since_stim_onset', 'trial_num_in_block']
    'output_names':     # ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']
    'output_values':    # [[0,1], [0,1,2], [0,1,2], [0,1,2]]
    'subject_idx':      # np.array(nsessions,), int32
    'brain_region_idx': # list[session] -> np.array(nneurons,), int32
    'subject_names':    # list of subject name strings
    'brain_region_names': # list of brain region name strings
}
```

### Outputs
- **choice**: 0=left, 1=right (per-trial)
- **prior**: 0=0.2, 1=0.5, 2=0.8 probabilityLeft (per-trial)
- **wheel_speed**: 0/1/2 discretized terciles (per-timestep)
- **whisker_motion_energy**: 0/1/2 discretized terciles (per-timestep)

## Decoder Results

Trained with PCA (100 components), balanced cross-entropy loss, 200 epochs:

| Output | Validation Balanced Acc | Chance |
|--------|------------------------|--------|
| choice | 0.633 | 0.500 |
| prior | 0.713 | 0.333 |
| wheel_speed | 0.654 | 0.333 |
| whisker_motion_energy | 0.640 | 0.333 |

## Usage

```bash
# Verify data format
python train_decoder.py converted_data_reduced.pkl --verify-only

# Train decoder
python train_decoder.py converted_data_reduced.pkl --plot-samples --cpu
```

## Processing Decisions

Key decisions documented in `CONVERSION_NOTES.md`:
1. **Alignment**: stimOn_times for all behaviors (matching reference code + decoder task)
2. **Bin size**: 20ms uniformly (matching reference code)
3. **Neurons**: All neurons used (no QC filtering, matching reference code qc=None)
4. **Neuron subsampling**: Max 500 neurons/session in reduced version (decoder uses PCA to 100 components)
5. **Discretization**: Session-wide terciles for wheel speed and whisker ME

## Source

- Data paper: "A brain-wide map of neural activity during complex behaviour" (IBL, 2023)
- Methods paper: "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al., 2025)

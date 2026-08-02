# Geometric Representations in CA1 - Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from hippocampal CA1 neurons recorded while mice freely explored 10 geometrically distinct environments. The data has been converted from the original format (Bhattarai et al., "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping") into a standardized decoder-compatible format.

### Experiment
- **Task**: Free exploration of geometric environments
- **Brain region**: CA1 (hippocampus)
- **Recording method**: Miniscope calcium imaging at 30 Hz
- **Session duration**: 40 minutes per session
- **Environments**: 10 distinct geometries formed by partitioning a 75×75 cm square arena into a 3×3 grid

### Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 7 mice |
| Sessions | 207 |
| Trials | 8,187 (1-minute segments) |
| Unique neurons | 5,413 |
| Active neuron-sessions | 69,744 |
| Mean neurons/session | 336.9 |
| Time bin size | 33.33 ms (30 Hz) |
| Frames per trial | 1,800 |

## How to Load the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 1800)

# Access environment geometry input for session 0, trial 0
input_data = data['input'][0][0]  # shape: (9,) - flattened 3x3 binary grid

# Access position output for session 0, trial 0
output_data = data['output'][0][0]  # shape: (1, 1800) - position bin 0-8
```

## Output Format

### Neural Data
- Binary rising-phase calcium transient vectors (0 or 1)
- Shape per trial: (n_neurons, 1800)
- Only active neurons (tracked on that day) are included

### Decoder Input: Environment Geometry
- 9-element binary vector (flattened 3×3 grid)
- 1 = accessible region, 0 = blocked region
- Static per trial (same for all timepoints)
- Names: env_grid_0 through env_grid_8

### Decoder Output: Mouse Position
- Discretized into 3×3 = 9 spatial bins (0-8)
- Time-varying (one value per frame)
- Bin index = row × 3 + column
- Each bin covers 25×25 cm of the 75×75 cm arena

### Metadata
- `subjects`: List of 7 mouse IDs
- `subject_idx`: Session-to-subject mapping
- `brain_regions`: ['CA1']
- `brain_region_idx`: All neurons mapped to CA1
- `metadata`: Recording parameters and session info

## Decoder Performance

| Output | Training Acc | Validation Acc | Chance |
|--------|-------------|---------------|--------|
| Position (3×3) | 61.91% | 54.51% | 11.11% |

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (19.98 GB) |
| `sample_data.pkl` | Sample dataset (2 animals, 5.21 GB) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `README.md` | This file |

## Reference

Bhattarai et al., "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"

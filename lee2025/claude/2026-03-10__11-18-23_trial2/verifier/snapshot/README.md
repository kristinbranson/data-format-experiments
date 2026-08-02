# QLAK-CA1 Geometric Representations - Decoder Dataset

## Dataset Description

Converted dataset from Lee et al. (2025) "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping" (Neuron 113(2): 307-320).

**Task**: Decode mouse position (3x3 spatial bins) from CA1 hippocampal neural activity during free exploration of geometrically varying environments.

### Experiment Summary
- 7 mice with CA1 calcium imaging (miniscope, 30 Hz)
- Each mouse explored a sequence of 10 distinct environment geometries created by partitioning a 75x75 cm square arena into a 3x3 grid
- Each session = 1 day recording (~40 min)
- 207 total sessions, 5,413 unique neurons, 69,744 rate maps

## Loading the Data

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 600)

# Access environment geometry input
env_input = data['input'][0][0]  # shape: (9,) - flattened 3x3 binary

# Access position output
position = data['output'][0][0]  # shape: (1, 600) - bin indices 0-8
```

## Output Format

### Key Fields
| Field | Description |
|-------|-------------|
| `neural` | List of sessions, each a list of trials. Each trial: (n_neurons, 600) float32 |
| `input` | Environment geometry as flattened 3x3 binary matrix. (9,) float32, static per trial |
| `output` | Mouse position as 3x3 bin index (0-8). (1, 600) int64, time-varying |
| `subjects` | 7 mouse IDs |
| `subject_idx` | Session-to-subject mapping |
| `brain_regions` | ['CA1'] |
| `brain_region_idx` | Per-session neuron-to-region mapping |

### Processing Details
- **Neural data**: Binary calcium events (rising phase), Gaussian smoothed (sigma=3 frames), then averaged in 3-frame bins (100 ms)
- **Position**: x,y tracking discretized into 3x3 grid (25 cm bins), temporally binned by 3 frames
- **Trials**: 1-minute segments within each ~40-min session (39-40 trials per session)
- **Time bin**: 100 ms (3 frames at 30 Hz) = 600 time bins per trial

### Environment Geometries (Input)
The 3x3 binary matrix indicates which partitions of the arena are accessible (1) or blocked (0):
```
Partition layout: [[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]]
```
10 geometries: square, o, t, u, rectangle, +, i, l, bit donut, glenn

### Position Bins (Output)
```
Output bin layout: [[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]]
```
Each bin covers 25x25 cm of the 75x75 cm arena.

## Key Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Sessions | 207 |
| Total trials | 8,187 |
| Time bins per trial | 600 |
| Total cell-days | 69,744 |
| Neurons per session | 113-564 (mean 337) |
| Decoder validation accuracy | 0.607 (chance: 0.111) |

## Running the Decoder

```bash
# Verify data format
python3 train_decoder.py converted_data.pkl --verify-only

# Train and evaluate decoder
python3 train_decoder.py converted_data.pkl --plot-samples

# Regenerate the converted data
python3 convert_data.py converted_data.pkl --full
```

## Reference
Lee, J.Q., Keinath, A.T., Cianfarano, E., & Brandon, M.P. (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron, 113(2), 307-320.

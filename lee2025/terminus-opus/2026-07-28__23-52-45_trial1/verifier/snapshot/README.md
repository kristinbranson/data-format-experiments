# CA1 Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging recordings from mouse hippocampal CA1 neurons during exploration of geometrically distinct environments. The data comes from:

> Lee, Keinath, Cianfarano & Brandon (2025). "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping." Neuron 113(2): 307-320.

### Experimental Setup
- 7 mice explored a 75×75 cm² arena
- The arena was partitioned into a 3×3 grid
- 10 unique geometric environments were created by blocking different grid partitions with 25 cm walls
- Each mouse was recorded for up to 31 daily sessions (~40 minutes each)
- Neural activity was recorded via calcium imaging at 30 Hz

### Decoder Task
- **Input**: Environment geometry (3×3 binary matrix indicating open/blocked partitions)
- **Output**: Mouse position discretized into 3×3 = 9 spatial bins (time-varying)
- **Neural**: Binary calcium events from CA1 neurons, binned into 1-second time bins

## Data Format

The converted dataset is saved in `converted_data.pkl` as a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure
- `data['neural']`: List of 207 sessions, each containing ~39-40 trials of shape (n_neurons, 60)
- `data['input']`: Environment geometry per trial, shape (9,) - static per trial
- `data['output']`: Position bin index per trial, shape (1, 60) - time-varying
- `data['subjects']`: 7 mouse IDs
- `data['subject_idx']`: Session-to-subject mapping
- `data['brain_regions']`: ['CA1']
- `data['brain_region_idx']`: All neurons are CA1
- `data['metadata']`: Task description, time bin size (1000ms), etc.

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Total sessions | 207 |
| Total trials | 8,187 |
| Trials/session | 39-40 |
| Total unique neurons | 5,413 |
| Valid neuron-sessions | 69,744 |
| Neurons/session | 113-564 (mean 337) |
| Time bins/trial | 60 (1 second each) |
| Input dimension | 9 (environment geometry) |
| Output classes | 9 (3×3 position bins) |

### Output Values
Position bins are indexed 0-8 in row-major order:
```
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
```

### Input Values
Environment geometry is a flattened 3×3 binary matrix:
- 1.0 = open partition
- 0.0 = blocked partition

## Decoder Performance
- Validation balanced accuracy: **0.6697** (chance: 0.1111, 6.0× above chance)
- Training balanced accuracy: 0.7937

## Files
- `converted_data.pkl` - Full converted dataset (667.8 MB)
- `sample_data.pkl` - Sample dataset (2 animals, 174.3 MB)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion notes
- `verification_full_out.txt` - Full data verification output
- `train_decoder_full_out.txt` - Full decoder training output

## Usage

### Convert data
```bash
# Full conversion
python -u convert_data.py converted_data.pkl --full

# Sample conversion (2 animals)
python -u convert_data.py sample_data.pkl --sample

# With processing visualizations
python -u convert_data.py sample_data.pkl --sample --show-processing
```

### Train decoder
```bash
python -u train_decoder.py converted_data.pkl
```

### Verify data format
```bash
python -u train_decoder.py converted_data.pkl --verify-only
```

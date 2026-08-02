# MAP Dataset - Neural Decoder Format

## Dataset Description

This dataset contains neural recordings from the Mesoscale Activity Map (MAP) project, converted from NWB format to a decoder-compatible Python dictionary structure.

**Source**: DANDI:000363 - "Brain-wide neural activity underlying memory-guided movement"
**Reference Papers**:
- Chen et al., "Brain-wide neural activity underlying memory-guided movement" (data paper)
- Wang, Kurgyis et al., "Brain-wide analysis reveals movement encoding structured across and within brain areas" (Nature Neuroscience 2025)

## Task Description

Mice performed an **auditory delayed response task**:
1. **Sample epoch** (-1.85 to -1.20s): Three pure tones (3kHz or 12kHz) played for 150ms with 100ms gaps
2. **Delay epoch** (-1.20 to 0.00s): 1.2s delay period
3. **Go cue** (0.00s): Auditory go cue signals response period
4. **Response epoch** (0.00 to 1.50s): Mice lick left or right port to report tone identity

## Data Format

The converted data is stored in `converted_data.pkl` as a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Keys:
- **`neural`**: List of sessions, each containing list of trials with shape `(n_neurons, 80)` - firing rates in Hz
- **`input`**: List of sessions/trials with shape `(2, 80)` - decoder inputs
  - `input[0]`: Time from tone onset (seconds)
  - `input[1]`: Photostimulation on/off (binary)
- **`output`**: List of sessions/trials with shape `(4, 80)` - decoder outputs
  - `output[0]`: Lick direction choice (0=left, 1=right)
  - `output[1]`: Outcome (0=ignore, 1=miss, 2=hit)
  - `output[2]`: Early lick (0=no, 1=yes)
  - `output[3]`: Tongue y-position (0=low, 1=mid, 2=high, discretized per session)
- **`subjects`**: List of 28 subject IDs
- **`subject_idx`**: Session-to-subject mapping
- **`brain_regions`**: List of brain region names
- **`brain_region_idx`**: Neuron-to-region mapping per session
- **`metadata`**: Task description, timing parameters, session info

### Key Parameters:
- **Time bin size**: 50 ms
- **Temporal alignment**: Go cue onset
- **Window**: -2.5 to +1.5 s relative to go cue
- **Time bins per trial**: 80

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 151 |
| Subjects | 28 |
| Total trials | 81,188 |
| Total neurons (good) | 60,324 |
| Mean trials/session | 537.7 |
| Mean neurons/session | 399.5 |
| Brain regions | 138 |

### Output Distributions
| Output | Values | Distribution |
|--------|--------|-------------|
| Choice | left (0), right (1) | 48.9%, 51.1% |
| Outcome | ignore (0), miss (1), hit (2) | 12.5%, 15.4%, 72.1% |
| Early lick | no (0), yes (1) | 88.8%, 11.2% |
| Tongue y | low (0), mid (1), high (2) | 40%, 20%, 40% |

## Session Selection Criteria

Sessions were selected based on:
1. At least 2 good units (classifier-based quality control)
2. Overall behavioral performance > 65% correct
3. At least 50 correct lick-left and lick-right trials each

## Neuron Quality Control

Neurons were filtered using classifier-based quality control (5 region-specific classifiers trained on manual curation labels). Only units classified as 'good' were included.

## Usage

```python
# Train decoder
python train_decoder.py converted_data.pkl

# Verify data format only
python train_decoder.py converted_data.pkl --verify-only

# Convert data (full)
python -u convert_data.py converted_data.pkl --full

# Convert data (sample, 2 sessions)
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Files

- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model implementation

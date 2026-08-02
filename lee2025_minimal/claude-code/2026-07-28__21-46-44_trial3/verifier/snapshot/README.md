# georepca1 Neural Decoder Dataset

## Overview
This dataset converts CA1 calcium imaging data from Lee et al. (2025) into a standardized format for neural decoding of mouse position from hippocampal neural activity.

**Reference**: Lee, Keinath, Cianfarano & Brandon (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2): 307-320.

## Task
Decode mouse position (3x3 = 9 spatial bins) from CA1 binary calcium transient activity, with environment geometry as context input.

## Dataset Structure

### Neural Data
- Binary calcium transient events (0/1) at 30 Hz
- Only registered (non-NaN) cells per session
- Shape per trial: (n_neurons, 1800) for 1-minute trials

### Input: Environment Geometry
- 3x3 binary grid (9 values): 1 = open partition, 0 = blocked
- Static per trial (same environment throughout a recording session)

### Output: Mouse Position
- 9 spatial bins (3x3 grid, each 25x25 cm in 75x75 cm arena)
- Time-varying: one bin index per timepoint
- Row-major: row0_col0, row0_col1, ..., row2_col2

## Key Statistics
| Metric | Value |
|--------|-------|
| Animals | 7 |
| Sessions | 207 |
| Trials per session | 39-40 |
| Total trials | 8,187 |
| Neurons per session | 113-564 (mean 337) |
| Unique neurons | 5,413 |
| Time bin | 33.33 ms (30 Hz) |
| Validation accuracy | 54.9% (chance: 11.1%) |

## Files
- `converted_data.pkl` - Full dataset
- `sample_data.pkl` - 2-animal subset for quick testing
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training/validation script
- `CONVERSION_NOTES.md` - Detailed conversion documentation

## Usage
```bash
# Convert data
python convert_data.py                    # Full dataset
python convert_data.py --sample           # Sample (2 animals)

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples
```

# Converted Track2p Dataset

This repository now contains a decoder-ready conversion of the Majnik et al. 2025 longitudinal barrel-cortex calcium imaging dataset.

## Dataset summary
- Source: Track2p developmental barrel cortex dataset from Majnik et al. 2025
- Subjects: 6 mice
- Sessions: 41 daily sessions
- Brain region: barrel cortex L2/3
- Trialization: fixed 60-second trials
- Time bin size: 333.333 ms (non-overlapping means of 10 native 30 Hz frames)
- Decoder input: `time_from_session_start_sec`
- Decoder output: `motion_energy_quintile` with 5 per-session equal-percentile classes

## Files
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: 2-session sample dataset
- `/app/convert_data.py`: conversion script
- `/app/CONVERSION_NOTES.md`: detailed processing log, decisions, and validation results
- `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, `/app/train_decoder_sample_out.txt`: sample-run logs
- `/app/conversion_full_out.txt`, `/app/verification_full_out.txt`, `/app/train_decoder_full_out.txt`: full-run logs

## Conversion choices
- Neural data are derived from the released suite2p traces using:
  - neuropil subtraction: `F - 0.7 * Fneu`
  - Suite2p baseline preprocessing (`maximin`, 60 s window, sigma 10 frames)
- Motion energy is aligned to the imaging timeline using the provided timestamps.
- Missing camera frames are linearly interpolated only where raw motion samples are absent.
- Motion is discretized into five equal-percentile bins separately for each session.

## Output structure
`converted_data.pkl` stores a Python dictionary with:
- `neural`: list of sessions, each a list of `(n_neurons, n_timepoints)` trial arrays
- `input`: list of sessions, each a list of `(1, n_timepoints)` trial arrays
- `output`: list of sessions, each a list of `(1, n_timepoints)` integer trial arrays
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## How to load
```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['input_names'])
print(data['output_names'])
print(len(data['neural']), 'sessions')
print(data['neural'][0][0].shape, 'first trial neural shape')
```

## How to regenerate
Full dataset:
```bash
python3 -u /app/convert_data.py /app/converted_data.pkl --full
```

Sample dataset:
```bash
python3 -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Validation
Format-only validation:
```bash
python3 -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Full decoder training:
```bash
python3 -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Key statistics
- Total 60-second trials: 1090
- Trial lengths: all 180 time bins
- Neurons per session: min 221, mean 498.66, max 746
- Output distribution: exactly 20% in each quintile by session construction

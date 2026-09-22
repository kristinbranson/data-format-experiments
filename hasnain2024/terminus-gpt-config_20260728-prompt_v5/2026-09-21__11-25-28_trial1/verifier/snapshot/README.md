# Converted Randomized-Delay ALM Dataset

This repository contains a converted neuroscience dataset for decoder training.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small sample conversion
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `train_decoder_full_out.txt`: full decoder training log
- `verification_full_out.txt`: full format verification log

## Dataset summary
- Task: randomized-delay licking task with DR/WC context
- Alignment: go cue onset
- Neural data: ALM spike-rate traces binned at 20 ms from -2.4 s to 2.0 s
- Inputs: time from go cue
- Outputs:
  - lick direction
  - behavioral context
  - outcome
  - tongue velocity (discretized)
  - paw velocity (discretized)
  - motion energy (discretized)

## Loading
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Format
The pickle stores a dictionary with keys:
`neural`, `input`, `output`, `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, `metadata`.

## Notes
See `CONVERSION_NOTES.md` for curation decisions, consistency checks, and decoder results.

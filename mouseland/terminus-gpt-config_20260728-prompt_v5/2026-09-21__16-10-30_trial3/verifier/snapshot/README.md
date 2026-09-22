# Converted Neural Decoder Dataset

This repository contains a converted dataset saved as `/app/converted_data.pkl` for decoder training with `/app/train_decoder.py`.

## Dataset summary
- Source paper: *Unsupervised pretraining in biological neural networks*
- Final converted dataset: filtered to rewarded imaging sessions with lick events
- Sessions: 27
- Subjects: 5 (`TX108`, `TX109`, `TX60`, `TX61`, `VR2`)
- Outputs decoded:
  - visual stimulus category
  - licking
  - position in 4 corridor bins
  - running speed in 4 bins

## File format
The pickle stores a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each `(4, n_timepoints)`
- `output`: list of sessions/trials, each `(4, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Notes
- Neural activity is loaded from processed deconvolved traces (`spks`) and concatenated across source groups.
- Time was downsampled by a stride of 5 frames to reduce file size.
- Trials are aligned to trial start / corridor entry for decoder compatibility.
- Some sessions use heuristic trial-boundary inference and some behavior matches use `swap1`/`swap2` session variants.

## Usage
Verify format:
```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Train decoder:
```bash
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

# CA1 Geometry Decoder Dataset

This repository contains a converted version of the dataset from:

- Lee, Keinath, Cianfarano, and Brandon (2025), "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"

The converted dataset is saved as `converted_data.pkl` and is formatted for the provided `train_decoder.py` pipeline.

## Source Data

- Raw animal files are stored in `data/` as joblib files and original MATLAB `.mat` files.
- Neural data are rise-extracted calcium-event traces from hippocampal CA1.
- Behavior is frame-aligned x-y position recorded at 30 Hz.
- Environment geometry is represented by 3x3 partition layouts with blocked partitions.

## Conversion Summary

The conversion follows the reference paper/code and then adapts the continuous sessions to the requested trial-based format.

Main processing steps:

1. Load per-animal joblib files from `data/`.
2. Infer the 3x3 geometry orientation that best matches occupancy and blocked partitions.
3. For each session, compute movement-valid frames from smoothed speed.
4. Keep session-valid cells and apply the reference decoder's low-activity cell filter.
5. Split each continuous session into consecutive 1-minute chunks.
6. Within each chunk, keep movement-valid frames, smooth neural traces, and average-pool every 3 frames.
7. Convert environment geometry to a static 9-feature input vector.
8. Convert position to a time-varying 9-class spatial-bin output, snapping invalid blocked-bin samples to the nearest open bin.

## Output Format

`converted_data.pkl` is a Python dictionary with keys:

- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each trial shaped `(9,)`
- `output`: list of sessions, each a list of trials, each trial shaped `(1, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

Input semantics:

- `partition_0_open` ... `partition_8_open`
- row-major 3x3 order
- `1 = open`, `0 = blocked`

Output semantics:

- `position_bin`
- categorical values `0..8`
- same row-major 3x3 order as the geometry input

## Key Statistics

Final converted dataset:

- Sessions: 207
- Subjects: 7
- Trials: 8109
- Mean trials/session: 39.17
- Mean active neurons/session: 332.67
- Mean pooled samples/trial: 313.92
- File size: about 3.2 GB

Validation and decoder results:

- `python train_decoder.py converted_data.pkl --verify-only` reports no errors or warnings
- Full decoder training:
  - Training balanced accuracy: `0.7767`
  - Validation balanced accuracy: `0.6856`
  - Chance level: `0.1111`

## Re-running Conversion

Sample conversion:

```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
python -u train_decoder.py sample_data.pkl --verify-only
python -u train_decoder.py sample_data.pkl
```

Full conversion:

```bash
python -u convert_data.py converted_data.pkl --full
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Notes

- The detailed conversion log and validation notes are in `CONVERSION_NOTES.md`.
- Sample and full run logs are saved in the corresponding `*_out.txt` files.

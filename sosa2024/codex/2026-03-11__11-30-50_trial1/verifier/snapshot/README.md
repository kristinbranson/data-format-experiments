# Converted Decoder Dataset

This repository now includes a trial-aligned conversion of the NWB release for:

- `A flexible hippocampal population code for experience relative to reward`

The converted dataset is saved as:

- `converted_data.pkl`

## Source Dataset

- Modality: 2-photon calcium imaging + virtual-reality behavior
- Brain region: hippocampus, CA1
- Subjects: 11 mice
- Sessions: 152
- Converted trials: 12,147
- Curated neurons/ROIs: 138,678
- Temporal alignment: trial start
- Time bin size: ~64.48 ms (`15.5078125 Hz`)

## Decoder Inputs

Each trial stores 4 input variables as a `(4, T)` array:

1. `time_from_trial_start_s`
2. `environment`
3. `trial_number`
4. `previous_trial_outcome`

Per-trial constants are repeated across all `T` frames.

## Decoder Outputs

Each trial stores 6 categorical outputs as a `(6, T)` array:

1. `distance_to_reward_zone_bin`
2. `absolute_position_bin`
3. `speed_bin`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

`reward_zone_location` and `reward_outcome` are per-trial variables repeated across frames.

## Neural Data

Each trial stores neural activity as a `(n_neurons, T)` array using NWB exported deconvolved calcium activity:

- source: `processing/ophys/Deconvolved`
- ROI filter: `iscell[:, 0] == 1`
- trial span: `trial_start` to `teleport` onset
- teleport frames excluded

## File Format

The pickle contains a dictionary with keys:

- `neural`
- `input`
- `output`
- `subjects`
- `subject_idx`
- `brain_regions`
- `brain_region_idx`
- `input_names`
- `output_names`
- `output_values`
- `metadata`

This matches the format expected by:

- `python train_decoder.py converted_data.pkl`

## Basic Usage

Verify the data:

```bash
python train_decoder.py converted_data.pkl --verify-only
```

Train the decoder:

```bash
python train_decoder.py converted_data.pkl --plot-samples
```

Create the dataset again:

```bash
python -u convert_data.py converted_data.pkl --full
```

Create a small sample:

```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Notes

- Reward-zone identity is derived from the NWB session identifier and the 30-trial switch rule described in the paper/code.
- Lick-sensor failure trials are excluded so `lick` can be represented without missing labels.
- See `CONVERSION_NOTES.md` for detailed validation, sanity checks, and decoder results.

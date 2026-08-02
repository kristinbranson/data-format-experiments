# CA1 Geometry Dataset Conversion

This repository contains a conversion of the Lee et al. (2025) CA1 geometry-remapping dataset into the decoder format expected by `train_decoder.py`.

## Source
- Paper: *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping*
- Data source in this workspace:
  - `data/QLAK-CA1-*` joblib animal files
  - `data/QLAK-CA1-*.mat` MATLAB files
- Reference code used for processing decisions:
  - `code/georepca1/src/utils.py`
  - `code/georepca1/main.py`

## Converted dataset
- Output file: `converted_data.pkl`
- Sample file: `sample_data.pkl`
- Converter script: `convert_data.py`

## What was converted
- Original recordings are continuous 40-minute CA1 miniscope sessions at 30 Hz.
- Each original recording day becomes one session in the converted dataset.
- Each session is split into non-overlapping 1-minute trials:
  - 1800 frames per trial
  - 39 or 40 trials per session depending on raw frame count
- Neural data:
  - released binary rising-phase calcium-event traces
  - session-present cells only (`NaN` rows removed per day)
- Decoder input:
  - static 3x3 environment geometry from the raw `blocked` field
  - converted to a 9D binary vector
  - transposed to match the coordinate frame of `position` and the spatial maps
- Decoder output:
  - time-varying 3x3 position bin
  - stored as a single 9-class categorical variable

## Key statistics
- Subjects: 7
- Sessions: 207
- Derived 1-minute trials: 8187
- Registered cells in source files: 5413
- Session-present neuron instances in converted data: 69744
- Mean neurons/session: 336.93
- Validation balanced accuracy on the full converted dataset:
  - `position_bin_3x3`: `0.5489`
  - uniform chance: `0.1111`

## How to regenerate
Full conversion:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Sample conversion:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

Format verification:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Full decoder training:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Output format
The pickle stores a dictionary with the required keys:
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

Important conventions:
- `neural[session][trial]` has shape `(n_neurons, 1800)`
- `input[session][trial]` has shape `(9,)`
- `output[session][trial]` has shape `(1, 1800)`
- `brain_regions == ['CA1']`
- `output_values[0] == ['x0_y0', ..., 'x2_y2']`

## Notes
- Cross-day cell identities are not preserved in the converted format because the target schema is session-centric. The converted dataset therefore stores session-present neurons rather than global registered-cell identities.
- Detailed processing decisions, validations, and decoder results are documented in `CONVERSION_NOTES.md`.

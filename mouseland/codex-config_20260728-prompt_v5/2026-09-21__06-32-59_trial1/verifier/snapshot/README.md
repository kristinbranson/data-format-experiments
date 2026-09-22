# Neural Decoder Dataset Conversion

Converted dataset for the paper *Unsupervised pretraining in biological neural networks* into the decoder-compatible format required by `/app/train_decoder.py`.

## Dataset Summary

- Source data: two-photon mesoscope recordings and behavior from `/app/data`
- Paper cohort represented: 89 recordings from 19 mice
- Final converted file: `/app/converted_data.pkl`
- Sample file: `/app/sample_data.pkl`
- Temporal reference: trial start / corridor entry
- Included frames: `ft_trInd == trial`, `ft_CorrSpc == True`, `ft_move > 0`

## Final Converted Statistics

- Sessions: 89
- Subjects: 19
- Trials: 38,110
- Neurons: 4,691,034
- Mean neurons/session: 52,708.25
- Neuron range/session: 20,547 to 89,577
- Full file size: about 81 GiB

## Decoder Inputs

Order in `data["input_names"]`:

1. `time_to_sound_cue_s`
2. `training_day`
3. `time_since_trial_start_s`
4. `reward_available`

Notes:

- `time_to_sound_cue_s` and `time_since_trial_start_s` are derived from corridor position using the fixed virtual speed of 60 cm/s used in the experiment.
- `training_day` is days since the mouse's first imaging session, with a small within-day block offset.
- `reward_available` is the rewarded-corridor identity (`isRew`), not actual reward delivery.

## Decoder Outputs

Order in `data["output_names"]`:

1. `visual_stimulus`
2. `licking`
3. `position_bin`
4. `speed_bin`

Output value definitions:

- `visual_stimulus`: canonical categories pooled across sessions
- `licking`: `0=no_lick`, `1=lick`
- `position_bin`: four 1 m bins over the 4 m corridor
- `speed_bin`: global quartiles over included running frames

## Data Layout

The pickle contains a Python dictionary with keys:

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

Each `neural[session][trial]` array has shape `(n_neurons, n_timepoints)`.

## Load Example

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))                 # sessions
print(data["input_names"])
print(data["output_names"])
print(data["neural"][0][0].shape)          # first session, first trial
```

## Reproduction Commands

Create sample dataset:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/train_decoder.py /app/sample_data.pkl
```

Create full dataset and validate:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Important Notes

- The public behavior files contain duplicated analysis views of some recordings. The converter merges these into 89 unique raw imaging sessions rather than duplicating neural data.
- Framewise data are kept on the original imaging frame grid after running-frame selection instead of using the paper's 60-bin position interpolation, because the decoder task requires time-varying outputs.
- Detailed decisions, checks, and validation results are documented in `/app/CONVERSION_NOTES.md`.

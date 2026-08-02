# Converted Decoder Dataset

This repository now includes a converted version of the Zhong et al. "Unsupervised pretraining in biological neural networks" imaging dataset in the decoder-ready pickle file `converted_data.pkl`.

## Dataset Summary

- Source data: calcium-imaging recordings and virtual-reality behavior from the local `data/` directory.
- Reference materials used: `paper.pdf`, `methods.txt`, and the original analysis code in `code/`.
- Final converted dataset:
  - 19 subjects
  - 89 sessions
  - 35,893 retained trials
  - 304,548 exported neurons total
  - median native frame bin size: about 314.7 ms

## Conversion Choices

- Session unit: one unique recording base (`<mouse>_<date>_<block>`), matching the paper’s 89 recordings.
- Neural signal: stored `spks` traces loaded directly from the raw files; no dF/F recomputation.
- Trial/frame mask: only retained running frames inside the 0-4 m textured corridor (`ft_CorrSpc & (ft_move > 0)`), matching the reference analyses.
- Neuron export: paper-style top 5% positive and top 5% negative stimulus-selective neurons per visual area (`V1`, `mHV`, `lHV`, `aHV`).
- Decoder-specific curation: trials with retained duration `> 60 s` or retained-frame gaps `> 10 s` are removed to avoid pathological trial-start timing outliers.

## File Format

Load the converted dataset with:

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)
```

Top-level fields:

- `neural`: list of sessions, each a list of `(n_neurons, T)` trial matrices
- `input`: list of sessions, each a list of `(4, T)` trial matrices
- `output`: list of sessions, each a list of `(4, T)` trial matrices
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

Decoder inputs:

1. `time_to_sound_cue_s`
2. `training_day`
3. `time_since_trial_start_s`
4. `reward_available`

Decoder outputs:

1. `visual_stimulus`
2. `licking`
3. `position_bin`
4. `running_speed_bin`

## Validation

Format validation:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Full decoder training:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

Final full-dataset validation balanced accuracy:

- `visual_stimulus`: `0.3911`
- `licking`: `0.7981`
- `position_bin`: `0.3159`
- `running_speed_bin`: `0.2968`

## Rebuilding

Recreate the full converted dataset with:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Create the sample dataset with:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

Detailed conversion decisions, checks, and review results are recorded in `CONVERSION_NOTES.md`.

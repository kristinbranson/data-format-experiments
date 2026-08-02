# Mesoscale Activity Map - Neural Decoder Dataset

## Overview

Converted neural electrophysiology data from the Mesoscale Activity Map dataset (DANDI:000363) into a format suitable for training neural decoders. The original data contains brain-wide Neuropixels recordings during an auditory delayed-response task in mice.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 173 |
| Subjects | 28 mice |
| Total neurons | 69,453 (QC-passed) |
| Total trials | 89,546 |
| Brain regions | 14 coarse regions |
| Time bins | 80 (50ms bins, -2.5s to 1.5s relative to go cue) |

## Files

- `converted_data.pkl` — Main converted dataset (pickle format)
- `convert_data.py` — Conversion script
- `CONVERSION_NOTES.md` — Detailed conversion documentation
- `train_decoder.py` / `decoder.py` — Decoder training scripts

## Data Format

The pickle file contains a dictionary with the following structure:

```python
{
    'neural': list[list[np.array]]     # [session][trial] -> (n_neurons, 80) float32
    'input': list[list[np.array]]      # [session][trial] -> (2, 80) float32
    'output': list[list[np.array]]     # [session][trial] -> (4, 80) int64
    'subjects': list[str]              # Subject IDs
    'subject_idx': np.array            # (n_sessions,) int64
    'brain_regions': list[str]         # 14 region names
    'brain_region_idx': list[np.array] # [session] -> (n_neurons,) int, region index per neuron
    'input_names': ['time_from_tone_onset', 'photostim_on']
    'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position']
    'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['below_40th','40th_to_60th','above_60th']]
    'metadata': dict                   # Dataset metadata
}
```

## Inputs

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time_from_tone_onset | continuous | Time (s) since sample tone onset |
| 1 | photostim_on | binary | 1 during optogenetic photostimulation, 0 otherwise |

## Outputs

| Index | Name | Classes | Description |
|-------|------|---------|-------------|
| 0 | choice | left(0), right(1) | Trial instruction (lick direction) |
| 1 | outcome | ignore(0), miss(1), hit(2) | Trial outcome |
| 2 | early_lick | no(0), yes(1) | Whether mouse licked during delay |
| 3 | tongue_y_position | below_40th(0), 40th-60th(1), above_60th(2) | Discretized tongue y-position (per-session percentiles) |

## Brain Regions

ALM, Cerebellum, CorticalSubplate, Hippocampus, Hypothalamus, Medulla, Midbrain, Olfactory, Orbital, OtherCortex, Pallidum, Pons, Striatum, Thalamus

## Decoder Performance

Trained with `train_decoder.py` using balanced loss, 100 PCs, 200 epochs:

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| choice | 0.7097 | 0.5000 |
| outcome | 0.6645 | 0.3333 |
| early_lick | 0.7515 | 0.5000 |
| tongue_y_position | 0.7427 | 0.3333 |

## Usage

```bash
# Convert from NWB (requires data/ directory with NWB files)
python3 convert_data.py converted_data.pkl --full

# Train decoder
python3 train_decoder.py converted_data.pkl --cpu --stats-json decoder_stats.json
```

## Source

- Dataset: https://dandiarchive.org/dandiset/000363
- Data paper: Chen, Liu et al. (2024) "Brain-wide neural activity underlying memory-guided movement"
- Method paper: Economo, Viber et al. (2022)

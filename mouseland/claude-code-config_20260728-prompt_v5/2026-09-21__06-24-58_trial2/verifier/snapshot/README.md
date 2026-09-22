# Zhong et al. 2025 - Dataset Conversion

Converts calcium imaging + VR corridor behavioral data from "Unsupervised pretraining in biological neural networks" (Zhong et al. 2025) into a decoder-compatible Python dictionary format.

## Dataset Overview

- **89 recordings** across **19 mice** (GCaMP6s, two-photon calcium imaging)
- **20,547-89,577 neurons per session** (Suite2p deconvolved fluorescence)
- **Virtual reality corridor**: 4m texture corridor + 2m grey space, constant 60 cm/s when running
- **4 texture categories**: circle, leaf, rock, wood
- **Frame rate**: 3.178 Hz (~315 ms per frame)

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full converted dataset (89 sessions, 38,110 trials, ~152 GB) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |
| `decoder.py` | Decoder module (provided) |
| `train_decoder.py` | Decoder training script (provided) |
| `cache/` | Cached copies of converted data |

## Usage

### Convert data
```bash
# Full dataset (89 sessions)
python -u convert_data.py converted_data.pkl --full

# Sample (2 sessions)
python -u convert_data.py sample_data.pkl --sample --show-processing
```

### Train decoder
```bash
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Data Format

The output pickle contains a dictionary with:

- **`neural`**: List of sessions, each a list of trials. Each trial is `(n_neurons, T)` float32 array of deconvolved fluorescence.
- **`input`**: List of sessions/trials. Each trial is `(4, T)` float32:
  - `time_to_sound_cue`: seconds, positive before cue, negative after
  - `day_of_training`: days since first session for each mouse
  - `time_since_trial_start`: seconds from corridor entry
  - `reward_availability`: 0 or 1
- **`output`**: List of sessions/trials. Each trial is `(4, T)` int64:
  - `stimulus_category`: 0=circle, 1=leaf, 2=rock, 3=wood
  - `licking`: 0=no lick, 1=lick
  - `position`: 0-3 (4 bins of 1m each in 4m corridor)
  - `running_speed`: 0-3 (quartile bins)
- **`subjects`**: List of mouse names
- **`subject_idx`**: Session-to-subject mapping
- **`brain_regions`**: ['V1', 'mHV', 'lHV', 'aHV']
- **`brain_region_idx`**: Per-session neuron-to-region mapping
- **`metadata`**: Task description, frame rate, speed quartile edges, etc.

## Processing Pipeline

1. Load neural data (concatenate imaging planes)
2. Filter neurons by brain region (exclude iarea=-1 and iarea=7)
3. For each trial, select frames where mouse is running (ft_move > 0) and in texture corridor (ft_CorrSpc)
4. Extract neural activity, compute input/output variables
5. Position and running speed discretized into 4 bins each

## Decoder Results (Full Dataset)

| Output | Train Bal. Acc | Val Bal. Acc | Chance |
|--------|---------------|-------------|--------|
| stimulus_category | 0.634 | 0.629 | 0.250 |
| licking | 0.936 | 0.828 | 0.500 |
| position | 0.317 | 0.316 | 0.250 |
| running_speed | 0.384 | 0.387 | 0.250 |

All outputs decode above chance, confirming meaningful neural encoding.

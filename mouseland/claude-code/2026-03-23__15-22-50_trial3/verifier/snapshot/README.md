# Zhong et al. 2025 — Decoder-Compatible Data Conversion

## Overview
Converts 2-photon calcium imaging data from "Unsupervised pretraining in biological neural networks" (Zhong et al., 2025) into a pickle format for neural decoder training.

## Dataset
- **Source**: Mouse visual cortex recordings during VR corridor navigation
- **89 sessions** across **19 mice** (GCaMP6s, excitatory neurons)
- **20,547–89,577 neurons/session** (Suite2p deconvolved traces)
- **38,110 total trials**, ~315ms time bins (~3.17 Hz frame rate)
- **5 brain regions**: V1, mHV, lHV, aHV, other

## Output Format
`converted_data.pkl` — pickle protocol 5, ~202 GB

### Structure
```python
{
    'neural': [[array(n_neurons, T), ...], ...],  # float16, per session/trial
    'input': [[array(4, T), ...], ...],            # float32
    'output': [[array(4, T), ...], ...],           # int64
    'subjects': ['DR10', ...],                     # 19 mice
    'subject_idx': array,                          # session → subject
    'brain_regions': ['V1', 'mHV', 'lHV', 'aHV', 'other'],
    'brain_region_idx': [array, ...],              # neuron → region per session
    'input_names': ['time_to_sound_cue', 'day_of_training',
                    'time_since_trial_start', 'reward_availability'],
    'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
    'output_values': [stimulus_names, ['no_lick','lick'], pos_bins, speed_bins],
    'metadata': {...}
}
```

### Inputs (4 dimensions)
| Input | Type | Description |
|-------|------|-------------|
| time_to_sound_cue | continuous | Seconds to/from sound cue (positive=before, negative=after) |
| day_of_training | per-trial | Ordinal session index within mouse (0-indexed) |
| time_since_trial_start | continuous | Seconds since corridor entry |
| reward_availability | per-trial | 1 if rewarded corridor, 0 otherwise |

### Outputs (4 dimensions, categorical)
| Output | Classes | Description |
|--------|---------|-------------|
| visual_stimulus | 15 | circle1/2/3, leaf1/2/3, leaf1_swap1/2, rock1/2, wood1/2/5, wood1_swap1/2 |
| licking | 2 | Binary per-frame licking (only in supervised sessions) |
| position | 4 | 0-1m, 1-2m, 2-3m, 3m+ (last bin includes 2m gray space) |
| running_speed | 4 | Quartile bins (Q25=0, Q50=9.67, Q75=31.46 cm/s) |

## Files
| File | Description |
|------|-------------|
| `converted_data.pkl` | Full dataset (202 GB) |
| `sample_data.pkl` | 2-session sample (15 GB) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training/validation script |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |
| `conversion_full_out.txt` | Full conversion log |
| `verification_full_out.txt` | Format verification results |
| `train_decoder_full_out.txt` | Decoder training results |

## Usage
```bash
# Full conversion (requires ~250 GB RAM, ~33 min)
python3 -u convert_data.py converted_data.pkl --full

# Sample conversion (2 sessions)
python3 -u convert_data.py sample_data.pkl --sample

# Verify format
python3 train_decoder.py converted_data.pkl --verify-only

# Train decoder (requires --cpu, subsamples neurons to 3000/session)
python3 train_decoder.py converted_data.pkl --plot-samples --cpu
```

## Decoder Results (Full Dataset)
| Output | Val Balanced Acc | Chance | Above Chance |
|--------|-----------------|--------|--------------|
| visual_stimulus | 0.188 | 0.067 | 2.8x |
| licking | 0.636 | 0.500 | 1.3x |
| position | 0.297 | 0.250 | 1.2x |
| running_speed | 0.298 | 0.250 | 1.2x |

All outputs above chance, confirming valid data conversion.

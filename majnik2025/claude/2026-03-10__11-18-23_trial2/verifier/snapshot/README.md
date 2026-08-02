# Track2p Neural Decoder Dataset

Converted dataset from Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p".

## Task
Decode **motion energy** (discretized into 5 equal-percentile bins) from **barrel cortex neural activity** during spontaneous behavior in developing mice (P7-P14).

## Dataset Summary
| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice (jm031-jm046) |
| Sessions | 41 total (6-7 per mouse) |
| Trials | 545 total (2-min blocks) |
| Neurons/mouse | 221-746 (mean 500) |
| Time bin size | 333.33 ms (10 frames @ 30 Hz) |
| Time bins/trial | 360 |
| Brain region | Barrel cortex |
| Output classes | 5 (motion energy bins) |

## Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data: (n_neurons, n_timepoints) per trial
neural_trial = data['neural'][0][0]  # session 0, trial 0

# Access input (time elapsed in seconds)
time_input = data['input'][0][0]  # shape (1, n_timepoints)

# Access output (motion energy bin 0-4)
output = data['output'][0][0]  # shape (1, n_timepoints)

# Metadata
print(data['subjects'])       # ['jm031', 'jm032', ...]
print(data['brain_regions'])  # ['barrel cortex']
print(data['input_names'])    # ['time_elapsed_s']
print(data['output_names'])   # ['motion_energy_bin']
```

## Processing Pipeline
1. Load raw fluorescence (F.npy) and neuropil (Fneu.npy) from Suite2p output
2. Neuropil correction: F_corr = F - 0.7 * Fneu
3. Baseline correction using Suite2p maximin filter (win=60s, sig=10 frames)
4. dF/F = (F_corr - baseline) / baseline
5. Bin neural and motion energy by averaging 10 consecutive frames
6. Split sessions into 2-minute trials
7. Discretize motion energy into 5 equal-percentile bins (global)

## Reproduction
```bash
python3 -u convert_data.py converted_data.pkl --full
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Decoder Performance
| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| motion_energy_bin | 0.355 | 0.200 |

## Reference
Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel J-C, Cossart R (2025). Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p. eLife 14:RP107540.

# Neural Decoder Dataset: Hasnain, Birnbaum et al., Nature Neuroscience 2024

Converted electrophysiology data from "Separating cognitive and motor processes in the behaving mouse" into a decoder-compatible pickle format.

## Source
- **Paper**: Hasnain, Birnbaum et al., Nature Neuroscience 2024
- **Data**: Zenodo DOI 10.5281/zenodo.13941415
- **Species**: Mus musculus (house mouse)
- **Brain region**: Anterior lateral motor cortex (ALM)
- **Tasks**: Delayed-response (DR) and water-cued (WC) licking paradigms

## Dataset Summary
| Statistic | Value |
|-----------|-------|
| Sessions | 44 (25 DR + 19 randomized delay) |
| Subjects | 14 mice |
| Total neurons | 2,456 |
| Total trials | 13,823 |
| Time bins per trial | 500 (10ms bins, -2.5 to 2.5s from go cue) |

## Output Files
| File | Description |
|------|-------------|
| `converted_data.pkl` | Full dataset (44 sessions, 1.8 GB) |
| `convert_data.py` | Conversion script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `conversion_full_out.txt` | Conversion log |
| `verification_full_out.txt` | Verification log |
| `train_decoder_full_out.txt` | Decoder training results |
| `predictions.png` | Decoder prediction samples |
| `sample_trials.png` | Sample trial visualizations |
| `cache/` | Intermediate files (sample runs, processing plots) |

## Pickle Format
```python
data = {
    'neural':           list of 44 sessions, each a list of trials, each (n_neurons, 500) float32
    'input':            list of 44 sessions, each a list of trials, each (1, 500) float32
    'output':           list of 44 sessions, each a list of trials, each (6, 500) int64
    'input_names':      ['time_from_go_cue']
    'output_names':     ['lick_direction', 'behavioral_context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy']
    'output_values':    [['left','right'], ['WC','DR'], ['incorrect','correct'],
                         ['low','high'], ['low','high'], ['low','high']]
    'subjects':         list of 14 subject names
    'subject_idx':      (44,) int64, maps session to subject
    'brain_regions':    ['ALM']
    'brain_region_idx': list of 44, each (n_neurons,) int64
    'metadata':         dict with processing parameters
}
```

## Processing Pipeline
1. Load MATLAB data objects (HDF5 and v5 formats)
2. Quality filter neurons (exclude garbage, noisy, gabrga, real?)
3. Filter trials (exclude early lick and photostimulation)
4. Align spikes to go cue onset
5. Bin at 10ms, smooth with causal Gaussian kernel (window=15)
6. Convert to firing rate (spks/sec)
7. Remove neurons with mean FR < 1 Hz
8. Extract trial variables (lick direction, context, outcome)
9. Compute tongue and paw velocity from DLC trajectories
10. Load and align motion energy
11. Discretize continuous variables at 50th percentile per session

## Decoder Results (Validation Balanced Accuracy)
| Output | Accuracy | Chance |
|--------|----------|--------|
| lick_direction | 0.663 | 0.50 |
| behavioral_context | 0.853 | 0.50 |
| outcome | 0.706 | 0.50 |
| tongue_velocity | 0.816 | 0.50 |
| paw_velocity | 0.585 | 0.50 |
| motion_energy | 0.816 | 0.50 |

All outputs decode above chance, confirming meaningful neural representations.

## Usage
```bash
# Convert from raw data
python3 convert_data.py

# Verify and train decoder
python3 train_decoder.py converted_data.pkl --plot-samples
```

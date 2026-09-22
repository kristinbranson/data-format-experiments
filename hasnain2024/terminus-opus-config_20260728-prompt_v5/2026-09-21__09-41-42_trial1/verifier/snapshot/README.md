# Neural Decoder Dataset: Hasnain, Birnbaum et al. (2024)

## Dataset Description

This dataset contains neural recordings from the anterior lateral motor cortex (ALM) of mice performing a two-context behavioral paradigm, converted from the paper "Separating cognitive and motor processes in the behaving mouse" (Hasnain, Birnbaum et al., Nature Neuroscience 2024).

### Task Description
Mice perform two tasks that alternate block-wise within each session:
1. **Delayed Response (DR)**: An auditory tone indicates reward location. After a delay, a go cue instructs the mouse to lick left or right.
2. **Water-Cued (WC)**: No auditory cues. Water is presented randomly at one lickport.

Some sessions use a **Randomized Delay** variant where the delay duration is randomly selected from 6 possible values.

### Recording Details
- **Brain region**: ALM (anterior lateral motor cortex)
- **Recording method**: High-density silicon probes (Neuropixels)
- **Spike sorting**: JRCLUST and/or Kilosort 3 with manual curation

## Data Format

The converted data is stored in `converted_data.pkl` as a Python dictionary.

### Loading the Data
```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Data Structure
- **neural**: List of 43 sessions, each containing a list of trials. Each trial is a (n_neurons, 1000) float32 array of firing rates in spk/s.
- **input**: List of sessions/trials. Each trial has shape (1, 1000) containing time from go cue in seconds.
- **output**: List of sessions/trials. Each trial has shape (6, 1000) containing:
  - [0] lick_direction: 0=left, 1=right, 2=none (per-trial)
  - [1] context: 0=DR, 1=WC (per-trial)
  - [2] outcome: 0=incorrect, 1=correct, 2=ignore (per-trial)
  - [3] tongue_velocity: 0=low, 1=high, 2=not_visible (time-varying)
  - [4] paw_velocity: 0=low, 1=high, 2=not_visible (time-varying)
  - [5] motion_energy: 0=low, 1=high, 2=no_video (time-varying)

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 43 |
| Subjects | 14 |
| Total neurons | 2,398 |
| Total trials | 11,981 |
| Neurons/session | 17-141 (mean 55.8) |
| Time bins | 1000 (5ms bins, -2.5 to 2.5s from go cue) |
| Brain region | ALM |

### Processing Parameters
- **Temporal alignment**: Go cue onset
- **Time bin size**: 5 ms
- **Smoothing**: 15-sample causal Gaussian kernel
- **Firing rate threshold**: 1 Hz minimum
- **Cluster quality**: All except garbage, noisy, real?
- **Trial filtering**: Excluded early lick, ignore, and stimulation trials
- **Session inclusion**: >40 correct DR trials per direction

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - 2-session sample for testing
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion documentation

## Running the Decoder
```bash
python train_decoder.py converted_data.pkl
```

## Source
Hasnain, M., Birnbaum, M. et al. "Separating cognitive and motor processes in the behaving mouse." Nature Neuroscience (2024).

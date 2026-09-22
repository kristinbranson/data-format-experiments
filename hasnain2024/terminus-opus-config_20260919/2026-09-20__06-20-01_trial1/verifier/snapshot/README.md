# ALM two-context / delayed-response dataset, converted for neural decoding

Converted from **Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the behaving mouse", Nature Neuroscience 28:640-653 (2025)**
(data: Zenodo 10.5281/zenodo.13941415; code: `github.com/economolab`, provided in `/app/code`).

## Dataset description

Head-fixed mice performed a **delayed-response (DR)** directional licking task in which an auditory sample tone
(1.3 s) instructed a left or right lick, followed by a delay epoch and an auditory go cue. Interleaved blocks of a
**water-cued (WC, "autowater")** task presented a water drop at a random port with no auditory cues. A subset of
mice performed a randomized-delay version of the DR task. Spiking activity was recorded with silicon / Neuropixels
probes in anterior lateral motor cortex (ALM; a few second probes targeted tongue-jaw motor cortex, tjM1), together
with 400 Hz side- and bottom-view video (DeepLabCut tracking + motion energy).

**Converted dataset**

| Statistic | Value |
|---|---|
| Sessions | 44 (25 fixed-delay + 19 randomized-delay) |
| Subjects (mice) | 14 |
| Neurons | 2,457 (ALM 2,312, tjM1 145); 17-141 per session, mean 55.8 |
| Trials | 13,762 (193-474 per session) |
| Time bins | 166 per trial, 30 ms, spanning -2.5 to +2.5 s around the go cue |
| Alignment | go-cue onset (water-drop onset on WC trials) |
| Neural signal | firing rate (spikes/s), causal-Gaussian smoothed (15 bins) |

## Loading the data

```python
import pickle, numpy as np
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]      # session 0, trial 0 -> (n_neurons, 166) firing rates in spk/s
inp    = data['input'][0][0]       # (1, 166) time from go cue, seconds
out    = data['output'][0][0]      # (6, 166) integer class labels

data['metadata']['time_axis']      # (166,) bin centres, -2.485 .. 2.465 s
data['metadata']['session_info'][0]  # per-session id, animal, task, thresholds, ...
```

## Output format specification

```
data['neural'][session][trial]  float32 (n_neurons, 166)   firing rate, spikes/s
data['input'][session][trial]   float32 (1, 166)           time from go cue (s)
data['output'][session][trial]  int64   (6, 166)           categorical labels
```

| # | `output_names` | Values (`output_values`) |
|---|---|---|
| 0 | `lick_direction` | 0 left, 1 right, 2 none (side of the first lickport contact after the go cue) |
| 1 | `context` | 0 WC (`bp.autowater`), 1 DR |
| 2 | `outcome` | 0 incorrect (`bp.miss`), 1 correct (`bp.hit`), 2 ignore (`bp.no`) |
| 3 | `tongue_velocity` | 0 below session median, 1 at/above median, 2 tongue not visible |
| 4 | `paw_velocity` | 0 below session median, 1 at/above median, 2 paws not visible |
| 5 | `motion_energy` | 0 below session median, 1 at/above median, 2 no video |

Outputs 0-2 are per-trial values broadcast over time; outputs 3-5 are genuinely time-varying.
Other keys: `subjects`, `subject_idx`, `brain_regions` (`['ALM','tjM1']`), `brain_region_idx`,
`input_names`, `output_names`, `output_values`, `metadata`.

## How the data were processed

Following the authors' MATLAB pipeline (`/app/code`):
1. Sessions and ALM probes taken from `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`.
2. Units: quality label not in {garbage, noisy, real?} (`findClusters.m`) and mean firing rate > 1 spk/s
   (`removeLowFRClusters.m`; the paper's criterion). Sessions need >= 10 units.
3. Trials: photoinactivation (`bp.stim.enable`) and early-lick (`bp.early`) trials removed, as in every reference
   condition string; trials with no ephys coverage removed. Ignore trials are **kept** (required output class).
4. Spikes aligned to `bp.ev.goCue` (`alignSpikes.m`), binned -2.5..2.5 s at 30 ms and converted to rates, then
   smoothed with the causal Gaussian kernel of `mySmooth.m` (N = 15, reflect boundary).
5. Video: DLC trajectories and motion energy interpolated onto the same time axis using
   `frameTimes - vidshift - goCue` with `vidshift` from `findVideoOffset.m`.
6. Tongue/paw speeds are the magnitude of the position derivative (`findVelocity.m`), computed only where
   DeepLabCut sees the feature; timepoints where it does not are the 'not visible' class.

## Validation

- `train_decoder.py --verify-only`: valid, no errors or warnings.
- Validation balanced accuracy (shared linear decoder, 44 sessions): lick_direction 0.678, context 0.865,
  outcome 0.648, tongue_velocity 0.609, paw_velocity 0.614, motion_energy 0.796 (chance 0.333 / 0.500).
- Replicating the paper's per-time-bin logistic-regression decoding on the converted data gives choice accuracy
  0.93 in the response epoch (peak 0.95) and context accuracy 0.80 during the ITI, matching Figs. 3b and 4b.

See `CONVERSION_NOTES.md` for the full decision log, consistency checks and known discrepancies,
and `convert_data.py` for the conversion code (`python -u convert_data.py out.pkl --full`).

# ALM two-context / delayed-response dataset, converted for neural decoding

Converted from **Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the
behaving mouse", Nature Neuroscience 28:640-653 (2025)** (data: Zenodo 10.5281/zenodo.13941415,
code: `/app/code`).

## Dataset description

Head-fixed mice performed two directional-licking tasks that alternated in blocks within a session:

* **DR (delayed response)**: an auditory tone (1.3 s) indicated the rewarded lickport, then a 0.9 s
  delay, then an auditory **go cue**; licking the correct port delivered ~3 ul of water.
* **WC (water cued)**: all auditory cues omitted; a water drop appeared at a random time at a random
  port and the animal consumed it.

A second cohort performed a **randomized-delay** version of the DR task (delays 0.3-3.6 s).
Extracellular recordings (Neuropixels 1.0 / Cambridge Neurotech H2) were made in anterior lateral
motor cortex (ALM); two high-speed cameras (400 Hz) plus DeepLabCut tracked the tongue, jaw, nose and
paws, and motion energy was computed per frame.

## Key statistics of the converted dataset

| | |
|---|---|
| Sessions | 44 (25 fixed-delay, 19 randomized-delay) |
| Mice | 14 |
| Trials | 13,762 (after removing early-lick, optogenetic-stim and unrecorded trials) |
| Units | 2,456 (2,311 ALM + 145 tjM1) |
| Units / session | 55.8 (17-141) |
| Time bins | 500 x 10 ms, from -2.5 s to +2.5 s around the go cue |
| Neural signal | smoothed firing rate (spikes/s), causal Gaussian, 15-bin window |
| Output distribution | lick: left 0.42 / right 0.45 / none 0.13; context: WC 0.10 / DR 0.90; outcome: incorrect 0.12 / correct 0.75 / ignore 0.13 |

## How to load and use

```python
import pickle, numpy as np
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]      # session 0, trial 0: (n_neurons, 500) firing rates in spikes/s
time   = data['input'][0][0][0]    # (500,) time from go cue, in seconds (-2.495 ... 2.495)
out    = data['output'][0][0]      # (6, 500) integer class labels
print(data['output_names'])        # names of the 6 outputs
print(data['output_values'][0])    # ['left', 'right', 'none']
print(data['metadata']['time_axis'])
```

To re-create the file:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~20 s with 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl
```

## Output format specification

```
data = {
  'neural':   [session][trial] -> float32 (n_neurons, 500)   firing rate, spikes/s
  'input':    [session][trial] -> float32 (1, 500)           time from go cue (s)
  'output':   [session][trial] -> int64   (6, 500)           class labels (see below)
  'subjects': list of 14 mouse ids
  'subject_idx': (44,) int index into 'subjects'
  'brain_regions': ['ALM', 'tjM1']
  'brain_region_idx': [session] -> (n_neurons,) int index into 'brain_regions'
  'input_names':  ['time_from_go_cue']
  'output_names': ['lick_direction','context','outcome','tongue_velocity','paw_velocity','motion_energy']
  'output_values': [['left','right','none'],
                    ['WC','DR'],
                    ['incorrect','correct','ignore'],
                    ['below_median','above_median','not_visible'],
                    ['below_median','above_median','not_visible'],
                    ['below_median','above_median','no_video']]
  'metadata': {task_description, time_bin_size (10.0 ms), temporal_alignment_event,
               off_start (-2.5), off_end (2.5), time_axis, smoothing, neural_units,
               neuron_curation, trial_curation, session_curation, session_info, ...}
}
```

Outputs 0-2 are per-trial values repeated over time; outputs 3-5 are genuinely time-varying.
The three movement outputs are discretised with a **per-session median** of the values at timepoints
where the feature is visible; class 2 means the feature was not visible (DLC NaN) or no video/motion
energy was available for that trial.

## Processing summary

1. Sessions and ALM probe(s) taken from the authors' meta scripts (`load<ANM>_ALMVideo.m`).
2. Units with quality garbage/noisy/real? removed; spikes aligned to `bp.ev.goCue`; binned at 10 ms
   from -2.5 to +2.5 s; divided by the bin width; smoothed with the authors' causal Gaussian kernel;
   units with mean rate <= 1 Hz removed; sessions with < 10 remaining units would be dropped (none were).
3. Early-lick trials, optogenetic-stimulation trials and trials outside the ephys recording removed.
4. DLC kinematics and 400 Hz motion energy interpolated onto the same time axis after correcting the
   video/ephys clock offset (`findVideoOffset.m`).

See `/app/CONVERSION_NOTES.md` for the full decision log and all validation results.

## Decoder performance (train_decoder.py, full dataset)

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| lick_direction | 0.333 | 0.636 |
| context | 0.500 | 0.854 |
| outcome | 0.333 | 0.625 |
| tongue_velocity | 0.333 | 0.567 |
| paw_velocity | 0.333 | 0.578 |
| motion_energy | 0.333 | 0.720 |

A per-time-bin logistic regression (the paper's own analysis) on these data gives 0.88-0.92 choice
decoding accuracy in the second after the go cue and ~0.5 early in the trial, matching Fig. 3b of the
paper, and 0.70-0.80 context decoding, matching Fig. 4b.

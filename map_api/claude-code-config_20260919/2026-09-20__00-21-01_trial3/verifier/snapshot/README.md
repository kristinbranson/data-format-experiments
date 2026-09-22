# Mesoscale Activity Map — decoder-ready conversion

`converted_data.pkl` is a decoder-ready reformatting of **DANDI:000363**, the *Mesoscale
Activity Map Dataset* (Chen, Nguyen, Li & Svoboda, 2023, v0.230822.0128), published with
*Brain-wide neural activity underlying memory-guided movement* (Chen et al.) and re-analysed in
*Brain-wide analysis reveals movement encoding structured across and within brain areas*
(Wang, Kurgyis et al., Nat Neurosci 2025).

## Dataset description

Head-fixed mice performed an **auditory delayed-response task**. During the *sample* epoch a
series of pure tones (3 kHz → lick right, 12 kHz → lick left) instructed the upcoming action;
a 1.2 s *delay* epoch followed in which licking had to be withheld; an auditory **go cue**
then opened a 1.5 s *response* epoch in which the mouse licked the left or right port, a
correct lick being rewarded with water. Licking during sample/delay ("early lick") replayed
the epoch. On ~20 % of randomly interleaved trials, anterolateral motor cortex (ALM) was
photoinhibited (473 nm, 40 Hz, 5 mW/hemisphere) for 0.5 s during the delay, always ending at
or before the go cue.

Two to five Neuropixels probes recorded extracellular activity brain-wide while a 300 Hz
side-view camera tracked the tongue, jaw and nose with DeepLabCut.

### Key statistics of the converted data
| | |
|---|---|
| Sessions | 142 |
| Mice | 28 (2–10 sessions each) |
| Trials | 73,845 (mean 520 per session, range 206–796) |
| Neurons (QC "good" units) | 57,023 (mean 402 per session, range 90–923) |
| Time bins per trial | 80 × 50 ms, from −2.5 s to +1.5 s around the go cue |
| Behavioural performance | 83.8 % correct (range 65.8–98.9 %) |
| Brain regions | 14 coarse Allen CCF groups (ALM, Orbital, OtherCortex, Olfactory, Hippocampus, CorticalSubplate, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum) |

## How to load and use

```python
import pickle
import numpy as np

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# firing rates (Hz) of every good unit in session 0, trial 12: (n_neurons, 80)
rates = data['neural'][0][12]

# the two decoder inputs for that trial: (2, 80)
time_since_tone, photostim_on = data['input'][0][12]

# the four decoded variables for that trial: (4, 80), integer class labels
choice, outcome, early_lick, tongue_y = data['output'][0][12]
print(data['output_values'][0][choice[0]])     # e.g. 'left'

# which mouse, which brain region each neuron is in
mouse = data['subjects'][data['subject_idx'][0]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][0]]

t = data['metadata']['bin_centers_s']          # bin centres relative to the go cue
```

Train and evaluate the reference decoder with:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl              # train + validate
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

## Output format specification

```python
data = {
  'neural':  [ [ (n_neurons, 80) float32, ... per trial ], ... per session ],  # firing rate, Hz
  'input':   [ [ (2, 80)        float32, ... ], ... ],
  'output':  [ [ (4, 80)        int8,    ... ], ... ],

  'subjects':     [str] * 28,                 # mouse names, e.g. 'SC015'
  'subject_idx':  int64 array, (142,),        # index into `subjects` per session

  'brain_regions':    [str] * 14,
  'brain_region_idx': [ int64 array (n_neurons,) per session ],

  'input_names':   ['time_from_tone_onset_s', 'photostim_on'],
  'output_names':  ['lick_direction_choice', 'outcome', 'early_lick', 'tongue_y_position'],
  'output_values': [['left', 'right', 'no lick'],
                    ['ignore', 'miss', 'hit'],
                    ['no', 'yes'],
                    ['y < 40th pct', 'y 40th-60th pct', 'y > 60th pct', 'not visible']],
  'metadata': {...},
}
```

### Inputs
| # | Name | Definition |
|---|------|------------|
| 0 | `time_from_tone_onset_s` | seconds elapsed since the onset of the instruction tone (the last sample-epoch onset at or before the go cue); negative before the tone. Continuous, per bin. |
| 1 | `photostim_on` | 1 if ALM photostimulation overlaps that 50 ms bin, else 0. |

### Outputs (all integer class labels, one per 50 ms bin)
| # | Name | Classes | Definition |
|---|------|---------|------------|
| 0 | `lick_direction_choice` | left / right / no lick | the port the mouse licked: the instructed side on `hit` trials, the opposite side on `miss` trials, `no lick` on `ignore` trials. Constant within a trial. |
| 1 | `outcome` | ignore / miss / hit | `trials['outcome']`. Constant within a trial. |
| 2 | `early_lick` | no / yes | `trials['early_lick']`. Constant within a trial. |
| 3 | `tongue_y_position` | <40th pct / 40–60th pct / >60th pct / not visible | per bin: `not visible` if no frame in the bin has DeepLabCut likelihood > 0.9; otherwise the mean vertical tongue position of the visible frames, discretised at that session's 40th and 60th percentiles. |

### Selected `metadata` fields
`task_description`, `time_bin_size` (50.0 ms), `temporal_alignment_event` ("go cue onset"),
`off_start` (−2.5 s), `off_end` (+1.5 s), `bin_centers_s`, `neural_units`,
`input_descriptions`, `output_descriptions`, `neuron_curation`, `trial_curation`,
`session_curation`, `known_limitation`, `tongue_likelihood_threshold`,
`neuron_annotation` / `neuron_hemisphere` / `neuron_ccf_coordinates` (per session, per neuron),
and `session_info` — one entry per session with its NWB identifier and file name, mouse,
trial and neuron counts, the indices of the trials and units it was built from, behavioural
performance and the session's tongue percentiles.

## Curation applied

- **Neurons**: only units labelled `good` by the dataset's per-region quality-control
  classifier (`units['classification']`), which is the unit set the papers report.
- **Trials**: dropped if they lie outside the electrophysiological recording (the units'
  `obs_intervals`), if they are auto-water or free-water trials, or if the video does not cover
  the analysis window. Early-lick, no-response and photostimulation trials are **kept** — they
  are exactly what the decoder is asked to predict (or to receive as input).
- **Sessions**: kept if they have at least one good unit and, on the curated trials, > 65 %
  correct control non-early-lick trials with at least 50 correct lick-left and 50 correct
  lick-right trials — the data paper's stated session-selection criteria.

### Known limitation
The NWB release stores spikes only inside each trial's `[start_time, stop_time]` interval.
Error (`miss`) trials are terminated a few hundred milliseconds after the go cue by the
time-out, so 15.6 % of trials end before +1.5 s and 3.1 % begin after −2.5 s. Those bins
contain no spikes and therefore have a firing rate of 0 (2.9 % of all bins on average, at most
8.5 % in any session). This is a property of the published data, not of the conversion.

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~26 s, 9.7 GB
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/sanity_checks.py /app/converted_data.pkl                # independent checks
```

| File | Purpose |
|---|---|
| `convert_data.py` | the conversion |
| `ccf_regions.py` | Allen CCF annotation → coarse brain-region group |
| `sanity_checks.py` | re-derives neural, input, output and metadata values straight from the NWB files and compares them with `np.allclose` |
| `CONVERSION_NOTES.md` | every decision, its provenance in the papers/reference code, and all validation results |
| `processing_<session>.png` | 8-panel diagnostic figure of every processing step |
| `accuracy_vs_time.png` | decoder validation accuracy per 50 ms bin |
| `cache/` | exploratory scripts and extracted paper text (see `cache/README_CACHE.md`) |

## Decoder performance

Validation balanced accuracy of the provided decoder on the full dataset
(`train_decoder_full_out.txt`):

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| lick_direction_choice | 0.333 | 0.682 |
| outcome | 0.333 | 0.656 |
| early_lick | 0.500 | 0.751 |
| tongue_y_position | 0.250 | 0.655 |

These are averages over the whole 4 s window. Resolved per bin (`accuracy_vs_time.png`) the
time course is the expected one: choice jumps at the go cue and peaks at 0.87, outcome peaks at
0.88 about 1 s after it, and early lick peaks at 0.82 during the sample/delay epoch when early
licks occur. An independent per-session logistic-regression control reaches a lick-direction
AUC of 0.998 ± 0.002 in the response epoch and 0.916 ± 0.029 in the delay epoch, matching the
0.99 ± 0.01 response-epoch AUC reported in the method paper.

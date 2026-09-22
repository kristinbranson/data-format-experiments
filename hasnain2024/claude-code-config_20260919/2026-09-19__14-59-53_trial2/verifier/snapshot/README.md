# ALM two-context / delayed-response dataset, decoder-ready

Converted from the data released with

> Hasnain, Birnbaum, Ugarte Nunez, Hartman, Chandrasekaran & Economo (2025),
> *Separating cognitive and motor processes in the behaving mouse*,
> **Nature Neuroscience** 28, 640–653. doi:10.1038/s41593-024-01859-1
> Data: Zenodo 10.5281/zenodo.13941415 · Code: `/app/code`

`converted_data.pkl` holds go-cue-aligned ALM spiking activity together with the task and
movement variables a decoder is asked to predict.

---

## Dataset description

Head-fixed mice performed two directional-licking tasks that alternate in blocks within a
session:

* **DR (delayed response)** — an auditory tone (1.3 s) indicates which lickport will be
  rewarded; after a delay (0.9 s fixed, or one of 0.3/0.6/1.2/1.8/2.4/3.6 s in the
  randomized-delay sessions) an auditory **go cue** instructs the animal to lick.
* **WC (water cued)** — all auditory cues are omitted and ~3 µl of water simply appears at a
  randomly chosen lickport at a random time; the animal drinks it.

Neural activity is extracellular spiking recorded in **anterior lateral motor cortex (ALM)**
with Neuropixels 1.0 or Cambridge Neurotech H2 probes. Two high-speed cameras (400 Hz, side
and bottom view) tracked the tongue, jaw, nose and paws with DeepLabCut, and whole-frame
motion energy was computed per video frame.

### Key statistics

| | |
|---|---|
| Sessions | **44** (25 fixed-delay + 19 randomized-delay; 12 of the fixed-delay sessions are two-context DR+WC) |
| Subjects | **14** mice |
| Trials | **13,762** (14,972 before curation) |
| Units | **2,457** ALM units (min 17, max 141, mean 55.8 per session) |
| Trial length | 500 bins × 10 ms = 5 s, from −2.5 s to +2.5 s about the go cue |
| Alignment | go cue onset (`bp.ev.goCue`; water-drop time on WC trials) |

| Output | class fractions |
|---|---|
| `lick_direction` | left 0.423 · right 0.446 · none 0.131 |
| `context` | WC 0.097 · DR 0.903 (WC is 0.315 within the 12 two-context sessions) |
| `outcome` | incorrect 0.120 · correct 0.749 · ignore 0.131 |
| `tongue_velocity` | below-median 0.046 · above-median 0.046 · not visible 0.909 |
| `paw_velocity` | 0.470 · 0.470 · not visible 0.059 |
| `motion_energy` | 0.480 · 0.481 · no video 0.039 |

---

## How to load and use

```python
import pickle
import numpy as np

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# one trial
X = data['neural'][0][0]     # (n_units, 500)  firing rate, spikes/s, float32
u = data['input'][0][0]      # (1, 500)        time from go cue, seconds, float32
y = data['output'][0][0]     # (6, 500)        categorical labels, int8

print(data['input_names'])   # ['time_from_go_cue']
print(data['output_names'])  # ['lick_direction', 'context', 'outcome',
                             #  'tongue_velocity', 'paw_velocity', 'motion_energy']
print(data['output_values'][0])  # ['left', 'right', 'none']

# which mouse / which session
sess = 0
print(data['subjects'][data['subject_idx'][sess]])
print(data['metadata']['session_info'][sess]['session'])   # e.g. 'JEB6_2021-04-18'
```

Train and score the reference decoder:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Regenerate the dataset from the raw `.mat` files (≈20 s on 16 cores):

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

---

## Output format specification

```
data = {
  'neural':  [session][trial] -> float32 (n_units[session], 500)   firing rate, spikes/s
  'input':   [session][trial] -> float32 (1, 500)                  time from go cue (s)
  'output':  [session][trial] -> int8    (6, 500)                  class index per timepoint

  'subjects':         list[str], 14 mouse ids
  'subject_idx':      int64 (44,)            index into 'subjects' for each session
  'brain_regions':    ['ALM']
  'brain_region_idx': [session] -> int64 (n_units[session],)   all zeros (all units are ALM)

  'input_names':   ['time_from_go_cue']
  'output_names':  ['lick_direction','context','outcome',
                    'tongue_velocity','paw_velocity','motion_energy']
  'output_values': [['left','right','none'],
                    ['WC','DR'],
                    ['incorrect','correct','ignore'],
                    ['below_median','above_median','not_visible'],
                    ['below_median','above_median','not_visible'],
                    ['below_median','above_median','no_video']]

  'metadata': { ... see below ... }
}
```

### Variable definitions

| Variable | Definition |
|---|---|
| `neural` | spike counts in 10 ms bins aligned to the go cue, divided by the bin width and smoothed with a **causal** Gaussian kernel (`gausswin(15)`, first 7 taps zeroed, `'reflect'` boundary) — i.e. exactly `obj.trialdat` from the authors' pipeline |
| `time_from_go_cue` | bin centre, −2.495 … +2.495 s (identical for every trial) |
| `lick_direction` | `left` if `(L&hit)|(R&miss)`, `right` if `(R&hit)|(L&miss)`, `none` on ignore trials (per-trial, constant over the trial) |
| `context` | `WC` if `bp.autowater`, else `DR` (per trial) |
| `outcome` | `correct` = `bp.hit`, `incorrect` = `bp.miss`, `ignore` = `bp.no` (per trial) |
| `tongue_velocity` | speed of the side-camera `tongue` marker, interpolated onto the 10 ms grid and split at that **session's** median over the timepoints where the tongue is tracked; `not_visible` where DeepLabCut does not label the tongue or the video does not cover the bin |
| `paw_velocity` | as above, for the bottom-camera paws — the mean speed over whichever of `top_paw` / `bottom_paw` is visible; `not_visible` only where neither is |
| `motion_energy` | 99th-percentile whole-frame motion energy, interpolated onto the 10 ms grid and split at that **session's** median; `no_video` where the video does not cover the bin |

### `metadata` keys

`task_description`, `time_bin_size` (10.0 ms), `temporal_alignment_event`,
`off_start` (−2.5), `off_end` (+2.5), `time_bin_centers_s`, `neural_units`, `smoothing`,
`neuron_curation`, `trial_curation`, `session_curation`, `video`, `discretization`,
`reference`, `sessions_dropped`, and `session_info` — a list with one dict per session
holding `anm`, `date`, `task` (`fixed`/`randomized`), `probes`, `two_context`,
`ntrials_raw`, `ntrials_kept`, `trial_index` (0-based indices into the raw session, so any
value can be traced back to the original `.mat`), `cluster_index` (probe, cluster) per unit,
`n_units_quality`, `n_units`, `n_trials_early`, `n_trials_stim`, `n_trials_no_ephys`,
`n_autowater_kept`, `vidshift`, and the three per-session discretisation thresholds.

---

## Curation applied

* **Sessions**: the 44 ALM recording sessions listed in the authors'
  `DataLoadingScripts/Recording and video/load*_ALMVideo.m`; sessions need ≥10 units
  (Methods). None was dropped.
* **Units**: only the ALM probe(s) designated in those scripts; clusters labelled
  `garbage`/`gabrga`/`noisy`/`real?` removed (`findClusters`); mean firing rate > 1 Hz
  over the analysis window (`removeLowFRClusters`, Methods).
* **Trials**: early-lick trials (976) and photoinactivation trials (187) removed, as in
  every `params.condition` of the reference code; 64 trials at the end of two sessions
  where the ephys recording had already stopped removed. Correct, error and ignore trials
  and both contexts are kept, because they are decoder targets.

## Decoder performance (reference harness, `train_decoder.py`)

| Output | chance | validation balanced accuracy |
|---|---|---|
| lick_direction | 0.333 | 0.643 |
| context | 0.500 | 0.845 |
| outcome | 0.333 | 0.619 |
| tongue_velocity | 0.333 | 0.576 |
| paw_velocity | 0.333 | 0.574 |
| motion_energy | 0.333 | 0.714 |

Running the paper's own analysis (per-session, per-time-bin logistic regression on balanced
correct trials) on this dataset reproduces its published curves: choice decoding rises from
chance before the sample tone to **0.894** just after the go cue (paper Fig. 3b ≈0.9) and
context decoding is **0.74** during the ITI, peaking at **0.914** (paper Fig. 4b). See
`CONVERSION_NOTES.md` Step 12 and `cache/paper_style_decoding.png`.

---

## Files

| File | |
|---|---|
| `convert_data.py` | conversion script |
| `converted_data.pkl` | full dataset (44 sessions, 1.6 GB) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | full record of decisions, checks and validation |
| `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` | run logs |
| `processing_<session>.png` | per-step diagnostic plots |
| `cache/` | analysis and verification scripts (see `cache/README_CACHE.md`) |

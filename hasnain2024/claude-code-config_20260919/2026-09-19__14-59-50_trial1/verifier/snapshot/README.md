# ALM two-context (DR / WC) decoding dataset

Decoder-ready conversion of the anterior lateral motor cortex (ALM) electrophysiology recorded
during the **two-context task-switching paradigm** of

> Hasnain, Birnbaum, Ulloa Severino, Zhang, Economo, *"Separating cognitive and motor processes in
> the behaving mouse"*, **Nature Neuroscience** 28:640–653 (2025).
> Data: Zenodo DOI [10.5281/zenodo.13941415](https://doi.org/10.5281/zenodo.13941415).

Produced by `convert_data.py` from `/app/data/Ephys_Behavior`. See `CONVERSION_NOTES.md` for the
full decision log and validation record.

---

## The experiment in one paragraph

Head-fixed mice alternate block-wise between two directional-licking tasks that demand the same
instructed movement but differ in their cognitive requirements.

* **DR (delayed response)** — a 1.3 s auditory sample tone indicates which lickport will be
  rewarded, a 0.9 s delay follows, and a brief auditory **go cue** releases the animal to lick.
* **WC (water cued)** — all auditory cues are omitted and ~3 µl of water simply appears at a
  randomly chosen port at a random time; the animal consumes it.

Blocks are 10–25 trials, with no cue signalling the switch, so the animal must hold an internal
representation of the current context. Every session starts with ~100 DR trials. Neural activity
was recorded in ALM with H2 or Neuropixels 1.0 probes; two 400 Hz cameras (side + bottom) were
tracked with DeepLabCut, and whole-frame motion energy was computed per video frame.

## What is in the converted dataset

| | |
|---|---|
| Sessions | 12 (the only ephys sessions containing both contexts) |
| Subjects | 7 mice (JEB6, JEB7 ×2, EKH1, EKH3, JGR2 ×2, JGR3, JEB19 ×4) |
| Brain region | ALM (left or right hemisphere) |
| Units | 521 (214 well-isolated single units) — paper reports 522 / 214 |
| Units per session | 27 – 67 (mean 43.4) |
| Trials | 3115 (210 – 390 per session, mean 260) |
| Alignment | go cue onset (`obj.bp.ev.goCue`) |
| Time window | −2.5 s … +2.5 s |
| Bin size | 10 ms → 500 time points per trial |
| Neural values | single-trial firing rate in spikes/s (causal-Gaussian smoothed) |

## Loading

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 17
X = data['neural'][session][trial]   # (n_units, 500)  float32, spikes/s
u = data['input'][session][trial]    # (1, 500)        float32, time from go cue (s)
y = data['output'][session][trial]   # (6, 500)        int64,   class labels

t = u[0]                                      # -2.495 ... 2.495 s
print(data['subjects'][data['subject_idx'][session]])            # 'JEB6'
print(data['metadata']['session_info'][session]['session_id'])   # 'JEB6_2021-04-18'
for i, name in enumerate(data['output_names']):
    print(name, '->', data['output_values'][i][y[i, 0]])
```

## Format

```
data = {
  'neural':  [session][trial] -> float32 (n_units, 500), firing rate in spikes/s
  'input':   [session][trial] -> float32 (1, 500)
  'output':  [session][trial] -> int64   (6, 500)
  'subjects': list[str]                      # 7 mouse IDs
  'subject_idx': int64 (12,)                 # index into 'subjects' per session
  'brain_regions': ['ALM']
  'brain_region_idx': [session] -> int64 (n_units,)   # all zeros (all units are ALM)
  'input_names':  ['time_from_go_cue']
  'output_names': ['lick_direction','context','outcome',
                   'tongue_velocity','paw_velocity','motion_energy']
  'output_values': [...]                     # class-name list per output
  'metadata': {...}
}
```

### Input

| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | `time_from_go_cue` | continuous, time-varying | bin centre in seconds relative to the go cue, −2.495 … 2.495 |

### Outputs

| # | Name | Values | Time-varying? | Definition | Fraction of samples |
|---|------|--------|---------------|------------|---------------------|
| 0 | `lick_direction` | `left`, `right`, `none` | per trial | `left = (L&hit)|(R&miss)`, `right = (R&hit)|(L&miss)`, `none = no` (`obj.bp`) | 0.398 / 0.377 / 0.225 |
| 1 | `context` | `WC`, `DR` | per trial | `obj.bp.autowater` (1 = WC) | 0.315 / 0.685 |
| 2 | `outcome` | `incorrect`, `correct`, `ignore` | per trial | `obj.bp.miss` / `hit` / `no` | 0.106 / 0.670 / 0.225 |
| 3 | `tongue_velocity` | `below_median`, `above_median`, `not_visible` | yes | speed of the side-camera `tongue` marker, split at the session's 50th percentile over visible samples; `not_visible` = DeepLabCut produced no tongue position | 0.041 / 0.041 / 0.919 |
| 4 | `paw_velocity` | `below_median`, `above_median`, `not_visible` | yes | mean speed of the bottom-camera `top_paw`/`bottom_paw` markers, same split; `not_visible` = neither paw detected | 0.492 / 0.492 / 0.015 |
| 5 | `motion_energy` | `below_median`, `above_median`, `no_video` | yes | whole-frame motion energy, same split; `no_video` = the trial has no video (all such trials were dropped, so this class is unpopulated) | 0.499 / 0.501 / 0.000 |

Per-trial variables are broadcast across the 500 time points so that all six outputs share one
`(6, T)` array.

### Metadata

`data['metadata']` records `task_description`, `time_bin_size` (10.0 ms),
`temporal_alignment_event`, `off_start` (−2.5), `off_end` (+2.5), `n_timepoints` (500),
`smoothing`, `neural_units`, `neuron_filtering`, `trial_filtering`, `session_selection`, `video`,
`source`, and `session_info` — a per-session dict with the animal, date, probe, cluster/trial
counts, video offset, the three discretisation thresholds, and `trial_ids` / `cluster_ids` /
`unit_qualities` that map every exported trial and unit back to the original `.mat` file.

## Processing (mirrors the authors' MATLAB pipeline)

1. **Session/probe selection** — the 12 two-context sessions and their ALM probe, taken from
   `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` (the same list loaded by
   `Scripts/Figure 8` and `Scripts/EDFigure 2a-left`).
2. **Neuron curation** — drop clusters of quality `garbage`/`gabrga`/`noisy`/`real?`
   (`findClusters.m`), then drop clusters whose condition-averaged mean rate is ≤ 1 Hz
   (`removeLowFRClusters.m`, `params.lowFR = 1`). 2287 → 529 → 521 units.
3. **Trial curation** — drop early-lick (`bp.early`) and photoinactivation (`bp.stim.enable`)
   trials, as every `params.condition` in the reference does, plus one trial with no video.
   Ignore trials are kept (the decoder task needs an `ignore` class).
4. **Alignment** — spikes: `trialtm − goCue(trial)` (`alignSpikes.m`); video and motion energy:
   `frameTimes − vidshift − goCue(trial)` with `vidshift` from `findVideoOffset.m`.
5. **Binning/smoothing** — 10 ms bins over [−2.5, 2.5], `/dt`, then the reference causal Gaussian
   (`mySmooth(x, 15, 'reflect')`), giving `obj.trialdat` in spikes/s (`getSeq.m`).
6. **Behavioural outputs** — DeepLabCut positions resampled onto the same 10 ms grid, velocity as
   the first derivative of position (`findPosition.m` / `findVelocity.m`), motion energy
   interpolated and nearest-filled (`loadMotionEnergy.m`), then discretised at each session's
   50th percentile.

## Reproducing

```bash
python -u convert_data.py converted_data.pkl --full                       # ~25 s
python -u convert_data.py sample_data.pkl --sample --show-processing      # 2 sessions + plots
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples
python -u cache/sanity_checks.py                                          # independent checks
python -u cache/paper_style_decoding.py                                   # reproduce paper Fig 3b/4b
```

## Decoder performance

`train_decoder.py` (one decoder shared across sessions, 100 PCs per session, per-time-point linear
readout, balanced loss, 80/20 trial split):

| Output | Chance | Train balanced acc | Validation balanced acc |
|---|---|---|---|
| lick_direction | 0.333 | 0.613 | **0.597** |
| context | 0.500 | 0.760 | **0.751** |
| outcome | 0.333 | 0.613 | **0.579** |
| tongue_velocity | 0.333 | 0.593 | **0.589** |
| paw_velocity | 0.333 | 0.542 | **0.507** |
| motion_energy | 0.333 (effectively 0.5) | 0.777 | **0.775** |

Because the decoder must predict at *every* time point, these whole-trial averages mix epochs in
which a variable is not yet determined with epochs in which it is. Resolved in time, choice
decoding is at chance 2 s before the go cue and reaches ≈0.72 in the response epoch, while context
decoding is high (≈0.78–0.84) throughout the trial, including the inter-trial interval — exactly
the pattern reported in the paper.

Re-running the **paper's own** analysis (per session, 75 ms bins, binary and class-balanced,
4-fold cross-validated logistic regression) on this converted data reproduces the published
curves: choice decoding peaks at **0.96** just after the go cue, context decoding at **0.97**,
with context already at 0.85–0.87 during the ITI.

## Files

| File | Contents |
|---|---|
| `converted_data.pkl` | full dataset (12 sessions, 361 MB) |
| `sample_data.pkl` | first 2 sessions, for quick tests |
| `convert_data.py` | the conversion script |
| `CONVERSION_NOTES.md` | decision log, consistency checks, validation record |
| `processing_<session>.png` | 12-panel per-session diagnostics of every processing step |
| `sample_trials.png`, `predictions.png` | decoder input/output visualisations |
| `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` | run logs |
| `cache/` | exploration and validation scripts (see `cache/README_CACHE.md`) |

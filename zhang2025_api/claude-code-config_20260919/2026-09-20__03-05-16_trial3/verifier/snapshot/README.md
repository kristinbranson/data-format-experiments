# IBL Brain-Wide Map → neural-decoder dataset

`converted_data.pkl` holds the International Brain Laboratory **Brain-Wide Map** dataset
(International Brain Laboratory et al., *A brain-wide map of neural activity during complex
behaviour*) reshaped into trial-aligned spike-count tensors with per-timestep task and
behaviour labels, ready for `train_decoder.py`.

## The experiment in one paragraph

Head-fixed mice turn a wheel to move a visual stimulus (a Gabor patch, contrast drawn from
{0, 6.25, 12.5, 25, 100}%) from the left or right of a screen to the centre. Each session
begins with 90 unbiased trials (stimulus equally likely on either side) and then alternates
between blocks of 20–100 trials in which the stimulus appears on the left with probability
0.2 or 0.8 — the *prior*, which mice learn and exploit. Neuropixels probes recorded
brain-wide while side-view cameras and a rotary encoder tracked whisking and wheel motion.

## What is in the file

| | |
|---|---|
| Sessions | 444 |
| Subjects (mice) | 136 |
| Trials | 188,922 (mean 425.5 per session, range 125–1445) |
| Neurons | 62,763 (mean 141.4 per session, range 1–516) |
| Brain regions (IBL Beryl) | 263 |
| Alignment | visual stimulus onset (`trials.stimOn_times`) |
| Window | −0.5 s to +1.5 s |
| Time bins | 100 × 20 ms |
| Size on disk | 11.7 GB |

## Loading

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# one trial
neural = data['neural'][session][trial]   # (n_neurons, 100) float32 spike counts
inputs = data['input'][session][trial]    # (2, 100) float32
outputs = data['output'][session][trial]  # (4, 100) int64 class labels

# where that neuron was recorded
region = data['brain_regions'][data['brain_region_idx'][session][0]]
# which mouse
subject = data['subjects'][data['subject_idx'][session]]
```

## Format

### `neural`
`neural[session][trial]` is `(n_neurons, 100)` **float32 spike counts** in 20 ms bins.
Bin *i* covers `[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`. Neurons are pooled
across all probes of a session and are the same set, in the same order, for every trial of
that session.

### `input` — what the decoder is given besides spikes
| idx | name | type | description |
|---|---|---|---|
| 0 | `time_from_stimulus_onset` | continuous, time-varying | bin-centre time in seconds, −0.49 … +1.49 |
| 1 | `trial_number_in_block` | continuous, per-trial | trials elapsed since the current `probabilityLeft` block began (0-based), broadcast across the 100 bins |

### `output` — what the decoder predicts
| idx | name | classes | description |
|---|---|---|---|
| 0 | `choice` | `left` = 0, `right` = 1 | the wheel turn the mouse made (IBL `choice` +1 → left, −1 → right); constant within a trial |
| 1 | `prior` | `p_left=0.2` = 0, `p_left=0.5` = 1, `p_left=0.8` = 2 | the block's prior probability that the stimulus is on the left; constant within a trial |
| 2 | `wheel_speed` | `low`/`medium`/`high` = 0/1/2 | \|wheel velocity\|, interpolated to the bin grid and split at the session's tertiles; time-varying |
| 3 | `whisker_motion_energy` | `low`/`medium`/`high` = 0/1/2 | whisker-pad motion energy (left camera, right as fallback), same treatment; time-varying |

All four are stored as `(4, 100)` arrays; the two per-trial variables are broadcast across
time so that every output is available at every timestep.

Global class fractions: choice `[0.508, 0.492]`, prior `[0.417, 0.141, 0.442]`,
wheel speed `[0.333, 0.333, 0.333]`, whisker `[0.334, 0.333, 0.333]`.

### `metadata`
`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`,
`off_start` (−0.5), `off_end` (+1.5), `n_time_bins`, descriptions of the neuron/trial
selection rules, the reference parameters used, `skipped_sessions`, and `session_info` —
one record per session with its `eid`, `subject`, probe count, neuron and trial counts,
which camera supplied the whisker trace, the tertile edges, and `trial_idx`, the row index
of every kept trial in that session's original trials table.

## How the data were selected

- **Sessions**: the 459 sessions of the BWM public release (`bwm_release.csv`). 15 are
  dropped because no whisker video covers their trials.
- **Neurons**: the data paper's *well-isolated neurons* — `clusters.label >= 1`, i.e. all
  three RIGOR single-unit metrics passed (amplitude > 50 µV, noise cutoff < 20 µV,
  refractory-period violation) — restricted to grey matter (Beryl acronym not `root`/`void`).
  The loader reproduces the paper's counts exactly: 621,733 clusters, 889.5 per probe,
  75,708 well-isolated, 108.3 per probe.
- **Trials**: `ibl_data_utils.load_trials_and_mask` with the reference's parameters — no
  NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`,
  `firstMovement_times` or `feedbackType`; reaction time in [0.08, 2.0] s; trial no longer
  than 10 s; no no-go trials. Trials whose wheel or whisker trace does not cover the 2 s
  window, or that fall outside the spike recording, are also dropped (89 trials in total).
  The 90-trial unbiased block is kept, because `probabilityLeft = 0.5` is one of the prior
  classes.

## Reproducing

```bash
python -u build_one_cache.py                        # rebuild the ONE cache tables (once)
python -u convert_data.py converted_data.pkl --full --n-workers 32   # ~70 s
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples          # ~11 min on a GPU
```

`--sample` converts 2 sessions; `--show-processing` writes per-session figures showing
raw versus binned spikes, raw versus interpolated behaviour, the discretisation
boundaries, and the trial×time alignment of all three streams.

## Decoder performance

Validation balanced accuracy of the reference decoder (`train_decoder.py`, 100 PCs,
80/20 per-session trial split):

| Output | Training | Validation | Chance |
|---|---|---|---|
| choice | 0.637 | **0.615** | 0.500 |
| prior | 0.680 | **0.662** | 0.333 |
| wheel_speed | 0.616 | **0.610** | 0.333 |
| whisker_motion_energy | 0.595 | **0.589** | 0.333 |

Choice is decoded per timestep, and a quarter of every trial precedes the stimulus, when
the upcoming choice is not yet represented in the brain. Decoding one 20 ms bin at a time
gives 0.53 (chance) before stimulus onset and 0.75 at +0.22 s — see
`cache/choice_timecourse.png`.

See `CONVERSION_NOTES.md` for the full record of decisions, validation and checks.

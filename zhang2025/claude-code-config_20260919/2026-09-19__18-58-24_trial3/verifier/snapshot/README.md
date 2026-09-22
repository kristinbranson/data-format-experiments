# IBL Brain-Wide Map → neural-decoder dataset

`converted_data.pkl` is the IBL Brain-Wide Map (BWM) Neuropixels release reformatted for
the decoder in `train_decoder.py`: stimulus-onset-aligned spike counts plus the task and
behavioural variables to decode from them.

## Dataset

Mice perform the IBL decision-making task: a Gabor patch of one of five contrasts
(100, 25, 12.5, 6.25, 0 %) appears 35° to the left or right and the mouse turns a wheel to
bring it to the centre for a water reward. The first 90 trials of a session are unbiased
(p(left) = 0.5); afterwards the stimulus side is drawn from uncued blocks of 20–100 trials
with p(left) = 0.2 or 0.8. Neuropixels probes recorded the left forebrain/midbrain and the
right hindbrain/cerebellum while a side camera tracked the whisker pad.

Source: IBL et al., *A brain-wide map of neural activity during complex behaviour*.
Processing follows Zhang et al., *Exploiting correlations across trials and behavioral
sessions to improve neural decoding* (`code/code_zhang2025/src/0_data_caching.py`).

## Key statistics

| | |
|---|---|
| Sessions | 442 (of 459 released) |
| Subjects (mice) | 136 (12 labs) |
| Probes | 699 released; 2 probes merged in 240 sessions |
| Neurons | 73,039 well-isolated (mean 165 / session, median 143, range 7–524) |
| Trials | 188,367 (mean 426 / session, median 392, range 125–1445) |
| Timepoints per trial | 100 |
| Time bin | 20 ms |
| Alignment | visual stimulus onset (`trials.stimOn_times`) |
| Window | −0.5 s → +1.5 s |
| Brain regions | 265 (Beryl atlas) |
| File size | 12.96 GB |

Class balance: choice left 0.508 / right 0.492; prior 0.417 / 0.140 / 0.442;
wheel speed and whisker motion energy 1/3 each by construction.

## Loading

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][s][k]   # (n_neurons[s], 100) float32 spike counts, trial k
inp    = data['input'][s][k]    # (2, 100)  float32
out    = data['output'][s][k]   # (4, 100)  int8 class labels

subject = data['subjects'][data['subject_idx'][s]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][s]]  # per neuron
```

Validate / train:

```
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl --plot-samples
```

Regenerate (≈3 min on 24 cores):

```
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + plots
```

## Format specification

| Key | Type | Description |
|---|---|---|
| `neural` | list[442] of list[n_trials] of `(n_neurons, 100)` float32 | spike counts per 20 ms bin, all probes of the session merged |
| `input` | list[442] of list[n_trials] of `(2, 100)` float32 | decoder inputs |
| `output` | list[442] of list[n_trials] of `(4, 100)` int8 | decoder targets, categorical |
| `subjects` | list[136] of str | mouse names |
| `subject_idx` | `(442,)` int64 | index into `subjects` per session |
| `brain_regions` | list[265] of str | Beryl acronyms |
| `brain_region_idx` | list[442] of `(n_neurons,)` int64 | index into `brain_regions` per neuron |
| `input_names` | list[2] of str | |
| `output_names` | list[4] of str | |
| `output_values` | list[4] of list[str] | class names per output |
| `metadata` | dict | see below |

### Inputs

| # | Name | Kind | Values |
|---|------|------|--------|
| 0 | `time_from_stim_on` | time-varying | bin-centre time in seconds, −0.49 … +1.49 |
| 1 | `trial_number_in_block` | per-trial (broadcast over time) | 0-based index of the trial within its block of constant `probabilityLeft`, counted over **all** trials of the session, 0 … 98 |

### Outputs

| # | Name | Kind | Classes |
|---|------|------|---------|
| 0 | `choice` | per-trial | 0 = `left`, 1 = `right` (IBL `choice` +1 → left, −1 → right) |
| 1 | `prior_prob_left` | per-trial | 0 = `p_left=0.2`, 1 = `p_left=0.5`, 2 = `p_left=0.8` |
| 2 | `wheel_speed` | time-varying | 0/1/2 = `low`/`medium`/`high`, split at the within-session tertiles of \|wheel velocity\| |
| 3 | `whisker_motion_energy` | time-varying | 0/1/2 = `low`/`medium`/`high`, split at the within-session tertiles of whisker-pad motion energy |

All four are stored as `(4, 100)` arrays; the per-trial ones are constant across the
100 bins.

### `metadata`

`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`,
`off_start` (−0.5), `off_end` (+1.5), `n_timepoints` (100), `neural_units`,
`neural_curation`, `trial_curation`, `brain_region_mapping`, `input_descriptions`,
`output_descriptions`, `source`, and `session_info` — one record per session with its
`eid`, `subject`, `lab`, probe/unit/neuron counts, the trial counts at each curation
stage, the per-criterion exclusion counts, which camera supplied the whisker signal, and
the two tertile edges actually used for each discretised output.

## Curation applied

- **Neurons**: only well-isolated units (`clusters.label == 1` — all three RIGOR
  single-unit metrics passed: amplitude > 50 µV, noise cut-off < 20 µV, refractory-period
  violation). Reproduces the data paper's 75,708 neurons exactly. No region filtering.
- **Trials**: no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`,
  `firstMovement_times`, `feedbackType`; reaction time within 0.08–2 s;
  `feedback_times − goCue_times ≤ 10 s`; `choice ≠ 0`; wheel and whisker traces covering the
  full window without NaN; at least one spike in the window.
- **Sessions**: dropped if whisker motion energy is missing or its camera timestamps do not
  cover the session (15), or if fewer than 5 well-isolated neurons remain (2).

`CONVERSION_NOTES.md` documents every decision, its justification, and the validation
results (including 52 independent spot-checks against the raw ALF files).

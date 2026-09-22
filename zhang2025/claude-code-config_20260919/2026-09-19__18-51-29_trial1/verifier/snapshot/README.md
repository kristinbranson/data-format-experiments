# IBL Brain-wide Map — decoder-ready dataset

`converted_data.pkl` holds the International Brain Laboratory **Brain-wide Map** public
release, reformatted for the neural decoder in `train_decoder.py`.

* Data paper: *A brain-wide map of neural activity during complex behaviour* (IBL et al.)
* Processing follows *Exploiting correlations across trials and behavioral sessions to
  improve neural decoding* (Zhang et al., **Neuron** 2026) — specifically its data-caching
  script `code/code_zhang2025/src/0_data_caching.py`.

Every conversion decision, the checks behind it and the validation results are written up
in [`CONVERSION_NOTES.md`](CONVERSION_NOTES.md).

---

## The experiment

Head-fixed mice turn a wheel to move a visual Gabor patch, which appears to the left or the
right of a screen, into the centre. After the first 90 trials (during which the stimulus is
equally likely on each side) the session runs in blocks of 20–100 trials in which the
stimulus appears on one side 80 % of the time; block switches are uncued, so the animal has
to track a **prior**. Neuropixels probes (1–2 per session) record while side cameras track
whisker-pad motion and a rotary encoder tracks the wheel.

## Key statistics

| | |
|---|---|
| Sessions | **440** (of the 459 in the release freeze) |
| Mice | **135** |
| Probe insertions | 670 |
| Neurons | **62,650** well-isolated grey-matter units (mean 142 / session, range 7–516) |
| Brain regions | **263** (Beryl acronyms) |
| Trials | **187,513** (mean 426 / session, range 125–1,445) |
| Time bins per trial | **100** × 20 ms, spanning −0.5 s → +1.5 s around stimulus onset |
| File size | 11.7 GB |

These reproduce the numbers quoted in the data paper: 891 clusters/probe (paper 889),
108.8 well-isolated units/probe (paper 108), raw trials per session mean 646.5 / median 602
/ range 401–1525 (paper 645 / 602 / 401–1525), and 81.6 % correct choices (paper 81.4 %).

## Loading

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# one trial
X = data['neural'][0][0]     # (n_neurons, 100)  spike counts per 20 ms bin, float32
U = data['input'][0][0]      # (2, 100)          float32
Y = data['output'][0][0]     # (4, 100)          int64 class labels

print(data['input_names'])   # ['time_from_stim_on', 'trial_num_in_block']
print(data['output_names'])  # ['choice', 'prior_prob_left', 'wheel_speed',
                             #  'whisker_motion_energy']
```

Validate / decode:

```bash
python train_decoder.py converted_data.pkl --verify-only     # format + summary
python train_decoder.py converted_data.pkl --plot-samples    # train and evaluate
```

Rebuild from the ONE cache:

```bash
python -u convert_data.py converted_data.pkl --full                   # ~4 min, 24 workers
python -u convert_data.py sample_data.pkl --sample --show-processing  # 2 sessions + figures
```

## Format

```
data = {
  'neural':  [session][trial] -> (n_neurons, 100) float32   spike counts / 20 ms bin
  'input':   [session][trial] -> (2, 100)         float32
  'output':  [session][trial] -> (4, 100)         int64
  'subjects':          list[str],  len 135
  'subject_idx':       (440,) int64  -> index into 'subjects'
  'brain_regions':     list[str],  len 263  (Beryl acronyms)
  'brain_region_idx':  [session] -> (n_neurons,) int64 -> index into 'brain_regions'
  'input_names', 'output_names', 'output_values',
  'metadata': {...}
}
```

### Inputs (decoder inputs)

| # | Name | Type | Values |
|---|------|------|--------|
| 0 | `time_from_stim_on` | time-varying, seconds | −0.50, −0.48, …, 1.48 (left edge of each bin) |
| 1 | `trial_num_in_block` | per-trial, broadcast over the 100 bins | 0–98; 0-based index of the trial inside its constant-`probabilityLeft` block |

### Outputs (what the decoder predicts) — all categorical

| # | Name | Classes | Meaning | Distribution |
|---|------|---------|---------|--------------|
| 0 | `choice` | `left`, `right` | the side the mouse reported, constant within a trial (`trials.choice == +1` → left → 0) | 0.508 / 0.492 |
| 1 | `prior_prob_left` | `p(left)=0.2`, `p(left)=0.5`, `p(left)=0.8` | block prior, constant within a trial | 0.418 / 0.140 / 0.442 |
| 2 | `wheel_speed` | `low`, `medium`, `high` | \|wheel velocity\|, time-varying, split at that session's 33rd/67th percentiles | 1/3 each |
| 3 | `whisker_motion_energy` | `low`, `medium`, `high` | whisker-pad motion energy, time-varying, split at that session's 33rd/67th percentiles | 1/3 each |

### `metadata`

`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`
(`trials.stimOn_times`), `off_start` (−0.5), `off_end` (+1.5), `n_time_bins` (100),
per-variable descriptions, the exact trial- and neuron-inclusion rules, the list of
`failed_sessions` with the reason each was skipped, and `session_info` — one record per
session with its eid, subject, lab, probe and cluster counts, trial counts at each stage of
curation, the indices of the retained trials in the raw trials table, the cluster UUIDs of
the retained neurons, the tertile thresholds used, and the mean firing rate.

## What was included, and what was not

**Neurons.** All probes of a session are merged. A unit is kept if
`clusters.label >= 1` — the data paper's *well-isolated neuron* (amplitude > 50 µV, noise
cut-off < 20 µV, no refractory-period violation) — and if its Beryl region is grey matter
(not `root`/`void`). Sessions need at least 5 such neurons.

**Trials.** The reference function `load_trials_and_mask(..., max_trial_len=10.0)` is used
unmodified: reaction time in [0.08, 2.0] s, `feedback_times − goCue_times ≤ 10 s`, no NaN in
`stimOn_times / choice / feedback_times / probabilityLeft / firstMovement_times /
feedbackType`, and `choice ≠ 0`. The unbiased (p = 0.5) block is kept, because it is one of
the three prior classes. A trial is dropped additionally if the wheel trace, the whisker
trace or the ephys recording of any probe does not cover its 2-s window. 65.9 % of raw
trials survive; 34.0 % are removed by the reference mask alone.

**Sessions.** 19 of the 459 are skipped and each reason is recorded in
`metadata['failed_sessions']`: 13 have no whisker-motion-energy dataset at all (only 445 of
459 sessions do), 3 have fewer than 5 well-isolated grey-matter neurons, 1 has no usable
trials, 1 has a broken whisker ROI whose motion energy is constant, and 1 has both.

## Decoder performance

`train_decoder.py` on the full dataset (70/30 trial split within every session, balanced
loss, 100 PCs, 200 epochs):

| Output | Chance | Train | **Validation** |
|---|---|---|---|
| choice | 0.500 | 0.6358 | **0.6160** |
| prior_prob_left | 0.333 | 0.6734 | **0.6560** |
| wheel_speed | 0.333 | 0.6125 | **0.6059** |
| whisker_motion_energy | 0.333 | 0.5969 | **0.5914** |

Balanced accuracy is computed at every timepoint. Choice and prior are constant within a
trial, so they are also scored over the 0.5 s *before* the stimulus, where choice is not
yet decodable — a per-timepoint analysis (`cache/time_resolved_decodability.png`) shows
choice sitting at 0.52 before stimulus onset and peaking at **0.786 at +0.20 s**, which is
the level the methods paper reports for per-trial choice decoding (AUC 0.72–0.79).

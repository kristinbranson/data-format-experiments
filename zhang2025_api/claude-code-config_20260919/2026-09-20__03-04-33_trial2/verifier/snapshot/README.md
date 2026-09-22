# IBL Brain-Wide Map → decoder-ready dataset

`converted_data.pkl` contains the International Brain Laboratory **Brain Wide Map** public
Neuropixels release, reformatted as trial-aligned neural / input / output arrays for
`train_decoder.py`.

* Source data: IBL et al. (2025), *A brain-wide map of neural activity during complex
  behaviour*, **Nature 645**, 177–191 — public ONE cache at `/app/data/one_cache`.
* Processing follows Zhang et al. (2026), *Exploiting correlations across trials and
  behavioral sessions to improve neural decoding*, **Neuron 114** — reference code in
  `/app/code/code_zhang2025`.
* Conversion script: `convert_data.py`. Decisions, validation and consistency checks:
  `CONVERSION_NOTES.md`.

---

## The experiment in one paragraph

Head-fixed mice turn a wheel to move a Gabor patch (contrast 0, 6.25, 12.5, 25 or 100%),
which appears 35° to the left or right, into the centre of a screen. The first 90 trials of
a session are *unbiased* (`p(left) = 0.5`); afterwards the stimulus side is drawn in blocks
of 20–100 trials with `p(left) = 0.8` or `0.2`, and the block is never cued, so the mouse
must infer it. Neuropixels probes (1–2 per session) record across the whole brain while
side-view cameras and a rotary encoder record whisking and wheel movement.

---

## How to load

```python
import pickle
import numpy as np

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 0
X = data['neural'][session][trial]   # (n_neurons, 100) float32 spike counts per 20 ms bin
U = data['input'][session][trial]    # (2, 100)  float32
Y = data['output'][session][trial]   # (4, 100)  int64 class labels

print(data['subjects'][data['subject_idx'][session]])                       # mouse name
print([data['brain_regions'][i] for i in data['brain_region_idx'][session]])  # region per neuron
print(data['metadata']['session_info'][session])                            # per-session details
```

Validate / train:

```bash
python train_decoder.py /app/converted_data.pkl --verify-only
python train_decoder.py /app/converted_data.pkl --plot-samples
```

Reproduce the conversion (≈3.5 min on 32 cores):

```bash
python -u convert_data.py /app/converted_data.pkl --full --n-workers 32
python -u convert_data.py /app/sample_data.pkl --sample --show-processing   # 2 sessions + plots
```

---

## Output format

Every trial is a 2 s window, **aligned to visual stimulus onset** (`trials.stimOn_times`),
spanning **−0.5 s to +1.5 s** in **100 non-overlapping 20 ms bins**.

| Key | Type | Meaning |
|---|---|---|
| `neural` | list[441] of list[n_trials] of `(n_neurons, 100)` float32 | spike counts per 20 ms bin, all well-isolated grey-matter neurons of the session pooled across probes |
| `input` | list[441] of list[n_trials] of `(2, 100)` float32 | decoder inputs, see below |
| `output` | list[441] of list[n_trials] of `(4, 100)` int64 | decoder targets, see below |
| `subjects` | list[136] of str | mouse names |
| `subject_idx` | `(441,)` int64 | index into `subjects` for each session |
| `brain_regions` | list[263] of str | Beryl (Allen CCF, summary-structure) acronyms |
| `brain_region_idx` | list[441] of `(n_neurons,)` int64 | region of each neuron |
| `input_names`, `output_names`, `output_values` | list[str] | names / class labels |
| `metadata` | dict | see below |

### Inputs (`input_names`)

| i | Name | Kind | Range | Definition |
|---|------|------|-------|------------|
| 0 | `time_from_stimulus_onset` | time-varying | −0.49 … +1.49 s | signed bin-centre time relative to stimulus onset |
| 1 | `trial_number_in_block` | per-trial (broadcast over time) | 0 … 98 | 0-based position of the trial inside its run of constant `probabilityLeft`, counted on the *complete* trials table |

### Outputs (`output_names`, `output_values`)

| i | Name | Kind | Classes | Definition |
|---|------|------|---------|------------|
| 0 | `choice` | per-trial | `left` (0), `right` (1) | `trials.choice == +1` → left, `== −1` → right (verified against correct high-contrast trials) |
| 1 | `prior_probability_left` | per-trial | `p(left)=0.2` (0), `p(left)=0.5` (1), `p(left)=0.8` (2) | `trials.probabilityLeft` |
| 2 | `wheel_speed` | time-varying | `low` (0), `medium` (1), `high` (2) | \|wheel velocity\| (from the ~1 kHz resampled wheel), resampled to bin right edges, split at the session's 33.3/66.7 percentiles |
| 3 | `whisker_motion_energy` | time-varying | `low` (0), `medium` (1), `high` (2) | whisker-pad motion energy from the left camera (60 Hz; right camera for 7 sessions), same resampling and per-session tertile split |

Tertile thresholds are computed **per session** because motion energy is in uncalibrated,
camera- and rig-dependent units and wheel vigour differs between mice; the exact
thresholds used for each session are stored in
`metadata['session_info'][s]['wheel_speed_tertiles' | 'whisker_me_tertiles']`.

### `metadata`

`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`,
`off_start` (−0.5), `off_end` (+1.5), `n_timepoints` (100), plus `neural_units`,
`input_units`, `output_discretization`, `behaviour_sampling`, `neuron_inclusion`,
`trial_inclusion`, `session_inclusion`, `reference_code`, dataset totals,
`failed_sessions`, and `session_info` — one record per session with the eid, subject, lab,
date, probe/cluster/neuron counts, trial counts at each curation stage, camera used,
tertile thresholds and fraction of correct trials.

---

## Curation

**Neurons** — clusters with ibllib quality `label >= 1`, i.e. the IBL *well-isolated
neuron* criterion (amplitude > 50 µV, noise cut-off < 20 µV, no refractory-period
violation), restricted to grey matter (Beryl acronym not `root`/`void`).

**Trials** — the reference `load_trials_and_mask(max_trial_len=10.0)`: no NaN in
`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`,
`feedbackType`; `0.08 s ≤ firstMovement_times − stimOn_times ≤ 2.0 s`; `choice ≠ 0`;
`feedback_times − goCue_times ≤ 10 s`. In addition a trial needs a wheel and a whisker
trace covering the full 2 s window without NaNs, and electrophysiology covering the full
window (no >0.5 s gap in the pooled spike train).

**Sessions** — the 459 eids of the BWM freeze file (`bwm_release.csv`) with ≥5
well-isolated grey-matter neurons, a usable whisker trace and ≥2 usable trials. 18 were
dropped: 14 have no motion-energy dataset on either camera, 3 have <5 neurons, 1 has no
usable trial.

---

## Key statistics

| Statistic | Value | Reference-paper value |
|---|---|---|
| Sessions | 441 | 459 released / 433 used by the method paper |
| Subjects (mice) | 136 | 139 |
| Probe insertions used | 673 | 699 |
| Brain regions (Beryl) | 263 | 270 (method paper) |
| Kilosort clusters per probe | 890.1 | 889 |
| Well-isolated neurons per probe | 108.5 | 108 |
| Neurons in the dataset (well-isolated, grey matter) | 62,757 | 62,857 ("canonical dataset") |
| Neurons per session | mean 142.3, median 123, range 7–516 | — |
| Raw trials per session | mean 646.3, median 601, range 401–1,525 | mean 645, median 602, range 401–1,525 |
| Trials retained | 187,901 (66.0% pass the trial mask; mean 426 per session) | — |
| Timepoints per trial | 100 (20 ms bins, −0.5…+1.5 s) | T = 100, 20 ms bins |
| Fraction of correct trials | 81.6% | 81.4 ± 0.4% |
| Mean firing rate | 11.4 spikes/s per neuron | — |
| File size | 11.7 GB | — |

Class balance: choice 50.8 / 49.2%; prior 41.7 / 14.0 / 44.2%; wheel speed and whisker
motion energy 33.3 / 33.3 / 33.3% by construction.

---

## Decoder performance (`train_decoder.py`, 200 epochs, 80/20 split per session)

| Output | Chance | Training balanced acc | Validation balanced acc |
|---|---|---|---|
| choice | 0.500 | 0.639 | **0.617** |
| prior_probability_left | 0.333 | 0.683 | **0.663** |
| wheel_speed | 0.333 | 0.617 | **0.609** |
| whisker_motion_energy | 0.333 | 0.600 | **0.593** |

These are averages over *every* timepoint of the window, including the 0.5 s before the
stimulus during which choice is not yet decodable. A per-timepoint analysis
(`CONVERSION_NOTES.md`, Step 12) shows choice decodability flat at 0.52 before stimulus
onset, rising sharply at t = 0 and peaking at 0.75 around +0.25 s.

---

## Files

| File | Contents |
|---|---|
| `converted_data.pkl` | the full converted dataset (441 sessions, 11.7 GB) |
| `sample_data.pkl` | 2-session sample used for quick validation |
| `convert_data.py` | the conversion script |
| `CONVERSION_NOTES.md` | decisions, consistency checks, validation log |
| `conversion_{sample,full}_out.txt` | conversion logs |
| `verification_{sample,full}_out.txt` | `train_decoder.py --verify-only` logs |
| `train_decoder_{sample,full}_out.txt` | decoder training logs |
| `processing_<eid>.png` | per-step processing/verification figures (2 sessions) |
| `sample_trials.png`, `predictions.png` | figures produced by `train_decoder.py` |
| `cache/` | investigation and sanity-check scripts (see `cache/README_CACHE.md`) |

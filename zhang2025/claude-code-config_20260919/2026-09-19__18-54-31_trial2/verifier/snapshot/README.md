# IBL Brain-Wide Map — decoder-ready conversion

`converted_data.pkl` contains the International Brain Laboratory **Brain-Wide Map** dataset
(Neuropixels recordings during the IBL decision-making task) reformatted for the neural decoder
in `train_decoder.py`.

## Dataset description

139 mice performed a visual decision-making task while 699 Neuropixels probes recorded across
279 brain areas (IBL et al., *A brain-wide map of neural activity during complex behaviour*,
Nature 645, 2025). On each trial a Gabor patch of one of five contrasts appears 35° to the left
or right and the mouse turns a wheel to bring it to the centre. The first 90 trials are
unbiased (p(stimulus left) = 0.5); afterwards the side follows 80:20 or 20:80 blocks of 20–100
trials, so the mouse maintains a *prior* about the stimulus side. Wheel movement and whisker-pad
motion energy are recorded continuously.

Processing follows the reference pipeline of Zhang et al. 2026 (Neuron)
(`code/code_zhang2025/src/0_data_caching.py`, `src/utils/ibl_data_utils.py`) and the inclusion
criteria of the data paper. See `CONVERSION_NOTES.md` for every decision and its justification.

| | |
|---|---|
| Sessions | **441** (of the 459 released; 18 excluded, see below) |
| Subjects (mice) | **136** |
| Probes merged | 672 |
| Neurons | **62,757** well-isolated grey-matter units (mean 142/session, range 7–516) |
| Brain regions (Beryl) | **263** |
| Trials | **187,934** (65.9 % of the 285,031 raw trials survive curation) |
| Alignment | visual stimulus onset (`trials.stimOn_times`) |
| Trial window | −0.5 s … +1.5 s |
| Time bins | 100 × 20 ms |
| File size | 11.7 GB |

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 0
X = data['neural'][session][trial]   # (n_neurons, 100) float32 spike counts per 20 ms bin
u = data['input'][session][trial]    # (2, 100) float32
y = data['output'][session][trial]   # (4, 100) int64 class labels

mouse  = data['subjects'][data['subject_idx'][session]]
region = [data['brain_regions'][i] for i in data['brain_region_idx'][session]]  # per neuron
```

Validate / train with:

```bash
python train_decoder.py converted_data.pkl --verify-only   # format + summary
python train_decoder.py converted_data.pkl --plot-samples  # train the decoder
```

Regenerate from the raw ONE cache with:

```bash
python -u convert_data.py converted_data.pkl --full                 # ~5 min, 24 workers
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural'  : [session][trial] -> (n_neurons, 100) float32  spike counts / 20 ms bin
  'input'   : [session][trial] -> (2, 100)        float32
  'output'  : [session][trial] -> (4, 100)        int64
  'subjects': list[str] (136),  'subject_idx': (441,) int64
  'brain_regions': list[str] (263),  'brain_region_idx': [session] -> (n_neurons,) int64
  'input_names', 'output_names', 'output_values', 'metadata'
}
```

### Inputs (`input_names`)
| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | `time_from_stim_onset` | continuous, time-varying | bin-centre time relative to stimulus onset, −0.49 … +1.49 s |
| 1 | `trial_number_in_block` | continuous, per trial (constant over the 100 bins) | 0-based position of the trial within its block of constant `probabilityLeft`, 0 … 98 |

### Outputs (`output_names` / `output_values`)
| # | Name | Classes | Description |
|---|------|---------|-------------|
| 0 | `choice` | `left` (0), `right` (1) | wheel-turn choice; `trials.choice == +1` → left, `== −1` → right. Constant within a trial. |
| 1 | `prior_prob_left` | `0.2` (0), `0.5` (1), `0.8` (2) | block prior `trials.probabilityLeft`. Constant within a trial. |
| 2 | `wheel_speed` | `low` / `medium` / `high` | \|wheel velocity\| interpolated to each bin's right edge, split at that session's 33.3 % / 66.7 % quantiles. |
| 3 | `whisker_motion_energy` | `low` / `medium` / `high` | whisker-pad motion energy (left camera, right camera as fallback), same interpolation and per-session terciles. |

`metadata` additionally holds `task_description`, `time_bin_size` (20.0 ms),
`temporal_alignment_event`, `off_start` (−0.5), `off_end` (+1.5), the neuron/trial curation
rules, the per-session `session_info` (eid, subject, lab, date, trial and neuron counts, camera
used, tercile edges) and `failed_sessions`.

## Curation summary

**Neurons** — well-isolated units only (`clusters.label >= 1`, i.e. passing all three RIGOR
single-unit metrics), restricted to grey matter (Beryl acronym not `root`/`void`), with at
least one spike in the session; sessions need ≥ 5 such neurons. Probes of a session are merged.

**Trials** — the reference `load_trials_and_mask`: no NaN in `stimOn_times`, `choice`,
`feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; reaction time
(`firstMovement_times − stimOn_times`) in 0.08–2.00 s; `feedback_times − goCue_times ≤ 10 s`;
`choice != 0`; and full wheel/whisker coverage of the −0.5…+1.5 s window.

**Sessions excluded (18)** — 14 with no whisker motion energy from either camera, 3 with fewer
than 5 well-isolated grey-matter neurons, 1 with no usable trial.

## Key statistics vs. the papers

| Statistic | Paper | This dataset |
|---|---|---|
| Units / well-isolated units in the release | 621,733 / 75,708 | 621,733 / 75,708 (verified from `clusters.metrics.pqt`) |
| Raw trials per session | mean 645, median 602, range 401–1,525 | mean 646.3, median 601, range 401–1,525 |
| Trials in the unbiased (p = 0.5) block | ~14 % (90 of ~645) | 14.0 % |
| Choice left/right | ≈ 50/50 | 50.8 / 49.2 |

## Decoder performance (validation balanced accuracy)

| Output | Chance | Accuracy |
|---|---|---|
| choice | 0.500 | 0.614 |
| prior_prob_left | 0.333 | 0.658 |
| wheel_speed | 0.333 | 0.606 |
| whisker_motion_energy | 0.333 | 0.588 |

Because `choice` is decoded at every time bin — including the 25 bins before the stimulus even
appears — its trial-average value understates the peak decodability: an independent per-time-bin
analysis gives ≈ 0.52 before stimulus onset and ≈ 0.78 at +0.22 s (see `CONVERSION_NOTES.md`,
Step 12).

## Files

| File | Contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` / `sample_data.pkl` | full / 2-session converted datasets |
| `CONVERSION_NOTES.md` | full record of decisions, checks and validation |
| `conversion_full_out.txt`, `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt`, `verification_sample_out.txt` | `--verify-only` output |
| `train_decoder_full_out.txt`, `train_decoder_sample_out.txt` | decoder training logs |
| `processing_<eid>.png` | per-step processing/alignment checks |
| `sample_trials.png`, `predictions.png` | decoder sample plots |
| `cache/` | analysis and verification scripts (see `cache/README_CACHE.md`) |

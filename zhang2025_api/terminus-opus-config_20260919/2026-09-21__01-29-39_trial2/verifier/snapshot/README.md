# IBL Brain-Wide Map -> neural-decoder dataset

This directory contains a conversion of the **International Brain Laboratory (IBL) Brain-Wide Map**
public data release into a decoder-ready format, together with the conversion code and its
validation.

## Dataset description

Mice perform the IBL decision-making task: a visual grating of one of five contrasts
(0, 6.25, 12.5, 25, 100%) appears on the left or right of a screen and the mouse turns a wheel to
bring it to the centre.  After 90 unbiased trials the probability that the stimulus appears on the
left is held constant within blocks of 20-100 trials at 0.2 or 0.8.  Neuropixels probes recorded
spiking activity across the brain while video cameras and a rotary encoder recorded behaviour.

Converted dataset:

| Statistic | Value |
|---|---|
| Sessions | 440 (of the 459 released) |
| Subjects (mice) | 136 |
| Trials | 187,547 |
| Neurons | 62,701 well-isolated units |
| Neurons / session | 142.5 mean (7-516) |
| Brain regions (Beryl) | 263 |
| Time bins per trial | 100 |
| Bin size | 20 ms |
| Alignment | stimulus onset (`trials.stimOn_times`), window -0.5 s to +1.5 s |

## How to load and use the converted data

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]     # session 0, trial 0: (n_neurons, 100) spike counts, float32
inp    = data['input'][0][0]      # (2, 100)
out    = data['output'][0][0]     # (4, 100), integer class labels
```

Train/validate the reference decoder:

```
python train_decoder.py converted_data.pkl              # train and report balanced accuracy
python train_decoder.py converted_data.pkl --verify-only  # format check and summary only
```

Re-create the dataset from the ONE cache:

```
python -u convert_data.py converted_data.pkl --full            # ~7 min with 12 workers
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural':  [session][trial] -> float32 (n_neurons, 100)   spike counts in 20 ms bins
  'input':   [session][trial] -> float32 (2, 100)
  'output':  [session][trial] -> int64   (4, 100)
  'subjects': list[str] (136), 'subject_idx': int array (440,)
  'brain_regions': list[str] (263), 'brain_region_idx': [int array (n_neurons,)] per session
  'input_names', 'output_names', 'output_values', 'metadata'
}
```

**Inputs** (`input_names`):

| idx | name | description |
|---|---|---|
| 0 | `time_from_stim_on` | time of the bin centre relative to stimulus onset, s (-0.49 ... 1.49) |
| 1 | `trial_number_in_block` | 0-based index of the trial within its `probabilityLeft` block |

**Outputs** (`output_names`, all time-varying with 100 bins):

| idx | name | classes |
|---|---|---|
| 0 | `choice` | 0 = left (`trials.choice == +1`), 1 = right (`trials.choice == -1`) |
| 1 | `prior_prob_left` | 0 = p(left) 0.2, 1 = 0.5, 2 = 0.8 |
| 2 | `wheel_speed` | 0 low / 1 medium / 2 high (per-session tertiles of \|wheel velocity\|) |
| 3 | `whisker_motion_energy` | 0 low / 1 medium / 2 high (per-session tertiles) |

`metadata` additionally holds `time_bin_size` (20.0 ms), `temporal_alignment_event`,
`off_start` (-0.5), `off_end` (1.5), the neuron/trial curation rules, and a `session_info` list with
per-session eid, subject, lab, trial and neuron counts, the camera used for whisker motion energy,
and the tertile thresholds.

## Key processing decisions

- Loading exclusively through the **ONE API** (`ONE(..., mode='remote')`, served from the staged
  cache) with `brainbox` `SessionLoader` / `SpikeSortingLoader`.
- Alignment, window and binning identical to the reference pipeline of Zhang et al.
  (`code/code_zhang2025/src/0_data_caching.py`): stimulus onset, (-0.5, 1.5) s, 20 ms bins.
- Probes of a session merged into one population (as in the BWM paper).
- Only **well-isolated** units (IBL unit QC `label == 1`) in grey matter (Beryl region not
  `void`/`root`); sessions need >= 5 such units.
- Trials filtered with the reference `load_trials_and_mask` criteria (no NaN in the key events,
  0.08 s <= reaction time <= 2 s, trial length <= 10 s, a choice was made), plus full coverage of
  the 2 s window by the wheel trace, the whisker trace and the spike-sorted recording.
- Wheel speed and whisker motion energy are interpolated onto the bin grid exactly as in the
  reference and then discretized into 3 per-session tertile classes.

## Validation summary

- `verification_full_out.txt`: "Data format is valid, no errors or warnings."
- Independent spot checks against the raw ALF data (`cache/sanity_checks.py`): 0 failures.
- Dataset statistics reproduce the papers exactly: 621,733 units, 75,708 well-isolated units,
  459 sessions, 699 insertions, 139 subjects, 645 trials/session.
- Decoder validation balanced accuracy: choice 0.614 (chance 0.5), prior 0.662, wheel speed 0.611,
  whisker motion energy 0.591 (chance 0.333).

Full details of every decision and check are in `CONVERSION_NOTES.md`.

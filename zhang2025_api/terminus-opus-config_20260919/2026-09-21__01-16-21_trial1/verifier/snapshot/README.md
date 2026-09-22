# IBL Brain-Wide Map -- decoder-ready dataset

`converted_data.pkl` is the IBL brain-wide map (BWM) public release, reformatted so that a
neural decoder can predict behavioural and task variables from population spiking activity.

## Dataset description

Mice perform the IBL decision-making task: a visual grating of one of five contrasts
(100, 25, 12.5, 6.25, 0 %) appears on the left or right of a screen and the mouse turns a
wheel to bring it to the centre. The first 90 trials of a session are unbiased
(P(left) = 0.5); afterwards the stimulus side is drawn in blocks of 20-100 trials with
P(left) = 0.2 or 0.8, and block changes are not cued. Neuropixels probes record
simultaneously from across the brain while wheel position and video-derived whisker-pad
motion energy are logged.

Each trial here is a **2 s window aligned to stimulus onset**, from **-0.5 s to +1.5 s**,
binned into **100 non-overlapping 20 ms bins**.

## Key statistics

| | |
|---|---|
| Sessions | 442 (of 459 released) |
| Subjects (mice) | 136 (of 139) |
| Brain regions (Allen/Beryl) | 263 |
| Neurons | 62,773 well-isolated, grey-matter |
| Trials | 188,044 |
| Neurons / session | mean 142, median 123, range 7-516 |
| Trials / session | mean 425, median 392, range 85-1445 |
| Time bins / trial | 100 (20 ms each) |
| Alignment | stimulus onset (`trials.stimOn_times`) |
| Window | -0.5 s to +1.5 s |
| File size | 11.7 GB |

Neurons are the data paper's **well-isolated** units (`clusters.label == 1`: median
amplitude > 50 uV, noise cut-off < 20 uV, and refractory-period violation all passed),
restricted to Allen grey matter. Probes recorded in the same session are merged into one
population.

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 0
X = data['neural'][session][trial]   # (n_neurons, 100)  spike counts per 20 ms bin
u = data['input'][session][trial]    # (2, 100)          decoder inputs
y = data['output'][session][trial]   # (4, 100)          decoder targets, integer classes

print(data['subjects'][data['subject_idx'][session]])                 # mouse name
print([data['brain_regions'][i] for i in data['brain_region_idx'][session]])  # region per neuron
```

Train and evaluate the reference decoder with:

```bash
python train_decoder.py converted_data.pkl              # train
python train_decoder.py converted_data.pkl --verify-only  # check format only
```

Regenerate the dataset from the ONE cache with:

```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + plots
```

## Output format specification

```
data = {
  'neural':   [session][trial] -> float32 (n_neurons, 100)   spike counts per 20 ms bin
  'input':    [session][trial] -> float32 (2, 100)
  'output':   [session][trial] -> int64   (4, 100)
  'subjects':          list[str], 136 mouse names
  'subject_idx':       int64 (442,)   index into 'subjects' per session
  'brain_regions':     list[str], 263 Beryl acronyms
  'brain_region_idx':  [session] -> int64 (n_neurons,)  index into 'brain_regions'
  'input_names', 'output_names', 'output_values', 'metadata'
}
```

### Inputs (`input_names`)
| # | Name | Type | Range | Meaning |
|---|------|------|-------|---------|
| 0 | `time_from_stim_on` | time-varying | -0.48 ... +1.50 s | time of the bin's right edge relative to stimulus onset |
| 1 | `trial_number_in_block` | per-trial (broadcast over the 100 bins) | 0.00 ... 0.98 | 0-based index of the trial within its block of constant P(left), **divided by 100** to keep it O(1) |

### Outputs (`output_names`, `output_values`)
| # | Name | Classes | Type |
|---|------|---------|------|
| 0 | `choice` | 0 = `left`, 1 = `right` | per-trial (broadcast) |
| 1 | `prior_prob_left` | 0 = `p_left_0.2`, 1 = `p_left_0.5`, 2 = `p_left_0.8` | per-trial (broadcast) |
| 2 | `wheel_speed` | 0 = `low`, 1 = `medium`, 2 = `high` | time-varying |
| 3 | `whisker_motion_energy` | 0 = `low`, 1 = `medium`, 2 = `high` | time-varying |

`choice` follows the required coding: `trials.choice == +1` is a leftward report -> 0.
`wheel_speed` (= |wheel velocity|) and `whisker_motion_energy` are discretised at the
**33.3rd and 66.7th percentiles of each session**, because their absolute scale varies by
an order of magnitude between sessions; the three classes are therefore equally populated
and mean the same thing in every session.

## Curation applied

- **Neurons**: `clusters.label >= 1` (well-isolated) and Beryl region not `root`/`void`.
- **Trials**: no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`,
  `firstMovement_times`, `feedbackType`; reaction time in [0.08, 2.0] s;
  `feedback_times - goCue_times <= 10 s`; `choice != 0`; the trial window must be covered
  by the wheel trace, the whisker trace and the spike sorting.
  The 90 unbiased trials are **kept** (they are class 1 of `prior_prob_left`).
- **Sessions**: dropped if they have no whisker motion-energy trace (14 sessions),
  fewer than 5 well-isolated grey-matter neurons (3), or fewer than 2 usable trials (0).

## Decoder performance (reference decoder, 442 sessions)

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| choice | 0.500 | 0.617 |
| prior_prob_left | 0.333 | 0.652 |
| wheel_speed | 0.333 | 0.618 |
| whisker_motion_energy | 0.333 | 0.602 |

Choice decodability is near chance before the stimulus (0.538) and peaks at **0.796 at
t = +0.24 s**, matching the oracle value reported in the methods paper.

## Provenance

- Data: IBL et al., *A brain-wide map of neural activity during complex behaviour*.
- Processing follows Zhang et al., *Exploiting correlations across trials and behavioral
  sessions to improve neural decoding* (`code/code_zhang2025`).
- All loading goes through the **ONE API** (`ONE()`, `brainbox.io.one.SessionLoader`,
  `brainbox.io.one.SpikeSortingLoader`, `iblatlas.regions.BrainRegions`).
- Every decision, check and validation result is documented in `CONVERSION_NOTES.md`.

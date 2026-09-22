# IBL Brain-Wide Map -> Neural Decoder Dataset

Converted from the International Brain Laboratory (IBL) **brain-wide map** electrophysiology
release into the decoder pickle format.

Sources:
- *A brain-wide map of neural activity during complex behaviour* (IBL et al.) - the data
- *Exploiting correlations across trials and behavioral sessions to improve neural decoding*
  (Zhang et al., Neuron 2026) - the processing pipeline that this conversion follows

## Dataset description

Head-fixed mice report on which side a visual grating appears by turning a wheel. After an
initial 90 unbiased trials, the stimulus side follows 20:80 / 80:20 blocks of 20-100 trials,
so the animal holds a slowly changing prior. Neuropixels probes record from across the brain.

Each trial is aligned to **stimulus onset** and spans **-0.5 s to +1.5 s**, binned at **20 ms**
into **100 timepoints**.

| Statistic | Value |
|---|---|
| Sessions | 441 |
| Subjects (mice) | 136 |
| Brain regions (Beryl, grey matter) | 263 |
| Trials | 187,934 (mean 426 per session) |
| Neurons | 62,757 (mean 142 per session) |
| Timepoints per trial | 100 (20 ms bins) |
| Mean firing rate | 10.2 Hz |

## Loading

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]   # session 0, trial 0 -> (n_neurons, 100) spike counts
inp    = data['input'][0][0]    # (3, 100)
out    = data['output'][0][0]   # (4, 100)
```

## Format

| Key | Type | Meaning |
|---|---|---|
| `neural` | list[session] of list[trial] of (n_neurons, 100) float32 | spike counts per 20 ms bin |
| `input` | list[session] of list[trial] of (3, 100) float32 | decoder inputs |
| `output` | list[session] of list[trial] of (4, 100) int16 | categorical decoder targets |
| `subjects` | list[str], len 136 | mouse names |
| `subject_idx` | (441,) int | index into `subjects` per session |
| `brain_regions` | list[str], len 263 | Beryl acronyms |
| `brain_region_idx` | list[session] of (n_neurons,) int | region of each neuron |
| `input_names` / `output_names` / `output_values` | list[str] | names and class labels |
| `metadata` | dict | see below |

### Inputs
| i | name | description |
|---|---|---|
| 0 | `time_from_stim_onset` | bin end time in seconds, -0.48 to 1.50 |
| 1 | `stim_onset` | binary, a single 1 in bin 25 (the bin containing t = 0) |
| 2 | `trial_in_block` | index of the trial within its probabilityLeft block |

### Outputs (all categorical, all time-varying)
| i | name | classes |
|---|---|---|
| 0 | `choice` | `left` (0), `right` (1) |
| 1 | `prior_prob_left` | `p_left_0.2` (0), `p_left_0.5` (1), `p_left_0.8` (2) |
| 2 | `wheel_speed` | `low`, `medium`, `high` (per-session tertiles) |
| 3 | `whisker_motion_energy` | `low`, `medium`, `high` (per-session tertiles) |

`choice` and `prior_prob_left` are constant within a trial; the other two vary over time.

### metadata
`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`, `off_start` (-0.5),
`off_end` (1.5), `n_timepoints`, `neural_units`, `alignment_note`, `neuron_curation`,
`trial_curation`, `discretisation`, `source`, `reference_code`, and `session_info` - a
per-session record with eid, subject, lab, probe and unit counts, trial counts, the camera
used for motion energy, and the tertile thresholds.

## Curation

**Neurons**: well-isolated units only (`clusters.label >= 1`, i.e. all three RIGOR single-unit
metrics passed - amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation),
restricted to grey matter (Beryl acronym not `root`/`void`). Probes are merged per session.
Sessions need at least 5 such neurons. *The `label >= 1` rule reproduces the paper's 75,708
well-isolated neurons exactly.*

**Trials**: the reference `load_trials_and_mask` criteria - no NaN in `stimOn_times`, `choice`,
`feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; reaction time in
[0.08, 2.0] s; `feedback_times - goCue_times <= 10 s`; `choice != 0` - plus complete wheel and
whisker-motion-energy coverage of the trial window. This retains 66 % of raw trials.

**Sessions**: 459 in the release freeze, minus 14 with no whisker motion energy, 3 with fewer
than 5 well-isolated grey-matter neurons, and 1 with no fully covered trial -> **441**.

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full        # ~3 min, 24 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl
```

`convert_data.py` rebuilds the ONE dataset index from disk on first use (see
`cache/build_one_cache.py`), which is required because the shipped release tables predate the
staged files and do not index their revision folders.

## Decoder performance

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| choice | 0.500 | 0.616 |
| prior_prob_left | 0.333 | 0.662 |
| wheel_speed | 0.333 | 0.610 |
| whisker_motion_energy | 0.333 | 0.598 |

Choice is scored at every timepoint, including the 25 bins before the stimulus appears where
it is necessarily at chance (measured: 0.525). Decoding choice from the whole window with the
reference's own protocol gives **0.833 +- 0.079**, matching the reference's published
0.848-0.877. See `CONVERSION_NOTES.md` Step 12.

See `CONVERSION_NOTES.md` for the full record of decisions, checks and validation.

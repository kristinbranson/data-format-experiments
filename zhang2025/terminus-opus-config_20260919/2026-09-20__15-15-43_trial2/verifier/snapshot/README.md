# IBL Brain Wide Map, converted for neural decoding

This directory contains the International Brain Laboratory (IBL) **Brain Wide Map** dataset
reformatted as trial-aligned arrays suitable for training a neural decoder.

- Data paper: *A brain-wide map of neural activity during complex behaviour* (`datapaper.pdf`)
- Methods paper: *Exploiting correlations across trials and behavioral sessions to improve
  neural decoding*, Zhang et al., Neuron 2026 (`methodpaper.pdf`)
- Reference code: `code/code_zhang2025`

## Dataset description

139 mice performed the IBL decision-making task while Neuropixels probes recorded neural
activity. On each trial a Gabor stimulus of one of five contrasts (0, 6.25, 12.5, 25, 100%)
appeared on the left or right of a screen and the mouse turned a wheel to bring it to the
centre. After an initial 90-trial unbiased block (P(left) = 0.5), the stimulus side followed
20:80 or 80:20 blocks of 20-100 trials, so the animal's *prior* about the stimulus side changes
through the session.

Trials are aligned to **stimulus onset** and span **-0.5 s to +1.5 s**, binned into
**100 non-overlapping 20 ms bins** (exactly the reference pipeline's configuration).

## Key statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 445 (of the 459 released; 14 lack whisker motion energy) |
| Subjects (mice) | 136 |
| Labs | 12 |
| Neurons | 62,779 well-isolated grey-matter units |
| Neurons / session | mean 141.1, median 123, range 1-516 |
| Trials | 189,057 after curation |
| Trials / session | mean 424.8, median 392, range 85-1,445 |
| Brain regions | 263 (IBL Beryl atlas mapping) |
| Time bins per trial | 100 (20 ms each) |
| File size | 11.28 GB |

Sanity checks against the data paper (all matched): 459 sessions, 699 insertions, 139 subjects,
621,733 total units, 75,708 well-isolated neurons, 645 mean trials/session (median 602, range
401-1,525), 81.4% correct, 58.7% correct on 0% contrast trials.

## How to load and use the data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# neural activity of session 3, trial 10: (n_neurons, 100) float32
x = data['neural'][3][10]

# decoder inputs and outputs for the same trial
u = data['input'][3][10]     # (3, 100) float32
y = data['output'][3][10]    # (4, 100) int8

print(data['input_names'])   # ['time_from_stim_onset', 'stim_onset', 'trial_num_in_block']
print(data['output_names'])  # ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']

# which mouse and which brain region each neuron came from
subject = data['subjects'][data['subject_idx'][3]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][3]]
```

Validate and train the reference decoder with:

```
python train_decoder.py converted_data.pkl --verify-only   # structure + summary only
python train_decoder.py converted_data.pkl --plot-samples  # full training
```

## Output format specification

```
data = {
  'neural':           list[n_sessions] of list[n_trials] of (n_neurons, 100) float32
  'input':            list[n_sessions] of list[n_trials] of (3, 100)         float32
  'output':           list[n_sessions] of list[n_trials] of (4, 100)         int8
  'subjects':         list[136] of str
  'subject_idx':      (445,) int64          index into 'subjects' per session
  'brain_regions':    list[263] of str      Beryl acronyms
  'brain_region_idx': list[445] of (n_neurons,) int64
  'input_names':      list[3] of str
  'output_names':     list[4] of str
  'output_values':    list[4] of list of str, names of each class
  'metadata':         dict (see below)
}
```

### Inputs (`input`, all time-varying with 100 bins)
| # | Name | Description |
|---|------|-------------|
| 0 | `time_from_stim_onset` | seconds from stimulus onset; bin centres -0.49 ... +1.49 |
| 1 | `stim_onset` | binary, 1 in the bin containing stimulus onset (bin 25) |
| 2 | `trial_num_in_block` | trials since the last change of `probabilityLeft` (constant within a trial) |

### Outputs (`output`, all stored time-varying with 100 bins)
| # | Name | Classes | Description |
|---|------|---------|-------------|
| 0 | `choice` | `left`=0, `right`=1 | side the mouse reported (constant within a trial) |
| 1 | `prior_prob_left` | `0.2`=0, `0.5`=1, `0.8`=2 | block prior probability that the stimulus is on the left (constant within a trial) |
| 2 | `wheel_speed` | `low`/`medium`/`high` | \|wheel velocity\|, per-session terciles |
| 3 | `whisker_motion_energy` | `low`/`medium`/`high` | whisker-pad motion energy, per-session terciles |

### Metadata
`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`
(`stimulus onset (trials.stimOn_times)`), `off_start` (-0.5), `off_end` (+1.5), `n_time_bins`,
`neural_data_type`, `neuron_inclusion`, `trial_inclusion`, `brain_region_mapping`,
`input_descriptions`, `output_descriptions`, `source`, plus `session_info` (a per-session record
with eid, subject, lab, trial and neuron counts, the camera used for motion energy, the tercile
thresholds and the mean firing rate) and `skipped_sessions`.

## Processing summary

- **Neurons**: only well-isolated units (IBL single-unit QC `label >= 1`: amplitude > 50 uV,
  noise cut-off < 20 uV, no refractory-period violation) located in grey matter (Beryl acronym
  not `root`/`void`). Probes of a session are merged, as in the reference analyses.
- **Trials**: the reference `load_trials_and_mask(max_trial_len=10)` mask, i.e. reaction time in
  [0.08, 2.0] s, no NaN in the six key trial events, feedback within 10 s of the go cue, and a
  response made; plus complete NaN-free wheel and whisker coverage of the 2 s window.
- **Binning**: 20 ms bins over [-0.5, +1.5] s from stimulus onset, verified bit-identical to the
  reference `bincount2D` routine.
- **Normalisation**: spike counts are standardised per time bin with a single scalar mean/std
  pooled over all neurons and trials of the session, matching the reference
  `standardize_spike_data`. Statistics are accumulated in float64 for numerical accuracy.

## Decoder performance

Validation balanced accuracy from `train_decoder.py` on the full dataset:

| Output | Chance | Validation balanced accuracy |
|--------|--------|------------------------------|
| choice | 0.500 | 0.6165 (0.81 peak at +0.3 s from a single 20 ms bin) |
| prior_prob_left | 0.333 | 0.6603 |
| wheel_speed | 0.333 | 0.6085 |
| whisker_motion_energy | 0.333 | 0.5999 |

Choice is at chance before stimulus onset and rises sharply afterwards, as it must be, which
confirms the temporal alignment. See `CONVERSION_NOTES.md` Step 12 for the full analysis.

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | the full converted dataset (445 sessions) |
| `sample_data.pkl` | a 2-session sample |
| `CONVERSION_NOTES.md` | full record of every decision, check and validation |
| `conversion_full_out.txt` / `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt` / `verification_sample_out.txt` | format-verification logs |
| `train_decoder_full_out.txt` / `train_decoder_sample_out.txt` | decoder training logs |
| `processing_<eid>.png` | per-step visualisations of the conversion |
| `sample_trials.png`, `predictions.png` | decoder sample and prediction plots |
| `cache/` | helper scripts, diagnostics and the rebuilt ONE cache index |

## Reproducing

```
python cache/build_one_cache.py                       # rebuild the local ONE index (needed once)
python -u convert_data.py converted_data.pkl --full   # about 2 minutes with 24 workers
python -u convert_data.py sample_data.pkl --sample --show-processing
python cache/sanity_checks.py converted_data.pkl      # 84 independent checks against raw files
```

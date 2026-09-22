# IBL Brain-Wide Map → neural-decoder dataset

`converted_data.pkl` is the IBL public **brain-wide map** (BWM) Neuropixels dataset
reformatted for the decoder in `train_decoder.py`: trial-aligned, 20 ms-binned spike counts
from well-isolated neurons, paired with task inputs and four categorical behavioural outputs.

Sources:
* IBL et al., *A brain-wide map of neural activity during complex behaviour* (`datapaper.pdf`)
* Zhang et al. 2026, *Neuron*, *Exploiting correlations across trials and behavioral sessions
  to improve neural decoding* (`methodpaper.pdf`; code in `code/code_zhang2025`)

All data are read through the `ONE` API (`one.api.ONE`) and the `brainbox` loaders
(`SpikeSortingLoader`, `SessionLoader`); no file in `data/` is opened directly.

---

## The experiment

Head-fixed mice turn a wheel to move a Gabor patch, which appears on the left or the right of
a screen, to the centre. Stimulus contrast is drawn from {0, 6.25, 12.5, 25, 100}%. The first
90 trials of a session are unbiased (p(left) = 0.5); afterwards the prior probability that the
stimulus appears on the left alternates between 0.2 and 0.8 in uncued blocks of 20–100 trials
(mean 51). Correct responses are rewarded with water. Wheel position, two side-view cameras
(whisker-pad motion energy) and Neuropixels recordings are acquired simultaneously.

## How the data were processed

| Step | Choice |
|------|--------|
| Sessions | the 459-session BWM freeze (`code_zhang2025/data/bwm_release.csv`); 441 survive |
| Probes | all insertions of a session merged into one population (`merge_probes`) |
| Neurons | IBL well-isolated units (`clusters.label >= 1`: amplitude > 50 µV, noise cut-off < 20 µV, no refractory-period violation) located in grey matter (Beryl acronym ∉ {root, void}); ≥ 5 per session |
| Trials | `brainbox`/reference `load_trials_and_mask(max_trial_len=10.0)`: no NaN in stimOn/choice/feedback/probabilityLeft/firstMovement/feedbackType, reaction time ∈ [0.08, 2.0] s, feedback − goCue ≤ 10 s, choice ≠ 0; plus full behavioural coverage of the window with no NaN, and at least one recorded spike |
| Alignment | stimulus onset (`trials.stimOn_times`) |
| Window | −0.5 s to +1.5 s |
| Binning | 20 ms non-overlapping ⇒ **T = 100**; bin *k* covers `[stimOn − 0.5 + 0.02k, stimOn − 0.5 + 0.02(k+1))` |
| Neural values | raw spike counts (float32), not normalised |
| Continuous behaviour | linearly interpolated onto the right edge of each bin, then split at the session's 33.3/66.7 percentiles |

## Key statistics

| | |
|---|---|
| Sessions | 441 |
| Subjects (mice) | 136 |
| Probe insertions | 672 |
| Neurons | 62,757 (mean 142.3/session, range 7–516) |
| Brain regions (Allen **Beryl**) | 263 (240 seen in ≥ 2 sessions) |
| Trials | 187,918 (mean 426/session, range 125–1445) |
| Timepoints per trial | 100 (20 ms bins, 2 s) |
| Mean firing rate of kept neurons | 11.8 Hz |
| choice | left 0.508 / right 0.492 |
| prior | 0.2 → 0.417, 0.5 → 0.140, 0.8 → 0.443 |
| wheel speed / whisker ME | ≈ 1/3 per class by construction |
| File size | 11.7 GB |

Cross-checks against the data paper: 891 Kilosort units per probe (paper: 889) and 108.6
well-isolated units per probe (paper: 108).

## Loading and using the data

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

sess, trial = 0, 0
X = data['neural'][sess][trial]    # (n_neurons, 100) float32 spike counts
U = data['input'][sess][trial]     # (2, 100)  float32
Y = data['output'][sess][trial]    # (4, 100)  int64 class labels

data['subjects'][data['subject_idx'][sess]]                       # mouse name
[data['brain_regions'][i] for i in data['brain_region_idx'][sess]]  # region of each neuron
data['metadata']['session_info'][sess]                            # per-session provenance
```

Re-create it with:

```
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~4 min, 28 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Output format specification

```
data = {
  'neural':  [session][trial] -> (n_neurons, 100) float32   spike counts per 20 ms bin
  'input':   [session][trial] -> (2, 100)        float32
  'output':  [session][trial] -> (4, 100)        int64
  'subjects':          list[str], 136 mouse names
  'subject_idx':       (441,) int64 index into 'subjects'
  'brain_regions':     list[str], 263 Beryl acronyms
  'brain_region_idx':  [session] -> (n_neurons,) int64 index into 'brain_regions'
  'input_names':       ['time_from_stim_onset_s', 'trial_number_in_block']
  'output_names':      ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']
  'output_values':     names of the classes of each output
  'metadata':          {...}
}
```

### Inputs (both time-varying, `(2, 100)`)
| # | Name | Meaning | Range |
|---|------|---------|-------|
| 0 | `time_from_stim_onset_s` | time of the right edge of each 20 ms bin relative to stimulus onset | −0.48 … 1.50 |
| 1 | `trial_number_in_block` | 0-based index of the trial within its constant-`probabilityLeft` block, counted on the unfiltered trials table; constant across the trial | 0 … 93 |

### Outputs (all time-varying, `(4, 100)`; the first two are constant within a trial)
| # | Name | Classes | Source |
|---|------|---------|--------|
| 0 | `choice` | `left` (0), `right` (1) | `trials.choice`: +1 → left, −1 → right |
| 1 | `prior` | `0.2` (0), `0.5` (1), `0.8` (2) | `trials.probabilityLeft` |
| 2 | `wheel_speed` | `low`, `medium`, `high` | \|wheel velocity\| from `SessionLoader.load_wheel`, split at the session's tertiles |
| 3 | `whisker_motion_energy` | `low`, `medium`, `high` | left (else right) camera whisker-pad ROI motion energy, split at the session's tertiles |

The two continuous behaviours are discretised **per session** because whisker motion energy is
in arbitrary units that depend on camera, illumination and ROI size and is not comparable
across sessions; tertiles keep the classes balanced for the balanced-accuracy metric. The
thresholds used are stored per session in `metadata['session_info'][s]`.

### `metadata`
`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`
(stimulus onset), `off_start` (−0.5), `off_end` (+1.5), `n_timepoints` (100), `neural_units`,
`neuron_selection`, `trial_selection`, `input_description`, `output_description`,
`skipped_sessions`, and `session_info` — one record per session with `eid`, `subject`, `lab`,
`n_probes`, `n_clusters_total`, `n_good_units`, `n_neurons`, `n_trials_raw`, `n_trials_mask`,
`n_trials`, `trial_idx` (row indices into the raw trials table), `whisker_source`,
`wheel_speed_tertiles`, `whisker_me_tertiles`, `mean_firing_rate_hz`, `behavior_drop_counts`.

## Decoder results (`train_decoder_full_out.txt`)

| Output | Chance | Training balanced acc | Validation balanced acc |
|--------|--------|----------------------|--------------------------|
| choice | 0.500 | 0.638 | 0.615 |
| prior | 0.333 | 0.679 | 0.660 |
| wheel_speed | 0.333 | 0.617 | 0.611 |
| whisker_motion_energy | 0.333 | 0.599 | 0.592 |

The decoder labels **each 20 ms bin independently**, so these averages include the 25
pre-stimulus bins in which choice is barely represented. Scored bin by bin, choice accuracy is
0.57 before the stimulus and peaks at **0.72 at +0.30 s**, the median first-wheel-movement
latency. Pooling a trial's bins into one prediction gives 0.70 per-trial balanced accuracy for
choice and 0.735 for prior. See `CONVERSION_NOTES.md`, Step 12.

## Files

| File | What it is |
|------|------------|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | the full dataset (441 sessions) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | every decision, validation and check, step by step |
| `conversion_{sample,full}_out.txt` | conversion logs |
| `verification_{sample,full}_out.txt` | format-verification logs |
| `train_decoder_{sample,full}_out.txt` | decoder training logs |
| `processing_<eid>.png` | per-session, per-step processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder sample plots |
| `cache/` | investigation and validation scripts (see `cache/README_CACHE.md`) |

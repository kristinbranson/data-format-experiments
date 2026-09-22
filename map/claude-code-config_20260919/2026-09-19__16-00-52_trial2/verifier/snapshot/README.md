# Mesoscale Activity Map (MAP) — decoder-ready conversion

Brain-wide Neuropixels recordings from mice performing a memory-guided directional
licking task, converted from NWB into the pickle format used by
`train_decoder.py`.

* **Source**: DANDI:000363 v0.230822.0128 — *Mesoscale Activity Map Dataset*
  (Chen, Nguyen, Li & Svoboda, 2023), 174 NWB files, 28 mice, 50 GB.
* **Data paper**: Chen et al., *Brain-wide neural activity underlying memory-guided
  movement*, Cell 187, 676–691 (2024).
* **Methods paper**: Wang, Kurgyis et al., *Brain-wide analysis reveals movement
  encoding structured across and within brain areas*, Nature Neuroscience (2025).
  Reference analysis code: `/app/code` (`druckmann-lab/MapVideoAnalysis`).

## The task

Mice heard a series of three 150 ms pure tones (12 kHz → lick left, 3 kHz → lick
right) during a 0.65 s **sample** epoch, withheld licking through a 1.2 s **delay**
epoch, and reported their choice after an auditory **go cue** by licking one of two
lick ports within a 1.5 s answer period. Licking during the sample or delay epoch
("early lick") replayed the epoch. On ~20 % of randomly interleaved trials, ALM was
photoinhibited for 0.5 s during the delay, always ending before the go cue.
Orofacial movements were tracked at 300 Hz with DeepLabCut (tongue, jaw, nose).

## Key statistics of the converted dataset

| | |
|---|---|
| Sessions | 136 (of 174 source files) |
| Subjects (mice) | 28 |
| Trials | 70,949 (mean 522 per session, range 206–796) |
| Neurons | 54,629 (median 392 per session, range 90–923) |
| Brain regions | 14 major regions × 2 hemispheres |
| Time bins | 80 × 50 ms, spanning −2.5 s to +1.5 s around the go cue |
| Behavioural performance | 83.8 % correct (range 65.8–98.9 %) |
| Pickle size | 9.45 GB |

Class fractions (over all time points):

| Output | Classes and fractions |
|---|---|
| `choice` | left 0.448, right 0.442, no lick 0.110 |
| `outcome` | ignore 0.110, miss 0.153, hit 0.737 |
| `early_lick` | no 0.884, yes 0.116 |
| `tongue_y_position` | <40th pct 0.106, 40th–60th pct 0.053, >60th pct 0.106, not visible 0.734 |

## Loading and using the data

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# neural activity of session 3, trial 17: (n_neurons, 80) firing rates in Hz
fr = data['neural'][3][17]

# the matching decoder inputs (2, 80) and outputs (4, 80)
x = data['input'][3][17]
y = data['output'][3][17]

# which mouse, and which brain region each neuron is in
mouse = data['subjects'][data['subject_idx'][3]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][3]]

# time axis, in seconds relative to the go cue
import numpy as np
t = np.arange(80) * 0.05 - 2.5 + 0.025
```

Train and evaluate the reference decoder:

```
python train_decoder.py converted_data.pkl              # train + validate
python train_decoder.py converted_data.pkl --verify-only  # format check + summary
python train_decoder.py converted_data.pkl --plot-samples # also save sample plots
```

Regenerate the pickle from the NWB files (≈36 s on 24 cores):

```
python -u convert_data.py converted_data.pkl --full --jobs 24
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + plots
```

## Output format specification

```python
data = {
  'neural':  [ [ (n_neurons, 80) float32 firing rate in Hz, ... per trial ], ... per session ],
  'input':   [ [ (2, 80) float32, ... ], ... ],
  'output':  [ [ (4, 80) int64,   ... ], ... ],

  'subjects':     list[str],                 # 28 mouse ids
  'subject_idx':  (136,) int64,              # index into `subjects` per session

  'brain_regions':    list[str],             # 28 'hemisphere Region' names
  'brain_region_idx': [ (n_neurons,) int64, ... ],   # one array per session

  'input_names':   ['time_from_tone_onset_s', 'photostim_on'],
  'output_names':  ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
  'output_values': [['left','right','no lick'],
                    ['ignore','miss','hit'],
                    ['no','yes'],
                    ['<40th pct','40th-60th pct','>60th pct','not visible']],
  'metadata': {...},
}
```

### Inputs (decoder inputs, time-varying)

| # | Name | Definition |
|---|------|------------|
| 0 | `time_from_tone_onset_s` | seconds from the onset of the instruction tone series (the last sample-epoch onset at or before the go cue) to the bin centre; negative before tone onset. 1.875 s at the go-cue bin on 87 % of trials |
| 1 | `photostim_on` | 1 while the ALM photoinhibition laser is on at the bin centre, else 0 |

### Outputs (variables to decode)

| # | Name | Classes | Definition |
|---|------|---------|------------|
| 0 | `choice` | 0 left, 1 right, 2 no lick | from `outcome` + `trial_instruction`: hit → instructed side, miss → opposite side, ignore → no lick. Constant within a trial |
| 1 | `outcome` | 0 ignore, 1 miss, 2 hit | `intervals/trials/outcome`. Constant within a trial |
| 2 | `early_lick` | 0 no, 1 yes | `intervals/trials/early_lick`. Constant within a trial |
| 3 | `tongue_y_position` | 0 <40th pct, 1 40th–60th pct, 2 >60th pct, 3 not visible | side-view DeepLabCut tongue y averaged over the visible frames of each 50 ms bin, discretised at the 40th/60th percentiles of that session; class 3 when no frame in the bin has likelihood > 0.5. Time-varying |

### `metadata` fields

`dataset`, `references`, `task_description`, `time_bin_size` (50.0 ms),
`temporal_alignment_event`, `off_start` (−2.5), `off_end` (+1.5), `neural_units`,
`n_timepoints`, `species`, `recording`, `neuron_curation`, `trial_curation`,
`session_curation`, `known_limitations`, `input_descriptions`,
`output_descriptions`, `session_info` (per-session record: id, subject,
performance, trial and neuron counts, tongue percentiles, …), `raw_trial_index`
(row indices into `intervals/trials` of each source NWB file, one list per session,
aligned with the converted trials), `n_sessions_available`, `n_sessions_kept`.

## What was filtered, and why

* **Neurons** — only units with `units/classification == 'good'` (the region-specific
  quality-control classifier of the MAP spike-sorting white paper, which both papers
  use) and a CCF annotation that maps to one of the 14 reference regions. This keeps
  25.90 % of all Kilosort2 clusters, matching the 25.9 % reported in the data paper.
* **Trials** — auto-water and free-water trials are removed (reward is not contingent
  on the animal's choice there, so `outcome` is not a behavioural report), as are
  trials whose behavioural video does not cover the full 4 s window. Photostimulation,
  early-lick and no-response trials are **kept**, because they are required decoder
  inputs/outputs.
* **Sessions** — the data paper's session-selection criteria (control-trial
  performance > 65 % and ≥ 50 correct lick-left and lick-right trials), plus one
  session whose DeepLabCut tongue tracking failed. 136 of 174 sessions survive.

## Known limitation

The published dataset only contains spikes inside `[trial.start_time,
trial.stop_time]`. On error (miss) trials that interval ends ~0.2–0.5 s after the go
cue, so the last bins of those trials necessarily read 0 Hz. 15.6 % of raw trials are
affected; no imputation is possible. See `metadata['known_limitations']`.

## Files

| File | Contents |
|---|---|
| `converted_data.pkl` | the full converted dataset (136 sessions) |
| `converted_data_session_info.json` | per-source-file record, including why each rejected session was rejected |
| `sample_data.pkl` | two sessions, for quick tests |
| `convert_data.py` | the conversion script |
| `region_map.py` | CCF annotation → major brain region table |
| `CONVERSION_NOTES.md` | full record of every decision, check and validation |
| `conversion_full_out.txt`, `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt`, `verification_sample_out.txt` | format-verification logs |
| `train_decoder_full_out.txt`, `train_decoder_sample_out.txt` | decoder training logs |
| `processing_<session>.png` | per-step diagnostic plots (`--show-processing`) |
| `cache/` | analysis and verification scripts used while developing the conversion |

# MAP dataset → neural-decoder format

`converted_data.pkl` contains the **Mesoscale Activity Map (MAP)** electrophysiology dataset
reformatted for training a neural decoder that predicts behavioural and task variables from
brain-wide spiking activity.

- **Source**: [DANDI:000363](https://dandiarchive.org/dandiset/000363) — Chen, S., Nguyen, T.,
  Li, N., Svoboda, K. (2023) *Mesoscale Activity Map Dataset*.
- **Data paper**: Chen et al., *Brain-wide neural activity underlying memory-guided movement*,
  **Cell** 187, 676–691 (2024).
- **Method paper**: Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding
  structured across and within brain areas*, **Nature Neuroscience** (2025).
- **Reference analysis code**: `MapVideoAnalysis` (`/app/code`).

---

## Dataset description

Head-fixed mice performed an **auditory delayed-response task**. During a 0.65 s sample epoch
one of two pure tones (3 kHz / 12 kHz, three 150 ms pulses separated by 100 ms) instructed the
animal to lick left or right. After a delay epoch (1.2 s in most sessions) an auditory go cue
(6 kHz, 0.1 s) released the animal, which reported its choice by licking one of two lick ports
during a 1.5 s answer period; a correct lick was rewarded with water. On a randomly interleaved
subset of trials, ALM was photoinhibited for 0.5 s during the delay epoch. Orofacial movements
were tracked at ~300 Hz with DeepLabCut (tongue, jaw, nose). Spiking activity was recorded with
up to five simultaneous Neuropixels probes covering the whole brain.

### Key statistics of the converted dataset

| Quantity | Value |
|---|---|
| Sessions | 173 |
| Subjects (mice) | 28 (3–10 sessions each) |
| Trials | 89,544 (mean 518/session, range 159–796) |
| Neurons (QC-passed units) | 69,453 (mean 401/session, range 90–923) |
| Brain areas | 14 |
| Time bins per trial | 80 (50 ms each, −2.5 s … +1.5 s around the go cue) |
| Alignment event | onset of the auditory go cue |
| Neural values | firing rate, spikes/s |

Neurons per brain area (this conversion vs. the numbers published in Fig. 2F of the data paper):

| Area | Ours | Paper | | Area | Ours | Paper |
|---|---|---|---|---|---|---|
| ALM | 8,464 | 8,717 | | Pallidum | 1,092 | 1,092 |
| Orbital | 10,223 | 10,223 | | Thalamus | 12,808 | 12,808 |
| Other cortex | 7,756 | 7,993 | | Hypothalamus | 815 | 815 |
| Olfactory | 4,137 | 4,137 | | Midbrain | 7,495 | 7,495 |
| Hippocampus | 1,944 | 1,944 | | Pons | 347 | 347 |
| Cortical subplate | 1,960 | 1,960 | | Medulla | 2,928 | 2,928 |
| Striatum | 7,664 | 7,664 | | Cerebellum | 1,820 | 1,820 |

Output class fractions (over all trials × time bins):

| Output | Distribution |
|---|---|
| `choice` | left 0.429, right 0.422, no lick 0.148 |
| `outcome` | ignore 0.148, miss 0.167, hit 0.685 |
| `early_lick` | no 0.884, yes 0.116 |
| `tongue_y_position` | <40th pct 0.101, 40th–60th pct 0.050, >60th pct 0.101, not visible 0.749 |

---

## How to load and use the data

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 10

fr   = data['neural'][session][trial]   # (n_neurons, 80) firing rate, spikes/s
inp  = data['input'][session][trial]    # (2, 80)  float32
out  = data['output'][session][trial]   # (4, 80)  int64 class labels

t = np.array(data['metadata']['bin_centers_s'])          # (80,) s, relative to the go cue
mouse   = data['subjects'][data['subject_idx'][session]]  # e.g. 'SC015'
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][session]]  # per neuron

# e.g. mean ALM firing rate on this trial
alm = np.array(regions) == 'ALM'
alm_psth = fr[alm].mean(0)
```

Train / evaluate the reference decoder:

```bash
python train_decoder.py converted_data.pkl --verify-only   # format + summary only
python train_decoder.py converted_data.pkl --plot-samples  # train, evaluate, plot
```

Reproduce the conversion from the raw NWB files:

```bash
python -u convert_data.py converted_data.pkl --full --workers 16          # ~40 s
python -u convert_data.py sample_data.pkl --sample --show-processing      # 2 sessions + plots
```

---

## Output format specification

```python
data = {
  'neural':  [ [ (n_neurons, 80) float32, ... per trial ], ... per session ],
  'input':   [ [ (2, 80)        float32, ... ], ... ],
  'output':  [ [ (4, 80)        int64,   ... ], ... ],

  'subjects':          list[str],          # 28 mouse names, e.g. 'SC015'
  'subject_idx':       (173,) int64,       # index into `subjects` for each session

  'brain_regions':     list[str],          # 14 area names
  'brain_region_idx':  [ (n_neurons,) int64, ... per session ],

  'input_names':   ['time_from_tone_onset', 'photostim_on'],
  'output_names':  ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
  'output_values': [['left', 'right', 'no lick'],
                    ['ignore', 'miss', 'hit'],
                    ['no', 'yes'],
                    ['<40th pct', '40th-60th pct', '>60th pct', 'not visible']],
  'metadata': {...},
}
```

### Inputs (per time bin)
| # | Name | Description |
|---|------|-------------|
| 0 | `time_from_tone_onset` | seconds elapsed since the onset of the instruction tone at the centre of the bin (negative before the tone). If an early lick triggered a replay of the sample epoch, the last tone before the go cue is used. Typically crosses 0 at −1.85 s relative to the go cue. |
| 1 | `photostim_on` | 1 if the 0.5 s ALM photoinhibition stimulus overlapped the bin, else 0. |

### Outputs (per time bin; the first three are constant within a trial)
| # | Name | Classes | Description |
|---|------|---------|-------------|
| 0 | `choice` | left, right, no lick | lick direction reported after the go cue |
| 1 | `outcome` | ignore, miss, hit | ignore = no lick in the answer period, miss = incorrect lick, hit = correct lick |
| 2 | `early_lick` | no, yes | whether the animal licked during the sample or delay epoch |
| 3 | `tongue_y_position` | <40th pct, 40th–60th pct, >60th pct, not visible | per-bin mean y-position of the tracked tongue (side camera), discretised with the 40th/60th percentiles of that session's visible bins |

### Metadata
`metadata` carries the task description, `time_bin_size` (50 ms), `temporal_alignment_event`,
`off_start` (−2.5), `off_end` (+1.5), `bin_centers_s`, descriptions of every input and output,
the curation rules, known limitations, and a `session_info` list with one record per session
(session id, mouse, file name, neuron/trial counts, the original trial indices, the tongue
percentile thresholds, mean firing rate, and per-session processing time).

---

## Curation applied

**Neurons** — units labelled `good` by the region-specific spike-sorting QC classifiers
(`units/classification`, the same QC the reference pipeline uses via `qc_mode='classifier'`)
and having a CCF histology annotation.

**Trials** — trials covered by the ephys recording, excluding auto-water and free-water trials
(as in the reference `get_regular_trial_mask`). Early-lick, no-response and photostimulation
trials are **kept**, because they are decoder outputs/inputs.

**Sessions** — all 173 sessions that contain at least one good unit (1 of the 174 NWB files has
none).

---

## Known limitations

1. The NWB export stores spikes only inside `[trial start, trial stop]`. On error (`miss`)
   trials the recorded interval ends about 0.8 s after the go cue, so the last bins of those
   trials contain no spikes (3.2 % of all bins dataset-wide); 0.2 % of bins likewise precede
   the start of the recorded interval. These bins are reported as a firing rate of 0.
   Measured effect on decoding: restricting the evaluation to bins where the recording is
   still live changes `outcome` accuracy by 0.007 and the other outputs not at all.
2. One session (`SC066_20210413_112028_s6`) has essentially no video, and five sessions
   (mice SC011, SC022) have video that stops around the go cue; their post-go tongue bins are
   labelled "not visible". One session (`SC027_20190803_150200_s21`) has a DeepLabCut tracking
   failure for the tongue marker; its tongue labels are unreliable.
3. ALM is defined here as motor/frontal-pole cortex ≥ 2.0 mm anterior to bregma, because the
   voxel mask used by the reference code is not distributed with it.

See `CONVERSION_NOTES.md` for the full derivation, all validation checks and the decoding
results.

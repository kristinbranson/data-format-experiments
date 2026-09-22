# Mesoscale Activity Map — decoder-ready conversion

`converted_data.pkl` is a decoder-ready reformatting of **DANDI:000363**, the
*Mesoscale Activity Map Dataset* (Chen, Nguyen, Li & Svoboda, 2023), published with

* Chen S. *et al.* **Brain-wide neural activity underlying memory-guided movement.**
  *Cell* 187, 676–691 (2024).
* Wang Z.A., Kurgyis B. *et al.* **Brain-wide analysis reveals movement encoding structured
  across and within brain areas.** *Nat. Neurosci.* (2025).

## Dataset description

28 head-fixed mice performed an auditory **memory-guided directional-licking task**:

```
 presample │ sample (0.65 s instruction tone train: 3 kHz or 12 kHz) │ delay (1.2 s) │ GO CUE │ answer (1.5 s) │ consumption
                                                                     └── ALM photoinhibition (0.5 s, 5.5 mW) on ~20 % of trials ──┘
```

Neuropixels probes (655 insertions, 3–5 simultaneously) recorded brain-wide from ALM down to
the medulla while two 300 Hz cameras tracked the tongue, jaw and nose with DeepLabCut.

Every trial in this conversion is a **80 × 50 ms window spanning −2.5 s to +1.5 s around the
Go cue**.

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 173 |
| Subjects (mice) | 28 (3–10 sessions each) |
| Neurons (QC-'good' single units) | 69,453 (mean 401 / session, median 390, range 90–923) |
| Trials | 89,544 (mean 518 / session, range 159–796) |
| Time bins per trial | 80 × 50 ms, −2.5 s … +1.5 s relative to the Go cue |
| Brain areas | 14 (ALM 8,882 · Thalamus 13,016 · Orbital 10,223 · Striatum 7,738 · Midbrain 7,480 · OtherCortex 7,338 · Olfactory 4,137 · Medulla 2,928 · Hippocampus 1,944 · CorticalSubplate 1,888 · Cerebellum 1,820 · Pallidum 1,090 · Hypothalamus 602 · Pons 367) |
| choice | left 42.9 % · right 42.2 % · no lick 14.8 % |
| outcome | hit 68.5 % · miss 16.7 % · ignore 14.8 % |
| early lick | 11.6 % of trials |
| photostimulation | 20.0 % of trials |
| tongue visible | 24.5 % of time bins |
| File size | 11.9 GB |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

sess, trial = 0, 0
rates  = data['neural'][sess][trial]   # (n_neurons, 80) firing rate in Hz, float32
inputs = data['input'][sess][trial]    # (2, 80)  float32
outs   = data['output'][sess][trial]   # (4, 80)  int64 class labels

t = data['metadata']['bin_centers_s']  # bin centres in s relative to the Go cue

# which brain area is neuron 17 of this session in?
data['brain_regions'][data['brain_region_idx'][sess][17]]

# which mouse ran this session?
data['subjects'][data['subject_idx'][sess]]
```

Validate / train the reference decoder:

```bash
python train_decoder.py converted_data.pkl --verify-only     # format + summary
python train_decoder.py converted_data.pkl --plot-samples    # train and evaluate
```

Regenerate the file from the raw NWB (~40 s on 14 cores):

```bash
python -u convert_data.py converted_data.pkl --full --jobs 14
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + plots
```

## Output format specification

```python
data = {
  'neural':  [ [ (n_neurons, 80) float32, ... per trial ], ... per session ],  # firing rate, Hz
  'input':   [ [ (2, 80)        float32, ... ], ... ],
  'output':  [ [ (4, 80)        int64,   ... ], ... ],

  'subjects':     ['SC011', ..., 'SC067'],          # 28 mouse IDs
  'subject_idx':  int64 array, shape (173,),        # index into 'subjects' per session

  'brain_regions':    ['ALM', 'Orbital', ..., 'Cerebellum'],   # 14 areas
  'brain_region_idx': [ int64 array (n_neurons,), ... ],       # per session

  'input_names':   ['time_from_tone_onset', 'photostim_on'],
  'output_names':  ['choice', 'outcome', 'early_lick', 'tongue_y'],
  'output_values': [['left', 'right', 'no lick'],
                    ['ignore', 'miss', 'hit'],
                    ['no', 'yes'],
                    ['<40th pct', '40-60th pct', '>60th pct', 'not visible']],

  'metadata': { 'task_description', 'time_bin_size' (50.0 ms),
                'temporal_alignment_event' ('Go cue onset'),
                'off_start' (-2.5), 'off_end' (1.5), 'n_time_bins' (80),
                'bin_centers_s', 'input_descriptions', 'output_descriptions',
                'neuron_curation', 'trial_curation', 'session_curation',
                'known_limitations', 'ccf_note',
                'session_info': [ per-session dict with session_id, subject, counts,
                                  tongue percentiles, timings, ... ] },
}
```

### Inputs
| # | Name | Description |
|---|---|---|
| 0 | `time_from_tone_onset` | seconds since the onset of that trial's instruction-tone (sample) epoch; continuous, ranges −1.5 … 11.9 s |
| 1 | `photostim_on` | 1 while ALM photoinhibition was on (0.5 s during the delay), else 0 |

### Outputs (all integer class labels, one per 50 ms bin)
| # | Name | Classes | Time-varying? |
|---|---|---|---|
| 0 | `choice` | left, right, no lick | constant within a trial |
| 1 | `outcome` | ignore, miss, hit | constant within a trial |
| 2 | `early_lick` | no, yes | constant within a trial |
| 3 | `tongue_y` | <40th pct, 40–60th pct, >60th pct, not visible | yes |

## Processing summary

* **Neurons** — only units with `units.classification == 'good'` (the region-specific
  logistic-regression quality-control classifier of Chen, Liu *et al.* 2023), as in the
  reference pipeline.
* **Alignment** — Go cue onset (`BehavioralEvents/go_start_times`), as in the reference
  pipeline.
* **Binning** — non-overlapping 50 ms bins, firing rate = spikes / bin width; numerically
  identical to the reference `sliding_histogram(bw=stride=0.05, rate=True)`.
* **Trials** — those covered by the ephys recording (`units.obs_intervals`), excluding
  auto-water and free-water trials. Photostimulation, early-lick and no-response trials are
  **kept**, because the decoding task uses them as an input / as output classes.
* **Tongue** — DeepLabCut side-view y coordinate, averaged over the frames of each bin whose
  likelihood exceeds 0.9; bins with no such frame are "not visible". Visible bins are split
  at the 40th and 60th percentile of the session's visible binned y values.
* **Brain areas** — from the per-unit Allen CCFv3 annotation; frontal motor cortex is called
  ALM only for units > 1.75 mm anterior to bregma and < 2.0 mm from the midline.

### Known limitation
The NWB release stores spikes and video only inside `[trial.start_time, trial.stop_time]`.
Error (*miss*) trials end ≈0.8 s after the Go cue because of the time-out, so the last
~0.7 s of the requested window on those trials contains no spikes (rate 0) and no video
(tongue "not visible"). 81.5 % of trials cover the full window. Two sessions
(`SC066_20210413_112028_s6`, `SC065_20210505_170309_s6`) have partially missing video.

## Reference decoder performance

Balanced accuracy of `train_decoder.py` (80 % / 20 % train-validation split per session):

| Output | Chance | Train | Validation |
|---|---|---|---|
| choice | 0.333 | 0.710 | **0.682** |
| outcome | 0.333 | 0.697 | **0.664** |
| early_lick | 0.500 | 0.795 | **0.755** |
| tongue_y | 0.250 | 0.687 | **0.656** |

Full details, validation and every processing decision are documented in
[`CONVERSION_NOTES.md`](CONVERSION_NOTES.md).

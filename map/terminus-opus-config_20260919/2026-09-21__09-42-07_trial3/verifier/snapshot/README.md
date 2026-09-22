# Mesoscale Activity Map (MAP) - decoder-ready conversion

This directory contains a conversion of the **Mesoscale Activity Map** Neuropixels dataset
(DANDI [000363](https://doi.org/10.48324/dandi.000363/0.231012.2129); Chen et al., *Cell* 2024,
"Brain-wide neural activity underlying memory-guided movement") into the pickle format used by
`train_decoder.py`. Processing follows the reference analysis code of Wang\*, Kurgyis\* et al.,
*Nature Neuroscience* 2025 ("Brain-wide analysis reveals movement encoding structured across and
within brain areas", `/app/code`) wherever applicable.

## Dataset description

Mice performed an **auditory delayed-response task**: a 3 kHz or 12 kHz tone train during the
sample epoch (0.65 s) instructed a lick to the right or left lick port; after a 1.2 s delay an
auditory go cue released the response. On ~20% of trials ALM was photoinhibited during the late
delay (0.5 s, always ending before the go cue). Neuropixels probes recorded brain-wide activity
and a 300 Hz side-view camera tracked the tongue, jaw and nose with DeepLabCut.

## Key statistics of the converted dataset

| Statistic | Value |
|---|---|
| Sessions | 138 (of 174 NWB files) |
| Subjects (mice) | 28 |
| Trials | 69,074 (mean 500 per session, range 159-796) |
| Neurons | 55,429 QC-'good' units (mean 402 per session, range 90-923) |
| Brain regions | 28 (14 coarse regions x hemisphere) |
| Time bins | 80 x 50 ms, from -2.5 s to +1.5 s around the **go cue** |
| Neural data | firing rate in Hz (spike count / 0.05 s) |
| Behavioural performance | mean 83.9% (65.8-98.9%) |

## Loading the data

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 80) firing rate in Hz
inputs = data['input'][session][trial]    # (2, 80)
outputs = data['output'][session][trial]  # (4, 80) integer class labels
```

## Format specification

| Key | Content |
|---|---|
| `neural` | list of sessions -> list of trials -> `(n_neurons, 80)` float32, firing rate in Hz |
| `input` | list of sessions -> list of trials -> `(2, 80)` float32 |
| `input_names` | `['time_from_tone_onset', 'photostim_on']` - signed seconds from the last sample-epoch tone onset before the go cue; binary ALM photoinhibition indicator |
| `output` | list of sessions -> list of trials -> `(4, 80)` int64 class labels |
| `output_names` | `['choice', 'outcome', 'early_lick', 'tongue_y']` |
| `output_values` | `choice`: left / right / no lick; `outcome`: ignore / miss / hit; `early_lick`: no / yes; `tongue_y`: <40th pctile / 40th-60th pctile / >60th pctile / not visible |
| `subjects`, `subject_idx` | 28 mouse ids; index of the mouse for each session |
| `brain_regions`, `brain_region_idx` | 28 "hemisphere region" names; region index for each neuron of each session |
| `metadata` | task description, `time_bin_size` (50 ms), `temporal_alignment_event` ('go cue onset'), `off_start` (-2.5), `off_end` (1.5), bin centres, per-session info (file, subject, trial/neuron counts, performance, tongue percentiles, raw trial and unit indices) and the curation rules |

## Processing summary

* **Alignment**: go cue (`acquisition/BehavioralEvents/go_start_times`), one per trial.
* **Neural**: spikes of QC-'good' units (`units/classification`) counted in 80 non-overlapping
  50-ms bins and divided by the bin width (Hz), as in the reference `sliding_histogram(rate=True)`.
* **Neuron curation**: `classification == 'good'` (white-paper QC classifier); units flagged
  unstable by `units/is_good_trials` on a retained trial are dropped.
* **Session curation** (data paper): performance > 65% and >= 50 correct lick-left and lick-right
  trials, at least one good unit, tongue tracking present.
* **Trial curation**: auto-water and free-water trials removed; trials must have been observed by
  the probes (`units/obs_intervals`) and have video frames in every bin. Early-lick, ignore/miss
  and photostimulation trials are kept because they are decoder outputs/inputs.
* **Brain regions**: Allen CCF `anno_name` of each unit mapped to coarse regions (ALM, Orbital,
  OtherCortex, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum,
  Hippocampus, Olfactory, CorticalSubplate), hemisphere from the CCF ML coordinate (midline 5,700 um).

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~40 s with 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Decoder performance (reference decoder, whole 4 s window)

| Output | Train balanced acc | Validation balanced acc | Chance |
|---|---|---|---|
| choice | 0.740 | 0.695 | 0.333 |
| outcome | 0.718 | 0.659 | 0.333 |
| early_lick | 0.776 | 0.742 | 0.500 |
| tongue_y | 0.714 | 0.684 | 0.250 |

Epoch-resolved control analysis (ALM populations, logistic regression, 5-fold CV) reproduces the
published ~0.9 late-delay choice decoding accuracy of the data paper.

Details of every decision, validation and sanity check are in `/app/CONVERSION_NOTES.md`.

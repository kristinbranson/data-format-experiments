# Mesoscale Activity Map → neural-decoder dataset

This directory contains a decoder-ready conversion of the **Mesoscale Activity Map** dataset
(DANDI:000363; Chen, Nguyen, Li & Svoboda 2023), the data behind

* Chen et al., *Brain-wide neural activity underlying memory-guided movement* (`datapaper.pdf`)
* Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding structured across and within brain areas*
  (`methodpaper.pdf`, reference code in `code/`)

## Dataset description

28 mice performed an auditory delayed-response task while up to five Neuropixels probes recorded
brain-wide spiking activity and a 300 Hz side-view camera tracked orofacial movements.
A sample tone (3 or 12 kHz) instructed licking left or right, a 1.2 s delay epoch required withholding
the response, and an auditory **go cue** opened a 1.5 s answer epoch. On ~20 % of trials ALM was
photoinhibited during the late delay.

The conversion keeps the units that pass the published spike-sorting quality-control classifier
(`units.classification == 'good'`), aligns everything to the **go cue**, and extracts
**−2.5 s … +1.5 s** in **50 ms** bins (80 bins per trial).

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 138 |
| Subjects (mice) | 28 |
| Trials | 72,251 (mean 524 per session, range 159–800) |
| Neurons (good units) | 55,437 (mean 402 per session, range 90–923) |
| Time bins per trial | 80 (50 ms, −2.5 … +1.5 s relative to the go cue) |
| Brain-region labels | 28 (14 coarse CCF regions × 2 hemispheres) |
| Behavioural performance | 83.9 % correct (paper: 84 %) |
| Photostimulation trials | 20.3 % |
| File size | 9.6 GB (`converted_data.pkl`) |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 80) firing rate in spikes/s
inputs = data['input'][session][trial]    # (2, 80) float32
outputs = data['output'][session][trial]  # (4, 80) int64 class indices
```

Train/evaluate the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl              # train
python train_decoder.py /app/converted_data.pkl --verify-only # format check + summary
```

Regenerate the dataset:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full             # ~35 s with 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

| Field | Type | Meaning |
|---|---|---|
| `neural` | list[session] of list[trial] of (n_neurons, 80) float32 | firing rate in spikes/s, 50 ms bins, go-cue aligned |
| `input` | list[session] of list[trial] of (2, 80) float32 | see below |
| `output` | list[session] of list[trial] of (4, 80) int64 | see below |
| `subjects` | list[str] | 28 mouse names (`SC0xx`) |
| `subject_idx` | (n_sessions,) int | index into `subjects` |
| `brain_regions` | list[str] | e.g. `left ALM`, `right Thalamus` |
| `brain_region_idx` | list[session] of (n_neurons,) int | region of each neuron |
| `input_names`, `output_names`, `output_values` | lists | names / class labels |
| `metadata` | dict | task description, binning, alignment, curation rules, per-session info |

### Inputs
| # | Name | Description |
|---|---|---|
| 0 | `time_from_tone_onset` | seconds from the onset of the instructing sample tone to the bin centre (continuous) |
| 1 | `photostim_on` | 1 while ALM photoinhibition is on, else 0 |

### Outputs (all categorical, emitted for every time bin)
| # | Name | Classes |
|---|---|---|
| 0 | `lick_direction_choice` | 0 = left, 1 = right, 2 = no lick |
| 1 | `outcome` | 0 = ignore, 1 = miss, 2 = hit |
| 2 | `early_lick` | 0 = no, 1 = yes |
| 3 | `tongue_y_position` | 0 = < 40th pct, 1 = 40–60th pct, 2 = > 60th pct, 3 = not visible (per-session percentiles) |

## Curation applied

* **Neurons**: only units labelled `good` by the published QC classifier.
* **Sessions**: ≥ 1 good unit; the data paper's behavioural criteria (performance > 65 % and ≥ 50 correct
  lick-left and lick-right trials); side-view video covering the analysis window.
* **Trials**: all trial types are kept (photostimulation, early lick, ignore, auto/free water) because they are
  required decoder inputs/outputs. Trials are dropped only when they are outside the units' `obs_intervals`,
  when the video covers < 90 % of the window, or when no good unit fired a single spike (acquisition dropout).

## Decoder performance (reference `train_decoder.py`)

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| lick_direction_choice | 0.333 | 0.680 |
| outcome | 0.333 | 0.653 |
| early_lick | 0.500 | 0.746 |
| tongue_y_position | 0.250 | 0.663 |

A per-time-bin logistic-regression control (`cache/epoch_decoding.py`) reproduces the published choice-decoding
time course: AUC ≈ 0.57 before the tone, 0.71–0.76 during sample/delay, 0.96–0.98 after the go cue
(papers: 0.51, 0.66, 0.99).

Full details of every decision and validation step are in `CONVERSION_NOTES.md`.

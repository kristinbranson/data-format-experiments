# Mesoscale Activity Map — decoder-ready conversion

`converted_data.pkl` contains the Neuropixels recordings and orofacial video tracking of
the **Mesoscale Activity Map** dataset (DANDI:000363, Chen, Nguyen, Li & Svoboda 2023),
re-organised as trials aligned to the go cue so that a neural decoder can be trained to
predict behavioral and task variables from population activity.

Source papers:
- Chen et al., *Brain-wide neural activity underlying memory-guided movement* (`datapaper.pdf`)
- Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding structured across
  and within brain areas*, Nat Neurosci 2025 (`methodpaper.pdf`)

## The experiment

Head-fixed mice performed an **auditory delayed-response task**. Three pure tones
(12 kHz → lick left, 3 kHz → lick right; 0.65 s *sample* epoch) instructed the upcoming
lick direction. After a 1.2 s *delay* epoch, an auditory **go cue** (6 kHz, 0.1 s)
released the animal to lick one of two ports during a 1.5 s answer period; the correct
port delivered a small water reward. Licking during the sample or delay epoch ("early
lick") replayed the epoch. On ~20 % of randomly interleaved trials, ALM was
photoinhibited (5.5 mW, 0.5 s, always ending at or before the go cue). Two to five
Neuropixels probes recorded simultaneously from ALM, orbital cortex, striatum, thalamus,
midbrain, medulla and other structures, while a 300 Hz side-view camera tracked the nose,
jaw and tongue with DeepLabCut.

## Key statistics of the converted dataset

| | |
|---|---|
| Sessions | 150 |
| Mice | 28 (1–10 sessions each) |
| Trials | 77,521 (mean 517 per session, range 159–796) |
| Neurons | 59,749 (mean 398 per session, range 90–923) |
| Time bins | 80 per trial, 50 ms, −2.5 s … +1.5 s around the go cue |
| Neural units | firing rate in Hz (spike count / 0.05 s); median 4.5 Hz per neuron |
| Behavioral performance of the included sessions | 83.6 % correct (range 65.8–98.9 %) |
| Photostimulated trials | 20.5 % |

Brain areas (neurons): thalamus 11,329 · orbital 9,226 · ALM 7,720 · midbrain 6,416 ·
striatum 6,225 · other cortex 6,108 · olfactory 3,471 · medulla 2,866 · cerebellum 1,820 ·
hippocampus 1,680 · pallidum 1,021 · hypothalamus 755 · cortical subplate 773 · pons 339.

## Curation applied

- **Sessions**: the data paper's criteria — behavioral performance on control trials
  (no photostimulation, no early lick, no auto/free water) > 65 %, and ≥ 50 correct
  lick-left and ≥ 50 correct lick-right trials. 23 of 174 sessions fail this; one further
  session has no unit passing quality control.
- **Neurons**: only units labelled `good` by the region-specific quality-control
  classifier of the Chen/Liu et al. 2023 spike-sorting white paper
  (`units/classification`). This reproduces the paper's per-area unit counts exactly.
- **Trials**: all trials except auto-water and free-water trials, trials outside the
  units' `obs_intervals` (9 sessions have electrophysiology for only part of the
  behavioral session), and one trial in which no neuron fired at all. Early-lick,
  no-response and photostimulation trials are **kept** — they are exactly what the
  decoder has to predict or receive.

## Loading the data

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 0
X = data['neural'][session][trial]   # (n_neurons, 80) float32, firing rate in Hz
u = data['input'][session][trial]    # (2, 80)  float32
y = data['output'][session][trial]   # (4, 80)  int64, class indices

regions = [data['brain_regions'][i] for i in data['brain_region_idx'][session]]
mouse = data['subjects'][data['subject_idx'][session]]
```

## Format specification

```
data = {
  'neural'          : list[n_sessions] of list[n_trials] of (n_neurons, 80) float32
  'input'           : list[n_sessions] of list[n_trials] of (2, 80) float32
  'output'          : list[n_sessions] of list[n_trials] of (4, 80) int64
  'subjects'        : list[28] of str                 # 'SC011' ... 'SC067'
  'subject_idx'     : (150,) int64                    # index into 'subjects'
  'brain_regions'   : list[15] of str
  'brain_region_idx': list[n_sessions] of (n_neurons,) int64
  'input_names'     : ['time_from_tone_onset', 'photostim']
  'output_names'    : ['choice', 'outcome', 'early_lick', 'tongue_y']
  'output_values'   : names of the classes of each output (see below)
  'metadata'        : dict (see below)
}
```

### Inputs (decoder inputs, time-varying)

| idx | name | units | description |
|---|---|---|---|
| 0 | `time_from_tone_onset` | s | bin centre minus the onset of the instruction-tone series (the last sample-epoch onset before the go cue). Crosses zero at −1.85 s relative to the go cue on a standard trial. Range [−1.53, 11.89] s. |
| 1 | `photostim` | binary | 1 while the ALM photoinhibition laser is on (0.5 s, always ending at or before the go cue), 0 otherwise. |

### Outputs (to be decoded, time-varying class indices)

| idx | name | classes |
|---|---|---|
| 0 | `choice` | 0 `left`, 1 `right`, 2 `no lick` — the port the animal actually licked |
| 1 | `outcome` | 0 `ignore` (no response), 1 `miss` (wrong port), 2 `hit` (rewarded) |
| 2 | `early_lick` | 0 `no`, 1 `yes` — the animal licked during the sample or delay epoch |
| 3 | `tongue_y` | 0 `<40th pctile`, 1 `40-60th pctile`, 2 `>60th pctile`, 3 `not visible` |

`choice`, `outcome` and `early_lick` are properties of the whole trial and are therefore
constant across the 80 bins. `tongue_y` is genuinely time-varying: within each 50 ms bin
the side-view tongue y-position is averaged over the video frames whose DeepLabCut
likelihood exceeds 0.9, and the result is discretised against the 40th and 60th
percentiles of all visible bins **of that session** (stored per session in
`metadata['session_info'][s]['tongue_percentile_values']`). Bins with no visible frame —
the tongue is in the mouth, or the camera was between trials — get class 3.

### Metadata

`task_description`, `time_bin_size` (50.0 ms), `temporal_alignment_event` (go cue onset),
`off_start` (−2.5 s), `off_end` (+1.5 s), `bin_centers` (80 values, s, relative to the go
cue), `neural_units`, `input_units`, `dataset`, `references`, `neuron_curation`,
`trial_curation`, `session_curation`, `tongue_likelihood_threshold`,
`tongue_percentiles`, `session_info` (one record per converted session: id, mouse,
performance, trial and neuron counts, tongue percentiles) and `excluded_sessions` (with
the reason for each exclusion).

## Reproducing the conversion

```bash
python -u convert_data.py converted_data.pkl --full --njobs 24     # ~45 s
python -u convert_data.py sample_data.pkl --sample --show-processing
python -u train_decoder.py converted_data.pkl --plot-samples
python3 cache/sanity_checks.py converted_data.pkl 8                # raw-data spot checks
```

`--show-processing` writes `processing_<session_id>.png`, a figure that walks through
every processing step (raw spikes vs bins, PSTH, inputs, tongue discretisation, output
cross-checks) for the first two sessions.

## Decoder performance

Validation balanced accuracy of the provided decoder (`train_decoder.py`, 150 sessions,
15,563 held-out trials):

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| choice | 0.333 | 0.677 |
| outcome | 0.333 | 0.648 |
| early_lick | 0.500 | 0.748 |
| tongue_y | 0.250 | 0.660 |

Time-resolved choice decoding (PCA + logistic regression at each bin) gives AUC 0.54
before the tone, 0.75 during sample + delay and 0.91 late in the response epoch, matching
the epoch structure reported in the method paper.

See `CONVERSION_NOTES.md` for the full record of decisions, consistency checks against the
papers, and validation results.

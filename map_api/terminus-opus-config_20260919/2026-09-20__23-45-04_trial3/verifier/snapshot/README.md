# MAP dataset converted for neural decoding

This directory contains a decoder-ready conversion of the **Mesoscale Activity Map (MAP)**
dataset (DANDI:000363, Chen, Nguyen, Li & Svoboda 2023), the dataset used in

- Chen et al., *Brain-wide neural activity underlying memory-guided movement* (`datapaper.pdf`)
- Wang, Kurgyis et al. 2025, *Brain-wide analysis reveals movement encoding structured across
  and within brain areas* (`methodpaper.pdf`, code in `code/`)

## Dataset description

Head-fixed mice performed an **auditory delayed-response task**: a 0.65 s tone (3 or 12 kHz,
three 150 ms pips) instructed lick-left vs lick-right, followed by a 1.2 s delay; an auditory
go cue (6 kHz, 0.1 s) opened a 1.5 s answer period in which the mouse licked one of two ports.
On ~20% of randomly interleaved trials, ALM was photoinhibited during the last 0.5 s of the
delay. Neuropixels probes (2-5 simultaneously) recorded brain-wide activity; the orofacial
movements were tracked at 300 Hz with DeepLabCut.

## Converted files

| File | Content |
|---|---|
| `converted_data.pkl` | full dataset (173 sessions, 11.9 GB) |
| `sample_data.pkl` | 2 sessions, for quick tests |
| `convert_data.py` | conversion script (`python -u convert_data.py out.pkl [--full|--sample] [--show-processing]`) |
| `CONVERSION_NOTES.md` | full record of all decisions, checks and validation results |
| `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` | logs |
| `cache/` | exploration, sanity-check and analysis scripts |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 80) firing rate in Hz
inp    = data['input'][session][trial]    # (2, 80)
out    = data['output'][session][trial]   # (4, 80) integer class labels
```

## Format specification

- **Alignment**: go cue onset. Window **-2.5 s to +1.5 s**, **50 ms** non-overlapping bins -> 80 time points.
- `neural[s][t]`: `(n_neurons, 80)` float32 **firing rate in spikes/s** (spike count per bin / 0.05 s).
- `input[s][t]`: `(2, 80)` float32
  | idx | name | description |
  |---|---|---|
  | 0 | `time_from_tone_onset` | seconds since the onset of the tone (sample epoch) that preceded the go cue; 1.85 s at the go cue in standard trials |
  | 1 | `photostim` | 1 while the ALM photoinhibition laser is on, else 0 |
- `output[s][t]`: `(4, 80)` int64
  | idx | name | values |
  |---|---|---|
  | 0 | `choice` | 0 left, 1 right, 2 no lick (direction of the first lick in the answer period; constant in time) |
  | 1 | `outcome` | 0 ignore, 1 miss, 2 hit (constant in time) |
  | 2 | `early_lick` | 0 no, 1 yes (constant in time) |
  | 3 | `tongue_y_position` | 0 < 40th pct, 1 40-60th pct, 2 > 60th pct of the session's visible tongue-y distribution, 3 not visible (time-varying) |
- `subjects` (28 mouse ids), `subject_idx` (per session), `brain_regions` (14 coarse CCF groups x 2 hemispheres), `brain_region_idx` (per neuron).
- `metadata`: task description, bin size, alignment event, window, per-session info (`session_id`,
  `nwb_file`, `trial_index` into the NWB trials table, `go_times`, unit/trial counts, tongue percentiles).

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 173 (paper: 173) |
| Subjects | 28 (paper: 28) |
| Neurons (QC 'good') | 69,453 (paper: 69,943) |
| Neurons / session | 401 mean (90-923) |
| Trials | 89,544 |
| Trials / session | 518 mean (159-796) |
| Time points / trial | 80 |
| choice | left 0.431, right 0.421, no lick 0.148 |
| outcome | ignore 0.148, miss 0.167, hit 0.685 |
| early lick | 0.116 |
| photostim | 20.0% of trials |
| tongue class | 0.096 / 0.048 / 0.096 / 0.760 |

## Decoder performance (`train_decoder.py`, full dataset)

| Output | Train balanced acc | Validation balanced acc | Chance |
|---|---|---|---|
| choice | 0.703 | 0.677 | 0.333 |
| outcome | 0.697 | 0.664 | 0.333 |
| early_lick | 0.799 | 0.752 | 0.500 |
| tongue_y_position | 0.685 | 0.660 | 0.250 |

Time-resolved control (logistic regression per 50 ms bin, as in the data paper): choice decoding
is at chance before the tone (0.54), rises during the sample/delay epochs (0.63) and reaches
0.85 (up to 0.94 per session) in the response epoch - the published pattern.

## Processing summary

- Loaded exclusively with `pynwb`.
- Neurons: `units.classification == 'good'` (region-specific QC classifiers of the Chen & Liu
  et al. 2023 white paper) with a valid CCF annotation; hemisphere from the CCF ML coordinate
  (midline 5,700 um), as in the reference code.
- Trials: auto-water and free-water trials excluded (reward not contingent on choice, as in the
  reference `get_regular_trial_mask`); trials without ephys coverage excluded. Early-lick,
  no-response and photostim trials are kept because they are required decoder outputs/inputs.
- Sessions: 1 of the 174 NWB files has no QC-good units and is dropped, giving the paper's 173.

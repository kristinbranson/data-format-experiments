# MAP dataset converted for neural decoding

This directory contains a decoder-ready conversion of the **Mesoscale Activity Map (MAP)** dataset
(DANDI 000363; Chen et al., *Brain-wide neural activity underlying memory-guided movement*, Cell 2024),
processed following the pipeline of Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding
structured across and within brain areas*, Nature Neuroscience 2025
(code: `druckmann-lab/MapVideoAnalysis`).

## Dataset description

Mice performed an auditory delayed-response task: a 3 kHz or 12 kHz instruction tone during the sample
epoch, a 1.2 s delay in which licking had to be withheld, and an auditory go cue after which the animal
reported the instruction by licking the left or right lick port. Neuropixels probes recorded up to five
brain areas simultaneously; a side-view 300 Hz camera tracked the tongue, jaw and nose with DeepLabCut.
On ~20% of the trials ALM was photoinhibited during the last 0.5 s of the delay.

| Statistic | Value |
|---|---|
| Sessions | 173 |
| Mice | 28 |
| Neurons (QC-'good' units) | 69,453 (mean 401 per session, range 90-923) |
| Trials | 89,544 (mean 518 per session, range 159-796) |
| Time bins per trial | 80 x 50 ms, from 2.5 s before to 1.5 s after the go cue |
| Brain regions | 28 (14 coarse CCF groups x hemisphere) |
| File size | 11.9 GB (`converted_data.pkl`) |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 80) float32, firing rate in Hz
inputs = data['input'][session][trial]    # (2, 80)  float32
outputs = data['output'][session][trial]  # (4, 80)  int64 category labels
```

## Format specification

| Key | Contents |
|---|---|
| `neural` | list of sessions -> list of trials -> `(n_neurons, 80)` float32 **firing rates in Hz** (spike count per 50 ms bin / 0.05 s) |
| `input` | `(2, 80)` float32 per trial: `time_from_tone_onset` (s, continuous) and `photostim_on` (0/1) |
| `output` | `(4, 80)` int64 per trial: `choice`, `outcome`, `early_lick`, `tongue_y_position` |
| `subjects` / `subject_idx` | 28 mouse ids; index of the mouse for each session |
| `brain_regions` / `brain_region_idx` | 28 region names (`'left ALM'`, `'right Medulla'`, ...); region index of every neuron |
| `input_names`, `output_names`, `output_values` | names of the inputs/outputs and of each output category |
| `metadata` | task description, `time_bin_size` = 50 ms, `temporal_alignment_event` = go cue onset, `off_start` = -2.5, `off_end` = +1.5, curation rules, per-session info |

### Output categories
| Output | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `choice` | no lick | left | right | - |
| `outcome` | ignore | miss | hit | - |
| `early_lick` | no | yes | - | - |
| `tongue_y_position` | < 40th pctile | 40-60th pctile | > 60th pctile | not visible |

The tongue percentiles are computed **per session** over all bins in which the DeepLabCut likelihood
exceeds 0.9.

## Curation applied

- **Neurons**: only units with `units/classification == 'good'` (the region-specific QC classifiers of
  Chen, Liu et al. 2023) that also carry a CCF annotation — the same criterion as the reference code.
- **Trials**: auto-water and free-water trials removed; trials outside the units' `obs_intervals`
  (i.e. not covered by the ephys recording) and trials without a single recorded spike removed.
  Early-lick, no-response and photostimulation trials are **kept** because they are decoder targets or
  inputs.
- **Sessions**: the one file without any QC-good unit is skipped, leaving the 173 sessions reported in
  the data paper.

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~45 s with 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Decoder performance (validation balanced accuracy)

| Output | Accuracy | Chance |
|---|---|---|
| choice | 0.682 | 0.333 |
| outcome | 0.661 | 0.333 |
| early_lick | 0.750 | 0.500 |
| tongue_y_position | 0.655 | 0.250 |

See `CONVERSION_NOTES.md` for the full record of decisions, consistency checks against the papers and
validation results.

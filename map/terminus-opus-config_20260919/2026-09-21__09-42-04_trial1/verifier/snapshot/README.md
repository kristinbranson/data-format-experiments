# MAP dataset converted for neural decoding

`converted_data.pkl` contains the Mesoscale Activity Map dataset (DANDI:000363; Chen et al.,
*Brain-wide neural activity underlying memory-guided movement*; analysed in Wang, Kurgyis et al. 2025,
*Brain-wide analysis reveals movement encoding structured across and within brain areas*) reformatted for
training a neural decoder.

## Dataset
- 28 mice, 173 sessions, 659 Neuropixels insertions, **69,453 quality-controlled neurons**, **89,544 trials**.
- Task: auditory delayed-response. A 3 kHz / 12 kHz tone (sample epoch) instructs licking left / right; after a
  1.2 s delay an auditory go cue releases a 1.5 s response epoch. On ~20% of trials ALM was photoinhibited during
  the last 0.5 s of the delay (ending at the go cue).
- Neural data: spike trains of the QC-good units, aligned to the **go cue**, binned into **80 non-overlapping
  50 ms bins** covering **-2.5 s to +1.5 s**, expressed as firing rates in spikes/s.

## How to load
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 80) float32, Hz
inp    = data['input'][session][trial]    # (2, 80) float32
out    = data['output'][session][trial]   # (4, 80) int64
```

## Format
| Key | Type | Description |
|-----|------|-------------|
| `neural` | list[session] of list[trial] of (n_neurons, 80) float32 | firing rate (Hz) per 50 ms bin |
| `input` | list[session] of list[trial] of (2, 80) float32 | decoder inputs |
| `output` | list[session] of list[trial] of (4, 80) int64 | decoder targets (categorical) |
| `subjects` | list[str], len 28 | mouse ids |
| `subject_idx` | (173,) int | mouse of each session |
| `brain_regions` | list[str], len 15 | coarse region names |
| `brain_region_idx` | list[session] of (n_neurons,) int | region of each neuron |
| `input_names`, `output_names`, `output_values` | | names of the variables and of their categories |
| `metadata` | dict | task description, bin size (50 ms), alignment event, off_start (-2.5), off_end (+1.5), curation rules, per-session info |

### Inputs
| idx | name | description |
|-----|------|-------------|
| 0 | `time_from_tone_onset` | signed seconds from the onset of the instruction tone to the bin centre (median tone onset = go - 1.85 s) |
| 1 | `photostim_on` | 1 while the ALM photoinhibition laser is on in that bin, else 0 |

### Outputs (all time-varying; the first three are constant within a trial)
| idx | name | categories |
|-----|------|-----------|
| 0 | `lick_direction` | 0 left, 1 right, 2 no lick |
| 1 | `outcome` | 0 ignore, 1 miss, 2 hit |
| 2 | `early_lick` | 0 no, 1 yes |
| 3 | `tongue_y_position` | 0 (< 40th percentile of the session), 1 (40-60th), 2 (> 60th), 3 not visible |

### Brain regions
ALM, Orbital, OtherCortex, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum,
Hippocampus, Olfactory, CorticalSubplate, Unknown.

## Curation
- Neurons: `units/classification == 'good'` (the region-specific QC classifiers of Chen, Liu et al. 2023) with a
  CCF annotation.
- Trials: auto-water and free-water trials removed; trials outside the probes' recording interval
  (`units/obs_intervals`) and trials without a single spike removed. Early-lick, no-response and photostimulation
  trials are kept because they are decoder variables.
- Sessions: those with no QC-good unit or fewer than two usable trials removed (1 of 174).

## Key statistics
| Statistic | Value |
|-----------|-------|
| Sessions / mice | 173 / 28 |
| Neurons | 69,453 (median 390 per session) |
| Trials | 89,544 (mean 518 per session) |
| Outcome | hit 0.685, miss 0.167, ignore 0.148 |
| Early lick | 0.116 |
| Lick direction | left 0.429, right 0.422, no lick 0.148 |
| Tongue class | 0.144 / 0.033 / 0.068 / 0.755 (not visible) |
| Photostim trials | 20.0% |

## Decoder performance (`train_decoder.py`, validation balanced accuracy)
| Output | Accuracy | Chance |
|--------|----------|--------|
| lick_direction | 0.683 | 0.333 |
| outcome | 0.661 | 0.333 |
| early_lick | 0.751 | 0.500 |
| tongue_y_position | 0.651 | 0.250 |

## Reproducing
```
python -u convert_data.py converted_data.pkl --full          # ~45 s with 16 processes, 11.9 GB output
python -u convert_data.py sample_data.pkl --sample --show-processing
python -u train_decoder.py converted_data.pkl --plot-samples
```
See `CONVERSION_NOTES.md` for the full record of decisions and validation.

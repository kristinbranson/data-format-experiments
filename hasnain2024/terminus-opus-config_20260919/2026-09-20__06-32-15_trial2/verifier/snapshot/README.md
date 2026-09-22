# ALM two-context (DR / WC) dataset converted for neural decoding

Converted from **Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the behaving mouse", Nature Neuroscience 2024** (data: Zenodo DOI 10.5281/zenodo.13941415; code: `/app/code`).

## Dataset description

Head-fixed mice alternate block-wise (10-25 trials per block) between two directional-licking tasks:

- **DR (delayed response)**: an auditory sample tone (1.3 s) instructs a left or right lick; after a delay epoch (0.9 s fixed, or randomly drawn from {0.3, 0.6, 1.2, 1.8, 2.4, 3.6} s in the randomized-delay sessions) an auditory go cue (10 ms chirp) releases the movement. Correct licks are rewarded with ~3 ul of water. No response within 3 s = *ignore* trial.
- **WC (water cued)**: no auditory cues; a water drop appears at a random time at a randomly chosen port and the mouse licks to collect it.

Neural activity is spike-sorted single/multi-unit activity recorded with Neuropixels/H2 probes in **ALM** (a few dual-probe sessions also contain **tjM1** units). Two high-speed cameras (400 Hz, side + bottom views) tracked the tongue, jaw, nose and paws with DeepLabCut, and per-frame motion energy was computed.

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 44 (25 fixed-delay + 19 randomized-delay) |
| Subjects | 14 mice |
| Trials | 13,762 (mean 313 per session) |
| Neurons | 2,456 (2,311 ALM, 145 tjM1); mean 56 per session |
| Time bins | 100 per trial, 50 ms each, spanning -2.5 to +2.5 s around the go cue |
| Outcome distribution | 74.9% correct, 12.0% incorrect, 13.1% ignore |
| Lick direction | 42.3% left, 44.6% right, 13.1% none |
| Context | 90.3% DR, 9.7% WC (33% WC in the 12 two-context sessions) |

## How to load

```python
import pickle, numpy as np
data = pickle.load(open('/app/converted_data.pkl', 'rb'))

neural = data['neural'][session][trial]   # (n_neurons, 100) float32, firing rate in spikes/s
inp    = data['input'][session][trial]     # (1, 100) float32, time from go cue (s)
out    = data['output'][session][trial]    # (6, 100) int64, categorical outputs
```

## Format specification

| Field | Description |
|---|---|
| `neural` | list (sessions) of list (trials) of `(n_neurons, 100)` float32 arrays: go-cue-aligned firing rates (spikes/s), 10 ms binning + causal Gaussian smoothing (as in the paper's `getSeq.m`/`mySmooth.m`), averaged into 50 ms bins |
| `input` | list of list of `(1, 100)` float32: `time_from_go_cue`, the bin centres from -2.475 to +2.475 s |
| `output` | list of list of `(6, 100)` int64 (see below) |
| `subjects` / `subject_idx` | 14 animal ids and the animal index of each session |
| `brain_regions` / `brain_region_idx` | `['ALM','tjM1']` and the region of each neuron |
| `input_names` | `['time_from_go_cue']` |
| `output_names` | `['lick_direction','context','outcome','tongue_velocity','paw_velocity','motion_energy']` |
| `output_values` | value names per output (see below) |
| `metadata` | task description, `time_bin_size` (50 ms), `temporal_alignment_event` ('go cue onset'), `off_start` (-2.5), `off_end` (+2.5), processing/curation descriptions and per-session `session_info` |

### Outputs

| idx | name | values |
|---|---|---|
| 0 | `lick_direction` | 0 = left, 1 = right, 2 = none (ignore trials) - per trial |
| 1 | `context` | 0 = WC (water-cued), 1 = DR (delayed response) - per trial |
| 2 | `outcome` | 0 = incorrect, 1 = correct, 2 = ignore - per trial |
| 3 | `tongue_velocity` | 0 = below the per-session median, 1 = at/above it, 2 = tongue not visible - per time bin |
| 4 | `paw_velocity` | same coding, averaged over the two bottom-view paw markers - per time bin |
| 5 | `motion_energy` | 0 = below the per-session median, 1 = at/above it, 2 = no video - per time bin |

## Reproducing the conversion

```
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~12 s with 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl
```

Processing decisions, validation and all consistency checks are documented in `CONVERSION_NOTES.md`.

## Decoder performance (validation balanced accuracy, full dataset)

| Output | Accuracy | Chance |
|---|---|---|
| lick_direction | 0.654 | 0.333 |
| context | 0.863 | 0.500 |
| outcome | 0.631 | 0.333 |
| tongue_velocity | 0.624 | 0.333 |
| paw_velocity | 0.620 | 0.333 |
| motion_energy | 0.804 | 0.333 |

Replicating the paper's own per-time-bin SVM decoders on this converted data gives 0.947 peak choice accuracy and 0.925 peak context accuracy, matching Figs 3b/4b of the paper.

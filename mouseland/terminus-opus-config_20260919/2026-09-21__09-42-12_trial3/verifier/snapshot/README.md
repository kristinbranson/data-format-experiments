# Zhong et al. 2025 virtual-reality dataset, converted for neural decoding

This directory contains a decoder-ready conversion of the two-photon mesoscope dataset from
**Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer et al. (2025), "Unsupervised pretraining in
biological neural networks"** (`paper.pdf`, reference code in `code/`, raw data in `data/`).

## Dataset description

Head-fixed mice ran on an air-floating ball through 4 m linear virtual-reality corridors whose walls
showed frozen crops of naturalistic textures (e.g. "leaf" vs "circle", or "rock" vs "wood"), separated
by 2 m of grey space. The virtual reality advanced at a constant 60 cm/s whenever the mouse ran faster
than 6 cm/s. A sound cue was played at a random position (0.5-3.5 m) on every trial. In the task
("sup") cohort, water was available after the cue in one of the two corridors; unsupervised, naive and
grating-exposed mice ran the same corridors without water and were not water restricted.
Neural activity was recorded with a two-photon mesoscope across V1 and higher visual areas and
processed with Suite2p (non-negative deconvolution, 0.75 s decay).

- 89 recordings (sessions) from 19 mice
- 20,547-89,577 neurons recorded per session (2,000 randomly subsampled neurons stored per session)
- 38,110 trials (corridor traversals); median 21 retained time bins per trial
- Time bin = one imaging frame, 314.7 ms (3.18 Hz)

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, n_timebins) float32, deconvolved activity
inputs = data['input'][session][trial]    # (4, n_timebins) float32
outputs = data['output'][session][trial]  # (4, n_timebins) int16 class labels
```

Train/evaluate the reference decoder:

```bash
python train_decoder.py converted_data.pkl            # train + validate
python train_decoder.py converted_data.pkl --verify-only
```

Reproduce the conversion:

```bash
python -u convert_data.py converted_data.pkl --full          # ~70 s, 6.6 GB
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Output format specification

`data` is a dict with:

| Key | Content |
|-----|---------|
| `neural` | list over 89 sessions of lists over trials of `(n_neurons, T)` float32 deconvolved activity |
| `input` | same nesting, `(4, T)` float32 |
| `output` | same nesting, `(4, T)` int16 categorical labels |
| `subjects` | 19 mouse names |
| `subject_idx` | `(89,)` int index into `subjects` |
| `brain_regions` | `['V1', 'mHV', 'lHV', 'aHV']` |
| `brain_region_idx` | list of `(n_neurons,)` int arrays, one per session |
| `input_names` | `['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']` |
| `output_names` | `['stimulus_category', 'licking', 'position_bin', 'speed_bin']` |
| `output_values` | class names for each output |
| `metadata` | task description, time bin size (ms), alignment event, bin edges, per-session info, curation rules |

### Inputs
| # | Name | Units | Description |
|---|------|-------|-------------|
| 0 | `time_to_sound_cue` | s | signed time until the sound cue (positive before, negative after), time-varying |
| 1 | `day_of_training` | days | days since the mouse's first imaging session, constant per trial |
| 2 | `time_since_trial_start` | s | time since corridor entry, time-varying |
| 3 | `reward_availability` | 0/1 | 1 if the trial is in the rewarded corridor of a task mouse, constant per trial |

### Outputs
| # | Name | Classes |
|---|------|---------|
| 0 | `stimulus_category` | 8: crops 1/2/3 of the non-rewarded texture (0, 1, 7), crops 1/2/3 of the rewarded texture (2, 3, 4), two spatial swaps of rewarded crop 1 (5, 6) |
| 1 | `licking` | 2: `no_lick`, `lick` (>= 1 lick in the time bin) |
| 2 | `position_bin` | 4 x 1 m bins of the 4 m corridor |
| 3 | `speed_bin` | 4 global running-speed quartiles (edges 12.42 / 25.35 / 40.85 cm/s) |

## Processing summary (matches the reference code)

- Neural signal: Suite2p deconvolved traces, `np.concatenate(d['spks'], 0)` (as `utils.load_spk`); no dF/F, no additional neuron quality filtering.
- Time bins retained: inside the 0-4 m texture corridor (`ft_CorrSpc`) **and** the mouse running so the VR moves (`ft_move > 0`), behaviour truncated to the number of imaging frames -- the `fr_valid` mask of `utils.Get_dprime_selective_neuron`.
- Trials: corridor traversals from `ft_trInd`, aligned to corridor entry (`StartFr`); all 38,110 trials retained.
- Brain areas from the retinotopic atlas `iarea` via the `utils.neu_area_ID` mapping (V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4); neurons outside these areas are excluded.
- 2,000 neurons per session are stored (random, seed 0), matching the decoder's own `svd_max_neurons = 2000` random projection, because storing all in-area neurons would need 152 GB.

## Key statistics

| Statistic | Value |
|-----------|-------|
| Sessions / mice | 89 / 19 |
| Trials | 38,110 (mean 428 per session, range 84-789) |
| Time bins per trial | median 21, min 11, max 178 |
| Time bin | 314.69 ms |
| Stimulus category (trials) | 11,964 / 2,266 / 12,339 / 6,538 / 2,279 / 1,173 / 1,242 / 309 |
| Licking | 3.5% of bins overall, 12.1% within the 28 task sessions |
| Position bins | 25.0 / 24.9 / 25.0 / 25.2 % |
| Speed bins | 25 / 25 / 25 / 25 % |
| Rewarded-corridor trials | 11.7% of all trials, 39.7% of task-session trials |

## Decoder performance (validation, balanced accuracy)

| Output | Accuracy | Chance |
|--------|----------|--------|
| stimulus_category | 0.814 | 0.125 |
| licking | 0.794 | 0.500 |
| position_bin | 0.869 | 0.250 |
| speed_bin | 0.622 | 0.250 |

See `CONVERSION_NOTES.md` for the full decision log, consistency checks and validation results.

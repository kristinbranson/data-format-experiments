# CA1 reward-relative coding — decoder-ready dataset

Converted from **DANDI:001361**, the two-photon imaging + virtual-reality dataset of

> Sosa M., Plitt M. H., Giocomo L. M. (2025) *A flexible hippocampal population code for experience
> relative to reward.* **Nature Neuroscience**.

`converted_data.pkl` holds the whole dataset in the decoder format described below;
`sample_data.pkl` holds two sessions in the same format for quick tests.

---

## Dataset description

Head-fixed mice run laps on a 450 cm virtual linear track while CA1 pyramidal neurons are imaged at
15.5 Hz. Water reward is delivered operantly when the mouse licks inside a hidden 50 cm reward zone
at one of three possible locations (**A** 80–130 cm, **B** 200–250 cm, **C** 320–370 cm); only one
zone is active at a time and reward is randomly omitted on ~15% of trials. On "switch" sessions the
reward zone moves to a new location after trial 30, on some days together with a switch between two
visually distinct environments (**ENV 1 / ENV 2**). Each trial is one lap, ending with a teleport
through a variable-length grey inter-trial interval.

| Statistic | Value |
|---|---|
| Subjects (mice) | 11 |
| Sessions | 152 (14 per mouse; m11 has 12 — imaging started on day 3) |
| Trials | 12,135 (79.8 ± 6.9 per session) |
| Neurons | 138,298 total; 910 ± 490 per session (154 – 2,320) |
| Timepoints | 2,576,026 imaging frames inside trials |
| Time bin | 64.484 ms (15.5078125 Hz per imaging plane), identical in every session |
| Brain region | dorsal CA1 (m17/m18 imaged in two planes, pooled) |
| Trial alignment | start of trial (entry to the track at 0 cm) |
| Trial end | the teleport frame (excluded); trial length varies, 96 – 3,359 frames |
| Rewarded trials | 84.7% |

**Neural signal.** ΔF/F of each curated neuron, computed exactly as in the paper: suite2p raw `F`
minus 0.7 × neuropil (trial mean added back), a per-trial *maximin* baseline (Gaussian σ = 15
samples, then a 20 s minimum filter followed by a 20 s maximum filter), ΔF/F = (F − base)/|base|,
smoothed with a 2-sample (~0.129 s) Gaussian. Baselines are taken within each trial, extended over
the preceding inter-trial interval only on the sessions where the laser was not blanked.

**Curation.** Neurons: suite2p `iscell` (the authors' manual curation, as archived in the NWB files)
followed by exclusion of putative interneurons (Pearson r between ΔF/F and running speed > 0.5;
380 cells, 0.32 ± 0.60% per session). Trials: the 81 trials with lick-sensor error (>30% of frames
with a cumulative lick count > 2) are dropped, reproducing the paper's count exactly.

---

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:      # ~9.6 GB, needs ~11 GB RAM
    data = pickle.load(f)

session, trial = 0, 5
X = data['neural'][session][trial]   # (n_neurons, T)  float32 dF/F
U = data['input'][session][trial]    # (4, T)          float32
Y = data['output'][session][trial]   # (6, T)          int64 class labels

data['subjects'][data['subject_idx'][session]]        # e.g. 'm11'
data['metadata']['session_info'][session]['scene']    # e.g. 'Env1_LocationA_to_C'
```

Train and evaluate the reference decoder:

```bash
python train_decoder.py converted_data.pkl --plot-samples     # full run, ~13 min on one GPU
python train_decoder.py converted_data.pkl --verify-only      # format check + summary only
```

Re-create the dataset from the NWB files:

```bash
python -u convert_data.py converted_data.pkl --full                     # ~1 min, 12 workers
python -u convert_data.py sample_data.pkl --sample --show-processing    # 2 sessions + diagnostics
python -u convert_data.py events.pkl --full --neural-signal events      # OASIS-deconvolved instead
python -u sanity_checks.py converted_data.pkl 6                         # re-verify against raw NWB
```

---

## Output format specification

```
data = {
  'neural'          : list[n_sessions] of list[n_trials] of (n_neurons, T) float32   # dF/F
  'input'           : list[n_sessions] of list[n_trials] of (4, T) float32
  'output'          : list[n_sessions] of list[n_trials] of (6, T) int64
  'subjects'        : list[11] of str
  'subject_idx'     : (152,) int64 index into 'subjects'
  'brain_regions'   : ['CA1']
  'brain_region_idx': list[n_sessions] of (n_neurons,) int64   (all zeros)
  'input_names', 'output_names', 'output_values', 'metadata'
}
```

### Inputs (`input_names`)
| # | Name | Units / coding |
|---|---|---|
| 0 | `time_from_trial_start_s` | seconds since the trial-start frame (time-varying) |
| 1 | `environment` | 0 = ENV 1, 1 = ENV 2 (constant within a trial) |
| 2 | `trial_number` | 0-indexed trial index within the session (constant within a trial) |
| 3 | `previous_trial_outcome` | 0 = previous trial omitted, 1 = rewarded; 0 for the first trial |

### Outputs (`output_names`, `output_values`) — all time-varying
| # | Name | Classes | Fraction of frames |
|---|---|---|---|
| 0 | `distance_to_reward_zone` | `< -50`, `-50…-10`, `-10…0`, `in zone (0)`, `0…+10`, `+10…+50`, `> +50` cm | .251 .102 .073 .238 .021 .072 .243 |
| 1 | `position` | `<90`, `90–180`, `180–270`, `270–360`, `>360` cm | .212 .177 .231 .226 .154 |
| 2 | `speed` | `<2`, `2–10`, `10–20`, `20–40`, `>40` cm/s | .117 .087 .134 .319 .343 |
| 3 | `lick` | `no lick`, `lick` | .777 .223 |
| 4 | `reward_zone_location` | `A`, `B`, `C` | .332 .336 .333 |
| 5 | `reward_outcome` | `omitted`, `rewarded` | .158 .842 |

`distance_to_reward_zone` is the signed distance to the **nearest point of the active reward zone**:
0 while the animal is inside the zone, negative before it, positive after it.
`reward_zone_location` and `reward_outcome` are constant within a trial but stored as time series.

### `metadata`
`task_description`, `time_bin_size` (ms), `temporal_alignment_event`, `off_start` (0.0),
`off_end` (`None` — trials end at the teleport and vary in length), `trial_end_event`,
`neural_signal` and `neural_signal_description`, `sampling_rate_hz`, dataset counts,
`neuron_curation`, `trial_curation`, `input_descriptions`, `output_descriptions`, `source`,
`brain_region_note`, and `session_info` — one dict per session with subject, experiment day, date,
scene, path, trial and neuron counts, dropped-trial counts, `keep_teleports`, reward fraction,
per-trial reward-zone labels and environment, and the per-neuron imaging-plane index.

---

## Decoder performance (reference decoder, 80/20 trial split within every session)

| Output | Chance | Train balanced acc. | Validation balanced acc. |
|---|---|---|---|
| distance_to_reward_zone | 0.143 | 0.798 | **0.621** |
| position | 0.200 | 0.891 | **0.761** |
| speed | 0.200 | 0.732 | **0.628** |
| lick | 0.500 | 0.796 | **0.768** |
| reward_zone_location | 0.333 | 0.963 | **0.873** |
| reward_outcome | 0.500 | 0.933 | **0.602** |

`reward_outcome` is the hardest output by design: whether a trial is rewarded is undetermined until
the animal licks in the reward zone, so the frames before the zone carry no information (validation
accuracy there is 0.531, rising to 0.653 after the zone).

See `CONVERSION_NOTES.md` for the full record of decisions, validation and consistency checks.

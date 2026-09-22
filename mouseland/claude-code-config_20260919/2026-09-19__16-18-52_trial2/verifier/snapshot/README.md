# Zhong et al. 2025 visual-cortex VR dataset — decoder-ready conversion

Converted form of the two-photon mesoscope dataset from

> Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer & Pachitariu,
> **"Unsupervised pretraining in biological neural networks"**, *Nature* **644**, 741 (2025).
> Data: figshare item 28811129. Code: `zhong-et-al-2025`.

`converted_data.pkl` holds every one of the paper's 89 recordings in the standard
`{neural, input, output, …}` decoder format, temporally aligned to corridor entry.

---

## The experiment in one paragraph

Head-fixed mice run on a ball through 4 m linear virtual-reality corridors whose walls
carry "frozen" crops of naturalistic textures (circle/leaf for most mice, rock/wood for
some), separated by 2 m of grey space. The VR advances at a constant 60 cm s⁻¹ whenever
the mouse runs faster than 6 cm s⁻¹ and is frozen otherwise. A sound cue is played at a
position drawn uniformly from 0.5–3.5 m on every trial. In the **task** cohort (28 of
89 sessions) the cue marks the start of the reward zone in one of the two corridors and
licking after it delivers water; **unsupervised** and **naive** mice see the same
corridors and hear the same cue but are never rewarded. Neural activity is Suite2p
non-negative deconvolved calcium (τ = 0.75 s) from V1 and higher visual areas, sampled
at 3.176 Hz.

---

## Loading

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

data['neural'][s][t]            # (2000, T) float32, z-scored deconvolved activity
data['input'][s][t]             # (4, T)    float32
data['output'][s][t]            # (4, T)    int64, class indices
data['subjects'][data['subject_idx'][s]]      # mouse name for session s
data['brain_regions'][data['brain_region_idx'][s][n]]  # area of neuron n in session s
data['metadata']['session_info'][s]           # provenance for session s
```

Sessions are ordered by (mouse, date, block); `metadata['session_info'][s]['session']`
gives the original `<mname>_<datexp>_<blk>` key.

## Format specification

### `neural`
89 sessions × (84–722 trials) × `(2000, T)` float32. `T` is the number of imaging
frames the animal spent running inside the 0–4 m texture region on that trial
(median 21, mean 21.6, range 11–178). One time bin = one imaging frame = **314.85 ms**.

### `input` — decoder inputs, `(4, T)` float32
| # | name | type | description |
|---|------|------|-------------|
| 0 | `time_to_sound_cue_s` | continuous, time-varying | signed seconds until the sound cue (positive before it, 0 at the cue, negative after) |
| 1 | `day_of_training` | continuous, per-trial | days since that mouse's first imaging session (0–92) |
| 2 | `time_since_trial_start_s` | continuous, time-varying | seconds of running since corridor entry |
| 3 | `reward_available` | binary, per-trial | 1 if this corridor is the rewarded one, else 0 |

Per-trial inputs are broadcast across the `T` columns. Times 0 and 2 are measured on the
"running clock" — elapsed time accumulated over running frames only, because
non-running samples are excluded from the dataset (see `CONVERSION_NOTES.md`, Step 5).

### `output` — variables to decode, `(4, T)` int64
| # | name | classes | `output_values` |
|---|------|---------|-----------------|
| 0 | `stimulus` | 7 | `circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2` |
| 1 | `licking` | 2 | `no_lick, lick` — ≥1 lick in this time bin |
| 2 | `position_bin` | 4 | `0-1m, 1-2m, 2-3m, 3-4m` |
| 3 | `speed_bin` | 4 | `speed_q1 … speed_q4` — quartiles of running speed over the whole dataset (edges 12.40, 25.28, 40.75 cm s⁻¹) |

`stimulus` uses the paper's own `stim_id` code, which is **role-based**, not name-based:
`leaf1` is always the exemplar that plays the rewarded role, so labels are comparable
across mice that were shown different texture sets (circle/leaf vs rock/wood).

### `metadata`
`task_description`, `time_bin_size` (314.85 ms), `temporal_alignment_event`
("trial start = entry into the visual corridor"), `off_start` (0.0), `off_end` (`None`,
trials have variable length), plus `sample_selection`, `neuron_selection`,
`trial_selection`, `time_base`, `speed_bin_edges`, `sampling_rate_hz`, and
`session_info` — a per-session record with mouse, date, block, cohort, reward mode,
rewarded wall, day of training, trial counts, neuron counts and frame interval.

---

## Key statistics

| | |
|---|---|
| Sessions | 89 (matches the paper's "89 recordings") |
| Mice | 19 (1–8 sessions each) |
| Trials | 37,801 (84–722 per session, median 429) |
| Timepoints | 815,485 |
| Neurons recorded | 20,547–89,577 per session (the paper's exact range), 4,691,034 total |
| Neurons in visual cortex | 4,105,393 |
| Neurons kept | 2,000 per session = 178,000 (V1 78,082 · mHV 50,617 · lHV 20,720 · aHV 28,581) |
| Cohorts | 28 task, 46 unsupervised, 15 naive/grating-control sessions |
| Output fractions | stimulus (.319, .060, .335, .170, .059, .027, .029) · licking (.963, .037) · position (.250, .249, .250, .252) · speed (.25, .25, .25, .25) |

### Curation applied
- samples: inside the 0–4 m texture region **and** the VR moving (`ft_CorrSpc &
  ft_move > 0`) — the paper's own analysis window;
- neurons: inside visual cortex (`iarea ∉ {−1, 7}`), non-zero variance, then a
  region-stratified random subsample of 2,000 per session (a full-resolution dataset
  would be 151.5 GB and is not trainable in practice — see `CONVERSION_NOTES.md`);
- trials: 309 `circle3` trials (0.81 %, 4 sessions) dropped — the reference assigns them
  no stimulus role; 21 boundary samples dropped where `ft_Pos` had already wrapped to
  the next corridor.

---

## Decoder performance

`python /app/train_decoder.py /app/converted_data.pkl`

| Output | Chance | Train balanced acc | **Validation balanced acc** |
|---|---|---|---|
| stimulus | 0.143 | 0.998 | **0.861** |
| licking | 0.500 | 0.991 | **0.773** |
| position_bin | 0.250 | 1.000 | **0.897** |
| speed_bin | 0.250 | 0.940 | **0.574** |

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full --workers 8   # ~60 s
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples      # ~5 min on an L4
python -u /app/cache/sanity_checks.py                                       # independent checks
```

Options: `--max-neurons N` (default 2000), `--no-zscore` (ablation),
`--workers K`, `--show-processing` (per-step figures for 2 sessions).

Full rationale for every decision, all validation results, and the ablations behind the
2,000-neuron cap and the z-scoring choice are in `CONVERSION_NOTES.md`.

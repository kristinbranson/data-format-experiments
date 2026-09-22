# Zhong et al. 2025 visual-cortex dataset — decoder-ready conversion

`converted_data.pkl` contains the two-photon mesoscope recordings of
**Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer & Pachitariu (2025),
"Unsupervised pretraining in biological neural networks", *Nature* 644:741**
(data doi:10.25378/janelia.28811129), reformatted for training a neural decoder.

---

## 1. The experiment

Head-fixed mice run on an air-floating ball through **4 m linear virtual-reality
corridors** whose walls are frozen crops of naturalistic textures, separated by 2 m of
grey space. Two corridors ("leaf" and "circle", or "rock"/"wood" in some mice) are
presented in pseudo-random order. A **sound cue** is played at a random position
(uniform 0.5–3.5 m) in every corridor. In the **task** cohort, water is available after
the cue in one of the two corridors and the mice learn to lick in anticipation of reward;
the **unsupervised**, **unsupervised-grating** and **naive** cohorts see the same corridors
without any reward. The virtual reality advances at a fixed 60 cm s⁻¹ whenever the mouse
runs faster than 6 cm s⁻¹ and is stationary otherwise. Neural activity is Suite2p
non-negative-deconvolved calcium fluorescence from up to ~90,000 neurons simultaneously
across V1 and the higher visual areas.

## 2. What is in the file

| | |
|---|---|
| Recordings (sessions) | **89** |
| Mice (subjects) | **19** |
| Trials (corridor traversals) | **37,801** |
| Timepoints | **815,506** |
| Neurons | **4,105,393** (17,363 – 78,815 per session) |
| Brain regions | V1, mHV (medial HVAs), lHV (lateral HVAs), aHV (anterior HVAs) |
| Time bin | **314.85 ms** (the native imaging frame, 3.18 Hz) |
| Alignment | trial start = **corridor entry** (`off_start = 0`) |
| File size | 150.8 GB |

Only the timepoints the paper analyses are kept: frames **inside the 0–4 m texture area**
while the **virtual reality is moving** (i.e. the mouse is running), and only neurons
assigned to a visual cortical area. Neural traces are **z-scored per neuron** over all
frames of their session.

## 3. Loading

```python
import pickle
with open('converted_data.pkl', 'rb') as f:      # needs ~155 GB of RAM
    data = pickle.load(f)

X = data['neural'][s][t]     # (n_neurons[s], T[s][t])  float32, z-scored
U = data['input'][s][t]      # (4, T[s][t])             float32
Y = data['output'][s][t]     # (4, T[s][t])             int64 class labels

data['subjects'][data['subject_idx'][s]]        # mouse name of session s
data['brain_regions'][data['brain_region_idx'][s][n]]   # region of neuron n
data['metadata']['session_info'][s]             # per-session provenance
```

## 4. Format specification

### `input` — decoder inputs, `input_names`, all rows length `T`

| i | name | type | description |
|---|------|------|-------------|
| 0 | `time_to_sound_cue_s` | continuous, time-varying | seconds until the sound cue; **positive before** the cue, negative after |
| 1 | `day_of_training` | continuous, per-trial | days since that mouse's first imaging session (0 – 92) |
| 2 | `time_since_trial_start_s` | continuous, time-varying | seconds since corridor entry (real elapsed time; ≥ 0) |
| 3 | `reward_available` | discrete, per-trial | 1 if this corridor is the rewarded one for this mouse, else 0 |

### `output` — decoder targets, `output_names` / `output_values`

| i | name | classes | description |
|---|------|---------|-------------|
| 0 | `stimulus` | 7: `circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2` | canonical visual-stimulus category of the corridor (per-trial). Mouse-specific textures are mapped onto this scheme (rock1→circle1, rock2→circle2, wood1→leaf1, wood2→leaf2, wood5→leaf3) |
| 1 | `licking` | 2: `no_lick, lick` | 1 if ≥ 1 lick occurred in that imaging frame |
| 2 | `position_bin` | 4: `0-1m, 1-2m, 2-3m, 3-4m` | 1-m bin of the corridor |
| 3 | `running_speed_bin` | 4: `Q1_slowest … Q4_fastest` | quartile of the running speed, edges from the pooled distribution of all kept timepoints (12.40, 25.28, 40.75) |

### Class distributions (all timepoints)

```
stimulus          circle1 .319  circle2 .060  leaf1 .335  leaf2 .170
                  leaf3 .059  leaf1_swap1 .027  leaf1_swap2 .029
licking           no_lick .963  lick .037
position_bin      .250 .249 .250 .252
running_speed_bin .250 .250 .250 .250
```

### `metadata`
`task_description`, `time_bin_size` (ms), `temporal_alignment_event`, `off_start`,
`off_end` (`None` — trial length is variable), `sampling_rate_hz`, `neural_signal`,
`frames_included`, `neurons_included`, `speed_bin_edges_cm_s`, `position_bin_edges_m`,
`stimulus_id_scheme`, `input_descriptions`, `source`, and `session_info` — a list of 89
dicts with `session_key`, `mouse`, `date`, `block`, `cohort`
(task / unsupervised / unsupervised_grating / naive), `exp_types`, `reward_mode`,
`day_of_training`, `n_neurons_all_rois`, `n_neurons_kept`, `n_frames_neural`,
`n_trials_raw`, `n_trials_kept`, `n_rewarded_trials`, `frame_interval_s`, `stimuli`.

## 5. Reproducing the conversion

```bash
python -u convert_data.py out.pkl --sample --show-processing   # 2 sessions + diagnostics
python -u convert_data.py out.pkl --full                       # all 89 (~8 min, 150.8 GB)
python -u train_decoder.py out.pkl --verify-only
python -u train_decoder.py out.pkl --plot-samples
```

`CONVERSION_NOTES.md` documents every decision and every validation;
`cache/` holds the exploration and verification scripts (see `cache/README_CACHE.md`).

## 6. Decoder performance

Balanced accuracy of the reference decoder (`train_decoder.py`, 100 PCs, 200 epochs,
80/20 trial split within every session; `train_decoder_full_out.txt`):

| Output | Chance | Train | **Validation** | Val / chance |
|---|---|---|---|---|
| `stimulus` | 0.143 | 0.769 | **0.704** | 4.93× |
| `licking` | 0.500 | 0.882 | **0.846** | 1.69× |
| `position_bin` | 0.250 | 0.692 | **0.686** | 2.75× |
| `running_speed_bin` | 0.250 | 0.541 | **0.530** | 2.12× |

The paper itself reports no decoding analysis, so there is no published accuracy to compare
against. Instead the conversion is validated against the paper's own results: the Fig. 1j
percentages of stimulus-selective neurons per visual region before and after learning are
reproduced from exactly the frames this conversion keeps (task medial HVAs 4.2 % → 18.0 %,
unsupervised 5.1 % → 12.9 %, V1 unchanged, anterior HVAs rising only in task mice), as is the
Fig. 1c,d anticipatory-licking result.

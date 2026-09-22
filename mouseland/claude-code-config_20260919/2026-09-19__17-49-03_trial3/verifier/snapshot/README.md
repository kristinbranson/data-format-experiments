# Zhong et al. 2025 — decoder-ready dataset

Converted form of the two-photon mesoscope dataset from **"Unsupervised pretraining in
biological neural networks"** (Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer &
Pachitariu), Figshare DOI [10.25378/janelia.28811129.v1](https://doi.org/10.25378/janelia.28811129.v1).

- Converted file: **`/app/converted_data.pkl`** (6.62 GB)
- 2-session sample: **`/app/sample_data.pkl`** (0.13 GB)
- Conversion script: **`/app/convert_data.py`**
- Full record of decisions and validation: **`/app/CONVERSION_NOTES.md`**

---

## 1. Dataset description

19 mice expressing GCaMP6s in excitatory neurons ran head-fixed on an air-floating ball
through 4-m virtual-reality corridors separated by 2 m of grey space. Each corridor's walls
were covered with frozen crops of one of two naturalistic texture categories (denoted
*leaf* and *circle*; *rock* and *brick/wood* in some mice). The VR advanced at a constant
60 cm s⁻¹ whenever the mouse ran faster than 6 cm s⁻¹ and was stationary otherwise. A sound
cue was played at a random position (0.5–3.5 m) in every trial; for the *task* cohort it
marked the start of the reward zone in the rewarded corridor only. Neural activity was
recorded from tens of thousands of neurons across V1 and higher visual areas simultaneously
with a two-photon mesoscope and deconvolved with Suite2p.

Three cohorts are present: **task** (water-rewarded, 5 mice / 28 sessions),
**unsupervised** (same corridors, never rewarded) and **naive** (first exposure), plus the
grating-exposure control mice.

## 2. Key statistics

| | |
|---|---|
| Sessions (recordings) | **89** |
| Subjects (mice) | **19** (1–8 sessions each) |
| Trials | **38,110** (84–789 per session, mean 428) |
| Timepoints (time bins) | **821,558** (mean 22.3 per trial, median 21) |
| Time bin | **314.69 ms** (native imaging frame, 3.178 Hz; 3.171–3.181 Hz across sessions) |
| Neurons recorded | 4,691,034 total; **20,547 – 89,577 per recording** (matches the paper) |
| Neurons in the converted file | **2,000 per session** (178,000 total) |
| Brain regions | V1 (78,082), mHV (50,617), lHV (20,720), aHV (28,581) |
| Rewarded-corridor trials | 4,446 (11.7 % overall; ≈48 % within the 28 task sessions) |
| Sessions containing licks | 28 (the task cohort only) |

Output class distributions (fraction of timepoints):

| Output | Distribution |
|---|---|
| stimulus | circle1 .317, circle2 .060, leaf1 .333, leaf2 .169, leaf3 .058, leaf1_swap1 .027, leaf1_swap2 .029, circle3 .007 |
| licking | no_lick .963, lick .037 |
| position_bin | .250 / .249 / .250 / .252 |
| running_speed_bin | .250 / .250 / .250 / .250 |

## 3. How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 5
X = data['neural'][session][trial]    # (2000 neurons, T bins), float32, z-scored
U = data['input'][session][trial]     # (4, T) float32
Y = data['output'][session][trial]    # (4, T) int64, class indices

print(data['subjects'][data['subject_idx'][session]])            # mouse name
print(data['brain_regions'])                                     # ['V1','mHV','lHV','aHV']
print(data['brain_region_idx'][session].shape)                   # (2000,)
print(data['metadata']['session_info'][session])                 # per-session provenance
```

Train / evaluate the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl --verify-only   # format check + summary
python train_decoder.py /app/converted_data.pkl --plot-samples  # train + evaluate
```

Reproduce the conversion (needs `/app/data`, ~90 s, 6 worker processes):

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## 4. Output format specification

```
data = {
  'neural':  [session][trial] -> float32 (n_neurons=2000, T)
  'input':   [session][trial] -> float32 (4, T)
  'output':  [session][trial] -> int64   (4, T)
  'subjects': list[str]                          # 19 mouse names
  'subject_idx': int64 (89,)                     # index into subjects, per session
  'brain_regions': ['V1', 'mHV', 'lHV', 'aHV']
  'brain_region_idx': [session] -> int64 (2000,)
  'input_names':  [...]
  'output_names': [...]
  'output_values': [ [names of each class], ... ]
  'metadata': {...}
}
```

### Inputs (`input_names`)

| i | name | type | definition |
|---|------|------|------------|
| 0 | `time_to_sound_cue_s` | continuous, time-varying | seconds until the sound cue (positive before, negative after), clipped to ±30 s |
| 1 | `day_of_training` | continuous, per-trial | days elapsed since that mouse's first recording (0–92) |
| 2 | `time_since_trial_start_s` | continuous, time-varying | seconds since corridor entry, clipped to 30 s |
| 3 | `reward_available` | discrete, per-trial | 1 if this corridor was the rewarded corridor for this mouse, else 0 |

Per-trial inputs are broadcast across the trial's time bins so every trial is a `(4, T)`
array.

### Outputs (`output_names`, `output_values`)

| i | name | classes | definition |
|---|------|---------|------------|
| 0 | `stimulus` | `circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, circle3` | canonical stimulus **role** of the corridor, following the reference code (role 2 = the trained/rewarded stimulus). Rock/brick mice map onto the same roles, as the paper does. Per trial. |
| 1 | `licking` | `no_lick, lick` | 1 if ≥1 lick was detected in that imaging frame. Time-varying. |
| 2 | `position_bin` | `0-1m, 1-2m, 2-3m, 3-4m` | position along the 4-m texture corridor in 4 equal 1-m bins. Time-varying. |
| 3 | `running_speed_bin` | `speed_q1 … speed_q4` | running-speed quartile; bin edges 12.42 / 25.35 / 40.85 cm s⁻¹ from the pooled distribution over all retained timepoints. Time-varying. |

### Metadata highlights

`task_description`, `time_bin_size` (314.69 ms), `temporal_alignment_event`
(`trial start = entry into the VR corridor`), `off_start = 0.0`, `off_end = None`
(trials end at the 4-m corridor exit and therefore have variable length),
`timepoint_selection`, `neural_signal`, `neural_subsampling`,
`position_bin_edges_m`, `running_speed_bin_edges_cm_s`, `time_input_clip_s`,
`frame_rate_hz`, `input_descriptions`, `output_descriptions`, and `session_info`
(one dict per session: id, mouse, date, block, cohort, day of training, experiment types,
neurons recorded / in visual cortex / kept, trials, timepoints, frame rate, stimulus map).

## 5. Processing summary

1. **Sessions** — the 142 `(recording × experiment type)` entries of
   `Imaging_Exp_info.npy` are de-duplicated to the 89 unique recordings; each recording's
   canonical stimulus map is merged over every experiment type it appears in.
2. **Timepoints** — kept if inside the 0–4 m texture corridor **and** the VR was moving
   (i.e. the mouse was running above the 6 cm s⁻¹ threshold), exactly the reference recipe
   `fr_valid = (ft_move > 0) & ft_CorrSpc` and the Methods statement "We only considered
   timepoints during running for analysis". 21 ambiguous corridor-boundary frames are
   dropped. No trials or sessions are excluded.
3. **Neurons** — Suite2p deconvolved traces, restricted to neurons assigned to a visual
   area (`iarea ∉ {−1, 7}`), then a seeded random subsample of 2,000 per session stratified
   over V1/mHV/lHV/aHV, then z-scored per neuron over the session's retained timepoints.
4. **Alignment** — every trial starts at corridor entry; each time bin is one imaging
   frame, so neural, input and output streams share the same frame index by construction.

The only deliberate departures from the reference pipeline are the neuron subsample, the
time-domain (rather than position-domain) binning, and the per-neuron z-score. Each is
required by the decoding task and is justified in `CONVERSION_NOTES.md` (Step 5).

## 6. Decoder performance

`python train_decoder.py /app/converted_data.pkl` (200 epochs, 100 PCs, balanced loss):

| Output | Chance | Training balanced acc. | Validation balanced acc. |
|---|---|---|---|
| stimulus | 0.125 | 0.998 | **0.873** |
| licking | 0.500 | 0.992 | **0.764** |
| position_bin | 0.250 | 1.000 | **0.894** |
| running_speed_bin | 0.250 | 0.937 | **0.570** |

The paper itself reports no decoding analyses, so these are not comparable to a published
number; all four outputs are well above chance.

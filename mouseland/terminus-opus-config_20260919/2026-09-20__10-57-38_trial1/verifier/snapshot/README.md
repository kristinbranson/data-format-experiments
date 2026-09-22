# Zhong et al. 2025 VR-corridor imaging data - decoder-ready conversion

## Dataset

Two-photon mesoscope recordings of visual cortex (V1 and higher visual areas) in head-fixed
mice running through 4 m long virtual-reality corridors whose walls carried naturalistic
texture patterns (leaf/circle, and rock/brick in some mice, plus spatially swapped versions
of the rewarded texture). A sound cue was played at a random position (0.5-3.5 m) in every
corridor; water-restricted "task" mice received water for licking after the cue in the
rewarded corridor. Unsupervised and naive mice ran the same corridors without water.

Source: Zhong, Baptista, Gattoni, Arnold, Flickinger, Stringer et al.,
*Unsupervised pretraining in biological neural networks* (`/app/paper.pdf`, code in `/app/code`).

| | |
|---|---|
| Sessions | 89 (matches the paper's "89 recordings") |
| Mice | 19 |
| Trials | 37,801 corridor traversals (mean 425/session) |
| Timepoints | 815,506 imaging frames |
| Time bin | 314.85 ms (3.17 Hz imaging frame rate) |
| Neurons recorded | 20,547-89,577 per session (matches the paper) |
| Neurons stored | 1,000 per session, sampled proportionally across V1/mHV/lHV/aHV |
| Neural signal | Suite2p deconvolved traces (0.75 s decay), z-scored per neuron |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]    # (1000, T)  z-scored deconvolved activity
inputs = data['input'][session][trial]     # (5, T)
outputs = data['output'][session][trial]   # (4, T)  categorical
```

## Format

- `neural[session][trial]`: float32 (n_neurons, T)
- `input[session][trial]`: float32 (5, T) - `time_to_sound_cue` (s, positive before the cue),
  `sound_cue_onset` (binary), `day_of_training` (days since that mouse's first session),
  `time_since_trial_start` (s), `reward_availability` (1 = rewarded corridor)
- `output[session][trial]`: int64 (4, T) - `stimulus_category` (0-6: circle1, circle2, leaf1,
  leaf2, leaf3, leaf1_swap1, leaf1_swap2), `licking` (0/1), `position_bin` (4 x 1 m bins over
  the 4 m corridor), `speed_bin` (global running-speed quartiles)
- `subjects` / `subject_idx`, `brain_regions` (`V1`, `mHV`, `lHV`, `aHV`) / `brain_region_idx`
- `metadata`: task description, time bin, alignment event, per-session `session_info`
  (mouse, date, cohort, experiment types, day of training, trial/neuron/frame counts),
  speed-bin edges, and descriptions of every input and output.

## Processing summary

- Trials are aligned to **corridor entry** and contain only frames inside the 4 m textured
  corridor while the mouse was running - the reference mask
  `fr_valid = (ft_move>0) & ft_CorrSpc` (utils.Get_dprime_selective_neuron), matching the
  paper's "we only considered timepoints during running".
- Neurons outside the visual cortex (`iarea in {-1,7}`) are excluded, as in the reference.
- Stimulus labels use the reference's canonical role-based ids (`beh['stim_id']`), merged
  across all experiment types in which a session appears.
- 309 `circle3` trials (4 sessions) are dropped because that stimulus has no canonical label.

## Key results (decoder, `train_decoder.py`)

| Output | Validation balanced accuracy | Chance |
|---|---|---|
| stimulus_category | 0.827 | 0.143 |
| licking | 0.812 | 0.500 |
| position_bin | 0.843 | 0.250 |
| speed_bin | 0.579 | 0.250 |

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~80 s with 8 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

See `/app/CONVERSION_NOTES.md` for the full decision log and validation report.

# Zhong et al. 2025 VR visual-discrimination dataset -- decoder-ready conversion

## Dataset description

Two-photon mesoscope recordings of mouse visual cortex during a virtual-reality (VR) visual
discrimination task (Zhong, Li et al., *Unsupervised pretraining in biological neural networks*,
Nature 2025; data: Figshare doi:10.25378/janelia.28811129.v1, local copy in `/app/data`).

Head-fixed mice run on an air-floating ball through 4 m VR corridors whose walls are frozen crops of
one of four naturalistic textures (**circle, leaf, rock, brick**), separated by 2 m of grey space.
The VR advances at a constant 60 cm/s whenever the mouse runs faster than 6 cm/s. A sound cue is
played at a random position (0.5-3.5 m) on every trial. In the **task (supervised)** cohort one
texture is rewarded: licking after the cue in the rewarded corridor delivers water. **Unsupervised**
and **naive** cohorts (and an **unsupervised-grating** control) run the same corridors without water
restriction or reward, and therefore never lick.

Converted dataset:

| | |
|---|---|
| Sessions | 89 (all recordings of the paper) |
| Mice | 19 |
| Trials | 38,110 (none dropped) |
| Timepoints | 821,579 |
| Neurons recorded | 20,547-89,577 per session (4.69 M total) |
| Neurons in the file | 2,000 per session (178,000 total), stratified across V1/mHV/lHV/aHV |
| Neural signal | suite2p deconvolved fluorescence (non-negative deconvolution, tau = 0.75 s) |
| Time bin | 314.7 ms (imaging frame, fs = 3.18 Hz); no re-binning |
| Alignment | trial start = entry into the VR corridor (`beh['StartFr']`), `off_start = 0` |
| Timepoints kept | frames inside the 4 m texture corridor **while the VR was moving** (the mouse was running), i.e. the reference mask `(ft_move>0) & ft_CorrSpc` |
| Trial length | variable, median 21 frames (~6.6 s = 4 m / 0.6 m/s) |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['neural'][0][0]      # session 0, trial 0: (2000 neurons, T frames), float32
U = data['input'][0][0]       # (4, T) float32   -- decoder inputs
Y = data['output'][0][0]      # (4, T) int64     -- decoder outputs (class indices)
mouse = data['subjects'][data['subject_idx'][0]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][0]]
info = data['metadata']['session_info'][0]   # ids, cohort, day, neuron/trial counts
```

Train/validate the reference decoder:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Re-create the dataset from the raw files:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~1 min, 6.6 GB
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

`data` is a dict with the fields required by `/app/decoder.py`:

- `neural`: list of 89 sessions; `neural[s][t]` is `(n_neurons, T)` float32 deconvolved activity.
- `input`: `input[s][t]` is `(4, T)` float32, `input_names` =
  1. `time_to_sound_cue` (s, positive before the cue, 0 at the cue frame)
  2. `day_of_training` (calendar days since the mouse's first imaging session; constant within a trial)
  3. `time_since_trial_start` (s since corridor entry)
  4. `reward_available` (1 = rewarded corridor, 0 otherwise; constant within a trial)
- `output`: `output[s][t]` is `(4, T)` int64 class indices, `output_names` / `output_values` =
  1. `stimulus_category`: `['circle', 'leaf', 'rock', 'brick']` (per trial, broadcast over time)
  2. `licking`: `['no lick', 'lick']` (1 if a lick occurred in that frame)
  3. `position_bin`: `['0-1m', '1-2m', '2-3m', '3-4m']` (4 equal 1-m bins of the corridor)
  4. `speed_bin`: `['Q1 (slowest 25%)', 'Q2', 'Q3', 'Q4 (fastest 25%)']` (global running-speed quartiles,
     edges 12.42 / 25.35 / 40.85 cm/s)
- `subjects` (19 names), `subject_idx` (89,), `brain_regions` = `['V1','mHV','lHV','aHV']`,
  `brain_region_idx[s]` (n_neurons,)
- `metadata`: `task_description`, `time_bin_size` (314.70 ms), `temporal_alignment_event`,
  `off_start` (0.0), `off_end` (None -- trials have variable length), plus `frame_rate_hz`,
  `neural_signal`, `neuron_selection`, `speed_bin_edges_cm_s`, `position_bin_edges_m`, `trial_definition`,
  `input_units`, `source_paper`, `source_data`, and `session_info` (per-session ids, mouse, date, cohort,
  exp_types, training day, neuron/trial counts, reward mode).

## Key statistics

Output class fractions (over all timepoints): stimulus circle 0.311 / leaf 0.470 / rock 0.085 / brick 0.135;
licking 0.037 (0.058-0.391 within the 28 rewarded sessions, exactly 0 in the other 61);
position 0.250 / 0.249 / 0.250 / 0.252; speed 0.25 each by construction.

Reference-decoder validation balanced accuracy (see `/app/train_decoder_full_out.txt`):

| Output | Chance | Validation |
|--------|--------|-----------|
| stimulus_category | 0.250 | **0.970** |
| licking | 0.500 | **0.767** |
| position_bin | 0.250 | **0.872** |
| speed_bin | 0.250 | **0.620** |

See `/app/CONVERSION_NOTES.md` for every processing decision, the comparison with the reference code and
paper, and the raw-data sanity checks (130 checks, 0 failures).

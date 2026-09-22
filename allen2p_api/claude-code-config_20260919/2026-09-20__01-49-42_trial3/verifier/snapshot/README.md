# Visual Behavior 2P → neural-decoder dataset

`/app/converted_data.pkl` (8.73 GB) is the Allen Brain Observatory **Visual Behavior 2P** dataset
reformatted for the decoder in `train_decoder.py`. It was produced by `/app/convert_data.py`, which
reads the AllenSDK cache in `/app/data` through `VisualBehaviorOphysProjectCache` (the NWB files are
never opened directly).

Full record of every decision, check and validation result: **`CONVERSION_NOTES.md`**.

---

## 1. Dataset description

Head-fixed mice perform a go/no-go **visual change-detection task** while two-photon calcium imaging
is performed in primary visual cortex (VISp). Natural images are flashed for 250 ms every 750 ms
(500 ms grey inter-stimulus interval; 5 % of repeats are omitted). The mouse earns water by licking
within 150–750 ms of a change in image identity. Running speed and pupil are recorded throughout.

Source: `visual-behavior-ophys-1.1.0`, `project_code == "VisualBehavior"` (the single-plane dataset
variant named in the Allen technical whitepaper), **active-behaviour sessions only**.

| | |
|---|---|
| Sessions (imaging planes) | **165** |
| Mice | **37** |
| Neurons | **28,821** (mean 174.7 / session, range 6–666) |
| Trials (go + catch) | **42,470** (37,143 go, 5,327 catch → 12.54 % catch) |
| Trials / session | mean 257.4 (39–409) |
| Timepoints | 11,192,974 (mean 262 per trial, range 217–389) |
| Time bin | **32.319 ms** (the 31 Hz two-photon frame interval) |
| Brain region | VISp |
| Cre lines | Slc17a7 (excitatory), Sst, Vip |
| Neural signal | detrended **dF/F** (`ds.dff_traces.dff`), Allen pipeline output |

Curation applied: `project_code == "VisualBehavior"`; passive-viewing sessions excluded (retracted
lick spout → no trial outcomes); 3 sessions without eye tracking excluded; trials restricted to
`go | catch` (which in the AllenSDK already excludes aborted and auto-rewarded trials); **no
additional neuron filtering** — the Allen pipeline's ROI filtering is already applied upstream.

## 2. How to load and use

```python
import pickle, numpy as np

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

n_sessions = len(data['neural'])                 # 165
X = data['neural'][0][5]                         # session 0, trial 5 -> (n_neurons, T) float32
Y = data['output'][0][5]                         # (5, T) int64 class labels
U = data['input'][0][5]                          # (0, T) float32 -- this task has no decoder inputs

mouse   = data['subjects'][data['subject_idx'][0]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][0]]

# decode the labels
for d, name in enumerate(data['output_names']):
    print(name, [data['output_values'][d][v] for v in Y[d, :5]])
```

Validate / train:

```bash
python train_decoder.py /app/converted_data.pkl --verify-only     # format + statistics
python train_decoder.py /app/converted_data.pkl --plot-samples    # train the decoder
```

Reproduce the conversion:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full                 # ~72 s, 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## 3. Output format specification

Top-level keys follow the task specification exactly.

| Key | Type | Meaning |
|---|---|---|
| `neural` | list[165] of list[n_trials] of `(n_neurons, T)` float32 | dF/F per cell per two-photon frame |
| `input` | list of list of `(0, T)` float32 | empty — the task specifies no decoder inputs |
| `output` | list of list of `(5, T)` int64 | the five decoded variables (below) |
| `subjects` | list[37] of str | mouse ids |
| `subject_idx` | int64 `(165,)` | index into `subjects` for each session |
| `brain_regions` | `['VISp']` | targeted structure |
| `brain_region_idx` | list[165] of int64 `(n_neurons,)` | region of each neuron |
| `input_names` | `[]` | |
| `output_names` | list[5] of str | |
| `output_values` | list[5] of list[str] | label names for each output's classes |
| `metadata` | dict | see below |

### Outputs

| # | Name | Classes | Definition |
|---|---|---|---|
| 0 | `image_identity` | 17: `grey` + 16 natural images (`im000`…`im106`) | the image on screen during its 250 ms presentation; `grey` during the inter-stimulus interval **and** during omitted flashes |
| 1 | `image_change` | 2: `no_change`, `change` | 1 for the 250 ms presentation of the image that constitutes a change (7–9 frames); 0 elsewhere. Catch trials contain no change (the sham change does not alter the image) |
| 2 | `running_speed_bin` | 5: `run_q1`…`run_q5` | running speed (cm/s), interpolated onto the ophys frames, cut at the **global** quintiles `[0.043, 4.403, 22.636, 36.295]` |
| 3 | `pupil_diameter_bin` | 5: `pupil_q1`…`pupil_q5` | pupil diameter `2·√(pupil_area/π)` in pixels, blink-interpolated and resampled, cut at the global quintiles `[75.42, 84.94, 94.18, 107.17]` |
| 4 | `trial_outcome` | 4: `hit`, `miss`, `false_alarm`, `correct_reject` | static per trial, broadcast across the trial's timepoints |

Outputs 0–3 are time-varying; output 4 is constant within a trial (the format requires one array per
trial, so it is broadcast rather than stored as a separate 1-D array).

### Trials and time base

A trial is one row of the AllenSDK `trials` table with `go` or `catch` true, spanning
`[start_time, stop_time)` — the experiment's own trial definition. The change (or sham change)
occurs 2.25–8.25 s after trial start and the trial ends 4.24 s after it, so `T` varies
(217–389 frames) while the **bin size is constant** at the two-photon frame interval. Accordingly
`metadata['off_start'] = 0.0` (relative to trial start) and `metadata['off_end'] = None`.

All streams are resampled onto `ds.ophys_timestamps`, the common clock the Allen sync board provides
for imaging, stimulus, running and eye tracking.

### `metadata`

`task_description`, `time_bin_size` (ms), `temporal_alignment_event`, `off_start`, `off_end`,
`trial_definition`, `neural_signal`, `sampling_rate_hz`, `curation`,
`running_speed_bin_edges_cm_s`, `pupil_diameter_bin_edges_px`, `pupil_note`, `n_sessions`,
`n_subjects`, `n_neurons_total`, `n_trials_total`, `source`, and `session_info` — a list of 165 dicts
with `ophys_experiment_id`, `ophys_session_id`, `behavior_session_id`, `mouse_id`, `cre_line`,
`session_type`, `experience_level`, `image_set`, `targeted_structure`, `imaging_depth`,
`equipment_name`, `n_neurons`, `n_trials`, `n_go_trials`, `n_catch_trials`, `ophys_frame_rate_hz`,
`median_frame_interval_s` and `cell_specimen_ids`.

## 4. Key statistics and validation

Class distributions over all 11.19 M timepoints:

| Output | Distribution |
|---|---|
| `image_identity` | grey 0.669; each of the 16 images 0.020–0.021 |
| `image_change` | no_change 0.974, change 0.026 |
| `running_speed_bin` | 0.200 / 0.200 / 0.200 / 0.200 / 0.200 |
| `pupil_diameter_bin` | 0.200 / 0.200 / 0.200 / 0.200 / 0.200 |
| `trial_outcome` | hit 0.316, miss 0.559, false_alarm 0.018, correct_reject 0.107 |

Decoder performance (`train_decoder.py`, 200 epochs, balanced loss, 100 PCs):

| Output | Chance | Training balanced acc | **Validation balanced acc** |
|---|---|---|---|
| image_identity | 0.059 | 0.428 | **0.400** |
| image_change | 0.500 | 0.669 | **0.623** |
| running_speed_bin | 0.200 | 0.382 | **0.363** |
| pupil_diameter_bin | 0.200 | 0.466 | **0.444** |
| trial_outcome | 0.250 | 0.437 | **0.295** |

Independent validation (see `CONVERSION_NOTES.md` Steps 10 and 12):
- every converted trial's neural matrix matches a freshly loaded `dff_traces` slice (`np.allclose`);
- every output was rebuilt frame-by-frame from the SDK tables and agreed on 100 % of frames;
- per-session trial, neuron and hit/miss/FA/CR counts match `behavior_session_table.csv` and
  `ophys_cells_table.csv` for all 165 sessions;
- re-implementing the reference paper's own decoders on the converted data reproduces its reported
  accuracies (change vs. repeat: 0.80 excitatory / 0.59 Vip; hit vs. miss: 0.77).

## 5. Files

| File | What it is |
|---|---|
| `converted_data.pkl` | the full converted dataset (165 sessions) |
| `sample_data.pkl` | 2-session sample produced by `--sample` |
| `convert_data.py` | the conversion script |
| `CONVERSION_NOTES.md` | full decision/validation record (Steps 0–13) |
| `conversion_full_out.txt`, `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt`, `verification_sample_out.txt` | `--verify-only` logs |
| `train_decoder_full_out.txt`, `train_decoder_sample_out.txt` | decoder training logs |
| `processing_<oeid>.png` | step-by-step processing verification plots |
| `sample_trials.png`, `predictions.png` | plots written by `train_decoder.py` |
| `cache/` | investigation scripts and their outputs (see `cache/README_CACHE.md`) |

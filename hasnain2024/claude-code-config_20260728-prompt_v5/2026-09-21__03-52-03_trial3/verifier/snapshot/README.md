# Neural Decoder Data Conversion

Converts electrophysiology and behavioral data from "Separating cognitive and motor processes in the behaving mouse" (Hasnain, Birnbaum et al., Nature Neuroscience 2024) into a Python dictionary format suitable for neural decoder training.

## Source Data

- **Paper**: Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the behaving mouse", Nature Neuroscience (2024)
- **Brain regions**: ALM (anterior lateral motor cortex), tjM1 (tongue/jaw motor cortex)
- **Task**: Head-fixed mice performed two directional licking tasks (Delayed-Response and Water-Cued) that alternated block-wise within sessions
- **Recording**: Extracellular recordings with high-density silicon probes (H2, Neuropixels 1.0)

## Output Format

The converted data is a Python dictionary (pickle) with the following structure:

| Key | Type | Description |
|-----|------|-------------|
| `neural` | list of arrays | Per-session neural data, each `(trials, neurons, 500)` — smoothed firing rates in Hz |
| `input` | list of arrays | Per-session input, each `(trials, 1, 500)` — time from go cue in seconds |
| `output` | list of arrays | Per-session output, each `(trials, 6, 500)` — categorical labels |
| `subjects` | list of str | Unique subject IDs |
| `subject_idx` | list of int | Subject index per session |
| `brain_regions` | list of str | `['ALM', 'tjM1']` |
| `brain_region_idx` | list of arrays | Per-session neuron-to-region mapping |
| `input_names` | list of str | `['time_from_go_cue']` |
| `output_names` | list of str | Names of the 6 output dimensions |
| `output_values` | list of lists | Category labels per output dimension |
| `metadata` | dict | Processing parameters and session info |

### Output Dimensions

| Dim | Name | Values | Description |
|-----|------|--------|-------------|
| 0 | lick_direction | left, right, none | Direction of instructed lick (none = ignore trial) |
| 1 | behavioral_context | WC, DR | Water-Cued vs Delayed-Response block |
| 2 | outcome | incorrect, correct, ignore | Trial outcome |
| 3 | tongue_velocity | below_50pct, above_50pct, not_visible | Discretized tongue speed from DLC |
| 4 | paw_velocity | below_50pct, above_50pct, not_visible | Discretized paw speed from DLC |
| 5 | motion_energy | below_50pct, above_50pct | Discretized video motion energy |

## Processing Pipeline

1. **Spike binning**: 10 ms bins, aligned to go cue, -2.5 to +2.5 s window
2. **Smoothing**: Causal Gaussian kernel (15-point window, reflect boundary)
3. **Unit filtering**: Exclude garbage/noisy quality labels, require >1 Hz mean firing rate
4. **Session filtering**: Require >=40 right-hit AND >=40 left-hit DR trials, >=10 neurons
5. **Behavioral variables**: Discretized at per-session 50th percentile thresholds
6. **DLC features**: Tongue and paw velocity from DeepLabCut tracking, with visibility detection

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full dataset (42 sessions, 14 subjects, 2354 neurons) |
| `sample_data.pkl` | Sample dataset (2 sessions) for quick testing |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder model and utilities |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and statistics |

## Usage

```bash
# Convert sample data (2 sessions)
python convert_data.py sample_data.pkl

# Convert full dataset (42 sessions)
python convert_data.py converted_data.pkl --full

# Verify data format
python train_decoder.py converted_data.pkl --verify-only --plot-samples

# Train decoder
python train_decoder.py converted_data.pkl --plot-samples --stats-json stats.json
```

## Decoder Results (Full Dataset)

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| lick_direction | 0.626 | 0.333 |
| behavioral_context | 0.850 | 0.500 |
| outcome | 0.610 | 0.333 |
| tongue_velocity | 0.550 | 0.333 |
| paw_velocity | 0.551 | 0.333 |
| motion_energy | 0.719 | 0.333 |

All outputs decode well above chance with minimal train/validation gap (~1-2%), indicating robust neural encoding of task variables, behavioral context, and movement-related signals in ALM.

## Dataset Statistics

- **42 sessions** from **14 mice** (2 skipped for insufficient trial counts)
- **2354 neurons** (2209 ALM, 145 tjM1)
- **14231 trials** total
- **500 time bins** per trial (10 ms resolution, 5 s window)
- Correct trial rate: ~75.5%, Ignore rate: ~12.0%
- DR context: ~90.9%, WC context: ~9.1%

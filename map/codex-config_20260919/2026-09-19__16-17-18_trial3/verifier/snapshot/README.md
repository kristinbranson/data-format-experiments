# Neural Decoder Dataset

`converted_data.pkl` contains the complete decoder-ready version of the brain-wide auditory delayed-response dataset. Neural activity is aligned to go-cue onset from −2.5 to +1.5 seconds in 80 adjacent 50-ms bins.

## Dataset summary

| Item | Value |
|------|-------|
| Subjects | 28 mice |
| Sessions | 173 |
| Trials | 89,068 |
| Curated neurons | 69,453 |
| Neurons/session | mean 401.46; median 390; range 90–923 |
| Trials/session | mean 514.84; median 517; range 159–796 |
| Brain-region labels | 293 exact Allen CCF annotations |
| Full pickle size | 10.826 GiB |

Units use the NWB classifier's `good` label. Trials are mapped to the exact ephys observation block, and trials invalid for any retained unit, auto/free-water trials, and two no-coverage boundary trials are excluded. Early-lick, ignore, and photostimulation trials are retained because they are required decoder variables.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

session = 0
trial = 0
neural = data["neural"][session][trial]  # (n_neurons, 80), float32 Hz
inputs = data["input"][session][trial]   # (2, 80), float32
outputs = data["output"][session][trial] # (4, 80), int8
```

Loading the complete pickle requires enough RAM for an approximately 10.8-GiB serialized object. `sample_data.pkl` provides the same structure for two sessions.

## Variables

Inputs are `time from tone onset (s)` and `photostimulation on`. Time from tone is evaluated at each bin center; stimulation is binary and derived from NWB start/stop intervals.

Outputs are:

| Row | Variable | Codes |
|-----|----------|-------|
| 0 | Lick direction choice | 0 left, 1 right, 2 no lick |
| 1 | Outcome | 0 ignore, 1 miss, 2 hit |
| 2 | Early lick | 0 no, 1 yes |
| 3 | Tongue y-position | 0 below session q40, 1 q40–q60, 2 above q60, 3 not visible |

Choice, outcome, and early lick are trial labels repeated over the 80 time points. Tongue position is time-varying. Side-camera tongue frames with DLC likelihood below 0.9 or no nearby preceding frame are not visible; >5-SD 2D velocity artifacts are interpolated before per-session quantiles are calculated.

`subjects`/`subject_idx` identify the mouse for each session. `brain_regions`/`brain_region_idx` identify every neuron's exact CCF annotation. `metadata.session_info` contains source-file provenance, retained source trial rows, filtering counts, tongue thresholds, and timing details.

## Reproduction and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The full verifier reports no errors or warnings. Held-out balanced accuracies are 0.6950 choice, 0.6627 outcome, 0.7567 early lick, and 0.6252 tongue position. Detailed decisions, reference comparisons, raw-data sanity checks, and validation results are in `CONVERSION_NOTES.md`.


# IBL Neural Decoder Dataset

`converted_data.pkl` is a stimulus-onset-aligned conversion of the staged IBL brain-wide-map electrophysiology release for neural decoding. Scientific data were loaded only through `one.api.One` and brainbox loaders.

## Contents

- 441 sessions from 135 mice
- 186,844 retained trials
- 594,965 session-neurons (all Kilosort clusters, matching the decoder reference code's `qc=None`)
- 100 time bins per trial, 20 ms each, from -0.5 to +1.5 s around visual stimulus onset
- Decoder inputs: time since stimulus onset and zero-based trial number within block
- Decoder outputs: choice, prior-left class, wheel-speed tertile, and whisker-motion-energy tertile

Eighteen of 459 candidate sessions were excluded because a mandatory source variable was absent or unusable: 14 lacked both whisker motion-energy streams, two lacked `probabilityLeft`, one had a collapsed continuous signal, and one had no valid trials. Full rationale and checks are in `CONVERSION_NOTES.md`.

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as stream:
    data = pickle.load(stream)
```

The pickle is approximately 98 GiB, so loading it requires substantial RAM. `sample_data.pkl` provides the identical schema for two sessions.

The top-level fields are `neural`, `input`, `output`, `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, and `metadata`. Each `neural[session][trial]` array is neuron × time. Inputs are 2 × time and outputs are 4 × time; static choice and prior labels are broadcast across time.

## Reproducing

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The complete decoder reached held-out balanced accuracies of 0.566 choice, 0.596 prior, 0.584 wheel speed, and 0.572 whisker energy. See `train_decoder_full_out.txt` for the complete run.


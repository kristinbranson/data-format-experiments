# IBL Brain-Wide Map Decoder Dataset

This directory contains a stimulus-onset-aligned conversion of the frozen IBL
brain-wide map release for neural decoding. The source freeze is defined by
`code/code_zhang2025/data/bwm_release.csv`; processing follows the bundled Zhang et
al. decoder cache wherever compatible with the requested inputs and categorical
outputs.

## Main files

- `converted_data.pkl`: complete converted dataset (98.796 GiB)
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible conversion program
- `CONVERSION_NOTES.md`: decisions, source reconciliation, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: complete logs
- `processing_<eid>.png`, `sample_trials.png`, `predictions.png`: visual audits

## Dataset summary

| Statistic | Value |
|-----------|------:|
| Sessions | 444 |
| Subjects | 136 |
| Trials | 188,925 |
| Session-specific neuron sum | 599,865 |
| Mean neurons/session | 1,351.05 |
| Brain-region labels (Beryl mapping) | 281 |
| Time bins/trial | 100 |
| Bin width | 20 ms |
| Window/alignment | -0.5 to +1.5 s around visual stimulus onset |

The conversion starts from the exact paper freeze (139 mice, 459 sessions, 699
probes, 621,733 sorted units). Fourteen sessions lack a paired whisker-motion stream;
one additional session has fewer than two trials with complete wheel/whisker
coverage. Every exclusion and retained source trial index is in pickle metadata.

## Loading and structure

The complete pickle needs substantially more than 100 GiB of available RAM while
loading because it expands into many Python/NumPy objects.

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

x = data["neural"][0][0]   # neuron x time, float32 spike counts
u = data["input"][0][0]    # 2 x time, float32
y = data["output"][0][0]   # 4 x time, int8 categorical labels
```

Top-level fields are `neural`, `input`, `output`, `subjects`, `subject_idx`,
`brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`,
and `metadata`. Lists are ordered session then trial. Neurons are concatenated across
simultaneously recorded probes, and `brain_region_idx[session]` has one entry per
neuron.

Inputs, in row order:

1. Time since stimulus onset: bin-ending coordinates -0.48 through +1.50 s.
2. Trial number in block: zero-based position in the original probability-left run,
   repeated over time.

Outputs, in row order:

1. Choice: left = 0, right = 1, repeated over time.
2. Prior probability left: 0.2 = 0, 0.5 = 1, 0.8 = 2, repeated over time.
3. Absolute filtered wheel speed: global low/medium/high tertiles = 0/1/2.
4. Whisker motion energy: global low/medium/high tertiles = 0/1/2.

Exact tertile thresholds, trial filtering, source IDs/indices, camera view, release
inventory, and all session-level information are stored under `data["metadata"]`.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full conversion took 455.48 seconds with eight workers. Verification completed with
no errors. It warns about three all-zero neural trials in one session; independent
raw-file checks proved that the source spike recording ends before those final three
behavioral trials, so zeros are the faithful reference-compatible representation.

The completed full decoder achieved validation balanced accuracy 0.5743 (choice),
0.5753 (prior), 0.5792 (wheel), and 0.7208 (whisker), versus chance 0.5/0.333/0.333/
0.333. See `CONVERSION_NOTES.md` for the full critical review and paper comparison.


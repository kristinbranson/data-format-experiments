# Neural Decoder Conversion

This repository now includes a converted decoder-ready dataset in [`converted_data.pkl`](/app/converted_data.pkl) built from the imaging dataset used in *Unsupervised pretraining in biological neural networks*.

## Dataset summary

- Source dataset: 89 unique imaging recordings from 19 mice.
- Task alignment: trials are aligned to corridor entry.
- Neural signal: deconvolved two-photon activity loaded from the paper's saved `spks` arrays.
- Trial filter: only running frames inside the textured corridor are retained (`ft_CorrSpc & (ft_move > 0)`).
- Time bin size: median native imaging frame interval, `314.69 ms`.

## Exported decoder variables

- Inputs:
  - `time_to_sound_cue`
  - `day_of_training`
  - `time_since_trial_start`
  - `reward_available`
- Outputs:
  - `visual_stimulus_category`
  - `licking`
  - `position_bin`
  - `running_speed_bin`

## Important caveat

The converted dataset does **not** include all raw neurons. It exports a reference-style subset chosen to match the paper's analysis logic while keeping `train_decoder.py` tractable:

- mHV familiar-stimulus selective neurons from odd running corridor frames using top/bottom 5% `d'`.
- aHV reward-prediction neurons using early-vs-late cue `d' >= 0.3`.

This yields 102,541 exported neurons across 89 sessions instead of the full raw-neuron count.

## Files

- [`convert_data.py`](/app/convert_data.py): conversion script.
- [`converted_data.pkl`](/app/converted_data.pkl): full converted dataset.
- [`sample_data.pkl`](/app/sample_data.pkl): 2-session sample dataset.
- [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md): full conversion log and validation record.
- [`train_decoder_full_out.txt`](/app/train_decoder_full_out.txt): full decoder training log.

## Usage

Inspect the pickle:

```bash
python3 - <<'PY'
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
print(data['metadata'])
PY
```

Run validation only:

```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train the decoder:

```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```

Rebuild the dataset:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

## Key converted statistics

- Sessions: 89
- Subjects: 19
- Trials: 38,110
- Exported neurons: 102,541
- Brain regions: `mHV`, `aHV`
- Validation balanced accuracy:
  - `visual_stimulus_category`: `0.8029`
  - `licking`: `0.8724`
  - `position_bin`: `0.6807`
  - `running_speed_bin`: `0.4361`

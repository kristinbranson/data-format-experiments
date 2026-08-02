# CA1 Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from hippocampal CA1 neurons recorded while mice freely explored 10 geometrically distinct environments. The data is from Lee et al. (2025) "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping" (Neuron 113(2): 307-320).

### Experiment
- 7 mice explored environments formed by partitioning a 75×75 cm square arena into a 3×3 grid
- 10 different geometric configurations were created by blocking different partitions
- Each mouse completed 2-3 sequences of all 10 geometries (plus square bookends)
- Sessions were 40 minutes long, recorded at 30 Hz with miniscope calcium imaging
- Neural data: binarized rising-phase calcium events from CA1 neurons

### Decoder Task
- **Input**: Environment geometry (3×3 binary matrix indicating which partitions are open)
- **Output**: Mouse position discretized into 3×3 spatial bins (9 categories)
- **Neural**: Smoothed and temporally binned calcium event rates

## Data Format

The converted dataset is saved as `converted_data.pkl`, a Python pickle file containing a dictionary with the following structure:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Keys

| Key | Type | Description |
|-----|------|-------------|
| `neural` | list of lists of arrays | Neural activity: `data['neural'][session][trial]` has shape `(n_neurons, 600)` |
| `input` | list of lists of arrays | Environment geometry: `data['input'][session][trial]` has shape `(9,)` |
| `output` | list of lists of arrays | Position bin: `data['output'][session][trial]` has shape `(1, 600)` |
| `subjects` | list of str | Animal IDs: 7 mice |
| `subject_idx` | array | Index into subjects for each session |
| `brain_regions` | list of str | `['CA1']` |
| `brain_region_idx` | list of arrays | All zeros (single brain region) |
| `input_names` | list of str | Names for each input dimension (env_00 through env_22) |
| `output_names` | list of str | `['position']` |
| `output_values` | list of lists | Position bin labels (top-left through bottom-right) |
| `metadata` | dict | Task description, time bin size, etc. |

### Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 207 |
| Trials | 8,187 (39-40 per session) |
| Subjects | 7 |
| Total registered cell-days | 69,744 |
| Neurons per session | 113-564 (mean: 337) |
| Time bins per trial | 600 |
| Time bin size | 100 ms |
| Trial duration | 60 seconds |
| Spatial bins | 3×3 = 9 |

### Neural Processing
1. Binary calcium events (0/1) are Gaussian-smoothed (σ=3 frames)
2. Temporally binned using average pooling (kernel=3 frames, stride=3)
3. Resulting in continuous firing rate estimates at 10 Hz (100 ms bins)

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset |
| `sample_data.pkl` | Sample dataset (2 sessions) for testing |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder library |
| `CONVERSION_NOTES.md` | Detailed conversion notes |

## Usage

### Convert data
```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing
```

### Train decoder
```bash
python -u train_decoder.py converted_data.pkl --plot-samples
python -u train_decoder.py sample_data.pkl --verify-only
```

### Decoder Performance
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| position | 0.695 | 0.607 | 0.111 |

## References

Lee, J.Q., Keinath, A.T., Cianfarano, E. & Brandon, M.P. (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron, 113(2), 307-320.

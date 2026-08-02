# Conversion Notes: georepca1 Dataset

## Source Dataset
Lee, Keinath, Cianfarano & Brandon (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2): 307-320.

## Data Summary
- **7 animals** (mice): QLAK-CA1-08, QLAK-CA1-30, QLAK-CA1-50, QLAK-CA1-51, QLAK-CA1-56, QLAK-CA1-74, QLAK-CA1-75
- **207 sessions** (recording days total across all animals)
- **5,413 unique neurons** tracked across sessions via CellReg
- **69,744 neuron-session observations** (rate maps)
- **10 distinct environment geometries** created by blocking partitions of a 3x3 grid in a 75x75 cm square arena
- **Recording**: 30 Hz calcium imaging with miniscopes, ~40 min sessions

## Decoder Task
Decode mouse position (discretized into 3x3 = 9 spatial bins) from CA1 neural activity, with environment geometry as decoder input.

## Conversion Decisions

### Sessions and Trials
- Each **session = one recording day** (one environment geometry per day)
- Sessions split into **1-minute trials** (1800 frames at 30 Hz)
- Animals with 31 days and ~71,866 frames/day yield 39 trials/session
- Animals with different frame counts (QLAK-CA1-51: 72,219 frames) yield 40 trials/session
- Remainder frames at end of session are discarded (< 1 min)
- Minimum 2 trials per session guaranteed (40 min session >> 2 min)

### Neural Data
- **Binary calcium transient events** from rise-phase extraction (z-score > 2.5 threshold)
- Values: 0 (no event) or 1 (significant transient event)
- Only **registered cells** (non-NaN traces) included per session
- No additional filtering applied - paper states "motivated the inclusion of all cells in subsequent analyses"
- Cell count varies per session due to cross-day registration (cells not always detected every day)

### Input (Decoder Input): Environment Geometry
- 3x3 binary matrix representing open (1) vs blocked (0) partitions
- Flattened to 9-element vector, static per trial
- Derived from `get_env_mat()` function in reference code
- Row-major ordering: grid_0_0, grid_0_1, grid_0_2, grid_1_0, ..., grid_2_2

### Output (Decoder Output): Mouse Position
- Continuous x-y position (0-75 cm) discretized into 3x3 = 9 spatial bins
- Each bin is 25 cm x 25 cm, matching the physical grid partitions
- Row-major indexing: bin = row * 3 + col (row0_col0=0, ..., row2_col2=8)
- Time-varying: one bin index per time step per trial

### Temporal Alignment
- All time series (neural, position) are natively aligned at 30 Hz recording rate
- DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz
- Time bin size: 33.33 ms (1000/30)
- Trials aligned to start of each 1-minute segment from session start

## Sanity Checks

### Matching Paper Statistics
- 207 sessions: **MATCHES** paper ("207 sessions")
- 69,744 neuron-session rate maps: **MATCHES** paper ("69,744 rate maps")
- 7 animals: **MATCHES** paper
- 10 environment geometries: **MATCHES** paper
- Mean cells per animal ~773: Paper reports "mean number of cells per animal = 773 +/- 68 SE"
  - Our data: unique cell counts per animal match total cells in dataset files (515, 875, 942, 554, 862, 713, 952)
  - Mean = (515+875+942+554+862+713+952)/7 = 773.3 **MATCHES**
- Minimum cells per animal = 515: **MATCHES** paper ("minimum cells per animal = 515")
- QLAK-CA1-51 has 21 sessions (2 sequences), others have 31 (3 sequences): **MATCHES** paper description of "up to three" repetitions

### Data Integrity
- Neural traces are binary (0/1 only): **VERIFIED**
- Position range [0, 75] cm: **VERIFIED**
- Environment geometries are consistent with blocked partition indices: **VERIFIED**
- All 10 environment shapes present across animals: **VERIFIED**

## Decoder Performance
- **Validation balanced accuracy: 54.9%** (chance = 11.1%)
- Approximately 5x above chance for 9-class classification
- Training balanced accuracy: 62.1%
- Position decoding from calcium imaging is well above chance, confirming spatial coding in CA1

## File Inventory
- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset (7 animals, 207 sessions)
- `sample_data.pkl` - Sample dataset (2 animals, 62 sessions)
- `conversion_full_out.txt` - Full conversion log
- `conversion_sample_out.txt` - Sample conversion log
- `verification_full_out.txt` - Full verification output
- `verification_sample_out.txt` - Sample verification output
- `train_decoder_full_out.txt` - Full decoder training log
- `train_decoder_sample_out.txt` - Sample decoder training log
- `README.md` - Dataset documentation
- `CONVERSION_NOTES.md` - This file

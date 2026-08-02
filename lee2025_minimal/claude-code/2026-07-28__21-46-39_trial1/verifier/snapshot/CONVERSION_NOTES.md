# Conversion Notes

## Data Source

Lee et al. (2025) "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping", Neuron 113, 307-320.

Data: Hippocampal CA1 calcium imaging in 7 mice navigating 10 geometrically distinct environments.

## Loading and Processing Decisions

### Data Loading
- Loaded from original MATLAB (.mat) files using `mat73.loadmat`
- Same loading approach as reference code (`load_dat` function in `utils.py`)
- 7 animals: QLAK-CA1-08, QLAK-CA1-30, QLAK-CA1-50, QLAK-CA1-51, QLAK-CA1-56, QLAK-CA1-74, QLAK-CA1-75

### Neural Data
- **Type**: Binary rising-phase calcium transients (0 or 1), matching the paper's description: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5, and 0 otherwise."
- **Sampling rate**: 30 Hz (original recording rate, matching paper)
- **Time bin size**: 33.33 ms (1000/30)
- **Neuron filtering**: Per session, only neurons with valid (non-NaN) data are included. NaN values indicate neurons not detected/tracked in that session (CellReg tracking across days). NaN neurons are excluded, and any remaining NaN values in valid neurons are set to 0.
- **Data type**: float32 for storage efficiency

### Session Structure
- Each recording session corresponds to one environment per day (40 min, as stated in methods: "All sessions were 40 min")
- Sessions are split into 1-minute trials as specified in the decoder task
- QLAK-CA1-08/30/50: 71,866 frames/session -> 39 complete 1-min trials (1800 frames each), 1266 frames unused
- QLAK-CA1-51: 72,219 frames/session -> 40 complete trials, 219 frames unused
- QLAK-CA1-56/74/75: ~72,060-72,091 frames/session -> 40 complete trials

### Decoder Input: Environment Geometry
- Represented as a 3x3 binary matrix (flattened to 9 values)
- 1 = accessible partition, 0 = blocked partition
- Matches the `get_env_mat` function in the reference code
- The paper states: "We partitioned an open square (75 x 75 cm) into a 3 x 3 grid space"
- Static per trial (same environment for all trials within a session)
- 10 unique environments: square (all 1s), o (center blocked), t, u, rectangle, +, i, l, bit donut, glenn

### Decoder Output: Position
- Mouse position (from DeepLabCut tracking) discretized into 3x3 = 9 spatial bins
- Binning: position values [0, ~75] divided into 3 equal bins per dimension
- Output values 0-8 correspond to row-major ordering of the 3x3 grid: bin_idx = row * 3 + col
- Time-varying at 30 Hz (same as neural data)
- The paper uses spatial binning for rate maps: "we generated rate maps" with spatial bins, consistent with our 3x3 discretization as specified in the task

### No Filtering Applied
- No velocity filtering or place cell filtering applied, as these are analysis-specific in the paper
- The paper's Bayesian decoder uses velocity filtering (v_thresh=5) and cell activity thresholds, but these are part of that specific analysis, not data preprocessing
- All valid neurons included for maximum information available to the decoder

## Sanity Checks and Validation

### Consistency with Paper Statistics
1. **Total unique neurons**: 5,413 - matches paper ("5,413 unique neurons")
2. **Total sessions**: 207 - matches paper ("207 sessions")
3. **Total rate maps**: 69,744 neuron-sessions - matches paper ("69,744 rate maps"), verified from brain_region distribution in verification output
4. **Number of animals**: 7 - matches paper
5. **Number of environments**: 10 - matches paper ("10 geometries")
6. **Session duration**: ~40 min at 30 Hz - matches paper ("All sessions were 40 min")
7. **Sessions per animal**: 31 for most, 21 for QLAK-CA1-51 (2 complete sequences + initial square vs 3 complete sequences + initial square for others)

### Neural Data Validation
- Trace data is binary (0/1) as expected from the paper's transient extraction
- Firing rate is very sparse (~0.2% nonzero) consistent with calcium imaging
- Valid neuron counts per session range from 113 to 564, consistent with calcium imaging populations

### Position Data Validation
- Position values range from 0 to ~75, consistent with the 75x75 cm arena
- All 9 position bins are represented in the full dataset
- Some bins have 0 occupancy in specific sessions (e.g., blocked regions), which is expected

### Decoder Performance
- Validation balanced accuracy: 0.5476 (chance = 0.1111)
- This is ~5x above chance, confirming neural data contains meaningful position information
- Consistent with the paper's demonstration of successful position decoding

## Output File Descriptions

- `converted_data.pkl`: Full dataset, 207 sessions, 8187 trials
- `sample_data.pkl`: Subset with first 2 sessions per animal (14 sessions, 554 trials)
- `conversion_full_out.txt`: Full conversion script output
- `conversion_sample_out.txt`: Sample data verification output
- `verification_full_out.txt`: Full dataset format verification output
- `verification_sample_out.txt`: Sample dataset format verification output
- `train_decoder_full_out.txt`: Full dataset decoder training output
- `train_decoder_sample_out.txt`: Sample dataset decoder training output

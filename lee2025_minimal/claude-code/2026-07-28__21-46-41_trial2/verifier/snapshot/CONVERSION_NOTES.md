# Conversion Notes: CA1 Geometric Deformation Dataset

## Source Paper
Lee, Keinath, Cianfarano & Brandon (2025). "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping." Neuron 113(2): 307-320.

## Dataset Overview
- 7 mice recorded with miniscope calcium imaging in hippocampal CA1
- 10 geometrically distinct environments created by blocking partitions of a 75x75 cm square arena (3x3 grid)
- Sequences of environments across days, starting and ending with the square
- Sessions: 40 minutes each at 30 Hz

## Verification Against Paper Statistics
- **Total unique neurons**: 5,413 (paper: 5,413) -- MATCH
- **Total sessions**: 207 (paper: 207) -- MATCH
- **Total rate maps**: 69,744 (paper: 69,744) -- MATCH
- **Animals**: 7 (QLAK-CA1-08 through QLAK-CA1-75)
- **Geometries**: 10 (square, o, t, u, rectangle, +, i, l, bit donut, glenn)

## Data Processing Decisions

### Neural Data
- **Source**: Binary calcium transient traces (`trace` field) - rise-extracted, binarized at z > 2.5 (per paper methods)
- **Cell filtering**: Only cells registered (non-NaN) on each day are included per session
- **Time binning**: Summed over 30 frames = 1-second bins, yielding event counts per bin
- No additional filtering (e.g., place cell selection) applied - all registered cells included, matching the decoding approach in the paper

### Trial Structure
- Each 40-minute session split into 1-minute trials
- QLAK-CA1-08 (71,866 frames): 39 trials per session (1,866 remainder frames dropped)
- QLAK-CA1-30 (71,866 frames): 39 trials per session
- QLAK-CA1-50 (71,866 frames): 39 trials per session
- QLAK-CA1-51 (72,219 frames): 40 trials per session (219 remainder frames dropped)
- QLAK-CA1-56 (72,091 frames): 40 trials per session (91 remainder frames dropped)
- QLAK-CA1-74 (72,060 frames): 40 trials per session (60 remainder frames dropped)
- QLAK-CA1-75 (72,071 frames): 40 trials per session (71 remainder frames dropped)
- Each trial: 1800 frames (60 seconds) -> 60 time bins (1 second each)

### Decoder Input: Environment Geometry
- 3x3 binary matrix representing accessible (1) vs blocked (0) partitions
- Flattened to 9 values, static per trial
- Derived from `get_env_mat()` function in reference code

### Decoder Output: Mouse Position
- Position (x, y) discretized into 3x3 = 9 spatial bins
- Arena: 75x75 cm -> each bin is ~25x25 cm
- Binning: floor(position / (max + epsilon) * 3) for each dimension
- Combined index: row * 3 + col (0-8)
- Time-varying: mode of 30 frames per 1-second time bin
- Output shape: (1, 60) per trial

### Session Structure
- Each day for each animal = one session
- Subject index maps session to animal
- All sessions included (no filtering by quality)

## Validation Results

### Format Verification
- Data format valid with no errors or warnings
- 207 sessions, 8,187 total trials
- Consistent time dimension (60 bins per trial)
- Neuron counts range: 113-564 per session

### Decoder Performance
- **Training balanced accuracy**: 89.8% (chance: 11.1%)
- **Validation balanced accuracy**: 63.4% (chance: 11.1%)
- Position decoding is 5.7x above chance on held-out data, confirming neural data encodes spatial information

### Sanity Checks
1. Total neuron count (5,413) matches paper exactly
2. Total session count (207) matches paper exactly
3. Total rate map count (69,744) matches paper exactly
4. Trace data is binary {0, 1} as expected from rise-extraction processing
5. Position ranges match arena size (75x75 cm)
6. Environment geometries match the 10 shapes described in the paper
7. Session sequences match paper description (geometric sequences starting/ending with square)
8. Decoder performance well above chance, confirming data quality

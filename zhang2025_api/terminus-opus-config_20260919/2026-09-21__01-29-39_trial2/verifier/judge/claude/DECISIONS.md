# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API (`ONE(base_url=..., mode='remote')`) to access the IBL cache. Session identifiers (eids) are read from `/app/code/code_zhang2025/data/bwm_release.csv` rather than discovered via `one.search`. For each eid, `SessionLoader` loads trials, wheel, and motion energy; `SpikeSortingLoader` loads spike sorting per probe. A new ONE instance is created per worker process.

ii.
```python
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

def get_one():
    from one.api import ONE
    return ONE(base_url=ONE_BASE_URL, mode='remote')

# In main():
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(bwm.eid.unique())

# In convert_session():
one = get_one()
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
```

iii. The AI chose to read the session list from the reference code's own CSV file (`bwm_release.csv`) rather than searching the ONE index. The CONVERSION_NOTES state this is the BWM release session list. This gives the same set of 459 sessions.

## 1-b. How are the data split into subjects?

i. Subject names are looked up from the `bwm_release.csv` file using a mapping from eid to subject. Subjects are sorted and indexed at assembly time.

ii.
```python
eid2subject = bwm.groupby('eid').subject.first().to_dict()
# ...
subjects = sorted({eid2subject[r['eid']] for r in ok})
subj_index = {s: i for i, s in enumerate(subjects)}
```

iii. The subject identity comes from the CSV rather than from the ONE API search results, but both resolve the same subject names.

## 1-c. How are the data split into sessions?

i. Each eid from `bwm_release.csv` is one session. Sessions are processed independently and assembled at the end.

ii.
```python
eids = list(bwm.eid.unique())
```

iii. Sessions are the natural unit of the BWM release; no splitting is required.

## 1-d. How are the data split into trials?

i. The trials table loaded by `SessionLoader.load_trials()` has one row per trial. No further splitting is needed.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
```

iii. Trials are already one row per trial in the IBL data format.

## 1-e. How are trials filtered based on quality controls?

i. The AI implements the reference code's `load_trials_and_mask` function as a pandas eval query. Trials are excluded if: reaction time (firstMovement_times - stimOn_times) is outside [0.08, 2.0] s; trial length (feedback_times - goCue_times) exceeds 10 s; any of 6 key columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType) is NaN; choice == 0 (no response). Additionally, trials must have full wheel and whisker motion energy coverage, be within the ephys recording window, and have at least one neuron fire.

ii.
```python
def compute_trials_mask(trials, min_rt=MIN_RT, max_rt=MAX_RT, max_trial_len=MAX_TRIAL_LEN,
                        nan_exclude=NAN_EXCLUDE, exclude_nochoice=True):
    query = f'(firstMovement_times - stimOn_times < {min_rt})'
    query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    return ~trials.eval(query)

# Additional coverage filter:
rec_t0, rec_t1 = max(probe_t0), min(probe_t1)
covered = (align_all + TIME_WINDOW[0] >= rec_t0) & (align_all + TIME_WINDOW[1] <= rec_t1)
mask = mask & covered

# Behavior coverage:
wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
good = wheel_valid & me_valid

# Zero-spike filter:
nonzero = binned_spikes.sum(axis=(1, 2)) > 0
```

iii. The AI closely follows the reference code's `load_trials_and_mask` defaults (min_rt=0.08, max_rt=2, max_trial_len=10, NaN exclusions, no-choice exclusion), plus adds additional data-quality filters (ephys coverage, behavior coverage, zero-spike trials).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader.load_spike_sorting()`, with cluster metadata (acronym, label) from `merge_clusters`.

ii.
```python
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
spike_times_l.append(spikes['times'])
spike_clu_l.append(spikes['clusters'] + offset)
```

iii. The same raw variables as the reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the [-0.5, 1.5] s trial window, giving spike counts per unit per bin. The counts are stored as float32 **without dividing by the bin width** (i.e., raw spike counts, not firing rates). When a session has multiple probes, their units are merged into one population with offset cluster IDs.

ii.
```python
def bin_spikes(spike_times, spike_clusters, n_clusters, align_times, ...):
    out = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    # ...
    b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
    np.clip(b, 0, n_bins - 1, out=b)
    idx = spike_clusters[s:e].astype(np.int64) * n_bins + b
    cnt = np.bincount(idx, minlength=n_clusters * n_bins)
    out[k] = cnt.reshape(n_clusters, n_bins)
    return out

# In convert_session — no division by BIN:
binned_spikes = bin_spikes(sp_t, sp_c, len(keep_ids), align)
neural.append(binned_spikes[k])
```

iii. The AI's metadata states `'neural_data_type': 'spike counts per 20 ms bin (not normalised)'`. This is a deliberate choice to store counts rather than firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated) are kept, AND clusters whose Beryl region is 'void' or 'root' are excluded. Sessions must have at least 5 such neurons (`MIN_NEURONS = 5`).

ii.
```python
QC_LABEL = 1.0
EXCLUDE_REGIONS = ('void', 'root')
MIN_NEURONS = 5

beryl = br.acronym2acronym(acronyms, mapping='Beryl')
keep = (labels >= QC_LABEL) & (~np.isin(beryl, EXCLUDE_REGIONS))
keep_ids = np.where(keep)[0]
if len(keep_ids) < MIN_NEURONS:
    return dict(eid=eid, skipped='too few good neurons')
```

iii. The AI follows the BWM data paper's criterion for well-isolated units (`label == 1`). It additionally excludes both 'void' (outside the brain) and 'root' (not mapped to a specific Beryl region) clusters, and requires at least 5 neurons per session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. Each trial window is [-0.5, 1.5] s around stimulus onset. Spike times are binned relative to `align_times + t_start`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_all = trials[ALIGN_TIME].to_numpy()
# ...
begs = align_times + t_start
b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
```

iii. Standard alignment to stimulus onset as specified in the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. Matches the reference code's `'binsize': 0.02` and the methods paper description of "20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the trial alignment event `stimOn_times` and the bin grid parameters. The input is the centre of each time bin.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)   # -0.49 ... 1.49
```

iii. The bin centres are computed from the window and bin size parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data; the input is defined analytically as the centre of each of the 100 time bins, from -0.49 to 1.49 s.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inputs.append(np.stack([bin_centres.astype(np.float32),
                        np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. This is a deterministic quantity defined by the binning grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centres used as the time input are the centres of the same bins used for spike counting, so they are aligned by construction.

ii. See 3-b above. The bin centres correspond to the midpoints of the bins used in `bin_spikes`.

iii. Both are derived from the same TIME_WINDOW and BINSIZE parameters.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`, which is constant within a block. Block boundaries are detected as changes in this value.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    newblock = np.ones(len(pl), dtype=bool)
    newblock[1:] = pl[1:] != pl[:-1]
    idx = np.zeros(len(pl), dtype=np.int64)
    c = 0
    for i in range(len(pl)):
        c = 0 if newblock[i] else c + 1
        idx[i] = c
    return idx
```

iii. The trials table has no explicit block identifier, so blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that resets to 0 at each block boundary (where `probabilityLeft` changes). Computed on the raw (unmasked) trial sequence so that filtering doesn't corrupt the count.

ii.
```python
tnb_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
# ... after masking:
tnb = tnb_all[idx_trials].astype(np.float32)
```

iii. Computing on the raw trial sequence before filtering ensures the count reflects the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which takes values +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice_raw = trials['choice'].to_numpy()[idx_trials]
choice_lab = (choice_raw < 0).astype(np.int64)   # left -> 0, right -> 1
```

iii. Uses the IBL convention where +1 is left and -1 is right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded to 0 (left, from +1) and 1 (right, from -1). No-response trials (choice == 0) are excluded by the trial mask. The value is broadcast as a constant over all 100 time bins.

ii.
```python
choice_lab = (choice_raw < 0).astype(np.int64)
# ...
outputs.append(np.stack([np.full(NBINS, choice_lab[k], dtype=np.int64), ...]))
```

iii. Simple recoding as specified in the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pleft_raw = trials['probabilityLeft'].to_numpy()[idx_trials]
prior_lab = np.select([pleft_raw == 0.2, pleft_raw == 0.5, pleft_raw == 0.8], [0, 1, 2],
                      default=-1).astype(np.int64)
```

iii. The three values correspond to the block types in the IBL task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped to categorical labels: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Trials with unexpected values get -1 and are filtered out. Broadcast as constant over all time bins.

ii. See 6-a above.

iii. Matches the mapping specified in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `SessionLoader.load_wheel()`, which provides wheel velocity. Speed is the absolute value of velocity.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].to_numpy()
wheel_speed_raw = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. Same as the reference code's `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1 kHz and differentiated with a Butterworth low-pass filter by `SessionLoader.load_wheel()`. The absolute value gives speed. This continuous trace is then linearly interpolated onto the trial bin grid using `np.linspace(beg + binsize, end, n_bins)` — i.e., at bin **right edges** rather than bin centres. Finally, discretized into 3 classes.

ii.
```python
def bin_behavior(times, values, align_times, ...):
    grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
    out[k] = np.interp(grid[k], tt, vv)

wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
```

iii. The AI explicitly follows the reference code's `get_behavior_per_interval` which uses `np.linspace(beg + binsize, end, n_bins)` as the interpolation grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 classes using per-session tertiles (33rd and 67th quantiles). Values at or below the 1/3 quantile are class 0, between 1/3 and 2/3 are class 1, above 2/3 are class 2.

ii.
```python
def discretize_tertiles(x, valid_rows):
    ref = x[valid_rows].ravel()
    q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
    lab = np.zeros(x.shape, dtype=np.int64)
    lab[x > q1] = 1
    lab[x > q2] = 2
    return lab, (float(q1), float(q2))
```

iii. Per-session tertiles ensure balanced classes despite different scales across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the trial bin grid at bin **right edges** (`np.linspace(beg + binsize, end, n_bins)`), which is offset by half a bin (10 ms) from the bin centres used for the time input and representing the neural data.

ii.
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
```

iii. The AI follows the reference Zhang et al. code's `get_behavior_per_interval` interpolation grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` (or `rightCamera` as fallback), loaded via `SessionLoader.load_motion_energy()`.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[key]
        me_times = df['times'].to_numpy()
        me_vals = df['whiskerMotionEnergy'].to_numpy()
        camera_used = view
        break
    except Exception:
        continue
```

iii. Left camera preferred, right as fallback, same as reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the bin grid at bin right edges (same grid as wheel speed), then discretized into 3 classes using per-session tertiles.

ii.
```python
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: per-session tertiles (1/3 and 2/3 quantiles), producing 3 approximately equal-sized classes.

ii. Same `discretize_tertiles` function as wheel speed (see 7-c).

iii. Ensures balanced classes across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at bin right edges, offset by half a bin from the neural bin centres.

ii. Same `bin_behavior` function (see 7-d).

iii. Follows the reference Zhang et al. code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of filtering: (1) Probes that fail to load are skipped. (2) Sessions without whisker motion energy are skipped. (3) Sessions with fewer than 5 good neurons or fewer than 2 usable trials are skipped. (4) Trials outside the ephys recording window are excluded. (5) Trials where wheel or whisker coverage is insufficient are excluded. (6) Trials with zero spikes across all neurons are excluded. (7) Trials with unexpected probabilityLeft values are excluded.

ii.
```python
# Missing whisker ME:
if me_times is None:
    return dict(eid=eid, skipped='no whisker motion energy')

# Ephys coverage:
rec_t0, rec_t1 = max(probe_t0), min(probe_t1)
covered = (align_all + TIME_WINDOW[0] >= rec_t0) & (align_all + TIME_WINDOW[1] <= rec_t1)

# Behavior coverage:
good = wheel_valid & me_valid

# Zero spikes:
nonzero = binned_spikes.sum(axis=(1, 2)) > 0
```

iii. The AI documents in CONVERSION_NOTES that these filters were added iteratively after discovering issues (e.g., all-zero neural trials from ephys recording ending before behavior).

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk, which involves reading large spike time and cluster arrays per probe (3-6 seconds per session).

ii.
```python
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The CONVERSION_NOTES timing breakdown shows spike loading dominates at 3-6s per session vs <1s for everything else.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function has a per-trial loop that could potentially be vectorized by offsetting spike indices across trials. The `bin_behavior` function similarly loops over trials. The `trial_number_in_block` function uses a Python for-loop that could use cumulative operations.

ii.
```python
# Per-trial spike binning loop:
for k in range(n_trials):
    s, e = i0[k], i1[k]
    b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
    # ...

# Per-trial behavior interpolation loop:
for k in range(n_trials):
    out[k] = np.interp(grid[k], tt, vv)
```

iii. The AI notes these per-trial loops but considers them fast enough (~0.01-0.03 s per session).

## 10-c. What processing does the code repeat multiple times?

i. A new `ONE` instance is created for each worker process (necessary for fork safety). Within a session, `SpikeSortingLoader` is called once per probe. No unnecessary repeated processing is evident.

ii.
```python
def _worker(args):
    eid, show = args
    return convert_session(eid, show_processing=show)

# In convert_session:
one = get_one()
```

iii. The per-process ONE instantiation is necessary, not wasteful.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads full spike arrays (including quality-filtered spikes) before filtering, meaning all spikes are read from disk even though only `label == 1` units are kept. The code also computes and stores detailed timing and quantile metadata in each session result dict that is not used by the decoder.

ii.
```python
# All spikes loaded, then filtered:
spike_times_l.append(spikes['times'])
spike_clu_l.append(spikes['clusters'] + offset)
# ...later filtered:
sel = np.isin(spike_clusters, keep_ids)
sp_t = spike_times[sel]
```

iii. Loading all spikes before filtering is unavoidable given the API, but the extra metadata tracking adds minor overhead.

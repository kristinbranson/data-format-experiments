# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the session list from `/app/code/code_zhang2025/data/bwm_release.csv` (the BWM data freeze CSV), which lists all 459 sessions with their eids, pids, probe names, subjects, and labs. For each session, it uses ONE in local mode (`ONE(cache_dir=..., mode='local')`) to load trials via `SessionLoader`, wheel via `sess_loader.load_wheel()`, motion energy via `sess_loader.load_motion_energy()`, and spike sorting via `SpikeSortingLoader`. The probe insertions per session are looked up from the CSV rather than `one.eid2pid()` (which requires network access). The AI also rebuilt the ONE cache tables from the filesystem to handle revision-folder mismatches.

ii.
```python
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
bwm = pd.read_csv(BWM_CSV, index_col=0)
eids = sorted(bwm.eid.unique())
# ...
tasks = [(eid, bwm[bwm.eid == eid][['pid', 'probe_name']].copy(), ...) for i, eid in enumerate(eids)]
```

```python
ONE_KWARGS = dict(cache_dir='/app/data/one_cache',
                  tables_dir='/app/cache/one_tables',
                  mode='local', silent=True)
_ONE = ONE(**ONE_KWARGS)
```

iii. The AI documented that `one.eid2pid()` requires a network connection not available in the environment, so it sourced probe information from `bwm_release.csv` instead (which is the same freeze file the reference code loads). The AI also rebuilt the ONE cache index tables to resolve revision-folder path mismatches.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. At assembly, unique subject names are sorted and each session is mapped to its subject via the CSV's eid-to-subject mapping.

ii.
```python
eid2subject = bwm.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
subjects = sorted({eid2subject[r['eid']] for r in ok})
subject_index = {s: i for i, s in enumerate(subjects)}
# ...
'subject_idx': np.array([subject_index[eid2subject[r['eid']]] for r in ok], dtype=np.int64),
```

iii. The CSV already contains the subject for each session, so no parsing is needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unique eids from `bwm_release.csv`. Each eid is processed independently.

ii.
```python
eids = sorted(bwm.eid.unique())
```

iii. Sessions are the natural unit of the BWM release; one eid = one session.

## 1-d. How are the data split into trials?

i. Trials are rows in the trials table loaded by `SessionLoader.load_trials()` (called internally by `load_trials_and_mask`).

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
```

iii. The trials table has one row per trial, so the split is given by the data.

## 1-e. How are trials filtered based on quality controls?

i. The AI imports the reference code's `load_trials_and_mask` function directly from `/app/code/code_zhang2025/src/utils/ibl_data_utils.py` with `max_trial_len=10.0`. This applies: reaction time (firstMovement_times - stimOn_times) in [0.08, 2.0] s; no NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType; feedback_times - goCue_times <= 10 s; choice != 0. Additionally, the AI requires complete wheel and whisker motion energy coverage of the 2s window (via `interp_behavior` validity checks).

ii.
```python
sys.path.insert(0, '/app/code/code_zhang2025/src')
from utils.ibl_data_utils import load_trials_and_mask

trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
# ...
keep = valid_wheel & valid_me
```

iii. The AI stated: "The reference trial mask is used directly by importing `load_trials_and_mask` from the reference code, so trial curation is guaranteed identical to the reference pipeline rather than re-implemented." The additional wheel/camera coverage check mirrors the reference `get_behavior_per_interval` validity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` and `spikes.clusters`, loaded via `SpikeSortingLoader.load_spike_sorting()`. The cluster table provides quality labels and anatomical locations for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. Spike times and cluster assignments are the raw data for building neural activity arrays.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the [-0.5, 1.5] s trial window using vectorized `searchsorted` + `np.add.at`. The raw spike counts are then z-scored per time bin: for each of the 100 time bins, a single mean and standard deviation is computed pooling over all neurons and all trials, and that scalar affine transform is applied to the entire (trials x neurons) block for that bin. This follows the reference `standardize_spike_data`. The statistics are accumulated in float64 before casting the result to float32. When a session has two probes, their units are merged (re-indexed and concatenated, sorted by spike time).

ii.
```python
spikes_binned = bin_spikes(spike_times, spike_clusters, n_neurons, align_times)
# ...
mu = spikes_binned.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
sd = spikes_binned.std(axis=(0, 1), keepdims=True, dtype=np.float64)
neural = ((spikes_binned.astype(np.float64) - mu)
          / np.where(sd > 0, sd, 1.0)).astype(np.float32)
```

```python
def bin_spikes(spike_times, spike_clusters, n_neurons, align_times):
    # ...
    bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
    np.clip(bin_idx, 0, N_BINS - 1, out=bin_idx)
    np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
    return out
```

iii. The AI documented: "The provided /app/decoder.py does NO normalisation of its own... so normalisation has to happen in the conversion. Per-neuron z-scoring is applied to the stored neural data." Later revised to per-time-bin standardization after discovering that per-neuron z-scoring destroys relative firing-rate differences, matching the reference `standardize_spike_data`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (IBL well-isolated neurons) are kept. Additionally, units whose Beryl acronym is `root` or `void` are dropped (defined as `NON_GREY = ('root', 'void')`). This removes units outside the brain (`void`) and units not mapped to a specific grey matter region (`root`).

ii.
```python
NON_GREY = ('root', 'void')
# ...
good = (clu['label'].values >= 1)
beryl = np.asarray(br.acronym2acronym(clu['acronym'].to_numpy(), mapping='Beryl'))
keep = good & ~np.isin(beryl, NON_GREY)
```

iii. The AI justified this as matching the BWM data paper: "Neurons were excluded if they failed one of the three criteria... Final analyses were additionally restricted to regions that were designated grey matter in the adult mouse Allen CCF." The AI explicitly notes this differs from the reference caching code which keeps all clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to stimulus onset (`stimOn_times`). Spike binning uses the window [stimOn - 0.5, stimOn + 1.5] for each trial, computed as `align_times + TIME_WINDOW`.

ii.
```python
ALIGN_EVENT = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
# ...
align_times = trials[ALIGN_EVENT].values[mask]
# ...
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
```

iii. Matches the reference code `align_time='stimOn_times'` and the Decoder Task instruction "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial. No rebinning or interpolation is applied to the neural data.

ii.
```python
BIN_SIZE = 0.02                # seconds
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))   # = 100
```

iii. Matches the reference code `binsize: 0.02` and the methods paper "each divided into 20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin grid itself. The bin centres are computed relative to the alignment event (stimOn_times), spanning -0.49 to +1.49 s in 0.02 s steps.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
time_axis = BIN_CENTRES.astype(np.float32)
# ...
arr[0] = time_axis
```

iii. The time input is defined by the binning grid, not by any raw data variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the bin centres from the window parameters. The values are the same for every trial: -0.49, -0.47, ..., +1.49.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
```

iii. N/A - the variable is defined by the analysis parameters.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input IS the neural binning grid: the spikes are binned on the same window edges, and the input is the centre of those bins. They are aligned by construction.

ii.
```python
# Neural binning uses the same window:
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
# Input is the centres of those bins:
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
```

iii. Aligned by construction.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table, which is constant within a block. A change in its value marks a new block boundary.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(pl), dtype=bool)
    new_block[1:] = pl[1:] != pl[:-1]
    block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(pl)), 0))
    return np.arange(len(pl)) - block_start
```

iii. The trials table has no explicit block identifier, so blocks must be inferred from where probabilityLeft changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected where `probabilityLeft` changes. The trial number is the 0-based index within its block, computed on the UNMASKED trials table before any trial filtering. This means dropped trials still count toward the position.

ii.
```python
# Computed before mask is applied:
tnb_all = trial_number_in_block(trials['probabilityLeft'].values)
# Then subsetted to kept trials:
tnb = tnb_all[mask]
# ...
tnb = tnb[keep]
# ...
arr[2] = tnb[k]
```

iii. The AI stated: "Computed on the *unmasked* trials table, because the position of a trial within its block is a property of the experiment and must not change when trials are dropped."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials['choice']`, which takes values +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice_raw = trials['choice'].values[mask]
```

iii. The choice column directly encodes the animal's reported stimulus side.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded from the ALF convention (+1 left, -1 right) to the decoder format (0 = left, 1 = right). No-choice trials (0) are excluded by the trial mask.

ii.
```python
choice = np.where(choice_raw > 0, 0, 1).astype(np.int8)
```

iii. The AI verified the ALF choice sign convention empirically: "On every one of the 459 sessions, 100% of CORRECT trials with the stimulus on the LEFT have choice == +1."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials['probabilityLeft']`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prob_left_raw = trials['probabilityLeft'].values[mask]
```

iii. The prior probability is directly available in the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped from continuous values to discrete categories: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_map[round(float(p), 4)] for p in prob_left_raw], dtype=np.int8)
```

iii. Follows the Decoder Task specification exactly: "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2."

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From the wheel position and timestamps (`_ibl_wheel.position` and `_ibl_wheel.timestamps`), loaded by `SessionLoader.load_wheel()` which interpolates position to 1 kHz and computes velocity with a Butterworth low-pass filter.

ii.
```python
sess_loader.load_wheel()
wheel_speed_raw, valid_wheel = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. The speed is the absolute value of the velocity computed by `SessionLoader`, following the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` interpolates wheel position onto a 1 kHz grid and differentiates with a 20 Hz Butterworth low-pass filter to produce velocity. The speed (absolute velocity) is then linearly interpolated onto the 100 bin RIGHT EDGES of each trial (matching the reference `get_behavior_per_interval` which evaluates on `np.linspace(t_beg + binsize, t_end, n_bins)`). Finally, it is discretized into 3 classes using per-session quantiles.

ii.
```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
# ...
def interp_behavior(beh_times, beh_values, align_times):
    from scipy.interpolate import interp1d
    # ...
    grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
    # ...
    f = interp1d(beh_times, beh_values, kind='linear', bounds_error=False,
                 fill_value=(beh_values[0], beh_values[-1]))
    vals[valid] = f(grid[valid])
    return vals, valid
```

iii. The AI documented that `BIN_RIGHT_EDGES` matches the reference `get_behavior_per_interval` which evaluates behaviour on `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tercile discretization: the 1/3 and 2/3 quantiles of all (trial, bin) values in the session define the thresholds. Values are classified as 0 (low), 1 (medium), or 2 (high) using `np.searchsorted`. Duplicate edges from zero-inflated signals are nudged apart with `np.nextafter`.

ii.
```python
def discretize(values, n_bins=N_DISCRETE_BINS):
    flat = values.ravel()
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(flat, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    labels = np.searchsorted(edges, flat, side='right').astype(np.int8)
    return labels.reshape(values.shape), edges
```

iii. The AI chose per-session terciles because wheel speed is in session-specific arbitrary units, so fixed global thresholds would not work. This ensures roughly balanced classes (~1/3 each) within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is evaluated at the bin right edges of each trial, using the same trial alignment event (stimOn_times) and window [-0.5, +1.5] s. This means the behavior value for bin k corresponds to time `stimOn + T_START + BIN_SIZE * (k + 1)` rather than the bin centre.

ii.
```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
```

iii. The AI documented this as matching the reference `get_behavior_per_interval` grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From the whisker-pad motion energy of the side camera: `leftCamera.ROIMotionEnergy` with `_ibl_leftCamera.times`, preferring the left camera and falling back to the right.

ii.
```python
for view in ('left', 'right'):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[view + 'Camera']
        me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                           me['whiskerMotionEnergy'].to_numpy(),
                                           align_times)
        me_view = view
        break
    except Exception:
        continue
```

iii. The left camera preference and right camera fallback follows the reference `bin_behaviors` logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the 100 bin right edges of each trial, then discretized into 3 classes using per-session quantiles, identical to the wheel speed processing.

ii.
```python
me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                   me['whiskerMotionEnergy'].to_numpy(),
                                   align_times)
# ...
me_lab, me_edges = discretize(me_raw)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same per-session tercile discretization as wheel speed: 1/3 and 2/3 quantiles of all (trial, bin) values, yielding categories 0 (low), 1 (medium), 2 (high).

ii.
```python
me_lab, me_edges = discretize(me_raw)
```

(Same `discretize` function as for wheel speed.)

iii. Same justification as wheel speed: per-session normalization handles arbitrary session-specific units.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at the bin right edges using the same trial window and alignment event.

ii.
```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
```

iii. Same alignment approach as wheel speed, matching the reference `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) The trial mask from `load_trials_and_mask` drops trials with NaN in key fields. (2) `interp_behavior` drops trials where the wheel or camera signal doesn't cover the full window or contains NaN. (3) Sessions with no whisker motion energy from either camera are skipped (14 sessions). (4) Sessions with no well-isolated grey-matter neurons are skipped. (5) Sessions with fewer than 2 surviving trials are skipped. (6) Probes with no spike sorting data are skipped. (7) Duplicate tercile edges from zero-inflated signals are handled with `np.nextafter`. (8) Silent neurons (zero variance) get 0 rather than NaN because the std denominator is clamped.

ii.
```python
if spikes is None or len(spikes) == 0 or 'times' not in spikes:
    continue
# ...
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return dict(eid=eid, skip='fewer than %d trials pass curation' % MIN_TRIALS_PER_SESSION, ...)
# ...
if me_raw is None:
    return dict(eid=eid, skip='no whisker motion energy from either camera', ...)
# ...
keep = valid_wheel & valid_me
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return dict(eid=eid, skip='fewer than %d trials with complete behaviour' % MIN_TRIALS_PER_SESSION, ...)
```

iii. The AI documented each skip reason and reported 14 sessions skipped (all for missing whisker ME).

## 10-a. What are the most time-consuming steps of the code?

i. Spike loading dominates at 78% of total worker time, followed by wheel loading at 17%. Binning is only 2%, trials 1%, motion energy 1%.

ii.
```python
# From CONVERSION_NOTES timing profile:
# spike loading 78%, wheel loading 17%, binning 2%, trials 1%, motion energy 1%
```

iii. The AI profiled the timing and documented it in CONVERSION_NOTES Step 9.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop runs once per trial. The AI notes this could theoretically be vectorized by offsetting each spike's bin index by its trial, but it is already fast (0.03s per session) due to the `searchsorted` + `add.at` approach.

ii.
```python
for k in range(n_trials):
    a, b = i0[k], i1[k]
    if b <= a:
        continue
    bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
    np.clip(bin_idx, 0, N_BINS - 1, out=bin_idx)
    np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
```

iii. The AI optimized the binning with vectorized searchsorted for boundaries but kept a per-trial loop for the actual counting. The behavior interpolation uses a single vectorized `interp1d` call.

## 10-c. What processing does the code repeat multiple times?

i. The AI's code does not appear to repeat any processing. Each step (loading, filtering, binning, interpolation, discretization) is done once per session.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies per-time-bin z-scoring to the neural data. This normalization is baked into the stored data, whereas the reference solution stores raw firing rates and lets the decoder handle normalization. The z-scoring could be considered unnecessary if the decoder or downstream analysis applies its own normalization. Additionally, the AI includes a `stim_onset` binary indicator input that the Decoder Task does not explicitly request.

ii.
```python
# Z-scoring:
mu = spikes_binned.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
sd = spikes_binned.std(axis=(0, 1), keepdims=True, dtype=np.float64)
neural = ((spikes_binned.astype(np.float64) - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)

# Extra input:
stim_indicator = np.zeros(N_BINS, dtype=np.float32)
stim_indicator[STIM_ONSET_BIN] = 1.0
arr[1] = stim_indicator
```

iii. The AI justified z-scoring by noting the provided decoder does no normalization, and without it high-rate neurons dominate the SVD. The stim_onset indicator was justified by the instruction "If an input is a time such as onset of some stimulus, represent it as a binary time series."

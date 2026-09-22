# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `.nwb` file under `/app/data` is found with a recursive glob and sorted with a "natural" (numeric-aware) key; each file is one session and is processed one at a time. Files are opened directly with `h5py` (not `pynwb`) and only the datasets actually needed are read: `processing/ophys/Deconvolved/<plane>/data`, the `ImageSegmentation/PlaneSegmentation` columns (`iscell`, `planeIdx`), the behavior streams `environment`, `position`, `speed`, `lick`, `trial number`, `trial_start`, `teleport`, the `position` timestamps, the sparse `Reward/timestamps`, plus `general/subject/subject_id` and `identifier`. `--sample` takes the first two sorted files, `--full` (default) takes all 152. All 152 sessions / 11 subjects / 12,216 complete trials in the release are processed.

ii.
```python
DATA_ROOT = '/app/data'
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
if args.sample: files=files[:2]
...
for i,f in enumerate(files):
    ns,xs,ys,info=process_session(f, make_plot=args.show_processing and i<2)
```
```python
    with h5py.File(path, 'r') as h:
        b = h['processing/behavior/BehavioralTimeSeries']
        names = ['environment', 'position', 'speed', 'lick', 'trial number']
        streams = {n: np.asarray(b[n+'/data'][:]) for n in names}
        timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` is a read-only 87 GB DANDI export (Dandiset 001361): 152 NWB 2.8.0 files… Files are organized as `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` for 11 mice." Reading only the required HDF5 datasets, one session at a time, was chosen for speed and to bound memory ("Read only released Deconvolved arrays… process sessions serially to bound memory"). The full conversion took 54.8 s.

## 1-b. How are the data split into subjects?

i. The subject identity is read from the NWB metadata field `general/subject/subject_id` of each file (not from the directory name). `data['subjects']` is built in order of first appearance across the naturally-sorted file list, and `subject_idx` is the index of that subject for each session. Result: 11 subjects (m3, m4, m7, m11–m15, m17–m19).

ii.
```python
        subject = h['general/subject/subject_id'][()].decode()
...
        if info['subject'] not in subjects: subjects.append(info['subject'])
        subject_idx.append(subjects.index(info['subject']))
```

iii. CONVERSION_NOTES Step 5 maps "subject ID → `subjects`, `subject_idx`: Unique mouse IDs in natural order; per-session index — NWB subject table — 11 subjects." Using the in-file subject table rather than the folder name is treated as the authoritative source; Step 9 verifies 11 subjects with 14 sessions each except m11 (12).

## 1-c. How are the data split into sessions?

i. One session per NWB file; no merging or splitting. Sessions appear in naturally sorted file order (sub-m3 ses-01 … sub-m19 ses-14), 152 in total. No cross-day ROI alignment is attempted.

ii.
```python
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
...
        neural.append(ns); inputs.append(xs); outputs.append(ys)
        region_idx.append(np.zeros(info['n_neurons'],dtype=np.int64)); infos.append(info)
```

iii. Step 2 of the notes states the file naming convention `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, i.e. one file = one recording day/session; Step 9 checks the resulting 152 sessions against the raw inventory ("Sessions … 152 … Exact").

## 1-d. How are the data split into trials?

i. Trials are cut from the behavior event streams, not from the `trial number` stream. Every sample where `trial_start > 0` is a trial onset; each onset is paired with the **first** sample where `teleport > 0` occurring before the next onset. The trial window is start-inclusive, teleport-exclusive: `slice(start, stop)`. The code raises if a start does not have exactly one teleport before the next start. All 12,216 starts matched.

ii.
```python
def find_complete_trials(b):
    """Pair each global start with first teleport before next start."""
    start_flags = np.asarray(b['trial_start/data'][:])
    teleport_flags = np.asarray(b['teleport/data'][:])
    starts = np.flatnonzero(start_flags > 0)
    teleports = np.flatnonzero(teleport_flags > 0)
    trials = []
    for j, start in enumerate(starts):
        next_start = starts[j+1] if j+1 < len(starts) else len(start_flags)
        candidates = teleports[(teleports > start) & (teleports < next_start)]
        if len(candidates) != 1:
            raise ValueError(f'start {start}: expected one following teleport before {next_start}, got {candidates}')
        trials.append((int(start), int(candidates[0])))  # stop is exclusive
    return trials
```

iii. Step 4 discrepancy table: "Trial-ID assignment crosses teleport boundaries: 2,562 IDs have no teleport and 2,562 have two … Ignore trial-ID boundaries for slicing. Match each global start to the first subsequent teleport before the next start; include start and exclude teleport. All 12,216 starts match." Step 5 adds that the teleport sample itself is excluded because "Teleport samples are explicitly excluded because their positions can be interpolated between track end and pre-track jitter", the same masking the reference `dff` performs.

## 1-e. How are trials filtered based on quality controls?

i. Two filters. (1) The paper's lick-sensor corruption criterion: a trial is dropped if more than 30% of its frames have a cumulative lick count > 2. This removed exactly 81 of 12,216 trials — the same number the paper reports. (2) Structural validity: only "complete" trials (a start paired with a teleport) are kept, which drops one incomplete trial ID at a session edge; a session with fewer than 2 remaining trials raises an error (never triggered). No minimum-duration filter is applied (min retained trial length is 96 frames).

ii.
```python
            lick_raw = streams['lick'][q]
            # Reference correct_lick_sensor_error: fraction of frames whose cumulative count is >2.
            if np.mean(lick_raw > 2) > 0.30:
                bad_lick += 1
                continue
...
        if len(neural_trials) < 2:
            raise ValueError(f'Only {len(neural_trials)} valid trials')
```

iii. Step 4/Step 5: "Since categorical target cannot represent NaN, exclude corrupt trials entirely; document exact final count"; "Exclude a complete trial if >30% of its samples have lick count >2. A NaN categorical lick target is impossible, and retaining it would train on known sensor artifact." Step 9 uses the count as a consistency check: "Corrupt lick trials | 81 (0.65%) | >30% samples with count >2 | 81 using complete boundaries | 81 excluded | Exact count."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The stored NWB array `processing/ophys/Deconvolved/<plane>/data` (Suite2p deconvolved activity, time × ROI), concatenated across planes in `plane0, plane1` order and then restricted to ROIs with `ImageSegmentation/PlaneSegmentation/iscell[:,0] > 0`. The raw `Fluorescence` (F) and `Neuropil` (Fneu) arrays are read by nothing in the script; no dF/F is computed.

ii.
```python
def load_neural_and_regions(h):
    """Concatenate ophys planes in PlaneSegmentation order and apply Suite2p iscell."""
    dec = h['processing/ophys/Deconvolved']
    plane_names = sorted((k for k, v in dec.items() if isinstance(v, h5py.Group)), key=natural_key)
    arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
    if not arrays or len({x.shape[0] for x in arrays}) != 1:
        raise ValueError('Missing planes or unequal plane time dimensions')
    full = np.concatenate(arrays, axis=1)
    seg = h['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = np.asarray(seg['iscell'][:, 0]) > 0
    plane_idx = np.asarray(seg['planeIdx'][:], dtype=np.int16)
    if full.shape[1] != len(iscell) or len(plane_idx) != len(iscell):
        raise ValueError(f'plane concatenation {full.shape} != segmentation {len(iscell)}')
    return full[:, iscell], plane_idx[iscell], plane_names
```

iii. The agent's reasoning (trajectory step 25): "The NWBs contain exactly three ophys arrays with matched time × ROI shapes: `Deconvolved`, `Fluorescence`, and `Neuropil`. Thus reference-processed deconvolved activity is directly available and no expensive recomputation is needed for conversion." CONVERSION_NOTES Step 4: "Neural representation … Use released `Deconvolved`; no recomputation or spatial binning", and Step 10: "converter reads released NWB Deconvolved streams; reference `multi_anim_sess` creates that stream through `dff(..., deconvolve=True)`. **Same signal**, avoiding redundant recomputation." Step 5 adds that plane matrices are concatenated in `planeIdx` order and only `iscell` ROIs retained.

## 2-b. How is the `neural` data processed?

i. Essentially no processing. The stored Deconvolved values are cast to float32, planes concatenated along the neuron axis, `iscell` ROIs selected, the array transposed to (n_neurons, n_timepoints) per trial, and sliced by the trial window. No neuropil subtraction, no maximin baseline, no dF/F, no Gaussian smoothing, no OASIS deconvolution, no normalization, no spatial binning, no smoothing. Ten two-plane sessions have exactly one more neural row than behavior samples; that trailing row is trimmed.

ii.
```python
        neural_excess_frames = int(neural_all.shape[0] - ntime)
        # Ten released two-plane NWBs have exactly one unmatched trailing neural row.
        if neural_excess_frames not in (0, 1):
            raise ValueError(f'Unexpected neural/behavior row difference: {neural_all.shape[0]} vs {ntime}')
        if neural_excess_frames:
            neural_all = neural_all[:ntime]
...
            neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 4/5: "The released NWBs are already temporally aligned, reference-processed data. Conversion should operate at the native per-plane imaging sample grid… concatenate planes, retain Suite2p-classified cells, and preserve full time-domain deconvolved events"; Step 6 lists "Recomputing dF/F/deconvolution or reading unused fluorescence would be costly" as the avoided inefficiency. Step 1 of the notes does record that "The reference pipeline **does compute dF/F** rather than treating raw fluorescence as activity", but the agent concluded the released `Deconvolved` array already *is* that product.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: Suite2p's curated classification `iscell[:,0] > 0` (138,678 of 312,110 ROIs; 155–2,341 per session). The paper's additional exclusion of putative interneurons (dF/F–speed Pearson r > 0.5) is deliberately **not** applied, and neither are place-cell / reward-relative / speed-threshold selections.

ii.
```python
    iscell = np.asarray(seg['iscell'][:, 0]) > 0
...
    return full[:, iscell], plane_idx[iscell], plane_names
```

iii. Step 4: "Apply `iscell[:,0] > 0` as imaging-quality curation. Do not impose place-cell, reward-relative, speed, or interneuron analysis selections on a general decoder." Step 5 Key Decision 4: "Quality-filtered cells, not analysis-selected cells… Place-cell, reward-relative, speed-correlation and k-means criteria answer paper-specific questions and would bias a general decoder." Step 10 repeats: "It intentionally does not apply place-cell/RR/interneuron selections because those are analysis-specific rather than recording-quality filters."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the `trial_start` pulse: the trial's first neural column is the frame at the start pulse index, and all streams are sliced with the identical `slice(start, stop)`. No pre-event window is taken (`off_start = 0.0`, `off_end = None`), and the time input is zeroed at the same frame. Neural and behavior rows are already frame-synchronized in the release, so no resampling or shifting is done; the only correction is trimming the single extra trailing neural frame present in 10 two-plane sessions.

ii.
```python
            q = slice(start, stop)
...
            neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
            inp = np.vstack([
                (timestamps[q]-timestamps[start]).astype(np.float32),
```
```python
              metadata=dict(..., temporal_alignment_event='start of trial (trial_start pulse)',
                            off_start=0.0,off_end=None, ...,
                            trial_window='trial_start inclusive to teleport exclusive', ...)
```

iii. Step 2: "samples assigned a trial ID can precede its `trial_start` pulse (pre-track/jitter), while `teleport` marks its end. Requested temporal alignment must therefore use the onset pulse, not merely the first sample with that trial ID." Step 10: "behavior and ophys rows are already VR-to-2P aligned; converter retains the shared ~15.5 Hz frame grid and subtracts start timestamp."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging/behavior frame grid is kept: median Δt = 0.0644836 s, recorded as `time_bin_size = 64.4836 ms`. No rebinning, downsampling, upsampling, or smoothing. Each session's Δt is checked against a hard-coded reference (1/15.5078125 s, tolerance 2e-5 s) and a session whose median Δt deviates raises. For two-plane sessions the stored scanner `rate` (31.015625 Hz) is ignored in favour of the behavior timestamps, since each plane has exactly one row per behavior sample.

ii.
```python
DT_REFERENCE = 1.0 / 15.5078125
...
        dt = float(np.median(np.diff(timestamps)))
        if not np.isclose(dt, DT_REFERENCE, rtol=0, atol=2e-5):
            raise ValueError(f'Unexpected behavior dt {dt}')
...
    dts=np.asarray([x['time_bin_s'] for x in infos])
    ... time_bin_size=float(np.median(dts)*1000) ...
```

iii. Step 4: "The 62.03 Hz value is aggregate scanner rate. Data are already synchronized at 15.5078 samples/s per plane; concatenate planes, do not downsample." Step 10: "no temporal or spatial rebinning is applied. This matches paper time-domain GLM and preserves requested framewise low-speed behavior."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `processing/behavior/BehavioralTimeSeries/position/timestamps` array (seconds). The other behavior streams are only checked for equal length, not for identical timestamps.

ii.
```python
        timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
        ntime = len(timestamps)
        if any(len(x) != ntime for x in streams.values()):
            raise ValueError('Behavior stream length mismatch')
```

iii. Step 5 mapping table: "behavior timestamps → `input[0]` time from trial start → `timestamp - timestamp[start]`, seconds, repeated samplewise". The notes state all frame-aligned behavior streams carry explicit timestamps on the common imaging-frame grid, so any one of them serves.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the trial's start frame is subtracted from the trial's timestamps, and the result is stored as float32. Nothing else (no rounding to a frame index, no rescaling). The first element is therefore exactly 0 and the last is the trial duration (max 216.5 s over the dataset).

ii.
```python
            inp = np.vstack([
                (timestamps[q]-timestamps[start]).astype(np.float32),
                np.full(n, env, dtype=np.float32),
                np.full(n, trial_id, dtype=np.float32),
                np.full(n, prev_outcome, dtype=np.float32),
            ])
```

iii. Straightforward implementation of the requested input, using the alignment event chosen in 2-d ("subtracts start timestamp", Step 10).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `slice(start, stop)` indexes both the neural matrix and the timestamp vector, and the neural array is forced to the behavior length beforehand (trailing extra frame trimmed, any other mismatch raises). A per-trial shape assertion checks `neu.shape[1] == n` and `inp.shape == (4, n)`.

ii.
```python
        if neural_excess_frames not in (0, 1):
            raise ValueError(...)
        if neural_excess_frames:
            neural_all = neural_all[:ntime]
...
            if neu.shape[1] != n or inp.shape != (4, n) or out.shape != (6, n):
                raise AssertionError('Trial shape mismatch')
```

iii. Step 4: behavior is already interpolated onto two-photon frames by the reference pipeline (`vr_align_to_2P`), so rows correspond one-to-one; Step 10 Check 5 documents the trailing-frame edge case and the strict +1-only assertion.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The frame-wise `processing/behavior/BehavioralTimeSeries/environment` stream (0 = Env1, 1 = Env2; −1 is a pre-recording sentinel).

ii.
```python
            env_values = np.unique(streams['environment'][q])
            env_values = env_values[env_values >= 0]
```

iii. Step 4: "Environment | `Env1` maps to 0 and `Env2` to 1 | Framewise environment is 0/1 and can switch within cross-environment sessions | … | Use framewise stream sampled at trial start/as constant within each trial; it is authoritative even for identifier naming-order anomalies." I.e. the data stream, not the scene name in the identifier, is trusted for environment.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial the unique non-negative environment values are taken; the code raises unless exactly one value remains and it is 0 or 1. That scalar is then broadcast over all timepoints of the trial as a float32 row.

ii.
```python
            if len(env_values) != 1 or env_values[0] not in (0, 1):
                raise ValueError(f'Non-binary/nonconstant environment trial {j}: {env_values}')
            env = int(env_values[0])
...
                np.full(n, env, dtype=np.float32),
```

iii. Step 5: "0=ENV1, 1=ENV2; use value at start and repeat … Binary per trial." The strict check doubles as a sanity check that no trial straddles an environment change (it never fired across the 152 sessions, including the cross-environment switch days).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The stored `trial number` behavior stream, read at the trial's start frame (`streams['trial number'][start]`), rounded to an int.

ii.
```python
            trial_id = int(round(float(streams['trial number'][start])))
```

iii. Step 5: "source trial order/ID → `input[2]` trial number → Original nonzero-based… Original nonnegative trial ID, repeated across samples… preserve zero-based source value." Step 2 notes "Trial IDs are contiguous in every session", so the stored ID is used rather than a re-derived counter, while the *boundaries* still come from `trial_start`/`teleport`. (Verified independently here: the stored ID at every one of the 12,216 start pulses equals the 0-based chronological index, so this is numerically identical to a loop counter.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the rounding above; the scalar is broadcast across the trial's timepoints as a float32 row. Excluded (lick-corrupt) trials do not renumber the remaining ones — the stored chronological ID is kept, so retained trial numbers can skip values. Range across the dataset is 0–99.

ii.
```python
                np.full(n, trial_id, dtype=np.float32),
```

iii. Step 5/Step 10: per-trial context variables are "repeated to produce a uniform (6, n_timepoints) trial array" / "(4, n)" for inputs; chronological source indices are used throughout so that trial-order semantics (including the reward switch at trial 30) are unaffected by exclusions.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `processing/behavior/BehavioralTimeSeries/Reward` event series — specifically its `timestamps` (delivered rewards) — combined with the trial boundaries and the behavior timestamps. Reward *amounts* and the `autoreward` stream are not used.

ii.
```python
def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. Step 5 Key Decision 7: "Reward from sparse timestamps: Delivered reward events are direct evidence of outcome; do not infer outcome from autoreward or zone entry."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial rewarded/omitted vector is computed once per session (`outcomes`), then for trial *j* the input is `outcomes[j-1]`, and 0 for the first trial of the session. "Previous" means the previous **chronological complete trial**, even if that trial was itself discarded for lick corruption. The value is broadcast over the trial's timepoints.

ii.
```python
        outcomes = reward_outcomes(b, timestamps, trials)
...
            prev_outcome = int(outcomes[j-1]) if j > 0 else 0
...
                np.full(n, prev_outcome, dtype=np.float32),
```

iii. Step 5: "0 for first trial/omitted previous trial, 1 if previous complete source trial contains reward… Based on chronological source trial even if that previous trial is excluded for bad licking." Step 10 Check 5: "Previous outcome also uses the immediately preceding source trial, not previous retained trial."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the frame-wise `position` stream, and the trial's reward-zone interval. The interval is **not** taken from the frame-wise `reward_zone` stream; it is derived from the session's scene name, parsed out of the NWB `identifier` (e.g. `/data/InVivoDA/GCAMP3/01_10_2022/Env1_LocationC`), mapped through the paper's coordinates A = 80–130 cm, B = 200–250 cm, C = 320–370 cm (the `X`/`Y`/`Z` entries of the reference `behavior.reward_zone_dict`). Switch sessions (two labels in the scene) use the first zone for chronological trials 0–29 and the second from trial 30 on, replicating `behavior.get_reward_zones(..., change_trial=30)`.

ii.
```python
ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}

def parse_scene(identifier):
    """Return ordered A/B/C schedule labels from final identifier component."""
    scene = identifier.rstrip('/').split('/')[-1]
    labels = re.findall(r'(?:Location)?([ABC])', scene)
    if len(labels) not in (1, 2):
        raise ValueError(f'Cannot parse reward schedule from {scene!r}: {labels}')
    return scene, labels

def zone_for_trial(labels, chronological_index):
    # Matches behavior.get_reward_zones(..., change_trial=30).
    return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]
...
            zone = zone_for_trial(labels, j)
            zstart, zstop = ZONE_COORDS[zone]
```

iii. Step 4: "Reward-zone code | `get_reward_zones` derives A/B/C and coordinates from scene, switching at trial 30 | Framewise `reward_zone` values 0–8 are brief event states near reward, not identity | … | Parse one/two A/B/C labels from NWB identifier and switch after first 30 complete trials, matching reference code. Use fixed coordinates from `reward_zone_dict`." Step 6 records that an initial transcription error (A = 50–100 cm) was caught and fixed to the reference's A→X = 80–130 cm mapping.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the 50 cm zone: `position − zone_start` before the zone (negative), exactly `0` anywhere inside the zone, `position − zone_stop` after the zone (positive). Computed vectorized over the trial's positions with `np.where`, then discretized (7-c). Observed per-trial min/max distances are also recorded in the session metadata.

ii.
```python
def distance_classes(position, start, stop):
    distance = np.where(position < start, position-start,
                        np.where(position > stop, position-stop, 0.0))
```

iii. Step 5: "Signed distance to nearest point in interval: position-start below, 0 inside, position-stop above; discretize 7 requested classes… Class 3 exactly covers all positions inside the 50 cm zone ('distance to any location in zone' = 0)."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit boolean masks rather than `digitize`, with the zero class reserved for exact 0 (inside the zone): 0 for < −50; 1 for [−50, −10); 2 for [−10, 0); 3 for == 0; 4 for (0, 10]; 5 for (10, 50]; 6 for > 50 cm. Stored as int64, one value per frame.

ii.
```python
    out = np.empty(distance.shape, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
```

iii. Step 5: "Distance classes: 0 `<-50`; 1 `[-50,-10)`; 2 `[-10,0)`; 3 `==0`; 4 `(0,10]`; 5 `(10,50]`; 6 `>50` cm. Boundary handling follows the wording exactly." The class-3 bin is defined to mean "in the reward zone" because the requested variable is distance to *any* location in the zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial slice as the neural data — position is indexed with the identical `q = slice(start, stop)` — so distance classes are frame-by-frame aligned with the neural columns; the shape assertion enforces `out.shape == (6, n)`.

ii.
```python
            q = slice(start, stop)
            n = stop-start
            pos = streams['position'][q].astype(np.float32)
            dclass, distance = distance_classes(pos, zstart, zstop)
...
            if neu.shape[1] != n or inp.shape != (4, n) or out.shape != (6, n):
                raise AssertionError('Trial shape mismatch')
```

iii. Step 10: the released NWB streams are already VR-to-2P aligned, so identical indexing is sufficient; verified by the independent raw-data sanity checks in `/app/cache` ("np.allclose passed for every array") and the `critical_alignment_check.png` overlay.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The frame-wise `processing/behavior/BehavioralTimeSeries/position` stream (cm along the 450 cm virtual corridor).

ii.
```python
            pos = streams['position'][q].astype(np.float32)
```

iii. Step 5 mapping: "`position/data` → `output[1]` absolute position → `np.digitize(position,[90,180,270,360])`, classes 0–4 → Track samples from start to pre-teleport only." Restricting to the start→teleport window keeps out the −500/−50 teleport-and-jitter samples noted in Step 2.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None besides a float32 cast and discretization; raw position values are used directly.

ii.
```python
            posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. Step 5: raw framewise position is the requested variable; no smoothing or spatial binning of behavior is applied because "Spatial speed thresholds and position binning are downstream paper analyses and are inappropriate for this framewise decoder".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges [90, 180, 270, 360] and `right=False`, i.e. half-open bins [-inf, 90), [90, 180), [180, 270), [270, 360), [360, inf) → classes 0–4, labelled '< 90 cm' … '>= 360 cm'. The open end bins absorb the few samples slightly outside 0–450 cm.

ii.
```python
            posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
...
              output_values=[..., ['< 90 cm','90 to < 180 cm','180 to < 270 cm','270 to < 360 cm','>= 360 cm'], ...]
```

iii. Step 5: "Position classes use edges 90, 180, 270 and 360 cm; values exactly on an edge enter the higher conventional half-open bin" — five equal 90 cm bins over the 450 cm track, as the instructions specify.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical `slice(start, stop)` indexing as the neural data; no shifting or interpolation.

ii.
```python
            q = slice(start, stop)
            pos = streams['position'][q].astype(np.float32)
```

iii. Same as 7-d: the released streams are already on the imaging-frame grid, verified by the independent `np.allclose` spot checks in Step 10 and the alignment plots.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The frame-wise `lick` behavior stream (cumulative lick counts per frame, values 0–8).

ii.
```python
            lick_raw = streams['lick'][q]
```

iii. Step 2: "Lick count/sample | 0 to 8 (must be binarized for requested output)"; Step 5 maps "`lick/data` → `output[3]` lick → `(lick > 0)` after paper corruption filtering".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps: (1) trial-level corruption screening — the whole trial is dropped if >30% of its frames have count > 2 (see 1-e); (2) binarization of the surviving counts, `lick > 0 → 1`, stored int64 per frame. No smoothing, no spatial binning, no lick-rate conversion.

ii.
```python
            if np.mean(lick_raw > 2) > 0.30:
                bad_lick += 1
                continue
...
            lickclass = (lick_raw > 0).astype(np.int64)
```

iii. Methods quoted in Step 3: "These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2… Remaining lick counts were converted to a binary vector." The paper's subsequent 10 cm spatial binning / lick-rate step is deliberately skipped as analysis-specific.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slice as the neural data, frame for frame; no shift.

ii.
```python
            lick_raw = streams['lick'][q]
            ...
            lickclass = (lick_raw > 0).astype(np.int64)
```

iii. As in 7-d/8-d: streams are pre-aligned to imaging frames; Step 12 re-verified lick values against the raw NWB for three named trials with `np.allclose`.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The NWB `identifier` string only (its final path component is the scene, e.g. `Env1_LocationC`, `Env1_LocationA_to_C`, `Env1_C_to_Env2_B`), plus the chronological trial index for switch sessions. The frame-wise `reward_zone` stream is explicitly rejected as a source.

ii.
```python
        identifier = h['identifier'][()].decode()
        scene, labels = parse_scene(identifier)
...
            zone = zone_for_trial(labels, j)
...
                np.full(n, ZONE_INDEX[zone], dtype=np.int64),
```

iii. Step 4: "Framewise `reward_zone` values 0–8 are brief event states near reward, not identity… Parse one/two A/B/C labels from NWB identifier and switch after first 30 complete trials, matching reference code." Step 5 Key Decision 6: "Zone identity from scene: Framewise `reward_zone` is an event-state code, whereas scene schedule and trial-30 switch exactly match reference `get_reward_zones`."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex extraction of the ordered A/B/C labels from the scene (error if not 1 or 2 labels), selection of label[0] for chronological trials < 30 and label[1] from trial 30 onwards in two-label sessions, mapping A→0, B→1, C→2, and broadcasting the value across the trial's timepoints. The same label also picks the zone coordinates used for the distance output.

ii.
```python
    labels = re.findall(r'(?:Location)?([ABC])', scene)
    if len(labels) not in (1, 2):
        raise ValueError(f'Cannot parse reward schedule from {scene!r}: {labels}')
...
def zone_for_trial(labels, chronological_index):
    # Matches behavior.get_reward_zones(..., change_trial=30).
    return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]
```

iii. Step 5: "Fixed coordinates X/A=80–130 cm, Y/B=200–250 cm, Z/C=320–370 cm from code"; Step 9 reports the resulting frame-weighted class balance [0.332, 0.336, 0.333] as a plausibility check, and Step 10 notes that "Trial-29/30 zone transitions use source chronological index, unaffected by lick exclusions."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` TimeSeries' `timestamps` (delivered rewards), compared against the behavior timestamps at the trial boundaries. Same source as the previous-trial-outcome input.

ii.
```python
def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. Step 5 Key Decision 7 (delivered reward events are the direct evidence of outcome, not `autoreward` or zone entry); Step 12 verified three specific trials' outcomes (including omissions and a reward at t = 28.3728 s) directly against the raw NWB timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each complete trial, 1 if at least one reward timestamp falls in the half-open interval [timestamps[start], timestamps[stop]) — i.e. the same start-inclusive / teleport-exclusive window used for the data — else 0. The scalar is broadcast across all timepoints of the trial as row 5 of the output. Computed once per session for all trials (list comprehension over trials), then reused for both the outcome output and the previous-outcome input.

ii.
```python
        outcomes = reward_outcomes(b, timestamps, trials)
...
            out = np.vstack([
                dclass, posclass, speedclass, lickclass,
                np.full(n, ZONE_INDEX[zone], dtype=np.int64),
                np.full(n, outcomes[j], dtype=np.int64),
            ])
```

iii. Step 5: "Any delivered reward timestamp in [start, teleport); repeat 0/1 — Per-trial categorical outcome". Step 12 addresses the low decoding accuracy for this output: "Much of each trial occurs before reward delivery, when neural activity cannot predict the later stochastic omission… Changing labels only after delivery would violate the requested per-trial output specification."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is deliberately fail-fast, with exactly one tolerated anomaly:
- **Neural vs behavior length**: a difference of 0 or +1 rows is allowed; +1 (10 two-plane sessions) is trimmed from the end and recorded in `session_info['neural_excess_frames_trimmed']`; anything else raises.
- **Behavior stream lengths**: any stream whose length differs from the timestamps raises.
- **Sampling rate**: median Δt more than 2e-5 s from 1/15.5078125 raises.
- **Trial structure**: a start without exactly one teleport before the next start raises; the single incomplete trial ID at a session edge is simply never emitted.
- **Sentinel/invalid samples**: excluded implicitly by slicing only start→teleport; environment sentinels (−1) are filtered out before the constancy check.
- **Corrupt lick sensor**: whole trial dropped (81 trials).
- **Per-trial integrity**: shape assertion on neural/input/output, and a non-finite check on neural and input values.
- **Session integrity**: a session with fewer than 2 usable trials raises.

ii.
```python
        if any(len(x) != ntime for x in streams.values()):
            raise ValueError('Behavior stream length mismatch')
        dt = float(np.median(np.diff(timestamps)))
        if not np.isclose(dt, DT_REFERENCE, rtol=0, atol=2e-5):
            raise ValueError(f'Unexpected behavior dt {dt}')
...
        if neural_excess_frames not in (0, 1):
            raise ValueError(f'Unexpected neural/behavior row difference: {neural_all.shape[0]} vs {ntime}')
        if neural_excess_frames:
            neural_all = neural_all[:ntime]
...
            if not (np.isfinite(neu).all() and np.isfinite(inp).all()):
                raise ValueError('Nonfinite neural/input values')
...
        if len(neural_trials) < 2:
            raise ValueError(f'Only {len(neural_trials)} valid trials')
```

iii. Step 9 Iteration 1: "The first full run stopped at session 114 (`m17` session 04): both neural planes had 22,791 rows versus 22,790 behavior samples. An all-file audit found exactly 10 two-plane sessions with the same +1 neural-row pattern and no other mismatch. In each case the extra row is after the final behavior sample… The script now permits only a 0 or +1 neural difference, trims that trailing row, records it, and rejects all other differences." Step 10 Check 5 lists the remaining edge cases (incomplete trial ID, start/teleport pairing, start-inclusive/teleport-exclusive boundary, chronological indices for switch/previous-outcome).

## 13-a. What are the most time-consuming steps of the code?

i. The whole full conversion takes 54.8 s for 152 sessions, so nothing is expensive in absolute terms. Within that, the costs are (1) reading the `Deconvolved` HDF5 arrays (hundreds of MB per session; this dominates per-session time), (2) the `np.concatenate` of planes plus the `iscell` fancy-index copy, (3) building and contiguous-copying the per-trial neural slices, and (4) writing the 9.66 GB pickle at the end. The script prints per-session timings (`info['seconds']`) and a total elapsed time.

ii.
```python
def process_session(path, make_plot=False):
    t0 = time.time()
...
                    seconds=float(time.time()-t0))
...
        print(f"[{i+1}/{len(files)}] {info['file']}: neurons={info['n_neurons']} trials={info['kept_trials']}/{info['complete_trials']} badlick={info['excluded_bad_lick']} time={info['seconds']:.2f}s",flush=True)
...
    print(f'SAVED {args.outpicklefile} sessions={len(neural)} trials={total_trials} summed_neurons={total_cells} elapsed={time.time()-t0:.1f}s ...')
```

iii. Step 6: "Large neural matrices dominate I/O and pickle size. Recomputing dF/F/deconvolution or reading unused fluorescence would be costly." Step 7 estimated 3–8 minutes for the full run (actual 54.8 s), below the 15-minute budget.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain: (1) the per-trial loop in `process_session` — the distance/position/speed/lick discretizations are vectorized within a trial but could have been computed once for the whole session and then sliced; (2) `find_complete_trials`'s Python loop over starts, which could be done with a single `np.searchsorted` of the teleport indices into the start indices; (3) `reward_outcomes`'s list comprehension over trials, likewise expressible as two `searchsorted` calls. None matters in practice (the loops are ~80 iterations/session against hundreds of MB of I/O). The per-trial work that genuinely cannot be vectorized away is the trial-shaped neural copy, which is required by the output format.

ii.
```python
    for j, start in enumerate(starts):
        next_start = starts[j+1] if j+1 < len(starts) else len(start_flags)
        candidates = teleports[(teleports > start) & (teleports < next_start)]
```
```python
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```
```python
        for j, (start, stop) in enumerate(trials):
            q = slice(start, stop)
            ...
            posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. Step 6: "Code speedups added: … vectorize class construction; process sessions serially to bound memory; use float32; concatenate planes once per session; slice trials from the in-memory session array." The agent treated further loop removal as unnecessary once the full run fit in under a minute.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each dataset read once; there is no separate survey/inventory pass inside the converter (the exploratory inventory scripts live in `/app/cache` and are not part of the pipeline). Per-trial reward outcomes are computed once per session and reused for both `input[3]` and `output[5]`. The mild repetitions are: the reward-timestamp array is re-scanned once per trial inside the `reward_outcomes` comprehension; `streams['lick'][q]` is thresholded twice (once at >2 for corruption, once at >0 for the label); and zone coordinates/labels are re-looked-up per trial from the dicts.

ii.
```python
        outcomes = reward_outcomes(b, timestamps, trials)   # once per session, used twice
...
            if np.mean(lick_raw > 2) > 0.30: ...
            lickclass = (lick_raw > 0).astype(np.int64)
```

iii. Step 6: "Read only released Deconvolved arrays… concatenate planes once per session; slice trials from the in-memory session array." The design goal was a single streaming pass per file.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of bookkeeping rather than computation: `plane_idx` is read and carried through (`curated_plane_idx`) although every neuron is labelled brain region 0 (`dorsal CA1`) downstream; a per-trial `kept_info` dict (indices, zone, outcome, environment, distance min/max) is built for all 12,135 trials and stored in `metadata['session_info']` though the decoder never reads it; per-trial `distance.min()/max()` are computed only for that record; `plane_names` and `scene`/`identifier` strings are likewise metadata only. The `trial number` stream is read for every session to produce `input[2]`, which is numerically identical to the loop index. Nothing large or slow is computed and thrown away — and conversely, the paper's dF/F pipeline is not run at all.

ii.
```python
            kept_info.append(dict(source_trial_index=j, trial_id=trial_id, start=start, stop=stop,
                                  zone=zone, outcome=int(outcomes[j]), environment=env,
                                  distance_min=float(distance.min()), distance_max=float(distance.max())))
...
        info = dict(file=..., planes=plane_names, curated_plane_idx=curated_planes, ...)
...
              brain_regions=['dorsal CA1'], brain_region_idx=region_idx,
```

iii. Step 5: "Region `dorsal CA1`; all neuron indices 0 — Record plane indices additionally in metadata session info." The per-trial records are kept deliberately as provenance for the sanity checks described in Steps 10 and 12 (they are what the independent raw-data comparisons index by).

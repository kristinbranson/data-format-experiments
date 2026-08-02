import argparse
import datetime as dt
import gc
import math
import os
import pickle
from collections import defaultdict

import numpy as np


SECONDS_PER_DAY = 24.0 * 3600.0
REGION_NAMES = ["V1", "mHV", "lHV", "aHV"]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Zhong et al. imaging data into decoder format.")
    parser.add_argument("--root", type=str, default="/app", help="Project root.")
    parser.add_argument("--output", type=str, default="/app/converted_data.pkl", help="Full dataset output pickle.")
    parser.add_argument("--sample-output", type=str, default="/app/sample_data.pkl", help="Sample dataset output pickle.")
    parser.add_argument("--neurons-per-session", type=int, default=128, help="Number of neurons retained per session.")
    parser.add_argument("--frames-per-bin", type=int, default=3, help="Number of retained imaging frames to average per decoder time bin.")
    parser.add_argument("--sample-sessions", type=int, default=6, help="Number of sessions kept in sample dataset.")
    parser.add_argument("--sample-trials-per-session", type=int, default=80, help="Max trials per session kept in sample dataset.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional limit on the number of converted sessions, for testing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used only for deterministic tie-breaking.")
    return parser.parse_args()


def session_base_from_key(key):
    return "_".join(str(key).split("_")[:5])


def load_exp_info(root):
    path = os.path.join(root, "data", "beh", "Imaging_Exp_info.npy")
    return np.load(path, allow_pickle=True).item()


def collect_unique_sessions(exp_info):
    sessions = []
    seen = set()
    for entries in exp_info.values():
        for item in entries:
            key = (item["mname"], item["datexp"], item["blk"])
            if key in seen:
                continue
            seen.add(key)
            sessions.append(
                {
                    "mname": item["mname"],
                    "datexp": item["datexp"],
                    "blk": item["blk"],
                    "key": key,
                    "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
                    "exp_item": item,
                }
            )
    return sessions


def behavior_files(root):
    beh_dir = os.path.join(root, "data", "beh")
    files = []
    for name in sorted(os.listdir(beh_dir)):
        if not name.startswith("Beh_") or not name.endswith(".npy"):
            continue
        full = os.path.join(beh_dir, name)
        try:
            files.append((name, np.load(full, allow_pickle=True).item()))
        except Exception:
            continue
    return files


def arrays_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            return a.shape == b.shape and np.allclose(a, b, equal_nan=True)
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


def build_behavior_lookup(root):
    beh_files = behavior_files(root)
    canonical = {}
    provenance = defaultdict(list)
    duplicate_mismatches = []
    keys_to_compare = [
        "ntrials",
        "WallName",
        "ft_trInd",
        "ft_Pos",
        "ft_CorrSpc",
        "ft_move",
        "ft_RunSpeed",
        "StartFr",
        "EndFr",
        "SoundFr",
        "LickFr",
    ]
    for filename, beh_dict in beh_files:
        for raw_key, beh in beh_dict.items():
            base = session_base_from_key(raw_key)
            provenance[base].append((filename, raw_key))
            if base not in canonical:
                canonical[base] = (filename, raw_key, beh)
                continue
            ref = canonical[base][2]
            mismatch = []
            for key in keys_to_compare:
                if key not in beh or key not in ref:
                    continue
                if not arrays_equal(ref[key], beh[key]):
                    mismatch.append(key)
            if mismatch:
                duplicate_mismatches.append((base, canonical[base][:2], (filename, raw_key), mismatch))
    return canonical, provenance, duplicate_mismatches


def parse_date(date_str):
    return dt.datetime.strptime(date_str, "%Y_%m_%d").date()


def compute_training_days(sessions):
    by_subject = defaultdict(list)
    for idx, session in enumerate(sessions):
        by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
    offsets = {}
    for subject, entries in by_subject.items():
        first_date = min(date for _, date in entries)
        for idx, date in entries:
            offsets[idx] = float((date - first_date).days)
    return offsets


def compute_frame_dt_ms(canonical_lookup, sessions):
    dts = []
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        ft = np.asarray(beh["ft"], dtype=np.float64)
        if ft.size < 2:
            continue
        diffs = np.diff(ft) * SECONDS_PER_DAY * 1000.0
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            dts.append(np.median(diffs))
    return float(np.median(dts))


def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
    out[(iarea == 5) | (iarea == 6)] = 2
    out[(iarea == 3) | (iarea == 4)] = 3
    return out


def trial_frame_indices(beh, nfr):
    ft_trind = np.asarray(beh["ft_trInd"][:nfr])
    is_corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    is_move = np.asarray(beh["ft_move"][:nfr], dtype=np.float64) > 0
    ntrials = int(beh["ntrials"])
    out = []
    for tr in range(ntrials):
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
    return out


def chunk_indices(indices, chunk_size):
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


def compute_speed_thresholds(canonical_lookup, sessions, frames_per_bin):
    speed_values = []
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        nfr = len(beh["ft"])
        ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
        for indices in trial_frame_indices(beh, nfr):
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
    speed_values = np.asarray(speed_values, dtype=np.float32)
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return speed_values, q.astype(np.float32)


def output_stimulus_names(canonical_lookup, sessions):
    names = set()
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        names.update(map(str, np.unique(beh["WallName"])))
    return sorted(names)


def proportional_region_targets(region_counts, total_target):
    region_counts = np.asarray(region_counts, dtype=np.int64)
    total_available = int(region_counts.sum())
    if total_available <= total_target:
        return region_counts.copy()
    scaled = total_target * region_counts / total_available
    base = np.floor(scaled).astype(np.int64)
    base = np.minimum(base, region_counts)
    remainder = total_target - int(base.sum())
    fractions = scaled - np.floor(scaled)
    while remainder > 0:
        valid = np.flatnonzero(base < region_counts)
        if valid.size == 0:
            break
        pick = valid[np.argmax(fractions[valid])]
        base[pick] += 1
        fractions[pick] = -1.0
        remainder -= 1
    return base


def variance_over_columns(matrix, cols, block_cols=1024, max_var_frames=2048):
    if cols.size > max_var_frames:
        keep = np.linspace(0, cols.size - 1, max_var_frames, dtype=np.int64)
        cols = cols[keep]
    nneurons = matrix.shape[0]
    sum_x = np.zeros(nneurons, dtype=np.float64)
    sum_x2 = np.zeros(nneurons, dtype=np.float64)
    n = 0
    for start in range(0, cols.size, block_cols):
        block = cols[start : start + block_cols]
        x = matrix[:, block].astype(np.float64, copy=False)
        sum_x += x.sum(axis=1)
        sum_x2 += np.square(x).sum(axis=1)
        n += x.shape[1]
    mean = sum_x / max(n, 1)
    var = sum_x2 / max(n, 1) - np.square(mean)
    return var.astype(np.float32)


def select_neurons(spk, region_idx_full, selected_frames, neurons_per_session):
    region_counts = [(region_idx_full == ridx).sum() for ridx in range(len(REGION_NAMES))]
    targets = proportional_region_targets(region_counts, neurons_per_session)
    var = variance_over_columns(spk, selected_frames)
    selected = []
    for ridx, target in enumerate(targets):
        if target <= 0:
            continue
        candidates = np.flatnonzero(region_idx_full == ridx)
        if candidates.size == 0:
            continue
        order = np.argsort(var[candidates])[::-1]
        picked = candidates[order[:target]]
        selected.append(np.sort(picked))
    if selected:
        selected = np.concatenate(selected)
    else:
        selected = np.empty(0, dtype=np.int64)
    if selected.size < min(neurons_per_session, np.sum(region_idx_full >= 0)):
        remaining = np.setdiff1d(np.flatnonzero(region_idx_full >= 0), selected, assume_unique=False)
        order = np.argsort(var[remaining])[::-1]
        need = min(neurons_per_session - selected.size, remaining.size)
        if need > 0:
            selected = np.concatenate([selected, remaining[order[:need]]])
    selected = np.sort(selected.astype(np.int64))
    return selected


def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))


def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))


def build_lick_frame_mask(beh, nfr):
    lick_mask = np.zeros(nfr, dtype=bool)
    lick_fr = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
    lick_mask[lick_idx] = True
    return lick_mask


def convert_session(
    session_idx,
    session,
    canonical_lookup,
    training_days,
    stimulus_to_idx,
    speed_thresholds,
    root,
    neurons_per_session,
    frames_per_bin,
):
    beh_file, beh_key, beh = canonical_lookup[session["base"]]
    spk_path = os.path.join(root, "data", "spk", f"{session['base']}_neural_data.npy")
    ret_path = os.path.join(root, "data", "retinotopy", f"{session['mname']}_{session['datexp']}_trans.npz")

    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=np.int64)

    nfr = min(spk.shape[1], len(beh["ft"]))
    region_idx_full = coarse_region_indices(iarea[: spk.shape[0]])
    frames_by_trial = trial_frame_indices(beh, nfr)
    selected_frames = np.concatenate(frames_by_trial)
    if selected_frames.size == 0:
        raise RuntimeError(f"No running corridor frames found for session {session['base']}")

    selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
    if selected_neurons.size == 0:
        raise RuntimeError(f"No visual-cortex neurons selected for session {session['base']}")
    spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
    brain_region_idx = region_idx_full[selected_neurons].astype(np.int64)

    del spk
    del spk_obj
    gc.collect()

    ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
    lick_mask = build_lick_frame_mask(beh, nfr)
    sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
    trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
    is_rew = np.asarray(beh["isRew"], dtype=bool)
    wall_name = np.asarray(beh["WallName"])
    day_value = np.float32(training_days[session_idx])

    neural_trials = []
    input_trials = []
    output_trials = []
    kept_trial_indices = []
    per_trial_bin_counts = []

    for tr, frame_idx in enumerate(frames_by_trial):
        chunks = chunk_indices(frame_idx, frames_per_bin)
        if not chunks:
            continue
        stim_idx = stimulus_to_idx[str(wall_name[tr])]
        trial_neural = []
        trial_input = []
        trial_output = []
        for chunk in chunks:
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            mean_speed = float(ft_speed[chunk].mean())
            mean_pos = float(ft_pos[chunk].mean())

            trial_neural.append(neural_bin)
            trial_input.append(
                np.array(
                    [
                        np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
                        day_value,
                        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
                        float(is_rew[tr]),
                    ],
                    dtype=np.float32,
                )
            )
            trial_output.append(
                np.array(
                    [
                        stim_idx,
                        int(lick_mask[chunk].any()),
                        position_to_bin(mean_pos),
                        speed_to_bin(mean_speed, speed_thresholds),
                    ],
                    dtype=np.int64,
                )
            )

        neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
        input_trials.append(np.stack(trial_input, axis=1).astype(np.float32))
        output_trials.append(np.stack(trial_output, axis=1).astype(np.int64))
        kept_trial_indices.append(tr)
        per_trial_bin_counts.append(len(chunks))

    session_info = {
        "session_id": session["base"],
        "behavior_file": beh_file,
        "behavior_key": beh_key,
        "subject": session["mname"],
        "date": session["datexp"],
        "block": session["blk"],
        "original_neurons": int(iarea.shape[0]),
        "retained_neurons": int(selected_neurons.size),
        "original_trials": int(beh["ntrials"]),
        "retained_trials": int(len(neural_trials)),
        "mean_bins_per_trial": float(np.mean(per_trial_bin_counts)),
    }

    return neural_trials, input_trials, output_trials, brain_region_idx, session_info


def remap_subjects(subject_names, subject_idx):
    used = sorted(set(int(x) for x in subject_idx.tolist()))
    mapping = {old: new for new, old in enumerate(used)}
    new_subjects = [subject_names[old] for old in used]
    new_subject_idx = np.array([mapping[int(x)] for x in subject_idx], dtype=np.int64)
    return new_subjects, new_subject_idx


def choose_sample_session_indices(full_data, nsessions):
    rows = []
    for i in range(len(full_data["neural"])):
        reward_any = int(any(float(tr[3].max()) > 0 for tr in full_data["input"][i]))
        lick_any = int(any(int(tr[1].max()) > 0 for tr in full_data["output"][i]))
        stim_ids = set()
        for tr in full_data["output"][i]:
            stim_ids.update(np.unique(tr[0]).tolist())
        rows.append(
            {
                "idx": i,
                "subject": full_data["metadata"]["session_info"][i]["subject"],
                "ntrials": len(full_data["neural"][i]),
                "nstim": len(stim_ids),
                "reward_any": reward_any,
                "lick_any": lick_any,
            }
        )

    def pick(predicate, selected, used_subjects):
        candidates = [row for row in rows if predicate(row) and row["idx"] not in selected]
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                row["subject"] in used_subjects,
                -row["nstim"],
                -(row["reward_any"] + row["lick_any"]),
                -row["ntrials"],
                row["idx"],
            )
        )
        return candidates[0]["idx"]

    selected = []
    used_subjects = set()
    predicates = [
        lambda row: row["reward_any"] == 0 and row["nstim"] <= 2,
        lambda row: row["reward_any"] == 1 and row["nstim"] <= 2,
        lambda row: row["reward_any"] == 0 and row["nstim"] >= 4,
        lambda row: row["reward_any"] == 1 and row["nstim"] >= 4,
        lambda row: row["reward_any"] == 0 and row["nstim"] >= 5,
        lambda row: row["reward_any"] == 1 and row["nstim"] >= 5,
    ]
    for predicate in predicates:
        if len(selected) >= nsessions:
            break
        idx = pick(predicate, selected, used_subjects)
        if idx is None:
            continue
        selected.append(idx)
        used_subjects.add(full_data["metadata"]["session_info"][idx]["subject"])

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            -(row["reward_any"] + row["lick_any"]),
            -row["nstim"],
            -row["ntrials"],
            row["idx"],
        ),
    )
    for row in rows_sorted:
        if len(selected) >= nsessions:
            break
        if row["idx"] in selected:
            continue
        selected.append(row["idx"])
        used_subjects.add(row["subject"])

    return np.array(sorted(selected[:nsessions]), dtype=np.int64)


def build_sample_dataset(full_data, nsessions, max_trials_per_session):
    sess_idx = choose_sample_session_indices(full_data, min(nsessions, len(full_data["neural"])))
    sample = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": full_data["subjects"][:],
        "subject_idx": full_data["subject_idx"][sess_idx].copy(),
        "brain_regions": full_data["brain_regions"][:],
        "brain_region_idx": [],
        "input_names": full_data["input_names"][:],
        "output_names": full_data["output_names"][:],
        "output_values": [vals[:] for vals in full_data["output_values"]],
        "metadata": dict(full_data["metadata"]),
    }
    for idx in sess_idx:
        sample["neural"].append(full_data["neural"][idx][:max_trials_per_session])
        sample["input"].append(full_data["input"][idx][:max_trials_per_session])
        sample["output"].append(full_data["output"][idx][:max_trials_per_session])
        sample["brain_region_idx"].append(full_data["brain_region_idx"][idx].copy())
    sample["subjects"], sample["subject_idx"] = remap_subjects(sample["subjects"], sample["subject_idx"])
    sample["metadata"]["sample_subset"] = {
        "n_sessions": int(len(sample["neural"])),
        "max_trials_per_session": int(max_trials_per_session),
        "session_indices_from_full": sess_idx.tolist(),
    }
    sample["metadata"]["session_info"] = [full_data["metadata"]["session_info"][int(i)] for i in sess_idx]
    return sample


def summarize_dataset(data):
    ntrials = np.array([len(x) for x in data["neural"]], dtype=np.int64)
    nneurons = np.array([x.shape[0] for x in data["brain_region_idx"]], dtype=np.int64)
    nbins = []
    for sess in data["neural"]:
        nbins.extend([trial.shape[1] for trial in sess])
    nbins = np.asarray(nbins, dtype=np.int64)
    return {
        "nsessions": int(len(data["neural"])),
        "nsubjects": int(len(data["subjects"])),
        "ntrials_total": int(ntrials.sum()),
        "trials_per_session_min": int(ntrials.min()),
        "trials_per_session_max": int(ntrials.max()),
        "neurons_per_session_min": int(nneurons.min()),
        "neurons_per_session_max": int(nneurons.max()),
        "bins_per_trial_mean": float(nbins.mean()),
        "bins_per_trial_median": float(np.median(nbins)),
        "bins_per_trial_max": int(nbins.max()),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    exp_info = load_exp_info(args.root)
    sessions = collect_unique_sessions(exp_info)
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    canonical_lookup, provenance, duplicate_mismatches = build_behavior_lookup(args.root)
    missing = [s["base"] for s in sessions if s["base"] not in canonical_lookup]
    if missing:
        raise RuntimeError(f"Missing behavior entries for sessions: {missing[:5]}")

    training_days = compute_training_days(sessions)
    frame_dt_ms = compute_frame_dt_ms(canonical_lookup, sessions)
    speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
    stim_names = output_stimulus_names(canonical_lookup, sessions)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}

    subjects = []
    subject_to_idx = {}
    for session in sessions:
        if session["mname"] not in subject_to_idx:
            subject_to_idx[session["mname"]] = len(subjects)
            subjects.append(session["mname"])
    subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)

    print("Reference checks:")
    print(f"  Unique recordings from Imaging_Exp_info.npy: {len(sessions)}")
    print(f"  Unique mice from Imaging_Exp_info.npy: {len(subjects)}")
    print(f"  Duplicate behavior mismatches found: {len(duplicate_mismatches)}")
    print(f"  Global running-speed quartiles: {speed_thresholds.tolist()}")
    print(f"  Global median imaging frame step (ms): {frame_dt_ms:.3f}")
    print(f"  Stimulus vocabulary ({len(stim_names)}): {stim_names}")

    neural = []
    inputs = []
    outputs = []
    brain_region_idx = []
    session_info = []

    for session_idx, session in enumerate(sessions):
        print(f"[{session_idx + 1:02d}/{len(sessions)}] converting {session['base']}")
        sess_neural, sess_input, sess_output, sess_region_idx, sess_info = convert_session(
            session_idx=session_idx,
            session=session,
            canonical_lookup=canonical_lookup,
            training_days=training_days,
            stimulus_to_idx=stim_to_idx,
            speed_thresholds=speed_thresholds,
            root=args.root,
            neurons_per_session=args.neurons_per_session,
            frames_per_bin=args.frames_per_bin,
        )
        neural.append(sess_neural)
        inputs.append(sess_input)
        outputs.append(sess_output)
        brain_region_idx.append(sess_region_idx)
        session_info.append(sess_info)

        print(
            "  retained trials/neuron/bin-mean:",
            sess_info["retained_trials"],
            sess_info["retained_neurons"],
            f"{sess_info['mean_bins_per_trial']:.2f}",
        )
        gc.collect()

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": REGION_NAMES[:],
        "brain_region_idx": brain_region_idx,
        "input_names": [
            "time_to_sound_cue_s",
            "training_day",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            stim_names,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            ["speed_q1", "speed_q2", "speed_q3", "speed_q4"],
        ],
        "metadata": {
            "task_description": "Two-photon calcium imaging from mice running through virtual-reality corridors with natural-image stimuli; decoder predicts stimulus identity, licking, 1 m corridor position bin, and running-speed quartile from neural activity plus task context.",
            "time_bin_size": float(frame_dt_ms * args.frames_per_bin),
            "temporal_alignment_event": "corridor entry / trial start",
            "off_start": 0.0,
            "off_end": None,
            "source_paper": "Unsupervised pretraining in biological neural networks",
            "recordings_reported_in_paper": 89,
            "mice_reported_in_paper": 19,
            "original_neuron_count_range": [20547, 89577],
            "running_only": True,
            "corridor_only": True,
            "frame_selection_rule": "Frames with ft_CorrSpc == True and ft_move > 0, grouped in order within each trial.",
            "time_binning_rule": f"Average every {args.frames_per_bin} retained imaging frames into one decoder time bin.",
            "training_day_rule": "Calendar days since the first imaging session for that mouse.",
            "speed_thresholds": [float(x) for x in speed_thresholds.tolist()],
            "speed_thresholds_definition": "Quartiles of mean ft_RunSpeed over all retained decoder bins in the full dataset.",
            "stimulus_rule": "Per-trial visual category taken directly from beh['WallName'] for each trial.",
            "neuron_selection_rule": f"Retain up to {args.neurons_per_session} retinotopically assigned visual-cortex neurons per session, sampled proportionally across V1/mHV/lHV/aHV and ranked by variance over retained frames.",
            "session_info": session_info,
            "behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
        },
    }

    sample_data = build_sample_dataset(
        full_data=data,
        nsessions=args.sample_sessions,
        max_trials_per_session=args.sample_trials_per_session,
    )

    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.sample_output, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Full dataset summary:", summarize_dataset(data))
    print("Sample dataset summary:", summarize_dataset(sample_data))
    print(f"Saved full dataset to {args.output}")
    print(f"Saved sample dataset to {args.sample_output}")


if __name__ == "__main__":
    main()

import argparse
import gc
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import utils  # noqa: E402


if __name__ == "__main__":
    sys.modules.setdefault("convert_data", sys.modules[__name__])


def dprime(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    u1, u2 = np.nanmean(x1, axis=1), np.nanmean(x2, axis=1)
    s1, s2 = np.nanstd(x1, axis=1), np.nanstd(x2, axis=1)
    denom = s1 + s2
    denom[denom == 0] = np.nan
    out = 2 * (u1 - u2) / denom
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class SessionRef:
    rec_id: str
    subject: str
    date_str: str
    blk: str
    canonical_exp_type: str
    canonical_beh_key: str
    refs: tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zhong et al. imaging data into decoder format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Process all sessions (default).")
    group.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y_%m_%d")


def as_str_array(x) -> np.ndarray:
    return np.asarray([str(v) for v in np.asarray(x).tolist()], dtype=object)


def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()


def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()


def choose_canonical_behavior(
    rec_id: str, refs: list[tuple[str, dict]]
) -> tuple[str, str]:
    candidates = []
    checked = []
    for exp_type, db in refs:
        beh_all = load_beh(exp_type)
        key = f"{rec_id}_{db['stimtype']}" if "stimtype" in db else rec_id
        if key not in beh_all:
            continue
        beh = beh_all[key]
        score = (
            int(np.sum(~np.isnan(np.asarray(beh.get("stim_id", []), dtype=float)))),
            len(np.unique(as_str_array(beh["WallName"]))),
            key,
        )
        candidates.append((score, exp_type, key, beh))
        checked.append((exp_type, key, beh))
    if not candidates:
        raise KeyError(f"No behavior entry found for recording {rec_id}")

    # Validate that duplicate references agree on the fields used for conversion.
    base_beh = candidates[0][3]
    for exp_type, key, beh in checked[1:]:
        same = (
            int(beh["ntrials"]) == int(base_beh["ntrials"])
            and np.array_equal(as_str_array(beh["WallName"]), as_str_array(base_beh["WallName"]))
            and np.array_equal(np.asarray(beh["isRew"]).astype(int), np.asarray(base_beh["isRew"]).astype(int))
            and np.allclose(np.asarray(beh["SoundPos"], dtype=float), np.asarray(base_beh["SoundPos"], dtype=float), equal_nan=True)
            and np.allclose(np.asarray(beh["ft_trInd"], dtype=float), np.asarray(base_beh["ft_trInd"], dtype=float), equal_nan=True)
        )
        if not same:
            raise ValueError(
                f"Duplicate references for {rec_id} are not equivalent between "
                f"{candidates[0][1]}:{candidates[0][2]} and {exp_type}:{key}"
            )

    _, exp_type, key, _ = max(candidates)
    return exp_type, key


def collect_sessions(sample: bool) -> list[SessionRef]:
    exp_info = load_exp_info()
    per_rec: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            per_rec[rec_id].append((exp_type, db))

    sessions = []
    for rec_id, refs in per_rec.items():
        first_db = refs[0][1]
        exp_type, beh_key = choose_canonical_behavior(rec_id, refs)
        sessions.append(
            SessionRef(
                rec_id=rec_id,
                subject=first_db["mname"],
                date_str=first_db["datexp"],
                blk=first_db["blk"],
                canonical_exp_type=exp_type,
                canonical_beh_key=beh_key,
                refs=tuple((exp, db.get("stimtype")) for exp, db in refs),
            )
        )

    sessions.sort(key=lambda s: (s.subject, parse_date(s.date_str), int(s.blk)))
    if sample:
        representative = []
        for sess in sessions:
            beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
            reward_frac = float(np.mean(np.asarray(beh["isRew"]).astype(float)))
            has_licks = len(np.asarray(beh["LickFr"])) > 0
            if reward_frac > 0 and has_licks:
                representative.append(sess)
            if len(representative) == 2:
                break
        if len(representative) == 2:
            return representative
        return sessions[:2]
    return sessions


def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    by_subject: dict[str, list[SessionRef]] = defaultdict(list)
    for sess in sessions:
        by_subject[sess.subject].append(sess)

    day_map: dict[str, dict[str, float]] = defaultdict(dict)
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
        for sess in sess_list:
            day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
    return day_map


def get_familiar_pair(uniq_walls: np.ndarray, wall_name: np.ndarray, is_rew: np.ndarray) -> tuple[str, str]:
    uniq = [str(v) for v in uniq_walls.tolist()]
    base = sorted([w for w in uniq if w.endswith("1") and "swap" not in w])
    if len(base) >= 2:
        if np.any(is_rew):
            reward_stim = str(wall_name[is_rew][0])
            if reward_stim in base:
                other = [w for w in base if w != reward_stim][0]
                return reward_stim, other
        return base[0], base[1]
    if len(uniq) >= 2:
        uniq_sorted = sorted(uniq)
        return uniq_sorted[0], uniq_sorted[1]
    raise ValueError(f"Could not determine familiar pair from walls {uniq}")


def percentile_union_mask(values: np.ndarray, valid_mask: np.ndarray, hi: float, lo: float) -> np.ndarray:
    if np.sum(valid_mask) == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    thr_hi, thr_lo = np.percentile(values[valid_mask], [hi, lo])
    return ((values >= thr_hi) | (values <= thr_lo)) & valid_mask


def compute_selected_neurons(
    spk_chunks: list[np.ndarray],
    beh: dict,
    iarea: np.ndarray,
    session_id: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    spk = np.concatenate(spk_chunks, axis=0)
    areas = utils.neu_area_ID(iarea)

    wall_name = as_str_array(beh["WallName"])
    uniq_walls = as_str_array(beh["UniqWalls"])
    is_rew = np.asarray(beh["isRew"]).astype(bool)

    stim_pos, stim_neg = get_familiar_pair(uniq_walls, wall_name, is_rew)

    nfr = spk.shape[1]
    ft_wall = as_str_array(beh["ft_WallID"][:nfr])
    ft_trial = np.asarray(beh["ft_trInd"][:nfr], dtype=float)
    ft_trial_int = np.full(ft_trial.shape, -1, dtype=np.int64)
    finite_trial = np.isfinite(ft_trial)
    ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
    ft_move = np.asarray(beh["ft_move"][:nfr], dtype=float) > 0
    ft_corr = np.asarray(beh["ft_CorrSpc"][:nfr]).astype(bool)
    ft_gray = np.asarray(beh["ft_GraySpc"][:nfr]).astype(bool)
    odd_train = finite_trial & ((ft_trial_int % 2) == 0)
    corr_train = ft_corr & ft_move & odd_train
    gray_train = ft_gray & ft_move & odd_train

    stim_pos_fr = (ft_wall == stim_pos) & corr_train
    stim_neg_fr = (ft_wall == stim_neg) & corr_train
    if stim_pos_fr.sum() == 0 or stim_neg_fr.sum() == 0:
        raise ValueError(f"{session_id}: familiar-pair frame masks are empty for {stim_pos} vs {stim_neg}")

    stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
    corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
        spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)
    )

    mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)

    if int(np.sum(mhv_mask)) < 128:
        mhv_candidates = np.where(corr_neu & areas["mHV"])[0]
        if len(mhv_candidates) > 0:
            order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
            take = mhv_candidates[order[: min(128, len(order))]]
            mhv_mask = np.zeros_like(mhv_mask)
            mhv_mask[take] = True

    ahv_mask = np.zeros_like(mhv_mask)
    if np.any(is_rew) and np.sum(areas["aHV"]) > 0:
        poscum = np.asarray(beh["ft_PosCum"][:nfr], dtype=float)
        move_idx = np.where(ft_move)[0]
        poscum_move = poscum[move_idx]
        if np.all(np.diff(poscum_move) >= 0):
            ahv_idx = np.where(areas["aHV"])[0]
            interp_spk = utils.get_interpPos_spk(
                spk[ahv_idx][:, move_idx],
                poscum_move,
                int(beh["ntrials"]),
                n_bins=60,
                lengths=float(beh["Corridor_Length"]),
            )
            mean_corr = interp_spk[:, :, 5:40].mean(axis=2)
            cue_pos = np.mod(np.asarray(beh["SoundDelPos"], dtype=float), float(beh["Corridor_Length"]))
            stim_trials = wall_name == stim_pos
            early = (cue_pos <= np.nanmean(cue_pos)) & stim_trials
            late = (cue_pos > np.nanmean(cue_pos)) & stim_trials
            if np.sum(early) > 0 and np.sum(late) > 0:
                reward_dp = dprime(mean_corr[:, late], mean_corr[:, early])
                local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)
                ahv_mask[ahv_idx[local_keep]] = True

    keep_mask = mhv_mask | ahv_mask
    if int(np.sum(keep_mask)) == 0:
        # Final fallback: take strongest mHV neurons by absolute familiar-pair selectivity.
        mhv_candidates = np.where(areas["mHV"])[0]
        order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
        take = mhv_candidates[order[: min(128, len(order))]]
        keep_mask[take] = True

    region_names = np.empty(np.sum(keep_mask), dtype=object)
    kept_indices = np.where(keep_mask)[0]
    region_names[:] = "mHV"
    region_names[np.isin(kept_indices, np.where(ahv_mask)[0])] = "aHV"
    region_idx = np.array([0 if name == "mHV" else 1 for name in region_names], dtype=np.int64)

    stats = {
        "n_total": int(len(iarea)),
        "n_mhv_selected": int(np.sum(mhv_mask)),
        "n_ahv_selected": int(np.sum(ahv_mask)),
        "n_selected": int(np.sum(keep_mask)),
        "stim_pos": stim_pos,
        "stim_neg": stim_neg,
    }
    return kept_indices.astype(np.int64), region_idx, stats


def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ft_trial_int = np.full(ft_trial.shape, -1, dtype=np.int64)
    finite_trial = np.isfinite(ft_trial)
    ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    masks = []
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        if len(frame_idx) == 0:
            raise ValueError(f"Trial {trial} has no retained running corridor frames")
        masks.append(frame_idx)
    return masks


def session_processing_summary(
    session: SessionRef,
    beh: dict,
    kept_stats: dict,
    trial_masks: list[np.ndarray],
    speed_edges: np.ndarray,
    sample_trial_neural: np.ndarray,
    sample_trial_input: np.ndarray,
    sample_trial_output: np.ndarray,
    sample_trial_idx: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_counts = np.array([len(x) for x in trial_masks])
    raw_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
    sample_frames = trial_masks[sample_trial_idx]
    ft = np.asarray(beh["ft"], dtype=float)
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
    lick_trial_frames = lick_fr[lick_tr == sample_trial_idx]

    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax = ax.ravel()

    ax[0].bar(["total", "selected"], [kept_stats["n_total"], kept_stats["n_selected"]], color=["0.7", "tab:blue"])
    ax[0].set_title(f"{session.rec_id}: neuron curation")
    ax[0].set_ylabel("count")
    ax[0].text(0.98, 0.95, f"mHV: {kept_stats['n_mhv_selected']}\naHV: {kept_stats['n_ahv_selected']}",
               transform=ax[0].transAxes, ha="right", va="top")

    ax[1].hist(frame_counts, bins=30, color="tab:green")
    ax[1].set_title("Retained frame counts / trial")
    ax[1].set_xlabel("frames")

    ax[2].plot(sample_frames, np.asarray(beh["ft_Pos"], dtype=float)[sample_frames], color="tab:purple", lw=1.5)
    ax[2].axhline(10, color="0.8", ls="--")
    ax[2].axhline(20, color="0.8", ls="--")
    ax[2].axhline(30, color="0.8", ls="--")
    cue_idx = int(np.argmin(np.abs(ft[sample_frames] - float(beh["SoundTime"][sample_trial_idx]))))
    ax[2].axvline(sample_frames[cue_idx], color="tab:red", ls="--", label="cue")
    for lf in lick_trial_frames:
        if sample_frames[0] <= lf <= sample_frames[-1]:
            ax[2].axvline(lf, color="tab:orange", alpha=0.2)
    ax[2].set_title(f"Sample trial {sample_trial_idx}: retained frames")
    ax[2].set_xlabel("raw frame index")
    ax[2].set_ylabel("position (dm)")
    ax[2].legend(loc="upper left")

    sample_neurons = np.linspace(0, sample_trial_neural.shape[0] - 1, min(64, sample_trial_neural.shape[0]), dtype=int)
    ax[3].imshow(sample_trial_neural[sample_neurons], aspect="auto", interpolation="nearest", cmap="magma")
    ax[3].set_title("Sample selected-neuron activity")
    ax[3].set_xlabel("time bin")
    ax[3].set_ylabel("sampled neurons")

    ax[4].plot(sample_trial_input[0], label="time_to_cue")
    ax[4].plot(sample_trial_input[2], label="time_since_start")
    ax[4].plot(sample_trial_output[1], label="lick", alpha=0.8)
    ax[4].set_title("Sample trial inputs / licking")
    ax[4].legend(loc="upper right")

    kept_speed = raw_speed[np.concatenate(trial_masks)]
    ax[5].hist(kept_speed, bins=50, color="0.6")
    for edge in speed_edges:
        ax[5].axvline(edge, color="tab:red", ls="--")
    ax[5].set_title("Running speed with quartile edges")
    ax[5].set_xlabel("ft_RunSpeed")

    fig.tight_layout()
    out_path = ROOT / f"processing_{session.rec_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_global_metadata(sessions: list[SessionRef]) -> tuple[list[str], dict[str, int], dict[str, float], np.ndarray]:
    subjects = []
    subject_to_idx = {}
    for sess in sessions:
        if sess.subject not in subject_to_idx:
            subject_to_idx[sess.subject] = len(subjects)
            subjects.append(sess.subject)

    day_map = subject_day_map(sessions)
    dts = []
    speed_values = []
    category_values = set()
    for sess in sessions:
        beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
        ft = np.asarray(beh["ft"], dtype=float)
        dts.append(np.median(np.diff(ft)) * 86400.0)
        trial_masks = compute_trial_masks(beh)
        speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
        category_values.update(str(v) for v in np.asarray(beh["WallName"]).tolist())

    speed_values = np.concatenate(speed_values).astype(np.float32)
    speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
    return subjects, subject_to_idx, day_map, speed_edges


def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)


def convert_dataset(sessions: list[SessionRef], show_processing: bool) -> dict:
    subjects, subject_to_idx, day_map, speed_edges = build_global_metadata(sessions)
    categories = sorted(
        {
            str(v)
            for sess in sessions
            for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
        }
    )
    category_to_idx = {name: idx for idx, name in enumerate(categories)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int64),
        "brain_regions": ["mHV", "aHV"],
        "brain_region_idx": [],
        "input_names": [
            "time_to_sound_cue",
            "day_of_training",
            "time_since_trial_start",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            categories,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            ["q1", "q2", "q3", "q4"],
        ],
        "metadata": {
            "task_description": (
                "Decode visual stimulus category, licking, corridor position, and running speed "
                "from deconvolved two-photon activity during running in the corridor."
            ),
            "time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
            "temporal_alignment_event": "corridor entry (trial start)",
            "off_start": 0.0,
            "off_end": None,
            "session_ids": [s.rec_id for s in sessions],
            "neuron_selection": (
                "Reference-style subset: union of mHV familiar-stimulus selective neurons "
                "(top/bottom 5% d' on odd running corridor frames) and aHV reward-prediction neurons "
                "(d'late-vs-early >= 0.3 among positively stimulus-selective neurons)."
            ),
            "trial_filter": "Retain only running corridor frames: ft_CorrSpc & (ft_move > 0).",
            "speed_bin_edges": [float(x) for x in speed_edges.tolist()],
        },
    }

    processing_plotted = 0
    t_all = time.perf_counter()
    for sess_idx, sess in enumerate(sessions):
        t0 = time.perf_counter()
        beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
        spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
        spk_chunks = list(spk_obj["spks"])
        ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
        iarea = np.asarray(ret["iarea"])

        kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
        chunk_local_rows = []
        start = 0
        for chunk in spk_chunks:
            end = start + chunk.shape[0]
            in_chunk = (kept_idx >= start) & (kept_idx < end)
            chunk_local_rows.append((kept_idx[in_chunk] - start).astype(np.int64, copy=False))
            start = end

        trial_masks = compute_trial_masks(beh)
        ft = np.asarray(beh["ft"], dtype=float)
        trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
        sound_time = np.asarray(beh["SoundTime"], dtype=float)
        lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
        lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
        wall_name = as_str_array(beh["WallName"])
        is_rew = np.asarray(beh["isRew"], dtype=float)
        ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
        ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
        day_value = np.float32(day_map[sess.subject][sess.rec_id])

        session_neural = []
        session_input = []
        session_output = []

        for trial_idx, frame_idx in enumerate(trial_masks):
            trial_chunks = []
            for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
                if len(local_rows) == 0:
                    continue
                trial_chunks.append(chunk[local_rows][:, frame_idx])
            neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)

            current_ft = ft[frame_idx]
            time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
            time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
            reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
            day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)

            trial_input = np.vstack(
                [time_to_cue, day_of_training, time_since_start, reward_available]
            ).astype(np.float32, copy=False)

            lick_trial_frames = lick_fr[lick_tr == trial_idx]
            licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
            stim_code = np.full(
                len(frame_idx),
                category_to_idx[str(wall_name[trial_idx])],
                dtype=np.int16,
            )
            pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
            pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
            speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
            trial_output = np.vstack([stim_code, licking, pos_bin, speed_bin]).astype(np.int16, copy=False)

            session_neural.append(neural_trial)
            session_input.append(trial_input)
            session_output.append(trial_output)

        data["neural"].append(session_neural)
        data["input"].append(session_input)
        data["output"].append(session_output)
        data["brain_region_idx"].append(region_idx)

        elapsed = time.perf_counter() - t0
        total_tp = sum(arr.shape[1] for arr in session_neural)
        print(
            f"[{sess_idx + 1}/{len(sessions)}] {sess.rec_id}: "
            f"{kept_stats['n_selected']} neurons, {len(session_neural)} trials, "
            f"{total_tp} retained timepoints, {elapsed:.1f}s"
        )

        if show_processing and processing_plotted < 2:
            sample_trial_idx = int(np.argmax([x.shape[1] for x in session_neural]))
            session_processing_summary(
                session=sess,
                beh=beh,
                kept_stats=kept_stats,
                trial_masks=trial_masks,
                speed_edges=speed_edges,
                sample_trial_neural=session_neural[sample_trial_idx],
                sample_trial_input=session_input[sample_trial_idx],
                sample_trial_output=session_output[sample_trial_idx],
                sample_trial_idx=sample_trial_idx,
            )
            processing_plotted += 1

        del spk_obj, spk_chunks, ret
        gc.collect()

    print(f"Conversion complete in {time.perf_counter() - t_all:.1f}s")
    return data


def main() -> None:
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True

    sessions = collect_sessions(sample=args.sample)
    print(f"Selected {len(sessions)} sessions")
    data = convert_dataset(sessions, show_processing=args.show_processing)

    out_path = ROOT / args.outpicklefile
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

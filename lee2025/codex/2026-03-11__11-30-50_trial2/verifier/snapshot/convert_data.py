#!/usr/bin/env python3

import argparse
import gc
import os
import pickle
import time
from dataclasses import dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


FPS = 30
SESSION_SECONDS = 40 * 60
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
POOL_SIZE = 3
TRACE_SMOOTH_SIGMA = 3
VELOCITY_SMOOTH_SIGMA = 5
VELOCITY_THRESHOLD_CM_S = 5.0
CELL_EVENT_THRESHOLD = 5
GEOMETRY_SIZE = 3
BRAIN_REGION = "CA1"
MIN_POOLED_SAMPLES_PER_TRIAL = 10


def identity(x):
    return x


TRANSFORMS = {
    "identity": identity,
    "rot90": lambda x: np.rot90(x, 1),
    "rot180": lambda x: np.rot90(x, 2),
    "rot270": lambda x: np.rot90(x, 3),
    "flipud": np.flipud,
    "fliplr": np.fliplr,
    "transpose": np.transpose,
    "anti_transpose": lambda x: np.fliplr(np.flipud(np.transpose(x))),
}


@dataclass
class SessionPlotPayload:
    session_id: str
    animal: str
    session_in_animal: int
    env_name: str
    blocked_idx: np.ndarray
    geometry_open: np.ndarray
    transform_name: str
    transform_scores: dict
    scale_cm: float
    raw_position: np.ndarray
    velocity_mask: np.ndarray
    valid_cell_count: int
    active_cell_count: int
    raw_trial_neural: np.ndarray
    raw_trial_position: np.ndarray
    trial_velocity_mask: np.ndarray
    pooled_trial_neural: np.ndarray
    pooled_trial_output: np.ndarray
    occupancy_raw: np.ndarray
    occupancy_aligned: np.ndarray
    occupancy_snapped: np.ndarray
    trial_lengths: list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the CA1 geometry dataset into the decoder pickle format."
    )
    parser.add_argument("outpicklefile", help="Output pickle filename.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions.",
    )
    return parser.parse_args()


def get_animal_files(data_dir):
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if (
            os.path.isfile(path)
            and not name.endswith(".mat")
            and name not in {"behav_dict"}
            and not name.startswith(".")
        ):
            animals.append(name)
    return animals


def parse_blocked_indices(blocked_entry):
    if isinstance(blocked_entry, list):
        if len(blocked_entry) == 0:
            return np.array([], dtype=np.int64)
        blocked_entry = blocked_entry[0]
    arr = np.array(blocked_entry, dtype=np.float64).reshape(-1)
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
    return np.sort(arr.astype(np.int64))


def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx


def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
    y = np.clip(y, 0, GEOMETRY_SIZE - 1)
    return y, x


def trial_average_pool(values, pool_size=POOL_SIZE):
    n_full = values.shape[-1] // pool_size
    if n_full <= 0:
        return values[..., :0]
    trimmed = values[..., : n_full * pool_size]
    new_shape = values.shape[:-1] + (n_full, pool_size)
    return trimmed.reshape(new_shape).mean(axis=-1)


def compute_velocity_mask(position_xy, fps=FPS, threshold_cm_s=VELOCITY_THRESHOLD_CM_S):
    velocity_mask = np.zeros(position_xy.shape[1], dtype=bool)
    if position_xy.shape[1] < 2:
        return velocity_mask
    delta = np.diff(position_xy, axis=1)
    speed = np.linalg.norm(delta, axis=0) * fps
    speed = gaussian_filter1d(speed, sigma=VELOCITY_SMOOTH_SIGMA, mode="nearest")
    velocity_mask[1:] = speed > threshold_cm_s
    return velocity_mask


def occupancy_matrix(position_xy, bin_size_cm):
    rows, cols = raw_position_to_rc(position_xy, bin_size_cm)
    occ = np.zeros((GEOMETRY_SIZE, GEOMETRY_SIZE), dtype=np.int64)
    np.add.at(occ, (rows, cols), 1)
    return occ


def build_coord_map(transform_name):
    template = np.arange(GEOMETRY_SIZE * GEOMETRY_SIZE).reshape(GEOMETRY_SIZE, GEOMETRY_SIZE)
    transformed = TRANSFORMS[transform_name](template)
    coord_map = np.zeros(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.int64)
    for new_idx, old_idx in enumerate(transformed.ravel()):
        coord_map[int(old_idx)] = new_idx
    return coord_map


def infer_transform(position_all, blocked_all, scale_cm):
    bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
    occupancies = [occupancy_matrix(position_all[sess], bin_size_cm) for sess in range(position_all.shape[0])]
    scores = {}
    for name, transform_fn in TRANSFORMS.items():
        blocked_mass = 0.0
        open_mass = 0.0
        for sess, occ in enumerate(occupancies):
            open_vec, _ = blocked_to_open_vector(blocked_all[sess])
            open_mat = open_vec.reshape(GEOMETRY_SIZE, GEOMETRY_SIZE)
            occ_aligned = transform_fn(occ)
            blocked_mass += float(occ_aligned[open_mat == 0].sum())
            open_mass += float(occ_aligned[open_mat == 1].sum())
        total = blocked_mass + open_mass
        blocked_frac = blocked_mass / total if total > 0 else np.inf
        scores[name] = {
            "blocked_mass": blocked_mass,
            "open_mass": open_mass,
            "blocked_fraction": blocked_frac,
        }
    best_name = sorted(
        scores,
        key=lambda name: (scores[name]["blocked_fraction"], -scores[name]["open_mass"], name),
    )[0]
    return best_name, scores


def snap_to_open_bins(row_idx, col_idx, geometry_open):
    geometry_mat = geometry_open.reshape(GEOMETRY_SIZE, GEOMETRY_SIZE)
    open_coords = np.argwhere(geometry_mat == 1)
    if open_coords.size == 0:
        raise ValueError("Geometry has no open bins.")
    row_idx = row_idx.copy()
    col_idx = col_idx.copy()
    for i in range(row_idx.shape[0]):
        if geometry_mat[row_idx[i], col_idx[i]] == 1:
            continue
        distances = np.sum((open_coords - np.array([row_idx[i], col_idx[i]])) ** 2, axis=1)
        nearest = open_coords[np.argmin(distances)]
        row_idx[i] = nearest[0]
        col_idx[i] = nearest[1]
    return row_idx, col_idx


def preprocess_session(
    trace_session,
    position_session,
    blocked_entry,
    env_name,
    scale_cm,
    coord_map,
    session_id,
    animal,
    session_idx,
):
    geometry_open, blocked_idx = blocked_to_open_vector(blocked_entry)
    geometry_mat = geometry_open.reshape(GEOMETRY_SIZE, GEOMETRY_SIZE)
    bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
    velocity_mask = compute_velocity_mask(position_session)

    finite_cells = np.isfinite(trace_session).all(axis=1)
    trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
    if trace_finite.size == 0:
        raise ValueError(f"{session_id}: no finite cells after NaN filtering")

    activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
    trace_active = trace_finite[activity_mask]
    if trace_active.shape[0] == 0:
        raise ValueError(f"{session_id}: no active cells after activity filtering")

    raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
    raw_ids_all = raw_rows_all * GEOMETRY_SIZE + raw_cols_all
    mapped_ids_all = coord_map[raw_ids_all]
    mapped_rows_all = mapped_ids_all // GEOMETRY_SIZE
    mapped_cols_all = mapped_ids_all % GEOMETRY_SIZE
    snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
    occupancy_raw = occupancy_matrix(position_session, bin_size_cm)

    occupancy_aligned = np.zeros_like(occupancy_raw)
    np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
    occupancy_snapped = np.zeros_like(occupancy_raw)
    np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)

    n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
    neural_trials = []
    input_trials = []
    output_trials = []
    trial_lengths = []

    plot_payload = None
    for trial_idx in range(n_full_trials):
        start = trial_idx * RAW_TRIAL_FRAMES
        end = start + RAW_TRIAL_FRAMES
        chunk_mask = velocity_mask[start:end]
        if int(chunk_mask.sum()) < POOL_SIZE:
            continue

        chunk_trace = trace_active[:, start:end][:, chunk_mask]
        chunk_rows = snapped_rows_all[start:end][chunk_mask]
        chunk_cols = snapped_cols_all[start:end][chunk_mask]

        chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
        pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
        pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
        pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)

        if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
            continue

        if not np.any(pooled_trace):
            continue

        pooled_rows = np.clip(pooled_rows, 0, GEOMETRY_SIZE - 1)
        pooled_cols = np.clip(pooled_cols, 0, GEOMETRY_SIZE - 1)
        invalid = geometry_mat[pooled_rows, pooled_cols] == 0
        if np.any(invalid):
            pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
        pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)

        neural_trials.append(pooled_trace)
        input_trials.append(geometry_open.copy())
        output_trials.append(pooled_bins)
        trial_lengths.append(int(pooled_trace.shape[1]))

        if plot_payload is None:
            plot_payload = SessionPlotPayload(
                session_id=session_id,
                animal=animal,
                session_in_animal=session_idx,
                env_name=env_name,
                blocked_idx=blocked_idx,
                geometry_open=geometry_open.copy(),
                transform_name="",
                transform_scores={},
                scale_cm=float(scale_cm),
                raw_position=position_session.copy(),
                velocity_mask=velocity_mask.copy(),
                valid_cell_count=int(finite_cells.sum()),
                active_cell_count=int(trace_active.shape[0]),
                raw_trial_neural=trace_active[:, start:end].copy(),
                raw_trial_position=position_session[:, start:end].copy(),
                trial_velocity_mask=chunk_mask.copy(),
                pooled_trial_neural=pooled_trace.copy(),
                pooled_trial_output=pooled_bins.copy(),
                occupancy_raw=occupancy_raw.copy(),
                occupancy_aligned=occupancy_aligned.copy(),
                occupancy_snapped=occupancy_snapped.copy(),
                trial_lengths=[],
            )

    if len(neural_trials) < 2:
        raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")

    return neural_trials, input_trials, output_trials, plot_payload, trial_lengths, int(finite_cells.sum()), int(trace_active.shape[0]), velocity_mask


def save_processing_plot(payload):
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.ravel()

    geometry_mat = payload.geometry_open.reshape(GEOMETRY_SIZE, GEOMETRY_SIZE)
    velocity_fraction = float(payload.velocity_mask.mean())

    info_lines = [
        f"Session: {payload.session_id}",
        f"Animal: {payload.animal}",
        f"Session index: {payload.session_in_animal}",
        f"Environment: {payload.env_name}",
        f"Blocked bins: {payload.blocked_idx.tolist() if payload.blocked_idx.size else [-1]}",
        f"Transform: {payload.transform_name}",
        f"Scale (cm): {payload.scale_cm:.3f}",
        f"Cells finite/active: {payload.valid_cell_count}/{payload.active_cell_count}",
        f"Movement-valid frames: {velocity_fraction:.3f}",
        f"Exported trial lengths (pooled): {payload.trial_lengths[:5]}{'...' if len(payload.trial_lengths) > 5 else ''}",
    ]
    score_lines = []
    for name, stats in sorted(payload.transform_scores.items()):
        score_lines.append(f"{name}: blocked={stats['blocked_fraction']:.4f}")
    axes[0].axis("off")
    axes[0].text(
        0.0,
        1.0,
        "\n".join(info_lines + ["", "Transform blocked-occupancy fractions:"] + score_lines),
        va="top",
        family="monospace",
    )

    im = axes[1].imshow(geometry_mat, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Geometry Input (1=open, 0=blocked)")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            axes[1].text(c, r, f"{idx}\n{int(geometry_mat[r, c])}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    axes[2].imshow(payload.occupancy_raw, cmap="magma")
    axes[2].set_title("Raw 3x3 Occupancy")
    axes[2].set_xticks(range(3))
    axes[2].set_yticks(range(3))

    axes[3].imshow(payload.occupancy_snapped, cmap="magma")
    axes[3].set_title("Aligned/Snapped 3x3 Occupancy")
    axes[3].set_xticks(range(3))
    axes[3].set_yticks(range(3))

    t_raw = np.arange(payload.raw_trial_position.shape[1]) / FPS
    axes[4].plot(t_raw, payload.raw_trial_position[0], label="x")
    axes[4].plot(t_raw, payload.raw_trial_position[1], label="y")
    axes[4].fill_between(
        t_raw,
        payload.raw_trial_position.min(),
        payload.raw_trial_position.max(),
        where=payload.trial_velocity_mask,
        alpha=0.2,
        color="tab:green",
        label="movement-valid",
    )
    axes[4].set_title("Raw Trial Position")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="upper right")

    t_pooled = np.arange(payload.pooled_trial_output.shape[1]) * (POOL_SIZE / FPS)
    axes[5].step(t_pooled, payload.pooled_trial_output[0], where="post")
    axes[5].set_title("Pooled Position Bin Output")
    axes[5].set_xlabel("Pooled valid-frame time (s)")
    axes[5].set_ylabel("Bin index")
    axes[5].set_ylim(-0.5, 8.5)

    neural_show = payload.raw_trial_neural[: min(40, payload.raw_trial_neural.shape[0]), :300]
    axes[6].imshow(neural_show, aspect="auto", interpolation="nearest", cmap="binary")
    axes[6].set_title("Raw Neural Activity (first 40 cells, 300 frames)")
    axes[6].set_xlabel("Frame")
    axes[6].set_ylabel("Cell")

    pooled_show = payload.pooled_trial_neural[: min(40, payload.pooled_trial_neural.shape[0]), :100]
    axes[7].imshow(pooled_show, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[7].set_title("Processed Neural Activity (first 40 cells, 100 pooled samples)")
    axes[7].set_xlabel("Pooled sample")
    axes[7].set_ylabel("Cell")

    fig.tight_layout()
    fig.savefig(f"processing_{payload.session_id}.png", dpi=150)
    plt.close(fig)


def build_output_values():
    values = []
    for row in range(GEOMETRY_SIZE):
        for col in range(GEOMETRY_SIZE):
            idx = row * GEOMETRY_SIZE + col
            values.append(f"bin_{idx}_r{row}c{col}")
    return [values]


def main():
    args = parse_args()
    full_mode = not args.sample
    data_dir = os.path.join(os.getcwd(), "data")
    animals = get_animal_files(data_dir)
    if not animals:
        raise FileNotFoundError("No animal joblib files found under data/.")

    if args.sample:
        selected_animals = [animals[0]]
        max_sessions_global = 2
    else:
        selected_animals = animals
        max_sessions_global = None

    print(f"Found animals: {selected_animals}")
    print(f"Mode: {'sample' if args.sample else 'full'}")
    print(
        "Reference preprocessing: movement filtering, session-level active-cell filtering, "
        "Gaussian smoothing, 3-frame pooling."
    )

    all_neural = []
    all_input = []
    all_output = []
    subject_idx = []
    brain_region_idx = []
    session_info = []
    processing_payloads = []

    total_trials = 0
    total_cells = 0
    total_valid_frames = 0
    total_raw_frames = 0
    sessions_processed = 0
    global_session_counter = 0
    started = time.time()

    for animal in selected_animals:
        animal_start = time.time()
        animal_path = os.path.join(data_dir, animal)
        print(f"Loading {animal_path}")
        animal_load_start = time.time()
        dat = joblib.load(animal_path)[animal]
        load_seconds = time.time() - animal_load_start
        print(f"  loaded in {load_seconds:.2f}s")

        scale_cm = float(np.nanmax(dat["position"]))
        transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
        coord_map = build_coord_map(transform_name)
        print(f"  selected transform {transform_name} for {animal}")

        for session_idx in range(dat["position"].shape[0]):
            if max_sessions_global is not None and global_session_counter >= max_sessions_global:
                break

            session_id = f"{animal}_s{session_idx:02d}"
            env_name = str(dat["envs"][session_idx, 0])
            print(f"  processing {session_id} ({env_name})")

            session_neural, session_input, session_output, plot_payload, trial_lengths, n_finite, n_active, velocity_mask = preprocess_session(
                trace_session=dat["trace"][session_idx],
                position_session=dat["position"][session_idx],
                blocked_entry=dat["blocked"][session_idx],
                env_name=env_name,
                scale_cm=scale_cm,
                coord_map=coord_map,
                session_id=session_id,
                animal=animal,
                session_idx=session_idx,
            )

            if plot_payload is not None:
                plot_payload.transform_name = transform_name
                plot_payload.transform_scores = transform_scores
                plot_payload.trial_lengths = trial_lengths
                if len(processing_payloads) < 2 and args.show_processing:
                    processing_payloads.append(plot_payload)

            all_neural.append(session_neural)
            all_input.append(session_input)
            all_output.append(session_output)
            subject_idx.append(animals.index(animal))
            brain_region_idx.append(np.zeros(n_active, dtype=np.int64))

            session_info.append(
                {
                    "session_id": session_id,
                    "animal": animal,
                    "session_in_animal": session_idx,
                    "environment": env_name,
                    "blocked_bins": parse_blocked_indices(dat["blocked"][session_idx]).tolist(),
                    "transform_name": transform_name,
                    "scale_cm": scale_cm,
                    "n_trials": len(session_neural),
                    "n_cells_finite": n_finite,
                    "n_cells_active": n_active,
                    "raw_frames": int(dat["trace"][session_idx].shape[1]),
                    "movement_valid_fraction": float(velocity_mask.mean()),
                    "trial_lengths_pooled": trial_lengths,
                }
            )

            total_trials += len(session_neural)
            total_cells += n_active
            total_valid_frames += sum(trial_lengths) * POOL_SIZE
            total_raw_frames += int(dat["trace"][session_idx].shape[1])
            sessions_processed += 1
            global_session_counter += 1

            mean_trial_len = float(np.mean(trial_lengths))
            print(
                f"    trials={len(session_neural)} active_cells={n_active} "
                f"mean_pooled_len={mean_trial_len:.2f}"
            )

        animal_seconds = time.time() - animal_start
        print(f"Finished {animal} in {animal_seconds:.2f}s")
        del dat
        gc.collect()

        if max_sessions_global is not None and global_session_counter >= max_sessions_global:
            break

    if args.show_processing:
        for payload in processing_payloads:
            save_processing_plot(payload)
            print(f"Saved processing plot processing_{payload.session_id}.png")

    data = {
        "neural": all_neural,
        "input": all_input,
        "output": all_output,
        "subjects": animals,
        "subject_idx": np.array(subject_idx, dtype=np.int64),
        "brain_regions": [BRAIN_REGION],
        "brain_region_idx": brain_region_idx,
        "input_names": [f"partition_{idx}_open" for idx in range(GEOMETRY_SIZE * GEOMETRY_SIZE)],
        "output_names": ["position_bin"],
        "output_values": build_output_values(),
        "metadata": {
            "task_description": (
                "Decode movement-period mouse position in a 3x3 arena partition grid from CA1 calcium activity, "
                "with static geometric context per one-minute chunk."
            ),
            "time_bin_size": 100.0,
            "temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session",
            "off_start": 0.0,
            "off_end": 60.0,
            "source_paper": "Lee, Keinath, Cianfarano and Brandon (2025), Neuron 113(2):307-320",
            "native_frame_rate_hz": FPS,
            "native_session_duration_s": SESSION_SECONDS,
            "raw_trial_duration_s": TRIAL_SECONDS,
            "movement_filtered": True,
            "velocity_threshold_cm_s": VELOCITY_THRESHOLD_CM_S,
            "velocity_smoothing_sigma_frames": VELOCITY_SMOOTH_SIGMA,
            "cell_event_threshold": CELL_EVENT_THRESHOLD,
            "trace_smoothing_sigma_frames": TRACE_SMOOTH_SIGMA,
            "temporal_pool_size_frames": POOL_SIZE,
            "geometry_input_encoding": "9 binary features in row-major 3x3 order, 1=open 0=blocked",
            "position_output_encoding": "single categorical variable with 9 row-major 3x3 bins after snapping invalid blocked bins to nearest open bin",
            "session_info": session_info,
        },
    }

    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - started
    print(f"Saved {args.outpicklefile}")
    print(
        f"Summary: sessions={sessions_processed}, trials={total_trials}, "
        f"mean_active_cells_per_session={total_cells / max(sessions_processed, 1):.2f}, "
        f"raw_to_valid_frame_fraction={(total_valid_frames / max(total_raw_frames, 1)):.4f}"
    )
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

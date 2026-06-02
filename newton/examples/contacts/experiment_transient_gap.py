# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# H7 — Transient-compliance contact-reduction gap (the sqrt(N/K) law).
#
# Drops a flat cube onto a hydroelastic plate and measures the maximum
# first-contact penetration for the dense vs reduced contact sets. The
# hypothesis (H7_transient_compliance_hypothesis.md) is that reduction preserves
# the static resultant but not transient compliance, with
#     delta_max(dense) / delta_max(reduced) = sqrt(N / K).
#
# Generation only: writes CSVs. Interpretation lives in
# tools/transient_gap_report.py.
#
# Command:
#   uv run --extra examples python -m \
#     newton.examples.contacts.experiment_transient_gap
###########################################################################

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

from newton.examples.contacts.experiment_cube_on_plate_impact import _ImpactRun
from newton.examples.contacts.experiment_cube_on_plate_settle import (
    BUFFER_FRACTION,
    BUFFER_MULT_CONTACT,
    BUFFER_MULT_ISO,
    MUJOCO_NCONMAX,
    MUJOCO_NJMAX,
    RIGID_CONTACT_MAX,
    ModeConfig,
    SceneConfig,
    _load_scene_config,
)

DEFAULT_CONFIG = "newton/examples/contacts/configs/cube_on_plate_baseline.yaml"
DEFAULT_OUTPUT_DIR = Path("output") / "H7_transient_gap"
GRAVITY = 9.81

# Drop heights kept small so the DENSE penetration stays inside the +/-5 mm SDF
# narrow band (probe: 2 mm drop -> 5.75 mm dense penetration, already out of band;
# delta ~ sqrt(h), so <=1 mm stays in band).
DEFAULT_HEIGHTS_M = (0.00025, 0.0005, 0.001)
DEFAULT_RESOLUTIONS = (16, 24, 32, 48)
DEFAULT_RES_HEIGHT_M = 0.0005

STEP_DT = 0.00025
RUN_SECONDS = 0.30
FINAL_TILT_LIMIT_DEG = 0.5
FINAL_DRIFT_LIMIT_M = 1.0e-4

SUMMARY_CSV = "transient_gap_summary.csv"
TIMESERIES_CSV = "transient_gap_timeseries.csv"

SUMMARY_COLUMNS = (
    "sweep",
    "sweep_value",
    "mode",
    "reduce_contacts",
    "sdf_resolution",
    "drop_height_m",
    "impact_velocity_m_per_s",
    "face_count_N",
    "rigid_count_K",
    "N_over_K",
    "max_penetration_m",
    "final_penetration_m",
    "final_fz_N",
    "final_fz_over_weight",
    "settled_time_s",
    "peak_fz_N",
    "peak_fz_over_weight",
    "final_tilt_deg",
    "final_drift_m",
    "buffer_overflow",
    "state_invalid",
    "in_band",
    "valid",
)

TIMESERIES_COLUMNS = (
    "sweep_value",
    "mode",
    "time_s",
    "cube_z_m",
    "cube_vz_m_per_s",
    "penetration_m",
    "solver_fz_N",
    "rigid_contact_count",
    "face_contact_count",
)


def _cube_mass(scene: SceneConfig) -> float:
    return scene.cube_density * (2.0 * scene.cube_half_extent) ** 3


def _cube_tilt_deg_from_row_q(body_q: np.ndarray) -> float:
    qx, qy = float(body_q[3]), float(body_q[4])
    local_z_z = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    return math.degrees(math.acos(local_z_z))


def run_drop(
    *,
    scene: SceneConfig,
    height: float,
    mode: ModeConfig,
    run_seconds: float,
    step_dt: float,
    device: str | None,
    collect_timeseries: bool,
) -> dict[str, object]:
    """Drop the cube once; return transient + static observables."""
    run = _ImpactRun(
        scene=scene,
        height=height,
        mode=mode,
        rigid_contact_max=RIGID_CONTACT_MAX,
        nconmax=MUJOCO_NCONMAX,
        njmax=MUJOCO_NJMAX,
        buffer_mult_iso=BUFFER_MULT_ISO,
        buffer_mult_contact=BUFFER_MULT_CONTACT,
        buffer_fraction=BUFFER_FRACTION,
        device=device,
    )
    n_steps = int(round(run_seconds / step_dt))
    band_hi = float(scene.sdf_narrow_band_range[1])

    max_pen = 0.0
    peak_fz = 0.0
    impact_v = 0.0
    contacted = False
    max_face = 0
    max_rigid = 0
    overflow = False
    state_invalid = False
    final_pen = 0.0
    final_fz = 0.0
    final_tilt = 0.0
    final_drift = 0.0
    # settle detection: body velocity below tol after contact, sustained
    settled_time = float("nan")
    timeseries: list[dict[str, object]] = []

    for i in range(n_steps):
        fd = run.simulate_step(step_dt=step_dt, time_s=i * step_dt)
        r = fd.timeseries_row
        pen = float(r["cube_penetration_depth_m"])
        fz = float(r["solver_fz_N"])
        vz = float(r["cube_vz_m_per_s"])
        if not contacted:
            impact_v = max(impact_v, -vz)
            if pen > 0.0:
                contacted = True
        max_pen = max(max_pen, pen)
        peak_fz = max(peak_fz, fz)
        max_face = max(max_face, int(r["face_contact_count"]))
        max_rigid = max(max_rigid, int(r["rigid_contact_count"]))
        overflow = overflow or bool(r["buffer_overflow"])
        state_invalid = state_invalid or bool(r["state_invalid"])
        final_pen = pen
        final_fz = fz
        final_drift = math.hypot(float(r["cube_x_m"]), float(r["cube_y_m"]))
        final_tilt = float(r["cube_tilt_deg"])
        if contacted and abs(vz) < 1.0e-4 and settled_time != settled_time:
            settled_time = i * step_dt
        if collect_timeseries:
            timeseries.append(
                {
                    "time_s": i * step_dt,
                    "mode": mode.name,
                    "cube_z_m": float(r["cube_z_m"]),
                    "cube_vz_m_per_s": vz,
                    "penetration_m": pen,
                    "solver_fz_N": fz,
                    "rigid_contact_count": int(r["rigid_contact_count"]),
                    "face_contact_count": int(r["face_contact_count"]),
                }
            )

    n_over_k = (max_face / max_rigid) if max_rigid > 0 else float("nan")
    in_band = max_pen < band_hi
    valid = (
        (not overflow)
        and (not state_invalid)
        and contacted
        and peak_fz > 0.0
        and final_tilt < FINAL_TILT_LIMIT_DEG
        and final_drift < FINAL_DRIFT_LIMIT_M
        and ((not mode.reduce_contacts) <= in_band)  # dense must be in band; reduced exempt
    )
    # dense in-band is mandatory; reduced reading is always trusted (it is shallow).
    if not mode.reduce_contacts:
        valid = valid and in_band

    return {
        "max_penetration_m": max_pen,
        "final_penetration_m": final_pen,
        "final_fz_N": final_fz,
        "settled_time_s": settled_time,
        "peak_fz_N": peak_fz,
        "impact_velocity_m_per_s": impact_v,
        "face_count_N": max_face,
        "rigid_count_K": max_rigid,
        "N_over_K": n_over_k,
        "final_tilt_deg": final_tilt,
        "final_drift_m": final_drift,
        "buffer_overflow": overflow,
        "state_invalid": state_invalid,
        "in_band": in_band,
        "valid": bool(valid),
        "_timeseries": timeseries,
    }


def _summary_row(
    *,
    sweep: str,
    sweep_value: float | int,
    scene: SceneConfig,
    height: float,
    mode: ModeConfig,
    weight: float,
    obs: dict[str, object],
) -> dict[str, object]:
    return {
        "sweep": sweep,
        "sweep_value": sweep_value,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "sdf_resolution": scene.sdf_max_resolution,
        "drop_height_m": height,
        "impact_velocity_m_per_s": obs["impact_velocity_m_per_s"],
        "face_count_N": obs["face_count_N"],
        "rigid_count_K": obs["rigid_count_K"],
        "N_over_K": obs["N_over_K"],
        "max_penetration_m": obs["max_penetration_m"],
        "final_penetration_m": obs["final_penetration_m"],
        "final_fz_N": obs["final_fz_N"],
        "final_fz_over_weight": float(obs["final_fz_N"]) / weight if weight > 0 else float("nan"),
        "settled_time_s": obs["settled_time_s"],
        "peak_fz_N": obs["peak_fz_N"],
        "peak_fz_over_weight": float(obs["peak_fz_N"]) / weight if weight > 0 else float("nan"),
        "final_tilt_deg": obs["final_tilt_deg"],
        "final_drift_m": obs["final_drift_m"],
        "buffer_overflow": obs["buffer_overflow"],
        "state_invalid": obs["state_invalid"],
        "in_band": obs["in_band"],
        "valid": obs["valid"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _print_pair(height_mm: float, dense: dict, reduced: dict) -> None:
    dpen = dense["max_penetration_m"] * 1e6
    rpen = reduced["max_penetration_m"] * 1e6
    ratio = dpen / rpen if rpen > 0 else float("nan")
    nk = reduced["N_over_K"]
    pred = math.sqrt(nk) if nk and nk == nk else float("nan")
    print(
        f"  h={height_mm:5.2f}mm | dense δ={dpen:8.1f}µm (band_ok={dense['in_band']}) "
        f"reduced δ={rpen:7.1f}µm | ratio={ratio:5.2f}  √(N/K)={pred:5.2f}  "
        f"N/K={nk:5.1f}  peakFz d/r={dense['peak_fz_N']:.1f}/{reduced['peak_fz_N']:.1f}N"
    )


def run_sweep_height(scene, heights, modes, weight, args, summary_rows, ts_rows):
    print("\n=== SWEEP 1: drop height (velocity) — ratio should be ~constant √(N/K) ===")
    primary_idx = len(heights) // 2
    for hi, height in enumerate(heights):
        results = {}
        for label, mode in modes.items():
            obs = run_drop(
                scene=scene,
                height=height,
                mode=mode,
                run_seconds=args.run_seconds,
                step_dt=args.step_dt,
                device=args.device,
                collect_timeseries=(hi == primary_idx),
            )
            results[label] = obs
            summary_rows.append(
                _summary_row(
                    sweep="height",
                    sweep_value=height,
                    scene=scene,
                    height=height,
                    mode=mode,
                    weight=weight,
                    obs=obs,
                )
            )
            if hi == primary_idx:
                for tr in obs["_timeseries"]:
                    tr["sweep_value"] = height
                    ts_rows.append(tr)
        _print_pair(height * 1000.0, results["dense"], results["reduced"])


def run_sweep_resolution(scene_base, resolutions, height, modes, weight, args, summary_rows):
    print("\n=== SWEEP 2: SDF resolution — ratio should track √(N/K) as N grows ===")
    for res in resolutions:
        scene = dataclasses.replace(scene_base, sdf_max_resolution=int(res))
        results = {}
        for label, mode in modes.items():
            obs = run_drop(
                scene=scene,
                height=height,
                mode=mode,
                run_seconds=args.run_seconds,
                step_dt=args.step_dt,
                device=args.device,
                collect_timeseries=False,
            )
            results[label] = obs
            summary_rows.append(
                _summary_row(
                    sweep="resolution",
                    sweep_value=res,
                    scene=scene,
                    height=height,
                    mode=mode,
                    weight=weight,
                    obs=obs,
                )
            )
        dpen = results["dense"]["max_penetration_m"] * 1e6
        rpen = results["reduced"]["max_penetration_m"] * 1e6
        ratio = dpen / rpen if rpen > 0 else float("nan")
        nk = results["reduced"]["N_over_K"]
        pred = math.sqrt(nk) if nk and nk == nk else float("nan")
        print(
            f"  res={res:3d} | N={results['reduced']['face_count_N']:5d} K={results['reduced']['rigid_count_K']:4d} "
            f"| dense δ={dpen:8.1f}µm (band_ok={results['dense']['in_band']}) reduced δ={rpen:7.1f}µm "
            f"| ratio={ratio:5.2f}  √(N/K)={pred:5.2f}"
        )


def run_sweep_static(scene, height, modes, weight, args, summary_rows):
    """Long settle from a near-zero drop: control that both modes reach the same
    static equilibrium (support force -> weight), proving the net force / 0th
    moment is preserved and the gap is purely transient."""
    print("\n=== CONTROL: static settle — both modes should support weight (net force preserved) ===")
    for label, mode in modes.items():
        obs = run_drop(
            scene=scene,
            height=height,
            mode=mode,
            run_seconds=args.static_seconds,
            step_dt=args.step_dt,
            device=args.device,
            collect_timeseries=False,
        )
        summary_rows.append(
            _summary_row(
                sweep="static",
                sweep_value=height,
                scene=scene,
                height=height,
                mode=mode,
                weight=weight,
                obs=obs,
            )
        )
        print(
            f"  {label:8s}: final_Fz={float(obs['final_fz_N']):7.3f}N (x weight "
            f"{float(obs['final_fz_N']) / weight:5.2f})  final_pen={float(obs['final_penetration_m']) * 1e6:8.1f}µm  "
            f"settled_t={float(obs['settled_time_s']):.4f}s  max_pen={float(obs['max_penetration_m']) * 1e6:8.1f}µm"
        )


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H7 transient-compliance reduction gap")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--heights", type=float, nargs="+", default=list(DEFAULT_HEIGHTS_M))
    p.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    p.add_argument("--res-height", type=float, default=DEFAULT_RES_HEIGHT_M)
    p.add_argument("--run-seconds", type=float, default=RUN_SECONDS)
    p.add_argument("--static-seconds", type=float, default=3.0)
    p.add_argument("--static-height", type=float, default=0.0001)
    p.add_argument("--step-dt", type=float, default=STEP_DT)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--sweeps",
        nargs="+",
        default=["static", "height", "resolution"],
        choices=["static", "height", "resolution"],
    )
    return p


def run_experiment(
    *,
    config: str = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    heights: tuple[float, ...] = DEFAULT_HEIGHTS_M,
    resolutions: tuple[int, ...] = DEFAULT_RESOLUTIONS,
    res_height: float = DEFAULT_RES_HEIGHT_M,
    run_seconds: float = RUN_SECONDS,
    static_seconds: float = 3.0,
    static_height: float = 0.0001,
    step_dt: float = STEP_DT,
    device: str | None = None,
    sweeps: tuple[str, ...] = ("static", "height", "resolution"),
    verbose: bool = True,
) -> Path:
    """Run the transient-gap sweeps and write the summary/time-series CSVs.

    Args:
        config: Path to the scene YAML config.
        output_dir: Directory the CSV files are written to.
        heights: Drop heights [m] for the velocity-independence sweep.
        resolutions: SDF resolutions for the ``sqrt(N/K)`` sweep.
        res_height: Drop height [m] used for the resolution sweep.
        run_seconds: Impact-run duration [s].
        static_seconds: Static-control settle duration [s].
        static_height: Drop height [m] for the static control.
        step_dt: Solver/log step [s].
        device: Warp device override, or ``None`` for the default.
        sweeps: Which sweeps to run (``static``/``height``/``resolution``).
        verbose: Print per-row progress lines.

    Returns:
        The directory the CSV files were written to.
    """

    args = argparse.Namespace(
        config=config,
        output_dir=str(output_dir),
        heights=list(heights),
        resolutions=list(resolutions),
        res_height=res_height,
        run_seconds=run_seconds,
        static_seconds=static_seconds,
        static_height=static_height,
        step_dt=step_dt,
        device=device,
        sweeps=list(sweeps),
    )
    scene = _load_scene_config(args.config)
    weight = _cube_mass(scene) * GRAVITY
    modes = {
        "dense": ModeConfig(name="unreduced", reduce_contacts=False),
        "reduced": ModeConfig(name="reduced", reduce_contacts=True),
    }
    if verbose:
        print(
            f"scene: mass={_cube_mass(scene):.3f}kg weight={weight:.3f}N kh={scene.kh:.2e} "
            f"band=±{scene.sdf_narrow_band_range[1] * 1000:.0f}mm base_res={scene.sdf_max_resolution}"
        )

    summary_rows: list[dict[str, object]] = []
    ts_rows: list[dict[str, object]] = []

    if "static" in args.sweeps:
        run_sweep_static(scene, args.static_height, modes, weight, args, summary_rows)
    if "height" in args.sweeps:
        run_sweep_height(scene, args.heights, modes, weight, args, summary_rows, ts_rows)
    if "resolution" in args.sweeps:
        run_sweep_resolution(scene, args.resolutions, args.res_height, modes, weight, args, summary_rows)

    out = Path(args.output_dir)
    _write_csv(out / SUMMARY_CSV, summary_rows, SUMMARY_COLUMNS)
    if ts_rows:
        _write_csv(out / TIMESERIES_CSV, ts_rows, TIMESERIES_COLUMNS)
    if verbose:
        print(f"\nwrote {out / SUMMARY_CSV}  ({len(summary_rows)} rows)")
        if ts_rows:
            print(f"wrote {out / TIMESERIES_CSV}  ({len(ts_rows)} rows)")
    return out


def main() -> None:
    args = create_parser().parse_args()
    run_experiment(
        config=args.config,
        output_dir=args.output_dir,
        heights=tuple(args.heights),
        resolutions=tuple(args.resolutions),
        res_height=args.res_height,
        run_seconds=args.run_seconds,
        static_seconds=args.static_seconds,
        static_height=args.static_height,
        step_dt=args.step_dt,
        device=args.device,
        sweeps=tuple(args.sweeps),
    )


if __name__ == "__main__":
    main()

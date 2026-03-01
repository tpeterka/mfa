#!/usr/bin/env python3
"""Practical 2D CT pipeline for Walnut and PCIR manuscript examples.

This script prepares a real 2D image slice, runs MFA forward projection
(`line_integral --sinogram`) in both spline and trapezoid modes, reconstructs
with scikit-image, and writes figure/metric artifacts for manuscript use.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


WALNUT_GT_URL = (
    "https://zenodo.org/records/1254206/files/GroundTruthReconstruction.mat?download=1"
)
PCIR_ZIP_URL = (
    "https://zenodo.org/records/18225140/files/PCIR_98890234_20010101_7.zip?download=1"
)


@dataclass
class PreparedSlice:
    dataset: str
    source_path: Path
    image: np.ndarray
    raw_path: Path
    metadata: dict[str, Any]


@dataclass
class SinogramGrid:
    alpha_rad: np.ndarray
    rho: np.ndarray
    values: np.ndarray


def import_required(module_name: str, pip_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency '{pip_name}'. Install with: python -m pip install {pip_name}"
        ) from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def download_if_needed(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing file: {destination}")
        return destination

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Downloaded to: {destination}")
    return destination


def select_2d_slice(array: np.ndarray, slice_index: Optional[int]) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim == 2:
        return squeezed.astype(np.float32)

    if squeezed.ndim == 3:
        axis = int(np.argmin(squeezed.shape))
        if slice_index is None:
            index = squeezed.shape[axis] // 2
        else:
            index = max(0, min(slice_index, squeezed.shape[axis] - 1))
        return np.take(squeezed, index, axis=axis).astype(np.float32)

    raise ValueError(
        f"Expected 2D or 3D array after squeeze, got shape {squeezed.shape}"
    )


def robust_normalize(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError("Image has no finite values")

    finite = data[finite_mask]
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))

    if high <= low:
        normalized = np.zeros_like(data, dtype=np.float32)
    else:
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)

    normalized[~finite_mask] = 0.0
    return normalized.astype(np.float32)


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    transform = import_required("skimage.transform", "scikit-image")
    resized = transform.resize(
        image,
        (size, size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.asarray(resized, dtype=np.float32)


def save_grayscale_png(path: Path, image: np.ndarray, title: str) -> None:
    plt = import_required("matplotlib.pyplot", "matplotlib")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(image, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_walnut_slice(
    mat_path: Path, slice_index: Optional[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    scipy_io = import_required("scipy.io", "scipy")
    mat = scipy_io.loadmat(str(mat_path))
    preferred_keys = [
        "GroundTruthReconstruction",
        "groundTruthReconstruction",
        "ground_truth_reconstruction",
        "reconstruction",
        "Recon",
    ]

    chosen_key = None
    chosen_array = None
    for key in preferred_keys:
        value = mat.get(key)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            chosen_key = key
            chosen_array = value
            break

    if chosen_array is None:
        candidates: list[tuple[int, str, np.ndarray]] = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.number)
                and value.size > 0
            ):
                candidates.append((value.size, key, value))

        if not candidates:
            raise ValueError(f"No numeric MATLAB arrays found in {mat_path}")

        candidates.sort(reverse=True)
        _, chosen_key, chosen_array = candidates[0]

    slice_2d = select_2d_slice(chosen_array, slice_index)
    metadata = {
        "mat_key": chosen_key,
        "mat_shape": list(np.squeeze(chosen_array).shape),
    }
    return slice_2d, metadata


def load_pcir_slice(
    zip_path: Path, slice_index: Optional[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    pydicom = import_required("pydicom", "pydicom")

    slices: list[tuple[bool, float, int, str, np.ndarray]] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        for member in members:
            with archive.open(member, "r") as binary_file:
                ds = pydicom.dcmread(io.BytesIO(binary_file.read()), force=True)
            if "PixelData" not in ds:
                continue

            image = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            image = image * slope + intercept

            has_pos = (
                hasattr(ds, "ImagePositionPatient")
                and len(ds.ImagePositionPatient) >= 3
            )
            z_pos = float(ds.ImagePositionPatient[2]) if has_pos else 0.0
            instance = int(getattr(ds, "InstanceNumber", 0))
            slices.append((not has_pos, z_pos, instance, member, image))

    if not slices:
        raise ValueError(f"No DICOM slices with pixel data found in {zip_path}")

    slices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    if slice_index is None:
        index = len(slices) // 2
    else:
        index = max(0, min(slice_index, len(slices) - 1))

    selected = slices[index]
    metadata = {
        "slice_count": len(slices),
        "selected_member": selected[3],
        "selected_index": index,
        "selected_shape": list(selected[4].shape),
    }
    return selected[4], metadata


def prepare_slice(
    dataset: str,
    source_path: Path,
    output_root: Path,
    image_size: int,
    slice_index: Optional[int],
) -> PreparedSlice:
    dataset_root = output_root / dataset
    prep_dir = dataset_root / "prepared"
    prep_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "walnut":
        raw_image, metadata = load_walnut_slice(source_path, slice_index)
    elif dataset == "pcir":
        raw_image, metadata = load_pcir_slice(source_path, slice_index)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    normalized = robust_normalize(raw_image)
    resized = resize_square(normalized, image_size)

    raw_path = prep_dir / "slice_float32.raw"
    resized.astype(np.float32).tofile(raw_path)
    np.save(prep_dir / "slice.npy", resized)
    save_grayscale_png(
        prep_dir / "slice.png", resized, f"{dataset.upper()} prepared slice"
    )

    metadata.update(
        {
            "source_path": str(source_path),
            "prepared_shape": list(resized.shape),
            "raw_file": str(raw_path),
        }
    )
    write_json(prep_dir / "prepare_metadata.json", metadata)

    return PreparedSlice(
        dataset=dataset,
        source_path=source_path,
        image=resized,
        raw_path=raw_path,
        metadata=metadata,
    )


def resolve_line_integral_binary(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path.resolve()
        raise FileNotFoundError(f"line_integral binary not found: {explicit_path}")

    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "mfa" / "build" / "examples" / "eval" / "line_integral",
        root.parent / "mfa" / "build" / "examples" / "eval" / "line_integral",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    found = shutil.which("line_integral")
    if found:
        return Path(found).resolve()

    raise FileNotFoundError(
        "Could not locate line_integral binary. Pass --line-integral-binary explicitly."
    )


def run_line_integral(
    line_integral_binary: Path,
    raw_slice_path: Path,
    run_dir: Path,
    ndomp: int,
    vars_nctrl: int,
    ray_nctrl: list[int],
    ray_samples: list[int],
    seed: int,
    discrete: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = raw_slice_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Prepared raw slice not found: {input_path}")

    command: list[str] = [
        str(line_integral_binary),
        "--pt_dim",
        "3",
        "--dom_dim",
        "2",
        "--ndomp",
        str(ndomp),
        "--input",
        "phantom",
        "--infile",
        str(input_path),
        "--sinogram",
        "--vars_nctrl",
        str(vars_nctrl),
        "--seed",
        str(seed),
    ]

    for value in ray_nctrl:
        command.extend(["--rv", str(value)])
    for value in ray_samples:
        command.extend(["--rn", str(value)])
    if discrete:
        command.append("--disc_int")

    log_path = run_dir / "line_integral.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=run_dir, check=True, stdout=log, stderr=subprocess.STDOUT
        )


def parse_sinogram(path: Path) -> SinogramGrid:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Sinogram file must have at least 4 columns: {path}")

    alpha = data[:, 0]
    rho = data[:, 1]
    values = data[:, 3]

    alpha_unique = np.unique(alpha)
    rho_unique = np.unique(rho)
    expected = alpha_unique.size * rho_unique.size
    if values.size != expected:
        raise ValueError(
            f"Unexpected sinogram size in {path}: expected {expected}, got {values.size}"
        )

    matrix = values.reshape(alpha_unique.size, rho_unique.size).T
    return SinogramGrid(alpha_rad=alpha_unique, rho=rho_unique, values=matrix)


def prepare_iradon_inputs(grid: SinogramGrid) -> tuple[np.ndarray, np.ndarray]:
    theta_deg = np.rad2deg(grid.alpha_rad.copy())
    sinogram = np.nan_to_num(grid.values.copy(), nan=0.0, posinf=0.0, neginf=0.0)

    if theta_deg.size > 1 and np.isclose(theta_deg[-1], 180.0, atol=1e-9):
        theta_deg = theta_deg[:-1]
        sinogram = sinogram[:, :-1]

    return sinogram, theta_deg


def reconstruct_from_sinogram(
    grid: SinogramGrid,
    output_size: int,
    run_sart: bool,
) -> dict[str, np.ndarray]:
    transform = import_required("skimage.transform", "scikit-image")

    sinogram, theta_deg = prepare_iradon_inputs(grid)
    fbp = transform.iradon(
        sinogram,
        theta=theta_deg,
        output_size=output_size,
        circle=False,
        filter_name="ramp",
    ).astype(np.float32)

    outputs: dict[str, np.ndarray] = {"fbp": fbp}
    if run_sart:
        sart = transform.iradon_sart(sinogram, theta=theta_deg).astype(np.float32)
        if sart.shape != (output_size, output_size):
            sart = resize_square(sart, output_size)
        outputs["sart"] = sart

    return outputs


def compute_metrics(
    reference: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    metrics = import_required("skimage.metrics", "scikit-image")

    if reconstruction.shape != reference.shape:
        reconstruction = resize_square(reconstruction, reference.shape[0])

    ref = robust_normalize(reference)
    rec = robust_normalize(reconstruction)
    diff = rec - ref

    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "psnr": float(metrics.peak_signal_noise_ratio(ref, rec, data_range=1.0)),
        "ssim": float(metrics.structural_similarity(ref, rec, data_range=1.0)),
    }


def find_sinogram_file(run_dir: Path, stem: str) -> Path:
    matches = sorted(run_dir.glob(f"{stem}_gid*.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No sinogram files matching {stem}_gid*.txt in {run_dir}"
        )
    return matches[0]


def save_comparison_figure(
    path: Path,
    dataset: str,
    reference: np.ndarray,
    mfa_grid: SinogramGrid,
    trap_grid: SinogramGrid,
    mfa_recon: np.ndarray,
    trap_recon: np.ndarray,
    mfa_metrics: dict[str, float],
    trap_metrics: dict[str, float],
) -> None:
    plt = import_required("matplotlib.pyplot", "matplotlib")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)

    axes[0, 0].imshow(reference, cmap="gray", origin="lower")
    axes[0, 0].set_title(f"{dataset.upper()} reference slice")

    axes[0, 1].imshow(mfa_grid.values, cmap="gray", origin="lower", aspect="auto")
    axes[0, 1].set_title("MFA sinogram")

    axes[0, 2].imshow(trap_grid.values, cmap="gray", origin="lower", aspect="auto")
    axes[0, 2].set_title("Trapezoid sinogram")

    axes[1, 0].imshow(mfa_recon, cmap="gray", origin="lower")
    axes[1, 0].set_title(
        f"MFA FBP\nRMSE={mfa_metrics['rmse']:.4f}, SSIM={mfa_metrics['ssim']:.4f}"
    )

    axes[1, 1].imshow(trap_recon, cmap="gray", origin="lower")
    axes[1, 1].set_title(
        f"Trapezoid FBP\nRMSE={trap_metrics['rmse']:.4f}, SSIM={trap_metrics['ssim']:.4f}"
    )

    diff = np.abs(robust_normalize(mfa_recon) - robust_normalize(trap_recon))
    axes[1, 2].imshow(diff, cmap="magma", origin="lower")
    axes[1, 2].set_title("|MFA recon - trapezoid recon|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_dataset_pipeline(
    dataset: str,
    source_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n=== Running dataset pipeline: {dataset} ===")
    prepared = prepare_slice(
        dataset=dataset,
        source_path=source_path,
        output_root=args.output_dir,
        image_size=args.image_size,
        slice_index=args.slice_index,
    )

    dataset_root = args.output_dir / dataset
    summary: dict[str, Any] = {
        "dataset": dataset,
        "source_path": str(source_path),
        "prepared": prepared.metadata,
    }

    if args.skip_forward:
        write_json(dataset_root / "summary.json", summary)
        return summary

    if prepared.image.ndim != 2 or prepared.image.shape[0] != prepared.image.shape[1]:
        raise ValueError(
            f"Prepared slice must be square for phantom input, got {prepared.image.shape}"
        )
    ndomp = int(prepared.image.shape[0])

    line_integral_binary = resolve_line_integral_binary(args.line_integral_binary)
    print(f"Using line_integral binary: {line_integral_binary}")

    mfa_dir = dataset_root / "mfa_forward"
    trap_dir = dataset_root / "trapezoid_forward"

    print("Running spline-based MFA forward projection...")
    run_line_integral(
        line_integral_binary=line_integral_binary,
        raw_slice_path=prepared.raw_path,
        run_dir=mfa_dir,
        ndomp=ndomp,
        vars_nctrl=args.vars_nctrl,
        ray_nctrl=args.ray_nctrl,
        ray_samples=args.ray_samples,
        seed=args.seed,
        discrete=False,
    )

    print("Running trapezoid forward projection...")
    run_line_integral(
        line_integral_binary=line_integral_binary,
        raw_slice_path=prepared.raw_path,
        run_dir=trap_dir,
        ndomp=ndomp,
        vars_nctrl=args.vars_nctrl,
        ray_nctrl=args.ray_nctrl,
        ray_samples=args.ray_samples,
        seed=args.seed,
        discrete=True,
    )

    mfa_sino_path = find_sinogram_file(mfa_dir, "sinogram_approx")
    trap_sino_path = find_sinogram_file(trap_dir, "sinogram_approx")
    mfa_grid = parse_sinogram(mfa_sino_path)
    trap_grid = parse_sinogram(trap_sino_path)

    mfa_recons = reconstruct_from_sinogram(
        mfa_grid, args.image_size, run_sart=args.with_sart
    )
    trap_recons = reconstruct_from_sinogram(
        trap_grid, args.image_size, run_sart=args.with_sart
    )

    mfa_metrics = compute_metrics(prepared.image, mfa_recons["fbp"])
    trap_metrics = compute_metrics(prepared.image, trap_recons["fbp"])

    figure_path = dataset_root / f"{dataset}_comparison.png"
    save_comparison_figure(
        path=figure_path,
        dataset=dataset,
        reference=prepared.image,
        mfa_grid=mfa_grid,
        trap_grid=trap_grid,
        mfa_recon=mfa_recons["fbp"],
        trap_recon=trap_recons["fbp"],
        mfa_metrics=mfa_metrics,
        trap_metrics=trap_metrics,
    )

    summary.update(
        {
            "line_integral_binary": str(line_integral_binary),
            "mfa_sinogram": str(mfa_sino_path),
            "trapezoid_sinogram": str(trap_sino_path),
            "mfa_fbp_metrics": mfa_metrics,
            "trapezoid_fbp_metrics": trap_metrics,
            "figure": str(figure_path),
        }
    )

    write_json(dataset_root / "summary.json", summary)
    print(f"Completed {dataset}. Summary: {dataset_root / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Practical CT manuscript pipeline: Walnut-first validation and first-class PCIR example."
        )
    )

    parser.add_argument(
        "command",
        choices=["walnut", "pcir", "manuscript"],
        help="Run one dataset or run manuscript sequence (walnut then pcir).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/practical_ct"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/practical_ct"))
    parser.add_argument(
        "--download", action="store_true", help="Download missing input datasets."
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download even if files exist."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=200,
        help="Square target size for prepared slices (must be >= 2).",
    )
    parser.add_argument("--slice-index", type=int, default=None)

    parser.add_argument("--walnut-gt-mat", type=Path, default=None)
    parser.add_argument("--pcir-zip", type=Path, default=None)

    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only prepare slices, do not run MFA/reconstruction.",
    )
    parser.add_argument("--line-integral-binary", type=Path, default=None)
    parser.add_argument("--vars-nctrl", type=int, default=11)
    parser.add_argument("--ray-nctrl", type=int, nargs=3, default=[20, 20, 20])
    parser.add_argument("--ray-samples", type=int, nargs=3, default=[50, 50, 50])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--with-sart", action="store_true", help="Also compute one SART reconstruction."
    )

    return parser.parse_args()


def resolve_walnut_path(args: argparse.Namespace) -> Path:
    default_path = args.data_dir / "GroundTruthReconstruction.mat"
    path = args.walnut_gt_mat if args.walnut_gt_mat is not None else default_path
    if args.download:
        return download_if_needed(WALNUT_GT_URL, path, force=args.force_download)
    if not path.exists():
        raise FileNotFoundError(
            f"Walnut file not found: {path}. Use --download or --walnut-gt-mat."
        )
    return path


def resolve_pcir_path(args: argparse.Namespace) -> Path:
    default_path = args.data_dir / "PCIR_98890234_20010101_7.zip"
    path = args.pcir_zip if args.pcir_zip is not None else default_path
    if args.download:
        return download_if_needed(PCIR_ZIP_URL, path, force=args.force_download)
    if not path.exists():
        raise FileNotFoundError(
            f"PCIR file not found: {path}. Use --download or --pcir-zip."
        )
    return path


def main() -> None:
    args = parse_args()
    if args.image_size < 2:
        raise ValueError(f"--image-size must be >= 2, got {args.image_size}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "walnut":
        walnut_path = resolve_walnut_path(args)
        summary = run_dataset_pipeline("walnut", walnut_path, args)
        write_json(args.output_dir / "walnut_summary.json", summary)
        return

    if args.command == "pcir":
        pcir_path = resolve_pcir_path(args)
        summary = run_dataset_pipeline("pcir", pcir_path, args)
        write_json(args.output_dir / "pcir_summary.json", summary)
        return

    # Manuscript mode: Walnut first, then PCIR.
    walnut_path = resolve_walnut_path(args)
    walnut_summary = run_dataset_pipeline("walnut", walnut_path, args)

    pcir_path = resolve_pcir_path(args)
    pcir_summary = run_dataset_pipeline("pcir", pcir_path, args)

    combined = {
        "order": ["walnut", "pcir"],
        "walnut": walnut_summary,
        "pcir": pcir_summary,
    }
    write_json(args.output_dir / "manuscript_summary.json", combined)
    print(
        f"Wrote combined manuscript summary: {args.output_dir / 'manuscript_summary.json'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise

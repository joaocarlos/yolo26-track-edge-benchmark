"""
NCNN export pipeline: .pt → <model>_<res>_ncnn/ folder

Runs in the project venv (ultralytics 8.4.x + torch 2.10.x).
Must run on an x86 desktop — PNNX does not have an ARM binary.

Setup:
    source .venv/bin/activate
    pip install ncnn pnnx

Usage:
    # Export default set (n/s/m at 640 and 576):
    python edge/export_ncnn.py

    # Export a single model at one resolution:
    python edge/export_ncnn.py --model yolo26n.pt --imgsz 640

    # Export larger variants (l/x — not in the default set):
    python edge/export_ncnn.py --models yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt

    # Override resolutions:
    python edge/export_ncnn.py --resolutions 640 576 512

    # Additionally produce INT8-quantized folders (post-training quantization):
    python edge/export_ncnn.py --int8

Outputs:
    models/yolo26n_640_ncnn_model/model.ncnn.param      # FP32
    models/yolo26n_640_ncnn_model/model.ncnn.bin
    models/yolo26n_576_ncnn_model/...
    (one folder per model × resolution combination)

    With --int8, each FP32 folder is additionally quantized to:
    models/yolo26n_int8_640_ncnn_model/...              # INT8
    The "_int8_<res>_ncnn_model" layout keeps the device_profile.baked_imgsz()
    parser and the NCNN autobackend loader working unchanged.

INT8 quantization (PNNX has no INT8 path; this uses the ncnn CLI tools):
    PNNX/ultralytics export FP32/FP16 only. INT8 is a separate post-export step
    via ncnn2table (KL-divergence calibration over representative frames) followed
    by ncnn2int8. Both binaries ship in the official ncnn release and live under
    edge/export/tools/. Calibration frames are drawn from MOT17 sequences held out
    from the evaluation set (MOT17-02/-09/-04) to avoid leakage.

Notes:
    - PNNX is an x86-only binary. This script cannot run on ARM devices.
    - The input resolution is baked into model.ncnn.param at export time.
      Each output folder is valid for exactly one resolution. Passing a different
      imgsz= at inference is a silent correctness error.
    - Ultralytics cleans up PNNX artifacts automatically since 8.4.x; this
      script asserts the two inference-critical files survive regardless.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Script lives at edge/export/ — project root is two levels up.
_ROOT       = Path(__file__).parents[2]
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default export set — conservative for 4 GB ARM devices (RPi 4).
# Extend via --models to add l/x variants for RPi 5 (8 GB).
_DEFAULT_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]

_DEFAULT_RESOLUTIONS = [640, 576]

# INT8 quantization toolchain — official ncnn release binaries (not in the pip
# wheel; PNNX cannot quantize). Extracted under edge/export/tools/.
_TOOLS_DIR  = Path(__file__).parent / "tools"
_NCNN2TABLE = _TOOLS_DIR / "ncnn2table"
_NCNN2INT8  = _TOOLS_DIR / "ncnn2int8"

# Calibration frames are drawn from MOT17 sequences NOT in the evaluation set
# (eval = MOT17-02/-09/-04) so the INT8 table never sees evaluated data.
# One detector variant per sequence number is enough — img1/ is byte-identical
# across the DPM/FRCNN/SDP variants.
_DATA_DIR           = _ROOT / "data" / "MOT17" / "train"
_DEFAULT_CALIB_SEQS = ["MOT17-05-FRCNN", "MOT17-10-FRCNN",
                       "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
_DEFAULT_CALIB_PER_SEQ = 125            # ~500 frames total across four sequences

# YOLO preprocessing for ncnn2table: RGB pixel order, zero mean, 1/255 scaling —
# matches Ultralytics' NCNN inference path so calibration statistics align with
# the activations seen at runtime.
_YOLO_NORM = 1.0 / 255.0

# Artifacts produced alongside the two inference-critical files (param + bin).
# Not needed on the device; removing them reduces transfer size.
_ARTIFACTS_TO_REMOVE = [
    "model.pnnx.param",
    "model.pnnx.bin",
    "model.pnnx.onnx",
    "model_ncnn.py",
    "model_pnnx.py",
    "model.pt",
    "__pycache__",
]


def _check_filesystem() -> None:
    """
    Guard against cross-device link errors during PNNX binary download.

    Ultralytics downloads the PNNX binary at export time and installs it via
    os.rename(). If the working directory and the Python environment are on
    different filesystems, os.rename() raises [Errno 18] and the export fails.
    """
    import site
    site_pkg = site.getsitepackages()[0]

    result_cwd  = subprocess.run(["df", "--output=source", "."],
                                 capture_output=True, text=True)
    result_site = subprocess.run(["df", "--output=source", site_pkg],
                                 capture_output=True, text=True)

    dev_cwd  = result_cwd.stdout.strip().splitlines()[-1]
    dev_site = result_site.stdout.strip().splitlines()[-1]

    if dev_cwd != dev_site:
        print(
            f"ERROR: cross-device link risk.\n"
            f"  Working dir filesystem : {dev_cwd}\n"
            f"  Python venv filesystem : {dev_site}\n"
            f"  Run the export from a directory on the same filesystem as the venv.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[env] filesystem check OK  ({dev_cwd})")


def export_ncnn(model_name: str, res: int, half: bool = False) -> Path | None:
    """
    Export a single YOLO26 .pt model to NCNN format for ARM Cortex-A Linux targets.

    The input resolution is baked into model.ncnn.param at export time. Every
    inference call against the resulting folder must pass imgsz=res to match.
    Passing a different resolution is a silent correctness error.

    half=True produces an FP16 model in a "<stem>_fp16_<res>_ncnn_model" folder;
    half=False produces the FP32 default in "<stem>_<res>_ncnn_model". The "_fp16"
    infix sits before the resolution so device_profile.baked_imgsz() still parses
    the resolution from the last token, exactly as for the INT8 layout.

    Returns the output directory path on success, None if the .pt is not found.
    """
    from ultralytics import YOLO

    pt_path = _MODELS_DIR / model_name
    if not pt_path.exists():
        print(f"[SKIP] {model_name} not found at {pt_path}")
        return None

    stem  = pt_path.stem                           # e.g. "yolo26n"
    infix = "_fp16" if half else ""
    dst   = _MODELS_DIR / f"{stem}{infix}_{res}_ncnn_model"

    if dst.exists() and (dst / "model.ncnn.param").exists() and (dst / "model.ncnn.bin").exists():
        print(f"[skip] {dst.name} already exists — delete to re-export")
        return dst

    prec = "FP16" if half else "FP32"
    print(f"\n--- {model_name} @ {res}px ({prec}) ---")
    model = YOLO(str(pt_path))

    # Ultralytics names the exported folder <stem>_ncnn_model/ in the working dir.
    # Using imgsz as int (not list) produces a square input, which is what we need.
    exported_raw = Path(model.export(format="ncnn", imgsz=res, half=half))

    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(exported_raw), dst)

    # Strip artifacts not required for inference
    for name in _ARTIFACTS_TO_REMOVE:
        p = dst / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    # Hard assertion — if these are missing the export was silently broken
    assert (dst / "model.ncnn.param").exists(), f"model.ncnn.param missing in {dst}"
    assert (dst / "model.ncnn.bin").exists(),   f"model.ncnn.bin missing in {dst}"

    remaining = sorted(f.name for f in dst.iterdir())
    print(f"  OK → {dst.name}  {remaining}")
    return dst


def build_calib_list(seqs: list[str], per_seq: int, dst_txt: Path) -> int:
    """
    Write a newline-delimited list of calibration image paths for ncnn2table.

    Frames are evenly subsampled across each sequence so the calibration set
    spans the full temporal range rather than clustering at the start. Sequences
    must be disjoint from the evaluation set to prevent calibration leakage.

    Returns the total number of frames written.
    """
    paths: list[str] = []
    for seq in seqs:
        img_dir = _DATA_DIR / seq / "img1"
        if not img_dir.is_dir():
            print(f"[calib] WARNING: {img_dir} not found — skipping")
            continue
        frames = sorted(img_dir.glob("*.jpg"))
        if not frames:
            print(f"[calib] WARNING: no frames in {img_dir} — skipping")
            continue
        step = max(1, len(frames) // per_seq)
        sampled = frames[::step][:per_seq]
        paths.extend(str(p) for p in sampled)
        print(f"[calib] {seq}: {len(sampled)} / {len(frames)} frames")

    dst_txt.write_text("\n".join(paths) + "\n")
    print(f"[calib] wrote {len(paths)} frames → {dst_txt.name}")
    return len(paths)


def quantize_int8(model_name: str, res: int, calib_list: Path,
                  threads: int) -> Path | None:
    """
    Quantize an existing FP32 NCNN folder to INT8 via ncnn2table + ncnn2int8.

    The FP32 folder must already exist (run export_ncnn first). The INT8 output
    folder follows the "<stem>_int8_<res>_ncnn_model" layout so the resolution
    parser and the NCNN autobackend loader treat it like any other NCNN model.

    Returns the INT8 folder path, or None if the FP32 source is missing.
    """
    stem      = Path(model_name).stem                       # e.g. "yolo26n"
    fp32_dir  = _MODELS_DIR / f"{stem}_{res}_ncnn_model"
    fp32_param = fp32_dir / "model.ncnn.param"
    fp32_bin   = fp32_dir / "model.ncnn.bin"

    if not (fp32_param.exists() and fp32_bin.exists()):
        print(f"[int8] SKIP {stem}@{res}: FP32 source missing at {fp32_dir.name}")
        return None

    int8_dir = _MODELS_DIR / f"{stem}_int8_{res}_ncnn_model"
    int8_dir.mkdir(parents=True, exist_ok=True)

    # Calibration table is intermediate provenance — keep it out of the
    # deployment folder so only inference files (param/bin/metadata) ship.
    tables_dir = _MODELS_DIR / "calib_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table = tables_dir / f"{stem}_{res}.table"

    # Stage 1: KL-divergence calibration table from representative activations
    table_cmd = [
        str(_NCNN2TABLE), str(fp32_param), str(fp32_bin), str(calib_list),
        str(table),
        "mean=[0.0,0.0,0.0]",
        f"norm=[{_YOLO_NORM},{_YOLO_NORM},{_YOLO_NORM}]",
        f"shape=[{res},{res},3]",
        "pixel=RGB",
        f"thread={threads}",
        "method=kl",
    ]
    print(f"\n[int8] {stem}@{res} — ncnn2table ({threads} threads)")
    subprocess.run(table_cmd, check=True)

    # Stage 2: apply the table to produce the INT8 param/bin
    int8_param = int8_dir / "model.ncnn.param"
    int8_bin   = int8_dir / "model.ncnn.bin"
    int8_cmd = [
        str(_NCNN2INT8), str(fp32_param), str(fp32_bin),
        str(int8_param), str(int8_bin), str(table),
    ]
    print(f"[int8] {stem}@{res} — ncnn2int8")
    subprocess.run(int8_cmd, check=True)

    # metadata.yaml is required by the Ultralytics NCNN autobackend loader
    shutil.copy2(fp32_dir / "metadata.yaml", int8_dir / "metadata.yaml")

    assert int8_param.exists() and int8_bin.exists(), \
        f"ncnn2int8 did not produce param/bin in {int8_dir}"
    size_mb = int8_bin.stat().st_size / (1024 * 1024)
    fp32_mb = fp32_bin.stat().st_size / (1024 * 1024)
    print(f"[int8] OK → {int8_dir.name}  ({fp32_mb:.1f}MB FP32 → {size_mb:.1f}MB INT8)")
    return int8_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 .pt models to NCNN folders for ARM Cortex-A targets"
    )
    parser.add_argument(
        "--model",
        help="Single .pt filename, e.g. yolo26n.pt (overrides --models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_DEFAULT_MODELS,
        metavar="MODEL",
        help=f"List of .pt filenames (default: {_DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Single resolution override (overrides --resolutions)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=_DEFAULT_RESOLUTIONS,
        metavar="RES",
        help=f"List of resolutions to export (default: {_DEFAULT_RESOLUTIONS})",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Additionally produce FP16 folders (<stem>_fp16_<res>_ncnn_model).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Additionally produce INT8-quantized folders via ncnn2table + ncnn2int8.",
    )
    parser.add_argument(
        "--calib-seqs",
        nargs="+",
        default=_DEFAULT_CALIB_SEQS,
        metavar="SEQ",
        help=f"MOT17 calibration sequences, held out from eval (default: {_DEFAULT_CALIB_SEQS})",
    )
    parser.add_argument(
        "--calib-per-seq",
        type=int,
        default=_DEFAULT_CALIB_PER_SEQ,
        help=f"Frames sampled per calibration sequence (default: {_DEFAULT_CALIB_PER_SEQ})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads for ncnn2table calibration (default: 4)",
    )
    args = parser.parse_args()

    _check_filesystem()

    models      = [args.model] if args.model else args.models
    resolutions = [args.imgsz] if args.imgsz else args.resolutions

    if args.int8:
        for tool in (_NCNN2TABLE, _NCNN2INT8):
            if not tool.exists():
                print(f"ERROR: INT8 tool missing: {tool}\n"
                      f"  Extract ncnn2table/ncnn2int8 from the official ncnn "
                      f"release into {_TOOLS_DIR}/", file=sys.stderr)
                sys.exit(1)

    results: list[tuple[str, int, str]] = []   # (model, res, status)

    for model_name in models:
        for res in resolutions:
            try:
                out = export_ncnn(model_name, res)
                results.append((model_name, res, "ok" if out else "skipped"))
            except AssertionError as exc:
                print(f"  FAIL assertion: {exc}", file=sys.stderr)
                results.append((model_name, res, f"failed: {exc}"))
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
                results.append((model_name, res, f"failed: {type(exc).__name__}: {exc}"))

    print("\n--- Export summary ---")
    for model_name, res, status in results:
        print(f"  {model_name} @ {res}px  →  {status}")

    failed = [(m, r) for m, r, s in results if s.startswith("failed")]
    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        sys.exit(1)

    # FP16 export: independent pass producing half-precision folders alongside FP32
    if args.fp16:
        fp16_results: list[tuple[str, int, str]] = []
        for model_name in models:
            for res in resolutions:
                try:
                    out = export_ncnn(model_name, res, half=True)
                    fp16_results.append((model_name, res, "ok" if out else "skipped"))
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
                    fp16_results.append((model_name, res, f"failed: {type(exc).__name__}: {exc}"))

        print("\n--- FP16 export summary ---")
        for model_name, res, status in fp16_results:
            print(f"  {model_name} @ {res}px  →  {status}")
        if [r for r in fp16_results if r[2].startswith("failed")]:
            sys.exit(1)

    # INT8 post-quantization: only over the FP32 folders just produced/verified
    if args.int8:
        calib_list = _MODELS_DIR / "ncnn_int8_calib_list.txt"
        n_calib = build_calib_list(args.calib_seqs, args.calib_per_seq, calib_list)
        if n_calib == 0:
            print("ERROR: no calibration frames found — check --calib-seqs", file=sys.stderr)
            sys.exit(1)

        int8_results: list[tuple[str, int, str]] = []
        for model_name, res, status in results:
            if status != "ok":
                continue
            try:
                out = quantize_int8(model_name, res, calib_list, args.threads)
                int8_results.append((model_name, res, "ok" if out else "skipped"))
            except subprocess.CalledProcessError as exc:
                print(f"  FAIL ncnn tool rc={exc.returncode}", file=sys.stderr)
                int8_results.append((model_name, res, f"failed: rc={exc.returncode}"))
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
                int8_results.append((model_name, res, f"failed: {type(exc).__name__}: {exc}"))

        print("\n--- INT8 quantization summary ---")
        for model_name, res, status in int8_results:
            print(f"  {model_name} @ {res}px  →  {status}")
        if [r for r in int8_results if r[2].startswith("failed")]:
            sys.exit(1)

    print(
        f"\nTransfer models/ to each ARM device.\n"
        f"Only the *_ncnn_model/ folders are needed — each is valid for its baked resolution."
    )


if __name__ == "__main__":
    main()

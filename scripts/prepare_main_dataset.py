"""
Prepare main dataset from raw downloads.

Takes raw data (Kaggle/Drive/local) and builds a single canonical layout
so all experiments load from one place: main_dataset/Alzheimer, main_dataset/Retinal, main_dataset/TB.

Usage:
    python scripts/prepare_main_dataset.py --raw-dir /content/datasets --main-dir /content/main_dataset
    python scripts/prepare_main_dataset.py --raw-dir ./datasets --main-dir ./main_dataset
"""

import argparse
import os
import shutil


# Canonical names experiments expect (config.DATASET_PATHS)
MAIN_ALZHEIMER = "Alzheimer"
MAIN_RETINAL = "Retinal"
MAIN_TB = "TB"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _dir_has_image_subdirs(path, min_classes=2):
    """True if path contains subdirs that look like class folders (have images)."""
    if not path or not os.path.isdir(path):
        return False
    try:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) < min_classes:
            return False
        # Check at least one subdir has images
        for d in subdirs[:3]:
            sub = os.path.join(path, d)
            for f in os.listdir(sub)[:5]:
                if os.path.splitext(f)[1].lower() in VALID_EXT:
                    return True
    except Exception:
        pass
    return False


def find_alzheimer_root(raw_root):
    """Return path that has class subdirs (MildDemented, etc.)."""
    candidates = [
        os.path.join(raw_root, "AugmentedAlzheimerDataset", "OriginalDataset"),
        os.path.join(raw_root, "AugmentedAlzheimerDataset"),
        os.path.join(raw_root, "augmented-alzheimer-mri-dataset", "OriginalDataset"),
        os.path.join(raw_root, "augmented-alzheimer-mri-dataset"),
    ]
    for p in candidates:
        if _dir_has_image_subdirs(p, min_classes=4):
            return p
    return None


def find_retinal_root(raw_root):
    """Return path that has Retinal class subdirs."""
    candidates = [
        os.path.join(raw_root, "Ratinal_Deasis"),
        os.path.join(raw_root, "ratinal-deasis"),
    ]
    for p in candidates:
        if _dir_has_image_subdirs(p, min_classes=2):
            return p
    return None


def find_tb_root(raw_root):
    """Return path that has Normal, Tuberculosis (or similar)."""
    candidates = [
        os.path.join(raw_root, "TB_Chest_Radiography_Database"),
        os.path.join(raw_root, "tuberculosis-tb-chest-xray-dataset"),
    ]
    for p in candidates:
        if _dir_has_image_subdirs(p, min_classes=2):
            return p
    return None


def copy_tree(src, dst):
    """Copy directory tree; skip non-dirs and non-image files at top level."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        sp = os.path.join(src, name)
        dp = os.path.join(dst, name)
        if os.path.isdir(sp):
            if os.path.exists(dp):
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
        # else skip files (e.g. README, xlsx) at dataset root


def main():
    p = argparse.ArgumentParser(description="Build main dataset from raw downloads.")
    p.add_argument("--raw-dir", required=True, help="Root where raw datasets are (e.g. /content/datasets)")
    p.add_argument("--main-dir", required=True, help="Output root for main dataset (e.g. /content/main_dataset)")
    p.add_argument("--symlink", action="store_true", help="Use symlinks instead of copy (faster, no extra disk)")
    args = p.parse_args()

    raw_root = os.path.abspath(args.raw_dir)
    main_root = os.path.abspath(args.main_dir)

    if not os.path.isdir(raw_root):
        raise SystemExit(f"Raw dir not found: {raw_root}")

    os.makedirs(main_root, exist_ok=True)
    done = []

    # Alzheimer
    alz_src = find_alzheimer_root(raw_root)
    alz_dst = os.path.join(main_root, MAIN_ALZHEIMER)
    if alz_src:
        if args.symlink:
            if os.path.exists(alz_dst):
                os.unlink(alz_dst) if os.path.islink(alz_dst) else shutil.rmtree(alz_dst)
            os.symlink(alz_src, alz_dst)
        else:
            copy_tree(alz_src, alz_dst)
        done.append(f"Alzheimer <- {alz_src}")
    else:
        print(f"  Skip Alzheimer: no suitable folder under {raw_root}")

    # Retinal
    ret_src = find_retinal_root(raw_root)
    ret_dst = os.path.join(main_root, MAIN_RETINAL)
    if ret_src:
        if args.symlink:
            if os.path.exists(ret_dst):
                os.unlink(ret_dst) if os.path.islink(ret_dst) else shutil.rmtree(ret_dst)
            os.symlink(ret_src, ret_dst)
        else:
            copy_tree(ret_src, ret_dst)
        done.append(f"Retinal <- {ret_src}")
    else:
        print(f"  Skip Retinal: no suitable folder under {raw_root}")

    # TB
    tb_src = find_tb_root(raw_root)
    tb_dst = os.path.join(main_root, MAIN_TB)
    if tb_src:
        if args.symlink:
            if os.path.exists(tb_dst):
                os.unlink(tb_dst) if os.path.islink(tb_dst) else shutil.rmtree(tb_dst)
            os.symlink(tb_src, tb_dst)
        else:
            copy_tree(tb_src, tb_dst)
        done.append(f"TB <- {tb_src}")
    else:
        print(f"  Skip TB: no suitable folder under {raw_root}")

    if not done:
        raise SystemExit("No datasets prepared. Check --raw-dir has AugmentedAlzheimerDataset, Ratinal_Deasis, TB_Chest_Radiography_Database (or Kaggle unpack names).")

    print("Main dataset ready at:", main_root)
    for line in done:
        print(" ", line)
    print("Set COLAB_DATASET_ROOT (or MAIN_DATASET_ROOT) to:", main_root)


if __name__ == "__main__":
    main()

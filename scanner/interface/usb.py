"""scanner.interface.usb — USB drive detection and file copy."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def list_usb_drives() -> list[dict]:
    """Detect mounted USB drives using lsblk.

    Returns a list of dicts with keys: device, mountpoint, size, label.
    Falls back to scanning /media/ and /mnt/ if lsblk is unavailable.
    """
    try:
        result = subprocess.run(
            ["lsblk", "-J", "-o", "NAME,MOUNTPOINT,SIZE,LABEL,TRAN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        data = json.loads(result.stdout)
        drives: list[dict] = []
        for device in data.get("blockdevices", []):
            _collect_usb_mounts(device, drives, parent_is_usb=False)
        return drives
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.debug("lsblk unavailable (%s), falling back to directory scan", exc)
        return _fallback_scan()


def _collect_usb_mounts(device: dict, drives: list, *, parent_is_usb: bool) -> None:
    """Recursively find mounted USB partitions in lsblk output."""
    is_usb = device.get("tran") == "usb" or parent_is_usb
    mp = device.get("mountpoint")
    if is_usb and mp:
        drives.append(
            {
                "device": device.get("name", ""),
                "mountpoint": mp,
                "size": device.get("size", ""),
                "label": device.get("label") or device.get("name", "USB"),
            }
        )
    for child in device.get("children", []):
        _collect_usb_mounts(child, drives, parent_is_usb=is_usb)


def _fallback_scan() -> list[dict]:
    """Scan /media/ and /mnt/ for mounted directories as a fallback."""
    drives: list[dict] = []
    for base in ("/media", "/mnt"):
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            path = os.path.join(base, entry)
            if os.path.ismount(path):
                drives.append(
                    {
                        "device": entry,
                        "mountpoint": path,
                        "size": "",
                        "label": entry,
                    }
                )
        # Also check one level deeper (/media/<user>/<drive>)
        if base == "/media":
            for user_dir in os.listdir(base):
                user_path = os.path.join(base, user_dir)
                if not os.path.isdir(user_path):
                    continue
                for entry in os.listdir(user_path):
                    path = os.path.join(user_path, entry)
                    if os.path.ismount(path):
                        drives.append(
                            {
                                "device": entry,
                                "mountpoint": path,
                                "size": "",
                                "label": entry,
                            }
                        )
    return drives


def copy_to_usb(source_path: str, mountpoint: str) -> str:
    """Copy a file to a USB drive mountpoint.

    Args:
        source_path: Absolute path to the file to copy.
        mountpoint: Absolute path to the USB mount point.

    Returns:
        The destination file path.

    Raises:
        ValueError: If the mountpoint is invalid or the source file is missing.
        OSError: If the copy operation fails.
    """
    if not os.path.isfile(source_path):
        raise ValueError(f"Source file does not exist: {source_path}")
    if not os.path.ismount(mountpoint):
        raise ValueError(f"Not a valid mount point: {mountpoint}")

    dest = os.path.join(mountpoint, os.path.basename(source_path))
    shutil.copy2(source_path, dest)
    logger.info("Copied %s to %s", source_path, dest)
    return dest

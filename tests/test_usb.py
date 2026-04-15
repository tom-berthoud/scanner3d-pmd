"""Tests for scanner.interface.usb — USB drive detection and copy."""

import json
import os
import tempfile
from unittest import mock

import pytest

from scanner.interface.usb import copy_to_usb, list_usb_drives


class TestListUsbDrives:
    """Tests for list_usb_drives()."""

    def test_parses_lsblk_output(self):
        """Should extract USB drives with mountpoints from lsblk JSON."""
        lsblk_output = json.dumps(
            {
                "blockdevices": [
                    {
                        "name": "sda",
                        "mountpoint": None,
                        "size": "16G",
                        "label": None,
                        "tran": "usb",
                        "children": [
                            {
                                "name": "sda1",
                                "mountpoint": "/media/pi/USBSTICK",
                                "size": "16G",
                                "label": "USBSTICK",
                                "tran": None,
                            }
                        ],
                    },
                    {
                        "name": "mmcblk0",
                        "mountpoint": None,
                        "size": "32G",
                        "label": None,
                        "tran": None,
                        "children": [
                            {
                                "name": "mmcblk0p1",
                                "mountpoint": "/boot",
                                "size": "256M",
                                "label": "boot",
                                "tran": None,
                            }
                        ],
                    },
                ]
            }
        )

        with mock.patch("scanner.interface.usb.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout=lsblk_output)
            drives = list_usb_drives()

        assert len(drives) == 1
        assert drives[0]["mountpoint"] == "/media/pi/USBSTICK"
        assert drives[0]["label"] == "USBSTICK"
        assert drives[0]["size"] == "16G"

    def test_empty_when_no_usb(self):
        """Should return empty list when no USB devices found."""
        lsblk_output = json.dumps(
            {
                "blockdevices": [
                    {
                        "name": "mmcblk0",
                        "mountpoint": "/",
                        "size": "32G",
                        "label": "rootfs",
                        "tran": None,
                    }
                ]
            }
        )

        with mock.patch("scanner.interface.usb.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout=lsblk_output)
            drives = list_usb_drives()

        assert drives == []

    def test_fallback_when_lsblk_missing(self):
        """Should fall back to directory scan when lsblk is not available."""
        with mock.patch(
            "scanner.interface.usb.subprocess.run", side_effect=FileNotFoundError
        ):
            with mock.patch("scanner.interface.usb._fallback_scan", return_value=[]) as fb:
                drives = list_usb_drives()
                fb.assert_called_once()
                assert drives == []


class TestCopyToUsb:
    """Tests for copy_to_usb()."""

    def test_raises_on_missing_source(self, tmp_path):
        """Should raise ValueError if source file doesn't exist."""
        with pytest.raises(ValueError, match="Source file does not exist"):
            copy_to_usb("/nonexistent/file.stl", str(tmp_path))

    def test_raises_on_invalid_mountpoint(self, tmp_path):
        """Should raise ValueError if mountpoint is not a mount point."""
        source = tmp_path / "scan.stl"
        source.write_text("dummy")
        fake_mount = tmp_path / "not_a_mount"
        fake_mount.mkdir()

        with pytest.raises(ValueError, match="Not a valid mount point"):
            copy_to_usb(str(source), str(fake_mount))

    def test_copies_file_successfully(self, tmp_path):
        """Should copy the file when source and mountpoint are valid."""
        source = tmp_path / "scan.stl"
        source.write_text("mesh data")
        usb_dir = tmp_path / "usb"
        usb_dir.mkdir()

        with mock.patch("os.path.ismount", return_value=True):
            dest = copy_to_usb(str(source), str(usb_dir))

        assert os.path.basename(dest) == "scan.stl"
        assert (usb_dir / "scan.stl").read_text() == "mesh data"

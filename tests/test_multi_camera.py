"""Tests for multi-camera configuration helpers."""

import numpy as np

from scanner.calibration import camera_configs, camera_ids, default_camera_id


def test_legacy_camera_config_becomes_main_camera() -> None:
    config = {"camera": {"resolution": [640, 480], "mock_shape": "cube"}}
    cameras = camera_configs(config)
    assert camera_ids(config) == ["main"]
    assert default_camera_id(config) == "main"
    assert cameras[0]["resolution"] == [640, 480]


def test_camera_list_preserves_ids_and_order() -> None:
    config = {
        "cameras": [
            {"id": "right", "type": "pi"},
            {"id": "left", "type": "usb", "device_index": 0},
        ]
    }
    assert camera_ids(config) == ["right", "left"]


def test_identity_extrinsics_do_not_change_triangulated_point() -> None:
    from scanner.processing import triangulate

    line = np.array([[320.0, 240.0]], dtype=np.float32)
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    plane = np.array([0.0, 0.0, 1.0, -300.0])
    result = triangulate(
        line,
        camera_matrix,
        dist_coeffs,
        plane,
        0.0,
        camera_to_platform_rotation=np.eye(3),
        camera_to_platform_translation=np.zeros(3),
    )
    np.testing.assert_allclose(result, np.array([[0.0, 0.0, 300.0]]), atol=1e-6)


def test_measured_pose_extrinsics_point_camera_at_target() -> None:
    from scanner.calibration.multi_camera import _load_extrinsics

    position = np.array([173.5, 140.0, -300.5])
    target = np.array([0.0, 0.0, 0.0])
    rotation, translation = _load_extrinsics(
        {"extrinsics": {"position_mm": position.tolist(), "target_mm": target.tolist()}}
    )

    forward = rotation[:, 2]
    expected_forward = (target - position) / np.linalg.norm(target - position)
    np.testing.assert_allclose(translation, position, atol=1e-6)
    np.testing.assert_allclose(forward, expected_forward, atol=1e-6)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)

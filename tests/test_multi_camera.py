"""Tests for multi-camera configuration helpers."""

import numpy as np
import pytest
import yaml

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


def test_camera_list_preserves_laser_sampling() -> None:
    config = {
        "cameras": [
            {"id": "right", "laser_sampling": {"x_stride": 1, "y_stride": 2}},
            {"id": "left", "laser_sampling": {"x_stride": 3, "y_stride": 4}},
        ]
    }

    cameras = camera_configs(config)

    assert cameras[0]["laser_sampling"] == {"x_stride": 1, "y_stride": 2}
    assert cameras[1]["laser_sampling"] == {"x_stride": 3, "y_stride": 4}


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


def test_rotation_matrix_extrinsics_override_look_at_when_file_loaded(tmp_path) -> None:
    from scanner.calibration.multi_camera import _load_extrinsics

    path = tmp_path / "camera_extrinsics_left.yaml"
    file_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    file_translation = np.array([1.0, 2.0, 3.0], dtype=float)
    path.write_text(
        yaml.dump(
            {
                "rotation_matrix": file_rotation.tolist(),
                "translation_mm": file_translation.tolist(),
            }
        ),
        encoding="utf-8",
    )

    rotation, translation = _load_extrinsics(
        {
            "extrinsics_path": str(path),
            "extrinsics": {
                "position_mm": [100.0, 100.0, 100.0],
                "target_mm": [0.0, 0.0, 0.0],
            },
        }
    )

    np.testing.assert_allclose(rotation, file_rotation)
    np.testing.assert_allclose(translation, file_translation)


def test_pnp_pose_conversion_round_trips_camera_to_platform_pose() -> None:
    pytest.importorskip("cv2")

    from scanner.calibration.extrinsics import (
        _camera_pose_from_pnp,
        _pnp_guess_from_camera_pose,
    )
    from scanner.calibration.multi_camera import _look_at_extrinsics

    rotation, translation = _look_at_extrinsics(
        {
            "position_mm": [173.5, 140.0, -300.5],
            "target_mm": [0.0, 13.7, 0.0],
            "up_mm": [0.0, 1.0, 0.0],
        }
    )

    rvec, tvec = _pnp_guess_from_camera_pose(rotation, translation)
    recovered_rotation, recovered_translation = _camera_pose_from_pnp(rvec, tvec)

    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-9)
    np.testing.assert_allclose(recovered_translation, translation, atol=1e-9)


def test_extrinsics_pnp_uses_mechanical_prior_for_planar_board() -> None:
    cv2 = pytest.importorskip("cv2")

    from scanner.calibration.extrinsics import (
        _camera_pose_from_pnp,
        _platform_board_points,
        _pnp_guess_from_camera_pose,
        _solve_extrinsics_pnp,
    )
    from scanner.calibration.multi_camera import _look_at_extrinsics

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5)
    object_points = _platform_board_points(
        (7, 5),
        15.0,
        np.array([-43.5, 105.3, -32.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    rotation, translation = _look_at_extrinsics(
        {
            "position_mm": [173.5, 140.0, -300.5],
            "target_mm": [0.0, 13.7, 0.0],
            "up_mm": [0.0, 1.0, 0.0],
        }
    )
    rvec, tvec = _pnp_guess_from_camera_pose(rotation, translation)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2).astype(np.float32)

    solved_rvec, solved_tvec, _selected_points, candidates, selection = _solve_extrinsics_pnp(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        initial_camera_to_platform_rotation=rotation,
        initial_camera_to_platform_translation=translation,
    )
    solved_rotation, solved_translation = _camera_pose_from_pnp(solved_rvec, solved_tvec)

    assert any(candidate["selected"] for candidate in candidates)
    assert selection["translation_source"] == "pnp"
    np.testing.assert_allclose(solved_rotation, rotation, atol=1e-5)
    np.testing.assert_allclose(solved_translation, translation, atol=1e-4)


def test_extrinsics_pnp_handles_opposite_checkerboard_corner_order() -> None:
    cv2 = pytest.importorskip("cv2")

    from scanner.calibration.extrinsics import (
        _camera_pose_from_pnp,
        _checkerboard_object_point_variants,
        _platform_board_points,
        _pnp_guess_from_camera_pose,
        _solve_extrinsics_pnp,
    )
    from scanner.calibration.multi_camera import _look_at_extrinsics

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5)
    board_size = (7, 5)
    object_points = _platform_board_points(
        board_size,
        15.0,
        np.array([-43.5, 105.3, -32.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    rotation, translation = _look_at_extrinsics(
        {
            "position_mm": [173.5, 140.0, -300.5],
            "target_mm": [0.0, 13.7, 0.0],
            "up_mm": [0.0, 1.0, 0.0],
        }
    )
    rvec, tvec = _pnp_guess_from_camera_pose(rotation, translation)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2).astype(np.float32)[::-1]

    solved_rvec, solved_tvec, _selected_points, candidates, selection = _solve_extrinsics_pnp(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        initial_camera_to_platform_rotation=rotation,
        initial_camera_to_platform_translation=translation,
        object_point_variants=_checkerboard_object_point_variants(object_points, board_size),
    )
    solved_rotation, solved_translation = _camera_pose_from_pnp(solved_rvec, solved_tvec)
    selected = next(candidate for candidate in candidates if candidate["selected"])

    assert selected["corner_order"] == "origin_opposite_corner"
    assert selection["translation_source"] == "pnp"
    np.testing.assert_allclose(solved_rotation, rotation, atol=1e-5)
    np.testing.assert_allclose(solved_translation, translation, atol=1e-4)


def test_extrinsics_pnp_rejects_pose_far_from_mechanical_prior() -> None:
    cv2 = pytest.importorskip("cv2")

    from scanner.calibration import CalibrationError
    from scanner.calibration.extrinsics import (
        _platform_board_points,
        _pnp_guess_from_camera_pose,
        _solve_extrinsics_pnp,
    )
    from scanner.calibration.multi_camera import _look_at_extrinsics

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5)
    object_points = _platform_board_points(
        (7, 5),
        15.0,
        np.array([-43.5, 105.3, -32.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    true_rotation, true_translation = _look_at_extrinsics(
        {
            "position_mm": [173.5, 140.0, -300.5],
            "target_mm": [0.0, 13.7, 0.0],
            "up_mm": [0.0, 1.0, 0.0],
        }
    )
    wrong_rotation, wrong_translation = _look_at_extrinsics(
        {
            "position_mm": [-155.0, -20.0, -260.0],
            "target_mm": [0.0, 92.8, 8.5],
            "up_mm": [0.0, 1.0, 0.0],
        }
    )
    rvec, tvec = _pnp_guess_from_camera_pose(true_rotation, true_translation)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2).astype(np.float32)

    with pytest.raises(CalibrationError, match="inconsistent with the mechanical"):
        _solve_extrinsics_pnp(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            initial_camera_to_platform_rotation=wrong_rotation,
            initial_camera_to_platform_translation=wrong_translation,
            max_prior_distance_mm=120.0,
        )


def test_extrinsics_pnp_allows_mechanical_translation_fallback() -> None:
    cv2 = pytest.importorskip("cv2")

    from scanner.calibration.extrinsics import (
        _platform_board_points,
        _pnp_guess_from_camera_pose,
        _solve_extrinsics_pnp,
    )
    from scanner.calibration.multi_camera import _look_at_extrinsics

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5)
    object_points = _platform_board_points(
        (7, 5),
        15.0,
        np.array([-43.5, 105.3, -32.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    true_rotation, true_translation = _look_at_extrinsics(
        {
            "position_mm": [173.5, 140.0, -300.5],
            "target_mm": [0.0, 13.7, 0.0],
            "up_mm": [0.0, 1.0, 0.0],
        }
    )
    offset_prior_translation = true_translation + np.array([130.0, 0.0, 0.0])
    rvec, tvec = _pnp_guess_from_camera_pose(true_rotation, true_translation)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2).astype(np.float32)

    _rvec, _tvec, _points, _candidates, selection = _solve_extrinsics_pnp(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        initial_camera_to_platform_rotation=true_rotation,
        initial_camera_to_platform_translation=offset_prior_translation,
        max_prior_distance_mm=120.0,
        max_translation_fallback_distance_mm=250.0,
        max_fixed_translation_reprojection_px=1000.0,
    )

    assert selection["translation_source"] == "mechanical_prior"

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODALITY_PATH = ROOT / "examples" / "KMU26AUV" / "kmu26_auv_real__modality.json"


def test_kmu26_modality_ranges_are_contiguous():
    modality = json.loads(MODALITY_PATH.read_text())

    expected_state_dims = {
        "prev_command": 4,
        "dvl_velocity": 3,
        "angular_velocity": 3,
        "linear_acceleration": 3,
        "attitude": 4,
        "depth": 1,
        "altitude": 1,
        "validity": 4,
    }

    next_start = 0
    for key, expected_dim in expected_state_dims.items():
        item = modality["state"][key]
        assert item["start"] == next_start
        assert item["end"] - item["start"] == expected_dim
        next_start = item["end"]

    assert next_start == 23
    assert modality["action"]["motion"] == {"start": 0, "end": 4}


def test_kmu26_modality_uses_two_distinct_camera_streams():
    modality = json.loads(MODALITY_PATH.read_text())

    assert modality["video"]["ego"]["original_key"] == "observation.images.ego"
    assert (
        modality["video"]["buoy_release"]["original_key"]
        == "observation.images.buoy_release"
    )

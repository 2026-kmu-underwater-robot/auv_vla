import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "inference_service_u0.py"


def test_kmu26_example_observation_contains_configured_keys():
    tree = ast.parse(SCRIPT_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_example_observation"
    )
    kmu_return = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "video.buoy_release"
            for key in node.value.keys
        )
    )
    keys = {key.value for key in kmu_return.value.keys if isinstance(key, ast.Constant)}

    assert keys == {
        "video.ego",
        "video.buoy_release",
        "state.prev_command",
        "state.dvl_velocity",
        "state.angular_velocity",
        "state.linear_acceleration",
        "state.attitude",
        "state.depth",
        "state.altitude",
        "state.validity",
        "annotation.human.action.task_description",
    }

"""Real-vehicle data and checkpoint regressions; no hardware or weights needed."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.kmu26_dataset import (
    Kmu26RealTrainingDataset,
    assert_disjoint_sessions,
    validate_demonstrations,
)
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
)
from gr00t.deployment.kmu26_training import (
    DATA_CONFIG,
    read_real_training_settings,
    validate_real_checkpoint,
)
from gr00t.experiment.data_config import load_data_config
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from gr00t.model.gr00t_n1 import GR00T_N1_5_Config


@pytest.fixture
def recording(tmp_path):
    meta = tmp_path / "meta"
    audit = meta / "acquisition/episode_000000"
    audit.mkdir(parents=True)
    data = tmp_path / "data/chunk-000"
    data.mkdir(parents=True)
    provenance = dict(
        collection_kind="task_demonstration",
        data_source="real",
        use_sim_time=False,
        session_id="pool-session-1",
        context={"fixture": True},
        expected_mode="STABILIZE",
        neutral_pwm=1500,
        pwm_span=300,
        action_channels=[5, 6, 3, 4],
        max_sensor_age_sec=0.25,
    )
    manifest = dict(
        frames=20, fps=10, provenance=provenance, termination_reason="operator_stop", success=True
    )
    (meta / "source_manifests.jsonl").write_text(json.dumps(manifest) + "\n")
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 20}) + "\n")
    rows = [
        dict(
            connected=True,
            armed=True,
            mode="STABILIZE",
            rc_publishers=1,
            state_receipt_age_wall_s=0.1,
        )
        for _ in range(20)
    ]
    (audit / "vehicle_state.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    pd.DataFrame(
        {
            "observation.state": [np.ones(23)] * 20,
            "action": [np.zeros(4)] * 20,
            "telemetry.source_timestamp": [np.full(7, 10 + i / 10) for i in range(20)],
            "telemetry.ros_timestamp": np.arange(20) / 10 + 10,
        }
    ).to_parquet(data / "episode_000000.parquet")
    return tmp_path


def test_v2_preserves_depth_acceleration_and_validity():
    config = load_data_config(DATA_CONFIG)
    values = {key: np.array([[0.0, 1.0, 9.81]], dtype=np.float32) for key in config.state_keys}
    result = {key: value.copy() for key, value in values.items()}
    for transform in config.transform().transforms:
        if isinstance(transform, (StateActionToTensor, StateActionSinCosTransform)) and set(
            transform.apply_to
        ) == set(config.state_keys):
            result = transform(result)
    for key in config.state_keys:
        np.testing.assert_equal(result[key].numpy(), values[key])
    assert any(
        isinstance(t, StateActionSinCosTransform)
        for t in load_data_config("kmu26_auv_real").transform().transforms
    )


def test_complete_chunks_exclude_terminal_padding():
    dataset = Kmu26RealTrainingDataset.__new__(Kmu26RealTrainingDataset)
    dataset.modality_configs = load_data_config(DATA_CONFIG).modality_config()
    dataset._trajectory_ids = np.array([0, 1, 2])
    dataset._trajectory_lengths = np.array([15, 16, 20])
    assert dataset._get_all_steps() == [(1, 0)] + [(2, i) for i in range(5)]
    # The legacy loader retains its behavior for existing datasets/checkpoints.
    assert len(LeRobotSingleDataset._get_all_steps(dataset)) == 51


def test_valid_real_data_and_recorded_pwm_span(recording):
    validate_demonstrations(recording)
    settings = read_real_training_settings(recording)
    assert settings["pwm_span"] == 300
    assert settings["data_config"] == DATA_CONFIG


@pytest.mark.parametrize(
    "field,value",
    [
        ("collection_kind", "connection_check"),
        ("data_source", "simulation"),
        ("use_sim_time", True),
        ("session_id", ""),
        ("context", {}),
    ],
)
def test_unreviewed_or_simulation_data_is_rejected(recording, field, value):
    path = recording / "meta/source_manifests.jsonl"
    manifest = json.loads(path.read_text())
    manifest["provenance"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_demonstrations(recording)


def test_mixed_command_ranges_are_rejected(recording):
    path = recording / "meta/source_manifests.jsonl"
    manifest = json.loads(path.read_text())
    manifest["provenance"]["pwm_span"] = 400
    path.write_text(path.read_text() + json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="Mixed"):
        read_real_training_settings(recording)


@pytest.mark.parametrize("age", [None, -1, float("nan"), 2.1])
def test_unverified_vehicle_state_is_rejected(recording, age):
    path = recording / "meta/acquisition/episode_000000/vehicle_state.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["state_receipt_age_wall_s"] = age
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Vehicle/control"):
        validate_demonstrations(recording)


def test_duplicate_camera_is_rejected(recording):
    path = recording / "data/chunk-000/episode_000000.parquet"
    table = pd.read_parquet(path)
    table.at[1, "telemetry.source_timestamp"] = table.iloc[0]["telemetry.source_timestamp"].copy()
    table.to_parquet(path)
    with pytest.raises(ValueError, match="duplicate"):
        validate_demonstrations(recording)


def test_session_leakage_is_rejected(recording):
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_sessions(recording, recording)


def test_checkpoint_roundtrip_and_deployment_mismatch(recording, tmp_path):
    settings = read_real_training_settings(recording)
    checkpoint = tmp_path / "model"
    config = GR00T_N1_5_Config(action_head_cfg={"target_loss_weight": 0.0}, kmu26_training=settings)
    config.save_pretrained(checkpoint)
    assert validate_real_checkpoint(checkpoint, DATA_CONFIG, pwm_span=300) == settings
    with pytest.raises(ValueError, match="pwm_span"):
        validate_real_checkpoint(checkpoint, DATA_CONFIG, pwm_span=400)
    with pytest.raises(ValueError, match="preprocessing"):
        validate_real_checkpoint(checkpoint, "kmu26_auv_real")
    config.action_head_cfg["target_loss_weight"] = 1.0
    config.save_pretrained(checkpoint)
    with pytest.raises(ValueError, match="CAP-off"):
        validate_real_checkpoint(checkpoint, DATA_CONFIG)


def test_legacy_checkpoint_cannot_silently_use_v2(tmp_path):
    assert validate_real_checkpoint(tmp_path, "kmu26_auv_real") is None
    with pytest.raises(ValueError, match="Missing v2"):
        validate_real_checkpoint(tmp_path, DATA_CONFIG)


def test_bf16_construction_can_sample_flow_matching_time():
    config = FlowmatchingActionHeadConfig(
        diffusion_model_cfg=dict(
            num_attention_heads=1, attention_head_dim=8, output_dim=8, num_layers=0
        ),
        input_embedding_dim=8,
        backbone_embedding_dim=8,
        hidden_size=8,
        max_state_dim=64,
        action_dim=4,
        action_horizon=16,
        max_num_embodiments=1,
        max_seq_len=32,
        use_vlln=False,
        num_target_vision_tokens=2,
    )
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        head = FlowmatchingActionHead(config)
        times = head.sample_time(16, "cpu", torch.bfloat16)
    finally:
        torch.set_default_dtype(original_dtype)
    assert times.dtype == torch.bfloat16
    assert torch.isfinite(times).all()
    assert head.beta_dist.concentration0.dtype == torch.float32


def test_training_entrypoint_persists_cap_off(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/gr00t_finetune.py"
    spec = importlib.util.spec_from_file_location("kmu26_test_finetune", path)
    trainer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, trainer)
    spec.loader.exec_module(trainer)
    model = SimpleNamespace(
        config=GR00T_N1_5_Config(
            backbone_cfg={},
            action_head_cfg={"target_loss_weight": 1.0},
            action_horizon=16,
            action_dim=4,
        ),
        action_head=SimpleNamespace(
            config=SimpleNamespace(action_horizon=16), target_model=torch.nn.Linear(1, 1)
        ),
    )
    monkeypatch.setattr(trainer.GR00T_N1_5, "from_pretrained", lambda **kwargs: model)
    monkeypatch.setattr(trainer, "LeRobotSingleDataset", lambda **kwargs: object())
    monkeypatch.setattr(trainer, "TrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs))

    class SaveRunner:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def train(self):
            self.model.config.save_pretrained(tmp_path / "checkpoint")

    monkeypatch.setattr(trainer, "TrainRunner", SaveRunner)
    trainer.main(
        trainer.ArgsConfig(
            dataset_path=[str(tmp_path)],
            data_config="kmu26_auv_real",
            target_loss_weight=0,
            lora_rank=0,
        )
    )
    saved = GR00T_N1_5_Config.from_pretrained(tmp_path / "checkpoint")
    assert saved.action_head_cfg["target_loss_weight"] == 0.0
    assert model.action_head.config.target_loss_weight == 0.0


def test_unreviewed_extra_episode_is_rejected(recording):
    path = recording / "meta/episodes.jsonl"
    path.write_text(path.read_text() + json.dumps({"episode_index": 1, "length": 20}) + "\n")
    with pytest.raises(ValueError, match="Episode index"):
        validate_demonstrations(recording)


@pytest.mark.parametrize("problem", ["sample_gap", "stale_imu", "invalid_action"])
def test_edited_export_cannot_bypass_recording_checks(recording, problem):
    path = recording / "data/chunk-000/episode_000000.parquet"
    table = pd.read_parquet(path)
    if problem == "sample_gap":
        table.at[1, "telemetry.ros_timestamp"] += 0.3
    elif problem == "stale_imu":
        stamps = table.iloc[1]["telemetry.source_timestamp"].copy()
        stamps[2] -= 1.0
        table.at[1, "telemetry.source_timestamp"] = stamps
    else:
        table.at[1, "action"] = np.full(4, np.nan)
    table.to_parquet(path)
    with pytest.raises(ValueError):
        validate_demonstrations(recording)

"""Check recorded command units and preserve them with real-vehicle checkpoints."""

import argparse
import json
import math
from pathlib import Path

DATA_CONFIG = "kmu26_auv_real_v2"


def _validate_settings(settings: dict) -> None:
    if settings.get("action_channels") != [5, 6, 3, 4]:
        raise ValueError("Expected action channels [5, 6, 3, 4]")
    neutral = settings.get("neutral_pwm")
    span = settings.get("pwm_span")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in (neutral, span)
    ):
        raise ValueError("PWM neutral and span must be finite numbers")
    if span <= 0 or neutral - span < 1000 or neutral + span > 2000:
        raise ValueError("Recorded PWM range must fit within 1000..2000")
    if not isinstance(settings.get("expected_mode"), str) or not settings["expected_mode"].strip():
        raise ValueError("Missing recorded control mode")
    if settings.get("fps") != 10:
        raise ValueError("KMU26 requires 10 Hz observations/actions")


def read_real_training_settings(dataset: Path) -> dict:
    """Read consistent command settings from reviewed real-vehicle recordings."""
    lines = (dataset / "meta/source_manifests.jsonl").read_text().splitlines()
    if not lines:
        raise ValueError("Empty source manifests")
    settings = None
    for line in lines:
        manifest = json.loads(line)
        provenance = manifest.get("provenance", {})
        if provenance.get("data_source") != "real" or provenance.get("use_sim_time") is not False:
            raise ValueError("This training profile requires real data with use_sim_time=false")
        current = {
            key: provenance.get(key)
            for key in ("action_channels", "neutral_pwm", "pwm_span", "expected_mode")
        }
        current["fps"] = manifest.get("fps")
        _validate_settings(current)
        if settings is not None and current != settings:
            raise ValueError("Mixed command ranges or control modes; export separately")
        settings = current
    return {"schema": 1, "data_config": DATA_CONFIG, "target_loss_weight": 0.0, **settings}


def validate_real_checkpoint(
    checkpoint: str | Path,
    data_config: str,
    *,
    neutral_pwm: int | None = None,
    pwm_span: int | None = None,
    expected_mode: str | None = None,
) -> dict | None:
    """Reject mismatched preprocessing, CAP settings or explicitly supplied RC settings.

    Existing checkpoints without v2 metadata retain their legacy preprocessing.
    This checks recorded settings; it does not read back the flight controller.
    """
    path = Path(checkpoint).expanduser() / "config.json"
    config = json.loads(path.read_text()) if path.is_file() else {}
    settings = config.get("kmu26_training")
    if settings is None and data_config != DATA_CONFIG:
        return None
    if not isinstance(settings, dict) or settings.get("schema") != 1:
        raise ValueError(
            "Missing v2 training settings; use the checkpoint's original preprocessing"
        )
    if settings.get("data_config") != DATA_CONFIG or data_config != DATA_CONFIG:
        raise ValueError("Checkpoint preprocessing mismatch; use kmu26_auv_real_v2")
    if (
        settings.get("target_loss_weight") != 0
        or config.get("action_head_cfg", {}).get("target_loss_weight") != 0
    ):
        raise ValueError("Checkpoint must persist CAP-off training and inference")
    _validate_settings(settings)
    for key, value in (
        ("neutral_pwm", neutral_pwm),
        ("pwm_span", pwm_span),
        ("expected_mode", expected_mode),
    ):
        if value is not None and settings[key] != value:
            raise ValueError(f"Recorded {key}={settings[key]} differs from requested {value}")
    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--neutral_pwm", type=int, required=True)
    parser.add_argument("--pwm_span", type=int, required=True)
    parser.add_argument("--expected_mode", required=True)
    args = parser.parse_args()
    settings = validate_real_checkpoint(
        args.checkpoint,
        DATA_CONFIG,
        neutral_pwm=args.neutral_pwm,
        pwm_span=args.pwm_span,
        expected_mode=args.expected_mode,
    )
    print(json.dumps(settings, indent=2))

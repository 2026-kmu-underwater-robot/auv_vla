# KMU26 real-vehicle training fixes

Use `kmu26_auv_real_v2` for new real-vehicle training. The existing
`kmu26_auv_real` configuration is retained for checkpoints trained with its
46-value sin/cos state representation. Do not switch an existing checkpoint to
v2 only at inference time: v2 needs separate fine-tuning with the matching input
representation. Simulator-specific integration is on the `uuvsim` branch.

## Corrections

- Preserve the 23 recorded state values. Depth, acceleration, previous commands,
  quaternion components and validity flags are not scalar periodic joint angles.
  Applying sin/cos indiscriminately both doubled state dimensions and aliased
  physically different readings. Only the new opt-in profile removes that step.
- Train only on complete 16-action windows. An episode of 20 frames contributes
  five starting points, not 20 with repeated terminal actions. The v2 training
  entry point requires one preselected dataset because the general mixture
  sampler independently samples terminal indices. Export compatible sessions
  together instead of bypassing this restriction.
- Reject connection checks, simulation data, interrupted/unreviewed recordings,
  stale vehicle state, duplicate camera captures and mixed PWM/mode settings.
  Read the recorded command range instead of forcing a simulator span of 400.
- Persist `target_loss_weight` to both runtime and saved model configurations.
  The v2 recorder supplies no CAP target labels, so v2 explicitly requires
  `--target-loss-weight 0`. Other training profiles retain their selected value.
- Construct Beta distribution parameters in float32 even when the model is
  constructed in bf16. Sampling then casts to the requested model dtype.
- Both inference service entry points check saved v2 preprocessing/CAP settings
  before loading model weights. The RC adapter accepts U0's CAP-free
  `[actions, null]` response while retaining shape, finite-value and range checks.

The physical adapter's dry-run, deadman, enable, command limit (0.3), slew limit,
neutral (1500), span (300), arm and mode checks are unchanged. No model weights,
experimental sampling heuristics, GUI, or PPO changes are included.

## New training and inference

Collect/export with the accompanying recorder integrity changes. Each real task
recording needs `data_source=real`, `use_sim_time=false`, its session/configuration
evidence, per-frame vehicle state, and an operator-reviewed successful outcome.
Legacy exports lacking that evidence are rejected rather than guessed. Failure or
recovery demonstrations need a separately designed, reviewed training selection.
Split by session before export; `assert_disjoint_sessions` in
`gr00t.data.kmu26_dataset` checks training/validation session overlap.

```bash
python scripts/gr00t_finetune.py \
  --dataset-path /path/to/reviewed-real-train \
  --output-dir /path/to/new-real-model \
  --data-config kmu26_auv_real_v2 \
  --target-loss-weight 0

python -m gr00t.deployment.kmu26_training /path/to/new-real-model \
  --neutral_pwm 1500 --pwm_span 300 --expected_mode STABILIZE

python scripts/inference_service_u0.py \
  --model-path /path/to/new-real-model \
  --data-config kmu26_auv_real_v2 \
  --server --http-server --host 127.0.0.1 --port 8000
```

The preflight values above are the existing adapter defaults, not a calibration
instruction. Supply the actual deployment settings and resolve any mismatch
before using a model. Saved `config.json` contains `kmu26_training` metadata in
addition to the existing experiment metadata/statistics. Keep both with every
checkpoint. The server cannot read the remote FCU's parameters; the preflight
compares explicitly supplied values and does not certify real vehicle response.

## Tests and evidence

```bash
python -m pytest -q tests/test_real_training_integrity.py \
  tests/test_kmu26_modality.py tests/test_kmu26_inference_contract.py \
  tests/test_kmu26_command_contract.py
```

The focused suite passed 40 tests. Six cases fail with the corresponding fixes
removed: state preservation, terminal-window indexing, bf16 sampling, CAP
save/reload, and both list/tuple CAP-free response forms. The legacy state config
and bounded RC behavior are covered. The training-entry test substitutes the
expensive model/runner but executes the real training function and saves/reloads
its configuration; the bf16 test constructs an actual small action head.

A synthetic recording also passed the real collector -> MP4/Parquet exporter ->
U0 decoder/transforms/model-input path: 20 frames, five complete chunks,
state `(1, 64)` with 23 valid entries and action `(16, 32)` with 64 valid entries.
These tests do not establish physical-vehicle success, calibrated sensor timing,
or task performance. No vehicle was connected and no production training ran.

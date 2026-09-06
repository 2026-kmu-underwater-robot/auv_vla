# KMU26 real-AUV embodiment

This configuration fine-tunes U0 on data recorded from the physical KMU26 AUV. It intentionally
uses normalized body commands instead of individual thruster PWM values. ArduSub remains responsible
for stabilization and motor mixing.

## Observation contract

Both videos must be present for every sample. Store a placeholder frame and set the corresponding
validity flag to `0` when a camera is unavailable.

| Key | Shape | Convention |
| --- | --- | --- |
| `video.ego` | `(T, H, W, 3)` | RGB forward camera |
| `video.buoy_release` | `(T, H, W, 3)` | RGB close-range release camera |
| `state.prev_command` | `(T, 4)` | previous `[surge, sway, heave, yaw]` command |
| `state.dvl_velocity` | `(T, 3)` | body-frame linear velocity in m/s |
| `state.angular_velocity` | `(T, 3)` | body-frame angular velocity in rad/s |
| `state.linear_acceleration` | `(T, 3)` | body-frame acceleration in m/s^2 |
| `state.attitude` | `(T, 4)` | normalized quaternion `[w, x, y, z]` |
| `state.depth` | `(T, 1)` | positive-down depth in metres |
| `state.altitude` | `(T, 1)` | DVL bottom altitude in metres |
| `state.validity` | `(T, 4)` | `[ego, buoy_release, dvl_velocity, altitude]` validity |

All vector measurements must be transformed into one documented body-frame convention before they
are written to the dataset. Do not silently replace an invalid DVL observation with a valid-looking
zero; use zero as a placeholder and clear its validity bit.

## Action contract

`action.motion` has shape `(T, 4)` and order `[surge, sway, heave, yaw]`. Every component is the
operator command normalized to `[-1, 1]`; it is not motor PWM. Record the actual MAVROS command and
FCU output as additional telemetry, but do not use those fields as the action label.

The policy predicts 16 steps at 10 Hz. A real robot controller should execute only the first few
steps before replanning and must apply independent command limits, slew-rate limits, stale-data
checks, a deadman switch, and an operator override.

## Training

Use the U0 checkpoint as the base model and select this data configuration:

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/kmu26_real/train \
    --output-dir /path/to/checkpoints/kmu26_real \
    --base-model-path /path/to/u0_final \
    --data-config kmu26_auv_real \
    --embodiment-tag new_embodiment \
    --tune-visual \
    --target-loss-weight 0
```

The dataset must provide its own `meta/stats.json`; do not reuse USIM normalization statistics.
Split train and validation data by complete episode and physical environment, never by individual
frames.

Start an HTTP inference server for a trained checkpoint with:

```bash
python scripts/inference_service_u0.py \
    --server \
    --http-server \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda:0 \
    --model-path /path/to/checkpoints/kmu26_real \
    --data-config kmu26_auv_real \
    --embodiment-tag new_embodiment
```

The same script's `--client` mode builds a shape-correct synthetic KMU26 observation when
`--data-config kmu26_auv_real` is selected. It is only a transport/shape smoke test; never forward
its action to a vehicle.

## Buoy-release trigger

This initial embodiment only predicts vehicle motion. A one-shot release actuator should not be
added as a sparsely positive fifth diffusion action without validating the device semantics and
safety interlock. Add a separately gated event head after the release mechanism and labels are
specified.

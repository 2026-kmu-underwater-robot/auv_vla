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

## Current ROV / MuJoCo execution adapter

Install the updated [auv_vla_data_collector](https://github.com/2026-kmu-underwater-robot/auv_vla_data_collector)
with `auv_dvl_a50_msg` in the ROS Humble workspace. This adapter requires its
`policy_observation` API and shares the recorder's IMX219, FRD-to-FLU, depth,
freshness and 23-dimensional observation contract. Build/source that workspace
before running the following commands from this repository root:

```bash
python3 -m pip install -r examples/KMU26AUV/requirements-ros.txt
# No GPU/torch install is needed in this ROS environment.
PYTHONPATH="$PWD:$PYTHONPATH" python3 -m gr00t.deployment.kmu26_ros \
  --ros-args --params-file examples/KMU26AUV/rov_runtime.yaml
```

The example uses simulation time. Set the collector's `use_sim_time` to false
for the vehicle. Both images must be fresh; verify camera 1 actually views the
release area. The shared collector runs inside the adapter: do not launch a
second collector with the same name. Its recording services remain available
for optional telemetry collection. Supply `/vla/task_description` as usual.

The adapter starts disabled and defaults to `/vla/proposed_rc` only. Start the
separate HTTP server with a locally trained KMU26 checkpoint. The server rejects
original U0 metadata when `kmu26_auv_real` is selected; changing a config name does
not train a four-axis policy. A metadata check verifies compatibility, not policy
quality or training provenance.

To exercise dry-run execution, publish `/vla/deadman` (`std_msgs/msg/Bool`, true)
continuously at 10 Hz, then call `/vla/enable` (`std_srvs/srv/SetBool`, true).
Connect this heartbeat to an operator-held control, not an unattended publisher
for live operation. False or a missing heartbeat releases the four owned primary
channels and requires a new enable request. Set `dry_run: false` only for a
validated policy; this changes the output to `/mavros/rc/override` and additionally
requires a recent connected/armed `/mavros/state` in `expected_mode`. The adapter
never arms or changes vehicle mode. Keep collection and execution in the same
mode with the same PWM span/sign conventions. Disable VLA before joystick or
another controller takes RC ownership; this node is not an RC arbitration mux.

HTTP inference runs outside the ROS callback thread. Commands are bounded and
slew-limited; expired observations, inference timeouts, clock resets/pauses and
control-loop stalls disable output. Only the still-current part of a 16-step
10 Hz chunk is used (default maximum age 0.3 s), never all 1.6 s open-loop.
Responses from before a disable/re-enable are discarded. Actual inference latency
must fit the chosen age budget; measure it before allowing commands. The four
channels are `[5, 6, 3, 4]`, neutral 1500, span 300; all other channels are untouched.
Release-actuator operation is still outside this motion policy.

## RTX 5080 environment boundary

Keep ROS Humble and GPU inference/training in separate environments. The existing
`.[base]` extra pins torch 2.5.1 and must not be used unchanged on the RTX 5080.
Use the dedicated `.[blackwell]` extra after installing its matching PyTorch CUDA
12.8 wheels. Do not combine `base` and `blackwell` extras. FlashAttention must be
validated separately for the installed GPU/CUDA/PyTorch combination; an import
success is not evidence that the CUDA kernels or complete model run correctly.
This PR does not claim measured 5080 latency, VRAM headroom, full-model training,
or a trained KMU26 checkpoint. Start with offline evaluation and small batches.

```bash
# Separate Python 3.10 GPU environment; not the ROS environment.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[blackwell]'
```

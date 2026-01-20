# Ship Landing Simulation Troubleshooting Guide

## Quick Diagnostics

### Check Gazebo Status
```bash
# List topics (should show /world/ship_landing_debug/...)
gz topic -l

# Check pose data
gz topic -e -t /world/ship_landing_debug/dynamic_pose/info -n 1

# Check simulation time is advancing
gz topic -e -t /world/ship_landing_debug/clock -n 3
```

### Run Telemetry Monitor
```bash
python3 telemetry_debug.py --log landing_debug.jsonl --duration 60
```

---

## Common Landing Abort Reasons

### 1. Sensor Noise from Waves (Heave Prediction Drift)

**Symptoms:**
- Telemetry shows large `heave_prediction_error` values (> 0.5m)
- Drone oscillates vertically during terminal phase
- Landing attempts repeatedly abort at low altitude

**Diagnosis:**
```bash
# Check heave prediction error in logs
python3 -c "
import json
errors = []
with open('landing_debug.jsonl') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            errors.append(abs(data['heave_prediction_error']))
print(f'Max heave error: {max(errors):.3f}m')
print(f'RMS heave error: {(sum(e**2 for e in errors)/len(errors))**0.5:.3f}m')
"
```

**Solutions:**
1. Verify wave model parameters match actual sea state
2. Increase `filter_alpha` in RobustPoseReader (less responsive but smoother)
3. Add predictive compensation for wave phase
4. In terminal phase, use pure velocity matching without position prediction

### 2. EKF Divergence (State Estimation Failure)

**Symptoms:**
- Velocity estimates become extremely large (> 30 m/s)
- Position jumps discontinuously
- Controller commands saturate then vehicle flies away

**Diagnosis:**
```bash
# Check for velocity spikes in log
python3 -c "
import json
with open('landing_debug.jsonl') as f:
    for i, line in enumerate(f):
        if line.strip():
            data = json.loads(line)
            vel = data['quad_vel']
            if max(abs(v) for v in vel) > 20:
                print(f'Line {i}: Velocity spike: {vel}')
"
```

**Solutions:**
1. Reduce `max_velocity` threshold in RobustPoseReader
2. Increase outlier rejection threshold
3. Use median filter instead of exponential filter
4. Check Gazebo physics step size (recommend 0.001s or smaller)

### 3. MAVLink/Communication Timeouts (Not applicable in direct Gazebo control)

For MAVLink-based systems (ArduPilot, PX4):

**Symptoms:**
- `gz topic` commands timeout
- Pose data stale (timestamp not advancing)
- Controller runs but vehicle doesn't respond

**Diagnosis:**
```bash
# Check topic publish rates
gz topic -i -t /world/ship_landing_debug/dynamic_pose/info

# Test wrench application
gz topic -t /world/ship_landing_debug/wrench/persistent \
  -m gz.msgs.EntityWrench \
  -p 'entity: {name: "quadcopter::base_link", type: LINK}, wrench: {force: {z: 20}}'
```

**Solutions:**
1. Verify `GZ_IP` and `GZ_PARTITION` environment variables
2. Check for network issues (especially in distributed simulation)
3. Restart Gazebo if transport layer is stuck
4. Increase timeout values in subprocess calls

---

## Physics Debugging

### Check Contact Forces
```bash
# Enable contact debugging
gz topic -e -t /world/ship_landing_debug/contact -n 10
```

### Verify Friction Parameters
```bash
# Check SDF loaded correctly
gz sdf -p worlds/ship_landing_debug.world | grep -A5 "friction"
```

### Common Physics Issues

1. **Quad slides off deck**: Increase friction coefficients (mu > 2.0)
2. **Quad bounces on contact**: Reduce restitution (< 0.1)
3. **Quad clips through deck**: Reduce physics step size
4. **Unstable oscillations**: Increase contact damping (kd)

---

## Logging Configuration

### Enable Gazebo State Logging
```bash
# Start with logging
gz sim -s -r --record-path /tmp/landing_log worlds/ship_landing_debug.world

# Playback
gz sim -p /tmp/landing_log/state.tlog
```

### Python Telemetry Logging
```bash
python3 telemetry_debug.py --log landing_$(date +%Y%m%d_%H%M%S).jsonl
```

### Post-Run Analysis
```python
import json
import matplotlib.pyplot as plt

frames = []
with open('landing_debug.jsonl') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            frames.append(json.loads(line))

times = [f['sim_time'] for f in frames]
heights = [f['height_above_deck'] for f in frames]
heave_err = [f['heave_prediction_error'] for f in frames]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(times, heights, label='Height above deck')
ax1.axhline(y=0, color='r', linestyle='--', label='Deck level')
ax1.set_ylabel('Height (m)')
ax1.legend()

ax2.plot(times, heave_err, label='Heave prediction error')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (m)')
ax2.legend()

plt.tight_layout()
plt.savefig('landing_analysis.png')
```

---

## Emergency Procedures

### Force Stop Simulation
```bash
killall -9 gz ruby
```

### Clear Persistent Wrenches
```bash
gz topic -t /world/ship_landing_debug/wrench/clear \
  -m gz.msgs.Entity \
  -p 'name: "quadcopter::base_link", type: LINK'
```

### Reset Vehicle Position
```bash
gz service -s /world/ship_landing_debug/set_pose \
  --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 \
  --req 'name: "quadcopter", position: {x: -20, y: 0, z: 15}'
```

---

## Performance Tuning

### If landing takes too long:
- Increase `max_accel` in guidance config
- Reduce `N_position` for faster response
- Decrease approach height threshold

### If landing is unstable:
- Reduce `max_accel` and `max_descent`
- Increase velocity filter strength (lower alpha)
- Add more aggressive force rate limiting

### If terminal phase fails:
- Increase `TERMINAL_HEIGHT` for earlier transition
- Tune velocity matching gains (kd_vert)
- Reduce `desired_descent_rate`

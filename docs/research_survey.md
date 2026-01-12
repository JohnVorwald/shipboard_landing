# Autonomous Shipboard Landing: Research Survey

## Overview

This document surveys the state-of-the-art in autonomous rotorcraft (helicopter and multirotor UAV) landing on moving ship decks. The survey covers guidance algorithms, control strategies, sensing approaches, and ship motion prediction methods from leading research institutions.

---

## 1. Research Institutions and Key Contributions

### 1.1 Pennsylvania State University

**Principal Investigators**: Dr. Joseph Horn, Dr. Eric Johnson

**Key Publications**:

1. **Pravitra, J. (2021)** - "Shipboard UAS Operations with Optimized Landing Trajectories" [PhD Dissertation]
   - Developed Ensemble iLQR (EiLQR) algorithm
   - GPU-parallel trajectory optimization for real-time replanning
   - RTK relative positioning between helicopter and ship
   - Flight validated on US Naval Academy yard patrol craft

2. **Truskin, B. (2013)** - "Vision-based deck state estimation for autonomous ship-board landing" [MS Thesis]
   - Monocular vision for deck pose estimation
   - Feature tracking in maritime environment

3. **Holmes, W. (2017)** - "Vision-based relative deck state estimation used with tau-based landings" [MS Thesis]
   - Tau (τ) guidance for time-to-contact control
   - Vision-based relative state estimation

4. **Yang, J. & Horn, J.F. (2017)** - "Landing path generation algorithm for autonomous shipboard helicopter recovery"
   - Path planning with deck motion forecasting
   - ONR-sponsored research (Contract N00014-14-C-0004)

**Key Technologies**:
- Ensemble iLQR for parallel trajectory optimization
- Tau-based landing guidance
- Vision-based deck tracking
- Ship motion prediction and forecasting

### 1.2 Georgia Institute of Technology

**Key Publications**:

1. **Johnson, E.N. & Schrage, D.P.** - "The Georgia Tech Unmanned Aerial Research Vehicle: GTMax"
   - Research UAV testbed for autonomous operations
   - Aggressive maneuver demonstration

2. **Autonomous Recovery System for Rotorcraft UAV in Rough Seas**
   - Extended Kalman Filter sensor fusion
   - Multi-sensor deck displacement estimation

**Key Technologies**:
- Four-phase landing: tracking → high hover → low hover → final descent
- EKF-based sensor fusion for deck tracking
- Robust recovery procedures for rough sea environments

### 1.3 University of Maryland - Alfred Gessow Rotorcraft Center

**Principal Investigator**: Dr. Inderjit Chopra

**Key Research Areas**:

1. **Vision-based ship deck landing**
   - Feature-based algorithm for tracking stochastic ship deck motion
   - Autonomous landing under visually degraded conditions
   - Quadrotor with integrated avionics for vision-based navigation

2. **VFS Forum 2021 Presentation**
   - Vision-based method to track ship deck motion
   - Validated on custom quadrotor platform

**Key Technologies**:
- Feature-based visual tracking
- Degraded visibility operation
- In-house quadrotor development

### 1.4 Rensselaer Polytechnic Institute (RPI)

**Key Publications**:

1. **"Trajectory optimization and control for autonomous helicopter shipboard landing"** [Dissertation]
   - Shrinking horizon MPC for optimal touchdown timing
   - Recursive implementation with ship motion forecast updates

2. **"A Differential-Flatness-Based Approach for Autonomous Helicopter Shipboard Landing"** [IEEE]
   - Differential flatness for trajectory planning
   - Time-optimal landing trajectories
   - Simplified nonlinear helicopter model

**Key Technologies**:
- Differential flatness trajectory generation
- Shrinking horizon model predictive control
- Optimal touchdown time selection

### 1.5 Naval Surface Warfare Center Carderock Division (NSWCCD)

**Facility**: Maneuvering and Seakeeping Basin (MASK)
- 360 ft × 240 ft wave basin
- 216 electro-mechanical waveboards
- 12 million gallons capacity
- Realistic sea state simulation

**Collaborations**:
- Penn State Applied Research Laboratories (PSU ARL)
- Indoor Vicon motion capture testing
- Ship model testing in simulated waves

---

## 2. Guidance and Control Algorithms

### 2.1 Tau-Based Landing Guidance

**Concept**: Use optical flow variable τ (tau) = range / range-rate as primary feedback

**Advantages**:
- Direct measurement from vision
- Natural deceleration profile
- Robust to scale ambiguity

**Implementation**:
```
τ = r / ṙ  (time-to-contact)
τ̇_desired = constant (e.g., 0.5 for smooth approach)
Control: adjust velocity to maintain τ̇ = τ̇_desired
```

**Reference**: Holmes (2017), Penn State

### 2.2 Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV)

**Concept**: Optimal interception guidance from missile theory

**Formulation**:
```
ZEM = r + v·t_go           (predicted miss distance)
ZEV = v - v_target         (velocity error at intercept)

a_cmd = -N·ZEM/t_go² - M·ZEV/t_go

where N ≈ 3-6, M ≈ 2-3
```

**Advantages**:
- Optimal for constant-velocity targets
- Natural deceleration profile
- Handles moving targets

### 2.3 Ensemble iLQR (EiLQR)

**Concept**: Solve multiple iLQR problems in parallel for different scenarios

**Algorithm**:
1. Define ensemble of target states (different deck predictions)
2. Solve iLQR for each scenario on GPU
3. Select best trajectory based on expected cost
4. Execute first control, repeat

**Advantages**:
- Robust to prediction uncertainty
- Real-time capable with GPU
- Handles multiple possible futures

**Reference**: Pravitra (2021), Penn State

### 2.4 Variable Horizon MPC (VH-MPC)

**Concept**: Evaluate multiple horizon lengths in parallel, select optimal

**Algorithm** (Penn State, AIAA SciTech 2024):
```
1. Define horizon options: N = [20, 30, 40, 50] steps
2. Solve MPC for each horizon in parallel (GPU)
3. Select horizon with lowest total cost J(N)
4. Apply first control, repeat
```

**Move Blocking**: Reduce decision variables by grouping timesteps
```
Instead of u[0], u[1], ..., u[N-1]  (N variables)
Use u_a for steps 0-4, u_b for 5-9, etc. (N/5 variables)
```

**Advantages**:
- Implicit touchdown time optimization
- Parallel computation (latency = max, not sum)
- Robust to prediction uncertainty

**Reference**: Ngo & Sultan, AIAA SciTech 2024; Penn State

### 2.5 Shrinking Horizon MPC

**Concept**: MPC with horizon that shrinks as touchdown approaches

**Algorithm**:
```
At time t with planned touchdown at T:
  horizon = T - t (shrinks over time)
  Solve MPC with terminal constraints at T
  Select optimal touchdown time within horizon
```

**Advantages**:
- Explicit touchdown time optimization
- Adapts to changing conditions
- Terminal constraint satisfaction

**Reference**: RPI

### 2.6 Differential Flatness Trajectory Planning

**Concept**: Transform nonlinear dynamics to equivalent linear system

**For quadrotor**: Flat outputs are position (x, y, z) and yaw (ψ)
- All states and inputs can be written as functions of flat outputs and derivatives
- Trajectory planning becomes polynomial optimization

**Advantages**:
- Simplifies trajectory generation
- Guarantees dynamic feasibility
- Efficient computation

**Reference**: RPI, IEEE

---

## 3. Ship Motion Prediction

### 3.1 ARMA Time Series Models

**Formulation**:
```
y(t) = Σ φ_i·y(t-i) + Σ θ_j·e(t-j) + e(t)
       i=1 to p        j=1 to q
```

**Typical parameters**: p=4-8, q=2-4

**Advantages**:
- Fast computation
- Adapts to current conditions
- No physical model required

**Our Results**: ~1.4m RMS error at 3s horizon (sea state 4)

### 3.2 Higher-Order ARMA

**Enhancement**: Include velocity and acceleration in state

**State**: [z, ż, z̈, φ, φ̇, θ, θ̇]

**Taylor expansion blending**:
```
x_pred = α·x_ARMA + (1-α)·(x + v·dt + 0.5·a·dt²)
```

**Our Results**: +12% improvement at 1s horizon

### 3.3 Wave-Based Prediction

**Concept**: Estimate wave components, propagate using physics

**Algorithm**:
1. Observe ship motion history
2. Invert using RAOs to estimate wave amplitudes/phases
3. Propagate wave model forward
4. Predict ship response using RAOs

**Challenges**:
- Phase estimation difficult
- RAO uncertainty
- Nonlinear effects

**Our Results**: Not competitive with ARMA (5-7m error vs 1.4m)

---

## 4. Sensing Approaches

### 4.1 RTK GPS

**Accuracy**: ~2cm relative positioning
**Advantages**: High accuracy, all-weather
**Disadvantages**: Requires base station on ship

### 4.2 Vision-Based

**Approaches**:
- Feature tracking (AprilTags, natural features)
- Optical flow for tau estimation
- Deep learning for pose estimation

**Advantages**: No ship-side infrastructure
**Challenges**: Lighting, weather, vibration

### 4.3 Sensor Fusion

**Typical fusion**: GPS + IMU + Vision + Altimeter

**Methods**: EKF, UKF, factor graphs

---

## 5. Implementation Summary

### 5.1 Implemented in This Codebase

| Technology | File | Status |
|------------|------|--------|
| ARMA Prediction | `ship_motion/ddg_motion.py` | Working |
| Higher-Order ARMA | `simulation/forecast_comparison.py` | Working |
| Wave Estimator | `ship_motion/ddg_motion.py` | Working (poor accuracy) |
| ZEM/ZEV Guidance | `simulation/mpc_simple.py` | Working (0.87m pos error) |
| PMP with Matching | `optimal_control/pmp_landing.py` | Working |
| Tau-Based Landing | `guidance/tau_guidance.py` | Working |
| Variable Horizon MPC | `optimal_control/variable_horizon_mpc.py` | Working |
| Controller Comparison | `simulation/controller_comparison.py` | Working |
| Differential Flatness | `trajectory/diff_flat.py` | To implement |

### 5.2 Performance Results

**Best Configuration**: Simple MPC with ZEM/ZEV + Higher-Order ARMA
- Position error: 2.4m mean
- Velocity error: 2.7 m/s mean
- Success rate: 10% (sea state 4)

**Forecasting**: Higher-Order ARMA with 0.5s updates
- 1s horizon: 1.57m Z error
- 3s horizon: 1.44m Z error

---

## 6. References

### Dissertations and Theses

1. Pravitra, J. (2021). "Shipboard UAS Operations with Optimized Landing Trajectories." PhD Dissertation, Pennsylvania State University.

2. Holmes, W. (2017). "Vision-based relative deck state estimation used with tau-based landings." MS Thesis, Pennsylvania State University.

3. Truskin, B. (2013). "Vision-based deck state estimation for autonomous ship-board landing." MS Thesis, Pennsylvania State University.

### Journal Papers

4. IEEE (2021). "A Differential-Flatness-Based Approach for Autonomous Helicopter Shipboard Landing." IEEE Transactions.

5. Journal of Intelligent & Robotic Systems (2021). "Autonomous Landing of Rotary Wing Unmanned Aerial Vehicles on Underway Ships in a Sea State."

6. Aerospace Science and Technology (2022). "Autonomous ship deck landing of a quadrotor UAV using feed-forward image-based visual servoing."

### Conference Papers

7. Yang, J. & Horn, J.F. (2017). "Landing path generation algorithm for autonomous shipboard helicopter recovery." 7th AHS Technical Meeting on VTOL UAS and Autonomy.

8. VFS Forum (2021). UMD ship deck landing research presentation.

9. Ngo, T.D. & Sultan, C. (2024). "Robust Variable Horizon MPC with Move Blocking for Helicopter Shipboard Landing on Moving Decks." AIAA SciTech Forum. https://arc.aiaa.org/doi/abs/10.2514/6.2024-2398

10. Ngo, T.D. & Sultan, C. "Variable Horizon Model Predictive Control for Helicopter Landing on Moving Decks." Journal of Guidance, Control, and Dynamics. https://arc.aiaa.org/doi/10.2514/1.G005789

### Technical Reports

11. ONR Contract N00014-14-C-0004. "Autonomous Control Modes and Optimized Path Guidance for Shipboard Landing."

---

## 7. Future Work

1. **Vicon Integration**: Test algorithms with motion capture at NSWCCD MASK
2. **Wave Basin Validation**: Compare simulation to physical wave basin tests
3. **Real-Time GPU**: Implement EiLQR on embedded GPU
4. **Vision Pipeline**: Integrate feature-based deck tracking
5. **Higher Sea States**: Extend to sea state 5+

---

*Document generated: January 2026*
*Contributors: Research collaboration with Penn State Aerospace Engineering*

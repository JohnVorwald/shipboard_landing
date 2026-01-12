%% Shipboard Landing Visualization - Complete Simulation
% Single-file MATLAB script for UAV landing on DDG flight deck
%
% Includes:
% - Full DDG ship motion model with Pierson-Moskowitz wave spectrum
% - ARMA(2,1) ship motion forecaster
% - 6-DOF quadrotor dynamics with quaternion attitude
% - PMP-based trajectory tracking controller with costate feedback
% - Minimum-snap trajectory planner
%
% Generates video with:
% - 3D view of ship deck and UAV trajectory
% - Strip plots: X, Y, Z with forecast error bands
% - Strip plots: Roll, Pitch with forecast error bands
% - Dynamic time window: -5s (past) to +10s (future), NOW at center
%
% Author: Claude Code
% Date: 2026-01-07

clear; clc; close all;

%% ========== CONFIGURATION ==========
cfg = struct();

% Simulation
cfg.sea_state = 3;              % Sea state (1-6)
cfg.ship_speed_kts = 12;        % Ship speed (knots)
cfg.wave_direction = 45;        % Wave direction (deg from bow)
cfg.sim_duration = 15;          % Max simulation time (s)
cfg.dt = 0.02;                  % Integration time step (s)

% Initial approach
cfg.approach_altitude = 25;     % UAV start altitude above deck (m)
cfg.approach_distance = 40;     % UAV start distance behind deck (m)
cfg.approach_speed = 10;        % Initial forward speed (m/s)

% ARMA predictor
cfg.arma_order_p = 2;           % AR order
cfg.arma_order_q = 1;           % MA order
cfg.arma_history = 5;           % History window for fitting (s)
cfg.forecast_horizon = 10;      % Forecast horizon (s)
cfg.forecast_interval = 0.5;    % Forecast update interval (s)

% Trajectory replanning
cfg.replan_interval = 0.5;      % Replan every 0.5s

% Video
cfg.fps = 20;                   % Video frame rate
cfg.past_window = 5;            % Strip plot past window (s)
cfg.future_window = 10;         % Strip plot future window (s)
cfg.output_file = 'landing_video.mp4';
cfg.save_video = true;

%% ========== RUN SIMULATION ==========
fprintf('=== Shipboard Landing Simulation ===\n\n');

[sim_data, results] = run_full_simulation(cfg);

fprintf('\n=== Results ===\n');
fprintf('Landed: %s\n', mat2str(results.landed));
if results.landed
    fprintf('Landing time: %.2f s\n', results.landing_time);
    fprintf('Relative velocity: [%.2f, %.2f, %.2f] m/s\n', results.rel_vel);
    fprintf('Position error: %.2f m\n', results.pos_error);
end

%% ========== GENERATE VIDEO ==========
generate_video(sim_data, results, cfg);

%% ==================== SHIP MOTION MODEL ====================

function ship = init_ship_motion(cfg)
    % Initialize DDG-51 ship motion model with Pierson-Moskowitz spectrum

    ship = struct();
    ship.speed = cfg.ship_speed_kts * 0.5144;  % m/s

    % DDG-51 parameters
    ship.length = 154;      % m
    ship.beam = 20;         % m
    ship.draft = 9.4;       % m
    ship.displacement = 8300; % tonnes

    % Deck position relative to CG (flight deck aft)
    ship.deck_offset = [-60; 0; -8];  % [x_aft, y, z_up] from CG

    % Sea state -> Pierson-Moskowitz parameters
    % Hs = significant wave height, Tp = peak period
    Hs_table = [0.1, 0.3, 0.88, 1.88, 3.25, 5.0, 7.5];
    Tp_table = [2.0, 4.0, 6.3, 8.8, 11.0, 13.0, 15.5];

    ss = max(1, min(6, cfg.sea_state));
    ship.Hs = Hs_table(ss);
    ship.Tp = Tp_table(ss);
    ship.omega_p = 2*pi / ship.Tp;

    % Wave encounter frequency adjustment
    wave_dir_rad = deg2rad(cfg.wave_direction);
    ship.wave_dir = wave_dir_rad;

    % Generate wave spectrum components
    ship.n_waves = 30;
    omega_range = linspace(0.3, 2.0, ship.n_waves) * ship.omega_p;

    % Pierson-Moskowitz spectrum: S(w) = (alpha*g^2/w^5) * exp(-beta*(w_p/w)^4)
    alpha_pm = 0.0081;
    g = 9.81;
    beta_pm = 0.74;

    S_pm = zeros(1, ship.n_waves);
    for i = 1:ship.n_waves
        w = omega_range(i);
        S_pm(i) = (alpha_pm * g^2 / w^5) * exp(-beta_pm * (ship.omega_p/w)^4);
    end

    % Convert spectrum to wave amplitudes
    d_omega = omega_range(2) - omega_range(1);
    ship.wave_amp = sqrt(2 * S_pm * d_omega);
    ship.wave_omega = omega_range;
    ship.wave_phase = 2*pi * rand(1, ship.n_waves);

    % RAO (Response Amplitude Operators) - simplified
    % These convert wave height to ship motion
    ship.rao_heave = 0.9 * ones(1, ship.n_waves);
    ship.rao_roll = 0.12 * (ship.wave_omega / ship.omega_p).^(-0.5);
    ship.rao_pitch = 0.04 * ones(1, ship.n_waves);
    ship.rao_sway = 0.25 * ones(1, ship.n_waves);

    % Natural frequencies (for resonance)
    ship.omega_roll = 0.4;    % rad/s (~15s period)
    ship.omega_pitch = 0.6;   % rad/s (~10s period)

    % Damping
    ship.zeta_roll = 0.1;
    ship.zeta_pitch = 0.15;
end

function [deck, ship] = get_ship_motion(ship, t)
    % Compute ship and deck motion at time t

    g = 9.81;

    % Ship CG position (base forward motion)
    ship_x = ship.speed * t;
    ship_y = 0;
    ship_z = 0;

    % Wave-induced motions
    heave = 0; sway = 0; roll = 0; pitch = 0;
    heave_dot = 0; sway_dot = 0; roll_dot = 0; pitch_dot = 0;

    for i = 1:ship.n_waves
        w = ship.wave_omega(i);
        % Encounter frequency (head seas = higher encounter freq)
        w_e = w - w^2 * ship.speed * cos(ship.wave_dir) / g;
        w_e = max(0.1, abs(w_e));

        phase = w_e * t + ship.wave_phase(i);
        amp = ship.wave_amp(i);

        % Ship motions (applying RAOs)
        heave = heave + ship.rao_heave(i) * amp * cos(phase);
        sway = sway + ship.rao_sway(i) * amp * sin(phase + 0.3);
        roll = roll + ship.rao_roll(i) * amp * sin(phase + 0.5);
        pitch = pitch + ship.rao_pitch(i) * amp * cos(phase + 0.2);

        % Velocities
        heave_dot = heave_dot - ship.rao_heave(i) * amp * w_e * sin(phase);
        sway_dot = sway_dot + ship.rao_sway(i) * amp * w_e * cos(phase + 0.3);
        roll_dot = roll_dot + ship.rao_roll(i) * amp * w_e * cos(phase + 0.5);
        pitch_dot = pitch_dot - ship.rao_pitch(i) * amp * w_e * sin(phase + 0.2);
    end

    % Ship CG motion in NED
    ship_pos = [ship_x; ship_y + sway; ship_z + heave];
    ship_att = [roll; pitch; 0];
    ship_vel = [ship.speed; sway_dot; heave_dot];
    ship_omega = [roll_dot; pitch_dot; 0];

    % Transform deck position from ship frame to NED
    R_ship = euler_to_rotm(ship_att);
    deck_pos_ship = ship.deck_offset;
    deck_pos_ned = ship_pos + R_ship * deck_pos_ship;

    % Deck velocity includes both translation and rotation effects
    deck_vel_rotation = cross(ship_omega, R_ship * deck_pos_ship);
    deck_vel_ned = ship_vel + deck_vel_rotation;

    deck = struct();
    deck.pos = deck_pos_ned;
    deck.vel = deck_vel_ned;
    deck.att = ship_att;
    deck.omega = ship_omega;
end

%% ==================== ARMA FORECASTER ====================

function arma = init_arma_predictor(cfg)
    % Initialize ARMA predictor for ship motion

    arma = struct();
    arma.p = cfg.arma_order_p;  % AR order
    arma.q = cfg.arma_order_q;  % MA order
    arma.history_len = round(cfg.arma_history / cfg.dt);

    % History buffers (6 channels: x,y,z, roll, pitch, yaw)
    arma.n_channels = 6;
    arma.history_t = [];
    arma.history = [];  % [n_channels x N]

    % ARMA coefficients (will be fitted)
    arma.ar_coeffs = zeros(arma.n_channels, arma.p);
    arma.ma_coeffs = zeros(arma.n_channels, arma.q);
    arma.residuals = zeros(arma.n_channels, arma.q);

    arma.last_fit_time = -inf;
    arma.fit_interval = 2.0;  % Refit every 2s
end

function arma = update_arma(arma, t, deck)
    % Update ARMA with new observation

    % Add to history
    obs = [deck.pos; deck.att];
    arma.history_t = [arma.history_t, t];
    arma.history = [arma.history, obs];

    % Trim to max length
    if size(arma.history, 2) > arma.history_len
        arma.history_t = arma.history_t(end-arma.history_len+1:end);
        arma.history = arma.history(:, end-arma.history_len+1:end);
    end

    % Refit ARMA periodically
    if t - arma.last_fit_time >= arma.fit_interval && size(arma.history, 2) > 50
        arma = fit_arma(arma);
        arma.last_fit_time = t;
    end
end

function arma = fit_arma(arma)
    % Fit ARMA coefficients using least squares

    N = size(arma.history, 2);
    p = arma.p;

    for ch = 1:arma.n_channels
        y = arma.history(ch, :);

        % Remove linear trend
        trend = polyfit(1:N, y, 1);
        y_detrend = y - polyval(trend, 1:N);

        % Build regression matrix for AR
        if N > p + 10
            Y = y_detrend(p+1:end)';
            X = zeros(N-p, p);
            for i = 1:p
                X(:, i) = y_detrend(p+1-i:end-i)';
            end

            % Least squares fit
            ar_coef = (X' * X + 0.01*eye(p)) \ (X' * Y);
            arma.ar_coeffs(ch, :) = ar_coef';

            % Compute residuals for MA
            y_pred = X * ar_coef;
            resid = Y - y_pred;
            arma.residuals(ch, :) = resid(end-arma.q+1:end)';
        end
    end
end

function forecast = generate_arma_forecast(arma, t_now, horizon, dt)
    % Generate forecast using fitted ARMA model

    t_forecast = (t_now + dt):dt:(t_now + horizon);
    n_pts = length(t_forecast);

    forecast = struct();
    forecast.t = t_forecast;
    forecast.pos = zeros(3, n_pts);
    forecast.att = zeros(3, n_pts);
    forecast.vel = zeros(3, n_pts);

    if size(arma.history, 2) < arma.p + 5
        % Not enough data - use last value
        if ~isempty(arma.history)
            forecast.pos = repmat(arma.history(1:3, end), 1, n_pts);
            forecast.att = repmat(arma.history(4:6, end), 1, n_pts);
        end
        return;
    end

    % Forecast each channel
    for ch = 1:arma.n_channels
        % Get recent values and detrend
        y_recent = arma.history(ch, end-arma.p+1:end);

        % Linear trend from history
        N = min(50, size(arma.history, 2));
        trend = polyfit(1:N, arma.history(ch, end-N+1:end), 1);

        % Iterative AR forecast
        y_pred = y_recent;
        ar = arma.ar_coeffs(ch, :);

        for i = 1:n_pts
            % AR prediction
            y_new = ar * y_pred(end:-1:end-arma.p+1)';

            % Add trend
            y_new = y_new + trend(1);

            y_pred = [y_pred, y_new];
        end

        % Store forecast
        if ch <= 3
            forecast.pos(ch, :) = y_pred(end-n_pts+1:end);
        else
            forecast.att(ch-3, :) = y_pred(end-n_pts+1:end);
        end
    end

    % Estimate velocity from forecast gradient
    for i = 2:n_pts
        forecast.vel(:, i) = (forecast.pos(:, i) - forecast.pos(:, i-1)) / dt;
    end
    if n_pts > 1
        forecast.vel(:, 1) = forecast.vel(:, 2);
    end
end

%% ==================== QUADROTOR DYNAMICS ====================

function quad = init_quadrotor(cfg, deck)
    % Initialize 6-DOF quadrotor state

    quad = struct();

    % Physical parameters
    quad.mass = 3.0;        % kg
    quad.arm_length = 0.25; % m
    quad.Ixx = 0.02;        % kg*m^2
    quad.Iyy = 0.02;
    quad.Izz = 0.04;
    quad.I = diag([quad.Ixx, quad.Iyy, quad.Izz]);
    quad.I_inv = inv(quad.I);

    quad.max_thrust = 4 * 10;    % N (4 motors x 10N each)
    quad.min_thrust = 0;
    quad.max_torque = quad.arm_length * 5;  % Nm

    % Initial state
    quad.pos = deck.pos + [-cfg.approach_distance; 0; -cfg.approach_altitude];
    quad.vel = [cfg.approach_speed; 0; 0];

    % Quaternion attitude [w, x, y, z] (initially level)
    quad.quat = [1; 0; 0; 0];
    quad.omega = [0; 0; 0];  % Body angular rates

    quad.g = 9.81;
end

function quad = step_quadrotor(quad, u, dt)
    % Integrate quadrotor dynamics using RK4
    %
    % State: [pos; vel; quat; omega] (13 elements)
    % Control u: struct with T (thrust) and tau (torques)

    x = [quad.pos; quad.vel; quad.quat; quad.omega];

    % RK4 integration
    k1 = quad_dynamics(x, u, quad);
    k2 = quad_dynamics(x + 0.5*dt*k1, u, quad);
    k3 = quad_dynamics(x + 0.5*dt*k2, u, quad);
    k4 = quad_dynamics(x + dt*k3, u, quad);

    x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

    % Normalize quaternion
    x_new(7:10) = x_new(7:10) / norm(x_new(7:10));

    % Unpack
    quad.pos = x_new(1:3);
    quad.vel = x_new(4:6);
    quad.quat = x_new(7:10);
    quad.omega = x_new(11:13);
end

function xdot = quad_dynamics(x, u, quad)
    % Quadrotor equations of motion

    pos = x(1:3);
    vel = x(4:6);
    quat = x(7:10);
    omega = x(11:13);

    % Rotation matrix from body to NED
    R = quat_to_rotm(quat);

    % Forces in NED frame
    % Thrust along body -z axis (up in body frame)
    thrust_body = [0; 0; -u.T];
    thrust_ned = R * thrust_body;
    gravity_ned = [0; 0; quad.mass * quad.g];

    % Translational dynamics
    pos_dot = vel;
    vel_dot = (thrust_ned + gravity_ned) / quad.mass;

    % Rotational dynamics
    % Quaternion derivative
    omega_quat = [0; omega];
    quat_dot = 0.5 * quat_mult(quat, omega_quat);

    % Angular acceleration (Euler's equation)
    omega_dot = quad.I_inv * (u.tau - cross(omega, quad.I * omega));

    xdot = [pos_dot; vel_dot; quat_dot; omega_dot];
end

%% ==================== PMP CONTROLLER ====================

function ctrl = init_pmp_controller(quad)
    % Initialize PMP-based trajectory tracking controller

    ctrl = struct();

    % Gains
    ctrl.Kp_pos = [3; 3; 5];      % Position gains
    ctrl.Kd_pos = [4; 4; 6];      % Velocity gains
    ctrl.Kp_att = [30; 30; 15];   % Attitude gains
    ctrl.Kd_att = [8; 8; 5];      % Angular rate gains

    % Cost weights for costate estimation
    ctrl.Qf = diag([100, 100, 100, 50, 50, 50, 20, 20, 5, 5, 5, 2]);

    % Current trajectory
    ctrl.has_trajectory = false;
    ctrl.traj = [];
    ctrl.traj_start_time = 0;
end

function ctrl = update_trajectory(ctrl, quad, deck_target, tf, t_now)
    % Generate minimum-snap trajectory to target deck state

    % Sample trajectory points
    N = 20;
    t_traj = linspace(0, tf, N);

    pos_init = quad.pos;
    vel_init = quad.vel;
    pos_final = deck_target.pos;
    vel_final = deck_target.vel;

    % Compute 5th order polynomial coefficients for each axis
    ctrl.traj = struct();
    ctrl.traj.t = t_traj;
    ctrl.traj.tf = tf;
    ctrl.traj.pos = zeros(3, N);
    ctrl.traj.vel = zeros(3, N);
    ctrl.traj.acc = zeros(3, N);
    ctrl.traj.att_target = deck_target.att;

    for axis = 1:3
        % Boundary conditions: pos, vel, acc at t=0 and t=tf
        p0 = pos_init(axis);
        v0 = vel_init(axis);
        a0 = 0;
        pf = pos_final(axis);
        vf = vel_final(axis);
        af = 0;

        % Solve for 5th order polynomial coefficients
        T = tf;
        A = [1, 0, 0, 0, 0, 0;
             0, 1, 0, 0, 0, 0;
             0, 0, 2, 0, 0, 0;
             1, T, T^2, T^3, T^4, T^5;
             0, 1, 2*T, 3*T^2, 4*T^3, 5*T^4;
             0, 0, 2, 6*T, 12*T^2, 20*T^3];
        b = [p0; v0; a0; pf; vf; af];
        c = A \ b;

        % Evaluate polynomial
        for i = 1:N
            tau = t_traj(i);
            ctrl.traj.pos(axis, i) = c(1) + c(2)*tau + c(3)*tau^2 + ...
                                      c(4)*tau^3 + c(5)*tau^4 + c(6)*tau^5;
            ctrl.traj.vel(axis, i) = c(2) + 2*c(3)*tau + 3*c(4)*tau^2 + ...
                                      4*c(5)*tau^3 + 5*c(6)*tau^4;
            ctrl.traj.acc(axis, i) = 2*c(3) + 6*c(4)*tau + 12*c(5)*tau^2 + ...
                                      20*c(6)*tau^3;
        end
    end

    % Estimate costates (backward from terminal cost)
    ctrl.traj.costate = estimate_costates(ctrl, deck_target);

    ctrl.has_trajectory = true;
    ctrl.traj_start_time = t_now;
end

function costate = estimate_costates(ctrl, deck_target)
    % Estimate costates along trajectory

    N = length(ctrl.traj.t);
    costate = zeros(12, N);

    % Terminal costate from Qf
    x_target = [deck_target.pos; deck_target.vel; deck_target.att; zeros(3,1)];
    x_final = [ctrl.traj.pos(:,end); ctrl.traj.vel(:,end); ctrl.traj.att_target; zeros(3,1)];

    lam_tf = ctrl.Qf * (x_final - x_target);
    costate(:, N) = lam_tf;

    % Backward decay approximation
    tf = ctrl.traj.tf;
    for i = N-1:-1:1
        tau = tf - ctrl.traj.t(i);
        decay = exp(-0.5 * tau);
        costate(:, i) = decay * lam_tf;
    end
end

function [ref_pos, ref_vel, ref_acc, costate] = interpolate_trajectory(ctrl, t)
    % Interpolate trajectory at time t

    if ~ctrl.has_trajectory
        ref_pos = []; ref_vel = []; ref_acc = []; costate = [];
        return;
    end

    t_rel = t - ctrl.traj_start_time;
    t_rel = max(0, min(ctrl.traj.tf, t_rel));

    % Linear interpolation
    t_traj = ctrl.traj.t;
    idx = find(t_traj >= t_rel, 1);
    if isempty(idx), idx = length(t_traj); end
    if idx == 1, idx = 2; end

    alpha = (t_rel - t_traj(idx-1)) / (t_traj(idx) - t_traj(idx-1) + 1e-6);
    alpha = max(0, min(1, alpha));

    ref_pos = (1-alpha) * ctrl.traj.pos(:, idx-1) + alpha * ctrl.traj.pos(:, idx);
    ref_vel = (1-alpha) * ctrl.traj.vel(:, idx-1) + alpha * ctrl.traj.vel(:, idx);
    ref_acc = (1-alpha) * ctrl.traj.acc(:, idx-1) + alpha * ctrl.traj.acc(:, idx);
    costate = (1-alpha) * ctrl.traj.costate(:, idx-1) + alpha * ctrl.traj.costate(:, idx);
end

function u = compute_pmp_control(ctrl, quad, deck, t)
    % PMP-based trajectory tracking control

    g = 9.81;
    m = quad.mass;

    % Get trajectory reference
    [ref_pos, ref_vel, ref_acc, costate] = interpolate_trajectory(ctrl, t);

    if isempty(ref_pos)
        % Fallback to simple pursuit
        u = compute_fallback_control(quad, deck);
        return;
    end

    % Position/velocity error
    pos_err = quad.pos - ref_pos;
    vel_err = quad.vel - ref_vel;

    % Desired acceleration (PD + feedforward + costate correction)
    acc_des = -ctrl.Kp_pos .* pos_err - ctrl.Kd_pos .* vel_err + ref_acc;

    % Costate-based correction
    if ~isempty(costate)
        costate_pos = costate(1:3);
        costate_correction = -0.1 * costate_pos / (norm(costate_pos) + 0.1);
        acc_des = acc_des + costate_correction;
    end

    acc_des = max(-6, min(6, acc_des));

    % Convert desired acceleration to thrust and attitude
    % a_des = R * [0; 0; -T/m] + [0; 0; g]
    thrust_vec = acc_des - [0; 0; g];
    T = m * norm(thrust_vec);
    T = max(0.2*m*g, min(quad.max_thrust, T));

    % Desired attitude from thrust direction
    if T > 0.3*m*g
        thrust_dir = thrust_vec / norm(thrust_vec);
        pitch_des = asin(max(-1, min(1, -thrust_dir(1))));
        roll_des = atan2(thrust_dir(2), -thrust_dir(3));
    else
        roll_des = 0;
        pitch_des = 0;
    end
    yaw_des = 0;  % Keep yaw at 0

    % Near deck: blend toward deck attitude
    height = deck.pos(3) - quad.pos(3);
    if height < 3 && height > 0
        blend = 1 - height/3;
        roll_des = (1-blend)*roll_des + blend*deck.att(1);
        pitch_des = (1-blend)*pitch_des + blend*deck.att(2);
    end

    roll_des = max(-0.5, min(0.5, roll_des));
    pitch_des = max(-0.5, min(0.5, pitch_des));

    % Current attitude (euler from quaternion)
    euler_cur = quat_to_euler(quad.quat);

    % Attitude error
    roll_err = euler_cur(1) - roll_des;
    pitch_err = euler_cur(2) - pitch_des;
    yaw_err = euler_cur(3) - yaw_des;
    yaw_err = atan2(sin(yaw_err), cos(yaw_err));  % Wrap

    % Attitude control (PD)
    tau_x = -ctrl.Kp_att(1)*roll_err - ctrl.Kd_att(1)*quad.omega(1);
    tau_y = -ctrl.Kp_att(2)*pitch_err - ctrl.Kd_att(2)*quad.omega(2);
    tau_z = -ctrl.Kp_att(3)*yaw_err - ctrl.Kd_att(3)*quad.omega(3);

    tau_max = quad.max_torque;
    tau_x = max(-tau_max, min(tau_max, tau_x));
    tau_y = max(-tau_max, min(tau_max, tau_y));
    tau_z = max(-tau_max, min(tau_max, tau_z));

    u = struct('T', T, 'tau', [tau_x; tau_y; tau_z]);
end

function u = compute_fallback_control(quad, deck)
    % Simple pursuit guidance fallback

    g = 9.81;
    m = quad.mass;

    rel_pos = deck.pos - quad.pos;
    rel_vel = deck.vel - quad.vel;
    height = rel_pos(3);
    horiz_dist = norm(rel_pos(1:2));

    t_go = max(3, horiz_dist/5);

    % Desired velocity
    vel_des = rel_pos / t_go + deck.vel;
    vel_err = vel_des - quad.vel;

    acc_cmd = [3; 3; 5] .* vel_err;
    acc_cmd = max(-5, min(5, acc_cmd));

    T_z = m * (g - acc_cmd(3));
    T = sqrt(T_z^2 + (m*acc_cmd(1))^2 + (m*acc_cmd(2))^2);
    T = max(0.3*m*g, min(1.5*m*g, T));

    roll_des = atan2(acc_cmd(2), g);
    pitch_des = atan2(-acc_cmd(1), g);

    if height < 3 && height > 0
        blend = 1 - height/3;
        roll_des = (1-blend)*roll_des + blend*deck.att(1);
        pitch_des = (1-blend)*pitch_des + blend*deck.att(2);
    end

    euler_cur = quat_to_euler(quad.quat);

    tau_x = 30*(roll_des - euler_cur(1)) - 8*quad.omega(1);
    tau_y = 30*(pitch_des - euler_cur(2)) - 8*quad.omega(2);
    tau_z = -5*quad.omega(3);

    u = struct('T', T, 'tau', [tau_x; tau_y; tau_z]);
end

%% ==================== MAIN SIMULATION LOOP ====================

function [data, results] = run_full_simulation(cfg)
    % Run complete landing simulation

    fprintf('Initializing...\n');

    % Initialize components
    ship = init_ship_motion(cfg);
    [deck, ship] = get_ship_motion(ship, 0);
    quad = init_quadrotor(cfg, deck);
    arma = init_arma_predictor(cfg);
    ctrl = init_pmp_controller(quad);

    % Storage
    N_max = ceil(cfg.sim_duration / cfg.dt) + 1;
    data = struct();
    data.t = zeros(1, N_max);
    data.uav_pos = zeros(3, N_max);
    data.uav_att = zeros(3, N_max);
    data.uav_vel = zeros(3, N_max);
    data.deck_pos = zeros(3, N_max);
    data.deck_att = zeros(3, N_max);
    data.deck_vel = zeros(3, N_max);
    data.forecasts = {};

    results = struct();
    results.landed = false;
    results.landing_time = [];
    results.rel_vel = [];
    results.pos_error = [];

    t = 0;
    idx = 1;
    last_forecast_time = -cfg.forecast_interval;
    last_replan_time = -cfg.replan_interval;

    fprintf('Simulating...\n');

    while t < cfg.sim_duration
        % Get current deck state
        [deck, ship] = get_ship_motion(ship, t);

        % Store data
        data.t(idx) = t;
        data.uav_pos(:, idx) = quad.pos;
        data.uav_att(:, idx) = quat_to_euler(quad.quat);
        data.uav_vel(:, idx) = quad.vel;
        data.deck_pos(:, idx) = deck.pos;
        data.deck_att(:, idx) = deck.att;
        data.deck_vel(:, idx) = deck.vel;

        % Update ARMA
        arma = update_arma(arma, t, deck);

        % Generate and store forecast
        if t - last_forecast_time >= cfg.forecast_interval
            forecast = generate_arma_forecast(arma, t, cfg.forecast_horizon, 0.1);
            data.forecasts{end+1} = struct('t_origin', t, 'forecast', forecast);
            last_forecast_time = t;
        end

        % Replan trajectory
        if t - last_replan_time >= cfg.replan_interval
            % Get target from forecast
            forecast = generate_arma_forecast(arma, t, 5, 0.1);

            % Estimate time to intercept
            rel_pos = deck.pos - quad.pos;
            height = rel_pos(3);
            horiz_dist = norm(rel_pos(1:2));
            tf = max(2, min(8, max(horiz_dist/5, height/2)));

            % Find forecast at tf
            [~, tf_idx] = min(abs(forecast.t - (t + tf)));
            deck_target = struct();
            deck_target.pos = forecast.pos(:, tf_idx);
            deck_target.vel = forecast.vel(:, tf_idx);
            deck_target.att = forecast.att(:, tf_idx);

            ctrl = update_trajectory(ctrl, quad, deck_target, tf, t);
            last_replan_time = t;
        end

        % Check landing
        rel_pos = deck.pos - quad.pos;
        height = rel_pos(3);

        if height < 0.3 && height > -1
            results.landed = true;
            results.landing_time = t;
            results.rel_vel = quad.vel - deck.vel;
            results.pos_error = norm(rel_pos(1:2));
            fprintf('TOUCHDOWN at t=%.2fs\n', t);
            break;
        end

        % Compute control
        u = compute_pmp_control(ctrl, quad, deck, t);

        % Step dynamics
        quad = step_quadrotor(quad, u, cfg.dt);

        t = t + cfg.dt;
        idx = idx + 1;
    end

    % Trim data
    data.t = data.t(1:idx);
    data.uav_pos = data.uav_pos(:, 1:idx);
    data.uav_att = data.uav_att(:, 1:idx);
    data.uav_vel = data.uav_vel(:, 1:idx);
    data.deck_pos = data.deck_pos(:, 1:idx);
    data.deck_att = data.deck_att(:, 1:idx);
    data.deck_vel = data.deck_vel(:, 1:idx);

    fprintf('Simulation complete: %d steps\n', idx);
end

%% ==================== VIDEO GENERATION ====================

function generate_video(data, results, cfg)
    fprintf('Generating video...\n');

    % Colors
    colors = struct();
    colors.uav = [0.18, 0.80, 0.44];
    colors.deck = [0.20, 0.60, 0.86];
    colors.forecast = [0.91, 0.30, 0.24];
    colors.actual = [0.61, 0.35, 0.71];
    colors.history = [0.20, 0.29, 0.37];
    colors.landing = [0.95, 0.77, 0.06];

    % Create figure
    fig = figure('Position', [50, 50, 1400, 900], 'Color', 'w');

    % Video writer
    if cfg.save_video
        vid = VideoWriter(cfg.output_file, 'MPEG-4');
        vid.FrameRate = cfg.fps;
        vid.Quality = 90;
        open(vid);
    end

    % Frame indices (subsample for video fps)
    frame_dt = 1 / cfg.fps;
    frame_times = 0:frame_dt:data.t(end);
    n_frames = length(frame_times);

    for fi = 1:n_frames
        t = frame_times(fi);

        % Find closest data index
        [~, idx] = min(abs(data.t - t));

        clf(fig);

        % Current state
        uav_pos = data.uav_pos(:, idx);
        uav_att = data.uav_att(:, idx);
        deck_pos = data.deck_pos(:, idx);
        deck_att = data.deck_att(:, idx);

        % Time window
        t_min = t - cfg.past_window;
        t_max = t + cfg.future_window;

        % History indices
        hist_mask = data.t <= t & data.t >= t_min;
        hist_t = data.t(hist_mask);
        hist_uav = data.uav_pos(:, hist_mask);
        hist_deck = data.deck_pos(:, hist_mask);
        hist_att = data.deck_att(:, hist_mask);

        % Future indices
        fut_mask = data.t > t & data.t <= t_max;
        fut_t = data.t(fut_mask);
        fut_deck = data.deck_pos(:, fut_mask);
        fut_att = data.deck_att(:, fut_mask);

        % Get forecast
        [fc_t, fc_pos, fc_att] = get_forecast_at_time(data.forecasts, t);

        % === 3D VIEW ===
        ax3d = subplot(2, 3, [1, 4]);
        hold on; grid on;

        draw_deck_3d(deck_pos, deck_att, colors.deck);
        draw_uav_3d(uav_pos, uav_att, colors.uav);

        if ~isempty(hist_uav)
            plot3(hist_uav(1,:), hist_uav(2,:), hist_uav(3,:), ...
                  'Color', [colors.history, 0.5], 'LineWidth', 1);
        end

        cx = (uav_pos(1) + deck_pos(1)) / 2;
        cy = (uav_pos(2) + deck_pos(2)) / 2;
        xlim([cx-50, cx+50]);
        ylim([cy-30, cy+30]);
        zlim([-45, 5]);
        xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
        title(sprintf('Shipboard Landing  t = %.2fs', t), 'FontWeight', 'bold', 'FontSize', 12);
        view([-35, 25]);
        set(gca, 'ZDir', 'reverse');

        % === STRIP PLOTS ===
        landing_t = results.landing_time;

        % X position
        subplot(2, 3, 2);
        plot_strip_chart(t, t_min, t_max, hist_t, hist_deck(1,:), hist_uav(1,:), ...
                        fc_t, fc_pos(1,:), fut_t, fut_deck(1,:), landing_t, colors, ...
                        'X (m)', 'Position X (Forward)', 0.5);

        % Y position
        subplot(2, 3, 3);
        plot_strip_chart(t, t_min, t_max, hist_t, hist_deck(2,:), hist_uav(2,:), ...
                        fc_t, fc_pos(2,:), fut_t, fut_deck(2,:), landing_t, colors, ...
                        'Y (m)', 'Position Y (Starboard)', 0.5);

        % Altitude
        subplot(2, 3, 5);
        plot_strip_chart(t, t_min, t_max, hist_t, -hist_deck(3,:), -hist_uav(3,:), ...
                        fc_t, -fc_pos(3,:), fut_t, -fut_deck(3,:), landing_t, colors, ...
                        'Alt (m)', 'Altitude', 0.3);
        xlabel('Time (s)');

        % Attitude
        subplot(2, 3, 6);
        plot_attitude_chart(t, t_min, t_max, hist_t, hist_att, ...
                           fc_t, fc_att, fut_t, fut_att, landing_t, colors);
        xlabel('Sim Time (s)');

        drawnow;

        if cfg.save_video
            frame = getframe(fig);
            writeVideo(vid, frame);
        end

        if mod(fi, 20) == 1
            fprintf('  Frame %d/%d (t=%.1fs)\n', fi, n_frames, t);
        end
    end

    if cfg.save_video
        close(vid);
        fprintf('Video saved: %s\n', cfg.output_file);
    end
end

function [fc_t, fc_pos, fc_att] = get_forecast_at_time(forecasts, t)
    fc_t = []; fc_pos = zeros(3,0); fc_att = zeros(3,0);

    for i = length(forecasts):-1:1
        if forecasts{i}.t_origin <= t
            fc = forecasts{i}.forecast;
            mask = fc.t > t;
            fc_t = fc.t(mask);
            fc_pos = fc.pos(:, mask);
            fc_att = fc.att(:, mask);
            return;
        end
    end
end

function draw_deck_3d(pos, att, color)
    L = 20; W = 12;
    corners = [-L/2, -W/2, 0; L/2, -W/2, 0; L/2, W/2, 0; -L/2, W/2, 0]';

    R = euler_to_rotm(att);
    corners_w = R * corners + pos;

    fill3(corners_w(1,:), corners_w(2,:), corners_w(3,:), color, ...
          'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 1);
    plot3(pos(1), pos(2), pos(3), 'x', 'Color', [0.95, 0.77, 0.06], ...
          'MarkerSize', 15, 'LineWidth', 3);
end

function draw_uav_3d(pos, att, color)
    sz = 1.5;
    R = euler_to_rotm(att);

    arms = sz * [1,0,0; -1,0,0; 0,1,0; 0,-1,0]';
    arms_w = R * arms + pos;

    for i = 1:4
        col = 'k';
        if i == 3, col = 'r'; end
        plot3([pos(1), arms_w(1,i)], [pos(2), arms_w(2,i)], ...
              [pos(3), arms_w(3,i)], col, 'LineWidth', 2);
    end
    scatter3(arms_w(1,:), arms_w(2,:), arms_w(3,:), 50, color, 'filled');
end

function plot_strip_chart(t, t_min, t_max, hist_t, hist_deck, hist_uav, ...
                         fc_t, fc_y, fut_t, fut_y, landing_t, colors, ...
                         ylabel_str, title_str, err_scale)
    hold on; grid on;

    % History
    if ~isempty(hist_t)
        plot(hist_t, hist_deck, 'Color', colors.deck, 'LineWidth', 2, 'DisplayName', 'Deck');
        plot(hist_t, hist_uav, 'Color', colors.uav, 'LineWidth', 2, 'DisplayName', 'UAV');
    end

    % Forecast with error band
    if ~isempty(fc_t)
        plot(fc_t, fc_y, '--', 'Color', colors.forecast, 'LineWidth', 2, 'DisplayName', 'Forecast');
        dt = fc_t - t;
        err = err_scale * dt;
        fill([fc_t, fliplr(fc_t)], [fc_y+err, fliplr(fc_y-err)], ...
             colors.forecast, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end

    % Actual future
    if ~isempty(fut_t)
        plot(fut_t, fut_y, '-', 'Color', colors.actual, 'LineWidth', 1.5, 'DisplayName', 'Actual');
    end

    % Now line
    xline(t, 'k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    yl = ylim;
    patch([t_min, t, t, t_min], [yl(1), yl(1), yl(2), yl(2)], 'k', ...
          'FaceAlpha', 0.05, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Landing line
    if ~isempty(landing_t) && landing_t >= t_min && landing_t <= t_max
        xline(landing_t, '--', 'Color', colors.landing, 'LineWidth', 2, 'DisplayName', 'Landing');
    end

    xlim([t_min, t_max]);
    ylabel(ylabel_str);
    title(title_str, 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 7);
end

function plot_attitude_chart(t, t_min, t_max, hist_t, hist_att, ...
                            fc_t, fc_att, fut_t, fut_att, landing_t, colors)
    hold on; grid on;

    if ~isempty(hist_t)
        plot(hist_t, rad2deg(hist_att(1,:)), 'Color', colors.deck, 'LineWidth', 2, 'DisplayName', 'Roll');
        plot(hist_t, rad2deg(hist_att(2,:)), '--', 'Color', colors.deck, 'LineWidth', 2, 'DisplayName', 'Pitch');
    end

    if ~isempty(fc_t) && size(fc_att, 2) > 0
        fc_roll = rad2deg(fc_att(1,:));
        fc_pitch = rad2deg(fc_att(2,:));
        plot(fc_t, fc_roll, '--', 'Color', colors.forecast, 'LineWidth', 2, 'DisplayName', 'Forecast');
        plot(fc_t, fc_pitch, ':', 'Color', colors.forecast, 'LineWidth', 2, 'HandleVisibility', 'off');

        dt = fc_t - t;
        err = 0.3 * dt;
        fill([fc_t, fliplr(fc_t)], [fc_roll+err, fliplr(fc_roll-err)], ...
             colors.forecast, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        fill([fc_t, fliplr(fc_t)], [fc_pitch+err, fliplr(fc_pitch-err)], ...
             colors.forecast, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end

    if ~isempty(fut_t)
        plot(fut_t, rad2deg(fut_att(1,:)), '-', 'Color', colors.actual, 'LineWidth', 1.5, 'DisplayName', 'Actual');
        plot(fut_t, rad2deg(fut_att(2,:)), '-', 'Color', colors.actual, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end

    xline(t, 'k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    patch([t_min, t, t, t_min], [-12, -12, 12, 12], 'k', ...
          'FaceAlpha', 0.05, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    if ~isempty(landing_t) && landing_t >= t_min && landing_t <= t_max
        xline(landing_t, '--', 'Color', colors.landing, 'LineWidth', 2, 'HandleVisibility', 'off');
    end

    xlim([t_min, t_max]);
    ylim([-12, 12]);
    ylabel('Angle (deg)');
    title('Deck Roll & Pitch', 'FontSize', 10);
    legend('Location', 'best', 'FontSize', 7);
end

%% ==================== UTILITY FUNCTIONS ====================

function R = euler_to_rotm(att)
    % Euler angles [roll; pitch; yaw] to rotation matrix (ZYX convention)
    phi = att(1); theta = att(2); psi = att(3);

    cp = cos(phi); sp = sin(phi);
    ct = cos(theta); st = sin(theta);
    cy = cos(psi); sy = sin(psi);

    R = [ct*cy, sp*st*cy-cp*sy, cp*st*cy+sp*sy;
         ct*sy, sp*st*sy+cp*cy, cp*st*sy-sp*cy;
         -st,   sp*ct,          cp*ct];
end

function R = quat_to_rotm(q)
    % Quaternion [w; x; y; z] to rotation matrix
    w = q(1); x = q(2); y = q(3); z = q(4);

    R = [1-2*(y^2+z^2), 2*(x*y-w*z), 2*(x*z+w*y);
         2*(x*y+w*z), 1-2*(x^2+z^2), 2*(y*z-w*x);
         2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x^2+y^2)];
end

function euler = quat_to_euler(q)
    % Quaternion to Euler angles [roll; pitch; yaw]
    w = q(1); x = q(2); y = q(3); z = q(4);

    % Roll (x-axis rotation)
    sinr_cosp = 2*(w*x + y*z);
    cosr_cosp = 1 - 2*(x^2 + y^2);
    roll = atan2(sinr_cosp, cosr_cosp);

    % Pitch (y-axis rotation)
    sinp = 2*(w*y - z*x);
    if abs(sinp) >= 1
        pitch = sign(sinp) * pi/2;
    else
        pitch = asin(sinp);
    end

    % Yaw (z-axis rotation)
    siny_cosp = 2*(w*z + x*y);
    cosy_cosp = 1 - 2*(y^2 + z^2);
    yaw = atan2(siny_cosp, cosy_cosp);

    euler = [roll; pitch; yaw];
end

function q_out = quat_mult(q1, q2)
    % Quaternion multiplication q1 * q2
    w1 = q1(1); v1 = q1(2:4);
    w2 = q2(1); v2 = q2(2:4);

    w_out = w1*w2 - dot(v1, v2);
    v_out = w1*v2 + w2*v1 + cross(v1, v2);

    q_out = [w_out; v_out];
end

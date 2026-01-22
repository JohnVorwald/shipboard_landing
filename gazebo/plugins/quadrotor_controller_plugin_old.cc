/*
 * Quadrotor Controller Plugin with PMP Optimal Control for Gazebo 11
 * Implements trajectory tracking for shipboard landing using costate feedback
 *
 * References:
 * - Pontryagin Minimum Principle for optimal trajectory tracking
 * - Cascaded position -> acceleration -> attitude control
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Matrix3.hh>
#include <cmath>
#include <vector>
#include <fstream>

namespace gazebo
{
  // Minimum-snap trajectory segment
  struct TrajectorySegment
  {
    double t_start, t_end;
    std::vector<double> coeffs_x, coeffs_y, coeffs_z;  // 6 coefficients each (5th order)

    ignition::math::Vector3d Position(double t) const
    {
      double tau = (t - t_start) / (t_end - t_start);
      double tau2 = tau * tau;
      double tau3 = tau2 * tau;
      double tau4 = tau3 * tau;
      double tau5 = tau4 * tau;

      return ignition::math::Vector3d(
        coeffs_x[0] + coeffs_x[1]*tau + coeffs_x[2]*tau2 + coeffs_x[3]*tau3 + coeffs_x[4]*tau4 + coeffs_x[5]*tau5,
        coeffs_y[0] + coeffs_y[1]*tau + coeffs_y[2]*tau2 + coeffs_y[3]*tau3 + coeffs_y[4]*tau4 + coeffs_y[5]*tau5,
        coeffs_z[0] + coeffs_z[1]*tau + coeffs_z[2]*tau2 + coeffs_z[3]*tau3 + coeffs_z[4]*tau4 + coeffs_z[5]*tau5
      );
    }

    ignition::math::Vector3d Velocity(double t) const
    {
      double dt = t_end - t_start;
      double tau = (t - t_start) / dt;
      double tau2 = tau * tau;
      double tau3 = tau2 * tau;
      double tau4 = tau3 * tau;

      return ignition::math::Vector3d(
        (coeffs_x[1] + 2*coeffs_x[2]*tau + 3*coeffs_x[3]*tau2 + 4*coeffs_x[4]*tau3 + 5*coeffs_x[5]*tau4) / dt,
        (coeffs_y[1] + 2*coeffs_y[2]*tau + 3*coeffs_y[3]*tau2 + 4*coeffs_y[4]*tau3 + 5*coeffs_y[5]*tau4) / dt,
        (coeffs_z[1] + 2*coeffs_z[2]*tau + 3*coeffs_z[3]*tau2 + 4*coeffs_z[4]*tau3 + 5*coeffs_z[5]*tau4) / dt
      );
    }

    ignition::math::Vector3d Acceleration(double t) const
    {
      double dt = t_end - t_start;
      double dt2 = dt * dt;
      double tau = (t - t_start) / dt;
      double tau2 = tau * tau;
      double tau3 = tau2 * tau;

      return ignition::math::Vector3d(
        (2*coeffs_x[2] + 6*coeffs_x[3]*tau + 12*coeffs_x[4]*tau2 + 20*coeffs_x[5]*tau3) / dt2,
        (2*coeffs_y[2] + 6*coeffs_y[3]*tau + 12*coeffs_y[4]*tau2 + 20*coeffs_y[5]*tau3) / dt2,
        (2*coeffs_z[2] + 6*coeffs_z[3]*tau + 12*coeffs_z[4]*tau2 + 20*coeffs_z[5]*tau3) / dt2
      );
    }
  };

  // Costate estimator for PMP
  class CostateEstimator
  {
  public:
    void Update(const ignition::math::Vector3d& pos_err,
                const ignition::math::Vector3d& vel_err,
                double dt)
    {
      // Costate dynamics: λ_dot = -∂H/∂x
      // For quadratic cost, costates are proportional to state errors
      lambda_pos_ = lambda_pos_ + pos_err * dt * kp_costate_;
      lambda_vel_ = lambda_vel_ + vel_err * dt * kv_costate_;

      // Decay for stability
      lambda_pos_ = lambda_pos_ * 0.995;
      lambda_vel_ = lambda_vel_ * 0.995;
    }

    ignition::math::Vector3d GetCorrectionAccel() const
    {
      // Optimal control correction from Hamiltonian: u* = -R^{-1} B^T λ
      return lambda_pos_ * 0.1 + lambda_vel_ * 0.3;
    }

  private:
    ignition::math::Vector3d lambda_pos_{0, 0, 0};
    ignition::math::Vector3d lambda_vel_{0, 0, 0};
    double kp_costate_ = 2.0;
    double kv_costate_ = 1.0;
  };

  class QuadrotorControllerPlugin : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr model, sdf::ElementPtr sdf)
    {
      this->model_ = model;
      this->world_ = model->GetWorld();

      // Get parameters
      if (sdf->HasElement("mass"))
        this->mass_ = sdf->Get<double>("mass");

      if (sdf->HasElement("target_model"))
        this->target_model_name_ = sdf->Get<std::string>("target_model");

      if (sdf->HasElement("deck_offset"))
        this->deck_offset_ = sdf->Get<ignition::math::Vector3d>("deck_offset");

      if (sdf->HasElement("use_optimal_control"))
        this->use_optimal_control_ = sdf->Get<bool>("use_optimal_control");

      // Controller gains
      if (sdf->HasElement("kp_pos"))
        this->kp_pos_ = sdf->Get<double>("kp_pos");
      if (sdf->HasElement("kd_pos"))
        this->kd_pos_ = sdf->Get<double>("kd_pos");
      if (sdf->HasElement("kp_att"))
        this->kp_att_ = sdf->Get<double>("kp_att");
      if (sdf->HasElement("kd_att"))
        this->kd_att_ = sdf->Get<double>("kd_att");

      // Get base link
      this->base_link_ = model->GetLink("base_link");
      if (!this->base_link_)
        this->base_link_ = model->GetLinks()[0];

      // Set up Gazebo transport for trajectory commands
      this->node_ = transport::NodePtr(new transport::Node());
      this->node_->Init(this->world_->Name());

      this->traj_sub_ = this->node_->Subscribe(
        "~/" + model->GetName() + "/trajectory",
        &QuadrotorControllerPlugin::OnTrajectory, this);

      this->land_sub_ = this->node_->Subscribe(
        "~/" + model->GetName() + "/land",
        &QuadrotorControllerPlugin::OnLand, this);

      // Connect update
      this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
          std::bind(&QuadrotorControllerPlugin::OnUpdate, this));

      // Initialize trajectory to hover at start position
      ignition::math::Pose3d start_pose = model->WorldPose();
      hover_target_ = start_pose.Pos();

      gzmsg << "QuadrotorControllerPlugin loaded for " << model->GetName() << "\n";
      gzmsg << "  Optimal control: " << (use_optimal_control_ ? "enabled" : "disabled") << "\n";
      gzmsg << "  Target ship: " << target_model_name_ << "\n";
    }

    void OnUpdate()
    {
      double t = this->world_->SimTime().Double();
      double dt = this->world_->Physics()->GetMaxStepSize();

      // Get current state
      ignition::math::Pose3d pose = base_link_->WorldPose();
      ignition::math::Vector3d pos = pose.Pos();
      ignition::math::Vector3d vel = base_link_->WorldLinearVel();
      ignition::math::Vector3d omega = base_link_->WorldAngularVel();
      ignition::math::Quaterniond quat = pose.Rot();

      // Get target based on mode
      ignition::math::Vector3d target_pos, target_vel, target_acc;
      ignition::math::Quaterniond target_quat;

      if (landing_mode_ && !trajectory_.empty())
      {
        // Follow landing trajectory
        GetTrajectoryState(t, target_pos, target_vel, target_acc);
        target_quat = GetTargetAttitude(target_acc);

        // Check if landed
        if (t > trajectory_.back().t_end)
        {
          landing_mode_ = false;
          landed_ = true;
          gzmsg << "Landing complete at t=" << t << "\n";
        }
      }
      else if (landed_)
      {
        // Track deck after landing
        target_pos = GetDeckPosition();
        target_vel = GetDeckVelocity();
        target_acc = ignition::math::Vector3d(0, 0, 0);
        target_quat = GetDeckAttitude();
      }
      else
      {
        // Hover mode
        target_pos = hover_target_;
        target_vel = ignition::math::Vector3d(0, 0, 0);
        target_acc = ignition::math::Vector3d(0, 0, 0);
        target_quat = ignition::math::Quaterniond(0, 0, 0);
      }

      // Compute control
      ignition::math::Vector3d force, torque;

      if (use_optimal_control_)
      {
        ComputePMPControl(pos, vel, quat, omega,
                          target_pos, target_vel, target_acc, target_quat,
                          dt, force, torque);
      }
      else
      {
        ComputePDControl(pos, vel, quat, omega,
                         target_pos, target_vel, target_acc, target_quat,
                         force, torque);
      }

      // Apply force and torque
      base_link_->AddForce(force);
      base_link_->AddTorque(torque);

      // Log state periodically
      if (fmod(t, 0.1) < dt)
      {
        LogState(t, pos, vel, target_pos, target_vel);
      }
    }

  private:
    void ComputePMPControl(
      const ignition::math::Vector3d& pos,
      const ignition::math::Vector3d& vel,
      const ignition::math::Quaterniond& quat,
      const ignition::math::Vector3d& omega,
      const ignition::math::Vector3d& target_pos,
      const ignition::math::Vector3d& target_vel,
      const ignition::math::Vector3d& target_acc,
      const ignition::math::Quaterniond& target_quat,
      double dt,
      ignition::math::Vector3d& force,
      ignition::math::Vector3d& torque)
    {
      // Position and velocity errors
      ignition::math::Vector3d pos_err = pos - target_pos;
      ignition::math::Vector3d vel_err = vel - target_vel;

      // Update costate estimator
      costate_estimator_.Update(pos_err, vel_err, dt);

      // Desired acceleration: feedforward + feedback + costate correction
      ignition::math::Vector3d a_ff = target_acc;
      ignition::math::Vector3d a_fb = -pos_err * kp_pos_ - vel_err * kd_pos_;
      ignition::math::Vector3d a_costate = -costate_estimator_.GetCorrectionAccel();

      ignition::math::Vector3d a_des = a_ff + a_fb + a_costate;

      // Add gravity compensation
      a_des.Z() -= 9.81;

      // Total thrust magnitude
      double thrust = mass_ * a_des.Length();
      thrust = std::max(0.0, std::min(thrust, max_thrust_));

      // Desired body z-axis (thrust direction)
      ignition::math::Vector3d z_des = -a_des.Normalized();

      // Desired attitude from thrust direction
      ignition::math::Vector3d x_c(cos(target_quat.Yaw()), sin(target_quat.Yaw()), 0);
      ignition::math::Vector3d y_des = z_des.Cross(x_c).Normalized();
      ignition::math::Vector3d x_des = y_des.Cross(z_des);

      // Build rotation matrix and convert to quaternion
      ignition::math::Matrix3d R_des;
      R_des.SetCol(0, x_des);
      R_des.SetCol(1, y_des);
      R_des.SetCol(2, z_des);

      ignition::math::Quaterniond quat_des;
      quat_des = ignition::math::Quaterniond(R_des);

      // Attitude error (in body frame)
      ignition::math::Quaterniond quat_err = quat.Inverse() * quat_des;
      ignition::math::Vector3d att_err(
        2.0 * quat_err.X(),
        2.0 * quat_err.Y(),
        2.0 * quat_err.Z()
      );

      // Attitude control torque
      torque = att_err * kp_att_ * inertia_ - omega * kd_att_ * inertia_;

      // Thrust in world frame
      ignition::math::Vector3d body_z = quat.RotateVector(ignition::math::Vector3d(0, 0, 1));
      force = -body_z * thrust;
    }

    void ComputePDControl(
      const ignition::math::Vector3d& pos,
      const ignition::math::Vector3d& vel,
      const ignition::math::Quaterniond& quat,
      const ignition::math::Vector3d& omega,
      const ignition::math::Vector3d& target_pos,
      const ignition::math::Vector3d& target_vel,
      const ignition::math::Vector3d& target_acc,
      const ignition::math::Quaterniond& target_quat,
      ignition::math::Vector3d& force,
      ignition::math::Vector3d& torque)
    {
      // Simple PD position control
      ignition::math::Vector3d pos_err = target_pos - pos;
      ignition::math::Vector3d vel_err = target_vel - vel;

      ignition::math::Vector3d a_des = pos_err * kp_pos_ + vel_err * kd_pos_ + target_acc;
      a_des.Z() += 9.81;  // Gravity compensation

      double thrust = mass_ * a_des.Z();
      thrust = std::max(0.0, std::min(thrust, max_thrust_));

      // Simple attitude from desired horizontal acceleration
      double roll_des = (a_des.Y() / 9.81);
      double pitch_des = -(a_des.X() / 9.81);
      roll_des = std::max(-0.5, std::min(0.5, roll_des));
      pitch_des = std::max(-0.5, std::min(0.5, pitch_des));

      ignition::math::Quaterniond quat_des;
      quat_des.Euler(roll_des, pitch_des, target_quat.Yaw());

      // Attitude error
      ignition::math::Quaterniond quat_err = quat.Inverse() * quat_des;
      ignition::math::Vector3d att_err(
        2.0 * quat_err.X(),
        2.0 * quat_err.Y(),
        2.0 * quat_err.Z()
      );

      torque = att_err * kp_att_ * inertia_ - omega * kd_att_ * inertia_;

      ignition::math::Vector3d body_z = quat.RotateVector(ignition::math::Vector3d(0, 0, 1));
      force = body_z * thrust;
    }

    void GetTrajectoryState(double t,
                            ignition::math::Vector3d& pos,
                            ignition::math::Vector3d& vel,
                            ignition::math::Vector3d& acc)
    {
      // Find active segment
      for (const auto& seg : trajectory_)
      {
        if (t >= seg.t_start && t <= seg.t_end)
        {
          pos = seg.Position(t);
          vel = seg.Velocity(t);
          acc = seg.Acceleration(t);
          return;
        }
      }

      // Past trajectory end - hold final position
      if (!trajectory_.empty())
      {
        const auto& last = trajectory_.back();
        pos = last.Position(last.t_end);
        vel = ignition::math::Vector3d(0, 0, 0);
        acc = ignition::math::Vector3d(0, 0, 0);
      }
    }

    ignition::math::Quaterniond GetTargetAttitude(const ignition::math::Vector3d& acc)
    {
      // Get deck attitude for yaw reference
      ignition::math::Quaterniond deck_att = GetDeckAttitude();

      // Attitude from desired acceleration
      ignition::math::Vector3d thrust_dir = (acc + ignition::math::Vector3d(0, 0, 9.81)).Normalized();

      double roll = asin(-thrust_dir.Y());
      double pitch = atan2(thrust_dir.X(), thrust_dir.Z());

      ignition::math::Quaterniond quat;
      quat.Euler(roll, pitch, deck_att.Yaw());
      return quat;
    }

    ignition::math::Vector3d GetDeckPosition()
    {
      physics::ModelPtr ship = world_->ModelByName(target_model_name_);
      if (!ship)
        return ignition::math::Vector3d(15, 0, 2);

      ignition::math::Pose3d ship_pose = ship->WorldPose();
      ignition::math::Vector3d deck_world = ship_pose.Pos() +
        ship_pose.Rot().RotateVector(deck_offset_);
      return deck_world;
    }

    ignition::math::Vector3d GetDeckVelocity()
    {
      physics::ModelPtr ship = world_->ModelByName(target_model_name_);
      if (!ship)
        return ignition::math::Vector3d(0, 0, 0);

      return ship->WorldLinearVel();
    }

    ignition::math::Quaterniond GetDeckAttitude()
    {
      physics::ModelPtr ship = world_->ModelByName(target_model_name_);
      if (!ship)
        return ignition::math::Quaterniond(0, 0, 0);

      return ship->WorldPose().Rot();
    }

    void GenerateLandingTrajectory(double t_start, double duration)
    {
      trajectory_.clear();

      ignition::math::Vector3d start_pos = base_link_->WorldPose().Pos();
      ignition::math::Vector3d start_vel = base_link_->WorldLinearVel();
      ignition::math::Vector3d end_pos = GetDeckPosition();
      ignition::math::Vector3d end_vel = GetDeckVelocity();

      // Generate minimum-snap trajectory
      TrajectorySegment seg;
      seg.t_start = t_start;
      seg.t_end = t_start + duration;

      // Solve for 5th order polynomial coefficients
      // p(tau) = c0 + c1*tau + c2*tau^2 + c3*tau^3 + c4*tau^4 + c5*tau^5
      // Boundary conditions: p(0), p'(0), p''(0), p(1), p'(1), p''(1)

      auto solve_minsnap = [&](double p0, double v0, double a0,
                               double pf, double vf, double af,
                               double T) -> std::vector<double>
      {
        std::vector<double> c(6);
        c[0] = p0;
        c[1] = v0 * T;
        c[2] = 0.5 * a0 * T * T;
        c[3] = 10*(pf - p0) - 6*v0*T - 3*vf*T - 1.5*a0*T*T + 0.5*af*T*T;
        c[4] = -15*(pf - p0) + 8*v0*T + 7*vf*T + 1.5*a0*T*T - af*T*T;
        c[5] = 6*(pf - p0) - 3*v0*T - 3*vf*T - 0.5*a0*T*T + 0.5*af*T*T;
        return c;
      };

      seg.coeffs_x = solve_minsnap(start_pos.X(), start_vel.X(), 0,
                                    end_pos.X(), end_vel.X(), 0, duration);
      seg.coeffs_y = solve_minsnap(start_pos.Y(), start_vel.Y(), 0,
                                    end_pos.Y(), end_vel.Y(), 0, duration);
      seg.coeffs_z = solve_minsnap(start_pos.Z(), start_vel.Z(), 0,
                                    end_pos.Z(), end_vel.Z(), 0, duration);

      trajectory_.push_back(seg);

      gzmsg << "Generated landing trajectory: " << start_pos << " -> " << end_pos
            << " over " << duration << "s\n";
    }

    void OnTrajectory(ConstVector3dPtr& msg)
    {
      hover_target_.Set(msg->x(), msg->y(), msg->z());
      gzmsg << "New hover target: " << hover_target_ << "\n";
    }

    void OnLand(ConstIntPtr& msg)
    {
      double t = world_->SimTime().Double();
      double duration = msg->data() > 0 ? msg->data() : 10.0;

      GenerateLandingTrajectory(t, duration);
      landing_mode_ = true;
      landed_ = false;

      gzmsg << "Landing initiated at t=" << t << " with duration=" << duration << "s\n";
    }

    void LogState(double t,
                  const ignition::math::Vector3d& pos,
                  const ignition::math::Vector3d& vel,
                  const ignition::math::Vector3d& target_pos,
                  const ignition::math::Vector3d& target_vel)
    {
      // Could write to file or ROS topic
      // For now, just log to terminal occasionally
      static double last_log = 0;
      if (t - last_log > 1.0)
      {
        ignition::math::Vector3d err = pos - target_pos;
        gzdbg << "t=" << t << " pos_err=" << err.Length() << "m\n";
        last_log = t;
      }
    }

    // Member variables
    physics::ModelPtr model_;
    physics::WorldPtr world_;
    physics::LinkPtr base_link_;
    event::ConnectionPtr update_connection_;

    transport::NodePtr node_;
    transport::SubscriberPtr traj_sub_;
    transport::SubscriberPtr land_sub_;

    std::string target_model_name_ = "ship";
    ignition::math::Vector3d deck_offset_{15, 0, 2.2};
    ignition::math::Vector3d hover_target_{0, 0, 10};

    std::vector<TrajectorySegment> trajectory_;
    bool landing_mode_ = false;
    bool landed_ = false;
    bool use_optimal_control_ = true;

    CostateEstimator costate_estimator_;

    // Physical parameters
    double mass_ = 2.0;
    double inertia_ = 0.02;
    double max_thrust_ = 50.0;

    // Controller gains
    double kp_pos_ = 6.0;
    double kd_pos_ = 4.0;
    double kp_att_ = 200.0;
    double kd_att_ = 30.0;
  };

  GZ_REGISTER_MODEL_PLUGIN(QuadrotorControllerPlugin)
}

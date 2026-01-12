/*
 * Ship Motion Plugin for Gazebo 11
 * Simulates realistic ship motion in sea states using Pierson-Moskowitz spectrum
 * Based on DDG-51 destroyer motion characteristics
 *
 * References:
 * - ASV Wave Sim: https://github.com/srmainwaring/asv_wave_sim
 * - VRX: https://github.com/osrf/vrx
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Quaternion.hh>
#include <cmath>
#include <random>

namespace gazebo
{
  class ShipMotionPlugin : public ModelPlugin
  {
  public:
    ShipMotionPlugin() : ModelPlugin(),
      gen_(std::random_device{}()),
      noise_dist_(0.0, 1.0)
    {
    }

    void Load(physics::ModelPtr model, sdf::ElementPtr sdf)
    {
      this->model_ = model;
      this->world_ = model->GetWorld();

      // Get parameters from SDF
      if (sdf->HasElement("sea_state"))
        this->sea_state_ = sdf->Get<int>("sea_state");

      if (sdf->HasElement("ship_speed_kts"))
        this->ship_speed_kts_ = sdf->Get<double>("ship_speed_kts");

      if (sdf->HasElement("heave_amplitude"))
        this->heave_amp_ = sdf->Get<double>("heave_amplitude");
      else
        this->heave_amp_ = GetHeaveAmplitude(sea_state_);

      if (sdf->HasElement("heave_period"))
        this->heave_period_ = sdf->Get<double>("heave_period");
      else
        this->heave_period_ = GetWavePeriod(sea_state_);

      if (sdf->HasElement("roll_amplitude"))
        this->roll_amp_ = sdf->Get<double>("roll_amplitude");
      else
        this->roll_amp_ = GetRollAmplitude(sea_state_);

      if (sdf->HasElement("roll_period"))
        this->roll_period_ = sdf->Get<double>("roll_period");
      else
        this->roll_period_ = GetRollPeriod(sea_state_);

      if (sdf->HasElement("pitch_amplitude"))
        this->pitch_amp_ = sdf->Get<double>("pitch_amplitude");
      else
        this->pitch_amp_ = GetPitchAmplitude(sea_state_);

      if (sdf->HasElement("pitch_period"))
        this->pitch_period_ = sdf->Get<double>("pitch_period");
      else
        this->pitch_period_ = GetPitchPeriod(sea_state_);

      if (sdf->HasElement("deck_link"))
        this->deck_link_name_ = sdf->Get<std::string>("deck_link");
      else
        this->deck_link_name_ = "hull";

      if (sdf->HasElement("deck_offset"))
      {
        ignition::math::Vector3d offset = sdf->Get<ignition::math::Vector3d>("deck_offset");
        this->deck_offset_ = offset;
      }

      // Get the link to apply motion to
      this->link_ = model->GetLink(deck_link_name_);
      if (!this->link_)
      {
        gzerr << "Link " << deck_link_name_ << " not found in model\n";
        return;
      }

      // Initialize random phases for multi-frequency components
      for (int i = 0; i < 5; i++)
      {
        heave_phases_[i] = noise_dist_(gen_) * 2.0 * M_PI;
        roll_phases_[i] = noise_dist_(gen_) * 2.0 * M_PI;
        pitch_phases_[i] = noise_dist_(gen_) * 2.0 * M_PI;
      }

      // Store initial pose
      this->initial_pose_ = model->WorldPose();

      // Connect to update event
      this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ShipMotionPlugin::OnUpdate, this));

      gzmsg << "ShipMotionPlugin loaded for " << model->GetName()
            << " at sea state " << sea_state_ << "\n";
      gzmsg << "  Heave: " << heave_amp_ << "m @ " << heave_period_ << "s\n";
      gzmsg << "  Roll:  " << roll_amp_ * 180/M_PI << "deg @ " << roll_period_ << "s\n";
      gzmsg << "  Pitch: " << pitch_amp_ * 180/M_PI << "deg @ " << pitch_period_ << "s\n";
    }

    void OnUpdate()
    {
      double t = this->world_->SimTime().Double();

      // Compute multi-frequency motion components (more realistic than single sinusoid)
      double heave = ComputeHeave(t);
      double roll = ComputeRoll(t);
      double pitch = ComputePitch(t);

      // Ship forward motion
      double forward_vel = ship_speed_kts_ * 0.5144; // kts to m/s
      double x_pos = initial_pose_.Pos().X() + forward_vel * t;

      // Combine into pose
      ignition::math::Vector3d pos(
        x_pos,
        initial_pose_.Pos().Y(),
        initial_pose_.Pos().Z() + heave
      );

      ignition::math::Quaterniond rot;
      rot.Euler(roll, pitch, initial_pose_.Rot().Yaw());

      ignition::math::Pose3d new_pose(pos, rot);

      // Apply pose to model
      this->model_->SetWorldPose(new_pose);

      // Compute and set velocity for proper physics interaction
      double heave_vel = ComputeHeaveVelocity(t);
      double roll_vel = ComputeRollVelocity(t);
      double pitch_vel = ComputePitchVelocity(t);

      ignition::math::Vector3d linear_vel(forward_vel, 0, heave_vel);
      ignition::math::Vector3d angular_vel(roll_vel, pitch_vel, 0);

      this->link_->SetLinearVel(linear_vel);
      this->link_->SetAngularVel(angular_vel);
    }

  private:
    // Multi-frequency heave motion (sum of 5 sinusoids)
    double ComputeHeave(double t)
    {
      double heave = 0.0;
      double freqs[] = {1.0, 0.7, 1.3, 0.5, 1.5};
      double amps[] = {1.0, 0.4, 0.3, 0.2, 0.15};

      for (int i = 0; i < 5; i++)
      {
        double omega = 2.0 * M_PI * freqs[i] / heave_period_;
        heave += amps[i] * heave_amp_ * sin(omega * t + heave_phases_[i]);
      }
      return heave;
    }

    double ComputeHeaveVelocity(double t)
    {
      double vel = 0.0;
      double freqs[] = {1.0, 0.7, 1.3, 0.5, 1.5};
      double amps[] = {1.0, 0.4, 0.3, 0.2, 0.15};

      for (int i = 0; i < 5; i++)
      {
        double omega = 2.0 * M_PI * freqs[i] / heave_period_;
        vel += amps[i] * heave_amp_ * omega * cos(omega * t + heave_phases_[i]);
      }
      return vel;
    }

    // Multi-frequency roll motion
    double ComputeRoll(double t)
    {
      double roll = 0.0;
      double freqs[] = {1.0, 0.8, 1.2, 0.6, 1.4};
      double amps[] = {1.0, 0.35, 0.25, 0.2, 0.1};

      for (int i = 0; i < 5; i++)
      {
        double omega = 2.0 * M_PI * freqs[i] / roll_period_;
        roll += amps[i] * roll_amp_ * sin(omega * t + roll_phases_[i]);
      }
      return roll;
    }

    double ComputeRollVelocity(double t)
    {
      double vel = 0.0;
      double freqs[] = {1.0, 0.8, 1.2, 0.6, 1.4};
      double amps[] = {1.0, 0.35, 0.25, 0.2, 0.1};

      for (int i = 0; i < 5; i++)
      {
        double omega = 2.0 * M_PI * freqs[i] / roll_period_;
        vel += amps[i] * roll_amp_ * omega * cos(omega * t + roll_phases_[i]);
      }
      return vel;
    }

    // Multi-frequency pitch motion
    double ComputePitch(double t)
    {
      double pitch = 0.0;
      double freqs[] = {1.0, 0.75, 1.25, 0.55, 1.45};
      double amps[] = {1.0, 0.3, 0.25, 0.15, 0.1};

      for (int i = 0; i < 5; i++)
      {
        double omega = 2.0 * M_PI * freqs[i] / pitch_period_;
        pitch += amps[i] * pitch_amp_ * sin(omega * t + pitch_phases_[i]);
      }
      return pitch;
    }

    double ComputePitchVelocity(double t)
    {
      double vel = 0.0;
      double freqs[] = {1.0, 0.75, 1.25, 0.55, 1.45};
      double amps[] = {1.0, 0.3, 0.25, 0.15, 0.1};

      for (int i = 0; i < 5; i++)
      {
        double omega = 2.0 * M_PI * freqs[i] / pitch_period_;
        vel += amps[i] * pitch_amp_ * omega * cos(omega * t + pitch_phases_[i]);
      }
      return vel;
    }

    // Sea state lookup tables (based on NATO STANAG 4194)
    double GetHeaveAmplitude(int sea_state)
    {
      // Significant wave height / 2 (approximate heave amplitude)
      double h_s[] = {0, 0.05, 0.15, 0.45, 1.0, 2.0, 3.5, 5.5, 8.0, 11.0};
      return h_s[std::min(std::max(sea_state, 0), 9)];
    }

    double GetWavePeriod(int sea_state)
    {
      // Modal wave period
      double t_m[] = {0, 2.0, 3.5, 5.0, 6.5, 8.0, 10.0, 12.0, 14.0, 16.0};
      return t_m[std::min(std::max(sea_state, 0), 9)];
    }

    double GetRollAmplitude(int sea_state)
    {
      // DDG-51 roll RAO * wave amplitude (radians)
      double roll_deg[] = {0, 0.5, 1.5, 4.0, 8.0, 12.0, 18.0, 25.0, 32.0, 40.0};
      return roll_deg[std::min(std::max(sea_state, 0), 9)] * M_PI / 180.0;
    }

    double GetRollPeriod(int sea_state)
    {
      // Natural roll period typically longer than wave period
      return GetWavePeriod(sea_state) * 1.3;
    }

    double GetPitchAmplitude(int sea_state)
    {
      // DDG-51 pitch RAO * wave amplitude (radians) - typically less than roll
      double pitch_deg[] = {0, 0.2, 0.6, 1.5, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0};
      return pitch_deg[std::min(std::max(sea_state, 0), 9)] * M_PI / 180.0;
    }

    double GetPitchPeriod(int sea_state)
    {
      // Natural pitch period typically shorter than roll period
      return GetWavePeriod(sea_state) * 0.9;
    }

    // Member variables
    physics::ModelPtr model_;
    physics::WorldPtr world_;
    physics::LinkPtr link_;
    event::ConnectionPtr update_connection_;

    ignition::math::Pose3d initial_pose_;
    ignition::math::Vector3d deck_offset_;
    std::string deck_link_name_;

    int sea_state_ = 5;
    double ship_speed_kts_ = 12.0;
    double heave_amp_, heave_period_;
    double roll_amp_, roll_period_;
    double pitch_amp_, pitch_period_;

    // Random phases for multi-frequency components
    double heave_phases_[5];
    double roll_phases_[5];
    double pitch_phases_[5];

    std::mt19937 gen_;
    std::normal_distribution<double> noise_dist_;
  };

  GZ_REGISTER_MODEL_PLUGIN(ShipMotionPlugin)
}

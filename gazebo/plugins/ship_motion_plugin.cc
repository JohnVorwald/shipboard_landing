// Ship Motion Plugin for Gazebo Ionic (gz-sim9)
// Implements realistic ship motion (heave, roll, pitch) based on sea state

#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Link.hh>
#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <cmath>

using namespace gz;
using namespace sim;

class ShipMotionPlugin :
  public System,
  public ISystemConfigure,
  public ISystemPreUpdate
{
public:
  void Configure(const Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 EntityComponentManager &_ecm,
                 EventManager &) override
  {
    this->model = Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      gzerr << "ShipMotionPlugin can only be attached to a model entity." << std::endl;
      return;
    }

    gzmsg << "ShipMotionPlugin loading for model: " << this->model.Name(_ecm) << std::endl;

    // Read sea state parameters from SDF
    if (_sdf->HasElement("sea_state"))
      this->seaState = _sdf->Get<int>("sea_state");

    if (_sdf->HasElement("ship_speed_kts"))
      this->shipSpeedKts = _sdf->Get<double>("ship_speed_kts");

    // Heave parameters (vertical motion)
    if (_sdf->HasElement("heave_amplitude"))
      this->heaveAmplitude = _sdf->Get<double>("heave_amplitude");
    if (_sdf->HasElement("heave_period"))
      this->heavePeriod = _sdf->Get<double>("heave_period");

    // Roll parameters (rotation about X axis)
    if (_sdf->HasElement("roll_amplitude"))
      this->rollAmplitude = _sdf->Get<double>("roll_amplitude");
    if (_sdf->HasElement("roll_period"))
      this->rollPeriod = _sdf->Get<double>("roll_period");

    // Pitch parameters (rotation about Y axis)
    if (_sdf->HasElement("pitch_amplitude"))
      this->pitchAmplitude = _sdf->Get<double>("pitch_amplitude");
    if (_sdf->HasElement("pitch_period"))
      this->pitchPeriod = _sdf->Get<double>("pitch_period");

    // Deck offset for logging
    if (_sdf->HasElement("deck_offset"))
    {
      auto offset = _sdf->Get<gz::math::Vector3d>("deck_offset");
      this->deckOffset = offset;
    }

    // Store initial pose
    auto poseComp = _ecm.Component<components::Pose>(_entity);
    if (poseComp)
    {
      this->initialPose = poseComp->Data();
    }

    // Add phase offsets to make motion more realistic
    this->rollPhaseOffset = 0.3;   // Roll slightly out of phase with heave
    this->pitchPhaseOffset = 0.7;  // Pitch with different phase

    gzmsg << "ShipMotionPlugin configured:" << std::endl;
    gzmsg << "  Sea state: " << this->seaState << std::endl;
    gzmsg << "  Heave: " << this->heaveAmplitude << "m @ " << this->heavePeriod << "s" << std::endl;
    gzmsg << "  Roll: " << this->rollAmplitude << "rad @ " << this->rollPeriod << "s" << std::endl;
    gzmsg << "  Pitch: " << this->pitchAmplitude << "rad @ " << this->pitchPeriod << "s" << std::endl;

    this->configured = true;
  }

  void PreUpdate(const UpdateInfo &_info,
                 EntityComponentManager &_ecm) override
  {
    if (!this->configured || _info.paused)
      return;

    // Get simulation time in seconds
    double t = std::chrono::duration<double>(_info.simTime).count();

    // Calculate ship motion components
    // Heave: vertical displacement (z)
    double heave = this->heaveAmplitude *
                   std::sin(2.0 * M_PI * t / this->heavePeriod);

    // Roll: rotation about X axis
    double roll = this->rollAmplitude *
                  std::sin(2.0 * M_PI * t / this->rollPeriod + this->rollPhaseOffset);

    // Pitch: rotation about Y axis
    double pitch = this->pitchAmplitude *
                   std::sin(2.0 * M_PI * t / this->pitchPeriod + this->pitchPhaseOffset);

    // Add some irregularity based on sea state (higher sea state = more irregular)
    if (this->seaState > 3)
    {
      double irregularity = (this->seaState - 3) * 0.1;
      heave += irregularity * this->heaveAmplitude *
               std::sin(2.0 * M_PI * t / (this->heavePeriod * 0.7) + 1.2);
      roll += irregularity * this->rollAmplitude *
              std::sin(2.0 * M_PI * t / (this->rollPeriod * 0.6) + 0.8);
      pitch += irregularity * this->pitchAmplitude *
               std::sin(2.0 * M_PI * t / (this->pitchPeriod * 0.8) + 0.5);
    }

    // Create new pose with motion applied
    gz::math::Pose3d newPose(
      this->initialPose.Pos().X(),
      this->initialPose.Pos().Y(),
      this->initialPose.Pos().Z() + heave,
      roll,
      pitch,
      this->initialPose.Rot().Yaw()
    );

    // Update the model pose
    auto poseComp = _ecm.Component<components::Pose>(this->model.Entity());
    if (poseComp)
    {
      *poseComp = components::Pose(newPose);
    }
  }

private:
  Model model{kNullEntity};
  bool configured{false};

  // Initial pose
  gz::math::Pose3d initialPose;

  // Sea state parameters
  int seaState{5};
  double shipSpeedKts{12.0};

  // Motion parameters
  double heaveAmplitude{2.5};    // meters
  double heavePeriod{8.0};       // seconds
  double rollAmplitude{0.20};    // radians (~11 degrees)
  double rollPeriod{10.0};       // seconds
  double pitchAmplitude{0.08};   // radians (~4.5 degrees)
  double pitchPeriod{7.0};       // seconds

  // Phase offsets for more realistic motion
  double rollPhaseOffset{0.0};
  double pitchPhaseOffset{0.0};

  // Deck offset for reference
  gz::math::Vector3d deckOffset{15, 0, 2.2};
};

// Register the plugin with Gazebo
GZ_ADD_PLUGIN(ShipMotionPlugin,
              System,
              ShipMotionPlugin::ISystemConfigure,
              ShipMotionPlugin::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(ShipMotionPlugin, "ShipMotionPlugin")

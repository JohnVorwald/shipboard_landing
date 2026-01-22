#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/LinearVelocity.hh>

#include "gz/transport/Node.hh"

using namespace gz;
using namespace sim;

class ShipMotionPlugin :
  public System,
  public ISystemConfigure,
  public ISystemPostUpdate
{
  public: void Configure(const Entity &_entity,
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

    // In the new Gazebo, you'd typically read sea state from SDF here
    // and set up wave models, etc. For now, we'll just log.
  }

  public: void PostUpdate(const UpdateInfo &_info,
                         const EntityComponentManager &_ecm) override
  {
    // This function is called every simulation step.
    // If the simulation is paused, don't do anything.
    if (_info.paused)
      return;
    
    // Add your ship motion logic here, for example:
    // auto linVel = _ecm.Component<components::LinearVelocity>(this->model.Entity());
    // if(linVel)
    // {
    //   linVel->Data().X(some_value);
    // }
  }

  private: Model model{kNullEntity};
};

// Register the plugin
GZ_PLUGIN_REGISTER_SYSTEM(ShipMotionPlugin)

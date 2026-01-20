#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/plugin/Register.hh>
#include <gz/common/Console.hh>

using namespace gz;
using namespace sim;

class QuadrotorControllerPlugin :
  public System,
  public ISystemConfigure
{
  public: void Configure(const Entity &_entity,
                      const std::shared_ptr<const sdf::Element> &_sdf,
                      EntityComponentManager &_ecm,
                      EventManager &) override
  {
    auto model = Model(_entity);
    gzmsg << "Quadrotor Controller Plugin Loaded for: " << model.Name(_ecm) << std::endl;
  }
}; // <-- THIS WAS THE MISSING PIECE

GZ_PLUGIN_REGISTER_SYSTEM(QuadrotorControllerPlugin)

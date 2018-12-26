#ifndef NEOHOOK_PARAMS_H__
#define NEOHOOK_PARAMS_H__

#include <deal.II/base/parameter_handler.h>

namespace params
{
  dealii::ParameterHandler GlobPrm;

  void DeclareParams() {
    GlobPrm.declare_entry("fem polynomial order","1",
        dealii::Patterns::Integer(),"");
    GlobPrm.declare_entry("domain geometry hyper shape", "rectangle",
        dealii::Patterns::Selection("rectangle"));
  }
}

Params::Params(const dealii::ParameterHandler & prm)
    :
    poly_order(prm.get_integer("fem polynomial order")),
    domain_shape(prm.get("domain geometry shape")),
    solver_type(prm.get("solver type")) {}


#endif // NEOHOOK_PARAMS_H__

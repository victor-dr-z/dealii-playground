#ifndef NEOHOOK_PARAMS_H__
#define NEOHOOK_PARAMS_H__

#include <deal.II/base/parameter_handler.h>

struct Params {
  Params(const dealii::ParameterHandler & prm);

  const int poly_order;//!< polynomial order in FEA
  const std::string domain_shape;//!< name of the domain geometry shape
  const std::string solver_type;//!< to determine if we use iterative or direct solver
};

Params::Params(const dealii::ParameterHandler & prm)
    :
    poly_order(prm.get_integer("fem polynomial order")),
    domain_shape(prm.get("domain geometry shape")),
    solver_type(prm.get("solver type")) {}


#endif // NEOHOOK_PARAMS_H__

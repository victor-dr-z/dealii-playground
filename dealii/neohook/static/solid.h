#ifndef NEOHOOK_SOLID_H__
#define NEOHOOK_SOLID_H__

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

namespace solid
{
template <int dim>
inline dealii::Tensor<2,dim> GetF(const dealii::Tensor< 2, dim> & 	grad_u) {
  return dealii::Physics::Elasticity::Kinematics::F(grad_u);
}

template <int dim>
inline dealii::SymmetricTensor<2,dim> GetCGTensor (const Tensor<2,dim>& F) {
  return dealii::Physics::Elasticity::Kinematics::C(F);
}
}

#endif // NEOHOOK_SOLID_H__
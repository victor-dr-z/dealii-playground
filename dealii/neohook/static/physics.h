#ifndef NEOHOOK_PHYSICS_H__
#define NEOHOOK_PHYSICS_H__

#include <cmath>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/elasticity.h>

namespace physics
{
template <int dim>
void GetGradUs(const dealii::FEValuesBase<dim> & fvb,
    const dealii::Vector<double>& local_sol,
    std::vector<dealii::Tensor<2,dim>>& grad_us) {
  auto n_quad = fvb.n_quadrature_points;
  grad_us.resize(n_quad);
  std::vector<dealii::Tensor<1,dim>> nabla_grads(n_quad);
  fvb.get_function_gradients(local_sol, nabla_grads);
  for (int q=0; q<n_quad; ++q)
    grad_us[q] = GetGradUs(nabla_grads[q]);
}

template <int dim>
inline dealii::Tensor<2,dim> GetGradU(
    const dealii::Tensor<2,dim>& nabla_grad_u) {
  Tensor<2,dim> grad_u;
  for (int d=0; d<dim; ++d)
    for (int c=0; c<dim; ++c) {
      grad_u[d][c] = nabla_grad_u[c][d];
    }
  return grad_u;
}

template <int dim>
inline dealii::Tensor<2,dim> GetF(const dealii::Tensor<2, dim> & grad_u) {
  return dealii::Physics::Elasticity::Kinematics::F(grad_u);
}

template <int dim>
inline dealii::SymmetricTensor<2,dim> GetCGTensor (const Tensor<2,dim>& F) {
  return dealii::Physics::Elasticity::Kinematics::C(F);
}

template <int dim>
inline double GetJ(const Tensor<2, dim> & F) {
  return dealii::determinant(F);
}

template <int dim>
inline double GetTrace(const dealii::SymmetricTensor<2,dim>& CG){
  return dealii::trace(CG);
}

template <int dim>
inline dealii::Tensor<2, dim> GetGradTensor(
    const std::vector<dealii::Tensor<1,dim>>& grads) {
  // gradient values for component c
  dealii::
}

template <int dim>
inline double GetNeoHookEnergy(
    double mu,
    double lambda
    const dealii::Tensor<2, dim> & grad_u) {
  return 0.5 * mu * (GetTrace(GetCGTensor(GetF(grad_u))) - 3.0)
      - mu * std::log(GetJ(GetF(grad_u)))
      + lambda * std::log(GetJ(GetF(grad_u)));
}
}

#endif // NEOHOOK_SOLID_H__
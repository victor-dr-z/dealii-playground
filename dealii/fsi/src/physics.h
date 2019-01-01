#ifndef FSI_UTIL_H__
#define FSI_UTIL_H__

#include <cmath>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/elasticity.h>

namespace util
{
template <int dim, typename Number>
void GetGradUs(const dealii::FEValuesBase<dim> & fvb,
    const dealii::Vector<double>& local_sol,
    std::vector<dealii::Tensor<2,dim,Number>>& grad_us) {
  auto n_quad = fvb.n_quadrature_points;
  grad_us.resize(n_quad);
  std::vector<dealii::Tensor<1,dim>> nabla_grads(n_quad);
  fvb.get_function_gradients(local_sol, nabla_grads);
  for (int q=0; q<n_quad; ++q)
    grad_us[q] = GetGradU<dim,Number>(nabla_grads[q]);
}

template <int dim, typename Number>
inline dealii::Tensor<2,dim,Number> GetGradU(
    const dealii::Tensor<2,dim>& nabla_grad_u) {
  Tensor<2,dim,Number> grad_u;
  for (int d=0; d<dim; ++d)
    for (int c=0; c<dim; ++c)
      grad_u[d][c] = nabla_grad_u[c][d];
  return grad_u;
}

template <int dim, typename Number>
inline dealii::Tensor<2,dim,Number> GetF(const dealii::Tensor<2,dim,Number> & grad_u) {
  return dealii::Physics::Elasticity::Kinematics::F<dim,Number>(grad_u);
}

template <int dim, typename Number>
inline dealii::Tensor<2,dim,Number> GetFT(const dealii::Tensor<2,dim,Number>& f) {
  return dealii::transpose<dim,Number>(f);
}

template <int dim, typename Number>
inline dealii::Tensor<2,dim,Number> GetFInv(const dealii::Tensor<2,dim,Number>& f) {
  return dealii::invert<dim,Number>(f);
}

template <int dim, typename Number>
inline dealii::SymmetricTensor<2,dim,Number> GetCGTensor (const Tensor<2,dim,Number>& F) {
  return dealii::Physics::Elasticity::Kinematics::C<dim,Number>(F);
}

template <int dim, typename Number>
inline Number GetJ(const Tensor<2,dim,Number> & F) {
  return dealii::determinant<dim,Number>(F);
}

template <int dim, typename Number>
inline Number GetTrace(const dealii::SymmetricTensor<2,dim,Number>& CG){
  return dealii::trace<dim,Number>(CG);
}
}

#endif // FSI_SOLID_H__

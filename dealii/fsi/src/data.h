#ifndef FSI_DATA_H__
#define FSI_DATA_H__

#include "mesh.h"

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/trilinos_parallel_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/conditional_ostream.h>

template <int dim>
struct Data {
  Data(dealii::Triangulation<dim> & tria);

  dealii::TrilinosWrappers::MPI::SparseMatrix sys_mat;
  dealii::TrilinosWrappers::MPI::Vector sys_sol;
  dealii::TrilinosWrappers::MPI::Vector sys_rhs;

  dealii::ConditionalOStream pcout;
  const int poly_order;
  const dealii::FESystem<dim> fe;
  const QGauss<dim> q_rule;
  const QGauss<dim-1> fq_rule;
  const int nq;
  const int n_fq;
  const dealii::UpdateFlags flags;
  const dealii::UpdateFlags face_flags;
  dealii::FEValues<dim> fv;
  dealii::FEFaceValues<dim> fvf;

  Mesh<dim> mesh;
  dealii::DoFHandler<dim> dof_handler;
};

template <int dim>
Data<dim>::Data (dealii::Triangulation<dim> & tria)
    :
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
    poly_order(params::GetParam("fem polynomial order")),
    fe(FE_Q<dim>(poly_order), dim),
    q_rule(poly_order+1),
    fq_rule(poly_order+1),
    nq(q_rule.size()),
    n_fq(fq_rule.size()),
    flags(dealii::update_values | dealii::update_gradients |
        update_quadrature_points | update_JxW_values),
    face_flags(dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values |
        dealii::update_normal_vectors),
    fv(fe, q_rule, flags),
    fvf(fe, fq_rule, face_flags),
    mesh(),
    dof_handler(tria) {}

#endif // FSI_DATA_H__

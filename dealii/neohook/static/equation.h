#ifndef NEOHOOK_EQUATION_H__
#define NEOHOOK_EQUATION_H__

#include "data.h"

using fadd = Sacado::Fad::DFad<double>;
using fv = Data<dim>::fv;

template <int dim>
class Equation {
 public:
  Equation(std::shared_ptr<Data<dim>> & dat_p);

  void AssembleSystem();

  void AssembleCellMat(
      typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
      dealii::FullMatrix<double> &cell_mat);

  void AssembleRHS();

  void AssembleCellRHS(
      typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
      dealii::Vector<double> &cell_rhs);

  /*!
   Wrap linear algebraic solving process in.
   */
  void SolveEqu();
 private:
  const std::string solver_type_;
  std::shared_ptr<Data<dim>> & dat_;
};

template <int dim>
Equation<dim>::Equation (std::shared_ptr<Data<dim>> & dat_)
    :
    solver_type_(params::GetParam<std::string>("solver type")),
    dat_(dat_p) {}

template <int dim>
Equation<dim>::AssembleSystem() {
  for (typename dealii::DoFHandler<dim>::active_cell_iterator
      cell=dat_->dof_handler.begin_active(); cell!=dat_->dof_handler.end(); ++cell) {
    if (cell->is_locally_owned()) {
      fv.reinit(cell);
      fadd epsilon = 0;
      std::vector<fadd> residuals(dofs_per_cell);
      ComputeResidual(residuals);
    }
  }
}

template <dim>
void Equation<dim>::SolveEqu() {
  std::unordered_map<std::string, int> solvers =
      {{"direct", 0}, {"iterative",1}};
  switch (solvers[solver_type_]) {
    case 0: {
      dealii::SolverControl cn(1,0);
      dealii::TrilinosWrappers::SolverDirect::AdditionalData additional_data (
          true, "Amesos_Superludist");
      dealii::TrilinosWrappers::SolverDirect direct(cn, additional_data);
      direct.solve(dat_->sys_mat, dat_->sys_sol, dat_->sys_rhs);
    }
    default: {
      std::cerr << "Iterative scheme not implemented yet";
      return;
    }
  }
}

#endif // NEOHOOK_EQUATION_H__

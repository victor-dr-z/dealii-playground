#ifndef NEOHOOK_EQUATION_H__
#define NEOHOOK_EQUATION_H__

#include "data.h"

template <int dim>
class Equation {
 public:
  Equation(std::shared_ptr<Data<dim>> & dat_p_);

  void AssembleSystem();

  void AssembleCellMat(
      typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
      dealii::FullMatrix<double> &cell_mat);

  void AssembleRHS();

  void AssembleCellRHS(
      typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
      dealii::Vector<double> &cell_rhs);

  void SolveEqu();
 private:
};



#endif // NEOHOOK_EQUATION_H__

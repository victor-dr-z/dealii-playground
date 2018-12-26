#ifndef NEOHOOK_DATA_H__
#define NEOHOOK_DATA_H__

template <int dim>
struct Data {
  Data(const dealii::)
  
  dealii::TrilinosWrappers::MPI::SparseMatrix sys_mat;
  dealii::TrilinosWrappers::MPI::Vector sys_sol;
  dealii::TrilinosWrappers::MPI::Vector sys_rhs;

  dealii::FESystem<dim> fe;
};

#endif // NEOHOOK_DATA_H__

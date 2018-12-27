#ifndef NEOHOOK_DATA_H__
#define NEOHOOK_DATA_H__

template <int dim>
class Driver {
 public:
  void Run();

 private:
  void SetupSystem();
  void MakeGrid();
  void NewtonIteration();
  void OutputResult();

  dealii::parallel::distrbited_triangulation<dim> tria_;
  std::shared_ptr<Data<dim>> dat_p_;
};

template <int dim>
void Driver<dim>::MakeGrid() {
  dat_p_->mesh.MakeGrid(tria_);
}

template <int dim>
void Driver<dim>::SetupSystem() {
  dat_p_->dof_handler.distribute_dofs (dat_p_->fe);
  local_owned_dofs_ = dat_p_->dof_handler.locally_owned_dofs ();
  dealii::DoFTools::extract_locally_relevant_dofs (dat_p_->dof_handler,
      local_relevant_dofs_);

  // init data structure for hanging nodes constraint
  dat_p_->constraints_.clear ();
  constraints_.reinit (local_relevant_dofs_);
  dealii::DoFTools::make_hanging_node_constraints (dat_p_->dof_handler,
      constraints_);
  constraints_.close ();

  dealii::DynamicSparsityPattern dsp (local_relevant_dofs_);

  //dealii::DoFTools::make_flux_sparsity_pattern (dat_p_->dof_handler, dsp, constraints_, false);
  dealii::DoFTools::make_sparsity_pattern (dat_p_->dof_handler,
      dsp, constraints_, false);

  // setting up dsp with telling communicator and relevant dofs
  dealii::SparsityTools::distribute_sparsity_pattern (dsp,
      dat_p_->dof_handler.n_locally_owned_dofs_per_processor (),
      MPI_COMM_WORLD, local_relevant_dofs_);

  // init distributed matrix and vectors
  dat_p_->sys_mat.reinit (local_owned_dofs_, local_owned_dofs_, dsp, MPI_COMM_WORLD);
  dat_p_->sys_sol.reinit (local_owned_dofs_, MPI_COMM_WORLD);
  dat_p_->sys_rhs.reinit (local_owned_dofs_, MPI_COMM_WORLD);
  }
}

template <int dim>
void Driver<dim>::NewtonIteration() {
  iter_.DoIteration(equ_p_);
}

template <int dim>
void Driver<dim>::Run() {
  MakeGrid();
  SetupSystem();
  NewtonIteration();
  OutputResult();
}

#endif // NEOHOOK_DATA_H__

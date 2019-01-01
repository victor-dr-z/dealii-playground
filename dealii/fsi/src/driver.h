#ifndef FSI_DRIVER_H__
#define FSI_DRIVER_H__

template <int dim>
class Driver {
 public:
  Driver();

  void Run();

 private:
  void SetupSystem();
  void MakeGrid();
  void Solve();
  void OutputResult();

  dealii::parallel::distributed::Triangulation<dim> tria;
  std::shared_ptr<Data<dim>> dat_;

  dealii::IndexSet local_dofs_;
  dealii::IndexSet relev_dofs_;
};

template <int dim>
Driver<dim>::Driver()
    :
    tria(MPI_COMM_WORLD),
    dat_(std::make_shared<Data<dim>>(tria)) {}

template <int dim>
Driver<dim>::Run() {
  // produce a grid
  MakeGrid ();
  // setup matrices and vector
  SetupSystem ();
  // Perform iterations
  Solve();
  // Output results
  OutputResult();
}

template <int dim>
Driver<dim>::MakeGrid() {
  dat_->mesh.MakeGrid(tria);
}

template <int dim>
Driver<dim>::SetupSystem() {
  dat_->pcout << "Set up system" << std::endl;
  dat_->dof_handler.distribute_dofs(dat_->fe);
  local_dofs_ = dat_->dof_handler.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(dat_->dof_handler,
      relev_dofs_);
  dat_->constraints.clear();
  dat_->constraints.reinit(relev_dofs_);
  dealii::DoFTools::make_hanging_node_constraints(dat_->dof_handler,
      dat_->constraints);
  dat_->constraints.close();
  dealii::DynamicSparsity
}

#endif // FSI_DRIVER_H__
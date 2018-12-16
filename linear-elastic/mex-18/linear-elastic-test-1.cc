// @sect3{Include files}

// The first few files have already been covered in previous examples and will
// thus not be further commented on.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

// From the following include file we will import the declaration of
// H1-conforming finite element shape functions. This family of finite
// elements is called <code>FE_Q</code>, and was used in all examples before
// already to define the usual bi- or tri-linear elements, but we will now use
// it for bi-quadratic elements:
#include <deal.II/fe/fe_q.h>
// We will not read the grid from a file as in the previous example, but
// generate it using a function of the library. However, we will want to write
// out the locally refined grids (just the grid, not the solution) in each
// step, so we need the following include file instead of
// <code>grid_in.h</code>:
#include <deal.II/grid/grid_out.h>


// When using locally refined grids, we will get so-called <code>hanging
// nodes</code>. However, the standard finite element methods assumes that the
// discrete solution spaces be continuous, so we need to make sure that the
// degrees of freedom on hanging nodes conform to some constraints such that
// the global solution is continuous. We are also going to store the boundary
// conditions in this object. The following file contains a class which is
// used to handle these constraints:
#include <deal.II/lac/constraint_matrix.h>

// In order to refine our grids locally, we need a function from the library
// that decides which cells to flag for refinement or coarsening based on the
// error indicators we have computed. This function is defined here:
#include <deal.II/grid/grid_refinement.h>

// Finally, we need a simple way to actually compute the refinement indicators
// based on some error estimate. While in general, adaptivity is very
// problem-specific, the error indicator in the following file often yields
// quite nicely adapted grids for a wide class of problems.
#include <deal.II/numerics/error_estimator.h>

// Finally, this is as in previous programs:
using namespace dealii;


// @sect3{The <code>LinearElastic</code> class template}

// The main class is again almost unchanged. Two additions, however, are made:
// we have added the <code>refine_grid</code> function, which is used to
// adaptively refine the grid (instead of the global refinement in the
// previous examples), and a variable which will hold the constraints. In
// addition, we have added a destructor to the class for reasons that will
// become clear when we discuss its implementation.
template <int dim>
class LinearElastic
{
public:
  LinearElastic ();
  ~LinearElastic ();

  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;


  // This is the new variable in the main class. We need an object which holds
  // a list of constraints to hold the hanging nodes and the boundary
  // conditions.
  ConstraintMatrix     constraints;

  // The sparsity pattern and sparse matrix are deliberately declared in the
  // opposite of the order used in step-2 through step-5 to demonstrate the
  // primary use of the Subscriptor and SmartPointer classes.
  SparseMatrix<double> system_matrix;
  SparsityPattern      sparsity_pattern;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};


// @sect3{Nonconstant coefficients}

// The implementation of nonconstant coefficients is copied verbatim from
// step-5:
template <int dim>
double coefficient (const Point<dim> &p)
{
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}



// @sect3{The <code>LinearElastic</code> class implementation}

// @sect4{LinearElastic::LinearElastic}

template <int dim>
LinearElastic<dim>::LinearElastic ()
  :
  fe (2),
  dof_handler (triangulation)
{}


// @sect4{LinearElastic::~LinearElastic}

template <int dim>
LinearElastic<dim>::~LinearElastic ()
{
  system_matrix.clear();
}


// @sect4{LinearElastic::setup_system}

template <int dim>
void LinearElastic<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  // We may now populate the ConstraintMatrix with the hanging node
  // constraints. Since we will call this function in a loop we first clear
  // the current set of constraints from the last system and then compute new ones:
  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);


 VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            Functions::ZeroFunction<dim>(),
                                            constraints);

  constraints.close ();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  // We may now, finally, initialize the sparse matrix:
  system_matrix.reinit (sparsity_pattern);
}


// @sect4{LinearElastic::assemble_system}
template <int dim>
void LinearElastic<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(3);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          const double current_coefficient = coefficient<dim>
                                             (fe_values.quadrature_point (q_index));
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (current_coefficient *
                                     fe_values.shape_grad(i,q_index) *
                                     fe_values.shape_grad(j,q_index) *
                                     fe_values.JxW(q_index));

              cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                              1.0 *
                              fe_values.JxW(q_index));
            }
        }

      // Finally, transfer the contributions from @p cell_matrix and
      // @p cell_rhs into the global objects.
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
}


// @sect4{LinearElastic::solve}
template <int dim>
void LinearElastic<dim>::solve ()
{
  SolverControl      solver_control (1000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

  constraints.distribute (solution);
}


// @sect4{LinearElastic::refine_grid}
template <int dim>
void LinearElastic<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);

  triangulation.execute_coarsening_and_refinement ();
}


// @sect4{LinearElastic::output_results}
template <int dim>
void LinearElastic<dim>::output_results (const unsigned int cycle) const
{
  {
    GridOut grid_out;
    std::ofstream output ("grid-" + std::to_string(cycle) + ".eps");
    grid_out.write_eps (triangulation, output);
  }

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();

    std::ofstream output ("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk (output);
  }
}


// @sect4{LinearElastic::run}
template <int dim>
void LinearElastic<dim>::run ()
{
  for (unsigned int cycle=0; cycle<8; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_ball (triangulation);
          triangulation.refine_global (1);
        }
      else
        refine_grid ();


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      setup_system ();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      assemble_system ();
      solve ();
      output_results (cycle);
    }
}


// @sect3{The <code>main</code> function}
int main ()
{

  // The general idea behind the layout of this function is as follows: let's
  // try to run the program as we did before...
  try
    {
      LinearElastic<2> laplace_problem_2d;
      laplace_problem_2d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  // If the exception that was thrown somewhere was not an object of a class
  // derived from the standard <code>exception</code> class, then we can't do
  // anything at all. We then simply print an error message and exit.
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  // If we got to this point, there was no exception which propagated up to
  // the main function (there may have been exceptions, but they were caught
  // somewhere in the program or the library). Therefore, the program
  // performed as was expected and we can return without error.
  return 0;
}

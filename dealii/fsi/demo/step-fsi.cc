/* Author: Thomas Wick, 2012 */
/* University of Heidelberg  */
/* Date: May 3, 2011, */
/*       Revised Jan 30, 2012 */
/* E-mail: thomas.wick@iwr.uni-heidelberg.de */
/*
/*                                                                */
/*    Copyright (C) 2011, 2012 by Thomas Wick                     */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the webpage XXX for the  text  and     */
/*    further information on this license.                        */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>


// C++
#include <fstream>
#include <sstream>

// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:
using namespace dealii;



namespace ALETransformations
{
  template <int dim>
    inline
    Tensor<2,dim>
    get_pI (unsigned int q,
            std::vector<Vector<double> > old_solution_values)
    {
      Tensor<2,dim> tmp;
      tmp[0][0] =  old_solution_values[q](dim+dim);
      tmp[1][1] =  old_solution_values[q](dim+dim);

      return tmp;
    }

  template <int dim>
    inline
    Tensor<2,dim>
    get_pI_LinP (const double phi_i_p)
    {
      Tensor<2,dim> tmp;
      tmp.clear();
      tmp[0][0] = phi_i_p;
      tmp[1][1] = phi_i_p;

      return tmp;
   }

 template <int dim>
   inline
   Tensor<1,dim>
   get_grad_p (unsigned int q,
               std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
   {
     Tensor<1,dim> grad_p;
     grad_p[0] =  old_solution_grads[q][dim+dim][0];
     grad_p[1] =  old_solution_grads[q][dim+dim][1];

     return grad_p;
   }

 template <int dim>
  inline
  Tensor<1,dim>
  get_grad_p_LinP (const Tensor<1,dim> phi_i_grad_p)
    {
      Tensor<1,dim> grad_p;
      grad_p[0] =  phi_i_grad_p[0];
      grad_p[1] =  phi_i_grad_p[1];

      return grad_p;
   }

 template <int dim>
   inline
   Tensor<2,dim>
   get_grad_u (unsigned int q,
               std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
   {
      Tensor<2,dim> structure_continuation;
      structure_continuation[0][0] = old_solution_grads[q][dim][0];
      structure_continuation[0][1] = old_solution_grads[q][dim][1];
      structure_continuation[1][0] = old_solution_grads[q][dim+1][0];
      structure_continuation[1][1] = old_solution_grads[q][dim+1][1];

      return structure_continuation;
   }

  template <int dim>
  inline
  Tensor<2,dim>
  get_grad_v (unsigned int q,
              std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
      Tensor<2,dim> grad_v;
      grad_v[0][0] =  old_solution_grads[q][0][0];
      grad_v[0][1] =  old_solution_grads[q][0][1];
      grad_v[1][0] =  old_solution_grads[q][1][0];
      grad_v[1][1] =  old_solution_grads[q][1][1];

      return grad_v;
   }

  template <int dim>
    inline
    Tensor<2,dim>
    get_grad_v_T (const Tensor<2,dim> tensor_grad_v)
    {
      Tensor<2,dim> grad_v_T;
      grad_v_T = transpose (tensor_grad_v);

      return grad_v_T;
    }

  template <int dim>
    inline
    Tensor<2,dim>
    get_grad_v_LinV (const Tensor<2,dim> phi_i_grads_v)
    {
        Tensor<2,dim> tmp;
        tmp[0][0] = phi_i_grads_v[0][0];
        tmp[0][1] = phi_i_grads_v[0][1];
        tmp[1][0] = phi_i_grads_v[1][0];
        tmp[1][1] = phi_i_grads_v[1][1];

        return tmp;
    }

  template <int dim>
    inline
    Tensor<2,dim>
    get_Identity ()
    {
      Tensor<2,dim> identity;
      identity[0][0] = 1.0;
      identity[0][1] = 0.0;
      identity[1][0] = 0.0;
      identity[1][1] = 1.0;

      return identity;
   }

 template <int dim>
 inline
 Tensor<2,dim>
 get_F (unsigned int q,
        std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {
      Tensor<2,dim> F;
      F[0][0] = 1.0 +  old_solution_grads[q][dim][0];
      F[0][1] = old_solution_grads[q][dim][1];
      F[1][0] = old_solution_grads[q][dim+1][0];
      F[1][1] = 1.0 + old_solution_grads[q][dim+1][1];
      return F;
   }

 template <int dim>
 inline
 Tensor<2,dim>
 get_F_T (const Tensor<2,dim> F)
    {
      return  transpose (F);
    }

 template <int dim>
 inline
 Tensor<2,dim>
 get_F_Inverse (const Tensor<2,dim> F)
    {
      return invert (F);
    }

 template <int dim>
 inline
 Tensor<2,dim>
 get_F_Inverse_T (const Tensor<2,dim> F_Inverse)
   {
     return transpose (F_Inverse);
   }

 template <int dim>
   inline
   double
   get_J (const Tensor<2,dim> tensor_F)
   {
     return determinant (tensor_F);
   }


 template <int dim>
 inline
 Tensor<1,dim>
 get_v (unsigned int q,
        std::vector<Vector<double> > old_solution_values)
    {
      Tensor<1,dim> v;
      v[0] = old_solution_values[q](0);
      v[1] = old_solution_values[q](1);

      return v;
   }

 template <int dim>
   inline
   Tensor<1,dim>
   get_v_LinV (const Tensor<1,dim> phi_i_v)
   {
     Tensor<1,dim> tmp;
     tmp[0] = phi_i_v[0];
     tmp[1] = phi_i_v[1];

     return tmp;
   }

 template <int dim>
 inline
 Tensor<1,dim>
 get_u (unsigned int q,
        std::vector<Vector<double> > old_solution_values)
   {
     Tensor<1,dim> u;
     u[0] = old_solution_values[q](dim);
     u[1] = old_solution_values[q](dim+1);

     return u;
   }

 template <int dim>
   inline
   Tensor<1,dim>
   get_u_LinU (const Tensor<1,dim> phi_i_u)
   {
     Tensor<1,dim> tmp;
     tmp[0] = phi_i_u[0];
     tmp[1] = phi_i_u[1];

     return tmp;
   }

 template <int dim>
 inline
 Tensor<1,dim>
 get_w (unsigned int q,
        std::vector<Vector<double> > old_solution_values)
   {
     Tensor<1,dim> w;
     w[0] = old_solution_values[q](dim+dim+1);
     w[1] = old_solution_values[q](dim+dim+2);

     return w;
   }

 template <int dim>
   inline
   Tensor<2,dim>
   get_grad_w (unsigned int q,
               std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
   {
      Tensor<2,dim>
      tmp;
      tmp[0][0] = old_solution_grads[q][dim+dim+1][0];
      tmp[0][1] = old_solution_grads[q][dim+dim+1][1];
      tmp[1][0] = old_solution_grads[q][dim+dim+2][0];
      tmp[1][1] = old_solution_grads[q][dim+dim+2][1];

      return tmp;
   }



 template <int dim>
 inline
 double
 get_J_LinU (unsigned int q,
             const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
             const Tensor<2,dim> phi_i_grads_u)
{
  return (phi_i_grads_u[0][0] * (1 + old_solution_grads[q][dim+1][1]) +
                   (1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[1][1] -
                   phi_i_grads_u[0][1] * old_solution_grads[q][dim+1][0] -
                   old_solution_grads[q][dim][1] * phi_i_grads_u[1][0]);
}

  template <int dim>
  inline
  double
  get_J_Inverse_LinU (const double J,
                      const double J_LinU)
    {
      return (-1.0/std::pow(J,2) * J_LinU);
    }

template <int dim>
 inline
 Tensor<2,dim>
  get_F_LinU (const Tensor<2,dim> phi_i_grads_u)
  {
    Tensor<2,dim> tmp;
    tmp[0][0] = phi_i_grads_u[0][0];
    tmp[0][1] = phi_i_grads_u[0][1];
    tmp[1][0] = phi_i_grads_u[1][0];
    tmp[1][1] = phi_i_grads_u[1][1];

    return tmp;
  }

template <int dim>
 inline
 Tensor<2,dim>
  get_F_Inverse_LinU (const Tensor<2,dim> phi_i_grads_u,
                       const double J,
                       const double J_LinU,
                       unsigned int q,
                       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads
                       )
  {
    Tensor<2,dim> F_tilde;
    F_tilde[0][0] = 1.0 + old_solution_grads[q][dim+1][1];
    F_tilde[0][1] = -old_solution_grads[q][dim][1];
    F_tilde[1][0] = -old_solution_grads[q][dim+1][0];
    F_tilde[1][1] = 1.0 + old_solution_grads[q][dim][0];

    Tensor<2,dim> F_tilde_LinU;
    F_tilde_LinU[0][0] = phi_i_grads_u[1][1];
    F_tilde_LinU[0][1] = -phi_i_grads_u[0][1];
    F_tilde_LinU[1][0] = -phi_i_grads_u[1][0];
    F_tilde_LinU[1][1] = phi_i_grads_u[0][0];

    return (-1.0/(J*J) * J_LinU * F_tilde +
            1.0/J * F_tilde_LinU);

  }

 template <int dim>
   inline
   Tensor<2,dim>
   get_J_F_Inverse_T_LinU (const Tensor<2,dim> phi_i_grads_u)
   {
     Tensor<2,dim> tmp;
     tmp[0][0] = phi_i_grads_u[1][1];
     tmp[0][1] = -phi_i_grads_u[1][0];
     tmp[1][0] = -phi_i_grads_u[0][1];
     tmp[1][1] = phi_i_grads_u[0][0];

     return  tmp;
   }


 template <int dim>
 inline
 double
 get_tr_C_LinU (unsigned int q,
                 const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
                 const Tensor<2,dim> phi_i_grads_u)
{
  return ((1 + old_solution_grads[q][dim][0]) *
          phi_i_grads_u[0][0] +
          old_solution_grads[q][dim][1] *
          phi_i_grads_u[0][1] +
          (1 + old_solution_grads[q][dim+1][1]) *
          phi_i_grads_u[1][1] +
          old_solution_grads[q][dim+1][0] *
          phi_i_grads_u[1][0]);
}


}

namespace NSEALE
{
  template <int dim>
 inline
 Tensor<2,dim>
 get_stress_fluid_ALE (const double density,
                       const double viscosity,
                       const Tensor<2,dim>  pI,
                       const Tensor<2,dim>  grad_v,
                       const Tensor<2,dim>  grad_v_T,
                       const Tensor<2,dim>  F_Inverse,
                       const Tensor<2,dim>  F_Inverse_T)
  {
    return (-pI + density * viscosity *
           (grad_v * F_Inverse + F_Inverse_T * grad_v_T ));
  }

  template <int dim>
  inline
  Tensor<2,dim>
  get_stress_fluid_except_pressure_ALE (const double density,
                                        const double viscosity,
                                        const Tensor<2,dim>  grad_v,
                                        const Tensor<2,dim>  grad_v_T,
                                        const Tensor<2,dim>  F_Inverse,
                                        const Tensor<2,dim>  F_Inverse_T)
  {
    return (density * viscosity * (grad_v * F_Inverse + F_Inverse_T * grad_v_T));
  }

  template <int dim>
  inline
  Tensor<2,dim>
  get_stress_fluid_ALE_1st_term_LinAll (const Tensor<2,dim>  pI,
                                        const Tensor<2,dim>  F_Inverse_T,
                                        const Tensor<2,dim>  J_F_Inverse_T_LinU,
                                        const Tensor<2,dim>  pI_LinP,
                                        const double J)
  {
    return (-J * pI_LinP * F_Inverse_T - pI * J_F_Inverse_T_LinU);
  }

  template <int dim>
  inline
  Tensor<2,dim>
  get_stress_fluid_ALE_2nd_term_LinAll_short (const Tensor<2,dim> J_F_Inverse_T_LinU,
                                              const Tensor<2,dim> stress_fluid_ALE,
                                              const Tensor<2,dim> grad_v,
                                              const Tensor<2,dim> grad_v_LinV,
                                              const Tensor<2,dim> F_Inverse,
                                              const Tensor<2,dim> F_Inverse_LinU,
                                              const double J,
                                              const double viscosity,
                                              const double density
                                              )
{
    Tensor<2,dim> sigma_LinV;
    Tensor<2,dim> sigma_LinU;

    sigma_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    sigma_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);

    return (density * viscosity *
            (sigma_LinV + sigma_LinU) * J * transpose(F_Inverse) +
            stress_fluid_ALE * J_F_Inverse_T_LinU);
  }

template <int dim>
inline
Tensor<2,dim>
get_stress_fluid_ALE_3rd_term_LinAll_short (const Tensor<2,dim> F_Inverse,
                                            const Tensor<2,dim> F_Inverse_LinU,
                                            const Tensor<2,dim> grad_v,
                                            const Tensor<2,dim> grad_v_LinV,
                                            const double viscosity,
                                            const double density,
                                            const double J,
                                            const Tensor<2,dim> J_F_Inverse_T_LinU)
{
  return density * viscosity *
    (J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
     J * transpose(F_Inverse) * transpose(grad_v_LinV) * transpose(F_Inverse) +
     J * transpose(F_Inverse) * transpose(grad_v) * transpose(F_Inverse_LinU));
}



template <int dim>
inline
double
get_Incompressibility_ALE (unsigned int q,
                           std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
{
  return (old_solution_grads[q][0][0] +
          old_solution_grads[q][dim+1][1] * old_solution_grads[q][0][0] -
          old_solution_grads[q][dim][1] * old_solution_grads[q][1][0] -
          old_solution_grads[q][dim+1][0] * old_solution_grads[q][0][1] +
          old_solution_grads[q][1][1] +
          old_solution_grads[q][dim][0] * old_solution_grads[q][1][1]);

}

template <int dim>
inline
double
get_Incompressibility_ALE_LinAll (const Tensor<2,dim> phi_i_grads_v,
                                  const Tensor<2,dim> phi_i_grads_u,
                                  unsigned int q,
                                  const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
{
  return (phi_i_grads_v[0][0] + phi_i_grads_v[1][1] +
          phi_i_grads_u[1][1] * old_solution_grads[q][0][0] -
          phi_i_grads_u[0][1] * old_solution_grads[q][1][0] -
          phi_i_grads_u[1][0] * old_solution_grads[q][0][1] +
          phi_i_grads_u[0][0] * old_solution_grads[q][1][1]);
}


  template <int dim>
  inline
  Tensor<1,dim>
  get_Convection_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
                               const Tensor<1,dim> phi_i_v,
                               const double J,
                               const double J_LinU,
                               const Tensor<2,dim> F_Inverse,
                               const Tensor<2,dim> F_Inverse_LinU,
                               const Tensor<1,dim> v,
                               const Tensor<2,dim> grad_v,
                               const double density
                               )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)v = rho J grad(v)F^{-1}v

    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * v +
                       J * grad_v * F_Inverse_LinU * v);

    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * (phi_i_grads_v * F_Inverse * v +
                            grad_v * F_Inverse * phi_i_v));

    return density * (convection_LinU + convection_LinV);
  }


  template <int dim>
  inline
  Tensor<1,dim>
  get_Convection_u_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
                                 const Tensor<1,dim> phi_i_u,
                                 const double J,
                                 const double J_LinU,
                                 const Tensor<2,dim>  F_Inverse,
                                 const Tensor<2,dim>  F_Inverse_LinU,
                                 const Tensor<1,dim>  u,
                                 const Tensor<2,dim>  grad_v,
                                 const double density
                                 )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u

    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * u +
                       J * grad_v * F_Inverse_LinU * u +
                       J * grad_v * F_Inverse * phi_i_u);

    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * u);

    return density * (convection_LinU + convection_LinV);
}



  template <int dim>
  inline
  Tensor<1,dim>
  get_Convection_u_old_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
                                     const double J,
                                     const double J_LinU,
                                     const Tensor<2,dim>  F_Inverse,
                                     const Tensor<2,dim>  F_Inverse_LinU,
                                     const Tensor<1,dim>  old_timestep_solution_displacement,
                                     const Tensor<2,dim>  grad_v,
                                     const double density
                                     )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u

    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * old_timestep_solution_displacement +
                       J * grad_v * F_Inverse_LinU * old_timestep_solution_displacement);

    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * old_timestep_solution_displacement);


    return density * (convection_LinU  + convection_LinV);
  }

template <int dim>
inline
Tensor<1,dim>
get_accelaration_term_LinAll (const Tensor<1,dim> phi_i_v,
                              const Tensor<1,dim> v,
                              const Tensor<1,dim> old_timestep_v,
                              const double J_LinU,
                              const double J,
                              const double old_timestep_J,
                              const double density)
{
  return density/2.0 * (J_LinU * (v - old_timestep_v) + (J + old_timestep_J) * phi_i_v);

}


}


namespace StructureTermsALE
{
  // Green-Lagrange strain tensor
  template <int dim>
  inline
  Tensor<2,dim>
  get_E (const Tensor<2,dim> F_T,
         const Tensor<2,dim> F,
         const Tensor<2,dim> Identity)
  {
    return 0.5 * (F_T * F - Identity);
  }

  template <int dim>
  inline
  double
  get_tr_E (const Tensor<2,dim> E)
  {
    return trace (E);
  }

  template <int dim>
  inline
  double
  get_tr_E_LinU (unsigned int q,
                 const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
                 const Tensor<2,dim> phi_i_grads_u)
  {
    return ((1 + old_solution_grads[q][dim][0]) *
            phi_i_grads_u[0][0] +
            old_solution_grads[q][dim][1] *
            phi_i_grads_u[0][1] +
            (1 + old_solution_grads[q][dim+1][1]) *
            phi_i_grads_u[1][1] +
            old_solution_grads[q][dim+1][0] *
            phi_i_grads_u[1][0]);
  }

}



template <int dim>
class BoundaryParabolic : public Function<dim>
{
  public:
  BoundaryParabolic (const double time)
    : Function<dim>(dim+dim+dim+1)
    {
      _time = time;
    }

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;

private:
  double _time;

};

// The boundary values are given to component
// with number 0.
template <int dim>
double
BoundaryParabolic<dim>::value (const Point<dim>  &p,
                             const unsigned int component) const
{
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

  const long double pi = 3.141592653589793238462643;

  // The maximum inflow depends on the configuration
  // for the different test cases:
  // FSI 1: 0.2; FSI 2: 1.0; FSI 3: 2.0
  //
  // For the two unsteady test cases FSI 2 and FSI 3, it
  // is recommanded to start with a smooth increase of
  // the inflow. Hence, we use the cosine function
  // to control the inflow at the beginning until
  // the total time 2.0 has been reached.
  double inflow_velocity = 1.0;

  if (component == 0)
    {
      if (_time < 2.0)
        {
          return   ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * inflow_velocity *
                     (1.0 - std::cos(pi/2.0 * _time))/2.0 *
                     (4.0/0.1681) *
                     (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
        }
      else
        {
          return ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * inflow_velocity *
                   (4.0/0.1681) *
                   (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );

        }

    }

  return 0;
}



template <int dim>
void
BoundaryParabolic<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = BoundaryParabolic<dim>::value (p, c);
}


template <int dim>
class FSIALEProblem
{
public:

  FSIALEProblem (const unsigned int degree);
  ~FSIALEProblem ();
  void run ();

private:

  void set_runtime_parameters ();
  void setup_system ();
  void assemble_system_matrix ();
  void assemble_system_rhs ();

  void set_initial_bc (const double time);
  void set_newton_bc ();

  void solve ();
  void newton_iteration(const double time);
  void output_results (const unsigned int refinement_cycle,
                       const BlockVector<double> solution) const;

  double compute_point_value (Point<dim> p,
                              const unsigned int component) const;

  void compute_drag_lift_fsi_fluid_tensor ();
  void compute_functional_values ();

  const unsigned int   degree;

  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  ConstraintMatrix     constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution, newton_update, old_timestep_solution;
  BlockVector<double> system_rhs;

  TimerOutput         timer;

  // Global variables for timestepping scheme
  unsigned int timestep_number;
  unsigned int max_no_timesteps;
  double timestep, theta, time;
  std::string time_stepping_scheme;

  // Fluid parameters
  double density_fluid, viscosity;

  // Structure parameters
  double density_structure;
  double lame_coefficient_mu, lame_coefficient_lambda, poisson_ratio_nu;

  // Other parameters to control the fluid mesh motion
  double cell_diameter;
  double alpha_u, alpha_w;





};


template <int dim>
FSIALEProblem<dim>::FSIALEProblem (const unsigned int degree)
                :
                degree (degree),
                triangulation (Triangulation<dim>::maximum_smoothing),
                fe (FE_Q<dim>(degree+1), dim,
                    FE_Q<dim>(degree+1), dim,
                    FE_DGP<dim>(degree), 1,
                    FE_Q<dim>(degree+1), dim),
                dof_handler (triangulation),
                timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)
{}


// This is the standard destructor.
template <int dim>
FSIALEProblem<dim>::~FSIALEProblem ()
{}


template <int dim>
void FSIALEProblem<dim>::set_runtime_parameters ()
{
   // Fluid parameters
  density_fluid = 1.0e+3;

  // FSI 1 & 3: 1.0e+3; FSI 2: 1.0e+4
  density_structure = 1.0e+3;
  viscosity = 1.0e-3;

  // Structure parameters
  // FSI 1 & 2: 0.5e+6; FSI 3: 2.0e+6
  lame_coefficient_mu = 2.0e+6;
  poisson_ratio_nu = 0.4;

  lame_coefficient_lambda =  (2 * poisson_ratio_nu * lame_coefficient_mu)/
    (1.0 - 2 * poisson_ratio_nu);

  // Diffusion parameters to control the fluid mesh motion
  // The higher these parameters the stiffer the fluid mesh.
  alpha_u = 1.0e-5;
  alpha_w = 1.0e-5;

  // Timestepping schemes
  //BE, CN, CN_shifted
  time_stepping_scheme = "BE";

  // Timestep size:
  // FSI 1: 1.0 (quasi-stationary)
  // FSI 2: <= 1.0e-2 (non-stationary)
  // FSI 3: <= 1.0e-3 (non-stationary)
  timestep = 1.0e-03;

  // Maximum number of timesteps:
  // FSI 1: 25 , T= 25   (timestep == 1.0)
  // FSI 2: 1500, T= 15  (timestep == 1.0e-2)
  // FSI 3: 10000, T= 10 (timestep == 1.0e-3)
  max_no_timesteps = 15000;

  // A variable to count the number of time steps
  timestep_number = 0;

  // Counts total time
  time = 0;

  // Here, we choose a time-stepping scheme that
  // is based on finite differences:
  // BE         = backward Euler scheme
  // CN         = Crank-Nicolson scheme
  // CN_shifted = time-shifted Crank-Nicolson scheme
  // For further properties of these schemes,
  // we refer to standard literature.
  if (time_stepping_scheme == "BE")
    theta = 1.0;
  else if (time_stepping_scheme == "CN")
    theta = 0.5;
  else if (time_stepping_scheme == "CN_shifted")
    theta = 0.5 + timestep;
  else
    std::cout << "No such timestepping scheme" << std::endl;

  // In the following, we read a *.inp grid from a file.
  // The geometry information is based on the
  // fluid-structure interaction benchmark problems
  // (Lit. J. Hron, S. Turek, 2006)
  std::string grid_name;
  grid_name  = "fsi.inp";

  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());
  Assert (dim==2, ExcInternalError());
  grid_in.read_ucd (input_file);

  Point<dim> p(0.2, 0.2);
  double radius = 0.05;
  static const SphericalManifold<dim> boundary(p);
  triangulation.set_manifold (80, boundary);
  triangulation.set_manifold (81, boundary);

  triangulation.refine_global (1);

}



// This function is similar to many deal.II tuturial steps.
template <int dim>
void FSIALEProblem<dim>::setup_system ()
{
  timer.enter_section("Setup system.");

  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters ();

  system_matrix.clear ();

  dof_handler.distribute_dofs (fe);
  DoFRenumbering::Cuthill_McKee (dof_handler);

  // We are dealing with 7 components for this
  // two-dimensional fluid-structure interacion problem
  // Precisely, we use:
  // velocity in x and y:                0
  // structure displacement in x and y:  1
  // scalar pressure field:              2
  // additional displacement in x and y: 3
  std::vector<unsigned int> block_component (7,0);
  block_component[dim] = 1;
  block_component[dim+1] = 1;
  block_component[dim+dim] = 2;
  block_component[dim+dim+1] = 3;
  block_component[dim+dim+dim] = 3;

  DoFRenumbering::component_wise (dof_handler, block_component);

  {
    constraints.clear ();
    set_newton_bc ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
  }
  constraints.close ();

  std::vector<unsigned int> dofs_per_block (4);
  DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
  const unsigned int n_v = dofs_per_block[0],
    n_u = dofs_per_block[1],
    n_p =  dofs_per_block[2],
    n_w =  dofs_per_block[3];

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << '+' << n_u << '+' << n_p << '+' << n_w <<  ')'
            << std::endl;




 {
    BlockDynamicSparsityPattern csp (4,4);

    csp.block(0,0).reinit (n_v, n_v);
    csp.block(0,1).reinit (n_v, n_u);
    csp.block(0,2).reinit (n_v, n_p);
    csp.block(0,3).reinit (n_v, n_w);

    csp.block(1,0).reinit (n_u, n_v);
    csp.block(1,1).reinit (n_u, n_u);
    csp.block(1,2).reinit (n_u, n_p);
    csp.block(1,3).reinit (n_u, n_w);

    csp.block(2,0).reinit (n_p, n_v);
    csp.block(2,1).reinit (n_p, n_u);
    csp.block(2,2).reinit (n_p, n_p);
    csp.block(2,3).reinit (n_p, n_w);

    csp.block(3,0).reinit (n_w, n_v);
    csp.block(3,1).reinit (n_w, n_u);
    csp.block(3,2).reinit (n_w, n_p);
    csp.block(3,3).reinit (n_w, n_w);

    csp.collect_sizes();


    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);

    sparsity_pattern.copy_from (csp);
  }

 system_matrix.reinit (sparsity_pattern);

  // Actual solution at time step n
  solution.reinit (4);
  solution.block(0).reinit (n_v);
  solution.block(1).reinit (n_u);
  solution.block(2).reinit (n_p);
  solution.block(3).reinit (n_w);

  solution.collect_sizes ();

  // Old timestep solution at time step n-1
  old_timestep_solution.reinit (4);
  old_timestep_solution.block(0).reinit (n_v);
  old_timestep_solution.block(1).reinit (n_u);
  old_timestep_solution.block(2).reinit (n_p);
  old_timestep_solution.block(3).reinit (n_w);

  old_timestep_solution.collect_sizes ();


  // Updates for Newton's method
  newton_update.reinit (4);
  newton_update.block(0).reinit (n_v);
  newton_update.block(1).reinit (n_u);
  newton_update.block(2).reinit (n_p);
  newton_update.block(3).reinit (n_w);

  newton_update.collect_sizes ();

  // Residual for  Newton's method
  system_rhs.reinit (4);
  system_rhs.block(0).reinit (n_v);
  system_rhs.block(1).reinit (n_u);
  system_rhs.block(2).reinit (n_p);
  system_rhs.block(3).reinit (n_w);

  system_rhs.collect_sizes ();

  timer.exit_section();
}


template <int dim>
void FSIALEProblem<dim>::assemble_system_matrix ()
{
  timer.enter_section("Assemble Matrix.");
  system_matrix=0;

  QGauss<dim>   quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_gradients |
                                    update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;

  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);


  // Now, we are going to use the
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); // 2
  const FEValuesExtractors::Scalar pressure (dim+dim); // 4
  const FEValuesExtractors::Vector displacements_w (dim+dim+1); // 4


  // We declare Vectors and Tensors for
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points,
                                                    Vector<double>(dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points,
                                                                std::vector<Tensor<1,dim> > (dim+dim+dim+1));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points,
                                                          Vector<double>(dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points,
                                                                     std::vector<Tensor<1,dim> > (dim+dim+dim+1));


  // We declare Vectors and Tensors for
  // the solution at the previous time step:
   std::vector<Vector<double> > old_timestep_solution_values (n_q_points,
                                                    Vector<double>(dim+dim+dim+1));


  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points,
                                          std::vector<Tensor<1,dim> > (dim+dim+dim+1));


  std::vector<Vector<double> >   old_timestep_solution_face_values (n_face_q_points,
                                                                    Vector<double>(dim+dim+dim+1));


  std::vector<std::vector<Tensor<1,dim> > >  old_timestep_solution_face_grads (n_face_q_points,
                                                                               std::vector<Tensor<1,dim> > (dim+dim+dim+1));

  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_v (dofs_per_cell);
  std::vector<Tensor<2,dim> > phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell);
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell);
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);
  std::vector<Tensor<1,dim> > phi_i_w (dofs_per_cell);
  std::vector<Tensor<2,dim> > phi_i_grads_w(dofs_per_cell);

  // This is the identity matrix in two dimensions:
  const Tensor<2,dim> Identity = ALETransformations
    ::get_Identity<dim> ();

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      local_matrix = 0;

      // We need the cell diameter to control the fluid mesh motion
      cell_diameter = cell->diameter();

      // Old Newton iteration values
      fe_values.get_function_values (solution, old_solution_values);
      fe_values.get_function_gradients (solution, old_solution_grads);

      // Old_timestep_solution values
      fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);

      // Next, we run over all cells for the fluid equations
      if (cell->material_id() == 0)
        {
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                  phi_i_v[k]       = fe_values[velocities].value (k, q);
                  phi_i_grads_v[k] = fe_values[velocities].gradient (k, q);
                  phi_i_p[k]       = fe_values[pressure].value (k, q);
                  phi_i_u[k]       = fe_values[displacements].value (k, q);
                  phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
                  phi_i_w[k]       = fe_values[displacements_w].value (k, q);
                  phi_i_grads_w[k] = fe_values[displacements_w].gradient (k, q);
                }

              // We build values, vectors, and tensors
              // from information of the previous Newton step. These are introduced
              // for two reasons:
              // First, these are used to perform the ALE mapping of the
              // fluid equations. Second, these terms are used to
              // make the notation as simple and self-explaining as possible:
              const Tensor<2,dim> pI = ALETransformations
                ::get_pI<dim> (q, old_solution_values);

              const Tensor<1,dim> v = ALETransformations
                ::get_v<dim> (q, old_solution_values);

              const Tensor<1,dim> u = ALETransformations
                ::get_u<dim> (q,old_solution_values);

              const Tensor<2,dim> grad_v = ALETransformations
                ::get_grad_v<dim> (q, old_solution_grads);

              const Tensor<2,dim> grad_v_T = ALETransformations
                ::get_grad_v_T<dim> (grad_v);

              const Tensor<2,dim> F = ALETransformations
                ::get_F<dim> (q, old_solution_grads);

              const Tensor<2,dim> F_Inverse = ALETransformations
                ::get_F_Inverse<dim> (F);

              const Tensor<2,dim> F_Inverse_T = ALETransformations
                ::get_F_Inverse_T<dim> (F_Inverse);

              const double J = ALETransformations
                ::get_J<dim> (F);

              // Stress tensor for the fluid in ALE notation
              const Tensor<2,dim> sigma_ALE = NSEALE
                ::get_stress_fluid_ALE<dim> (density_fluid, viscosity, pI,
                                             grad_v, grad_v_T, F_Inverse, F_Inverse_T );

              // Further, we also need some information from the previous time steps
              const Tensor<1,dim> old_timestep_v = ALETransformations
                ::get_v<dim> (q, old_timestep_solution_values);

              const Tensor<1,dim> old_timestep_u = ALETransformations
                ::get_u<dim> (q, old_timestep_solution_values);

              const Tensor<2,dim> old_timestep_F = ALETransformations
                ::get_F<dim> (q, old_timestep_solution_grads);

              const double old_timestep_J = ALETransformations
                ::get_J<dim> (old_timestep_F);

              // Outer loop for dofs
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  const Tensor<2,dim> pI_LinP = ALETransformations
                    ::get_pI_LinP<dim> (phi_i_p[i]);

                  const Tensor<2,dim> grad_v_LinV = ALETransformations
                    ::get_grad_v_LinV<dim> (phi_i_grads_v[i]);

                  const double J_LinU =  ALETransformations
                    ::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);

                  const Tensor<2,dim> J_F_Inverse_T_LinU = ALETransformations
                    ::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);

                  const Tensor<2,dim> F_Inverse_LinU = ALETransformations
		    ::get_F_Inverse_LinU (phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);

                  const Tensor<2,dim>  stress_fluid_ALE_1st_term_LinAll = NSEALE
                    ::get_stress_fluid_ALE_1st_term_LinAll<dim>
                    (pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                  const Tensor<2,dim> stress_fluid_ALE_2nd_term_LinAll = NSEALE
		    ::get_stress_fluid_ALE_2nd_term_LinAll_short
                    (J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                     F_Inverse, F_Inverse_LinU, J, viscosity, density_fluid);

                  const Tensor<1,dim> convection_fluid_LinAll_short = NSEALE
                    ::get_Convection_LinAll_short<dim>
                    (phi_i_grads_v[i], phi_i_v[i], J,J_LinU,
                     F_Inverse, F_Inverse_LinU, v, grad_v, density_fluid);

                  const double incompressibility_ALE_LinAll = NSEALE
                    ::get_Incompressibility_ALE_LinAll<dim>
                    (phi_i_grads_v[i], phi_i_grads_u[i], q, old_solution_grads);

                  const Tensor<1,dim> accelaration_term_LinAll = NSEALE
		    ::get_accelaration_term_LinAll
                    (phi_i_v[i], v, old_timestep_v, J_LinU,
                     J, old_timestep_J, density_fluid);

                  const Tensor<1,dim> convection_fluid_u_LinAll_short =  NSEALE
                    ::get_Convection_u_LinAll_short<dim>
                    (phi_i_grads_v[i], phi_i_u[i], J,J_LinU, F_Inverse,
                     F_Inverse_LinU, u, grad_v, density_fluid);

                  const Tensor<1,dim> convection_fluid_u_old_LinAll_short = NSEALE
                    ::get_Convection_u_old_LinAll_short<dim>
                    (phi_i_grads_v[i], J, J_LinU, F_Inverse,
                     F_Inverse_LinU, old_timestep_u, grad_v, density_fluid);

                  // Inner loop for dofs
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                      // Fluid , NSE in ALE
                      const unsigned int comp_j = fe.system_to_component_index(j).first;
                      if (comp_j == 0 || comp_j == 1)
                        {
                          local_matrix(j,i) += (accelaration_term_LinAll * phi_i_v[j] +
                                                timestep * theta *
                                                convection_fluid_LinAll_short * phi_i_v[j] -
                                                convection_fluid_u_LinAll_short * phi_i_v[j] +
                                                convection_fluid_u_old_LinAll_short * phi_i_v[j] +
                                                timestep * scalar_product(stress_fluid_ALE_1st_term_LinAll, phi_i_grads_v[j]) +
                                                timestep * theta *
                                                scalar_product(stress_fluid_ALE_2nd_term_LinAll, phi_i_grads_v[j])
                                                ) * fe_values.JxW(q);
                        }
                      else if (comp_j == 2 || comp_j == 3)
                        {
                          local_matrix(j,i) += (alpha_u * scalar_product(phi_i_grads_w[i], phi_i_grads_u[j])
                                                ) * fe_values.JxW(q);
                        }
                      else if (comp_j == 4)
                        {
                          local_matrix(j,i) += (incompressibility_ALE_LinAll *  phi_i_p[j]
                                                ) * fe_values.JxW(q);
                        }
                      else if (comp_j == 5 || comp_j == 6)
                        {
                          local_matrix(j,i) += (alpha_w * (phi_i_w[i] * phi_i_w[j] - scalar_product(phi_i_grads_u[i],phi_i_grads_w[j]))
                                                ) * fe_values.JxW(q);
                        }

                      // end j dofs
                    }
                  // end i dofs
                }
              // end n_q_points
            }

          // We compute in the following
          // one term on the outflow boundary.
          // This relation is well-know in the literature
          // as "do-nothing" condition. Therefore, we only
          // ask for the corresponding color at the outflow
          // boundary that is 1 in our case.
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->face(face)->at_boundary() &&
                  (cell->face(face)->boundary_id() == 1)
                  )
                {

                  fe_face_values.reinit (cell, face);

                  fe_face_values.get_function_values (solution, old_solution_face_values);
                  fe_face_values.get_function_gradients (solution, old_solution_face_grads);

                  for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                      for (unsigned int k=0; k<dofs_per_cell; ++k)
                        {
                          phi_i_v[k]       = fe_face_values[velocities].value (k, q);
                          phi_i_grads_v[k] = fe_face_values[velocities].gradient (k, q);
                          phi_i_grads_u[k] = fe_face_values[displacements].gradient (k, q);
                        }

                      const Tensor<2,dim> pI = ALETransformations
                        ::get_pI<dim> (q, old_solution_face_values);

                      const Tensor<1,dim> v = ALETransformations
                        ::get_v<dim> (q, old_solution_face_values);

                      const Tensor<2,dim>  grad_v = ALETransformations
                        ::get_grad_v<dim> (q, old_solution_face_grads);

                      const Tensor<2,dim> grad_v_T = ALETransformations
                        ::get_grad_v_T<dim> (grad_v);

                      const Tensor<2,dim> F = ALETransformations
                        ::get_F<dim> (q, old_solution_face_grads);

                      const Tensor<2,dim> F_Inverse = ALETransformations
                        ::get_F_Inverse<dim> (F);

                      const Tensor<2,dim> F_Inverse_T = ALETransformations
                        ::get_F_Inverse_T<dim> (F_Inverse);

                      const double J = ALETransformations
                        ::get_J<dim> (F);


                      for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                          const Tensor<2,dim> grad_v_LinV = ALETransformations
                            ::get_grad_v_LinV<dim> (phi_i_grads_v[i]);

                          const double J_LinU = ALETransformations
                            ::get_J_LinU<dim> (q, old_solution_face_grads, phi_i_grads_u[i]);

                          const Tensor<2,dim> J_F_Inverse_T_LinU = ALETransformations
                            ::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);

                          const Tensor<2,dim> F_Inverse_LinU = ALETransformations
			    ::get_F_Inverse_LinU
                            (phi_i_grads_u[i], J, J_LinU, q, old_solution_face_grads);

                          const Tensor<2,dim> stress_fluid_ALE_3rd_term_LinAll =  NSEALE
                            ::get_stress_fluid_ALE_3rd_term_LinAll_short<dim>
                            (F_Inverse, F_Inverse_LinU, grad_v, grad_v_LinV,
                             viscosity, density_fluid, J, J_F_Inverse_T_LinU);

                          // Here, we multiply the symmetric part of fluid's stress tensor
                          // with the normal direction.
                          const Tensor<1,dim> neumann_value
                            = (stress_fluid_ALE_3rd_term_LinAll * fe_face_values.normal_vector(q));

                          for (unsigned int j=0; j<dofs_per_cell; ++j)
                            {
                              const unsigned int comp_j = fe.system_to_component_index(j).first;
                              if (comp_j == 0 || comp_j == 1)
                                {
                                  local_matrix(j,i) -= (timestep * theta *
                                                        neumann_value * phi_i_v[j]
                                                        ) * fe_face_values.JxW(q);
                                }
                              // end j
                            }
                          // end i
                        }
                      // end q_face_points
                    }
                  // end if-routine face integrals
                }
              // end face integrals
            }


          // Next, we compute the face integrals on the interface between fluid and structure.
          // The appear because of partial integration of the additional equations for
          // the fluid mesh motion.
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->neighbor_index(face) != -1)
                if (cell->material_id() !=  cell->neighbor(face)->material_id() &&
                    cell->face(face)->boundary_id()!=81)
                  {

                    fe_face_values.reinit (cell, face);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                      {
                        for (unsigned int k=0; k<dofs_per_cell; ++k)
                          {
                            phi_i_u[k]       = fe_face_values[displacements].value (k, q);
                            phi_i_grads_u[k] = fe_face_values[displacements].gradient (k, q);

                            phi_i_w[k]       = fe_face_values[displacements_w].value (k, q);
                            phi_i_grads_w[k] = fe_face_values[displacements_w].gradient (k, q);
                          }

                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                          {
                            const Tensor<1,dim> neumann_value_w
                              = (phi_i_grads_w[i] * fe_face_values.normal_vector(q));

                            const Tensor<1,dim> neumann_value_u
                              = (phi_i_grads_u[i] * fe_face_values.normal_vector(q));

                            for (unsigned int j=0; j<dofs_per_cell; ++j)
                              {
                                const unsigned int comp_j = fe.system_to_component_index(j).first;
                                if (comp_j == 2 || comp_j == 3)
                                  {
                                    local_matrix(j,i) -= (alpha_u * neumann_value_w * phi_i_u[j] * fe_face_values.JxW(q));

                                  }
                                else if (comp_j == 5 || comp_j == 6)
                                  {
                                    local_matrix(j,i) -= (alpha_w * neumann_value_u * phi_i_w[j] *  fe_face_values.JxW(q));

                                  }

                              }  // end j

                          }   // end i

                      }  // end q_face_points
                  }  // end if-routine face integrals


            }   // end face artificial Gamma i //


          // This is the same as discussed in step-22:
          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (local_matrix, local_dof_indices,
                                                  system_matrix);

          // Finally, we arrive at the end for assembling the matrix
          // for the fluid equations and step to the computation of the
          // structure terms:
        }
      else if (cell->material_id() == 1)
        {
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                  phi_i_v[k]       = fe_values[velocities].value (k, q);
                  phi_i_grads_v[k] = fe_values[velocities].gradient (k, q);
                  phi_i_p[k]       = fe_values[pressure].value (k, q);
                  phi_i_u[k]       = fe_values[displacements].value (k, q);
                  phi_i_grads_u[k] = fe_values[displacements].gradient (k, q);
                  phi_i_w[k]       = fe_values[displacements_w].value (k, q);
                  phi_i_grads_w[k] = fe_values[displacements_w].gradient (k, q);
                }

              // It is here the same as already shown for the fluid equations.
              // First, we prepare things coming from the previous Newton
              // iteration...
              const Tensor<2,dim> pI = ALETransformations
                ::get_pI<dim> (q, old_solution_values);

              const Tensor<2,dim> grad_v = ALETransformations
                ::get_grad_v<dim> (q, old_solution_grads);

              const Tensor<2,dim> grad_v_T = ALETransformations
                ::get_grad_v_T<dim> (grad_v);

              const Tensor<1,dim> v = ALETransformations
                ::get_v<dim> (q, old_solution_values);

              const Tensor<2,dim> F = ALETransformations
                ::get_F<dim> (q, old_solution_grads);

              const Tensor<2,dim> F_Inverse = ALETransformations
                ::get_F_Inverse<dim> (F);

              const Tensor<2,dim> F_Inverse_T = ALETransformations
                ::get_F_Inverse_T<dim> (F_Inverse);

              const Tensor<2,dim> F_T = ALETransformations
                ::get_F_T<dim> (F);

              const double J = ALETransformations
                ::get_J<dim> (F);

              const Tensor<2,dim> E = StructureTermsALE
                ::get_E<dim> (F_T, F, Identity);

              const double tr_E = StructureTermsALE
                ::get_tr_E<dim> (E);

              // ... and then things coming from the previous time steps
              const Tensor<1,dim> old_timestep_v = ALETransformations
                ::get_v<dim> (q, old_timestep_solution_values);

              const Tensor<2,dim> old_timestep_F = ALETransformations
                ::get_F<dim> (q, old_timestep_solution_grads);

              const double old_timestep_J = ALETransformations
                ::get_J<dim> (old_timestep_F);


              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  const Tensor<2,dim> pI_LinP = ALETransformations
                    ::get_pI_LinP<dim> (phi_i_p[i]);

                  const double J_LinU = ALETransformations
                    ::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);

                  const Tensor<2,dim> F_LinU = ALETransformations
                    ::get_F_LinU<dim> (phi_i_grads_u[i]);

                  const Tensor<2,dim> F_Inverse_LinU = ALETransformations
                    ::get_F_Inverse_LinU<dim>
                    (phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);

                  const Tensor<2,dim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);

                  const Tensor<2,dim> J_F_Inverse_T_LinU = ALETransformations
                    ::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);

                  const Tensor<1,dim> accelaration_term_LinAll = NSEALE
		    ::get_accelaration_term_LinAll
                    (phi_i_v[i], v, old_timestep_v, J_LinU, J, old_timestep_J, density_structure);

                  // STVK: Green-Lagrange strain tensor derivatives
                  const Tensor<2,dim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                  const double tr_E_LinU = StructureTermsALE
                    ::get_tr_E_LinU<dim> (q,old_solution_grads, phi_i_grads_u[i]);


                  // STVK
                  // Piola-kirchhoff stress structure STVK linearized in all directions
                  Tensor<2,dim> piola_kirchhoff_stress_structure_STVK_LinALL;
                  piola_kirchhoff_stress_structure_STVK_LinALL = lame_coefficient_lambda *
                    (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
                    + 2 * lame_coefficient_mu * (F_LinU * E + F * E_LinU);


                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                      // STVK
                      const unsigned int comp_j = fe.system_to_component_index(j).first;
                      if (comp_j == 0 || comp_j == 1)
                        {
                          local_matrix(j,i) += (density_structure * phi_i_v[i] * phi_i_v[j] +
                                                timestep * theta * scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL,
                                                                                  phi_i_grads_v[j])
                                                ) * fe_values.JxW(q);
                        }
                      else if (comp_j == 2 || comp_j == 3)
                        {
                          local_matrix(j,i) += (density_structure * 1.0/(cell_diameter*cell_diameter) *
                                                (phi_i_u[i] * phi_i_u[j] - timestep * theta * phi_i_v[i] * phi_i_u[j])
                                                ) *  fe_values.JxW(q);
                        }
                      else if (comp_j == 4)
                        {
                          local_matrix(j,i) += (phi_i_p[i] * phi_i_p[j]) * fe_values.JxW(q);
                        }
                      else if (comp_j == 5 || comp_j == 6)
                        {
                          local_matrix(j,i) += (alpha_w * (phi_i_w[i] * phi_i_w[j] - scalar_product(phi_i_grads_u[i],phi_i_grads_w[j]) )
                                                ) * fe_values.JxW(q);
                        }
                      // end j dofs
                    }
                  // end i dofs
                }
              // end n_q_points
            }


          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (local_matrix, local_dof_indices,
                                                  system_matrix);
          // end if (second PDE: STVK material)
        }
      // end cell
    }

  timer.exit_section();
}



template <int dim>
void
FSIALEProblem<dim>::assemble_system_rhs ()
{
  timer.enter_section("Assemble Rhs.");
  system_rhs=0;

  QGauss<dim>   quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_gradients |
                                    update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;

  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim);
  const FEValuesExtractors::Scalar pressure (dim+dim);
  const FEValuesExtractors::Vector displacements_w (dim+dim+1);

  std::vector<Vector<double> >
    old_solution_values (n_q_points, Vector<double>(dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > >
    old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+dim+1));


  std::vector<Vector<double> >
    old_solution_face_values (n_face_q_points, Vector<double>(dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > >
    old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+dim+1));

  std::vector<Vector<double> >
    old_timestep_solution_values (n_q_points, Vector<double>(dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > >
    old_timestep_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+dim+1));

  std::vector<Vector<double> >
    old_timestep_solution_face_values (n_face_q_points, Vector<double>(dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > >
    old_timestep_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+dim+1));


  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      local_rhs = 0;

      cell_diameter = cell->diameter();

      // old Newton iteration
      fe_values.get_function_values (solution, old_solution_values);
      fe_values.get_function_gradients (solution, old_solution_grads);

      // old timestep iteration
      fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);

      // Again, material_id == 0 corresponds to
      // the domain for fluid equations
      if (cell->material_id() == 0)
        {
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const Tensor<2,dim> pI = ALETransformations
                ::get_pI<dim> (q, old_solution_values);

              const Tensor<1,dim> v = ALETransformations
                ::get_v<dim> (q, old_solution_values);

              const Tensor<2,dim> grad_v = ALETransformations
                ::get_grad_v<dim> (q, old_solution_grads);

              const Tensor<2,dim> grad_u = ALETransformations
                ::get_grad_u<dim> (q, old_solution_grads);

              const Tensor<2,dim> grad_v_T = ALETransformations
                ::get_grad_v_T<dim> (grad_v);

              const Tensor<1,dim> u = ALETransformations
                ::get_u<dim> (q, old_solution_values);

              const Tensor<1,dim> w = ALETransformations
                ::get_w<dim> (q, old_solution_values);

              const Tensor<2,dim> grad_w = ALETransformations
                ::get_grad_w<dim> (q, old_solution_grads);

              const Tensor<2,dim> F = ALETransformations
                ::get_F<dim> (q, old_solution_grads);

              const Tensor<2,dim> F_Inverse = ALETransformations
                ::get_F_Inverse<dim> (F);

              const Tensor<2,dim> F_Inverse_T = ALETransformations
                ::get_F_Inverse_T<dim> (F_Inverse);

              const double J = ALETransformations
                ::get_J<dim> (F);

              // This is the fluid stress tensor in ALE formulation
              const Tensor<2,dim> sigma_ALE = NSEALE
                ::get_stress_fluid_except_pressure_ALE<dim>
                (density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );

              // We proceed by catching the previous time step values
              const Tensor<2,dim> old_timestep_pI = ALETransformations
                ::get_pI<dim> (q, old_timestep_solution_values);

              const Tensor<1,dim> old_timestep_v = ALETransformations
                ::get_v<dim> (q, old_timestep_solution_values);

              const Tensor<2,dim> old_timestep_grad_v = ALETransformations
                ::get_grad_v<dim> (q, old_timestep_solution_grads);

              const Tensor<2,dim> old_timestep_grad_v_T = ALETransformations
                ::get_grad_v_T<dim> (old_timestep_grad_v);

              const Tensor<1,dim> old_timestep_u = ALETransformations
                     ::get_u<dim> (q, old_timestep_solution_values);

              const Tensor<2,dim> old_timestep_grad_u = ALETransformations
                ::get_grad_u<dim> (q, old_timestep_solution_grads);

              const Tensor<2,dim> old_timestep_F = ALETransformations
                ::get_F<dim> (q, old_timestep_solution_grads);

              const Tensor<2,dim> old_timestep_F_Inverse = ALETransformations
                ::get_F_Inverse<dim> (old_timestep_F);

              const Tensor<2,dim> old_timestep_F_Inverse_T = ALETransformations
                ::get_F_Inverse_T<dim> (old_timestep_F_Inverse);

              const double old_timestep_J = ALETransformations
                ::get_J<dim> (old_timestep_F);

              // This is the fluid stress tensor in the ALE formulation
              // at the previous time step
              const Tensor<2,dim> old_timestep_sigma_ALE = NSEALE
                ::get_stress_fluid_except_pressure_ALE<dim>
                (density_fluid, viscosity, old_timestep_grad_v, old_timestep_grad_v_T,
                 old_timestep_F_Inverse, old_timestep_F_Inverse_T );

              Tensor<2,dim> stress_fluid;
              stress_fluid.clear();
              stress_fluid = (J * sigma_ALE * F_Inverse_T);

              Tensor<2,dim> fluid_pressure;
              fluid_pressure.clear();
              fluid_pressure = (-pI * J * F_Inverse_T);

              Tensor<2,dim> old_timestep_stress_fluid;
              old_timestep_stress_fluid.clear();
              old_timestep_stress_fluid =
                (old_timestep_J * old_timestep_sigma_ALE * old_timestep_F_Inverse_T);

              // Divergence of the fluid in the ALE formulation
              const double incompressiblity_fluid = NSEALE
                ::get_Incompressibility_ALE<dim> (q, old_solution_grads);

              // Convection term of the fluid in the ALE formulation.
              // We emphasize that the fluid convection term for
              // non-stationary flow problems in ALE
              // representation is difficult to derive.
              // For adequate discretization, the convection term will be
              // split into three smaller terms:
              Tensor<1,dim> convection_fluid;
              convection_fluid.clear();
              convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);

              // The second convection term for the fluid in the ALE formulation
              Tensor<1,dim> convection_fluid_with_u;
              convection_fluid_with_u.clear();
              convection_fluid_with_u =
                density_fluid * J * (grad_v * F_Inverse * u);

              // The third convection term for the fluid in the ALE formulation
              Tensor<1,dim> convection_fluid_with_old_timestep_u;
              convection_fluid_with_old_timestep_u.clear();
              convection_fluid_with_old_timestep_u =
                density_fluid * J * (grad_v * F_Inverse * old_timestep_u);

              // The convection term of the previous time step
              Tensor<1,dim> old_timestep_convection_fluid;
              old_timestep_convection_fluid.clear();
              old_timestep_convection_fluid =
                (density_fluid * old_timestep_J *
                 (old_timestep_grad_v * old_timestep_F_Inverse * old_timestep_v));

              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  // Fluid, NSE in ALE
                  const unsigned int comp_i = fe.system_to_component_index(i).first;
                  if (comp_i == 0 || comp_i == 1)
                    {
                      const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);
                      const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);

                      local_rhs(i) -= (density_fluid * (J + old_timestep_J)/2.0 *
                                       (v - old_timestep_v) * phi_i_v +
                                       timestep * theta * convection_fluid * phi_i_v +
                                       timestep * (1.0-theta) *
                                       old_timestep_convection_fluid * phi_i_v -
                                       (convection_fluid_with_u -
                                        convection_fluid_with_old_timestep_u) * phi_i_v +
                                       timestep * scalar_product(fluid_pressure, phi_i_grads_v) +
                                       timestep * theta * scalar_product(stress_fluid, phi_i_grads_v) +
                                       timestep * (1.0-theta) *
                                       scalar_product(old_timestep_stress_fluid, phi_i_grads_v)
                                       ) *  fe_values.JxW(q);

                    }
                  else if (comp_i == 2 || comp_i == 3)
                    {
                      const Tensor<1,dim> phi_i_u = fe_values[displacements].value (i, q);
                      const Tensor<2,dim> phi_i_grads_u = fe_values[displacements].gradient (i, q);

                      local_rhs(i) -= (alpha_u * scalar_product(grad_w, phi_i_grads_u)
                                       ) * fe_values.JxW(q);
                    }
                  else if (comp_i == 4)
                    {
                      const double phi_i_p = fe_values[pressure].value (i, q);
                      local_rhs(i) -= (incompressiblity_fluid * phi_i_p) *  fe_values.JxW(q);
                    }
                  else if (comp_i == 5 || comp_i == 6)
                    {
                      const Tensor<1,dim> phi_i_w = fe_values[displacements_w].value (i, q);
                      const Tensor<2,dim> phi_i_grads_w = fe_values[displacements_w].gradient (i, q);

                      local_rhs(i) -= alpha_w * (w * phi_i_w - scalar_product(grad_u,phi_i_grads_w)) *
                        fe_values.JxW(q);
                    }
                  // end i dofs
                }
              // close n_q_points
            }

          // As already discussed in the assembling method for the matrix,
          // we have to integrate some terms on the outflow boundary:
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->face(face)->at_boundary() &&
                  (cell->face(face)->boundary_id() == 1)
                  )
                {

                  fe_face_values.reinit (cell, face);

                  fe_face_values.get_function_values (solution, old_solution_face_values);
                  fe_face_values.get_function_gradients (solution, old_solution_face_grads);

                  fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
                  fe_face_values.get_function_gradients (old_timestep_solution, old_timestep_solution_face_grads);

                  for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                      // These are terms coming from the
                      // previous Newton iterations ...
                      const Tensor<1,dim> v = ALETransformations
                        ::get_v<dim> (q, old_solution_face_values);

                      const Tensor<2,dim> grad_v = ALETransformations
                        ::get_grad_v<dim> (q, old_solution_face_grads);

                      const Tensor<2,dim> grad_v_T = ALETransformations
                        ::get_grad_v_T<dim> (grad_v);

                      const Tensor<2,dim> F = ALETransformations
                        ::get_F<dim> (q, old_solution_face_grads);

                      const Tensor<2,dim> F_Inverse = ALETransformations
                        ::get_F_Inverse<dim> (F);

                      const Tensor<2,dim> F_Inverse_T = ALETransformations
                        ::get_F_Inverse_T<dim> (F_Inverse);

                      const double J = ALETransformations
                        ::get_J<dim> (F);

                      // ... and here from the previous time step iteration
                      const Tensor<1,dim> old_timestep_v = ALETransformations
                        ::get_v<dim> (q, old_timestep_solution_face_values);

                      const Tensor<2,dim> old_timestep_grad_v = ALETransformations
                        ::get_grad_v<dim> (q, old_timestep_solution_face_grads);

                      const Tensor<2,dim> old_timestep_grad_v_T = ALETransformations
                        ::get_grad_v_T<dim> (old_timestep_grad_v);

                      const Tensor<2,dim> old_timestep_F = ALETransformations
                        ::get_F<dim> (q, old_timestep_solution_face_grads);

                      const Tensor<2,dim> old_timestep_F_Inverse = ALETransformations
                        ::get_F_Inverse<dim> (old_timestep_F);

                      const Tensor<2,dim> old_timestep_F_Inverse_T = ALETransformations
                        ::get_F_Inverse_T<dim> (old_timestep_F_Inverse);

                      const double old_timestep_J = ALETransformations
                        ::get_J<dim> (old_timestep_F);

                      Tensor<2,dim> sigma_ALE_tilde;
                      sigma_ALE_tilde.clear();
                      sigma_ALE_tilde =
                        (density_fluid * viscosity * F_Inverse_T * grad_v_T);

                      Tensor<2,dim> old_timestep_sigma_ALE_tilde;
                      old_timestep_sigma_ALE_tilde.clear();
                      old_timestep_sigma_ALE_tilde =
                        (density_fluid * viscosity * old_timestep_F_Inverse_T * old_timestep_grad_v_T);

                      // Neumann boundary integral
                      Tensor<2,dim> stress_fluid_transposed_part;
                      stress_fluid_transposed_part.clear();
                      stress_fluid_transposed_part = (J * sigma_ALE_tilde * F_Inverse_T);

                      Tensor<2,dim> old_timestep_stress_fluid_transposed_part;
                      old_timestep_stress_fluid_transposed_part.clear();
                      old_timestep_stress_fluid_transposed_part =
                        (old_timestep_J * old_timestep_sigma_ALE_tilde * old_timestep_F_Inverse_T);

                      const Tensor<1,dim> neumann_value
                        = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));

                      const Tensor<1,dim> old_timestep_neumann_value
                        = (old_timestep_stress_fluid_transposed_part * fe_face_values.normal_vector(q));

                      for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                          const unsigned int comp_i = fe.system_to_component_index(i).first;
                          if (comp_i == 0 || comp_i == 1)
                            {
                              local_rhs(i) +=  (timestep * theta *
                                                 neumann_value * fe_face_values[velocities].value (i, q) +
                                                 timestep * (1.0-theta) *
                                                 old_timestep_neumann_value *
                                                 fe_face_values[velocities].value (i, q)
                                                 ) * fe_face_values.JxW(q);
                            }
                          // end i
                        }
                      // end face_n_q_points
                    }
                }
            }  // end face integrals do-nothing condition


           // The computation of these face integrals on the interface has
          // already been discussed in the matrix section.
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->neighbor_index(face) != -1)
                if (cell->material_id() !=  cell->neighbor(face)->material_id() &&
                    cell->face(face)->boundary_id()!=81)
                  {
                    fe_face_values.reinit (cell, face);
                    fe_face_values.get_function_gradients (solution, old_solution_face_grads);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                      {
                        const Tensor<2,dim> grad_u = ALETransformations
                          ::get_grad_u<dim> (q, old_solution_face_grads);

                        const Tensor<2,dim> grad_w = ALETransformations
                          ::get_grad_w<dim> (q, old_solution_face_grads);

                        const Tensor<1,dim> neumann_value_u
                          = (grad_u * fe_face_values.normal_vector(q));

                        const Tensor<1,dim> neumann_value_w
                          = (grad_w * fe_face_values.normal_vector(q));

                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                          {
                            const unsigned int comp_i = fe.system_to_component_index(i).first;
                            if (comp_i == 2 || comp_i == 3)
                              {
                                local_rhs(i) +=  (alpha_w * neumann_value_w *
                                                  fe_face_values[displacements].value (i, q)) *
                                  fe_face_values.JxW(q);
                              }
                            else if (comp_i == 5 || comp_i == 6)
                              {
                                local_rhs(i) +=  (alpha_u * neumann_value_u *
                                                  fe_face_values[displacements_w].value (i, q)) *
                                  fe_face_values.JxW(q);
                              }

                          }  // end i

                      }   // end face_n_q_points

                  }
            }   // end face for interface conditions


          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (local_rhs, local_dof_indices,
                                                  system_rhs);

          // Finally, we arrive at the end for assembling
          // the variational formulation for the fluid part and step to
          // the assembling process of the structure terms:
        }
      else if (cell->material_id() == 1)
        {
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const Tensor<2,dim> pI = ALETransformations
                ::get_pI<dim> (q, old_solution_values);

              const Tensor<1,dim> v = ALETransformations
                ::get_v<dim> (q, old_solution_values);

              const Tensor<2,dim> grad_v = ALETransformations
                ::get_grad_v<dim> (q, old_solution_grads);

              const Tensor<2,dim> grad_v_T = ALETransformations
                ::get_grad_v_T<dim> (grad_v);

              const Tensor<1,dim> u = ALETransformations
                ::get_u<dim> (q, old_solution_values);

              const Tensor<2,dim> grad_u = ALETransformations
                ::get_grad_u<dim> (q, old_solution_grads);

              const Tensor<1,dim> w = ALETransformations
                ::get_w<dim> (q, old_solution_values);

              const Tensor<2,dim> grad_w = ALETransformations
                ::get_grad_w<dim> (q, old_solution_grads);

              const Tensor<2,dim> F = ALETransformations
                ::get_F<dim> (q, old_solution_grads);

              const Tensor<2,dim> F_T = ALETransformations
                ::get_F_T<dim> (F);

              const Tensor<2,dim> Identity = ALETransformations
                ::get_Identity<dim> ();

              const Tensor<2,dim> F_Inverse = ALETransformations
                ::get_F_Inverse<dim> (F);

              const Tensor<2,dim> F_Inverse_T = ALETransformations
                ::get_F_Inverse_T<dim> (F_Inverse);

              const double J = ALETransformations
                ::get_J<dim> (F);

              const Tensor<2,dim> E = StructureTermsALE
                ::get_E<dim> (F_T, F, Identity);

              const double tr_E = StructureTermsALE
                ::get_tr_E<dim> (E);

              // Previous time step values
              const Tensor<2,dim> old_timestep_pI = ALETransformations
                ::get_pI<dim> (q, old_timestep_solution_values);

              const Tensor<1,dim> old_timestep_v = ALETransformations
                ::get_v<dim> (q, old_timestep_solution_values);

              const Tensor<2,dim> old_timestep_grad_v = ALETransformations
                ::get_grad_v<dim> (q, old_timestep_solution_grads);

              const Tensor<2,dim> old_timestep_grad_v_T = ALETransformations
                ::get_grad_v_T<dim> (old_timestep_grad_v);

              const Tensor<1,dim> old_timestep_u = ALETransformations
                ::get_u<dim> (q, old_timestep_solution_values);

              const Tensor<2,dim> old_timestep_F = ALETransformations
                ::get_F<dim> (q, old_timestep_solution_grads);

              const Tensor<2,dim> old_timestep_F_Inverse = ALETransformations
                ::get_F_Inverse<dim> (old_timestep_F);

              const Tensor<2,dim> old_timestep_F_T = ALETransformations
                ::get_F_T<dim> (old_timestep_F);

              const Tensor<2,dim> old_timestep_F_Inverse_T = ALETransformations
                ::get_F_Inverse_T<dim> (old_timestep_F_Inverse);

              const double old_timestep_J = ALETransformations
                ::get_J<dim> (old_timestep_F);

              const Tensor<2,dim> old_timestep_E = StructureTermsALE
                ::get_E<dim> (old_timestep_F_T, old_timestep_F, Identity);

              const double old_timestep_tr_E = StructureTermsALE
                ::get_tr_E<dim> (old_timestep_E);


              // STVK structure model
              Tensor<2,dim> sigma_structure_ALE;
              sigma_structure_ALE.clear();
              sigma_structure_ALE = (1.0/J *
                                     F * (lame_coefficient_lambda *
                                          tr_E * Identity +
                                          2 * lame_coefficient_mu *
                                          E) *
                                     F_T);


              Tensor<2,dim> stress_term;
              stress_term.clear();
              stress_term = (J * sigma_structure_ALE * F_Inverse_T);

              Tensor<2,dim> old_timestep_sigma_structure_ALE;
              old_timestep_sigma_structure_ALE.clear();
              old_timestep_sigma_structure_ALE = (1.0/old_timestep_J *
                                                  old_timestep_F * (lame_coefficient_lambda *
                                                                    old_timestep_tr_E * Identity +
                                                                    2 * lame_coefficient_mu *
                                                                    old_timestep_E) *
                                                  old_timestep_F_T);

              Tensor<2,dim> old_timestep_stress_term;
              old_timestep_stress_term.clear();
              old_timestep_stress_term = (old_timestep_J * old_timestep_sigma_structure_ALE * old_timestep_F_Inverse_T);

              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  // STVK structure model
                  const unsigned int comp_i = fe.system_to_component_index(i).first;
                  if (comp_i == 0 || comp_i == 1)
                    {
                      const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);
                      const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);

                      local_rhs(i) -= (density_structure * (v - old_timestep_v) * phi_i_v +
                                       timestep * theta * scalar_product(stress_term,phi_i_grads_v) +
                                       timestep * (1.0-theta) * scalar_product(old_timestep_stress_term, phi_i_grads_v)
                                       ) * fe_values.JxW(q);

                    }
                  else if (comp_i == 2 || comp_i == 3)
                    {
                      const Tensor<1,dim> phi_i_u = fe_values[displacements].value (i, q);
                      local_rhs(i) -=  (density_structure * 1.0/(cell_diameter*cell_diameter) *
                                        ((u - old_timestep_u) * phi_i_u -
                                         timestep * (theta * v + (1.0-theta) *
                                                     old_timestep_v) * phi_i_u)
                                        ) * fe_values.JxW(q);

                    }
                  else if (comp_i == 4)
                    {
                      const double phi_i_p = fe_values[pressure].value (i, q);
                      local_rhs(i) -= (old_solution_values[q](dim+dim) * phi_i_p) * fe_values.JxW(q);

                    }
                  else if (comp_i == 5 || comp_i == 6)
                    {
                      const Tensor<1,dim> phi_i_w = fe_values[displacements_w].value (i, q);
                      const Tensor<2,dim> phi_i_grads_w = fe_values[displacements_w].gradient (i, q);

                      local_rhs(i) -= alpha_w * (w * phi_i_w - scalar_product(grad_u,phi_i_grads_w)) *
                        fe_values.JxW(q);
                    }
                  // end i
                }
              // end n_q_points
            }

          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (local_rhs, local_dof_indices,
                                                  system_rhs);

        // end if (for STVK material)
        }

    }  // end cell

  timer.exit_section();
}


template <int dim>
void
FSIALEProblem<dim>::set_initial_bc (const double time)
{
    std::map<unsigned int,double> boundary_values;
    std::vector<bool> component_mask (dim+dim+dim+1, true);
    // (Scalar) pressure
    component_mask[dim+dim] = false;

    // Additional displacement w:
    component_mask[dim+dim+1] = false;
    component_mask[dim+dim+dim] = false;

    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryParabolic<dim>(time),
                                              boundary_values,
                                              component_mask);

    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              boundary_values,
                                              component_mask);

    VectorTools::interpolate_boundary_values (dof_handler,
                                              80,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              boundary_values,
                                              component_mask);

    VectorTools::interpolate_boundary_values (dof_handler,
                                              81,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              boundary_values,
                                              component_mask);

    component_mask[0] = false;
    component_mask[1] = false;

    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              boundary_values,
                                              component_mask);

    for (typename std::map<unsigned int, double>::const_iterator
           i = boundary_values.begin();
         i != boundary_values.end();
         ++i)
      solution(i->first) = i->second;

}

template <int dim>
void
FSIALEProblem<dim>::set_newton_bc ()
{
    std::vector<bool> component_mask (dim+dim+dim+1, true);
    component_mask[dim+dim] = false;
    component_mask[dim+dim+1] = false;
    component_mask[dim+dim+dim] = false;

    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              constraints,
                                              component_mask);

    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              constraints,
                                              component_mask);

    VectorTools::interpolate_boundary_values (dof_handler,
                                              80,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              constraints,
                                              component_mask);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              81,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              constraints,
                                              component_mask);
    component_mask[0] = false;
    component_mask[1] = false;

    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              ZeroFunction<dim>(dim+dim+dim+1),
                                              constraints,
                                              component_mask);
}


template <int dim>
void FSIALEProblem<dim>::newton_iteration (const double time)

{
  Timer timer_newton;
  const double lower_bound_newton_residuum = 1.0e-8;
  const unsigned int max_no_newton_steps  = 40;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1;

  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 10;
  const double line_search_damping = 0.6;
  double new_newton_residuum;

  // Application of the initial boundary conditions to the
  // variational equations:
  set_initial_bc (time);
  assemble_system_rhs();

  double newton_residuum = system_rhs.linfty_norm();
  double old_newton_residuum= newton_residuum;
  unsigned int newton_step = 1;

  if (newton_residuum < lower_bound_newton_residuum)
    {
      std::cout << '\t'
                << std::scientific
                << newton_residuum
                << std::endl;
    }

  while (newton_residuum > lower_bound_newton_residuum &&
         newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residuum = newton_residuum;

      assemble_system_rhs();
      newton_residuum = system_rhs.linfty_norm();

      if (newton_residuum < lower_bound_newton_residuum)
        {
          std::cout << '\t'
                    << std::scientific
                    << newton_residuum << std::endl;
          break;
        }

      if (newton_residuum/old_newton_residuum > nonlinear_rho)
        assemble_system_matrix ();

      // Solve Ax = b
      solve ();

      line_search_step = 0;
      for ( ;
            line_search_step < max_no_line_search_steps;
            ++line_search_step)
        {
          solution += newton_update;

          assemble_system_rhs ();
          new_newton_residuum = system_rhs.linfty_norm();

          if (new_newton_residuum < newton_residuum)
              break;
          else
            solution -= newton_update;

          newton_update *= line_search_damping;
        }

      timer_newton.stop();

      std::cout << std::setprecision(5) <<newton_step << '\t'
                << std::scientific << newton_residuum << '\t'
                << std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
        std::cout << "r" << '\t' ;
      else
        std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t'
                << std::scientific << timer_newton ()
                << std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;
    }
}

 template <int dim>
void
FSIALEProblem<dim>::solve ()
{
  timer.enter_section("Solve linear system.");
  Vector<double> sol, rhs;
  sol = newton_update;
  rhs = system_rhs;

  SparseDirectUMFPACK A_direct;
  A_direct.factorize(system_matrix);
  A_direct.vmult(sol,rhs);
  newton_update = sol;

  constraints.distribute (newton_update);
  timer.exit_section();
}


/* This function is known from almost all other
 * tutorial steps in deal.II.
 * However, we emphasize that the FSI problem
 * is computed on a fixed mesh (instead of
 * moving the mesh as done in other references;
 * we refer the reader to the accompanying article
 * and the comments made therein).
 * For this reason, the output of the
 * solution in *.vtk format corresponds to the
 * solution on the fixed mesh and, therefore,
 * in a visualization program, the reader
 * has to postprocess the solution.
 * We also refer to the MappingQEulerian in
 * the deal.II librar, which is able
 * to tansform the solution to the current (i.e.,
 * the physical mesh)
 */
template <int dim>
void
FSIALEProblem<dim>::output_results (const unsigned int refinement_cycle,
                              const BlockVector<double> output_vector)  const
{

  std::vector<std::string> solution_names (dim, "velocity");
  solution_names.push_back ("displacement");
  solution_names.push_back ("displacement");
  solution_names.push_back ("p_fluid");
  solution_names.push_back ("displace_w");
  solution_names.push_back ("displace_w");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim+dim, DataComponentInterpretation::component_is_part_of_vector);

  data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);

  data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);

  data_out.add_data_vector (output_vector, solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);

  data_out.build_patches ();

  std::string filename_basis;
  filename_basis  = "solution_fsi_1_"; 

  std::ostringstream filename;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
  filename << filename_basis
           << Utilities::int_to_string (refinement_cycle, 5)
           << ".vtk";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtk (output);

}

template <int dim>
double FSIALEProblem<dim>::compute_point_value (Point<dim> p,
                                               const unsigned int component) const
{

  Vector<double> tmp_vector(dim+dim+dim+1);
  VectorTools::point_value (dof_handler,
                            solution,
                            p,
                            tmp_vector);

  return tmp_vector(component);
}

template <int dim>
void FSIALEProblem<dim>::compute_drag_lift_fsi_fluid_tensor()
{

  const QGauss<dim-1> face_quadrature_formula (3);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values | update_gradients | update_normal_vectors |
                                    update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<Vector<double> >  face_solution_values (n_face_q_points,
                                                      Vector<double> (dim+dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > >
    face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+dim+1));

  Tensor<1,dim> drag_lift_value;

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

   for (; cell!=endc; ++cell)
     {

       // First, we are going to compute the forces that
       // act on the cylinder. We notice that only the fluid
       // equations are defined here.
       for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
         if (cell->face(face)->at_boundary() &&
             cell->face(face)->boundary_id()==80)
           {
             fe_face_values.reinit (cell, face);
             fe_face_values.get_function_values (solution, face_solution_values);
             fe_face_values.get_function_gradients (solution, face_solution_grads);

             for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
               {
                 const Tensor<2,dim> pI = ALETransformations
                   ::get_pI<dim> (q_point, face_solution_values);

                 const Tensor<1,dim> v = ALETransformations
                   ::get_v<dim> (q_point, face_solution_values);

                 const Tensor<2,dim> grad_v = ALETransformations
                   ::get_grad_v<dim> (q_point, face_solution_grads);

                 const Tensor<2,dim> grad_v_T = ALETransformations
                   ::get_grad_v_T<dim> (grad_v);

                 const Tensor<2,dim> F = ALETransformations
                   ::get_F<dim> (q_point, face_solution_grads);

                 const Tensor<2,dim> F_Inverse = ALETransformations
                   ::get_F_Inverse<dim> (F);

                 const Tensor<2,dim> F_Inverse_T = ALETransformations
                   ::get_F_Inverse_T<dim> (F_Inverse);

                 const double J = ALETransformations
                   ::get_J<dim> (F);

                 const Tensor<2,dim> sigma_ALE = NSEALE
                   ::get_stress_fluid_except_pressure_ALE<dim>
                   (density_fluid, viscosity,
                    grad_v, grad_v_T, F_Inverse, F_Inverse_T );

                 Tensor<2,dim> stress_fluid;
                 stress_fluid.clear();
                 stress_fluid = (J * sigma_ALE * F_Inverse_T);

                 Tensor<2,dim> fluid_pressure;
                 fluid_pressure.clear();
                 fluid_pressure = (-pI * J * F_Inverse_T);

                 drag_lift_value -= (stress_fluid + fluid_pressure) *
                   fe_face_values.normal_vector(q_point)* fe_face_values.JxW(q_point);

               }
           } // end boundary 80 for fluid

       // Now, we compute the forces that act on the beam. Here,
       // we have two possibilities as already discussed in the paper.
       // We use again the fluid tensor to compute
       // drag and lift:
       if (cell->material_id() == 0)
         {
           for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
             if (cell->neighbor_index(face) != -1)
               if (cell->material_id() !=  cell->neighbor(face)->material_id() &&
                   cell->face(face)->boundary_id()!=80)
                 {

                   fe_face_values.reinit (cell, face);
                   fe_face_values.get_function_values (solution, face_solution_values);
                   fe_face_values.get_function_gradients (solution, face_solution_grads);

                   for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                     {
                       const Tensor<2,dim> pI = ALETransformations
                         ::get_pI<dim> (q_point, face_solution_values);

                       const Tensor<1,dim> v = ALETransformations
                         ::get_v<dim> (q_point, face_solution_values);

                       const Tensor<2,dim> grad_v = ALETransformations
                         ::get_grad_v<dim> (q_point, face_solution_grads);

                       const Tensor<2,dim> grad_v_T = ALETransformations
                         ::get_grad_v_T<dim> (grad_v);

                       const Tensor<2,dim> F = ALETransformations
                         ::get_F<dim> (q_point, face_solution_grads);

                       const Tensor<2,dim> F_Inverse = ALETransformations
                         ::get_F_Inverse<dim> (F);

                       const Tensor<2,dim> F_Inverse_T = ALETransformations
                         ::get_F_Inverse_T<dim> (F_Inverse);

                       const double J = ALETransformations
                         ::get_J<dim> (F);

                       const Tensor<2,dim> sigma_ALE = NSEALE
                         ::get_stress_fluid_except_pressure_ALE<dim>
                         (density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );

                       Tensor<2,dim> stress_fluid;
                       stress_fluid.clear();
                       stress_fluid = (J * sigma_ALE * F_Inverse_T);

                       Tensor<2,dim> fluid_pressure;
                       fluid_pressure.clear();
                       fluid_pressure = (-pI * J * F_Inverse_T);

                       drag_lift_value -= (stress_fluid + fluid_pressure) *
                         fe_face_values.normal_vector(q_point)* fe_face_values.JxW(q_point);
                     }
                 }
         }
     }

   std::cout << "Drag: " << drag_lift_value[0] << std::endl;
   std::cout << "Lift: " << drag_lift_value[1] << std::endl;
}

template<int dim>
void FSIALEProblem<dim>::compute_functional_values()
{
  double x1,y1;
  x1 = compute_point_value(Point<dim>(0.6,0.2), dim);
  y1 = compute_point_value(Point<dim>(0.6,0.2), dim+1);

  std::cout << "------------------" << std::endl;
  std::cout << "DisX: " << x1 << std::endl;
  std::cout << "DisY: " << y1 << std::endl;
  std::cout << "------------------" << std::endl;

  compute_drag_lift_fsi_fluid_tensor();

  std::cout << std::endl;
}

 template <int dim>
void FSIALEProblem<dim>::run ()
{
  setup_system();

  std::cout << "\n=============================="
            << "====================================="  << std::endl;
  std::cout << "Parameters\n"
            << "==========\n"
            << "Density fluid:     "   <<  density_fluid << "\n"
            << "Density structure: "   <<  density_structure << "\n"
            << "Viscosity fluid:   "   <<  viscosity << "\n"
            << "alpha_u:           "   <<  alpha_u << "\n"
            << "alpha_w:           "   <<  alpha_w << "\n"
            << "Lame coeff. mu:    "   <<  lame_coefficient_mu << "\n"
            << std::endl;


  const unsigned int output_skip = 5;
  do
    {
      std::cout << "Timestep " << timestep_number
                << " (" << time_stepping_scheme
                << ")" <<    ": " << time
                << " (" << timestep << ")"
                << "\n=============================="
                << "====================================="
                << std::endl;

      std::cout << std::endl;

      // Compute next time step
      old_timestep_solution = solution;
      newton_iteration (time);
      time += timestep;

      // Compute functional values: dx, dy, drag, lift
      std::cout << std::endl;
      compute_functional_values();

      // Write solutions
      if ((timestep_number % output_skip == 0))
        output_results (timestep_number,solution);


      ++timestep_number;

    }
  while (timestep_number <= max_no_timesteps);


}

// The main function looks almost the same
// as in all other deal.II tuturial steps.
int main ()
{
  try
    {
      deallog.depth_console (0);

      FSIALEProblem<2> fsi_problem(1);
      fsi_problem.run ();
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

  return 0;
}

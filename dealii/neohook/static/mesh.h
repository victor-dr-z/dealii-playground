#ifndef NEOHOOK_MESH__
#define NEOHOOK_MESH__

#include "params.h"

// deal.II section
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

template <int dim>
class Mesh {
 public:
  Mesh(const dealii::ParameterHandler & prm);

  void MakeGrid(dealii::Triangulation<dim>& tria);
 private:
  const std::string domain_shape_;
}

#endif //NEOHOOK_MESH__

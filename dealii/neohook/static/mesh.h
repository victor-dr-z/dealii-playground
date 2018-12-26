#ifndef NEOHOOK_MESH__
#define NEOHOOK_MESH__

#include "params.h"


// std section
#include <unordered_map>

// deal.II section
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

template <int dim>
class Mesh {
 public:
  Mesh();
  void MakeGrid(dealii::Triangulation<dim>& tria);
 private:
  const std::string domain_shape_;
  const int refinements_;//!< number of uniform refinements
  const std::vector<int> cells_;
  const std::vector<double> lengths_;
};

template <int dim>
Mesh<dim>::Mesh()
    :
    domain_shape_(params::GetParam<std::string>("domain geometry shape")),
    refinements_(params::GetParam<std::string>("uniform refinements")),
    cells_(params::GetParam<std::vector<int>>("number of cells")),
    lengths_(params::GetParam<std::vector<double>>("side lengths")) {}

template <int dim>
void Mesh<dim>::MakeGrid(dealii::Triangulation<dim> & tria) {
  std::unordered_map<std::string, int> m = {{"rectangle", 1}};
  switch (m[domain_shape_]) {
    case 1: {
      std::vector<unsigned int> n_cells(cells_.begin(), cells_.end());
      dealii::Point<dim> orgin, diag(&lengths_[0]);
      dealii::GridGenerator::subdivided_hyper_rectangle (tria, n_cells,
          origin, diag);
    }
    default: {
      std::cerr << "Not implemented yet for other geometries";
      return;
    }
  }
}

#endif //NEOHOOK_MESH__

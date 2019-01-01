#ifndef FSI_MESH_BASE__
#define FSI_MESH_BASE__

#include "params.h"


// std section
#include <unordered_map>

// deal.II section
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_generator.h>

template <int dim>
class MeshBase {
 public:
  /*!
   Constructor of the MeshBase class
   */
  MeshBase();
  virtue void BuildGrid(dealii::Triangulation<dim>& tria)=0;
  virtue void ReadGrid(dealii::Triangulation<dim>& tria);
  virtue void MakeGrid(dealii::Triangulation<dim>& tria);
 private:
  const std::string domain_shape_;
  const int refinements_;//!< number of uniform refinements
  const std::vector<int> cells_;
  const std::vector<double> lengths_;
};

template <int dim>
MeshBase<dim>::MeshBase()
    :
    domain_shape_(params::GetParam<std::string>("domain geometry shape")),
    refinements_(params::GetParam<std::string>("uniform refinements")),
    cells_(params::GetParam<std::vector<int>>("number of cells")),
    lengths_(params::GetParam<std::vector<double>>("side lengths")) {}

template <int dim>
void MeshBase<dim>::ReadGrid(dealii::Triangulation<dim>& tria) {
  // TODO: maybe read-in mesh functionality in the future
}

template <int dim>
void MeshBase<dim>::MakeGrid(dealii::Triangulation<dim> & tria) {
  std::unordered_map<std::string, int> m = {{"build", 0}, {"read", 1}};
  switch (m[mesh_option_]) {
    case 0: {
      BuildGrid(tria);
    }
    case 1: {
      ReadGrid(tria);
    }
    default: {
      std::cerr << "mesh option entry not valid";
      break;
    }
  }
}

#endif //FSI_MESH__

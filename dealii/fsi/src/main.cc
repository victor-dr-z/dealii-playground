
#include "driver.h"

int main(int argc, char * argv[]) {
  if (argc!=2) {
    std::cerr << "*** Call this program as ./FSI params.prm ***"
    return 1;
  }
  params::DeclareParams();
  params::ParseInput(argv);
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,
      dealii::numbers::invalid_unsigned_int);
  auto dim = params::GetParam("dimension");
  switch (dim) {
    case 2: {
      Driver<2> problem;
      problem.run();
      break;
    }
    case 3: {
      Driver<3> problem;
      problem.run();
      break;
    }
    default: {
      std::cerr << "only 3d is implemented";
      return 1;
    }
  }
}

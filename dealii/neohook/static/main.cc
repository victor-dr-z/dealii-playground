
#include "driver.h"

int main(int argc, char * argv[]) {
  if (argc!=2) {
    std::cerr << "*** Call this program as ./neohook params.prm ***"
    return 1;
  }
  params::DeclareParams();
  params::ParseInput(argv);
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,
      dealii::numbers::invalid_unsigned_int);
  auto dim = params::GetParam("dimension");
  switch (dim) {
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

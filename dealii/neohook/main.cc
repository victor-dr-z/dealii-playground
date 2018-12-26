#include "driver.h"

int main(int argc, char * argv[]) {
  if (argc!=2) {
    std::cerr << "*** Call this program as ./neohook params.prm ***"
    return 1;
  }
  params::DeclareParams();
  params::GlobPrm.parse_input(argv[1]);
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,
      dealii::numbers::invalid_unsigned_int);
  Driver<3> problem;
  problem.run();
}

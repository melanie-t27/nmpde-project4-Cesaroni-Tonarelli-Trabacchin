//
// Created by Melanie Tonarelli on 24/01/24.
//
#include "Solver.hpp"
#include "utils.hpp"
#include "Coupler.hpp"
#include "IonicModel.hpp"
#include "BuenoOrovioIonicModelIon1.hpp"
#include "SVICoupler.hpp"
#include <array>
#include <memory>

using namespace dealii;

int main(int argc, char *argv[]){
    const unsigned int dim = 3;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    const unsigned int degree = 2;

    const double T     = 1.0;
    const double theta_fe = 0.5;
    const double theta_ode = 0.5;
    const double deltat = 0.1;

    std::unique_ptr<U_0<dim>> u_0 = std::make_unique<U_0<dim>>();
    std::array<std::unique_ptr<Function<dim>>, 3> gating_variables_0({std::make_unique<GatingVariable_V0<dim>>(), std::make_unique<GatingVariable_W0<dim>>(), std::make_unique<GatingVariable_S0<dim>>() });
    std::shared_ptr<BuenoOrovioIonicModelIon1<1,3>> ionic_model = std::make_shared<BuenoOrovioIonicModelIon1<1,3>>();
    std::shared_ptr<SVICoupler<1,1,3>> coupler = std::make_shared<SVICoupler<1,1,3>>();
    std::unique_ptr<Iapp<dim>> I_app = std::make_unique<Iapp<dim>>();
    std::unique_ptr<D<dim>> d = std::make_unique<D<dim>>();
    Solver<1,1,3> solver("../meshes/mesh-cube-20.msh", degree, T, deltat, theta_fe, theta_ode, ionic_model, coupler, std::move(d), std::move(I_app), std::move(u_0), gating_variables_0);
    solver.solve();
    return 0;
}
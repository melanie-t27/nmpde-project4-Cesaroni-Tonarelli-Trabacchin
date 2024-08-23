//
// Created by Melanie Tonarelli on 24/01/24.
//
#include "Solver.hpp"
#include "utils.hpp"
#include "Coupler.hpp"
#include "IonicModel.hpp"
#include "BuenoOrovioIonicModelIon2.hpp"
#include "BuenoOrovioIonicModelIon1.hpp"
#include "SVICoupler.hpp"
#include "ICICoupler.hpp"
#include "QCoupler.hpp"
#include <array>
#include <memory>

using namespace dealii;

int main(int argc, char *argv[]){
    std::cout << "main started" << std::endl;
    const unsigned int dim = 3;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    const unsigned int degree = 1;
    // Default values
    double T     = 0.05;
    double theta_fe = 0.5;
    double theta_ode = 0.5;
    double deltat = 0.05/1000;
    std::string filename = "../meshes/cuboid.msh";

    // Process command line arguments
    for (int i = 1; i < argc; ++i)
    {
        /*if (std::string(argv[i]) == "-tt" && i + 1 < argc) // tissue type
        {
            tissue_type = std::stoi(argv[i + 1]);
            ++i;
        }
        else */if (std::string(argv[i]) == "-fn" && i + 1 < argc) // mesh name
        {
            filename = argv[i + 1];
            ++i;
        }
        else if (std::string(argv[i]) == "-T" && i + 1 < argc){ // Total time
            T = std::stod(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-dT" && i + 1 < argc){
            deltat = std::stod(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-tfe" && i + 1 < argc){
            theta_fe = std::stod(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-tode" && i + 1 < argc){
            theta_ode = std::stod(argv[i + 1]);
            ++i;
        } else {
            if(mpi_rank == 0){
                std::cout << "Wrong parameters!" << std::endl;
                return 1;
            }
        }
    }


    std::unique_ptr<U_0<dim>> u_0 = std::make_unique<U_0<dim>>();
    std::array<std::unique_ptr<Function<dim>>, 3> gating_variables_0{{std::make_unique<GatingVariable_V0<dim>>(), std::make_unique<GatingVariable_W0<dim>>(), std::make_unique<GatingVariable_S0<dim>>() }};
    std::shared_ptr<BuenoOrovioIonicModelIon1<1,3>> ionic_model = std::make_shared<BuenoOrovioIonicModelIon1<1,3>>();
    std::shared_ptr<ICICoupler<1,1,3>> coupler = std::make_shared<ICICoupler<1,1,3>>();
    std::unique_ptr<Iapp<dim>> I_app = std::make_unique<Iapp<dim>>();
    std::unique_ptr<D<dim>> d = std::make_unique<D<dim>>();
    Solver<1,1,3> solver(filename, degree, T, deltat, theta_fe, theta_ode, ionic_model, coupler, std::move(d), std::move(I_app), std::move(u_0), gating_variables_0);
    solver.solve();
    return 0;
}
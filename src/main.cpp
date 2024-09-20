#include "MonodomainSolver/Solver.hpp"
#include "utils.hpp"
#include "Couplers/Coupler.hpp"
#include "IonicModels/IonicModel.hpp"
#include "IonicModels/BuenoOrovioModel/BOModel.hpp"
#include "Couplers/SVICoupler.hpp"
#include "Couplers/ICICoupler.hpp"
#include "Couplers/GICoupler.hpp"
#include <array>
#include <memory>

using namespace dealii;

int main(int argc, char *argv[]){
    const unsigned int dim = 3;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    const unsigned int degree = 1;
    constexpr int n_ion = 3;
    // Default values
    double T     = 0.05;
    double theta_fe = 0.5;
    double theta_ode = 0.5;
    double deltat = 0.05/1000;
    int tissue_type = 0;
    int coupler_type = 0;
    std::string filename = "../meshes/cuboid-step-0.0005.msh";
    std::string output_folder = "./";
    int os = 100;

    // Process command line arguments to override default values with user-specified ones
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "-tt" && i + 1 < argc) // tissue type
        {
            tissue_type = std::stoi(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-fn" && i + 1 < argc) // mesh name
        {
            filename = argv[i + 1];
            ++i;
        }
        else if (std::string(argv[i]) == "-ct" && i + 1 < argc){ // coupler type
            coupler_type = std::stoi(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-T" && i + 1 < argc){ // Total time
            T = std::stod(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-dT" && i + 1 < argc){ // time step
            deltat = std::stod(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-tfe" && i + 1 < argc){ // theta_fe
            theta_fe = std::stod(argv[i + 1]);
            ++i;
        }
        else if (std::string(argv[i]) == "-tode" && i + 1 < argc){ // theta_ode
            theta_ode = std::stod(argv[i + 1]);
            ++i;
        } 
        else if(std::string(argv[i]) == "-o" && i + 1 < argc) { // output
            output_folder = argv[i + 1];
            ++i;
        } else if(std::string(argv[i]) == "-os" && i + 1 < argc) {
            os =  std::stoi(argv[i + 1]);
            ++i;
        }
        else {
            if(mpi_rank == 0){
                std::cout << "Wrong parameters!" << std::endl;
                return 1;
            }
        }
    }

    // We initialize all the objects passed as parameters to the constructor of the Solver class:

    // Initial condition (defined in utils)
    std::unique_ptr<U_0<dim>> u_0 = std::make_unique<U_0<dim>>();

    // Initial gating variables (defined in utils)
    std::array<std::unique_ptr<Function<dim>>, n_ion> gating_variables_0{{std::make_unique<GatingVariable_V0<dim>>(), std::make_unique<GatingVariable_W0<dim>>(), std::make_unique<GatingVariable_S0<dim>>() }};

    // Ionic model, with the tissue_type passed as a parameter to the constructor
    std::shared_ptr<BuenoOrovioIonicModel<n_ion>> ionic_model = std::make_shared<BuenoOrovioIonicModel<n_ion>>(tissue_type);

    // Coupler, which will be chosen based on the command line argument
    std::shared_ptr<Coupler<n_ion>> coupler;

    //Based on coupler_type value, we create an instance of the corresponding coupler
    switch (coupler_type)
    {
    case 0:
        coupler = std::make_shared<ICICoupler<n_ion>>();
        break;

    case 1:
        coupler = std::make_shared<GICoupler<n_ion>>();
        break;
    
    case 2:
        coupler = std::make_shared<SVICoupler<n_ion>>();
        break;
    
    default:
        throw -1;
        break;
    }

    // Applied external stimulus current
    std::unique_ptr<Iapp<dim>> I_app = std::make_unique<Iapp<dim>>();

    // Tissue conductivity tensor
    std::unique_ptr<D<dim>> d = std::make_unique<D<dim>>();
    Solver<n_ion> solver(filename, degree, T, deltat, theta_fe, theta_ode, os, output_folder, ionic_model, coupler, std::move(d), std::move(I_app), std::move(u_0), gating_variables_0);
    solver.solve();
    return 0;
}
#ifndef SOLVER_H
#define SOLVER_H

#include "../IonicModels/IonicModel.hpp"
#include "../Couplers/Coupler.hpp"
#include "../utils.hpp"
#include "../IonicModels/ODESolver.hpp"
#include "FESolver.hpp"
#include <deque>
#include <fstream>
#include <iostream>

using namespace dealii;

// The class coordinates the 3 modules: Coupler, IonicModel and MonodomaiSolver
// N_ion is the number of ODEs in the ionic model.
// In our case N_ion is set to 3
template<int N_ion>
class Solver {
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 3;

public:
    // Constructor. We provide mesh_file_name, polynomial degree, final time, time step Delta t, theta method, output steps
    // parameters, mass lumping parameter, output_folder, ionic model, coupler, tissue conductivity tensor, I_app,
    // initial condition and the initial gating variables as constructor parameters.
    Solver(
            const std::string  &mesh_file_name_,
            const unsigned int &r_,
            const double       &T_,
            const double       &deltat_,
            const double       &fe_theta_,
            const double       &ode_theta_,
            const int          &os_,
            std::string output_folder_,
            std::shared_ptr<IonicModel<N_ion>> ionic_model_,
            std::shared_ptr<Coupler<N_ion>> coupler_,
            std::unique_ptr<TensorFunction<2, dim>> d_,
            std::unique_ptr<Function<dim>> I_app_,
            std::unique_ptr<Function<dim>> u_0,
            std::array<std::unique_ptr<Function<dim>>, N_ion>& gate_vars_0
    )
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , fe_theta(fe_theta_)
    , ode_theta(ode_theta_)
    , output_folder(output_folder_)
    , mesh(MPI_COMM_WORLD)
    , ionic_model(ionic_model_)
    , coupler(coupler_)
    , ode_solver(ode_theta_, deltat_, ionic_model_)
    , os(os_)
    {
        // this method is responsible for:
        // - Initializing the mesh
        // - Initializing the finite element space and quadrature object
        // - Initializing the DoF handler
        init();
        // we initialize the fe_solver object with the needed constructor parameters
        fe_solver = std::make_unique<FESolver>(r_, T_, deltat_, fe_theta_, output_folder_, mesh, fe, quadrature, dof_handler, std::move(d_), std::move(I_app_));
        // The setup method initializes the linear system, including the sparsity pattern, matrices, system rhs,
        // and solution vector. It also assembles the matrices M and K, which are time-independent and can be
        // assembled once outside the temporal loop.
        fe_solver->setup();
        // Solution is set with initial value. The method takes in input  the initial condition u_0 and evaluates it on dofs
        fe_solver->setInitialSolution(std::move(u_0));
        // The gating variables are initialized with the inital values.
        // Depending on the chosen coupler, they are evaluated on dofs (ICICoupler and SVICoupler)
        // or they are interpolated on quadrature nodes (GICoupler)
        coupler->setInitialGatingVariables(*this, std::move(gate_vars_0));
        #ifdef CHECK_ACTIVATION_TIMES
        std::cout << "checking activation times" << std::endl;
        // we initialize the vector containing the activation times
        activation_times_init();
        #endif
        
    }

    // Methods to retrieve private members
    std::shared_ptr<FiniteElement<dim>> getFiniteElement() {
        return fe;
    }

    std::shared_ptr<Quadrature<dim>> getQuadrature() {
        return quadrature;
    }

    DoFHandler<dim>& getDofHandler() {
        return dof_handler;
    }

    parallel::fullydistributed::Triangulation<dim>& getMesh() {
        return mesh;
    }

    double& getIonicCurrent(int cell_index, int q) {
        return fe_solver->getIonicCurrent(cell_index, q);
    }

    TrilinosWrappers::MPI::Vector& getFESolution() {
        return fe_solver->getSolution();
    }

    TrilinosWrappers::MPI::Vector& getFESolutionOwned() {
        return fe_solver->getSolutionOwned();
    }


    ODESolver<N_ion>& getOdeSolver() {
        return ode_solver;
    }

    std::unique_ptr<FESolver>& getFESolver() {
        return fe_solver;
    }

    auto getIonicModel() {
        return ionic_model;
    }

    double getDeltaT() {
        return deltat;
    }

    double getT() {
        return T;
    }

    double getThetaFE() {
        return fe_theta;
    }

    double getThetaODE() {
        return ode_theta;
    }

    unsigned int getR() {
        return r;
    }

    void solve() {
        time = 0.0;
        unsigned int time_step = 0;
        #ifndef CHECK_ACTIVATION_TIMES
        fe_solver->output(time_step);
        #endif
        while(time < T) {
            time += deltat;
            time_step++;
            pcout << "solving time step " << time_step << std::endl;
            // we first solve the system of ODEs (the method takes in input a reference to
            // an instance of the class Solver)
            coupler->solveOde(*this);
            // we then solve the linear system assembled using FE method for the monodomain equation
            // (the method takes in input a reference to an instance of the class Solver and current time)
            coupler->solveFE(*this, time);
    
            if(time_step % os == 0) {
                fe_solver -> output(time_step);
            }

            #ifdef CHECK_ACTIVATION_TIMES
            // we compute the activation time
            compute_activation_times(time);
            #endif
        }
        
        #ifdef CHECK_ACTIVATION_TIMES
        // we write the activation times to a file
        output_activation_times();
        #endif
    }



private:
    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition. ///////////////////////////////////////////////////////

    // Current time.
    double time;

    // Final time.
    double T;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree.
    unsigned int r;

    // Time step.
    double deltat;

    // Theta parameter of the theta method used be the FESolver.
    double fe_theta;

    // Theta parameter of the theta method used by the ODESolver.
    double ode_theta;

    //output folder
    std::string output_folder;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite Element solver.
    std::unique_ptr<FESolver> fe_solver;

    // Finite element space.
    std::shared_ptr<FiniteElement<dim>> fe;

    // Ionic model.
    std::shared_ptr<IonicModel<N_ion>> ionic_model;

    // Coupling of the monodomain and ionic models.
    std::shared_ptr<Coupler<N_ion>> coupler;

    // ODE Solver.
    ODESolver<N_ion> ode_solver;

    // Quadrature formula.
    std::shared_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    //time steps between two output files
    int os;



    void init() {
        // Create the mesh.
        {
            pcout << "Initializing the mesh " <<  std::endl;

            Triangulation<dim> mesh_serial;

            GridIn<dim> grid_in;
            grid_in.attach_triangulation(mesh_serial);

            std::ifstream grid_in_file(mesh_file_name);
            grid_in.read_msh(grid_in_file);

            GridTools::partition_triangulation(mpi_size, mesh_serial);
            const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
            mesh.create_triangulation(construction_data);

            pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
        }

        pcout << "-----------------------------------------------" << std::endl;

        // Initialize the finite element space.
        {
            pcout << "Initializing the finite element space" << std::endl;

            fe = std::make_shared<FE_SimplexP<dim>>(r);

            pcout << "  Degree                     = " << fe->degree << std::endl;
            pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

            quadrature = std::make_shared<QGaussSimplex<dim>>(r + 1);

            pcout << "  Quadrature points per cell = " << quadrature->size()  << std::endl;
        }

        pcout << "-----------------------------------------------" << std::endl;

        // Initialize the DoF handler.
        {
            pcout << "Initializing the DoF handler" << std::endl;

            dof_handler.reinit(mesh);
            dof_handler.distribute_dofs(*fe);

            pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
        }
    }

    #ifdef CHECK_ACTIVATION_TIMES
    void compute_activation_times(double time) {
        // provided current time, we retrieve the range of the solution owned by current process
      auto [first, last] = getFESolutionOwned().local_range();
      // loop over the solution and check if it has raised from its resting value (about −84 mV) to a positive value.
      for(size_t i=first; i<last; i++) {
          if(getFESolutionOwned()[i] > 0 && activation_times_owned[i] < 0.000001){
              // If It's > 0 we store the time in the corresponding position in activation_times_owned
              activation_times_owned[i] = time;
          }
      }
      //perform communication
      activation_times = activation_times_owned;
    }

    void activation_times_init() {
      IndexSet locally_relevant_dofs;
      IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      activation_times_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
      activation_times.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
      activation_times = activation_times_owned;
    }

    void output_activation_times(){
      DataOut<dim> data_out;
      data_out.add_data_vector(dof_handler, activation_times, "u");
      std::vector<unsigned int> partition_int(mesh.n_active_cells());
      GridTools::get_subdomain_association(mesh, partition_int);
      const Vector<double> partitioning(partition_int.begin(), partition_int.end());
      data_out.add_data_vector(partitioning, "partitioning");
      data_out.build_patches();
      data_out.write_vtu_with_pvtu_record(output_folder, "output_activation_times", 0, MPI_COMM_WORLD, 3);
    }

    TrilinosWrappers::MPI::Vector activation_times;
    TrilinosWrappers::MPI::Vector activation_times_owned;
    #endif

};

#endif
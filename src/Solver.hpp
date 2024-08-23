#ifndef SOLVER_H
#define SOLVER_H

#include "IonicModel.hpp"
#include "Coupler.hpp"
#include "utils.hpp"
#include "ODESolver.hpp"
#include "FESolver.hpp"
#include <deque>
#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing monodomain problem.
template<int K_ode, int K_ion, int N_ion>
class Solver {
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 3;

public:
    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    Solver(
            const std::string &mesh_file_name_,
            const unsigned int &r_,
            const double       &T_,
            const double       &deltat_,
            const double       &fe_theta_,
            const double       &ode_theta_,
            std::shared_ptr<IonicModel<K_ion, N_ion>> ionic_model_,
            std::shared_ptr<Coupler<K_ode, K_ion, N_ion>> coupler_,
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
    , mesh(MPI_COMM_WORLD)
    , ionic_model(ionic_model_)
    , coupler(coupler_)
    , ode_solver(ode_theta_, deltat_, ionic_model_)
    {

        init();
        fe_solver = std::make_unique<FESolver>(r_, T_, deltat_, fe_theta_, mesh, fe, quadrature, dof_handler, std::move(d_), std::move(I_app_));
        fe_solver->setup();
        fe_solver->setInitialSolution(std::move(u_0));
        coupler->setInitialGatingVariables(*this, std::move(gate_vars_0));
#ifdef CHECK_ACTIVATION_TIMES
        std::cout << "checking activation times" << std::endl;
        activation_times_init();
#endif

    }

    std::vector<double>& getLastSolution() {
        return u[u.size() - 1];
    }

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

    double& getExplicitCoefficient(int cell_index, int q) {
        return fe_solver->getExplicitCoefficient(cell_index,q);
    }

    double& getImplicitCoefficient(int cell_index, int q) {
        return fe_solver->getImplicitCoefficient(cell_index,q);
    }

    TrilinosWrappers::MPI::Vector& getFESolution() {
        return fe_solver->getSolution();
    }

    TrilinosWrappers::MPI::Vector& getFESolutionOwned() {
        return fe_solver->getSolutionOwned();
    }


    ODESolver<K_ode, K_ion, N_ion>& getOdeSolver() {
        return ode_solver;
    }

    std::unique_ptr<FESolver>& getFESolver() {
        return fe_solver;
    }

    auto & getGatingVars() {
        return gate_vars;
    }

    auto & getSol(int k) {
        return u[u.size() - k];
    }

    int getSolSize() {
        return u.size();
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

    void setODESolution(std::vector<GatingVariables<N_ion>>& sol) {
        gate_vars = sol;
        //std::cout << "ode sol length 1 = " << gate_vars.size() << std::endl;
    }

    void setODESolution(std::vector<GatingVariables<N_ion>>&& sol) {
        gate_vars = sol;
        //std::cout << "ode sol length 2 = " << gate_vars.size() << std::endl;
    }

    void setFESolution(const TrilinosWrappers::MPI::Vector& sol){
        std::vector<double> s;
        s.resize(sol.local_size());
        auto [first, last] = sol.local_range();
        int i = 0;
        for(unsigned int j = first; j < last; i++, j++){
            s[i] = sol[j];
        }
        u.emplace_front(s);
        if(K_ion < u.size())
            u.pop_back();
    }

    auto const & getSolQuadrature() { // never used???
        return u_quadrature;
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
            auto start2 = std::chrono::high_resolution_clock::now();
            coupler->solveOde(*this);
            auto stop2 = std::chrono::high_resolution_clock::now();
            //std::cout << "mpi rank " << mpi_rank << " ODE time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2).count() << std::endl;
            coupler->solveFE(*this, time);
            auto start1 = std::chrono::high_resolution_clock::now();
#ifndef CHECK_ACTIVATION_TIMES
            fe_solver->output(time_step);
#endif
            if(time_step % 100 == 0) {
                fe_solver -> output(time_step);
            }
            auto stop1 = std::chrono::high_resolution_clock::now();
            //std::cout << "mpi rank " << mpi_rank  << " output time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count() << " start time : " << std::chrono::time_point_cast<std::chrono::microseconds>(start1).time_since_epoch().count() << " stop time : " << std::chrono::time_point_cast<std::chrono::microseconds>(stop1).time_since_epoch().count()  << std::endl;
#ifdef CHECK_ACTIVATION_TIMES
            compute_activation_times(time);
#endif
        }
        //std::cout << "finishing" << std::endl;
        //for(auto& f : futures) {
        //    f.get();
        //}
#ifdef CHECK_ACTIVATION_TIMES
        output_activation_times();
        fe_solver->output(time_step);
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

    std::vector<double> u_quadrature; // never set ?????????

    std::deque<std::vector<double>> u;//defined over dof; remember push_front

    std::vector<GatingVariables<N_ion>> gate_vars;//defined according to coupling method

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

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite Element solver.
    std::unique_ptr<FESolver> fe_solver;

    // Finite element space.
    std::shared_ptr<FiniteElement<dim>> fe;

    // Ionic model.
    std::shared_ptr<IonicModel<K_ion,N_ion>> ionic_model;

    // Coupling of the monodomain and ionic models.
    std::shared_ptr<Coupler<K_ode, K_ion, N_ion>> coupler;

    // ODE Solver.
    ODESolver<K_ode, K_ion, N_ion> ode_solver;

    // Quadrature formula.
    std::shared_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;




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

      auto [first, last] = getFESolutionOwned().local_range();
      for(size_t i=first; i<last; i++) {
          if(getFESolutionOwned()[i] > 0 && activation_times_owned[i] < 0.000001){
              //std::cout << "updated activation time" << std::endl;
              activation_times_owned[i] = time;
          }
      }

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
      data_out.write_vtu_with_pvtu_record("./", "output_activation_times", 0, MPI_COMM_WORLD, 3);
  }

  TrilinosWrappers::MPI::Vector activation_times;
  TrilinosWrappers::MPI::Vector activation_times_owned;
#endif

};

#endif
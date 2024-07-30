#ifndef SOLVER_H
#define SOLVER_H

#include "IonicModel.hpp"
#include "Coupler.hpp"
#include "utils.hpp"
#include "ODESolver.hpp"
#include "FESolver.hpp"
#include <deque>
using namespace dealii;


template<int K_ode, int K_ion, int N_ion>
class Solver {
        static constexpr unsigned int dim = 3;

public:
    Solver( 
         const std::string &mesh_file_name_,
         const unsigned int &r_,
         const double       &T_,
         const double       &deltat_,
         const double       &fe_theta_,
         const double       &ode_theta_,
         std::shared_ptr<IonicModel<K_ion, N_ion>> ionic_model_,
         std::shared_ptr<Coupler<K_ode, K_ion, N_ion>> coupler_
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
            , fe_solver(mesh_file_name_, r_, T_, deltat_, fe_theta_)
            , ionic_model(ionic_model_)
            , coupler(coupler_)
            , ode_solver(ode_theta_, deltat_)
    {}

    void notify_FE_ready(TrilinosWrappers::MPI::Vector solution) {

    }

    void notify_ODE_ready() {
        
    }

    double& getImplicitCoefficient(int cell_index, int q)  {
        return implicit_coefficients[cell_index * quadrature->size() + q];
    }

    double& getExplicitCoefficient(int cell_index, int q)  {
        return explicit_coefficients[cell_index * quadrature->size() + q];
    }

    std::vector<double>& getLastSolution() {
        return u[u.size() - 1];
    }

   /* void update_ode_input() {
        u_ode.emplace_front(u[u.size() - 1]);
        if(v.size() >= K_ode) {
            u_ode.pop_back();
        }
    }*/

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



    /*auto getActiveCellIterators() const {
        return dof_handler.active_cell_iterators();
    }*/

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

    void setODESolution(std::vector<GatingVariables<N_ion>> sol) {
        gate_vars = sol;
    }

    void setFESolution(TrilinosWrappers::MPI::Vector& sol){
        std::vector<double> s;
        for(unsigned int i = 0; i < sol.size(); i++){
            s[i] = sol[i];
        }
        u.emplace_front(s);
        if(K_ion < u.size())
            u.pop_back();
    }

    auto const & getSolQuadrature() {
        return u_quadrature;
    }


    void solve() {
        time = 0.0;
        time_step = 0;
        unsigned k = 1;
        fe_solver.output(time_step);
        while(time < T) {
            //solve ode
            time += deltat;
            time_step++;
            ode_solver.solve(coupler.from_fe_to_ode(), gate_vars);
            coupler.from_ode_to_fe();
            fe_solver.solve(time, time_step);
            fe_solver.output(time_step);
        }
    }



private:
    const unsigned int mpi_size;

    const unsigned int mpi_rank;

    ConditionalOStream pcout;
    
    double T;

    const std::string mesh_file_name;

    unsigned int r;

    double deltat;

    double fe_theta;

    double ode_theta;



    parallel::fullydistributed::Triangulation<dim> mesh;

    FESolver<K_ode, K_ion, N_ion> fe_solver;

    std::shared_ptr<FiniteElement<dim>> fe;

    std::shared_ptr<IonicModel<K_ion,N_ion>> ionic_model;

    std::shared_ptr<Coupler<K_ode, K_ion, N_ion>> coupler;

    ODESolver<K_ode, K_ion, N_ion> ode_solver;

    std::shared_ptr<Quadrature<dim>> quadrature;

    
    

    


    DoFHandler<dim> dof_handler;

    double time;

    unsigned int time_step;

    std::vector<double> implicit_coefficients;

    std::vector<double> explicit_coefficients;

    std::vector<double> u_quadrature;

    std::deque<std::vector<double>> u;//defined over dof; remember push_front

    std::vector<GatingVariables<N_ion>> gate_vars;//defined according to coupling method

};

#endif
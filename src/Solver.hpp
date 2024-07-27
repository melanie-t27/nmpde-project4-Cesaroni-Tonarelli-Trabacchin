#ifndef SOLVER_H
#define SOLVER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


#include "IonicModel.hpp"
#include "Coupler.hpp"
#include <deque>



template<int K_ode, int K_ion, int N_ion>
class Solver {
        static constexpr unsigned int dim = 3;
public:

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

    std::vector<double>& getLastSolution() const {
        return sol[sol.size() - 1];
    }

    void update_ode_input() {
        u_ode.emplace_front(sol[sol.size() - 1]);
        if(v.size() >= K_ode) {
            u_ode.pop_back();
        }
    }

    std::unique_ptr<FiniteElement<dim>> getFiniteElement() {
        return fe;
    }

    std::unique_ptr<Quadrature<dim>> getQuadrature() {
        return quadrature;
    }

    DofHandler<dim>& getDofHandler() {
        return dof_handler;
    }

    parallel::fullydistributed::Triangulation<dim> getMesh() {
        return mesh;
    }



    /*auto getActiveCellIterators() const {
        return dof_handler.active_cell_iterators();
    }*/

    auto const & getGateingVars() {
        return gate_vars;
    }

    auto const & getSol(int k) {
        return u[u.size() - k];
    }

    int getSolSize() {
        return sol.size();
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

    auto const & getSolQuadrature() {
        return u_quadrature;
    }



    void solve() {
        time = 0.0;
        time_step = 0;
        unsigned k = 1;
        while(time < T) {
            //solve ode
            time += deltat;
            time_step++;
            ode_solver.solve(coupler.from_fe_to_ode(), gate_vars);
            coupler.from_ode_to_fe();
            fe_solver.solve(time, time_step);
        }
    }




private:
    std::unique_ptr<FiniteElement<dim>> fe;

    std::unique_ptr<Quadrature<dim>> quadrature;

    std::unique_ptr<IonicModel<K_ion>> ionic_model;

    std::unique_ptr<Coupler> coupler;

    std::unique_ptr<ODESolver> ode_solver;

    parallel::fullydistributed::Triangulation<dim> mesh;


    DoFHandler<dim> dof_handler;

    double deltat;

    double time;

    unsigned int time_step;

    double T;

    unsigned int r;

    double fe_theta;

    double ode_theta;









    std::vector<double> implicit_coefficients;

    std::vector<double> explicit_coefficients;

    std::vector<double> u_quadrature;






    std::deque<std::vector<double>> u;//defined over dof; remember push_front

    std::vector<GatingVariables<N_ion>> gate_vars;//defined according to coupling method

};

#endif
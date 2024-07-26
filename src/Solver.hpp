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
#include "FixedSizeDeque.hpp"
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

    auto getActiveCellIterators() const {
        return dof_handler.active_cell_iterators();
    }

    auto const & getOdeOutput() {
        return gate_ode_output;
    }

    auto const & getSol(int k) {
        return sol[sol.size() - k];
    }

    int getSolSize() {
        return sol.size();
    }

    auto getIonicModel() {
        return ionic_model;
    }




private:
    std::unique_ptr<FiniteElement<dim>> fe;

    std::unique_ptr<Quadrature<dim>> quadrature;

    std::unique_ptr<IonicModel<K_ion>> ionic_model;

    std::unique_ptr<Coupler> coupler;

    DoFHandler<dim> dof_handler;

    std::deque<std::vector<double>, K_ode> u_ode_input;

    std::deque<std::vector<GatingVariable<N_ion>>, K_ode> gate_ode_input;

    std::vector<GatingVariable<N_ion>> gate_ode_output;


    std::vector<double> implicit_coefficients;

    std::vector<double> explicit_coefficients;

    std::vector<std::vector<double>> sol;

};

#endif
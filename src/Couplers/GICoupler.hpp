#ifndef GICOUPLER_H
#define GICOUPLER_H

#include <vector>
#include "Coupler.hpp"
#include "../utils.hpp"
#include <cstring>
#include <chrono>

using namespace dealii;

// Class responsible for the implementation of the Gauss integration approach
template<int N_ion>
class GICoupler : public Coupler<N_ion> {

    static constexpr unsigned int dim = 3;

public:

    // In this case the solveOde method is different with respect to the other couplers, as we solve the system of
    // ODEs on each quadrature node. Therefore, we interpolate the FE Solution on quadrature nodes,
    // create its view and pass it as a parameter, together with the view of gate_vars (defined on quadrature nodes),
    // to the solve method of the ODESolver class

    void solveOde(Solver<N_ion>& solver) override {
        // we retrieve  and initialize all the objects needed in order to perform the interpolation of the FE solution
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        // interpolated_value will store the value of the solution interpolated on current quadrature node
        double interpolated_value = 0;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            for(unsigned int q = 0; q < n_q; q++) {
                interpolated_value = 0.0;
                // for each quadrature node we loop over dofs
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    // we interpolate the FE solution, which is defined on dofs, on current quadrature node
                    interpolated_value += solver.getFESolution()[local_index] * shape_value;
                }
                // store the value in the corresponding position of interpolated_u
                interpolated_u[cell->active_cell_index() * n_q + q] = interpolated_value;
            }
        }
        // solve the system of ODEs
        solver.getOdeSolver().solve(interpolated_u, gate_vars);
    }


    void solveFE(Solver<N_ion>& solver, double time) override {
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int n_q           = quadrature->size();

        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            for(unsigned int q = 0; q < n_q; q++) {
                // we store in vars the value of gating variables at current quadrature node
                GatingVariables<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = gate_vars[j][cell->active_cell_index() * n_q + q];
                }
                // store the value of I_ion at current quadrature node
                solver.getIonicCurrent(cell->active_cell_index(), q) = ionicModel->ionic_current(interpolated_u[cell->active_cell_index() * n_q + q], vars);
            }
        }
        // solve the linear system
        solver.getFESolver()->solve_time_step(time);
    }


    // This method is different compared to the other couplers as
    // the initial gating variables need to be interpolated on quadrature nodes
    void setInitialGatingVariables(Solver<N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
        // we retrieve all the objects that we need in order to perform interpolation
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        // interpolated_values will store the value of the initial gating variables interpolated on quadrature nodes
        std::array<double, N_ion> interpolated_values;
        IndexSet locally_relevant_dofs;
        IndexSet locally_owned_dofs = solver.getDofHandler().locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(solver.getDofHandler(), locally_relevant_dofs);
        memset(&interpolated_values, 0, sizeof(double) * (N_ion));
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars_temp;
        std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars_temp_owned;
        for(int i = 0; i < N_ion; i++) {
            gate_vars[i].resize(solver.getMesh().n_active_cells() * n_q);
        }
        interpolated_u.resize(solver.getMesh().n_active_cells() * n_q);
        // we initialize gate_vars_temp and gate_vars_temp_owned with the initial values (evaluated at dofs)
        for(int i = 0; i < N_ion; i++) {
            gate_vars_temp_owned[i].reinit(locally_owned_dofs, MPI_COMM_WORLD);
            gate_vars_temp[i].reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
            VectorTools::interpolate(solver.getDofHandler(), *gate_vars_0[i], gate_vars_temp_owned[i]);
            gate_vars_temp[i] = gate_vars_temp_owned[i];
        }

        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for(unsigned int q = 0; q < n_q; q++) {
                memset(&interpolated_values, 0, sizeof(double) * (N_ion));
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    for(int j = 0; j < N_ion; j++) {
                        // we interpolate initial gating variables at current quadrature node
                        interpolated_values[j] += gate_vars_temp[j][local_index] * shape_value;
                    }
                }
                // store in gate_vars
                for(int j = 0; j < N_ion; j++) {
                    gate_vars[j][cell->active_cell_index()*n_q + q] = interpolated_values[j];
                }
            }
        }
    }


    ~GICoupler(){}

private:
    std::vector<double> interpolated_u; // solution on quadrature nodes
    std::deque<std::vector<double>> history;//defined over quadrature nodes
    std::array<std::vector<double>, N_ion> gate_vars;//defined over quadrature nodes
};
#endif
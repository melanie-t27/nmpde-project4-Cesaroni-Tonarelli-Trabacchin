#ifndef GICOUPLER_H
#define GICOUPLER_H

#include <vector>
#include "Coupler.hpp"
#include "../utils.hpp"
#include "../profile.hpp"
#include "../VectorView.hpp"
#include <cstring>
#include <chrono>


using namespace dealii;

template<int N_ion>
class GICoupler : public Coupler<N_ion> {

    static constexpr unsigned int dim = 3;

public:

    void solveOde(Solver<N_ion>& solver) override {
        auto start1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        double interpolated_value = 0;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            for(unsigned int q = 0; q < n_q; q++) {
                interpolated_value = 0.0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    interpolated_value += solver.getFESolution()[local_index] * shape_value;
                }

                interpolated_u[cell->active_cell_index() * n_q + q] = interpolated_value;
            }
        }
        

        std::vector<VectorView<std::vector<double>>> gate_vars_views;
        for(int i = 0; i < N_ion; i++) {
            gate_vars_views.emplace_back(gate_vars[i], 0, gate_vars.size());
        }
        VectorView<std::vector<double>> sol_view(interpolated_u, 0, interpolated_u.size());
        auto stop1 = std::chrono::high_resolution_clock::now();
        profileData.interpolation += std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count();
        start1 = std::chrono::high_resolution_clock::now();
        solver.getOdeSolver().solve(sol_view, gate_vars_views);
        stop1 = std::chrono::high_resolution_clock::now();
        profileData.ode_solve += std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count();
    }


    void solveFE(Solver<N_ion>& solver, double time) override {
        auto start1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int n_q           = quadrature->size();

        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            for(unsigned int q = 0; q < n_q; q++) {
                GatingVariables<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = gate_vars[j][cell->active_cell_index() * n_q + q];
                }
                solver.getIonicCurrent(cell->active_cell_index(), q) = ionicModel->ionic_current(interpolated_u[cell->active_cell_index() * n_q + q], vars);
            }
        }
        auto stop1 = std::chrono::high_resolution_clock::now();
        profileData.interpolation += std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count();
        start1 = std::chrono::high_resolution_clock::now();
        solver.getFESolver()->solve_time_step(time);
        stop1 = std::chrono::high_resolution_clock::now();
        profileData.fe_solve_tot += std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count();
    }



    void setInitialGatingVariables(Solver<N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
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
                        interpolated_values[j] += gate_vars_temp[j][local_index] * shape_value;
                    }
                }
                for(int j = 0; j < N_ion; j++) {
                    gate_vars[j][cell->active_cell_index()*n_q + q] = interpolated_values[j];
                }
            }
        }
    }


    ~GICoupler(){}

private:
    std::vector<double> interpolated_u;
    std::deque<std::vector<double>> history;//defined over quadrature nodes
    std::array<std::vector<double>, N_ion> gate_vars;//defined over quadrature nodes
};
#endif
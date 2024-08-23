#ifndef QCOUPLER_H
#define QCOUPLER_H

#include <vector>
#include "Coupler.hpp"
//#include "Solver.hpp"
#include "utils.hpp"
#include "VectorView.hpp"
#include <cstring>
#include <chrono>

using namespace dealii;

template<int K_ode, int K_ion, int N_ion>
class QCoupler : public Coupler<K_ode, K_ion, N_ion> {

    static constexpr unsigned int dim = 3;
public:



    void solveOde(Solver<K_ode, K_ion, N_ion>& solver) override {
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<K_ion, N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        double interpolated_value = 0;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::array<double, K_ion> history_u;
        std::vector<double> new_history_item;
        new_history_item.resize(solver.getMesh().n_active_cells() * n_q);
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

                new_history_item[cell->active_cell_index()*n_q + q] = interpolated_value;
            }
        }
        history.push_front(new_history_item);
        if(history.size() > K_ion) {
            history.pop_back();
        }

        std::vector<VectorView<std::vector<double>>> gate_vars_views;
        for(int i = 0; i < N_ion; i++) {
            gate_vars_views.emplace_back(gate_vars[i], 0, gate_vars.size());
        }
        VectorView<std::vector<double>> sol_view(history[0], 0, history[0].size());
        solver.getOdeSolver().solve(sol_view, gate_vars_views);
    }


    void solveFE(Solver<K_ode, K_ion, N_ion>& solver, double time) override {
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<K_ion, N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int n_q           = quadrature->size();
        double interpolated_value = 0;
        std::array<double, K_ion> history_u;
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            for(unsigned int q = 0; q < n_q; q++) {
                for(int j = 0; j < K_ion ;j++) {
                    history_u[j] = history[j][cell->active_cell_index()*n_q + q];
                }
                GatingVariables<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = gate_vars[j][cell->active_cell_index()*n_q + q];
                }
                solver.getImplicitCoefficient(cell->active_cell_index(), q) = ionicModel->implicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
                solver.getExplicitCoefficient(cell->active_cell_index(), q) = ionicModel->explicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
            }
        }

        solver.getFESolver()->solve_time_step(time);
    }



    void setInitialGatingVariables(Solver<K_ode, K_ion, N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<K_ion, N_ion>> ionicModel = solver.getIonicModel();
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
        //std::cout << "finished phase 1" << std::endl;

        for(int i = 0; i < N_ion; i++) {
            gate_vars_temp_owned[i].reinit(locally_owned_dofs, MPI_COMM_WORLD);
            gate_vars_temp[i].reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
            VectorTools::interpolate(solver.getDofHandler(), *gate_vars_0[i], gate_vars_temp_owned[i]);
            gate_vars_temp[i] = gate_vars_temp_owned[i];
        }
        //std::cout << "finished phase 2" << std::endl;
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            //std::cout << "finished phase 3" << std::endl;
            cell->get_dof_indices(local_dof_indices);
            //std::cout << "finished phase 4" << std::endl;

            for(unsigned int q = 0; q < n_q; q++) {
                //std::cout << "finished phase 5" << std::endl;
                memset(&interpolated_values, 0, sizeof(double) * (N_ion));
                //std::cout << "finished phase 6" << std::endl;
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    //std::cout << "finished phase 7" << std::endl;
                    size_t local_index = local_dof_indices[i];
                    for(int j = 0; j < N_ion; j++) {
                        //std::cout << "finished phase 8" << std::endl;
                        interpolated_values[j] += gate_vars_temp[j][local_index] * shape_value;
                    }
                }
                for(int j = 0; j < N_ion; j++) {
                    gate_vars[j][cell->active_cell_index()*n_q + q] = interpolated_values[j];
                }
            }
        }
        //std::cout << "finished init" << std::endl;
    }


    ~QCoupler(){}

private:

    std::deque<std::vector<double>> history;//defined over quadrature nodes
    std::array<std::vector<double>, N_ion> gate_vars;//defined over quadrature nodes
};
#endif
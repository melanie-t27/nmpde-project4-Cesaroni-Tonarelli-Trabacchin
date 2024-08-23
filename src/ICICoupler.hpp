#ifndef ICICOUPLER_H
#define ICICOUPLER_H

#include <vector>
#include "Coupler.hpp"
//#include "Solver.hpp"
#include "utils.hpp"
#include "VectorView.hpp"
#include <cstring>
#include <chrono>

using namespace dealii;

template<int K_ode, int K_ion, int N_ion>
class ICICoupler : public Coupler<K_ode, K_ion, N_ion> {

    static constexpr unsigned int dim = 3;
public:



    void solveOde(Solver<K_ode, K_ion, N_ion>& solver) override {
        const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        std::vector<VectorView<TrilinosWrappers::MPI::Vector>> gate_vars_views;
        for(int i = 0; i < N_ion; i++) {
            auto [first, last] = gate_vars_owned[i].local_range();
            gate_vars_views.emplace_back(gate_vars_owned[i], first, last - first);
        }
        auto [first, last] = solver.getFESolutionOwned().local_range();
        VectorView<TrilinosWrappers::MPI::Vector> sol_view(solver.getFESolutionOwned(), first, last - first);
        solver.getOdeSolver().solve(sol_view, gate_vars_views);
        for(int i = 0; i < N_ion; i++) {
            gate_vars[i] = gate_vars_owned[i];
        }
    }

    void solveFE(Solver<K_ode, K_ion, N_ion>& solver, double time) override {
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<K_ion, N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        double interpolated_value_explicit = 0;
        double interpolated_value_implicit = 0;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::array<double, K_ion> history_u;
        bool done = false;
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for(unsigned int q = 0; q < n_q; q++) {

                interpolated_value_implicit = 0;
                interpolated_value_explicit = 0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    history_u[0] = solver.getFESolution()[local_index];
                    for(int j = 1; j < K_ion ;j++) {
                        history_u[j] = history[j-1][local_index];
                    }
                    GatingVariables<N_ion> vars;
                    for(int j = 0; j < N_ion; j++) {
                        vars.get(j) = gate_vars[j][local_index];
                    }

                    interpolated_value_implicit += ionicModel->implicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars) * shape_value;
                    interpolated_value_explicit += ionicModel->explicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars) * shape_value;
                }

                solver.getImplicitCoefficient(cell->active_cell_index(), q) = interpolated_value_implicit;
                solver.getExplicitCoefficient(cell->active_cell_index(), q) = interpolated_value_explicit;

            }
        }

        solver.getFESolver()->solve_time_step(time);
        std::vector<double> new_history_item;
        auto [first, last] = solver.getFESolution().local_range();
        new_history_item.resize(last - first);
        for(size_t i = first; i < last; i++) {
            new_history_item[i-first] = solver.getFESolution()[i];
        }
        history.push_front(new_history_item);
        if(history.size() > K_ion) {
            history.pop_back();
        }

    }



    void setInitialGatingVariables(Solver<K_ode, K_ion, N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
        IndexSet locally_relevant_dofs;
        IndexSet locally_owned_dofs = solver.getDofHandler().locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(solver.getDofHandler(), locally_relevant_dofs);
        for(int i = 0; i < N_ion; i++) {
            gate_vars_owned[i].reinit(locally_owned_dofs, MPI_COMM_WORLD);
            gate_vars[i].reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
            VectorTools::interpolate(solver.getDofHandler(), *gate_vars_0[i], gate_vars_owned[i]);
            gate_vars[i] = gate_vars_owned[i];
        }
    }


    ~ICICoupler(){}

private:

    std::deque<std::vector<double>> history;//defined over dofs
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars;//defined over dofs
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars_owned;//defined over dofs
};
#endif
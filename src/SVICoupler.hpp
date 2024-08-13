#ifndef SVICOUPLER_H
#define SVICOUPLER_H

#include <vector>
#include "Coupler.hpp"
//#include "Solver.hpp"
#include "utils.hpp"
#include "VectorView.hpp"
#include <cstring>
#include <chrono>

using namespace dealii;

template<int K_ode, int K_ion, int N_ion>
class SVICoupler : public Coupler<K_ode, K_ion, N_ion> {

    static constexpr unsigned int dim = 3; 
public:



    void solveOde(Solver<K_ode, K_ion, N_ion>& solver) override {
        const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        std::vector<VectorView<TrilinosWrappers::MPI::Vector>> gate_vars_views;
        for(int i = 0; i < N_ion; i++) {
            auto [first, last] = gate_vars_owned[i].local_range();
            gate_vars_views.emplace_back(gate_vars_owned[i], first, last - first);
        }
        auto [first, last] = solver.getFESolutionOwned().local_range();
        VectorView<TrilinosWrappers::MPI::Vector> sol_view(solver.getFESolutionOwned(), first, last - first);
        solver.getOdeSolver().solve(sol_view, gate_vars_views);
        auto stop1 = std::chrono::high_resolution_clock::now();
       // std::cout << "mpi rank " << mpi_rank << " size = " << last - first << " bare ode time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count() << " start time : " << std::chrono::time_point_cast<std::chrono::microseconds>(start1).time_since_epoch().count() << " stop time : " << std::chrono::time_point_cast<std::chrono::microseconds>(stop1).time_since_epoch().count()  << std::endl;        auto start2 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < N_ion; i++) {
            gate_vars[i] = gate_vars_owned[i];
        }
        auto stop2 = std::chrono::high_resolution_clock::now();
        //std::cout << "mpi rank " << mpi_rank << " size = " << last - first << " ode communication solve time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2).count() << " start time : " << std::chrono::time_point_cast<std::chrono::microseconds>(start2).time_since_epoch().count() << " stop time : " << std::chrono::time_point_cast<std::chrono::microseconds>(stop2).time_since_epoch().count()  << std::endl;
    }

    void solveFE(Solver<K_ode, K_ion, N_ion>& solver, double time) override {
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<K_ion, N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        std::array<double, N_ion + 1> interpolated_values;
        memset(&interpolated_values, 0, sizeof(double) * (N_ion + 1));
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::array<double, K_ion> history_u;
        std::vector<double> new_history_item;
        new_history_item.resize(solver.getMesh().n_active_cells() * n_q+1);
        bool done = false;
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for(unsigned int q = 0; q < n_q; q++) {
                
                memset(&interpolated_values, 0, sizeof(double) * (N_ion + 1));
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    for(int j = 0; j < N_ion; j++) {
                        interpolated_values[j] += gate_vars[j][local_index] * shape_value;
                    }
                    interpolated_values[N_ion] += solver.getFESolution()[local_index] * shape_value;
                }

                history_u[0] = interpolated_values[N_ion];
                new_history_item[cell->active_cell_index()*n_q + q] = interpolated_values[N_ion];

                for(int j = 1; j < K_ion ;j++) {
                    history_u[j] = history[j-1][cell->active_cell_index()*n_q + q];
                }

                GatingVariables<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = interpolated_values[j];
                }
                solver.getImplicitCoefficient(cell->active_cell_index(), q) = ionicModel->implicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
                solver.getExplicitCoefficient(cell->active_cell_index(), q) = ionicModel->explicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
                if(!done && fe_values.quadrature_point(q).norm_square() < 0.1) {
                    std::cout << "u = " << interpolated_values[N_ion] << std::endl;
                    for(int i = 0; i < N_ion; i++) {
                        std::cout << "gate var " << i << "  = " << gate_vars[i][local_dof_indices[0]] << std::endl;
                    }
                    std::cout << "ionic current = " << 0.01 * 140 * 85.7 * (solver.getImplicitCoefficient(cell->active_cell_index(), q) * (interpolated_values[N_ion] + 84.0)/85.7  + solver.getExplicitCoefficient(cell->active_cell_index(), q)) << std::endl;
                    std::cout << "ionic current FI = " << 0.01 * 140 * 85.7 * ionicModel->get_FI(interpolated_values[N_ion], vars) << std::endl;
                    std::cout << "ionic current SO = " << 0.01 * 140 * 85.7 * ionicModel->get_SO(interpolated_values[N_ion], vars) << std::endl;
                    std::cout << "ionic current SI = " << 0.01 * 140 * 85.7 * ionicModel->get_SI(interpolated_values[N_ion], vars) << std::endl;
                    done = true;
                }
                /*if(solver.getExplicitCoefficient(cell->active_cell_index(), q) != 0.0 && solver.getImplicitCoefficient(cell->active_cell_index(), q) != 0.0) {
                    std::cout << "ok" << std::endl;
                }

                if(solver.getExplicitCoefficient(cell->active_cell_index(), q) == 0.0 && solver.getImplicitCoefficient(cell->active_cell_index(), q) == 0.0 && history_u[0] > -57.0) {
                    std::cout << "wtf" << std::endl;
                }*/
            }
        }


        history.push_front(new_history_item);
        if(history.size() > K_ion) {
            history.pop_back();
        }

        solver.getFESolver()->solve_time_step(time);

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


    ~SVICoupler(){}

private:
    std::deque<std::vector<double>> history;// defined over quadrature nodes
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars;
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars_owned;
     

};
#endif
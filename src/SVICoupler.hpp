#ifndef SVICOUPLER_H
#define SVICOUPLER_H

#include <vector>
#include "Coupler.hpp"
//#include "Solver.hpp"
#include "utils.hpp"
#include <cstring>

using namespace dealii;

template<int K_ode, int K_ion, int N_ion>
class SVICoupler : public Coupler<K_ode, K_ion, N_ion> {

    static constexpr unsigned int dim = 3; 
public:

    std::vector<double>& from_fe_to_ode(Solver<K_ode, K_ion, N_ion>& solver) override {
        return solver.getSol(1);
    }

    void from_ode_to_fe(Solver<K_ode, K_ion, N_ion>& solver) override {
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
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);

            for(unsigned int q = 0; q < n_q; q++) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    //std::cout << "gating vars length = " << solver.getGatingVars().size() << std::endl;
                    //std::cout << "sol length = " << solver.getSol(1).size() << std::endl;
                    for(int j = 0; j < N_ion; j++) {
                        interpolated_values[j] += solver.getGatingVars()[local_index].get(j) * shape_value;
                    }
                    interpolated_values[N_ion] += solver.getSol(1)[local_index] * shape_value;
                }

                history_u[0] = interpolated_values[N_ion];
                new_history_item.at(cell->active_cell_index()*n_q + q) = interpolated_values[N_ion];

                for(int j = 1; j < K_ion ;j++) {
                    history_u[j] = history[j-1][cell->active_cell_index()*n_q + q];
                }

                GatingVariables<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = interpolated_values[j];
                }
                solver.getImplicitCoefficient(cell->active_cell_index(), q) = ionicModel->implicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
                solver.getExplicitCoefficient(cell->active_cell_index(), q) = ionicModel->explicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
            }
        }


        history.push_front(new_history_item);
        if(history.size() > K_ion) {
            history.pop_back();
        }

    }

    void setInitialGatingVariables(Solver<K_ode, K_ion, N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
        TrilinosWrappers::MPI::Vector tmp;
        tmp.reinit(solver.getDofHandler().locally_owned_dofs(), MPI_COMM_WORLD);

        std::vector<GatingVariables<N_ion>> gate_interp_0;
        gate_interp_0.resize(solver.getDofHandler().n_dofs());
        for(int i = 0; i < N_ion; i++) {
            VectorTools::interpolate(solver.getDofHandler(), *gate_vars_0[i], tmp);
            for(size_t j = 0; j < gate_interp_0.size(); j++){
                gate_interp_0[j].get(i) = tmp[j];
            }
        }
        //std::cout << "gate interp size = " << gate_interp_0.size() << std::endl;
        solver.setODESolution(gate_interp_0);
    }


    ~SVICoupler(){}

private:
    std::deque<std::vector<double>> history;// defined over quadrature nodes
};
#endif
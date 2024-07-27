#ifndef SVICOUPLER_H
#define SVICOUPLER_H

#include <vector>

#include "Coupler.hpp"
#include "Solver.hpp"
template<int K_ode, int K_ion, int N_ion>
class SVICoupler : public Coupler<K_ode, K_ion, N_ion> {
    static constexpr unsigned int dim = 3; 
public:
    std::vector<double>& from_fe_to_ode(Solver<K_ode, K_ion, N_ion>& solver) override {
        return solver.getSol();
    }
    void from_ode_to_fe(Solver<K_ode, K_ion, N_ion>& solver) override {
        std::unique_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::unique_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DofHandler<dim>& dofHandler = solver.getDofhandler();
        std::unique_ptr<IonicModel> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        std::array<double, N_ion + 1> interpolated_values;
        std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
        std::array<double, K_ion> history_u;
        cell->get_dof_indices(local_dof_indices);

        std::vector<double> new_history_item;
        new_history_item.resize(dofhandler.n_locally_owned_active_cells() * n_q);
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);

            for(unsigned int q = 0; q < n_q; q++) {
                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                {
                    for(int j = 0; j < N_ion; j++) {
                        interpolated_values[j] += solver.getGatingVars()[local_dof_indices[i]].get(j) * fe_values.shape_value(i, q);
                    }

                    interpolated_values[N_ion] += solver.getSol(0)[local_dof_indices[i]] * fe_values.shape_value(i, q);

                    
                }

                history_u[0] = interpolated_values[N_ion];
                new_history_item[cell->active_cell_index()*n_q + q] = interpolated_values[N_ion];

                for(int j = 1; j < K_ion ;j++) {
                    history_u[j] = history[j-1][cell->active_cell_index()*n_q + q];
                }



                GatingVariable<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = interpolated_values[j];
                }
                solver.getImplicitCoefficient(cell->active_cell_index(), q) = IonicModel->implicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
                solver.getExplicitCoefficient(cell->active_cell_index(), q) = IonicModel->explicit_coefficient(history_u, std::min(K_ion, solver.getSolSize()), vars);
            }
        }
        history.emplace_front(new_history_item);
        if(history.size()>=K_ion) {
            history.pop_back();
        }
    }
private:
    std::deque<std::vector<double>> history;// defined over quadrature nodes
};
#endif
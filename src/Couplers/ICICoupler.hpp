#ifndef ICICOUPLER_H
#define ICICOUPLER_H

#include <vector>
#include "Coupler.hpp"
#include "../utils.hpp"
#include "../VectorView.hpp"
#include <cstring>
#include <chrono>

using namespace dealii;
// Class responsible for the implementation of the ionic current interpolation approach
template<int N_ion>
class ICICoupler : public Coupler<N_ion> {
    static constexpr unsigned int dim = 3;

public:
    // In this case the solveOde method is identical to the one in SVICoupler.
    // Also in this case the gating variables are defined on dofs.
    void solveOde(Solver<N_ion>& solver) override {
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

    // In this case the I_ion will be evaluated at the gating variables and then interpolated on each
    // quadrature node
    void solveFE(Solver<N_ion>& solver, double time) override {
        // we retrieve and initialize all the elements needed in order to perform the interpolation of the I_ion
        // and the solution on quadrature nodes
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        // interpolates_ionic_current will store the value of I_ion interpolated at current quadrature node
        double interpolated_ionic_current = 0;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
       
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for(unsigned int q = 0; q < n_q; q++) {
                interpolated_ionic_current = 0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q);
                    size_t local_index = local_dof_indices[i];
                    // we store in vars the values of gating variables at current dof index
                    GatingVariables<N_ion> vars;
                    for(int j = 0; j < N_ion; j++) {
                        vars.get(j) = gate_vars[j][local_index];
                    }
                    // we evaluate I_ion at the gating variables and interpolate it on current quadrature node
                    interpolated_ionic_current += ionicModel->ionic_current(solver.getFESolution()[local_index], vars) * shape_value;
                }
                // we store in the corresponding position of IonicCurrents (member of FESolver class) the value of interpolated I_ion
                solver.getIonicCurrent(cell->active_cell_index(), q) = interpolated_ionic_current;
            }
        }
        // solve the linear system
        solver.getFESolver()->solve_time_step(time);
    }


   // This method is identical to the one in SVICoupler, we just initialize the gating variables
   // with the initial values evaluating them on dofs and storing the result in gate_vars_owned
    void setInitialGatingVariables(Solver<N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
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
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars;//defined over dofs
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars_owned;//defined over dofs

};
#endif
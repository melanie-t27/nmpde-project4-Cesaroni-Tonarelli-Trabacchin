#ifndef SVICOUPLER_H
#define SVICOUPLER_H

#include <vector>
#include "Coupler.hpp"
#include "../utils.hpp"
#include "../VectorView.hpp"
#include <cstring>
#include <chrono>

using namespace dealii;

//Class responsible for implementing state variable interpolation approach:
// both the potential variable and the gating variables
// are interpolated at the quadrature nodes of the mesh
template<int N_ion>
class SVICoupler : public Coupler<N_ion> {
    static constexpr unsigned int dim = 3;

public:
    void solveOde(Solver<N_ion>& solver) override {
        // We construct views of the gating variables for efficient access and manipulation
        std::vector<VectorView<TrilinosWrappers::MPI::Vector>> gate_vars_views;
        for(int i = 0; i < N_ion; i++) {
            auto [first, last] = gate_vars_owned[i].local_range();
            gate_vars_views.emplace_back(gate_vars_owned[i], first, last - first);
        }
        // We construct the view of the FE solution
        auto [first, last] = solver.getFESolutionOwned().local_range();
        VectorView<TrilinosWrappers::MPI::Vector> sol_view(solver.getFESolutionOwned(), first, last - first);
        // We solve the system of ODEs by calling the solve method, member of the OdeSolver class
        solver.getOdeSolver().solve(sol_view, gate_vars_views);
        // We perform the necessary communication
        for(int i = 0; i < N_ion; i++) {
            gate_vars[i] = gate_vars_owned[i];
        }
    }

    void solveFE(Solver<N_ion>& solver, double time) override {
        // we retrieve and initialize all the elements needed in order to perform the interpolation of the gating variables
        // and the solution on quadrature nodes
        std::shared_ptr<FiniteElement<dim>> fe = solver.getFiniteElement();
        std::shared_ptr<Quadrature<dim>> quadrature = solver.getQuadrature();
        FEValues<dim>  fe_values(*fe,*quadrature,update_values | update_gradients | update_quadrature_points | update_JxW_values);
        DoFHandler<dim>& dofHandler = solver.getDofHandler();
        std::shared_ptr<IonicModel<N_ion>> ionicModel = solver.getIonicModel();
        const unsigned int dofs_per_cell = fe->dofs_per_cell;
        const unsigned int n_q           = quadrature->size();
        // we declare an array to store the interpolated values:
        // - the first N_ion components will store the interpolated values of the gating variables
        //   on the current quadrature node
        // - the last component will store the interpolated value of the solution
        //   on the current quadrature node
        std::array<double, N_ion + 1> interpolated_values;
        memset(&interpolated_values, 0, sizeof(double) * (N_ion + 1));
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        // we loop over the elements and for each element we iterate over the quadrature nodes
        for(const auto &cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned())
                continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for(unsigned int q = 0; q < n_q; q++) {
                memset(&interpolated_values, 0, sizeof(double) * (N_ion + 1));
                // we iterate over dofs
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    double shape_value = fe_values.shape_value(i, q); // phi_i(x_q)
                    size_t local_index = local_dof_indices[i];
                    // we interpolate each gate variable, on the current quadrature node (gate_vars is defined on dofs)
                    for(int j = 0; j < N_ion; j++) {
                        interpolated_values[j] += gate_vars[j][local_index] * shape_value;
                    }
                    // we interpolate the FEsolution, on the current quadrature node (FESolution is defined on dofs)
                    interpolated_values[N_ion] += solver.getFESolution()[local_index] * shape_value;
                }

                // we store in vars the values of the gating variables interpolated on current quadrature node
                GatingVariables<N_ion> vars;
                for(int j = 0; j < N_ion; j++) {
                    vars.get(j) = interpolated_values[j];
                }
                // We store the value of Ion, evaluated at current quadrature node, in the corresponding
                // position in the IonicCurrents vector (member of the FESolver class).
                // getIonicCurrent provides access to the IonicCurrents vector at the position corresponding to the
                // current quadrature node
                solver.getIonicCurrent(cell->active_cell_index(), q) = ionicModel->ionic_current(interpolated_values[N_ion],  vars);
            }
        }
        // finally we can call the solve_time_step method, method of the FESolver class, which is responsible
        // for solving, at each time step, the assembled linear system
        solver.getFESolver()->solve_time_step(time);
    }

    // the method takes as input parameter also gate_vars_0, which stores the initial values of the gating variables
    // (members of the utils class).
    void setInitialGatingVariables(Solver<N_ion>& solver, std::array<std::unique_ptr<Function<dim>>, N_ion>  gate_vars_0) {
        IndexSet locally_relevant_dofs;
        IndexSet locally_owned_dofs = solver.getDofHandler().locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(solver.getDofHandler(), locally_relevant_dofs);
        for(int i = 0; i < N_ion; i++) {
            // we initialize and evaluate each gate variable at dofs
            gate_vars_owned[i].reinit(locally_owned_dofs, MPI_COMM_WORLD);
            gate_vars[i].reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
            VectorTools::interpolate(solver.getDofHandler(), *gate_vars_0[i], gate_vars_owned[i]);
            // perform necessary communication
            gate_vars[i] = gate_vars_owned[i];
        }
    }

    ~SVICoupler(){}

private:
    // gate_vars and gate_vars_owned are defined on dofs
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars;
    std::array<TrilinosWrappers::MPI::Vector, N_ion> gate_vars_owned;

};
#endif
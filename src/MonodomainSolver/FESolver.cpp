#include "FESolver.hpp"
#include "../utils.hpp"
#include "Solver.hpp"
#include <chrono>

using namespace dealii;

void FESolver::setup()
{
    // Initialize linear system.
    ionic_currents.resize(mesh.n_active_cells() * quadrature->size());
    pcout << "Initializing the linear system" << std::endl;
    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    assemble_matrices();
}



void FESolver::assemble_matrices() {
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system matrices" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix      = 0.0;
    stiffness_matrix = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_mass_matrix      = 0.0;
        cell_stiffness_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // Evaluate coefficients on this quadrature node.
            const typename TensorFunction<2, dim>::value_type D_loc = d->value(fe_values.quadrature_point(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_mass_matrix(i, j) += chi * C_m * fe_values.shape_value(i, q) *
                                              fe_values.shape_value(j, q) /
                                              deltat * fe_values.JxW(q);

                    cell_stiffness_matrix(i, j) +=
                            (D_loc * fe_values.shape_grad(i, q)) *
                            fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        mass_matrix.add(dof_indices, cell_mass_matrix);
        stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

    mass_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);

    // We build the matrix on the left-hand side of the algebraic problem (the one
    // that we'll invert at each timestep).
    lhs_matrix.copy_from(mass_matrix);
    lhs_matrix.add(theta, stiffness_matrix);

    // We build the matrix on the right-hand side (the one that multiplies the old
    // solution un).
    rhs_matrix.copy_from(mass_matrix);
    rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
}



void FESolver::assemble_rhs(const double time) {
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators()){
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q) {
            // We need to compute the forcing term at the current time (tn+1) and
            // at the old time (tn).

            // Compute Iapp(tn+1)
            I_app->set_time(time);
            const double I_app_new_loc =
                    I_app->value(fe_values.quadrature_point(q));

            // Compute Iapp(tn)
            I_app->set_time(time - deltat);
            const double I_app_old_loc = I_app->value(fe_values.quadrature_point(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) += (theta * I_app_new_loc + (1.0 - theta) * I_app_old_loc 
                                    -  chi * C_m * 85.7 * getIonicCurrent(cell->active_cell_index(), q)) *
                               fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

        cell->get_dof_indices(dof_indices);
        system_rhs.add(dof_indices, cell_rhs);
    }

    system_rhs.compress(VectorOperation::add);

    // Add the term that comes from the old solution.
    rhs_matrix.vmult_add(system_rhs, solution_owned);

}



TrilinosWrappers::MPI::Vector&
FESolver::solve_time_step(double time)
{
    I_app->set_time(time);
    assemble_rhs(time);
    SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver_ls(solver_control);
    TrilinosWrappers::PreconditionILU      preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionILU::AdditionalData(0,0.0,1.01,0));
    solver_ls.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);

    solution = solution_owned;
    return solution_owned;
}


void
FESolver::output(unsigned int time_step)
{
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "u");
    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(output_folder, "output", time_step, MPI_COMM_WORLD, 3);
}


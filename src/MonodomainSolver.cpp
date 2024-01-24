//
// Created by Melanie Tonarelli on 22/01/24.
//
#include "MonodomainSolver.h"

void
MonodomainSolver::setup(){
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        fe = std::make_unique<FE_SimplexP<dim>>(r);

        pcout << "  Degree                     = " << fe->degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        pcout << "  Initializing the sparsity pattern" << std::endl;

        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                                   MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);
        sparsity.compress();

        pcout << "  Initializing the matrices" << std::endl;
        jacobian_matrix.reinit(sparsity);

        pcout << "  Initializing the system right-hand side" << std::endl;
        residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        solution_old = solution;
    }
}

void
MonodomainSolver::assemble_matrices()
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system matrices" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_residual(dofs_per_cell);


    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    // Value and gradient of the solution on current cell.
    std::vector<double>         solution_loc(n_q);
    std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);

    // Value of the solution at previous timestep (un) on current cell.
    std::vector<double> solution_old_loc(n_q);

    std::vector<Tensor<1, dim>> solution_old_gradient_loc(n_q);


    mass_matrix      = 0.0;
    stiffness_matrix = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_mass_matrix      = 0.0;
        cell_stiffness_matrix = 0.0;
        cell_residual = 0.0;

        fe_values.get_function_values(solution, solution_loc);
        fe_values.get_function_gradients(solution, solution_gradient_loc);
        fe_values.get_function_gradients(solution_old, solution_old_gradient_loc);
        fe_values.get_function_values(solution_old, solution_old_loc);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // mass matrix
                    cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                              fe_values.shape_value(j, q) /
                                              deltat * fe_values.JxW(q);

                    // diffusion term
                    cell_stiffness_matrix(i, j) += diffusion_matrix * theta *
                                                    fe_values.shape_grad(j, q) *
                                                    fe_values.shape_grad(i, q) *
                                                    fe_values.JxW(q);

                    // J ion term
                    cell_stiffness_matrix(i,j) -= 1 / h *
                            (J_ion(theta*solution_loc[q] + 2 * theta*  h * fe_values.shape_value(j, q) + (1-theta) * solution_old_loc[q])
                                - J_ion(solution_loc[q] - 2 * theta* h * fe_values.shape_value(j, q) + (1-theta) * solution_old_loc[q]))
                            * fe_values.shape_value(i, q) / 2 * fe_values.JxW(q);
                }

                // Assemble the residual vector (with changed sign).

                // Time derivative term.
                cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                                    deltat * fe_values.shape_value(i, q) *
                                    fe_values.JxW(q);

                // Diffusion terms
                cell_residual(i) -= diffusion_matrix * theta * solution_gradient_loc[q]
                                        * fe_values.shape_grad(i, q) * fe_values.JxW(q);

                cell_residual(i) -= diffusion_matrix * (1-theta) * solution_old_gradient_loc[q]
                                        * fe_values.shape_grad(i, q) * fe_values.JxW(q);

                // J ion term
                cell_residual(i) += J_ion(theta*solution_loc[q] + (1-theta)*solution_old_loc[q])*fe_values.shape_value(i,q)
                                        *fe_values.JxW(q);

                // Forcing term
                j_app.set_time(getCurrentTime());
                cell_residual(i) += theta * j_app.value(fe_values.shape_value(i, q)) * fe_values.JxW(q);

                j_app.set_time(getCurrentTime() - deltat);
                cell_residual(i) += (1-theta) * j_app.value(fe_values.shape_value(i, q)) * fe_values.JxW(q);
                j_app.set_time(getCurrentTime());
            }
        }

        cell->get_dof_indices(dof_indices);

        mass_matrix.add(dof_indices, cell_mass_matrix);
        stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
        residual_vector.add(dof_indices, cell_residual);
    }
    mass_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);
    residual_vector.compress(VectorOperation::add);

    // We build the matrix on the left-hand side of the algebraic problem (the one
    // that we'll invert at each timestep).
    lhs_matrix.copy_from(mass_matrix);
    lhs_matrix.add(1, stiffness_matrix);


}
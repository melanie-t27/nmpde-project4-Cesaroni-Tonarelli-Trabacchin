#include "FESolver.hpp"
#include "utils.hpp"
#include "Solver.hpp"
#include <chrono>

using namespace dealii;

void FESolver::setup() 
{
    // Initialise linear system.
    implicit_coefficients.resize(mesh.n_active_cells() * quadrature->size());
    explicit_coefficients.resize(mesh.n_active_cells() * quadrature->size());
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
    Z_matrix.reinit(sparsity);

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



void FESolver::assemble_Z_matrix() {
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the Z matrix" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  FullMatrix<double> cell_Z_matrix(dofs_per_cell, dofs_per_cell);
  Z_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_Z_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              cell_Z_matrix(i,j) += chi * getImplicitCoefficient(cell->active_cell_index(), q) * 
                            fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
            }
        }
      }

    cell->get_dof_indices(dof_indices);

    Z_matrix.add(dof_indices, cell_Z_matrix);
  }

  Z_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta, stiffness_matrix);
  lhs_matrix.add(1.0, Z_matrix);
}

void FESolver::assemble_Z_matrix_on_one_cell(const auto& cell, ScratchData& scratch, PerTaskData& data){
    const unsigned int dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q           = scratch.fe_values.get_quadrature().size();
    scratch.fe_values.reinit(cell);
    data.cell_matrix = 0.0;
    for (unsigned int q = 0; q < n_q; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                data.cell_matrix(i,j) += chi * getImplicitCoefficient(cell->active_cell_index(), q) *
                                      scratch.fe_values.shape_value(i, q) * scratch.fe_values.shape_value(j, q) * scratch.fe_values.JxW(q);
            }
        }
    }

    cell->get_dof_indices(data.dof_indices);


}

void FESolver::copy_local_to_global(const PerTaskData& data){
    Z_matrix.add(data.dof_indices, data.cell_matrix);
}

void FESolver::assemble_Z_matrix_in_parallel(){
    ScratchData scratch_data(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    PerTaskData per_task_data(*fe);
    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), [this](const auto& cell, ScratchData& scratch, PerTaskData& data){this->assemble_Z_matrix_on_one_cell(cell, scratch, data);} ,[this](const PerTaskData& data){this->copy_local_to_global(data);}, scratch_data, per_task_data);
    Z_matrix.compress(VectorOperation::add);

    lhs_matrix.copy_from(mass_matrix);
    lhs_matrix.add(theta, stiffness_matrix);
    lhs_matrix.add(1.0, Z_matrix);
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
      // at the old time (tn). deal.II Functions can be computed at a
      // specific time by calling their set_time method.

      // Compute f(tn+1)
      I_app->set_time(time);//I_app is yet to be defined
      const double I_app_new_loc =
      I_app->value(fe_values.quadrature_point(q));

      // Compute f(tn)
      I_app->set_time(time - deltat);
      const double I_app_old_loc = I_app->value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        cell_rhs(i) += (theta * I_app_new_loc + (1.0 - theta) * I_app_old_loc -  chi * getExplicitCoefficient(cell->active_cell_index(), q)) *
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
  auto start1 = std::chrono::high_resolution_clock::now();
  assemble_Z_matrix_in_parallel();
  auto stop1 = std::chrono::high_resolution_clock::now();
  //std::cout << "mpi rank " << mpi_rank << " matrix Z assemble time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count() << std::endl;
  std::cout << "mpi rank " << mpi_rank << " matrix Z assemble time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count() << " start time : " << std::chrono::time_point_cast<std::chrono::microseconds>(start1).time_since_epoch().count() << " stop time : " << std::chrono::time_point_cast<std::chrono::microseconds>(stop1).time_since_epoch().count()  << std::endl;

  auto start2 = std::chrono::high_resolution_clock::now();
  assemble_rhs(time);
  auto stop2 = std::chrono::high_resolution_clock::now();
  std::cout << "mpi rank " << mpi_rank << " rhs assemble time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2).count() << std::endl;

  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver_ls(solver_control);
  TrilinosWrappers::PreconditionILU      preconditioner;
  preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionILU::AdditionalData(0,0.0,1.01,0));

  auto start3 = std::chrono::high_resolution_clock::now();
  solver_ls.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  auto stop3 = std::chrono::high_resolution_clock::now();
  std::cout << "mpi rank " << mpi_rank << " CG time : " << std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3).count() << std::endl;

  std::cout << "mpi rank " << mpi_rank << " " << solver_control.last_step() << " CG iterations" << std::endl;
  solution = solution_owned;
  return solution_owned;
}


std::future<void>
FESolver::output(unsigned int time_step)
{
  TrilinosWrappers::MPI::Vector solution_copy(solution);
  auto f = std::async(std::launch::async, &FESolver::parallelOutput, this, solution_copy , time_step);
  return f;
}

void
FESolver::parallelOutput(TrilinosWrappers::MPI::Vector solution_copy, unsigned int time_step) {
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution_copy, "u");
    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record("./", "output", time_step, MPI_COMM_WORLD, 3);
}
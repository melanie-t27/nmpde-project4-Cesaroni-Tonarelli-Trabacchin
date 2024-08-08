#ifndef FE_SOLVER_H
#define FE_SOLVER_H

#include "utils.hpp"

using namespace dealii;

class FESolver {
    // Physical dimension (1D, 2D, 3D)
    static constexpr int dim = 3;
public:
    FESolver(
            const unsigned int &r_,
            const double       &T_,
            const double       &deltat_,
            const double       &theta_,
            parallel::fullydistributed::Triangulation<dim>& mesh_, 
            std::shared_ptr<FiniteElement<dim>> fe_,
            std::shared_ptr<Quadrature<dim>> quadrature_, 
            DoFHandler<dim>& dof_handler_,
            std::unique_ptr<TensorFunction<2, dim>> d_,
            std::unique_ptr<Function<dim>> I_app_
            )
            : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
            , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
            , pcout(std::cout, mpi_rank == 0)
            , T(T_)
            , r(r_)
            , deltat(deltat_)
            , theta(theta_)
            , mesh(mesh_)
            , fe(fe_)
            , quadrature(quadrature_)
            , dof_handler(dof_handler_)
            , d(std::move(d_))
            , I_app(std::move(I_app_))
    {
        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    }

    // Initialization.
    void setup(); 

    // Assemble the mass and stiffness matrices.
    void assemble_matrices();

    // Assemble the right-hand side of the problem.
    void assemble_rhs(double time);

    // Assemble left-hand side of the problem.
    void assemble_Z_matrix();

    void assemble_Z_matrix_on_one_cell(const auto & cell, ScratchData& scratch, PerTaskData& data);

    void copy_local_to_global(const PerTaskData& data);

    void assemble_Z_matrix_in_parallel();

    // Solve the problem for one time step.
    TrilinosWrappers::MPI::Vector& solve_time_step(double time);

    // Output.
    void output(unsigned int time_step);

    double& getImplicitCoefficient(int cell_index, int q)  {
        return implicit_coefficients[cell_index * quadrature->size() + q];
    }

    double& getExplicitCoefficient(int cell_index, int q)  {
        return explicit_coefficients[cell_index * quadrature->size() + q];
    }

    TrilinosWrappers::MPI::Vector& getSolution() {
        return solution;
    }

    TrilinosWrappers::MPI::Vector& getSolutionOwned() {
        return solution_owned;
    }

    void setInitialSolution(std::unique_ptr<Function<dim>> u_0){
        VectorTools::interpolate(dof_handler, *u_0, solution_owned);
        solution = solution_owned;
    }
     void parallelOutput(TrilinosWrappers::MPI::Vector solution_copy, unsigned int time_step);
private:

    double chi = 140 /*1/mm*/ ;

    double C_m = 0.01 /*microF/(mm)^-2*/;
  

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
    const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim>& mesh;

  // Finite element space.
  std::shared_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::shared_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim>& dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix K.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand size Z.
  TrilinosWrappers::SparseMatrix Z_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;



  std::unique_ptr<TensorFunction<2, dim>> d;
  std::unique_ptr<Function<dim>> I_app;

  std::vector<double> implicit_coefficients; 

  std::vector<double> explicit_coefficients; 
};
#endif
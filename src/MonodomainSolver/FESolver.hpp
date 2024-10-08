#ifndef FE_SOLVER_H
#define FE_SOLVER_H

#include "../utils.hpp"

using namespace dealii;

class FESolver {
    // Physical dimension (1D, 2D, 3D)
    static constexpr int dim = 3;
public:
    // Constructor. We provide polynomial degree, final time, time step Delta t, theta method
    // parameter, mass lumping parameter, output_folder, mesh, fe, quadrature, dof_handler, tissue conductivity tensor, I_app.
    FESolver(
            const unsigned int &r_,
            const double       &T_,
            const double       &deltat_,
            const double       &theta_,
            std::string output_folder_,
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
            , output_folder(output_folder_)
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

    // Solve the problem for one time step.
    TrilinosWrappers::MPI::Vector& solve_time_step(double time);

    // Output.
    void output(unsigned int time_step);

    // method to return the value of Iion on current quadrature node,
    // provided current cell_index and current quadrature node
    double& getIonicCurrent(int cell_index, int q) {
        return ionic_currents[cell_index * quadrature->size() + q];
    }

    TrilinosWrappers::MPI::Vector& getSolution() {
        return solution;
    }

    TrilinosWrappers::MPI::Vector& getSolutionOwned() {
        return solution_owned;
    }

    // It takes initial condition, evaluates it on dofs and stores
    // in solution_owned
    void setInitialSolution(std::unique_ptr<Function<dim>> u_0){
        VectorTools::interpolate(dof_handler, *u_0, solution_owned);
        solution = solution_owned;
    }

private:

    // cellular surface-to-volume ratio
    double chi = 140000 /*1/mm*/ ;

    // membrane capacitance
    double C_m = 0.01 /*microF/(mm)^2*/;


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

    //output folder
    std::string output_folder;

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

    // tissue conductivity tensor
    std::unique_ptr<TensorFunction<2, dim>> d;

    //applied external stimulus current
    std::unique_ptr<Function<dim>> I_app;

    // vector storing Iion at each quadrature node
    std::vector<double> ionic_currents;

};
#endif
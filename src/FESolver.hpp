#ifndef FE_SOLVER_H
#define FE_SOLVER_H
//#include "Solver.hpp"
#include "utils.hpp"
using namespace dealii;

template<int K_ode, int K_ion, int N_ion>
class Solver;

template<int K_ode, int K_ion, int N_ion>
class FESolver {
    static constexpr int dim = 3;
public:
    FESolver( const std::string &mesh_file_name_,
            const unsigned int &r_,
            const double       &T_,
            const double       &deltat_,
            const double       &theta_)
            : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
            , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
            , pcout(std::cout, mpi_rank == 0)
            , T(T_)
            , mesh_file_name(mesh_file_name_)
            , r(r_)
            , deltat(deltat_)
            , theta(theta_)
            , mesh(MPI_COMM_WORLD)
    {}

    template <int dim>
    class D : public TensorFunction<2, dim>
    {
    public:
        D() : TensorFunction<2, dim>() {}
        void value_list(const std::vector<Point<dim>> &/*points*/,
                   std::vector<Tensor<2, dim>> & /*values*/) const override {} // do not use it

        typename TensorFunction<2, dim>::value_type value(const Point<dim> &/*p*/) const override {
            return unit_symmetric_tensor<dim>();
        }
    };

    class Iapp : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
            return 0.1;
        }
    };


    void setup();
    void assemble_matrices();
    void assemble_rhs(double time);
    void assemble_Z_matrix();
    void solve_time_step(double time, unsigned int time_step);
    void output(const unsigned int &time_step) const;
    //void solve();

private:
    D<dim> d;
    Iapp I_app;

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;


    // Final time.
    const double T;

    const std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;

    // Theta parameter of the theta method.
    const double theta;


    std::shared_ptr<Solver<K_ode, K_ion, N_ion>> solver;


  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix K.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

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
};
#endif
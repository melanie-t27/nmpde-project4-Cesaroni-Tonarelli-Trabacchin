//
// Created by Melanie Tonarelli on 22/01/24.
//

#ifndef NMPDE_PROJECT4_CESARONI_TONARELLI_TRABACCHIN_MONODOMAINSOLVER_H
#define NMPDE_PROJECT4_CESARONI_TONARELLI_TRABACCHIN_MONODOMAINSOLVER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class MonodomainSolver{
public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 3;

    // Diffusion Matrix
    class DiffusionMatrix: public TensorFunction<2, dim, double>
    {
    public:
        DiffusionMatrix(){
            for(unsigned i = 0; i < dim; i++){
                for (unsigned j = 0; j < dim; j++){
                    if( i == j )
                        _tensor[i][j] = 1;
                    else _tensor[i][j] = 0;
                }
            }
        }

        virtual double
        value(const Point<dim> &/*p*/) const override {
            return _tensor;
        }
    private:
        Tensor<2,dim> _tensor;
    };

    // J app
    class J_app: public Function<dim>{
    public:
        virtual double
        value(const Point<dim> &/*p*/,
              const unsigned int /*component*/ = 0) const override
        {
            if(getCurrentTime() < 0.001)
                return 1.0;
            else return 0.0;
        }
    };

    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    MonodomainSolver(const std::string  &mesh_file_name_,
         const unsigned int &r_,
         const double       &T_,
         const double       &deltat_,
         const double       &theta_)
            : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
            , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
            , pcout(std::cout, mpi_rank == 0)
            , mesh_file_name(mesh_file_name_)
            , r(r_)
            , deltat(deltat_)
            , theta(theta_)
            , mesh(MPI_COMM_WORLD)
    {}

    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

    // Set current time
    void
    setCurrentTime(double _time){
        time = _time;
    }

    // Get current time
    double
    getCurrentTime(){
        return time;
    }

    // Set current time
    void
    setCurrentTime(double _time){
        time = _time;
    }

    // Get current time
    double
    getCurrentTime() const{
        return time;
    }
    // Set current time
    void
    setStateVariables(const std::vector<double>& _w){
        w = _w;
    }

    // Get current time
    std::vector<double>
    getStateVariables() const{
        return w;
    }

protected:
    // Assemble the mass and stiffness matrices.
    void
    assemble_matrices();

    // Assemble the right-hand side of the problem.
    void
    assemble_rhs(const double &time);

    // Solve the problem for one time step.
    void
    solve_time_step();

    // Output.
    void
    output(const unsigned int &time_step) const;

    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition. ///////////////////////////////////////////////////////

    // Diffusion Matrix
    DiffusionMatrix diffusion_matrix;

    // State Variable
    std::vector<double> w;

    // J_app
    J_app j_app;

    // Constant variables
    const double u_0 = 0;

    const double u_u = 1.55;

    const double theta_v = 0.3;

    const double theta_w = 0.13;

    const double theta_v_minus = 0.006;

    const double theta_0 = 0.006;

    const double tau_v1_minus = 60;

    const double tau_v2_minus = 1150;

    const double tau_v_plus = 1.4506;

    const double tau_w1_minus = 60;

    const double tau_w2_minus = 15;

    const double k_w_minus = 65;

    const double u_w_minus = 0.03;

    const double tau_w_plus = 200;

    const double tau_fi = 0.11;

    const double tau_01 = 400;

    const double tau_02 = 6;

    const double tau_so_1 = 30.0181;

    const double tau_so_2 = 0.9957;

    const double k_so = 2.0458;

    const double u_so = 0.65;

    const double tau_s1 = 2.7342;

    const double tau_s2 = 16;

    const double k_s = 2.0994;

    const double u_s = 0.9087;

    const double tau_si = 1.8875;

    const double tau_w_inf = 0.07;

    const double w_inf_star = 0.94;

    const double h = 1e-10;


    // Heaviside function
    double H(double x){
        if (x < 0){
            return 0;
        } else {
            return 1;
        }
    }

    // Tau v minus
    double tau_v_minus(double u){
        return (1 - H(u-theta_v_minus)) * tau_v1_minus + H(u - theta_v_minus) * tau_v2_minus;
    }

    // Tau w minus
    double tau_v_minus(double u){
        return tau_w1_minus + (tau_w2_minus - tau_w1_minus) * (1 + std::tanh(k_w_minus * (u - u_w_minus))) / 2;
    }

    // Tau so
    double tau_so(double u){
        return tau_so_1 + (tau_so_2 + tau_so_1) * (1 + std::tanh(k_so * (u - u_so))) / 2;
    }

    // Tau s
    double tau_s(double u){
        return (1 - H(u - theta_v)) * tau_s1 + H(u - theta_w) * tau_s2;
    }

    // Tau o
    double tau_o(double u){
        return (1 - H(u - theta_0)) * tau_01 + H(u - theta_0) * tau_02;
    }

    // v infinity
    double v_inf(double u){
        if(u < theta_v_minus)
            return 1;
        else return 0;
    }

    // w infinity
    double w_inf(double u){
        return (1 - H(u - theta_0)) * (1 - u / tau_w_inf) + H(u - theta_0) * w_inf_star;
    }

    // J fi
    double J_fi(double u){
        return - w[0] * H(u - theta_v) * (u - theta_v) * (u_u - u) / tau_fi;
    }

    // J so
    double J_so(double u){
        return (u - u_0) * (1 - H(u - theta_w)) / tau_o(u) + H(u - theta_w) / t_so(u);
    }

    // J si
    double J_si(double u){
        return - H(u - theta_w) * w[1] * w[2] / tau_si;
    }

    // J ion
    double J_ion(double u){
        return J_si(u) + J_so(u) * J_fi(u);
    }

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;

    // Current time
    double time;

    // Theta parameter of the theta method.
    const double theta;

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

    // Stiffness matrix A.
    TrilinosWrappers::SparseMatrix stiffness_matrix;

    // Matrix on the left-hand side (M / deltat + theta A).
    TrilinosWrappers::SparseMatrix lhs_matrix;

    // Matrix on the right-hand side (M / deltat - (1 - theta) A).
    TrilinosWrappers::SparseMatrix rhs_matrix;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::Vector residual_vector;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::Vector solution;

    // System solution at previous time step.
    TrilinosWrappers::MPI::Vector solution_old;

};





#endif //NMPDE_PROJECT4_CESARONI_TONARELLI_TRABACCHIN_MONODOMAINSOLVER_H

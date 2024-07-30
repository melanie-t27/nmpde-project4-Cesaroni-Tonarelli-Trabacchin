#ifndef COUPLER_H
#define COUPLER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

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
#include <vector>
//#include "Solver.hpp"
#include "utils.hpp"

using namespace dealii;
template<int K_ode, int K_ion, int N_ion>
class Solver;

template<int K_ode, int K_ion, int N_ion>
class Coupler {
public:
    virtual std::vector<double>& from_fe_to_ode(std::shared_ptr<Solver<K_ode, K_ion, N_ion>> solver);
    virtual void from_ode_to_fe(std::shared_ptr<Solver<K_ode, K_ion, N_ion>> solver);

    virtual ~Coupler() {

    }
};
#endif
#ifndef UTILS_H
#define UTILS_H
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

using namespace dealii;

template<int N>
struct GatingVariables {
    double& get(int k) {
        //assert(k < N);
        return var[k];
    }
private:
    double var[N];
};

template <int dim>
class D : public TensorFunction<2, dim>
{
    public:
        D() : TensorFunction<2, dim>() {}
        void value_list(const std::vector<Point<dim>> &/*points*/, std::vector<Tensor<2, dim>> & /*values*/) const override {} // do not use it

        typename TensorFunction<2, dim>::value_type value(const Point<dim> &/*p*/) const override {
            return unit_symmetric_tensor<dim>()/100.0;
        }
};

template <int dim>
class Iapp : public Function<dim>
{
public:
    virtual double
    value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override
    {
        
        if(p[0] < 0.1 && p[1] < 0.1 && p[2] < 0.1) {
            return 0.1;
        }
        return 0.0;
    }
};

template <int dim>
class U_0 : public Function<dim> {
    public:
        virtual double
        value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }
};

template <int dim>
class GatingVariable_V0 : public Function<dim> {
    public:
        virtual double
        value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }
};

template <int dim>
class GatingVariable_W0 : public Function<dim> {
    public:
        virtual double
        value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }
};

template <int dim>
class GatingVariable_S0 : public Function<dim> {
    public:
        virtual double
        value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }
};


#endif
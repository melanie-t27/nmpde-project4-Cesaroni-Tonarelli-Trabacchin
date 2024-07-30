#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H
#include <array>
#include <vector>
#include "utils.hpp"
//#include "Solver.hpp"
template<int K_ode, int K_ion, int N_ion>
class Solver;

template<int K_ode, int K_ion, int N_ion>
class ODESolver {
public:

    ODESolver(double theta_, double deltat_) :
        theta(theta_),
        deltat(deltat_)
    {}


    void solve(const std::vector<double>& u, const std::vector<GatingVariables<N_ion>>& vars, std::shared_ptr<Solver<K_ode, K_ion, N_ion>>  solver) {
        int size = u.size();
        std::vector<GatingVariables<N_ion>> new_vars;
        new_vars.resize(size);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < N_ion; j++) {
                auto [impl, expl] = solver->getIonicModel().getExpansionCoefficients(j, u[i]);
                new_vars[i].get(j) = (u[i]/deltat + theta*expl + (1-theta)*vars[i].get(j))/(1/deltat - theta*impl);
            }
        }
        solver->setODESolution(new_vars);
    }

    
private:
    double theta;
    double deltat;

};
#endif
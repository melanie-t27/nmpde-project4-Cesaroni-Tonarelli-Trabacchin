#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H
#include <array>
#include <vector>
template<int K_ode, int K_ion, int N_ion>
class ODESOlver {
public:
    void solve(const std::vector<double>& u, const std::vector<GatingVariable<N_ion>>& vars) {
        int size = u.size();
        std::vector<GatingVariables<N_ion>> new_vars;
        new_vars.resize(size);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < N_ion; j++) {
                auto [impl, expl] = solver.getIonicModel().getExpansionCoefficients(j, u[i]);
                new_vars[i].get(j) = (u[i]/deltat + theta*expl + (1-theta)*vars[i].get(j))/(1/deltat - theta*impl);
            }
        }
        solver.setODESolution(new_vars);
    }
private:
    double theta;
    double deltat;
    Solver<K_ode, K_ion, N_ion>& solver;

};
#endif
#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H
#include <array>
#include <vector>
template<int N_ion, int K_ode, int K_ion>
class ODESOlver {
public:
    virtual double getNext(const std::array<double, K_ode>& u, const std::array<GatingVariable<N_ion>, K_ode>& vars, size_t valid, IonicModel<K_ode, N_ion>& ionicModel);
    double solve(Solver<K_ode, K_ion, N_ion>& solver, const std::vector<double>6 u, const std::vector<GatingVariable<N_ion>>& vars, double h, const IonicModel<K_ode, N_ion>& ionicModel) {
        
    }

};
#endif
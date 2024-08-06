#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H
#include <array>
#include <vector>
#include "utils.hpp"

template<int K_ode, int K_ion, int N_ion>
class ODESolver {
public:

    ODESolver(double theta_, double deltat_, std::shared_ptr<IonicModel<K_ion, N_ion>> ionic_model_) :
        theta(theta_),
        deltat(deltat_),
        ionic_model(ionic_model_)
    {}


    std::vector<GatingVariables<N_ion>> solve(std::vector<double>& u, std::vector<GatingVariables<N_ion>>& vars) {
        int size = u.size();
        std::vector<GatingVariables<N_ion>> new_vars;
        new_vars.resize(size);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < N_ion; j++) {
                auto [impl, expl] = ionic_model->getExpansionCoefficients(j, u[i]);
                new_vars[i].get(j) = (vars[i].get(j)*(1/deltat + (1 - theta)*impl) + expl)/(1/deltat - theta*impl);
            }
        }
        return new_vars;
    }

    
private:
    double theta;
    double deltat;
    std::shared_ptr<IonicModel<K_ion, N_ion>> ionic_model;

};
#endif
#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H
#include <array>
#include <vector>
#include "utils.hpp"
#include "ODESolver.hpp"
#include "VectorView.hpp"

template<int K_ode, int K_ion, int N_ion>
class ODESolver {
public:

    ODESolver(double theta_, double deltat_, std::shared_ptr<IonicModel<K_ion, N_ion>> ionic_model_) :
        theta(theta_),
        deltat(deltat_),
        ionic_model(ionic_model_)
    {}

    template<typename VectorType>
    void solve(VectorView<VectorType>& u, std::vector<VectorView<VectorType>>& vars) {
        for(size_t i = 0; i < u.size(); i++) {
            for(size_t j = 0; j < N_ion; j++) {
                auto [impl, expl] = ionic_model->getExpansionCoefficients(j, u.get(i));
                //vars[j].set(i, (vars[j].get(i) * (1 + deltat*(1 - theta)*impl) + deltat * expl)/(1 - deltat*theta*impl));
                vars[j][i] = (vars[j][i] * (1 + deltat*(1 - theta)*impl) + deltat * expl)/(1 - deltat*theta*impl);
            }
        }
    }

    
private:
    double theta;
    double deltat;
    std::shared_ptr<IonicModel<K_ion, N_ion>> ionic_model;

};
#endif
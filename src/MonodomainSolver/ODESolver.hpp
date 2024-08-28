#ifndef ODE_SOLVER_H
#define ODE_SOLVER_H
#include <array>
#include <vector>
#include "../utils.hpp"
#include "../VectorView.hpp"

// class responsible for solving the system of ODEs
template<int N_ion>
class ODESolver {
public:
    // Constructor. We provide theta method parameter and the ionic model.
    ODESolver(double theta_, double deltat_, std::shared_ptr<IonicModel<N_ion>> ionic_model_) :
            theta(theta_),
            deltat(deltat_),
            ionic_model(ionic_model_)
    {}

    // the method takes in input a view of the FE Solution and a view of the gating variables
    template<typename VectorType>
    void solve(VectorView<VectorType>& u, std::vector<VectorView<VectorType>>& vars) {
        for(size_t i = 0; i < u.size(); i++) {
            for(size_t j = 0; j < N_ion; j++) {
                auto [impl, expl] = ionic_model->getExpansionCoefficients(j, u.get(i));
                vars[j].set(i, (vars[j][i] * (1 + deltat * (1 - theta) * impl) + deltat * expl)/(1 - deltat * theta * impl));
            }
        }
    }


private:
    double theta;
    double deltat;
    std::shared_ptr<IonicModel<N_ion>> ionic_model;

};
#endif
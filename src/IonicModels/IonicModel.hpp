#ifndef IONIC_MODEL_H
#define IONIC_MODEL_H
#include <array>
#include "../utils.hpp"

template<int N_ion>
class IonicModel {
public:

    virtual double getDerivative(int index, double u, GatingVariables<N_ion> vars);

    virtual double ionic_current(const double u, GatingVariables<N_ion>& vars);

    virtual std::tuple<double, double> getExpansionCoefficients(int index, double u);

    virtual double get_FI(double u, GatingVariables<N_ion>& vars) = 0;

    virtual double get_SO(double u, GatingVariables<N_ion>& vars) = 0;

    virtual double get_SI(double u, GatingVariables<N_ion>& vars) = 0;
    
    virtual ~IonicModel() {}
};
#endif
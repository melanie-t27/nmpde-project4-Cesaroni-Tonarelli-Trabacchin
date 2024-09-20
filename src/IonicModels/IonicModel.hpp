#ifndef IONIC_MODEL_H
#define IONIC_MODEL_H
#include <array>
#include "../utils.hpp"

// Abstract class representing the ionic model, from which the BuenoOrovioModel class inherits
template<int N_ion>
class IonicModel {
public:

    virtual double getDerivative(int index, double u, GatingVariables<N_ion> vars);

    virtual double ionic_current(const double u, GatingVariables<N_ion>& vars);

    virtual std::tuple<double, double> getExpansionCoefficients(int index, double u);

    virtual ~IonicModel() {}
};
#endif
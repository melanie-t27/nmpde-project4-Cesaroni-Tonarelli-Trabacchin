#ifndef IONIC_MODEL_H
#define IONIC_MODEL_H
#include <array>
#include "utils.hpp"
template<int K_ion, int N_ion>
class IonicModel {
public:
    /*virtual double dv(double u, double v, double w, double s);
    virtual double dw(double u, double v, double w, double s);
    virtual double ds(double u, double v, double w, double s);*/
    virtual double getDerivative(int index, double u, GatingVariables<N_ion> vars);

    virtual double implicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, GatingVariables<N_ion>& vars);
    virtual double explicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, GatingVariables<N_ion>& vars);
    virtual std::tuple<double, double> getExpansionCoefficients(int index, double u);

    /*virtual double implicit_coefficient(std::array<double, K> u, double v, double w, double s);
    virtual double explicit_coefficient(std::array<double, K> u, double v, double w, double s);*/

    virtual ~IonicModel();
};
#endif
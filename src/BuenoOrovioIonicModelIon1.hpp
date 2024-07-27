#ifndef BUENO_OROVIO_IONIC_MODEL_ION1_H
#define BUENO_OROVIO_IONIC_MODEL_ION1_H
#include "BuenoOrovioIonicModel.hpp"
template<int K>
class BuenoOrovioIonicModelIon1 : public BuenoOrovioIonicModel<K> {
public:
    //u(i) = u_k-i
    double implicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, const GatingVariables<N>& vars) override {
        return -(vars.get(0)*H(u[0] - theta_v)*(u_u - u[0]))/tau_fi + (1 - H(u[0] - theta_w))/tau_o(u[0]);
    }
    double explicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, const GatingVariables<N>& vars) override {
        return -(theta_v * vars.get(0) * H(u[0] - theta_v) * (u_u - u[0]))/tau_fi - u_0 / tau_o(u[0]) * (1 - H(u[0] - theta_w)) + 1/tau_so(u[0]) * H(u[0] - theta_w) - H(u[0] - theta_w)*vars.get(1)*vars.get(2)/tau_si;
    }

};

#endif
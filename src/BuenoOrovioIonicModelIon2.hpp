#ifndef BUENO_OROVIO_IONIC_MODEL_ION2_H
#define BUENO_OROVIO_IONIC_MODEL_ION2_H
#include "BuenoOrovioIonicModel.hpp"
#include "utils.hpp"

template<int K_ion, int N_ion>
class BuenoOrovioIonicModelIon2 : public BuenoOrovioIonicModel<K_ion, N_ion> {
public:
    //u(i) = u_k-i
    double implicit_coefficient(const std::array<double, K_ion>& u, size_t /*valid_u*/, GatingVariables<N_ion>& vars) override {
        double u_adim = this->getAdimensionalU(u[0]);
        return (1 - this->H(u_adim - this->theta_w))/this->tau_o(u_adim)
               + vars.get(0) / this->tau_fi * this->H(u_adim - this->theta_v) * (u_adim - this->theta_v);
    }

    double explicit_coefficient(const std::array<double, K_ion>& u, size_t /*valid_u*/, GatingVariables<N_ion>& vars) override {
        double u_adim = this->getAdimensionalU(u[0]);
        return -this->u_u * vars.get(0) / this->tau_fi  * this->H(u_adim - this->theta_v) * (u_adim - this->theta_v) -
               this->u_0 * (1 - this->H(u_adim - this->theta_w))/this->tau_o(u_adim) + this->H(u_adim - this->theta_w) / this->tau_so(u_adim) -
               this->H(u_adim - this->theta_w) * vars.get(1) * vars.get(2) / this->tau_si;
    }

    ~BuenoOrovioIonicModelIon2() override{}

};

#endif
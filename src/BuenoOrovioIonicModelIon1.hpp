#ifndef BUENO_OROVIO_IONIC_MODEL_ION1_H
#define BUENO_OROVIO_IONIC_MODEL_ION1_H
#include "BuenoOrovioIonicModel.hpp"
#include "utils.hpp"

template<int K_ion, int N_ion>
class BuenoOrovioIonicModelIon1 : public BuenoOrovioIonicModel<K_ion, N_ion> {
public:
    //u(i) = u_k-i
    double implicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, GatingVariables<N_ion>& vars) override {
        return -(vars.get(0) * this->H(u[0] - this->theta_v) * (this->u_u - u[0])) / this->tau_fi 
                    + (1 - this->H(u[0] - this->theta_w)) / this->tau_o(u[0]);
    }

    double explicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, GatingVariables<N_ion>& vars) override {
        return -(this->theta_v * vars.get(0) * this->H(u[0] - this->theta_v) * (this->u_u - u[0])) / this->tau_fi 
                - this->u_0 / this->tau_o(u[0]) * (1 - this->H(u[0] - this->theta_w)) 
                + 1 / this->tau_so(u[0]) * this->H(u[0] - this->theta_w) 
                - this->H(u[0] - this->theta_w) * vars.get(1) * vars.get(2) / this->tau_si;
    }

    ~BuenoOrovioIonicModelIon1() override{}

};

#endif
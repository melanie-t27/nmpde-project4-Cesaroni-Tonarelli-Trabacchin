#ifndef BUENO_OROVIO_IONIC_MODEL_ION1_H
#define BUENO_OROVIO_IONIC_MODEL_ION1_H
#include "BuenoOrovioIonicModel.hpp"
#include "utils.hpp"

template<int K_ion, int N_ion>
class BuenoOrovioIonicModelIon1 : public BuenoOrovioIonicModel<K_ion, N_ion> {
public:
    //u(i) = u_k-i
    double implicit_coefficient(const std::array<double, K_ion>& u, size_t /*valid_u*/, GatingVariables<N_ion>& vars) override {
        double u_adim = this->getAdimensionalU(u[0]);
        return -(vars.get(0) * this->H(u_adim - this->theta_v) * (this->u_u - u_adim)) / this->tau_fi 
                    + (1 - this->H(u_adim - this->theta_w)) / this->tau_o(u_adim);
    }

    double explicit_coefficient(const std::array<double, K_ion>& u, size_t /*valid_u*/, GatingVariables<N_ion>& vars) override {
        double u_adim = this->getAdimensionalU(u[0]);
        return +(this->theta_v * vars.get(0) * this->H(u_adim - this->theta_v) * (this->u_u - u_adim)) / this->tau_fi 
                - this->u_0 / this->tau_o(u_adim) * (1 - this->H(u_adim - this->theta_w)) 
                + 1 / this->tau_so(u_adim) * this->H(u_adim - this->theta_w) 
                - this->H(u_adim - this->theta_w) * vars.get(1) * vars.get(2) / this->tau_si;
                
        //return this->get_FI(u[0], vars) + this->get_SI(u[0], vars) + this->get_SO(u[0], vars);
        //return 0.0;
    }

    ~BuenoOrovioIonicModelIon1() override{}

};

#endif
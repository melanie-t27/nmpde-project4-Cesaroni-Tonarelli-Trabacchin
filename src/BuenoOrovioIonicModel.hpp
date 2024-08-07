#ifndef BUENO_OROVIO_IONIC_MODEL_H
#define BUENO_OROVIO_IONIC_MODEL_H
#include "IonicModel.hpp"
#include "utils.hpp"
#include <cmath>
#include <tuple>

template<int K_ion, int N_ion>
class BuenoOrovioIonicModel : public IonicModel<K_ion, N_ion> {
public:

   double getAdimensionalU(double v){
        return (v + 84) / 85.7;
   }

   double getDerivative(int index, double u, GatingVariables<N_ion> vars) override{
        u = getAdimensionalU(u);
        if(index == 0) {
            return ((1 - H(u - theta_v))*(v_inf(u) - vars.get(0))) / (tau_v_minus(u)) - ((H(u - theta_v))*vars.get(0))/(tau_v_plus);
        } else if(index == 1) {
            return ((1 - H(u - theta_w))*(w_inf(u) - vars.get(1)))/(tau_w_minus(u)) - (H(u - theta_w)*vars.get(1))/(tau_w_plus);
        } else if(index == 2) {
            return 1/tau_s(u) * ((1+std::tanh(k_s * (u - u_s)))/2 - vars.get(2));
        } else {
            assert(false);
            throw -1;
        }
    }

    //first is implicit, second is explicit
    std::tuple<double, double> getExpansionCoefficients(int index, double u) override{
        double A,B;
        u = getAdimensionalU(u);
        if(index == 0) {
            A = (1 - H(u - theta_v))/tau_v_minus(u);
            B = H(u - theta_v)/(tau_v_plus);
            return {-A-B, A * v_inf(u)};
        } else if(index == 1) {
            A = (1 - H(u - theta_w))/tau_w_minus(u);
            B = H(u - theta_w)/(tau_w_plus);
            return {-A-B, + A * w_inf(u)};
        } else if(index == 2) {
            A = 1/tau_s(u) * (1 + std::tanh(k_s * (u - u_s)))/2.0;
            B = (-1) * 1/tau_s(u);
            return {B, A};
        } else {
            assert(false);
            throw -1;
        }
    }

    virtual double implicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, GatingVariables<N_ion>& vars) override;
    virtual double explicit_coefficient(const std::array<double, K_ion>& u, size_t valid_u, GatingVariables<N_ion>& vars) override;



    virtual ~BuenoOrovioIonicModel()  override {

    }

protected:
    const double u_0 = 0;

    const double u_u = 1.61;

    const double theta_v = 0.3;

    const double theta_w = 0.13;

    const double theta_v_minus = 0.1;

    const double theta_0 = 0.005;

    const double tau_v1_minus = 80000;

    const double tau_v2_minus = 1.4506 * 1e3;

    const double tau_v_plus = 1.4506 * 1e3;

    const double tau_w1_minus = 70000;

    const double tau_w2_minus = 8 * 1e3;

    const double k_w_minus = 200;

    const double u_w_minus = 0.016;

    const double tau_w_plus = 280 * 1e3;

    const double tau_fi = 0.078 * 1e3;

    const double tau_01 = 410 * 1e3;

    const double tau_02 = 7 * 1e3;

    const double tau_so_1 = 91 * 1e3;

    const double tau_so_2 = 0.8 * 1e3;

    const double k_so = 2.1;

    const double u_so = 0.6;

    const double tau_s1 = 2.7342 * 1e3;

    const double tau_s2 = 4 * 1e3;

    const double k_s = 2.0994;

    const double u_s = 0.9087;

    const double tau_si = 3.3849 * 1e3;

    const double tau_w_inf = 0.01 * 1e3;

    const double w_inf_star = 0.5;

    double H(double x){
        if (x < 0){
            return 0;
        } else {
            return 1;
        }
    }

    // Tau v minus
    double tau_v_minus(double u){
        return (1 - H(u-theta_v_minus)) * tau_v1_minus + H(u - theta_v_minus) * tau_v2_minus;
    }

    // Tau w minus
    double tau_w_minus(double u){
        return tau_w1_minus + (tau_w2_minus - tau_w1_minus) * (1 + std::tanh(k_w_minus * (u - u_w_minus))) / 2;
    }

    // Tau so
    double tau_so(double u){
        return tau_so_1 + (tau_so_2 + tau_so_1) * (1 + std::tanh(k_so * (u - u_so))) / 2;
    }

    // Tau s
    double tau_s(double u){
        return (1 - H(u - theta_v)) * tau_s1 + H(u - theta_w) * tau_s2;
    }

    // Tau o
    double tau_o(double u){
        return (1 - H(u - theta_0)) * tau_01 + H(u - theta_0) * tau_02;
    }

    // v infinity
    double v_inf(double u){
        if(u < theta_v_minus)
            return 1;
        else return 0;
    }

    // w infinity
    double w_inf(double u){
        return (1 - H(u - theta_0)) * (1 - u / tau_w_inf) + H(u - theta_0) * w_inf_star;
    }

};
#endif
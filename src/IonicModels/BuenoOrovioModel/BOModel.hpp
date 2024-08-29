#ifndef BUENO_OROVIO_IONIC_MODEL_H
#define BUENO_OROVIO_IONIC_MODEL_H
#include "../IonicModel.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <tuple>

template<int N_ion>
class BuenoOrovioIonicModel : public IonicModel<N_ion> {
public:

    BuenoOrovioIonicModel(const int tissue_type){
        switch(tissue_type){
            case 0 : // EPI
                u_0 = 0;
                u_u = 1.55;
                theta_v = 0.3;
                theta_w = 0.13;
                theta_v_minus = 0.006;
                theta_0 = 0.006;
                tau_v1_minus = 60;
                tau_v2_minus = 1150;
                tau_v_plus = 1.4506;
                tau_w1_minus = 60;
                tau_w2_minus = 15;
                k_w_minus = 65;
                u_w_minus = 0.03;
                tau_w_plus = 280;
                tau_fi = 0.11;
                tau_01 = 400;
                tau_02 = 6;
                tau_so_1 = 30.0181;
                tau_so_2 = 0.9957;
                k_so = 2.0458;
                u_so = 0.65;
                tau_s1 = 2.7342;
                tau_s2 = 16;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 1.8875;
                tau_w_inf = 0.07;
                w_inf_star = 0.94;
                break;
            
            case 1 : // MID
                u_0 = 0;
                u_u = 1.61;
                theta_v = 0.3;
                theta_w = 0.13;
                theta_v_minus = 0.1;
                theta_0 = 0.005;
                tau_v1_minus = 80;
                tau_v2_minus = 1.4506;
                tau_v_plus = 1.4506;
                tau_w1_minus = 70;
                tau_w2_minus = 8;
                k_w_minus = 200;
                u_w_minus = 0.016;
                tau_w_plus = 280;
                tau_fi = 0.078;
                tau_01 = 410;
                tau_02 = 7;
                tau_so_1 = 91;
                tau_so_2 = 0.8;
                k_so = 2.1;
                u_so = 0.6;
                tau_s1 = 2.7342;
                tau_s2 = 4;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 3.3849;
                tau_w_inf = 0.01;
                w_inf_star = 0.5;
                break;

            case 2 : // ENDO
                u_0 = 0;
                u_u = 1.56;
                theta_v = 0.3;
                theta_w = 0.13;
                theta_v_minus = 0.2;
                theta_0 = 0.006;
                tau_v1_minus = 75;
                tau_v2_minus = 10;
                tau_v_plus = 1.4506;
                tau_w1_minus = 6;
                tau_w2_minus = 140;
                k_w_minus = 200;
                u_w_minus = 0.016;
                tau_w_plus = 280;
                tau_fi = 0.1;
                tau_01 = 470;
                tau_02 = 6;
                tau_so_1 = 40;
                tau_so_2 = 1.2;
                k_so = 2;
                u_so = 0.65;
                tau_s1 = 2.7342;
                tau_s2 = 2;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 2.9013;
                tau_w_inf = 0.0273;
                w_inf_star = 0.78;
                break;

            case 3: //ten tuscer
                u_0 = 0;
                u_u = 1.58;
                theta_v = 0.3;
                theta_w = 0.015;
                theta_v_minus = 0.015;
                theta_0 = 0.006;
                tau_v1_minus = 60;
                tau_v2_minus = 1150;
                tau_v_plus = 1.4506;
                tau_w1_minus = 70;
                tau_w2_minus = 20;
                k_w_minus = 65;
                u_w_minus = 0.03;
                tau_w_plus = 280;
                tau_fi = 0.11;
                tau_01 = 6;
                tau_02 = 6;
                tau_so_1 = 43;
                tau_so_2 = 0.2;
                k_so = 2;
                u_so = 0.65;
                tau_s1 = 2.7342;
                tau_s2 = 3;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 2.8723;
                tau_w_inf = 0.07;
                w_inf_star = 0.94;
                break;
            default :
                throw -1;
                break;
        }
    }

    double getAdimensionalU(double v){
        return (1000.0 * v + 84.0) / 85.7;
    }

    double getDerivative(int index, double u, GatingVariables<N_ion> vars) override
    {
        u = getAdimensionalU(u);
        if (index == 0) {
            return ((1 - H(u - theta_v)) * (v_inf(u) - vars.get(0))) / (tau_v_minus(u)) - ((H(u - theta_v)) * vars.get(0)) / (tau_v_plus);
        } else if (index == 1) {
            return ((1 - H(u - theta_w)) * (w_inf(u) - vars.get(1))) / (tau_w_minus(u)) - (H(u - theta_w) * vars.get(1)) / (tau_w_plus);
        } else if (index == 2) {
            return 1 / tau_s(u) * ((1 + std::tanh(k_s * (u - u_s))) / 2 - vars.get(2));
        } else {
            assert(false);
            throw -1;
        }
    }

    //first is implicit, second is explicit
    std::tuple<double, double> getExpansionCoefficients(int index, double u) override
    {
        double A, B;
        u = getAdimensionalU(u);
        if (index == 0 ){
            A = (1 - H(u - theta_v)) / tau_v_minus(u);
            B = H(u - theta_v) / (tau_v_plus);
            return {-A - B, A * v_inf(u)};
        } else if (index == 1) {
            A = (1 - H(u - theta_w)) / tau_w_minus(u);
            B = H(u - theta_w) / (tau_w_plus);
            return {-A - B, +A * w_inf(u)};
        } else if (index == 2) {
            A = 1 / tau_s(u) * (1 + std::tanh(k_s * (u - u_s))) / 2.0;
            B = (-1) * 1 / tau_s(u);
            return {B, A};
        } else {
            assert(false);
            throw -1;
        }
    }

    double ionic_current(const double u, GatingVariables<N_ion>& vars) override 
    {
        return this->get_FI(u, vars) + this->get_SI(u, vars) + this->get_SO(u, vars);
    }

    double get_FI(double u, GatingVariables<N_ion>& vars) override
    {
        u = getAdimensionalU(u);
        return - vars.get(0) * H(u - theta_v) * (u - theta_v) * (u_u - u)/tau_fi;
    }

    double get_SO(double u, GatingVariables<N_ion>& /*vars*/) override
    {
        u = getAdimensionalU(u);
        return u*(1 - H(u - theta_w))/tau_o(u) + H(u - theta_w)/tau_so(u);
    }

    double get_SI(double u, GatingVariables<N_ion>& vars) override
    {
        u = getAdimensionalU(u);
        return -H(u - theta_w) * vars.get(1) * vars.get(2)/tau_si;
    }

    ~BuenoOrovioIonicModel() override {}

protected:
    double u_0;
    double u_u;
    double theta_v;
    double theta_w;
    double theta_v_minus;
    double theta_0;
    double tau_v1_minus;
    double tau_v2_minus;
    double tau_v_plus;
    double tau_w1_minus;
    double tau_w2_minus;
    double k_w_minus;
    double u_w_minus;
    double tau_w_plus;
    double tau_fi;
    double tau_01;
    double tau_02;
    double tau_so_1;
    double tau_so_2;
    double k_so;
    double u_so;
    double tau_s1;
    double tau_s2;
    double k_s;
    double u_s;
    double tau_si;
    double tau_w_inf;
    double w_inf_star;

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
        return tau_so_1 + (tau_so_2 - tau_so_1) * (1 + std::tanh(k_so * (u - u_so))) / 2;
    }

    // Tau s
    double tau_s(double u){
        return (1 - H(u - theta_w)) * tau_s1 + H(u - theta_w) * tau_s2;
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
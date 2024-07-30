//
// Created by Melanie Tonarelli on 24/01/24.
//
#include "Solver.hpp"
#include "utils.hpp"
#include "Coupler.hpp"
#include "IonicModel.hpp"
#include "BuenoOrovioIonicModelIon1.hpp"
#include "SVICoupler.hpp"

int main(){
    std::shared_ptr<BuenoOrovioIonicModelIon1<3,3>> ionic_model = std::make_shared<BuenoOrovioIonicModelIon1<3,3>>();
    std::shared_ptr<SVICoupler<3,3,3>> coupler = std::make_shared<SVICoupler<3,3,3>>();
    Solver<3,3,3> solver("PDIO", 2, 1.0, 0.5, 1.5, 1.0, ionic_model, coupler);

    return 0;
}
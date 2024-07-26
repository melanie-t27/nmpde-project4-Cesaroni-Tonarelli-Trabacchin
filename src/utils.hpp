#ifndef UTILS_H
#define UTILS_H

template<int N>
struct GatingVariables {
    double& get(int k) {
        assert(k < N);
        return var[k];
    }
private:
    double var[N];

};

#endif
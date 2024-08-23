#ifndef VECTOR_VIEW_H
#define VECTOR_VIEW_H
#include "utils.hpp"
template<typename VectorType>
class VectorView {
public:
    VectorView(VectorType& vec, size_t offset, size_t size) : vec(vec), offset(offset), size_(size) {}
    double get(size_t index) {
        return vec[index + offset];
    }

    /*TrilinosWrappers::internal::VectorReference operator[](int index) {
        return vec(index + offset);
    }*/

    double operator[](int index) const {
        return vec[index + offset];
    }

    void set(size_t index, double value) {
        vec[index + offset] = value;
    }


    size_t size() {
        return size_;
    }
private:
    VectorType& vec;
    size_t offset;
    size_t size_;
};


#endif
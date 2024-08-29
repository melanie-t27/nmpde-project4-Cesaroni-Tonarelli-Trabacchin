#ifndef VECTOR_VIEW_H
#define VECTOR_VIEW_H
#include "utils.hpp"
template<typename VectorType>
// The class provides a view of a portion of a vector of type VectorType
// The purpose of the class is to allow access and manipulation of a subvector
// of a large vector, without having to copy the data.
// The class helps us manage and interact with parts of a vector more efficiently,
// both in terms of memory and performance

class VectorView {
public:
    // constructor that takes as parameters a reference to a vector of type VectorType,
    // the offset (starting index) and the size
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
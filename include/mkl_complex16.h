#pragma once

#include "mkl.h"


MKL_Complex16 operator*(const MKL_Complex16 &a, const MKL_Complex16 &b) {
    MKL_Complex16 result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

MKL_Complex16 operator+(const MKL_Complex16 &a, const MKL_Complex16 &b) {
    MKL_Complex16 result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

MKL_Complex16 operator+=(MKL_Complex16 &a, const MKL_Complex16 &b) {
    a.real += b.real;
    a.imag += b.imag;
    return a;
}

MKL_Complex16 operator*=(MKL_Complex16 &a, const MKL_Complex16 &b) {
    double real = a.real;
    a.real = real * b.real - a.imag * b.imag;
    a.imag = real * b.imag + a.imag * b.real;
    return a;
}

MKL_Complex16 operator*(double a, const MKL_Complex16 &b) {
    MKL_Complex16 result;
    result.real = a * b.real;
    result.imag = a * b.imag;
    return result;
}

MKL_Complex16 operator*(const MKL_Complex16 &a, double b) {
    MKL_Complex16 result;
    result.real = b * a.real;
    result.imag = b * a.imag;
    return result;
}

MKL_Complex16 operator/(const MKL_Complex16& a, double b) {
    MKL_Complex16 result;
    result.real = a.real/b;
    result.imag = a.imag/b;
    return result;
}

bool operator!=(const MKL_Complex16& a, const MKL_Complex16& b) {
    return not((a.real == b.real) and (a.imag == b.imag));
}

bool operator==(const MKL_Complex16& obj, double val) {
    return (obj.real == val and obj.imag == 0.);
}

bool operator!=(const MKL_Complex16& obj, double val) {
    return obj.real != val or obj.imag != 0.;
}

MKL_Complex16 Conjugate(MKL_Complex16 number) {
    number.imag = -number.imag;
    return number;
}
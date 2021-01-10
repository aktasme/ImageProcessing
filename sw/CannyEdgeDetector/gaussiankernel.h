/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#ifndef GAUSSIANKERNEL_H
#define GAUSSIANKERNEL_H

#include "basekernel.h"

class GaussianKernel : public BaseKernel
{
public:
    GaussianKernel(const float sigma, const unsigned int size);
    virtual ~GaussianKernel();
};

#endif // GAUSSIANKERNEL_H

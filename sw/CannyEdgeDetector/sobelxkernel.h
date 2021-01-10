/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#ifndef SOBELXKERNEL_H
#define SOBELXKERNEL_H

#include "basekernel.h"

class SobelXKernel : public BaseKernel
{
public:
    SobelXKernel();
    virtual ~SobelXKernel();
};

static const int sobelXKernelDefinition[3][3] =
{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

#endif // SOBELXKERNEL_H

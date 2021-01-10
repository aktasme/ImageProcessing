/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#ifndef SOBELYKERNEL_H
#define SOBELYKERNEL_H

#include "basekernel.h"

class SobelYKernel : public BaseKernel
{
public:
    SobelYKernel();
    virtual ~SobelYKernel();
};

static const int sobelYKernelDefinition[3][3] =
{
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1}
};

#endif // SOBELYKERNEL_H

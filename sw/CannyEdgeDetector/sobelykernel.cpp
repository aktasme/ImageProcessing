/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#include "sobelykernel.h"

SobelYKernel::SobelYKernel()
    : BaseKernel(3)
{
    for (unsigned int i = 0; i < height; ++i)
    {
        for (unsigned int j = 0; j < width; ++j)
        {
            kernel[i][j] = sobelYKernelDefinition[i][j];
        }
    }
}

SobelYKernel::~SobelYKernel()
{

}

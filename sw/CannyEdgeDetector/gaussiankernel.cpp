/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#include "gaussiankernel.h"
#include <cmath>

GaussianKernel::GaussianKernel(const float sigma, const unsigned int size)
    : BaseKernel(size)
{
    float r;
    float s = 2.0 * sigma * sigma;

    /* sum for normalization */
    float sum = 0.0;

    int upperIndex = size / 2;
    int lowerIndex = -1 * upperIndex;

    /* generating Gaussian kernel */
    for (int x = lowerIndex; x <= upperIndex; x++)
    {
        for (int y = lowerIndex; y <= upperIndex; y++)
        {
            r = sqrt(x * x + y * y);
            kernel[x + upperIndex][y + upperIndex] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += kernel[x + upperIndex][y + upperIndex];
        }
    }

    /* normalization */
    for (unsigned int i = 0; i < size; ++i)
    {
        for (unsigned int j = 0; j < size; ++j)
        {
            kernel[i][j] /= sum;
        }
    }
}

GaussianKernel::~GaussianKernel()
{

}

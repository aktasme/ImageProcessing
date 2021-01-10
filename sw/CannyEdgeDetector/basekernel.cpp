/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#include "basekernel.h"

BaseKernel::BaseKernel(const unsigned int size)
    : width(size)
    , height(size)
{
    qDebug() << "BaseKernel::BaseKernel:Enter";

    /* Allocate kernel from heap */
    kernel = new float*[size];
    for(unsigned int i = 0; i < size; i++)
    {
        kernel[i] = new float[size];
    }

    /* Initialize kernel */
    for (unsigned int i = 0; i < size; i++)
    {
        for (unsigned int j = 0; j < size; j++)
        {
            kernel[i][j] = 0.0f;
        }
    }
}

BaseKernel::~BaseKernel()
{
    /* Deallocate heap */
    for(unsigned int i = 0; i < height; i++)
    {
       delete[] kernel[i];
    }

    delete[] kernel;
}

unsigned int BaseKernel::getWidth() const
{
    return width;
}

void BaseKernel::setWidth(unsigned int value)
{
    width = value;
}

unsigned int BaseKernel::getHeight() const
{
    return height;
}

void BaseKernel::setHeight(unsigned int value)
{
    height = value;
}

float BaseKernel::value(const unsigned int x, const unsigned y) const
{
    return kernel[x][y];
}

/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#ifndef BASEKERNEL_H
#define BASEKERNEL_H

#include <QDebug>

class BaseKernel
{
public:
    BaseKernel(const unsigned int size);
    virtual ~BaseKernel();

    unsigned int getWidth() const;
    void setWidth(unsigned int value);

    unsigned int getHeight() const;
    void setHeight(unsigned int value);

    float value(const unsigned int x, const unsigned y) const;

protected:
    unsigned int width;
    unsigned int height;

    float** kernel;
};

#endif // BASEKERNEL_H

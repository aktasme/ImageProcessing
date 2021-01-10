/*
 * CannyEdgeDetecor: Software for edge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include "basekernel.h"
#include <queue>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

    typedef std::queue<std::pair<int, int>> Edges;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void displayImage(const QImage& inputImage);

    QImage convolution(const BaseKernel* kernel, const QImage& inputImage);
    void gradientMagnitude(QImage& inputImage, QImage& gradientX, QImage& gradientY);


    void convertToGrayScale(QImage& inputImage);
    void smoothing(QImage& inputImage);
    void findingGradients(QImage& inputImage, QImage& gradientX, QImage& gradientY);
    void nonMaximumSuppression(QImage& inputImage, QImage& gradientX, QImage& gradientY);
    void doubleThresholding(QImage& inputImage, const float thresholdMin, const float thresholdMax);
    void edgeTracking(QImage& inputImage, const float thresholdMin, const float thresholdMax);

public slots:
    void onSelectImageClicked();
    void onCalculateCanny();
    void onSaveClicked();
    void onSigmaChanged(int value);
    void onKernelSizeChanged(int value);
    void onThresholdLowChanged();
    void onThresholdHighChanged();

private:
    Ui::MainWindow *ui;

    float sigma = 1.0f;
    unsigned int kernelSize = 5;
    float thresholdMin = 40.0f;
    float thresholdMax = 120.0f;

    QString fileName;
    QImage baseImage;
    QImage resultImage;
    QGraphicsScene graphicScene;
    Edges strongEdges;
    Edges weakEdges;
};
#endif // MAINWINDOW_H

/*
 * CannyEdgeDetecor: Software foredge detection
 *
 * BLG 513E Image Processing HW#1
 *
 * Copyright (C)  Mehmet AKTAS 0504181576
 *
 */
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "gaussiankernel.h"
#include "sobelxkernel.h"
#include "sobelykernel.h"
#include <cmath>
#include <queue>
#include <utility>
#include <QFileDialog>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    /* UI connections */
    connect(ui->pushButtonSelect, &QPushButton::clicked, this, &MainWindow::onSelectImageClicked);
    connect(ui->pushButtonCanny, &QPushButton::clicked, this, &MainWindow::onCalculateCanny);
    connect(ui->pushButtonSave, &QPushButton::clicked, this, &MainWindow::onSaveClicked);
    connect(ui->spinBoxSigma, SIGNAL(valueChanged(int)), this, SLOT(onSigmaChanged(int)));
    connect(ui->spinBoxKernelSize, SIGNAL(valueChanged(int)), this, SLOT(onKernelSizeChanged(int)));
    connect(ui->spinBoxThresholdLow, SIGNAL(editingFinished()), this, SLOT(onThresholdLowChanged()));
    connect(ui->spinBoxThresholdHigh, SIGNAL(editingFinished()), this, SLOT(onThresholdHighChanged()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

/* Displays given image */
void MainWindow::displayImage(const QImage& inputImage)
{
    qDebug() << "displayImage::Enter";

    graphicScene.clear();
    graphicScene.addPixmap(QPixmap::fromImage(inputImage));
    graphicScene.setSceneRect(inputImage.rect());
    ui->graphicsView->setScene(&graphicScene);
    ui->graphicsView->fitInView(graphicScene.sceneRect(), Qt::KeepAspectRatio);
}

/* Converts image to grayscale */
void MainWindow::convertToGrayScale(QImage& inputImage)
{
    qDebug() << "convertToGrayScale::Enter";

    if(inputImage.isGrayscale())
    {
        qDebug() << "onSelectImage::Already grayscale!";
    }
    else
    {
        QImage grayScaledImage(inputImage.size(), QImage::Format_Grayscale8);
        for(int rawIndex = 0; rawIndex < inputImage.height(); rawIndex++)
        {
            quint32* inputLine = reinterpret_cast<QRgb*>(inputImage.scanLine(rawIndex));
            quint8* outputLine = grayScaledImage.scanLine(rawIndex);

            for(int columnIndex = 0; columnIndex < inputImage.width(); columnIndex++)
            {
                QRgb* inputPixel = inputLine + columnIndex;
                quint8 grayPixel = qGray(*inputPixel);
                outputLine[columnIndex] = grayPixel;
            }
        }

        inputImage = grayScaledImage;
    }
}

/* Convols image with given kernel */
QImage MainWindow::convolution(const BaseKernel* kernel, const QImage& inputImage)
{
    qDebug() << "convolution::Enter";

    unsigned int kernelWidth = kernel->getWidth();
    unsigned int kernelHeight = kernel->getHeight();
    unsigned int offsetx = kernelWidth / 2;
    unsigned int offsety = kernelHeight / 2;

    unsigned int imageWidth = static_cast<unsigned int>(inputImage.width());
    unsigned int imageheight = static_cast<unsigned int>(inputImage.height());

    QImage out(inputImage.size(), inputImage.format());
    float sum;

    quint8 *line;
    const quint8 *lookupLine;

    for(unsigned int y = 0; y < imageheight; y++)
    {
        line = out.scanLine(y);
        for(unsigned  x = 0; x < imageWidth; x++)
        {
            sum = 0;

            for(unsigned int j = 0; j < kernelHeight; j++)
            {
                if (y + j < offsety || y + j >= imageheight)
                {
                    continue;
                }

                lookupLine = inputImage.constScanLine(y + j - offsety);

                for(unsigned  i = 0; i < kernelWidth; i++)
                {
                    if (x + i < offsetx || x + i >= imageWidth)
                    {
                        continue;
                    }

                    sum += kernel->value(j, i) * lookupLine[x + i - offsetx];
                }
            }

            sum = abs(sum);
            line[x] = qBound(0x00, static_cast<int>(sum), 0xFF);
        }
    }

    return out;
}

/* Calculates gradient magnitude with given gradients */
void MainWindow::gradientMagnitude(QImage& inputImage, QImage& gradientX, QImage& gradientY)
{
    qDebug() << "gradientMagnitude::Enter";

    quint8 *line;
    const quint8 *gradientXline;
    const quint8 *gradientYline;

    for(int y = 0; y < inputImage.height(); y++)
    {
        line = inputImage.scanLine(y);
        gradientXline = gradientX.scanLine(y);
        gradientYline = gradientY.scanLine(y);

        for(int x = 0; x < inputImage.width(); x++)
        {
            line[x] = qBound(0x00, static_cast<int>(hypot(gradientXline[x], gradientYline[x])), 0xFF);
        }
    }
}

/* Smooths image with using Gaussian Kernel */
void MainWindow::smoothing(QImage& inputImage)
{
    qDebug() << "smoothing::Enter";

    if(sigma < 1.0)
    {
        inputImage = baseImage;
    }
    else
    {
        GaussianKernel* kernel = new GaussianKernel(sigma, kernelSize);
        inputImage = convolution(kernel, inputImage);
    }
}

/* Finds gradients using Sobel kernel */
void MainWindow::findingGradients(QImage& inputImage, QImage& gradientX, QImage& gradientY)
{
    SobelXKernel sobelX;
    SobelYKernel sobelY;
    gradientX = convolution(&sobelX, inputImage);
    gradientY = convolution(&sobelY, inputImage);

    gradientMagnitude(inputImage, gradientX, gradientY);
}

/* Thinning the edges */
void MainWindow::nonMaximumSuppression(QImage& inputImage, QImage& gradientX, QImage& gradientY)
{
    qDebug() << "nonMaximumSuppression::Enter";

    quint8 *line;
    quint8 *thinLine;
    const quint8* previousLine;
    const quint8* nextLine;
    const quint8* gradientXLine;
    const quint8* gradientYLine;

    QImage thinImage = inputImage;

    for(int y = 1; y < inputImage.height() - 1; y++)
    {
        line = inputImage.scanLine(y);
        thinLine = thinImage.scanLine(y);

        previousLine = inputImage.scanLine(y - 1);
        nextLine = inputImage.scanLine(y + 1);

        gradientXLine = gradientX.scanLine(y);
        gradientYLine = gradientY.scanLine(y);

        for(int x = 1; x < inputImage.width() - 1; x++)
        {
            if(line[x] == 0x00)
            {
                continue;
            }

            double angle = atan2(gradientYLine[x], gradientXLine[x]) * 180 / M_PI;

            if( ((angle > -22.5) && (angle <= 22.5 )) || ((angle > 157.5) && (angle <= -157.5 )) )
            {
                /* Horizantal edge */
                if(line[x] < previousLine[x] || line[x] < nextLine[x])
                {
                    thinLine[x] = 0x00;
                }
            }
            else if(((angle > -112.5) && (angle < -67.5 )) || ((angle > 67.5) && (angle < 112.5 )))
            {
                /* Vertical edge */
                if(line[x] < line[x+1] || line[x] < line[x-1])
                {
                    thinLine[x] = 0x00;
                }
            }
            else if(((angle > -67.5) && (angle < -22.5 )) || ((angle > 112.5) && (angle < 157.5 )))
            {
                /* -45 degree edge */
                if(line[x] < previousLine[x-1] || line[x] < nextLine[x+1])
                {
                    thinLine[x] = 0x00;
                }
            }
            else if(((angle > -157.5) && (angle < -112.5 )) || ((angle > 22.5) && (angle < 67.5 )))
            {
                /* 45 degree edge */
                if(line[x] < previousLine[x+1] || line[x] < nextLine[x-1])
                {
                    thinLine[x] = 0x00;
                }
            }
            else
            {
                thinLine[x] = line[x];
            }
        }
    }

    inputImage = thinImage;
}

/* Finds strong and weak edges. Clears the image. */
void MainWindow::doubleThresholding(QImage& inputImage, const float thresholdMin, const float thresholdMax)
{
    qDebug() << "doubleThresholding::Enter";

    quint8 *line;

    for(int y = 0; y < inputImage.height(); y++)
    {
        line = inputImage.scanLine(y);

        for(int x = 0; x < inputImage.width(); x++)
        {
            if(line[x] < thresholdMin)
            {
                /* No edge */
                line[x] = 0x00;
            }
            else if(line[x] > thresholdMax)
            {
                /* Strong edges */
                strongEdges.push(std::make_pair(x, y));
                line[x] = 0xFF;
            }
            else
            {
                /* Weak edge */
                weakEdges.push(std::make_pair(x, y));
            }
        }
    }
}

/* Finds weak edges connected to strong ones and make them strong. */
void MainWindow::edgeTracking(QImage& inputImage, const float thresholdMin, const float thresholdMax)
{
    qDebug() << "edgeTracking::Enter";

    quint8 *line;

    while(!strongEdges.empty())
    {
        int x = strongEdges.front().first;
        int y = strongEdges.front().second;
        strongEdges.pop();

        /* Looks the 2nd level neighbourhood to strong edges */
        for(int i = -2; i <= 2; i++)
        {
            int neighbourY = y + i;

            if(neighbourY >= inputImage.height() || (neighbourY < 0))
            {
                continue;
            }

            line = inputImage.scanLine(neighbourY);

            /* Looks the 2nd level neighbourhood to strong edges */
            for(int j = -2; j <= 2; j++)
            {
                int neighbourX = x + j;

                if(neighbourX >= inputImage.width() || (neighbourX < 0))
                {
                    continue;
                }

                if(line[neighbourX] > thresholdMin && line[neighbourX] < thresholdMax)
                {
                    line[x] = 0xFF;
                    strongEdges.push(std::make_pair(neighbourX, neighbourY));
                }
            }
        }
    }

    /* Lastly, clear the image from remaining weak edges */
    while(!weakEdges.empty())
    {
        /* Clean remaining */
        int x = weakEdges.front().first;
        int y = weakEdges.front().second;
        weakEdges.pop();

        if(inputImage.pixel(x, y) != 0xFF)
        {
            inputImage.setPixel(x, y, 0x00);
        }
    }
}

/* Called on image selection */
void MainWindow::onSelectImageClicked()
{
    qDebug() << "onSelectImage::Enter";

    fileName = QFileDialog::getOpenFileName(this, tr("Select Image"));
    baseImage = QImage(fileName);
    resultImage = baseImage;
    displayImage(resultImage);
    ui->pushButtonCanny->setEnabled(true);
}

/* Called on clicking Canny button */
void MainWindow::onCalculateCanny()
{
    qDebug() << "onCannyButtonClicked::Enter";

    resultImage = baseImage;

    QImage gradientX;
    QImage gradientY;

    convertToGrayScale(resultImage);
    smoothing(resultImage);
    findingGradients(resultImage, gradientX, gradientY);
    nonMaximumSuppression(resultImage, gradientX, gradientY);
    doubleThresholding(resultImage, thresholdMin, thresholdMax);
    edgeTracking(resultImage, thresholdMin, thresholdMax);

    displayImage(resultImage);

    ui->spinBoxSigma->setEnabled(true);
    ui->spinBoxKernelSize->setEnabled(true);
    ui->spinBoxThresholdLow->setEnabled(true);
    ui->spinBoxThresholdHigh->setEnabled(true);
    ui->pushButtonSave->setEnabled(true);
}

/* Called on clicking Save button */
void MainWindow::onSaveClicked()
{
    qDebug() << "onSaveClicked::Enter";

    QString path = QFileDialog::getExistingDirectory(this, tr("Destination"));
    QString name = "Canny_" + QFileInfo(fileName).fileName();
    resultImage.save(QDir(path).filePath(name));
}

/* Called when sigma changed */
void MainWindow::onSigmaChanged(int value)
{
    ui->spinBoxSigma->setEnabled(false);

    sigma = static_cast<float>(value);

    qDebug() << "onSaveClicked::Enter sigma:" << sigma;

    emit onCalculateCanny();
}

/* Called when kernel size changed */
void MainWindow::onKernelSizeChanged(int value)
{
    ui->spinBoxKernelSize->setEnabled(false);

    kernelSize = static_cast<unsigned int>(value);

    qDebug() << "onSaveClicked::Enter kernelSize:" << kernelSize;

    emit onCalculateCanny();
}

/* Called when low thresold changed */
void MainWindow::onThresholdLowChanged()
{
    ui->spinBoxThresholdLow->setEnabled(true);

    thresholdMin = static_cast<float>(ui->spinBoxThresholdLow->value());

    qDebug() << "onSaveClicked::Enter thresholdMin:" << thresholdMin;

    emit onCalculateCanny();
}

/* Called when highs thresold changed */
void MainWindow::onThresholdHighChanged()
{
    ui->spinBoxThresholdHigh->setEnabled(true);

    thresholdMax = static_cast<float>(ui->spinBoxThresholdHigh->value());

    qDebug() << "onSaveClicked::Enter thresholdMax:" << thresholdMax;

    emit onCalculateCanny();
}

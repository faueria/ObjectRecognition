/*
  Nihal Sandadi

  Header file for image thresholding operations, greyscale and HSV.
*/

#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include <opencv2/opencv.hpp>

double findOptimalThreshold(const cv::Mat& image, int sampleFraction = 16);
cv::Mat grayscaleThreshold(const cv::Mat& frame);
cv::Mat customThreshold(const cv::Mat& frame);

#endif
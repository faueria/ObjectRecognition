/*
  Nihal Sandadi

  Header file for morphological image processing operations for noise removal 
  and to clean the thresholded images.
*/

#ifndef MORPHOLOGICAL_H
#define MORPHOLOGICAL_H

#include <opencv2/opencv.hpp>

cv::Mat morphologicalClean(const cv::Mat& thresholded);
cv::Mat enhancedCleanThreshold(const cv::Mat& thresholded);
cv::Mat basicCleanThreshold(const cv::Mat& thresholded);

#endif
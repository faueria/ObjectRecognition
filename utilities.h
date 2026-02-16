// utilities.h
#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Function to prepare image for CNN embedding
void prepEmbeddingImage(cv::Mat& frame, cv::Mat& embimage,
    int cx, int cy, float theta,
    float minE1, float maxE1, float minE2, float maxE2,
    int debug = 0);

// Function to get CNN embedding from prepared image
int getEmbedding(cv::Mat& src, cv::Mat& embedding,
    cv::dnn::Net& net, int debug = 0);

#endif // UTILITIES_H
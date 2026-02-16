/*
  Nihal Sandadi

  an adaptive thresholding methods using k-means clustering
  for threshold selection and multiple thresholding strategies
  for object segmentation.
*/

#include "thresholding.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

using namespace cv;
using namespace std;

/*
  image : input image for threshold calculation
  sampleFraction : fraction of total pixels to sample for k-means clustering

  k-means clustering on randomly sampled pixels to determine the
  binary threshold value by separating image into two clusters.
*/
double findOptimalThreshold(const Mat& image, int sampleFraction) {
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    int totalPixels = gray.rows * gray.cols;
    int sampleSize = totalPixels / sampleFraction;
    vector<float> samples;
    samples.reserve(sampleSize);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, totalPixels - 1);

    for (int i = 0; i < sampleSize; ++i) {
        int idx = dis(gen);
        int row = idx / gray.cols;
        int col = idx % gray.cols;
        samples.push_back(static_cast<float>(gray.at<uchar>(row, col)));
    }

    Mat samplesMat(static_cast<int>(samples.size()), 1, CV_32F, samples.data());
    Mat labels, centers;
    kmeans(samplesMat, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);
    float mean1 = centers.at<float>(0, 0);
    float mean2 = centers.at<float>(1, 0);
    return (min(mean1, mean2) + max(mean1, mean2)) / 2.0;
}

/*
  frame : color frame from video capture

  Converts image to grayscale and applies thresholding.
*/
Mat grayscaleThreshold(const Mat& frame) {
    Mat gray, blurred, result;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);
    double thresholdValue = findOptimalThreshold(blurred);
    threshold(blurred, result, thresholdValue, 255, THRESH_BINARY);
    return result;
}

/*
  frame : color frame from video capture

  Uses HSV color space combination of saturation and value channels 
  for segmentation of colored objects against background.
*/
Mat customThreshold(const Mat& frame) {
    Mat hsv, saturation, value, result;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);
    saturation = hsvChannels[1];
    value = hsvChannels[2];
    Mat combined;
    addWeighted(value, 0.7, saturation, 0.3, 0, combined, CV_8U);
    Mat blurred;
    GaussianBlur(combined, blurred, Size(5, 5), 1.5);
    double thresholdValue = findOptimalThreshold(blurred);
    threshold(blurred, result, thresholdValue, 255, THRESH_BINARY);
    return result;
}
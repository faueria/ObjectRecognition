/*
  Nihal Sandadi

  Header file for region feature computation and visualization, including
  shape descriptions, moment invariants, and feature display functions.
*/

#ifndef REGION_FEATURES_H
#define REGION_FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

/*
  feature set for object characterization including geometric
  properties, moment invariants, and oriented bounding box for classification.
*/
struct RegionFeatures {
    int regionId;
    double area;
    double percentFilled;
    double aspectRatio;
    double elongation;
    std::vector<double> huMoments;
    double centroidX;
    double centroidY;
    cv::RotatedRect orientedBoundingBox;
};

RegionFeatures computeRegionFeatures(const cv::Mat& regionMask, int regionId);
void drawRegionFeatures(cv::Mat& image, const RegionFeatures& features, const cv::Scalar& color = cv::Scalar(0, 255, 255));
cv::Mat createFeatureDisplay(const std::vector<RegionFeatures>& features, const cv::Size& size);

#endif
/*
  Nihal Sandadi

  Header file for region analysis and connected components processing, including
  region detection, filtering, and visualization for object recognition.
*/

#ifndef REGION_ANALYSIS_H
#define REGION_ANALYSIS_H

#include <opencv2/opencv.hpp>
#include <vector>

/*
  Stores the region properties including identification, geometric characteristics,
  and visual representation data for connected component analysis.
*/
struct Region {
    int id;
    int area;
    cv::Point centroid;
    cv::Rect boundingBox;
    cv::Scalar color;
};

std::vector<Region> analyzeRegions(const cv::Mat& binaryImage,
    int minArea = 1000,
    int maxRegions = 5,
    bool ignoreBoundaryRegions = true);
cv::Mat createRegionMap(const cv::Mat& binaryImage,
    const std::vector<Region>& regions,
    bool showCentroids = true,
    bool showBoundingBoxes = true);
cv::Mat createColoredRegionMap(const std::vector<Region>& regions,
    const cv::Mat& labels,
    const cv::Size& imageSize);

#endif // REGION_ANALYSIS_H
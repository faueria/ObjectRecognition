/*
  Nihal Sandadi

  Implementation of region feature extraction including area based metrics,
  Hu moments, and visualization functions for object analysis and classification.
*/

#include "regionFeatures.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace std;

/*
  regionMask : binary mask image of the region to analyze
  regionId : identifier for the region being processed

  Computes a set of rotation-invariant features including area,
  oriented bounding box properties, Hu moments, and shape characteristics
  for object classification and recognition.
*/
RegionFeatures computeRegionFeatures(const Mat& regionMask, int regionId) {
    RegionFeatures features;
    features.regionId = regionId;
    features.area = countNonZero(regionMask);
    Moments m = moments(regionMask, true);
    if (m.m00 != 0) {
        features.centroidX = m.m10 / m.m00;
        features.centroidY = m.m01 / m.m00;
    }

    HuMoments(m, features.huMoments);
    vector<vector<Point>> contours;
    findContours(regionMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return features;
    }

    features.orientedBoundingBox = minAreaRect(contours[0]);
    Size2f obbSize = features.orientedBoundingBox.size;

    double width = max(obbSize.width, obbSize.height);
    double height = min(obbSize.width, obbSize.height);
    features.aspectRatio = (height > 0) ? width / height : 0;

    double obbArea = width * height;
    features.percentFilled = (obbArea > 0) ? (features.area / obbArea) : 0;

    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;

    double common = sqrt(4 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02));
    double lambda1 = 0.5 * ((mu20 + mu02) + common);
    double lambda2 = 0.5 * ((mu20 + mu02) - common);

    features.elongation = (lambda1 > 0) ? (1 - sqrt(lambda2 / lambda1)) : 0;

    return features;
}

/*
  image : output image for drawing visualization elements
  features : RegionFeatures object containing geometric properties to display
  color : drawing color for features and annotations

  Draws oriented bounding box, centroid marker, and feature information
  overlay on the input image for visual analysis and debugging.
*/
void drawRegionFeatures(Mat& image, const RegionFeatures& features, const Scalar& color) {
    if (features.area == 0) return;

    Point2f vertices[4];
    features.orientedBoundingBox.points(vertices);
    for (int i = 0; i < 4; i++) {
        line(image, vertices[i], vertices[(i + 1) % 4], color, 2);
    }

    Point centroid(static_cast<int>(features.centroidX), static_cast<int>(features.centroidY));
    circle(image, centroid, 5, color, -1);
    circle(image, centroid, 8, Scalar(255, 255, 255), 2);

    stringstream info;
    info << "R" << features.regionId
        << " PF:" << fixed << setprecision(2) << features.percentFilled
        << " AR:" << setprecision(2) << features.aspectRatio
        << " E:" << setprecision(2) << features.elongation;

    putText(image, info.str(),
        Point(static_cast<int>(features.centroidX) + 15, static_cast<int>(features.centroidY)),
        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
}

/*
  features : vector of RegionFeatures objects to display
  size : dimensions of the output display panel

  Creates a formatted text display panel showing computed features for all
  analyzed regions with organized layout and numerical formatting.
*/
Mat createFeatureDisplay(const vector<RegionFeatures>& features, const Size& size) {
    Mat display = Mat::zeros(size, CV_8UC3);

    int yPos = 30;
    int lineHeight = 20;

    putText(display, "Region-Based Features Only", Point(10, yPos),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    yPos += 35;

    for (const auto& feature : features) {
        stringstream header;
        header << "Region " << feature.regionId << " (Area: " << fixed << setprecision(0) << feature.area << ")";
        putText(display, header.str(), Point(10, yPos),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        yPos += lineHeight;
        stringstream featureLine;
        featureLine << "Percent Filled: " << fixed << setprecision(3) << feature.percentFilled
            << " | Aspect Ratio: " << setprecision(3) << feature.aspectRatio;
        putText(display, featureLine.str(), Point(15, yPos),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
        yPos += lineHeight;
        stringstream featureLine2;
        featureLine2 << "Elongation: " << setprecision(3) << feature.elongation
            << " | Hu1: " << scientific << setprecision(1) << feature.huMoments[0];
        putText(display, featureLine2.str(), Point(15, yPos),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 100), 1);
        yPos += lineHeight;
        line(display, Point(10, yPos), Point(size.width - 10, yPos), Scalar(100, 100, 100), 1);
        yPos += 10;

        if (yPos > size.height - 30) break;
    }
    return display;
}
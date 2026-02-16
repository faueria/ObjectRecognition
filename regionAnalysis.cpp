/*
  Nihal Sandadi

  Implementation of region analysis functions for connected component, region filtering, 
  and visualization processing with bounding boxes and centroids.
*/

#include "regionAnalysis.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace cv;
using namespace std;

/*
  binaryImage : binary image from thresholding operation
  minArea : minimum pixel area for region consideration
  maxRegions : maximum number of regions to return
  ignoreBoundaryRegions : whether to exclude regions touching image boundaries

  Analyze connected components in binary image, filters by area and boundary
  conditions, and returns a sorted list of significant regions with properties.
*/
vector<Region> analyzeRegions(const Mat& binaryImage,
    int minArea,
    int maxRegions,
    bool ignoreBoundaryRegions) {

    vector<Region> regions;
    Mat invertedBinary;
    bitwise_not(binaryImage, invertedBinary);
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(invertedBinary, labels, stats, centroids, 8);
    vector<Scalar> colors;
    RNG rng(12345);
    for (int i = 0; i < numLabels; i++) {
        colors.push_back(Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
    }

    vector<Region> allRegions;
    for (int i = 1; i < numLabels; i++) {
        Region region;
        region.id = i;
        region.area = stats.at<int>(i, CC_STAT_AREA);
        region.centroid = Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        region.boundingBox = Rect(
            stats.at<int>(i, CC_STAT_LEFT),
            stats.at<int>(i, CC_STAT_TOP),
            stats.at<int>(i, CC_STAT_WIDTH),
            stats.at<int>(i, CC_STAT_HEIGHT)
        );
        region.color = colors[i];

        allRegions.push_back(region);
    }

    vector<Region> filteredRegions;
    for (const auto& region : allRegions) {
        if (region.area >= minArea) {
            if (ignoreBoundaryRegions) {
                if (region.boundingBox.x > 0 &&
                    region.boundingBox.y > 0 &&
                    region.boundingBox.x + region.boundingBox.width < binaryImage.cols - 1 &&
                    region.boundingBox.y + region.boundingBox.height < binaryImage.rows - 1) {
                    filteredRegions.push_back(region);
                }
            }
            else {
                filteredRegions.push_back(region);
            }
        }
    }

    sort(filteredRegions.begin(), filteredRegions.end(),
        [](const Region& a, const Region& b) { return a.area > b.area; });
    int count = min(maxRegions, static_cast<int>(filteredRegions.size()));
    if (count > 0) {
        regions.assign(filteredRegions.begin(), filteredRegions.begin() + count);
    }

    for (int i = 0; i < regions.size(); i++) {
        regions[i].id = i + 1;
    }

    return regions;
}

/*
  binaryImage : input binary image for visualization background
  regions : vector of Region objects to display
  showCentroids : whether to draw centroid markers and crosshairs
  showBoundingBoxes : whether to draw bounding boxes around regions

  Creates a visualization map with region boundaries, centroids,
  and an overlay for analyzed connected components.
*/
Mat createRegionMap(const Mat& binaryImage,
    const vector<Region>& regions,
    bool showCentroids,
    bool showBoundingBoxes) {

    Mat regionMap;
    cvtColor(binaryImage, regionMap, COLOR_GRAY2BGR);
    Mat invertedBinary;
    bitwise_not(binaryImage, invertedBinary);
    Mat labels, stats, centroids;
    connectedComponentsWithStats(invertedBinary, labels, stats, centroids, 8);
    Mat coloredMap = createColoredRegionMap(regions, labels, binaryImage.size());
    Mat temp;
    addWeighted(regionMap, 0.5, coloredMap, 0.5, 0, temp);

    for (const auto& region : regions) {
        if (showBoundingBoxes) {
            rectangle(temp, region.boundingBox, Scalar(0, 255, 0), 2);

            int cornerSize = 8;
            rectangle(temp,
                Point(region.boundingBox.x, region.boundingBox.y),
                Point(region.boundingBox.x + cornerSize, region.boundingBox.y + cornerSize),
                Scalar(0, 255, 0), -1);
            rectangle(temp,
                Point(region.boundingBox.x + region.boundingBox.width - cornerSize, region.boundingBox.y),
                Point(region.boundingBox.x + region.boundingBox.width, region.boundingBox.y + cornerSize),
                Scalar(0, 255, 0), -1);
            rectangle(temp,
                Point(region.boundingBox.x, region.boundingBox.y + region.boundingBox.height - cornerSize),
                Point(region.boundingBox.x + cornerSize, region.boundingBox.y + region.boundingBox.height),
                Scalar(0, 255, 0), -1);
            rectangle(temp,
                Point(region.boundingBox.x + region.boundingBox.width - cornerSize, region.boundingBox.y + region.boundingBox.height - cornerSize),
                Point(region.boundingBox.x + region.boundingBox.width, region.boundingBox.y + region.boundingBox.height),
                Scalar(0, 255, 0), -1);
        }

        if (showCentroids) {
            circle(temp, region.centroid, 6, Scalar(255, 0, 0), -1);
            circle(temp, region.centroid, 10, Scalar(255, 255, 255), 2);

            line(temp,
                Point(region.centroid.x - 15, region.centroid.y),
                Point(region.centroid.x + 15, region.centroid.y),
                Scalar(255, 255, 255), 2);
            line(temp,
                Point(region.centroid.x, region.centroid.y - 15),
                Point(region.centroid.x, region.centroid.y + 15),
                Scalar(255, 255, 255), 2);
        }

        string info = "Obj " + to_string(region.id) + " (Area:" + to_string(region.area) + ")";
        int baseline = 0;
        Size textSize = getTextSize(info, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

        rectangle(temp,
            Point(region.boundingBox.x, region.boundingBox.y - textSize.height - 5),
            Point(region.boundingBox.x + textSize.width, region.boundingBox.y),
            Scalar(0, 0, 0), -1);

        putText(temp, info,
            Point(region.boundingBox.x, region.boundingBox.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        string centroidInfo = "(" + to_string(region.centroid.x) + "," +
            to_string(region.centroid.y) + ")";
        textSize = getTextSize(centroidInfo, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        rectangle(temp,
            Point(region.centroid.x + 10, region.centroid.y - textSize.height / 2),
            Point(region.centroid.x + 10 + textSize.width, region.centroid.y + textSize.height / 2),
            Scalar(0, 0, 0), -1);

        putText(temp, centroidInfo,
            Point(region.centroid.x + 10, region.centroid.y + textSize.height / 4),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 0), 1);
    }
    return temp;
}

/*
  regions : vector of Region objects with color assignments
  labels : connected components label matrix from analysis
  imageSize : dimensions of the original image for output map

  creates color-coded visualization map where each detected region is
  filled with its assigned random color for clear visual distinction.
*/
Mat createColoredRegionMap(const vector<Region>& regions,
    const Mat& labels,
    const Size& imageSize) {

    Mat coloredMap = Mat::zeros(imageSize, CV_8UC3);

    vector<int> labelToRegionId(labels.rows * labels.cols, -1);
    for (const auto& region : regions) {
        if (region.centroid.x >= 0 && region.centroid.x < labels.cols &&
            region.centroid.y >= 0 && region.centroid.y < labels.rows) {
            int originalLabel = labels.at<int>(region.centroid.y, region.centroid.x);
            if (originalLabel > 0 && originalLabel < labelToRegionId.size()) {
                labelToRegionId[originalLabel] = region.id;
            }
        }
    }

    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label > 0 && label < labelToRegionId.size() && labelToRegionId[label] != -1) {
                int regionIndex = labelToRegionId[label] - 1;
                if (regionIndex >= 0 && regionIndex < regions.size()) {
                    coloredMap.at<Vec3b>(y, x) = Vec3b(
                        regions[regionIndex].color[0],
                        regions[regionIndex].color[1],
                        regions[regionIndex].color[2]
                    );
                }
            }
        }
    }

    return coloredMap;
}
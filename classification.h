/*
  Nihal Sandadi

  Header file for object classification functions using both handcrafted features
  and CNN for one-shot classification.
*/

#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include "trainingData.h"

/*
  Holds the classification results: predicted label, distance to nearest
  neighbor, and if object is unknown.
*/
struct ClassificationResult {
    std::string label;
    double distance;
    bool isUnknown;
};

ClassificationResult classifyObject(const std::vector<double>& features,
    const std::vector<TrainingSample>& trainingData,
    double distanceThreshold = 2.0);

ClassificationResult classifyObjectCNN(const std::vector<float>& cnnEmbedding,
    const std::vector<TrainingSample>& trainingData,
    float distanceThreshold = 100000.0f);

#endif
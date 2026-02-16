/*
  Nihal Sandadi

  Classification algorithms for object recognition classic features and 
  CNN embeddings.
*/

#include "classification.h"
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

/*
  features : vector of 4 classic features from region analysis
  trainingData : collection of training samples for comparison
  distanceThreshold : maximum allowed distance

  Classifies objects using weighted scaled euclidean distance for classic features
*/
ClassificationResult classifyObject(const vector<double>& features,
    const vector<TrainingSample>& trainingData,
    double distanceThreshold) {
    ClassificationResult result;
    result.isUnknown = true;
    result.distance = numeric_limits<double>::max();

    if (trainingData.empty() || features.empty()) {
        result.label = "Unknown";
        return result;
    }

    vector<double> stdDevs(features.size(), 1.0);

    if (trainingData.size() > 1) {
        for (size_t i = 0; i < features.size(); i++) {
            double mean = 0.0;
            for (const auto& sample : trainingData) {
                mean += sample.features[i];
            }
            mean /= trainingData.size();

            double variance = 0.0;
            for (const auto& sample : trainingData) {
                variance += pow(sample.features[i] - mean, 2);
            }
            variance /= trainingData.size();
            stdDevs[i] = sqrt(variance);

            if (stdDevs[i] < 0.001) stdDevs[i] = 1.0;
        }
    }

    vector<double> weights = { 1.0, 1.0, 1.0, 2.0 };

    double minDistance = numeric_limits<double>::max();
    string bestLabel = "Unknown";

    for (const auto& sample : trainingData) {
        double distance = 0.0;

        for (size_t i = 0; i < features.size(); i++) {
            double diff = (features[i] - sample.features[i]) / stdDevs[i];
            distance += weights[i] * diff * diff;
        }
        distance = sqrt(distance);

        if (distance < minDistance) {
            minDistance = distance;
            bestLabel = sample.label;
        }
    }

    double adjustedThreshold = distanceThreshold * 1.5;

    result.label = bestLabel;
    result.distance = minDistance;
    result.isUnknown = (minDistance > adjustedThreshold);

    return result;
}

/*
  cnnEmbedding : feature vector for CNN processing
  trainingData : collection of training samples with CNN embeddings
  distanceThreshold : maximum allowed distance 

  Classifies objects using L2 distance on CNN embeddings for one-shot
  recognition with deep feature representations.
*/
ClassificationResult classifyObjectCNN(const std::vector<float>& cnnEmbedding,
    const std::vector<TrainingSample>& trainingData,
    float distanceThreshold) {
    ClassificationResult result;
    result.isUnknown = true;
    result.distance = std::numeric_limits<float>::max();

    if (trainingData.empty() || cnnEmbedding.empty()) {
        result.label = "Unknown";
        return result;
    }

    float minDistance = std::numeric_limits<float>::max();
    std::string bestLabel = "Unknown";

    for (const auto& sample : trainingData) {
        if (sample.cnnEmbedding.empty()) continue;

        float distance = 0.0f;
        for (size_t i = 0; i < cnnEmbedding.size(); i++) {
            float diff = cnnEmbedding[i] - sample.cnnEmbedding[i];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);

        if (distance < minDistance) {
            minDistance = distance;
            bestLabel = sample.label;
        }
    }

    result.label = bestLabel;
    result.distance = minDistance;
    result.isUnknown = (minDistance > distanceThreshold);

    return result;
}
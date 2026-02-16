/*
  Nihal Sandadi

  Header file for training data including storage, JSON,
  and training status display for object recognition system.
*/

#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "regionFeatures.h"

/*
  training sample storing both classic features and CNN embeddings
  with metadata.
*/
struct TrainingSample {
    std::string label;
    std::vector<double> features;
    std::vector<float> cnnEmbedding;
    std::string timestamp;
};

bool saveTrainingData(const std::vector<TrainingSample>& samples, const std::string& filename);
TrainingSample createTrainingSample(const std::string& label, const RegionFeatures& features);
void displayTrainingStatus(cv::Mat& image, const std::vector<TrainingSample>& samples, bool waitingForInput = false);

#endif
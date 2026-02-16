/*
  Nihal Sandadi

  Implementation of training data management (JSON), timestamp, 
  and training visualization for object recognition.
*/

#include "trainingData.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
  Generates current timestamp string in standardized format for training sample metadata.
*/
string getCurrentTimestamp() {
    time_t now = time(0);
    struct tm localTime;
    localtime_s(&localTime, &now);
    stringstream ss;
    ss << put_time(&localTime, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

/*
  label : object label provided by user
  features : computed region features from object analysis

  Creates a training sample with classic features and timestamp, initializing
  CNN embedding for training capture.
*/
TrainingSample createTrainingSample(const std::string& label, const RegionFeatures& features) {
    TrainingSample sample;
    sample.label = label;
    sample.timestamp = getCurrentTimestamp();
    sample.features = {
        features.percentFilled,
        features.aspectRatio,
        features.elongation,
        features.huMoments[0]
    };
    sample.cnnEmbedding.clear();
    return sample;
}

/*
  input : input string might have special JSON characters

  removes special JSON characters in input strings.
*/
string escapeJsonString(const string& input) {
    string output;
    for (char c : input) {
        switch (c) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default: output += c; break;
        }
    }
    return output;
}

/*
  samples : vector of TrainingSample objects
  filename : output JSON file path for saving training data

  writes training data to JSON format including classic features,
  CNN embeddings, and timestamp.
*/
bool saveTrainingData(const vector<TrainingSample>& samples, const string& filename) {
    try {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file for writing: " << filename << endl;
            return false;
        }
        // creating json
        file << "{" << endl;
        file << "  \"version\": \"1.0\"," << endl;
        file << "  \"created\": \"" << getCurrentTimestamp() << "\"," << endl;
        file << "  \"feature_names\": [\"percent_filled\", \"aspect_ratio\", \"elongation\", \"hu_moment_1\"]," << endl;
        file << "  \"total_samples\": " << samples.size() << "," << endl;
        file << "  \"samples\": [" << endl;
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& sample = samples[i];
            file << "    {" << endl;
            file << "      \"label\": \"" << escapeJsonString(sample.label) << "\"," << endl;
            file << "      \"timestamp\": \"" << escapeJsonString(sample.timestamp) << "\"," << endl;
            file << "      \"features\": [";
            for (size_t j = 0; j < sample.features.size(); ++j) {
                file << fixed << setprecision(6) << sample.features[j];
                if (j < sample.features.size() - 1) {
                    file << ", ";
                }
            }
            file << "]," << endl;
            file << "      \"cnn_embedding\": [";
            for (size_t j = 0; j < sample.cnnEmbedding.size(); ++j) {
                file << fixed << setprecision(6) << sample.cnnEmbedding[j];
                if (j < sample.cnnEmbedding.size() - 1) {
                    file << ", ";
                }
            }
            file << "]" << endl;
            if (i < samples.size() - 1) {
                file << "    }," << endl;
            }
            else {
                file << "    }" << endl;
            }
        }
        file << "  ]" << endl;
        file << "}" << endl;
        file.close();

        cout << "Training data saved to: " << filename << endl;
        return true;
    }
    catch (const exception& e) {
        cerr << "Error saving training data: " << e.what() << endl;
        return false;
    }
}

/*
  image : output image for status display
  samples : collection of training samples
  waitingForInput : flag if we're waiting user label input

  Displays training mode infor on the video feed including the
  sample count, recent labels, and control instructions for user guidance.
*/
void displayTrainingStatus(cv::Mat& image, const std::vector<TrainingSample>& samples, bool waitingForInput) {
    int yPos = 250;
    cv::putText(image, "=== TRAINING MODE ===", cv::Point(10, yPos),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    yPos += 30;
    if (waitingForInput) {
        cv::putText(image, "ENTER LABEL IN CONSOLE...", cv::Point(10, yPos),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        yPos += 30;
    }
    cv::putText(image, "Collected Samples: " + std::to_string(samples.size()),
        cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    yPos += 20;
    int startIdx = std::max(0, (int)samples.size() - 3);
    for (int i = startIdx; i < samples.size(); i++) {
        std::string sampleInfo = "Sample " + std::to_string(i + 1) + ": " + samples[i].label;
        cv::putText(image, sampleInfo, cv::Point(15, yPos),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 200, 255), 1);
        yPos += 15;
    }
    yPos = image.rows - 80;
    cv::putText(image, "TRAINING CONTROLS:", cv::Point(10, yPos),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 100), 1);
    yPos += 20;
    cv::putText(image, "n: Save current object with label", cv::Point(15, yPos),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    yPos += 15;
    cv::putText(image, "s: Save training data to file", cv::Point(15, yPos),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    yPos += 15;
    cv::putText(image, "t: Exit training mode", cv::Point(15, yPos),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}
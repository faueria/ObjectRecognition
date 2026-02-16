/*
  Nihal Sandadi

  Main application for real time object recognition system with dual classification
  (classic features + CNN embeddings) and region analysis.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "thresholding.h"
#include "morphological.h"
#include "regionAnalysis.h"
#include "regionFeatures.h"
#include "trainingData.h"
#include "classification.h"
#include "utilities.h"
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

/*
  Gets object label from user input thru console for training samples.
*/
string getLabelFromUser() {
    string label;
    cout << "Enter label for this object: ";
    getline(cin, label);
    return label;
}

/*
  Main loop which is in charge of the windows and processing the video feed
*/
int main() {
    Mat frame, thresholded, cleaned, regionMap, featureDisplay;
    int mode = 0;
    bool useMorphologicalClean = true;
    bool showRegionAnalysis = true;
    bool showFeatures = true;
    bool trainingMode = false;
    bool ignoreBoundaryRegions = true;
    bool waitingForLabelInput = false;
    int minArea = 1000;
    int maxRegions = 5;
    // this is for classic feature recognition
    vector<TrainingSample> trainingSamples;
    string trainingFilename = "C:\\Users\\Nihal Sandadi\\Desktop\\training_data.json";
    double classificationThreshold = 2.0;
    // this is for the cnn
    cv::dnn::Net cnnNet;
    std::string modelPath = "C:\\Users\\Nihal Sandadi\\Desktop\\computer vision\\hw3\\ObjectRecognition\\ObjectRecognition\\resnet18-v2-7.onnx";

    cout << "Initializing webcam..." << endl;
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open webcam" << endl;
        return -1;
    }
    cout << "Webcam started!" << endl;

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    try {
        std::ifstream fileCheck(modelPath);
        if (!fileCheck.good()) {
            cout << "CNN model file not found: " << modelPath << endl;
        }
        else {
            cout << "CNN model file found: " << modelPath << endl;
            cnnNet = cv::dnn::readNetFromONNX(modelPath);
            cout << "CNN model loaded successfully!" << endl;
        }
    }
    catch (const std::exception& e) {
        cout << "Error loading CNN model: " << e.what() << endl;
    }

    namedWindow("Original Video", WINDOW_AUTOSIZE);
    namedWindow("Thresholded Video", WINDOW_AUTOSIZE);
    namedWindow("Cleaned Video", WINDOW_AUTOSIZE);
    namedWindow("Region Analysis", WINDOW_AUTOSIZE);
    namedWindow("Region Features", WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Error: Captured empty frame" << endl;
            break;
        }

        if (mode == 0) {
            thresholded = grayscaleThreshold(frame);
        }
        else {
            thresholded = customThreshold(frame);
        }

        if (useMorphologicalClean) {
            cleaned = enhancedCleanThreshold(thresholded);
        }
        else {
            cleaned = basicCleanThreshold(thresholded);
        }

        vector<Region> regions;
        vector<RegionFeatures> regionFeatures;
        if (showRegionAnalysis) {
            regions = analyzeRegions(cleaned, minArea, maxRegions, ignoreBoundaryRegions);
            regionMap = createRegionMap(cleaned, regions, true, true);

            // if check on whether to show features for objects in the region
            if (showFeatures && !regions.empty()) {
                for (const auto& region : regions) {
                    Mat regionMask = Mat::zeros(cleaned.size(), CV_8UC1);
                    Mat labels, stats, centroids;
                    Mat invertedBinary;
                    bitwise_not(cleaned, invertedBinary);
                    connectedComponentsWithStats(invertedBinary, labels, stats, centroids, 8);

                    if (region.centroid.x >= 0 && region.centroid.x < labels.cols &&
                        region.centroid.y >= 0 && region.centroid.y < labels.rows) {
                        int originalLabel = labels.at<int>(region.centroid.y, region.centroid.x);
                        regionMask = (labels == originalLabel);
                    }

                    RegionFeatures features = computeRegionFeatures(regionMask, region.id);
                    regionFeatures.push_back(features);

                    drawRegionFeatures(regionMap, features, region.color);

                    if (!trainingMode && !trainingSamples.empty()) {
                        vector<double> currentFeatures = {
                            features.percentFilled,
                            features.aspectRatio,
                            features.elongation,
                            features.huMoments[0]
                        };

                        ClassificationResult result = classifyObject(currentFeatures, trainingSamples, classificationThreshold);
                        string classificationText = result.isUnknown ? "Unknown" : result.label;
                        Scalar color = result.isUnknown ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
                        putText(regionMap, "Class: " + classificationText,
                            Point(features.centroidX - 50, features.centroidY - 30),
                            FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                        putText(regionMap, "Dist: " + to_string(result.distance).substr(0, 5),
                            Point(features.centroidX - 50, features.centroidY - 60),
                            FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

                        if (!cnnNet.empty()) {
                            try {
                                cv::Mat embeddingImage;
                                prepEmbeddingImage(frame, embeddingImage,
                                    features.centroidX, features.centroidY,
                                    features.orientedBoundingBox.angle * CV_PI / 180.0,
                                    -features.orientedBoundingBox.size.width / 2, features.orientedBoundingBox.size.width / 2,
                                    -features.orientedBoundingBox.size.height / 2, features.orientedBoundingBox.size.height / 2,
                                    0);

                                cv::Mat embedding;
                                getEmbedding(embeddingImage, embedding, cnnNet, 0);

                                std::vector<float> cnnEmbedding;
                                cnnEmbedding.assign((float*)embedding.datastart, (float*)embedding.dataend);

                                ClassificationResult cnnResult = classifyObjectCNN(cnnEmbedding, trainingSamples, 100000.0f);

                                std::string cnnClassificationText = cnnResult.isUnknown ? "CNN: Unknown" : "CNN: " + cnnResult.label;
                                cv::Scalar cnnColor = cnnResult.isUnknown ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);

                                putText(regionMap, cnnClassificationText,
                                    Point(features.centroidX - 50, features.centroidY + 30),
                                    FONT_HERSHEY_SIMPLEX, 0.6, cnnColor, 2);

                                putText(regionMap, "CNN Dist: " + std::to_string(cnnResult.distance).substr(0, 8),
                                    Point(features.centroidX - 50, features.centroidY + 60),
                                    FONT_HERSHEY_SIMPLEX, 0.5, cnnColor, 1);

                            }
                            catch (const std::exception& e) {
                            }
                        }
                    }
                }

                featureDisplay = createFeatureDisplay(regionFeatures, Size(400, 300));
            }
            else {
                cvtColor(cleaned, regionMap, COLOR_GRAY2BGR);
                featureDisplay = Mat::zeros(Size(400, 300), CV_8UC3);
                putText(featureDisplay, "Feature display disabled", Point(50, 150),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
            }
        }
        else {
            cvtColor(cleaned, regionMap, COLOR_GRAY2BGR);
            featureDisplay = Mat::zeros(Size(400, 300), CV_8UC3);
            putText(featureDisplay, "Region analysis disabled", Point(50, 150),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        }

        string modeText = (mode == 0) ? "Grayscale" : "Custom";
        string cleanText = useMorphologicalClean ? "Morph Clean" : "Basic Clean";
        string regionText = showRegionAnalysis ? "ON" : "OFF";
        string featureText = showFeatures ? "ON" : "OFF";
        string trainingText = trainingMode ? "TRAINING MODE" : "CLASSIFICATION MODE";
        string boundaryText = ignoreBoundaryRegions ? "Ignore Boundary" : "All Regions";

        putText(frame, "Mode: " + modeText, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        putText(frame, "Cleaning: " + cleanText, Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);

        if (showRegionAnalysis) {
            putText(frame, "Region Analysis: " + regionText, Point(10, 90),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2);
            putText(frame, "Features: " + featureText, Point(10, 120),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            putText(frame, trainingText, Point(10, 150),
                FONT_HERSHEY_SIMPLEX, 0.6, trainingMode ? Scalar(0, 255, 0) : Scalar(255, 255, 0), 2);
            putText(frame, "Min Area: " + to_string(minArea), Point(10, 180),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            putText(frame, boundaryText, Point(10, 210),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 100), 2);
        }

        if (trainingMode) {
            displayTrainingStatus(frame, trainingSamples, waitingForLabelInput);
        }

        string instructions = "g/c: Modes | m: Cleaning | r: Regions | f: Features | t: Training | +/-: Area | q: Quit";
        if (trainingMode) {
            instructions += " | n: Save Object | s: Save Data";
        }
        putText(frame, instructions, Point(10, frame.rows - 10),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);

        imshow("Original Video", frame);
        imshow("Thresholded Video", thresholded);
        imshow("Cleaned Video", cleaned);
        imshow("Region Analysis", regionMap);
        imshow("Region Features", featureDisplay);

        // checking for key presses to modify current mode/save/record objects
        char key = waitKey(1);
        if (key == 'q' || key == 'Q') {
            break;
        }
        else if (key == 'g' || key == 'G') {
            mode = 0;
            cout << "Switched to grayscale thresholding" << endl;
        }
        else if (key == 'c' || key == 'C') {
            mode = 1;
            cout << "Switched to custom thresholding" << endl;
        }
        else if (key == 'm' || key == 'M') {
            useMorphologicalClean = !useMorphologicalClean;
            cout << "Morphological cleaning: " << (useMorphologicalClean ? "ENABLED" : "DISABLED") << endl;
        }
        else if (key == 'r' || key == 'R') {
            showRegionAnalysis = !showRegionAnalysis;
            cout << "Region analysis: " << (showRegionAnalysis ? "ENABLED" : "DISABLED") << endl;
        }
        else if (key == 'f' || key == 'F') {
            showFeatures = !showFeatures;
            cout << "Feature computation: " << (showFeatures ? "ENABLED" : "DISABLED") << endl;
        }
        else if (key == 't' || key == 'T') {
            trainingMode = !trainingMode;
            cout << "Training mode: " << (trainingMode ? "ENABLED" : "DISABLED") << endl;
            if (trainingMode) {
                cout << "Press 'n' to save objects with labels, 's' to save data" << endl;
            }
            else {
                cout << "Classification mode active" << endl;
            }
        }
        else if (key == '+' || key == '=') {
            minArea += 100;
            cout << "Min region area: " << minArea << endl;
        }
        else if (key == '-' || key == '_') {
            minArea = max(100, minArea - 100);
            cout << "Min region area: " << minArea << endl;
        }
        else if (trainingMode && (key == 'n' || key == 'N')) {
            if (!regionFeatures.empty()) {
                waitingForLabelInput = true;
                destroyAllWindows();

                string label = getLabelFromUser();

                if (!label.empty()) {
                    TrainingSample sample = createTrainingSample(label, regionFeatures[0]);

                    if (!cnnNet.empty()) {
                        try {
                            cv::RotatedRect obb = regionFeatures[0].orientedBoundingBox;

                            cv::Mat embeddingImage;
                            prepEmbeddingImage(frame, embeddingImage,
                                regionFeatures[0].centroidX, regionFeatures[0].centroidY,
                                obb.angle * CV_PI / 180.0,
                                -obb.size.width / 2, obb.size.width / 2,
                                -obb.size.height / 2, obb.size.height / 2,
                                0);

                            cv::Mat embedding;
                            getEmbedding(embeddingImage, embedding, cnnNet, 0);

                            sample.cnnEmbedding.assign((float*)embedding.datastart,
                                (float*)embedding.dataend);

                            cout << "CNN embedding captured! Size: " << sample.cnnEmbedding.size() << endl;

                        }
                        catch (const std::exception& e) {
                            cout << "CNN embedding failed: " << e.what() << endl;
                        }
                    }

                    trainingSamples.push_back(sample);
                    cout << "Saved training sample for '" << label << "'" << endl;

                    if (!sample.cnnEmbedding.empty()) {
                        cout << "CNN embedding size: " << sample.cnnEmbedding.size() << endl;
                    }
                }
                else {
                    cout << "No label provided, sample not saved." << endl;
                }

                waitingForLabelInput = false;
                namedWindow("Original Video", WINDOW_AUTOSIZE);
                namedWindow("Thresholded Video", WINDOW_AUTOSIZE);
                namedWindow("Cleaned Video", WINDOW_AUTOSIZE);
                namedWindow("Region Analysis", WINDOW_AUTOSIZE);
                namedWindow("Region Features", WINDOW_AUTOSIZE);
            }
            else {
                cout << "No regions detected to save!" << endl;
            }
        }
        else if (trainingMode && (key == 's' || key == 'S')) {
            if (saveTrainingData(trainingSamples, trainingFilename)) {
                cout << "Training data saved successfully!" << endl;
            }
        }
    }

    cap.release();
    destroyAllWindows();

    cout << "Application ended successfully" << endl;
    return 0;
}
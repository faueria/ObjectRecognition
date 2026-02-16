/*
  Nihal Sandadi

  Implementation of morphological operations for cleaning binary images,
  including dilation/erosion combinations and noise removal techniques.
*/

#include "morphological.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
  thresholded : binary image from thresholding operation

  Applies morphological dilation (8x8) followed by erosion (4x4),
  better for curved objects, like real world applications
*/
Mat morphologicalClean(const Mat& thresholded) {
    Mat cleaned = thresholded.clone();

    Mat kernel_dilate_8 = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
    Mat kernel_erode_4 = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));

    dilate(cleaned, cleaned, kernel_dilate_8);
    erode(cleaned, cleaned, kernel_erode_4);

    return cleaned;
}

/*
  thresholded : binary image from thresholding operation

  Applies morphological dilation/erosion, this is good for more noisy images
*/
Mat enhancedCleanThreshold(const Mat& thresholded) {
    Mat cleaned = thresholded.clone();

    Mat kernel_small = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(cleaned, cleaned, MORPH_OPEN, kernel_small);
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel_small);

    cleaned = morphologicalClean(cleaned);

    return cleaned;
}

/*
  thresholded : binary input image from thresholding operation

  Applies minimal morphological opening and closing with small element.
*/
Mat basicCleanThreshold(const Mat& thresholded) {
    Mat cleaned = thresholded.clone();

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(cleaned, cleaned, MORPH_OPEN, kernel);
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel);

    return cleaned;
}
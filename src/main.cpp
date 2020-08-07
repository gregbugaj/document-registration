#include <iostream>
#include <set>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <version.h>
#include<string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace mxnet::cpp;
using namespace std::chrono;
using namespace cv;

int iterators_mxnet();

void detectEdges(cv::Mat &imgGray, cv::Mat &edges);

void detectContours(cv::Mat &src);

Mat findSquares(const Mat &image, std::vector<std::vector<cv::Point>> &squares);

void alignForms(const std::string &src, const std::string &target);

void alignRegistrationPoints(Mat &refImage, Mat &targetImage, Mat &alignmentImage, Mat &h);

/*The global context, change them if necessary*/
static mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);
// static Context global_ctx(mxnet::cpp::kCPU,0);

/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char *source_window = "Source image";
char *corners_window = "Corners detected";

/// Function header
void cornerHarris_demo(int, void *);

/** @function cornerHarris_demo */
void cornerHarris_demo(int, void *) {

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros(src.size(), CV_32FC1);

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    /// Normalizing
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    /// Drawing a circle around corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int) dst_norm.at<float>(j, i) > thresh) {
                circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    /// Showing the result
    namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
    imshow(corners_window, dst_norm_scaled);
}


/** @function main */
int mainXX(int argc, char **argv) {
    /// Load source image and convert it to gray
//    src = imread( "/home/greg/dev/document-registration/test-deck/form-cms1500-1.png", 1 );
    src = imread("/tmp/ips/refSquareImageXXX.png", 1);
    cvtColor(src, src_gray, CV_BGR2GRAY);

    /// Create a window and a trackbar
    namedWindow(source_window, CV_WINDOW_AUTOSIZE);
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
    imshow(source_window, src);

    cornerHarris_demo(0, 0);

    waitKey(0);
    return (0);
}

int main(int argc, char const *argv[]) {
    int version;
    MXGetVersion(&version);

    LG << "Commit : " << GIT_COMMIT_HASH;
    LG << "MxNet     version : " << version;
    LG << "Leptonica version : " << getLeptonicaVersion();
    LG << "OpenCV    version : " << CV_VERSION;

    cv::Mat imgOriginal;
    cv::Mat imgGrayscale;
    cv::Mat imgBinarized;

/*    //3508 x 2480
    Size size(2480, 3508);// 2480 x 3508
    Mat dst;//dst image
    Mat src =  cv::imread("/home/greg/dev/document-registration/test-deck/form-cms1500-1.png");//src image
    resize(src,dst,size);//resize image
    cv::imwrite("/home/greg/dev/document-registration/test-deck/form-cms1500-2.png", dst);
*/

    // "/home/greg/dev/document-registration/test-deck/form-cms1500-1.png"
    alignForms("/home/greg/dev/document-registration/test-deck/form-cms1500-2.png",
               "/home/greg/dev/document-registration/test-deck/dd-allstate/269698_202006290005720_001.tif");

//    imgOriginal = cv::imread("/home/greg/dev/document-registration/test-deck/hicfa1500.jpg");
//    imgOriginal = cv::imread("/home/greg/dev/document-registration/test-deck/form-cms1500-1.png");

    if (false) {
        imgOriginal = cv::imread(
                "/home/greg/dev/document-registration/test-deck/dd-allstate/269698_202006290005720_001.tif");
        if (imgOriginal.empty()) {
            std::cout << "error: image not preprocess from file\n\n";
            return (0);
        }

        cv::cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);
        cv::threshold(imgGrayscale, imgBinarized, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

        cv::imwrite("/home/greg/dev/document-registration/test-deck/hicfa1500-binary.png", imgBinarized);

        cv::Mat edges;
        detectEdges(imgBinarized, edges);
        cv::imwrite("/home/greg/dev/document-registration/test-deck/hicfa1500-edges.png", edges);
        //detectContours(edges);

        std::vector<std::vector<cv::Point>> squares;
        findSquares(imgGrayscale, squares);
    }

    return 0;
}

/**
 * Load and preprocess image
 * @param filename
 * @return
 */
Mat preprocess(const std::string &filename) {
    Mat gray;
//    Mat bin;
    Mat src = imread(filename);
    cv::cvtColor(src, gray, CV_BGR2GRAY);
//    cv::threshold(gray, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    Mat pyr;
    pyrDown(gray, pyr, Size(src.cols / 2, src.rows / 2));
    pyrUp(pyr, gray, src.size());

    return gray;
}

void cornerHarris_label(Mat &src_gray) {
    int thresh = 1;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros(src_gray.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
/*    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int) dst_norm.at<float>(i, j) > thresh) {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }*/

    imwrite("/tmp/ips/dst.png", dst);
}


void alignForms(const std::string &refFilename, const std::string &targetFilename) {
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    std::cout << "Reading reference image : " << refFilename << std::endl;
    std::cout << "Reading target image : " << targetFilename << std::endl;

    Mat refImage = preprocess(refFilename);
    Mat targetImage = preprocess(targetFilename);

    imwrite("/tmp/ips/refImage.png", refImage);
    imwrite("/tmp/ips/targetImage.png", targetImage);

    cv::Mat refEdged;
    cv::Mat targetEdges;

    detectEdges(refImage, refEdged);
    detectEdges(targetImage, targetEdges);

    std::vector<std::vector<cv::Point>> refSquares;
    std::vector<std::vector<cv::Point>> targetSquares;

    detectContours(targetEdges);

    Mat refSquareImage = findSquares(refEdged, refSquares);
    Mat targetSquareImage = findSquares(targetEdges, targetSquares);

    imwrite("/tmp/ips/refSquareImage.png", refSquareImage);
    imwrite("/tmp/ips/targetSquareImage.png", targetSquareImage);

    imwrite("/tmp/ips/refImage.png", refImage);
    imwrite("/tmp/ips/targetImage.png", targetImage);

    imwrite("/tmp/ips/refEdged.png", refEdged);
    imwrite("/tmp/ips/targetEdges.png", targetEdges);

    // Registered image will be stored in aligmentImage.
    // The estimated homography will be stored in h.
    Mat alignmentImage, h;
    // Align images
    std::cout << "Aligning images ..." << std::endl;
    alignRegistrationPoints(refSquareImage, targetSquareImage, alignmentImage, h);
    cv::imwrite("/tmp/ips/alignmentImage.png", alignmentImage);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took " << time_span.count() << " seconds.";
    std::cout << std::endl;
}

void matchFeatures(const cv::Mat &query, const cv::Mat &target,
                   std::vector<cv::DMatch> &goodMatches) {
    float RATIO = .75f;
    std::vector<std::vector<cv::DMatch>> matches;
//    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    // Find 2 best matches for each descriptor to make later the second neighbor test.
    matcher.knnMatch(query, target, matches, 2);
    // Second neighbor ratio test.
    /*   for (unsigned int i = 0; i < matches.size(); ++i) {
           if (matches[i][0].distance < matches[i][1].distance * RATIO)
               goodMatches.push_back(matches[i][0]);
       }*/
}

void alignRegistrationPoints(Mat &refImage, Mat &targetImage, Mat &alignmentImage, Mat &h) {
    float GOOD_MATCH_PERCENT = 0.25f;
    // Convert images to grayscale
    Mat im1Gray = refImage, im2Gray = targetImage;

    // Variables to store keypoints and descriptors
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    // Detect ORB features and compute descriptors.
    Ptr<Feature2D> detector = ORB::create(500);
    detector->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1, false);
    detector->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2, false);

    cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);

    cv::Mat kpRefImage(refImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    drawKeypoints(refImage, keypoints1, kpRefImage, color, DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imwrite("/tmp/ips/kpRefImage.jpg", kpRefImage);

    cv::Mat kpTargetImage(targetImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    drawKeypoints(targetImage, keypoints2, kpTargetImage, color, DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imwrite("/tmp/ips/kpTargetImage.jpg", kpTargetImage);

    // Match features.
    std::vector<cv::DMatch> matches;
    auto matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());

/*    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    // Match features.
    matchFeatures(descriptors1, descriptors2, matches);*/

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Draw top matches
    Mat imMatches;
    drawMatches(refImage, keypoints1, targetImage, keypoints2, matches, imMatches);
    imwrite("/tmp/ips/matches.jpg", imMatches);

    // Extract location of good matches
    std::vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Find homography RANSAC LMEDS
//    h = findHomography(points1, points2, LMEDS);
    h = findHomography(points1, points2, RANSAC, 5.0);
    // Use homography to warp image
    warpPerspective(refImage, alignmentImage, h, targetImage.size());
}

/**
 * finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
 * @param pt1
 * @param pt2
 * @param pt0
 * @return
 */
double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

/**
 * returns sequence of squares detected on the image.
 * @param image
 * @param squares
 */
Mat findSquares(const Mat &image, std::vector<std::vector<cv::Point>> &squares) {
    squares.clear();
    Mat gray, gray0;
    std::vector<std::vector<cv::Point >> contours;

    // find contours and store them all as a list
    cv::findContours(image, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> approx;

    // test each contour
    for (size_t i = 0; i < contours.size(); i++) {
        // approximate contour with accuracy proportional  to the contour perimeter
        cv::approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because area may be positive or negative - in accordance with the contour orientation

        if (approx.size() == 4 &&
            fabs(contourArea(Mat(approx))) > 500 &&
            isContourConvex(Mat(approx))) {
            double maxCosine = 0;

            for (int j = 2; j < 5; j++) {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            // if cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence
            if (maxCosine < 0.1) {
                squares.push_back(approx);
            }
        }
    }

    // shows all the squares in the image
    cv::Mat squareImage(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < squares.size() - 1; i++) {
        cv::Rect r = cv::boundingRect(cv::Mat(squares[i]));
        cv::Mat s = image(r);
//        std::cout << r.x << " :: " << r.y << std::endl;
        /*
         std::stringstream temp_stream;
        temp_stream << "/home/greg/dev/document-registration/test-deck/squared/square" << " - " << i << ".jpg";
        std::cout << r;
        */
        cv::Rect r2 = cv::Rect_(r.x, r.y, 2, 2);
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
        cv::rectangle(squareImage, r, color);
    }

    cv::imwrite("/home/greg/dev/document-registration/test-deck/squareImage.png", squareImage);
    return squareImage;
}

void detectEdges(cv::Mat &imgGray, cv::Mat &edges) {
/*    cv::Mat imgBlurred;
    cv::GaussianBlur(imgGray, imgBlurred, cv::Size(3, 3), 0);
    cv::Canny(imgBlurred, edges, 40, 120);*/
    int lowThreshold = 40;
    const int ratio = 3;
    const int kernel_size = 3;

    blur(imgGray, edges, Size(3, 3));
    Canny(edges, edges, lowThreshold, lowThreshold * ratio, kernel_size);
}

void detectContours(cv::Mat &src) {
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<Vec4i> hierarchy;

    //CV_RETR_CCOMP CV_RETR_EXTERNAL
    cv::findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
    }

    std::vector<cv::Point> approx;
    cv::Mat contourImage(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t idx = 0; idx < contours.size(); idx++) {
        // approximate contour with accuracy proportional  to the contour perimeter
        cv::approxPolyDP(Mat(contours[idx]), approx, arcLength(Mat(contours[idx]), false) * 0.02, false);

        std::cout << approx.size() << std::endl;
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        auto contour = contours[idx];
        auto bb = cv::boundingRect(contour);
        auto aspect = bb.width / (float) bb.height;
        auto area = bb.area();

        // approx.size() == 4 &&
        if (
                fabs(contourArea(Mat(approx))) > 500
                ) {

            cv::drawContours(contourImage, contours, idx, color);
        }
    }

    cv::imwrite("/tmp/ips/contourImage.png", contourImage);
    cv::imwrite("/tmp/ips/contourImage-dst.png", dst);
}

/*!
 * Dump existing data iterators
 * @return
 */
int iterators_mxnet() {
    mx_uint num_data_iter_creators;
    DataIterCreator *data_iter_creators = nullptr;

    int r = MXListDataIters(&num_data_iter_creators, &data_iter_creators);
    CHECK_EQ(r, 0);
    LG << "num_data_iter_creators = " << num_data_iter_creators;
    //output: num_data_iter_creators = 8

    const char *name;
    const char *description;
    mx_uint num_args;
    const char **arg_names;
    const char **arg_type_infos;
    const char **arg_descriptions;

    for (mx_uint i = 0; i < num_data_iter_creators; i++) {
        r = MXDataIterGetIterInfo(data_iter_creators[i], &name, &description,
                                  &num_args, &arg_names, &arg_type_infos,
                                  &arg_descriptions);
        CHECK_EQ(r, 0);
        LG << " i: " << i << ", name: " << name;
    }

    MXNotifyShutdown();
    return 0;
}


#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

    /**
     * lambda function to compute the nearest point in z in the vector of lidar points.
     * @param points [in] lidar point to analyze (units of m). Will be sorted in place.
     * @param sigma_to_keep number of standard deviations around the median to keep (in z)
     * @param percent_to_average percentage of the outlier filtered point to average to get the range to the target (in z).
     * @param range to the target in m.
     */
    auto compute_nearest = [](std::vector<LidarPoint> & points, double sigma_to_keep = 3.0, double percent_to_average = 5.0)->double {
        std::sort(points.begin(), points.end(), [](const LidarPoint & a, const LidarPoint & b) -> bool {
            return a.x < b.x;
        });

        double mean = 0.0;
        for (const auto & point: points) {
            mean += point.x;
        }
        mean /= points.size();
        double stddev = 0.0;
        for (const auto & point: points) {
            stddev += std::pow(point.x - mean, 2);
        }
        stddev = std::sqrt(stddev/(points.size()-1));
        double bottom_threshold = mean - sigma_to_keep * stddev;
        size_t bottom_threshold_idx = 0;

        for (bottom_threshold_idx = 0; bottom_threshold_idx < points.size(); ++bottom_threshold_idx) {
            if (points[bottom_threshold_idx].x > bottom_threshold) {
                break;
            }
        }

        size_t remaining_points = points.size() - bottom_threshold_idx;
        size_t points_to_average = remaining_points * percent_to_average / 100;
        if (points_to_average == 0) {
            points_to_average = 1;
        }

        double range = 0.0;
        for (size_t i = bottom_threshold_idx; i < bottom_threshold_idx + points_to_average; ++i) {
            range += points[i].x;
        }

        return range / points_to_average;
    };

    double curr_range = compute_nearest(lidarPointsCurr);
    double prev_range = compute_nearest(lidarPointsPrev);

    double dt = 1.0/frameRate;

    TTC = curr_range * dt / (prev_range - curr_range);
}


void matchBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    const std::vector<cv::DMatch> &matches = currFrame.kptMatches; //removed from the function signature, since it was redundant.
    bbBestMatches.clear();
    const int min_match_threshold(2);

    // This vector of vector counts how many keypoints appear in the current frame bounding box and the previous frame bounding box
    // Each keypoint that appears in a current bounding box and also appears in a previous bounding box increments the appropriate value in the 
    // vector by 1. After processing the keypoints and boxes, the pairs of current and previous boxes with the highest association score are paired
    // (if the number of matches is greater than min_match_threshold)
    std::vector<std::vector<int>> box_match_count (currFrame.boundingBoxes.size(), std::vector<int>(prevFrame.boundingBoxes.size(), 0));
    
    for ( const auto & match : matches ) {
        for (auto curr_box = 0; curr_box < currFrame.boundingBoxes.size(); ++curr_box) {

            //query index is the first frame to be called, train is the second. With the convention we are using, curr is the train index, 
            // prev is the query index
            if (currFrame.boundingBoxes[curr_box].roi.contains (currFrame.keypoints[match.trainIdx].pt)) {
                for (auto prev_box = 0; prev_box < prevFrame.boundingBoxes.size(); ++prev_box) {
                    if (prevFrame.boundingBoxes[prev_box].roi.contains (prevFrame.keypoints[match.queryIdx].pt)) {
                        box_match_count[curr_box][prev_box] += 1;
                        continue;
                    }
                }
                continue; // These continues ensure that we only match 1 keypoint match to 1 box.
            }
        }
    }

    for (auto i = 0; i < box_match_count.size(); ++i) {
        for (auto j = 0; j < box_match_count[i].size(); ++j) {
            if (box_match_count[i][j] > min_match_threshold) {
                auto idx_curr = currFrame.boundingBoxes[i].boxID;
                auto idx_prev = prevFrame.boundingBoxes[j].boxID;
                bbBestMatches[idx_curr] = idx_prev;
            }
        }
    }
}


#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"


void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);


/**
 * This method matches the bounding box between two images/dataframes. The keypoint matches that are substantially common 
 *      between two data frames are assumed contained by the same bounding box. The currFrame must have populated the kptMatches
 *      field with matches from the previous frame for this function to work.
 * @param bbBestMatches [out] This map contains a map of the best match bounding boxes betweeen the two frames, by their Dataframe bounding box vector index.
 * @param prevFrame [in] previous dataframe
 * @param currFrame [in] current dataframe 
 */
void matchBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);

/**
 * This method returns the time to collision based on lidar between two sets of lidar points that have already been associated by bounding boxes. 
 * @param lidarPointsPrev [in] a vector of lidar points from the previous image. Will be sorted in place by z.
 * @param lidarPointsCurr [in] a vector of lidar points from the current image. Will be sorted in place by z.
 * @param framerate [in] the camera capture rate in Hz.
 * @param TTC [out] Time to Collision in seconds.
 */
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);                  
#endif /* camFusion_hpp */

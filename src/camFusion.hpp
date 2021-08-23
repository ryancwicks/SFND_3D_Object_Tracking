
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"


void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);

/**
 * This method takes a set of keypoints from two frames and matches, and associates them with a bounding box. It filters the keypoints 
 *      by rejecting matches with a distance greater than 3 standard deviations from the mean of the other keypoints. It does not use a size filter, 
 *      as it should reject non-vehicle points using the Gausian filter. Be aware that the train and query indexes in the bounding box kpt 
 *      matches will still be associated with the original kptsCurr index, and not the boundingBox.keypoint indices.
 * @param boundingBox [in/out] reference to the bounding box. Keypoints and matches within this box are copied into this structure as output.
 * @param kptsCurr [in] input vector of keypoints in the current image.
 * @param kptMatches [in] input vector of keypoint matches.
 */
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);


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


/**
 * Compute TTC using camera data, filter out TTC's outside of the 3 sigma average. 
 * @param kptsPrev [in] keypoints in the previous frame (entire image)
 * @param kptsCurr [in] keypoints in the current frame (entire image)
 * @param kptMatches[in] keypoint matches that have been clustered by a ROI
 * @param frameRate [in] image frame rate in Hz.
 * @param TTC [out] time to collision in s, returned by reference.
 * @param visImage [in/out]
 */
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);

/**
 * This method returns the time to collision based on lidar between two sets of lidar points that have already been associated by bounding boxes. 
 * @param lidarPointsPrev [in] a vector of lidar points from the previous image. Will be sorted in place by z.
 * @param lidarPointsCurr [in] a vector of lidar points from the current image. Will be sorted in place by z.
 * @param framerate [in] the camera capture rate in Hz.
 * @param TTC [out] Time to Collision in seconds.
 * @param offset_lidar_bumper [in] offset in m between lidar and camera. Set to 0.27 by default (rough approximation from here: http://www.cvlibs.net/datasets/kitti/setup.php).
 */
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, double offset_lidar_camera = 0.0);//0.27);                  
#endif /* camFusion_hpp */

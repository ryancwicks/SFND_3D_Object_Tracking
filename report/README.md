# 3D Object Tracking - Final Report - Ryan Wicks
------------------------------------------------

This report decribes the work done for the 3D object tracking part of the Udacity Sensor Fusion Nano Degree.

## FP.1 - Matching 3D Objects
-----------------------------

To complete this section, I implemented the matchBoundingBoxes function. My implementation involved setting up a vector of vectors. The first level of vector corresponds to the number of bounding boxes in the current frame, and each element contains the number of boxes in the previous frame. For every box in the current frame, matched keypoints within that box are compared with the matches in the previous frame. If the matched points in the previous frame are within a box, the count for that box is incremented. 

After filling this vector of vector of counts of associated keypoints, each of the current boxes takes all boxes in the previous frame that contain matched keypoints above a minimum threshold and looks at the overlap between the bounding boxes. Rather than just using the largest number of keypoint matches, I use the region with the most overlap with the current box. This method of filtering assumes that the overlap of the bounding boxes changes slowly, which is a reasonable assumption for images of traffic taken at 10's of Hz.

In practice, my box overlap technique ended up giving the same result as simply choosing the highest number of matched keypoints.

As the evaluation rubric says to select based on the largest number of matched keypoints, I commented out the bounding box matching code and used the highest number of keypoints code.

## FP.2 - Compute LIDAR Based TTC
---------------------------------

To complete this section, I implemented the computeTTCLidar function. To do this, I wrote a function called compute_nearest. This function first sorts the laser points in increasing distance from the front of the vehicle, and then calculates the mean and standard deviation of the points x values (distance from the vehicle front). 

I attempted two methods to remove outliers. In the first, I filter out any points that are more than three standard deviations below the mean. I then choose the lowest remaining point within this range as the distance to the back of the vehicle.

As discussed in part 5, this worked well when the noise was close to the main cluster of points. However, this didn't work as well when there were isolated points far from the main cluster (these artificially increased the mean value.) To address this, I rewrote the filter to use a RANSAC method to find the outlier rejected mean of the point ranges, and then filtered as above using this new mean.

Then, in both cases, I averaged the closes 10% of the remaining x distances to help reduce random noise.

Finally, I also subtract off the difference between the camera position and the LIDAR position (based on the KITTI dataset paper) to put the two measurements in the same frame of reference.

## FP.3 - Associate Keypoint Correspondences with Bounding Boxes
----------------------------------------------------------------

I begin by clearing the kptMatches vector in the boundingBox object. I then add all matched keypoints that lie within the box to the kptMatches vector. I then proceed to calculate the mean and standard deviation of the distance parameter of the matches. I then filter out any keypoint matches with a distance of greater than 3*standard deviation. Finally, I fill the boundingBox.keypoints with the current keypoints from the filtered matches.

## FP.4 - Compute Camera-based TTC
----------------------------------

In filling in the computeTTCCamera function, I begin by pulling out every unique combination of each set of keypoint pairs, and calculating the radial distance between the two current keypoints and the two previous keypoints. These are then put into the equation given in the notes to calculate the TTC from image keypoints. The mean and standard deviation are calculated, but due to the large number of outliers due to features inside the box but not on the car itself, the median was used as the reference point for filtering, and the range was set to a single standard deviation, rather than 3 sigma, to better reject outliers due to points in the distance.

## FP.5 - Performance Evaluation 1
----------------------------------

There was a lot of variability in the LIDAR range measurement.

In some cases, this is a jump to a large value due to a small difference in relative range. This is expected, as the distance between the two vehicles didn't change, so the expected TTC should go to infinity.

The filtering seems to work well when the LIDAR data is grouped together. Little jumps away from the main set of points are handled correctly. In the image shown below, the outlier sits at 7.205 m, but the filtered range to target is 7.344 m (distance to the main clump). Not properly filtering these small jumps can cause very large jumps in the range to target.

One instance where this filtering doesn't work is when there are very large jumps away from the mean far from the central cluster of points, shown in the image below. In this case, the near outlier is no longer being filtered due to the large outliers, presumably from other vehicles or the roadway in the bounding box. This caused a jump in the TTC from about 7s to about 2s. The only way to deal with this would be to come up with a more advanced filter, perhaps RANSAC based that would better find the mean of the main cluster and reject these outliers.

I decided to take a simpler route, based on the idea that the majority of the points on the vehicle picked up by LIDAR will be picked up on the back bumper. To perform the outlier rejection, I first reject any points further than 0.25m from the median laser point, and then continue with the calculation of the mean and standard deviation. This new algorithm now rejects points from other vehicles and the roadway that are being picked up within the bounding box, and allows the small outliers to continue to be rejected even in cases with a lot of outlier noise. Adding this functionality to the filter maintained the filter performance for well clustered LIDAR point clouds, and effectively pre-filtered the large outliers, so that the algorithm could effectively filter out small outliers on the back bumper (in the case previously discussed, the TTC jumped back up to about 7s, in line with the previous measurements).

## FP.6 - Performance Evaluation 2
----------------------------------
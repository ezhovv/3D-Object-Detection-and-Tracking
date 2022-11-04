
# Write-Up for Sensor Fusion Projects 

This README file documents the results for both __Mid-Term Project:3D Object Detection__ and 
__Final Project: Sensor Fusion and Object Tracking__ as a part of the Udacity _"Self Driving Car Engineer Nanodegree"_

Since the project is split into two main parts (__Object Detection__ and __Object Tracking__), the diagram attached below captures the bigger picture, outlining steps that make up the complete algorithm.

![Diagram of Complete Project](https://video.udacity-data.com/topher/2021/January/5ff8fd46_sf-project-diagram/sf-project-diagram.png)

# Midterm Project: 3D Object Detection

The fist part of the projects involves detecting objects in 3D point clouds using real-world data from the Waymo Open Dataset. 
A deep learning approach is used to detect vehicles based on a birds-eye view perspective of LiDAR point cloud. 

### __Step 1: Computing Lidar Point-Cloud from Range Image__
In the Waymo Open dataset, lidar data is stored as a range image. Two of the data channels within the image (namely 'range' and 'intensity')
were extracted and converted to an 8-bit integer value range (from floating-point data). The resulting range image was then cropped to +/- 
90 degrees left and right of the forward-facing x-axis to capture the relevant field of view of the sensor. Afterwards, cropped range and 
instensity image were stacked together to create a visualization shown below.

![Range Image](/report%20images/range_image.png)

### __Step 2: Visualizing lidar point-cloud__

This task leverages the Open3D library to display lidar point-cloud in a responsive 3d viewer. Upon successful implementation, point 
cloud segment was closely inspected to locate vehicles with varying degrees of visibility. In doing so, some stable features on most vehicles were established, as there are plenty of points marking those parts of a car in a LiDAR point cloud. 

![PCL](/report%20images/pcl.png)

__Varying degrees of visibility are presented below:__

![Deg Visibility 1](/report%20images/var_vis_1.png)

All of the vehicles could be clearly detected with no obstructions or omissions. 

![Deg Visibility 2](/report%20images/var_vis_2.png)

Vehicles in the blind-spot zone of a roof-top installation of the LiDAR are hard to detect for obvious physical constraints.

![Deg Visibility 3](/report%20images/var_vis_3.png)
![Deg Visibility 4](/report%20images/var_vis_4.png)

A couple of images presented above show that parts of vehicles tend to be omitted when they stand close behind each other

![Deg Visibility 5](/report%20images/var_vis_5.png)

Vehicles that are located a bit farther away still look identifiable

![Deg Visibility 6](/report%20images/var_vis_6.png)

Father away the vehicle is located less identifiable it might get.

__Well Recognizable Features in the Point Cloud:__

1. An automobile roof from BEV 

![](/report%20images/roof.png)

2. Windshield

![](/report%20images/windshield.png)

3. Rear bumper shape

![](/report%20images/rear_bumper.png)

4. Tire shapes

![](/report%20images/tires.png)

5. Side shape 

![](/report%20images/side_view.png)

### __Step 3: Convert sensor coordinates to BEV-map coordinates__

This step serves as a starting point in converting the lidar point-cloud into a birds-eye view (BEV) perspective. 
Based on the (x,y)- coordinates in sensor space, the respective coordinates within the BEV were computed. The BEV map was then filled with lidar data from the point-cloud. 

The resulting visualization is shown below: 

![](/report%20images/convert_coord_to_bev.png)

### __Step 4: Computing intensity layer of the BEV map__

This task fills in the 'intensity' channel of the BEV map with data from the point-cloud. 
In doing so, all points with the same (x,y)- coordinates within the BEV map were identified and the top-most lidar point intensity value was assigned to the respective BEV pixel.
The resulting list of points is stored in `lidar_pc_top` as it will be re-used in later tasks. 

Points were sorted using `np.lexsort()` in a way that elements are sorted first by x, then y, and only then by z. It should be noted that the sorted sequence needs to be inverted for descending sort for z-coordinate, since the default order is ascending.

To mitigate the influence of outliers calues (very bright and very dark regions) and 
make sure vehicles are clearly separated from the background the resulting intensity image was normalized using percentiles. 

![](/report%20images/img_intensity.png)

### __Step 5: Commputing height layer of the BEV map__ 

This step fills in the 'height' channel of the BEV map. 
The sorted and pruned point cloud from previous exercise was used to normalize the heights in each BEV map pixel by the difference of max. and min. 
height which is defined in the `configs` structures.

![](/report%20images/height_layer.png)

## Model-based Object Detection in BEV Image

The object deep learning algorithm utilizes 2D projections to perform 3D object detections. Thus, the point cloud 
needed to be converted onto a horizontal plane as a bird's eye view map (as outlined abov). 
The BEV map is then fed into the neural network that was pre-trained using similar point clouds. 

![Complex_Yolo](/report%20images/complex_yolo.png)

As the model input is a three-channel BEV map, the detected objects will be returned wit coordinates and properties in the BEV cooordinate space. 
Thus, before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space. 
This can be acccomplished by re-projecting detection into the front camera coordinate system using transformations based on the lidar and camera calibration data.  

#### __Step 1: Adding a second model from a [GitHub repo](https://github.com/maudzung/SFA3D)__

First, the repo is cloned to the local directory. 
By going over `test.py` script I familiarized myself with the inference process and extracted relevant parameters from `parse_test_configs()`
 to the `configs` structure in `load_configs_model`. The model was then instantiated as `fpn_resnet` in `create_model` function. 
 After the inference had beed performed, the output was decoded and post-processed. 

#### __Step 2: 3D Bounding Boxes Extraction__

This task allowed to convert all detection to have format `[1, x, y, z, h, w, l, yaw]`, where `1` denotes the class id for object type `vehicle`.

The result of the last two steps described above is as follows:

![ResNet_Det](/report%20images/fpn_resnet_detections.png)

### __Performance Evaluation for Object Detection__

Based on the labels within the Waymo Open Dataset, the geometrical ovelap 
between the bounding boxes of labels and detected objects was computed. 
Non-maximum suppression was used to find the best match for object detection with a respective ground trutg object. 
After, calculatd geometrical ovelap was used to calculate the percentage percentage of this overlap in relation to the area of the bounding boxes was calculated (_IOU metric_).

Based on the pairings between ground-truth labels and detected objects, the number of false positives and false positives for the current fram was determined. 
Detections with an IOU larger than 0.5 were considered potential true positives (TP).
Number of false negatives was calculated by subtracting number of true positive from all positive. 
Number of false positives was calculated by subtracting number of true positives from the number of overall detections.

To evaluate the model in a meaningful way, measures of _"precision"_ and _"recall"_  were used.


Precision = TP / (TP + FN)

Recall = TP / (TP + FP)

First the functions were tested under the assumption that ground-truth labels are considered objects to see whether the code produces plausible results (`configs_det.use_labels_as_objects = True`).

![Plausbility_Check](/report%20images/eval_conf_true.png)

**Precision=1.0**

**Recall=1.0**

After evaluation the network on the actual object predictions results were the following: 

![Eval_Model](/report%20images/eval_conf_false.png)

**Precision=0.95065789**

**Recall=0.944444444**

#  Final Project: Sensor Fusion and Object Tracking 

The final builds upon the midterm project by adding additional functionality of Object Tracking. 
Extended Kalmat filter was implemented to perform the 3D objects tracking with a constant velocity model including height estimation using camera and lidar sensor fusion. 
Simple straightforward track management was coded to take care of track initialization, updating 
track state depending on its score, and timely deletion of a track. Simple nearest neighbor data 
association and gating were implemented to reduce computational complexity when matching new 
incoming measurements to existing tracks. In the last part of the project, sensor visibility check for 
camera sensor was coded, which allowed to apply sensor fucion with nonlineaar camera measurement model. 

Upon completion, the system was able to track vehicles over time with real-world camera and lidar measurements.

### __Step 1: Implementing EKF__

Extended Kalman Filter was immplemented under the assumption of constant velocity for objects. 
It means that a vehcle is represented by the center of mass of a rigiid body that is subject to zero net applied force (or torque).
Despite its simplicity, constant velocity motion model is widely used  motion model for visual tracking, which even outperforms 
state-of-the-art neural model in some scenarios. 
Since the project is aimed at tracking vehicles moving on a highway (parallel to the ego vehicle) such a simplified assumption works well. 

However, if camera and lidar sensors were to capture more difficult vehicle maneuvers, a non-linear motion model, e.g. bicycle mode, would prove to be more effective 
because it mirrors the motion of a vehicle much better. 

Even though the timestep for measurements might vary, it was fixed for this task. EKF was initialized with some fixed values. 
For now, measurement updates were only generated from 3D lidar detections. Since we have linear measurement model for lidar, 
measurement function `h(x)` is also linear, resulting in constant Jacobian matrix `H`. 

In later steps, EKF would be expanded to update states using nonlinear camera measurement model and corresponding Jacobian matrix.

![Single Object Tracking](/report%20images/second%20part/final_screenshot1.png)
_Figure. Tracking over a single target object over time using EKF with a constant velocity model_


![Single Object Tracking](/report%20images/second%20part/step1.gif)


### __Step 2: Track Management__

A simple track management system was coded to initialize and delete tracks, set a track state and a track score. All the valid tracks are kept track of by storing them in a list. For each new object appearing in the field of view of the sensor, an individual EKF is initialized. In case no more associated measurements come in after the initial assigment, the track maintains unassigned status with its score being incrementally decreased. If either the track score falls below the threshold or the state estimation covariance matrix _P_ gets too big, the unassigned track gets deleted from the list. 

Upon first initialization, the new incoming track gets the 'initialized' status. In case the system receives more measurements associated with that track, its track score gets increased and status updated. The status upgrades are as follows: _initialize -> tentative -> confirmed_. The rollback is also possible in absence of incoming measurements. 

![Single Object Track Management](/report%20images/second%20part/final_screenshot2.png)
_Figure. Single-Object Track Management_

![Single Object Tracking](/report%20images/second%20part/step2.gif)


### __Step 3: Measurements Association__

A single nearest neighbor (SNN) data association is implemented based on minimizing the Mahalanobis distance to associate measurements to tracks. Gattins is applied prior to association to reduce the association complexity by removing unlikely association pairs. A gate around each track is define in form of an elipse. The tracks gets updated only in case if measurement being detected inside the gate. The remaining combinations are kept in association matrix with respective entries holding the Mahalanobis distance of all yet unassigned measurements and tracks. The minimum entry of the matrix marks the best track-measurement association pair under the assumption of using SNN. The corresponding track and measurement get mapped on one another and the minimum entry (row=track and col=measurement) in the association matrix gets removed. The procedure is iteratively repeated until the matrix is empy or all of the remanining entries are infinity (measurements lying outsdide all gates). As described above, a new EKF is initialized for each unassigned measurement, giving it an initial track score that can be either incremented or decremented based on further associated measurements. 

![Multiple Objects Track Management](/report%20images/second%20part/final_screenshot4.png)
_Figure. Multiple Vehicles Track Management_

![Multiple Objects Track Management](/report%20images/second%20part/step3.gif)


### __Step 4: Completion of Camera-lidar Sensor Fusion Module__

The restrictions for Kalman Filter to use Lidar measurements are now lifted and the sensor fusion module is expanded to take into consideration nonlinear camera measurement model. It should be noted, however, that for the reason of optimizing the processing time objects position measurements are generated based on the ground truth labels rather than object detector predictions. Artificial measurement is added onto the ground truth values to simulate conditions more similar to real-like situations. Since camera measurements come from ground truth vehicle c oordinates, a nonlinear functions needs to be implemented to transform original poisition first into the camera coordinate system and then into pixel coordinates. 
For the lidar sensor, a 3D object detector implemented in previous steps still serves as a source of detections. 

Since we are now trying to fuse measurements from multiple sensors, their respective FOVs and overlap betweem them need to be noted. The top lidar sennsor position on the roof of a car captures 360 angle of its surrounding. 
The fron of the car is covered by the Lidar's FOV of [-1.57, +1.57] (in radians). The camera, on the other hand, has a smaller FOV ([-0.35,0.35]) but overlaps with lidar's view in the forward direction. Due to those differences, an additional function was coded as a part of track-measurement association to remove measurements that are not visible for the respective sensor prior to sensor fusion. 

![Multiple Objects Track Management](/report%20images/second%20part/final_screenshot3.png)
_Figure. Multiple Vehicles Track Management_

![Multiple Objects Track Management](/report%20images/second%20part/step4.gif)

# Final Thoughts

The final camera-lidar sensor fusion resulted in the overall more robust system, resulting in fewer false positive tentative tracks. The camera model successfully complements the lidar system with its superior ability to capture more details. As it can be seen from step 3, the lidar-only system sometimes mistakenly believes that the detection of roadside bushes is that of a driving car and tends to keep corresponding track active for a while. The camera helps overcome this challenge by providing supplementary support to gather more details in such cases.   

The current sensor fusion system was implemented under a set of assumptions that may not hold for other real-life cases but hold for the highway scenario explored above. For example, a constant velocity model proved to be a good approximation to the behavior of cars moving on the highway. This assumption breaks down when target objects display more complicated maneuvers, in which casae a 'bicycle model' would be a better choice. 

Another simplifications made in the projects is the use of SNN for data association, which works for the highway situations when conflict situations of overlapping paths rarely arise. However, urban traffic situations may introduce situations when multiple measurements fall within a single gate or when multiple gates overlap. Thus, these situations would require more sophisticate algorithm that looks for globally-optimal combination of all possible track-measurement associations (an example of which is global nearest neighbor GNN). Despiter it being more computationally costly, such an algorithm would yield better results in crowded scenarios. 



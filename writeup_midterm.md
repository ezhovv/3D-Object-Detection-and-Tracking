# 3D Object Detection

This project implements a deep-learning approach to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. A series of performance measures were conducted to evaluate the performance of the detection approach. 

Making use of the real-world data provided by the [Waymo Open Dataset](https://waymo.com/open/), the following was achieved:
- Convert two of the data channels withing the range image ("range" and "intensity") to an 8-bit integer value range and visualize both range and intenisty image (ID_S1_EX1)
- Utilize Open3D library to display the lidar point-cloud for close inspection (ID_S1_EX2)
Multiple steps were taken to create *Birds-Eye* view from Lidar PCL
- Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
- Compute intensity later of the BEV map (ID_S2_EX2)
- Compute height layer of the BEV map (ID_S2_EX3)
Object detection components:
- In addition to **YOLOv4** used for final testing, a second model was integrated into an existing framework ([SFA3D](https://github.com/maudzung/SFA3D)) (ID_S3_EX1)
- Convert BEV coordinates into pixel coordinates to extract bounding boxes (ID_S3_EX2)
Performance Evaluation:
- Compute IoU between labels and detections (ID_S4_EX1)
- Compute false-negatives and false-positives (ID_S4_EX2)
- Compute precision and recall and visualize evaluations (ID_S4_EX3)

## LiDAR Point-Cloud from Range Image

### Visualizing range image channels 

Lidar data in the Waymo Open dataset is stored as a range image. Therefore, we first need to extract channels withing the range image. This step converts to of such channels ("range" and "intensity") to an 8-bit integer value range and visualizes the results using the OpenCV library. 

Initial setup in `loop_over_dataset.py`:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [0, 1]
    exec_data = []
    exec_detection = []
    exec_tracking = []
    exec_visualization = ['show_range_image']

Implementation of the function `show_range_image` in `student/objdet_pcl.py` : 

    # visualize range image
    def show_range_image(frame, lidar_name):

        ####### MODIFICATIONS MADE #######     
        #######
        
        # step 1 : extract lidar data and range image for the roof-mounted lidar
        lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
        if len(lidar.ri_return1.range_image_compressed) > 0:
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
            ri = np.array(ri.data).reshape(ri.shape.dims)
        # step 2 : extract the range and the intensity channel from the range image
        # step 3 : set values <0 to zero
        ri[ri < 0] = 0.0
        ri_intensity = ri[:,:,1]
        ri_range = ri[:,:,0]
        # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
        ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
        # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
        percentile_1_99 = [np.percentile(ri_intensity,1), np.percentile(ri_intensity,99)]
        ri_intensity = np.clip(ri_intensity, percentile_1_99[0], percentile_1_99[1]) * 255 / (percentile_1_99[1] - percentile_1_99[0])
        # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
        img_range_intensity = np.vstack((ri_range,ri_intensity)).astype(np.uint8)
        # crop range image to +/- 90 degrees (left and right of the forward facing x-axis)
            # center of image -- positive x-axis, dist btwn center and left as well as center and right is 180deg
            # 90 deg corresponds to 1/4 of range cols
        deg90 = int(img_range_intensity.shape[1] / 4)
        ri_center = int(img_range_intensity.shape[1]/2)
        img_range_intensity = img_range_intensity[:,ri_center-deg90:ri_center+deg90]
        #######
        ####### END #######     
        
        return img_range_intensity


A sample output:

![](/report%20images/range_image.png)


### Visualizing lidar point-cloud 

Initial setup in `loop_over_dataset.py`:

    data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord'
    show_only_frames = [0, 200]
    exec_data = []
    exec_detection = []
    exec_tracking = []
    exec_visualization = ['show_pcl']

Implementation of the function `show_pcl` in `student/objdet_pcl.py` : 

    # visualize lidar point-cloud
    def show_pcl(pcl):

        ####### MODIFICATIONS MADE #######     
        #######

        # step 1 : initialize open3d with key callback and create window
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name = 'Open3D', width = 1920, height = 1080,
        left = 50, top = 50, visible = True)
        global temp_var
        temp_var = True
        def next_frame(vis):
            global temp_var
            print("right arrow pressed")
            temp_var = False
            return
        vis.register_key_callback(262, next_frame)
        

        # step 2 : create instance of open3d point-cloud class
        pcd = o3d.geometry.PointCloud()
        # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
        pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
        # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
        vis.add_geometry(pcd)
        # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
        while temp_var:
            vis.poll_events()
            vis.update_renderer()

        #######
        ####### END #######  

A sample ouput (responsive):

![](/report%20images/pcl.png)

The point-cloud was inspected to locate examples of vehicles with varying degrees of visibility and identify stable vehicle features. 

![](/report%20images/var_vis_1.png)

All of the vehicles could be clearly detected with no obstructions or omissions. 

![](/report%20images/var_vis_2.png)

Vehicles in the blind-spot zone of a roof-top installation of the LiDAR are hard to detect for obvious physical constraints.

![](/report%20images/var_vis_3.png)
![](/report%20images/var_vis_4.png)

A couple of images presented above show that parts of vehicles tend to be omitted when they stand close behind each other

![](/report%20images/var_vis_5.png)

Vehicles that are located a bit farther away still look identifiable

![](/report%20images/var_vis_6.png)

The father away the vehicle is located the less identifiable it might get.


Some of the stable features of most vehicles were noted, as these could be the foundation of classification decision of the detection algorithm. 

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



## Creating BEV from Lidar PCL 

### Transform sensor coordinates to BEV-map coordinates 

Initial setup in `loop_over_dataset.py`:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [0, 1]
    exec_data = ['pcl_from_rangeimage']
    exec_detection = ['bev_from_pcl']
    exec_tracking = []
    exec_visualization = []

Modification made to the function `bev_from_pcl` in `student/objdet_pcl.py` :

    def bev_from_pcl(lidar_pcl, configs):

        ####### MODIFICATIONS MADE #######     
        #######
        # remove lidar points outside detection area and with too low reflectivity
        mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                        (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                        (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
        lidar_pcl = lidar_pcl[mask]
        
        # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
        lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

        # convert sensor coordinates to bev-map coordinates (center is bottom-middle)


        ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
        bev_discreet = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
        ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
        lidar_pcl_cpy = np.copy(lidar_pcl)
        lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0]/ bev_discreet))
        # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
        lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discreet) + (configs.bev_width + 1) / 2)
        lidar_pcl_cpy[:, 1] = np.abs(lidar_pcl_cpy[:,1])

        # step 4 : visualize point-cloud using the function show_pcl from a previous task
        show_pcl(lidar_pcl_cpy)
        #######
        #######  END ####### 

A sample output: 

![](/report%20images/convert_coord_to_bev.png)

### Compute intensity layer of the BEV map

Additions made to the `bev_from_pcl` in `student/objdet_pcl.py` :

    def bev_from_pcl(lidar_pcl, configs):
        ....

        ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
        intensity_map = np.zeros((configs.bev_height, configs.bev_width))
        # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
        lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0
        idx_intensity = np.lexsort((-lidar_pcl_cpy[:,2], lidar_pcl_cpy[:,1], lidar_pcl_cpy[:, 0]))
        lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]
        ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
        ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
        _, idx_height_unique, count = np.unique(lidar_pcl_cpy[:, 0:2], axis = 0, return_index= True, return_counts= True)
        lidar_pcl_top = lidar_pcl_cpy[idx_height_unique]
        ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
        ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
        ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
        intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3])-np.amin(lidar_pcl_top[:, 3]))

        ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
        img_intensity = intensity_map * 256 / (np.amax(intensity_map))- np.amin(intensity_map)
        img_intensity = img_intensity.astype(np.uint8)
        
        cv2.imshow('image intensity', img_intensity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

A sample output:

![](/report%20images/img_intensity.png)

### Compute height layer of the BEV map

Additions made to the `bev_from_pcl` in `student/objdet_pcl.py` :

    def bev_from_pcl(lidar_pcl, configs):
    ...


    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
        height_map = np.zeros((configs.bev_height, configs.bev_width))
        ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
        ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
        ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
        height_map[np.int_(lidar_pcl_top[:, 0]), 
                np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / np.float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
        ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
        img_height = height_map * 256
        img_height = img_height.astype(np.uint8)
        cv2.imshow('height map', height_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

A sample output: 

![](/report%20images/height_layer.png)

## Model-based Object Detection in BEV image

### Add a second model from a Github repo

Initial setup in `loop_over_dataset.py`:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [50, 51]
    exec_data = ['pcl_from_rangeimage', 'load_image']
    exec_detection = ['bev_from_pcl', 'detect_objects']
    exec_tracking = []
    exec_visualization = ['show_objects_in_bev_labels_in_camera']
    configs_det = det.load_configs(model_name="fpn_resnet")

Modifications made to `detect_objects`, `load_configs_model` and `load_configs_model` in `student/objdet_detect.py`


    # load model-related parameters into an edict
    def load_configs_model(model_name='darknet', configs=None):

        # init config file, if none has been passed
        if configs==None:
            configs = edict()  

        # get parent directory of this file to enable relative paths
        curr_path = os.path.dirname(os.path.realpath(__file__))
        parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
        
        # set parameters according to model type
        if model_name == "darknet":
            configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
            configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
            configs.arch = 'darknet'
            configs.batch_size = 4
            configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
            configs.conf_thresh = 0.5
            configs.distributed = False
            configs.img_size = 608
            configs.nms_thresh = 0.4
            configs.num_samples = None
            configs.num_workers = 4
            configs.pin_memory = True
            configs.use_giou_loss = False
            configs.min_iou = .5 # was added
        

        elif model_name == 'fpn_resnet':
            ####### MODIFICATIONS MADE #######     
            #######
            configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
            configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
            configs.arch = 'fpn_resnet'
            configs.batch_size = 4
            #configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
            configs.conf_thresh = 0.5
            configs.distributed = False
            configs.img_size = 608
            configs.nms_thresh = 0.4
            configs.num_samples = None
            configs.num_workers = 4
            configs.pin_memory = True
            configs.use_giou_loss = False
            configs.K=50

            configs.head_conv = 64
            configs.down_ratio = 4
            configs.peak_thresh = 0.2
            configs.imagenet_pretrained = False
            configs.num_classes = 3
            configs.num_center_offset = 2
            configs.num_z = 1
            configs.num_dim = 3
            configs.num_direction = 2  # sin, cos
            configs.heads = {
                'hm_cen': configs.num_classes,
                'cen_offset': configs.num_center_offset,
                'direction': configs.num_direction,
                'z_coor': configs.num_z,
                'dim': configs.num_dim
            }
            configs.num_layers = 18   
            #######
            ####### END #######     

        else:
            raise ValueError("Error: Invalid model name")

        # GPU vs. CPU
        configs.no_cuda = True # if true, cuda is not used
        configs.gpu_idx = 0  # GPU index to use.
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

        return configs

.


    # create model according to selected model type
    def create_model(configs):

        # check for availability of model file
        assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

        # create model depending on architecture name
        if (configs.arch == 'darknet') and (configs.cfgfile is not None):
            print('using darknet')
            model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
        
        elif 'fpn_resnet' in configs.arch:
            print('using ResNet architecture with feature pyramid')
            
            ####### MODIFICATIONS MADE #######     
            #######
            model = fpn_resnet.get_pose_net(num_layers=configs.num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)
            #######
            ####### END #######     
        
        else:
            assert False, 'Undefined model backbone'

        # load model weights
        model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
        print('Loaded weights from {}\n'.format(configs.pretrained_filename))

        # set model to evaluation state
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
        model = model.to(device=configs.device)  # load model to either cpu or gpu
        model.eval()          

        return model


    # detect trained objects in birds-eye view
    def detect_objects(input_bev_maps, model, configs):

        # deactivate autograd engine during test to reduce memory usage and speed up computations
        with torch.no_grad():  

            # perform inference
            outputs = model(input_bev_maps)

            # decode model output into target object format
            if 'darknet' in configs.arch:

                # perform post-processing
                output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
                detections = []
                for sample_i in range(len(output_post)):
                    if output_post[sample_i] is None:
                        continue
                    detection = output_post[sample_i]
                    for obj in detection:
                        x, y, w, l, im, re, _, _, _ = obj
                        yaw = np.arctan2(im, re)
                        detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

            elif 'fpn_resnet' in configs.arch:
                # decode output and perform post-processing
                
                ####### MODIFICATIONS MADE #######     
                #######
                outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
                outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
                detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                        outputs['dim'], K=configs.K)
                detections = detections.cpu().numpy().astype(np.float32)
                detections = post_processing(detections, configs)
                detections = detections[0][1]
                #######
                ####### END #######    


### Extract 3D bounding boxes from model response

Additions made to `detect_objects` in `student/objdet_detect.py`:

    def detect_objects(input_bev_maps, model, configs):
        ....

        ....

        ####### ADDITIONS MADE #######     
        #######
        # Extract 3d bounding boxes from model response
        objects = [] 

        # step 1 : check whether there are any detections
        if len(detections)!=0:
            # step 2 : loop over all detections
            for obj in detections:
                id, bev_x, bev_y, z, h, bev_w, bev_l, yaw = obj
                ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
                x = bev_y * (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
                y = bev_x * (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width - (configs.lim_y[1] - configs.lim_y[0])/2.0
                w = bev_w * (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width
                l = bev_l * (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
                
                if ((x >= configs.lim_x[0]) and (x <= configs.lim_x[1])
                and (y >= configs.lim_y[0]) and (y <= configs.lim_y[1])
                and (z >= configs.lim_z[0]) and (z <= configs.lim_z[1])):
                
                    ## step 4 : append the current object to the 'objects' array
                    objects.append([1, x, y, z, h, w, l, yaw])
            
        #######
        ####### END #######   
        
        return objects 

Output: 

![](/report%20images/fpn_resnet_detections.png)
![](/report%20images/second)


## Performance Evaluation for Object Detetcion

### Compute IoU between labels and detections, false-negatives and false-positives
### Also compute precision and recall

Initial setup in `loop_over_dataset.py`:

    data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
    show_only_frames = [50, 51]
    exec_data = ['pcl_from_rangeimage']
    exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
    exec_tracking = []
    exec_visualization = ['show_detection_performance']
    configs_det = det.load_configs(model_name="darknet")

Modifications made to `measure_detection_performance` in `student/objdet_eval.py`:

    # compute various performance measures to assess object detection
    def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
        
        # find best detection for each valid label 
        true_positives = 0 # no. of correctly detected objects
        center_devs = []
        ious = []
        for label, valid in zip(labels, labels_valid):
            matches_lab_det = []
            if valid: # exclude all labels from statistics which are not considered valid
                
                # compute intersection over union (iou) and distance between centers

                ####### MODIFICATIONS MADE #######     
                #######

                ## step 1 : extract the four corners of the current label bounding-box
                bbox = label.box
                bbox_l = tools.compute_box_corners(bbox.center_x, bbox.center_y, bbox.width, bbox.length, bbox.heading)
                ## step 2 : loop over all detected objects
                for det in detections: 

                    ## step 3 : extract the four corners of the current detection
                    idx, x, y, z, h, w, l, yaw = det
                    bbox_d = tools.compute_box_corners(x,y,w,l,yaw)
                    ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                    dist_x = np.array(bbox.center_x - x).item()
                    dist_y = np.array(bbox.center_y - y).item()
                    dist_z = np.array(bbox.center_z - z).item()
                    ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                    poly_l = Polygon(bbox_l)
                    poly_d = Polygon(bbox_d)
                    intersection = poly_l.intersection(poly_d).area
                    union = poly_l.union(poly_d).area
                    iou = intersection / union
                    
                    ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                    if iou > min_iou:
                        matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                        true_positives += 1
                        
                #######
                ####### END #######     
                
            # find best match and compute metrics
            if matches_lab_det:
                best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
                ious.append(best_match[0])
                center_devs.append(best_match[1:])
            


        ####### MODIFICATIONS MADE #######     
        #######
        
        # compute positives and negatives for precision/recall
        
        ## step 1 : compute the total number of positives present in the scene
        all_positives = labels_valid.sum()

        ## step 2 : compute the number of false negatives
        false_negatives = all_positives - true_positives

        ## step 3 : compute the number of false positives
        false_positives = len(detections) - true_positives
        
        #######
        ####### END #######     
        
        pos_negs = [all_positives, true_positives, false_negatives, false_positives]
        det_performance = [ious, center_devs, pos_negs]
        return det_performance

Modifications made to `compute_performance_stats` in `student/objdet_eval.py`:

    # evaluate object detection performance based on all frames
    def compute_performance_stats(det_performance_all):

        # extract elements
        ious = []
        center_devs = []
        pos_negs = []
        for item in det_performance_all:
            ious.append(item[0])
            center_devs.append(item[1])
            pos_negs.append(item[2])
        
        ####### MODIFICATIONS MADE #######     
        #######    
        print('student task ID_S4_EX3')
        pos_negs_arr = np.asarray(pos_negs)
        ## step 1 : extract the total number of positives, true positives, false negatives and false positives
        positives = sum(pos_negs_arr[:,0])
        true_pos = sum(pos_negs_arr[:,1])
        false_negs =  sum(pos_negs_arr[:,2])
        false_pos = sum(pos_negs_arr[:,3])
        ## step 2 : compute precision
        precision = true_pos / (true_pos + false_pos)

        ## step 3 : compute recall 
        recall = true_pos / (true_pos + false_negs)

        #######    
        ####### END #######     
        print('precision = ' + str(precision) + ", recall = " + str(recall))   
        ....
        ....


First the functions were tested under the assumption that ground-truth labels are considered objects to see whether the code produces plausible results (`configs_det.use_labels_as_objects = True`).

![](/report%20images/eval_conf_true.png)

**Precision=1.0**
**Recall=1.0**

After evaluation the network on the actual object predictions results were the following: 

![](/report%20images/eval_conf_false.png)

**Precision=0.95065789**
**Recall=0.944444444**



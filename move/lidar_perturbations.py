#!/usr/bin/env python3
import rospy
import numpy as np
import traceback
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from ros_numpy_master.src.ros_numpy import registry as ros_numpy
import tf.transformations as tr
from jsk_recognition_msgs.msg import BoundingBoxArray

from perturbationdrive.perturbationdrive import LidarPerturbation

class LidarPerturberNode:
    def __init__(self):
        rospy.init_node('lidar_perturbation_node')

        # ==========================================
        #           USER CONFIGURATION
        # ==========================================
        # Below is a list of all available perturbations. You can select one and set the intensity.

        PERTURBATION_NAMES = [
            # --- 0: Custom / Ghost Injection ---
            "lidar_inject_ghost_points",   # 0

            # --- 1-4: MultiCorrupt Framework (NOTE: Intensity only 0-2) ---
            "pts_motion",                  # 1
            "transform_points",            # 2
            "reduce_LiDAR_beamsV2",        # 3
            "pointsreducing",              # 4

            # --- 5-12: Global Sensor Noise & Dropout ---
            "scene_glare_noise",           # 5
            "lidar_crosstalk_noise",       # 6
            "density_dec_global",          # 7
            "cutout_local",                # 8
            "gaussian_noise_lidar",        # 9
            "uniform_noise",               # 10
            "impulse_noise_lidar",         # 11
            "fov_filter",                  # 12

            # --- 13-15: Motion & Alignment Corruption ---
            #"fulltrajectory_noise",        # 13, exclude too heavy plus needs torch
            "spatial_alignment_noise",     # 14
            "temporal_alignment_noise",    # 15

            # --- 16-18: Fast Real-Time Weather Approximations ---
            "fast_rain",               # 16
            "fast_fog",                    # 17
            "fast_snow",                   # 18

            # --- 19-27: Local Bounding Box (Object-Level) Corruptions ---
            "moving_noise_bbox",           # 19
            "density_dec_bbox",            # 20
            "cutout_bbox",                 # 21
            "gaussian_noise_bbox",         # 22
            "uniform_noise_bbox",          # 23
            "impulse_noise_bbox",          # 24
            "shear_bbox",                  # 25 4.5hz
            "scale_bbox",                  # 26 3.5hz
            "rotation_bbox",               # 27 3.5hz
        ]
        idx = rospy.get_param('~perturbation_index', 0)
        
        # Add a safety check so it doesn't crash on bad inputs
        if idx < 0 or idx >= len(PERTURBATION_NAMES):
            rospy.logwarn(f"Index {idx} is out of bounds. Defaulting to 0.")
            idx = 0
        self.perturbation_name = PERTURBATION_NAMES[idx]
        
        #self.perturbation_name = PERTURBATION_NAMES[9] 
        self.intensity = 2
        
        self.input_topic = "/velodyne_points" 
        self.output_topic = "/velodyne_points/perturbed"
        self.odom_topic = "/odom"  # Only needed for dynamic perturbations that require ego-motion data
        
        self.bbox_topic = "/detected_objects_bboxes"
        self.latest_bboxes = None
        # State variables for the Main Loop
        self.latest_msg = None
        self.latest_odom = None
        self.new_data_available = False
        # ==========================================

        rospy.loginfo(f"Initializing LiDAR Perturbation: {self.perturbation_name} at intensity {self.intensity}")

        self.perturber = LidarPerturbation(funcs=[self.perturbation_name])
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.lidar_callback)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.bbox_sub = rospy.Subscriber(self.bbox_topic, BoundingBoxArray, self.bbox_callback)
        
        rospy.loginfo("LiDAR Perturbation Node Ready.")

    def odom_callback(self, msg):
        """ PRODUCER: Caches the freshest ego-vehicle pose """
        self.latest_odom = msg

    def bbox_callback(self, msg):
        """ PRODUCER: Caches the freshest bounding box array """
        self.latest_bboxes = msg

    def lidar_callback(self, msg):
        """
        PRODUCER: This simply caches the latest message and exits instantly.
        It never blocks the ROS communication thread.
        """
        self.latest_msg = msg
        self.new_data_available = True

    def convert_bboxes_to_numpy(self, bbox_array_msg):
        if not bbox_array_msg.boxes:
            return np.empty((0, 7))
            
        bboxes_np = np.zeros((len(bbox_array_msg.boxes), 7))
        
        for i, box in enumerate(bbox_array_msg.boxes):
            # Center coordinates (x, y)
            bboxes_np[i, 0] = box.pose.position.x
            bboxes_np[i, 1] = box.pose.position.y
            
            # --- SUPERVISOR'S HACK ---
            # Center the box vertically at the sensor's height
            bboxes_np[i, 2] = 0.0 
            
            # --- THE DIMENSION SWAP FIX ---
            # Map the ROS Y-dimension to the Numpy X-dimension (Length)
            bboxes_np[i, 3] = box.dimensions.y  
            # Map the ROS X-dimension to the Numpy Y-dimension (Width)
            bboxes_np[i, 4] = box.dimensions.x  
            
            # --- SUPERVISOR'S HACK ---
            # Make the box 20 meters tall so it catches the wheels AND the roof
            bboxes_np[i, 5] = 20.0 
            
            # Extract heading (yaw) from quaternion
            quat = [
                box.pose.orientation.x,
                box.pose.orientation.y,
                box.pose.orientation.z,
                box.pose.orientation.w
            ]
            _, _, yaw = tr.euler_from_quaternion(quat)
            
            # Note: Because we swapped X and Y dimensions, we might need to 
            # rotate the yaw by 90 degrees (pi/2) to align the new length axis.
            # Try it as-is first. If the box is still missing points, change 
            # the line below to: bboxes_np[i, 6] = yaw + (np.pi / 2.0)
            bboxes_np[i, 6] = yaw 
            
        return bboxes_np
    
    def run(self):
        """
        CONSUMER: The main loop. It processes the freshest data at a controlled rate.
        """
        # Run at 10Hz 
        # It will automatically sleep to maintain this rate, or run instantly if behind.
        rate = rospy.Rate(10) 
        
        while not rospy.is_shutdown():
            if self.new_data_available and self.latest_msg is not None:
                # 1. Grab the freshest message and reset the flag
                msg = self.latest_msg
                self.new_data_available = False
                
                try:
                    # 2. Convert ROS PointCloud2 -> Numpy
                    pc_struct = ros_numpy.numpify(msg)  
                    
                    if pc_struct.size == 0:
                        continue 
                        
                    points_np = np.zeros((pc_struct.shape[0],5), dtype=np.float32)
                    points_np[:,0] = pc_struct['x']
                    points_np[:,1] = pc_struct['y'] 
                    points_np[:,2] = pc_struct['z']
                    points_np[:,3] = pc_struct['intensity']
                    points_np[:,4] = pc_struct['ring']

                    # 3. Apply Perturbation
                    if self.perturbation_name in ["spatial_alignment_noise", "fulltrajectory_noise"]:
                        # Construct the 4x4 original pose matrix from /odom
                        trans = [self.latest_odom.pose.pose.position.x, self.latest_odom.pose.pose.position.y, self.latest_odom.pose.pose.position.z]
                        rot = [self.latest_odom.pose.pose.orientation.x, self.latest_odom.pose.pose.orientation.y, self.latest_odom.pose.pose.orientation.z, self.latest_odom.pose.pose.orientation.w]
                        
                        ori_pose = tr.quaternion_matrix(rot)
                        ori_pose[0:3, 3] = trans
                        
                        if self.perturbation_name == "spatial_alignment_noise":
                            # Apply static sensor misalignment
                            noisy_pose = self.perturber.perturbation(ori_pose, self.perturbation_name, self.intensity)
                            error_matrix = np.dot(np.linalg.inv(ori_pose), noisy_pose)
                            xyz1 = np.ones((points_np.shape[0], 4))
                            xyz1[:, :3] = points_np[:, :3]
                            transformed_xyz = np.dot(xyz1, error_matrix.T)
                            perturbed_points = points_np.copy()
                            perturbed_points[:, :3] = transformed_xyz[:, :3]
                            
                        elif self.perturbation_name == "fulltrajectory_noise":
                            # Apply dynamic trajectory drift! 
                            # Notice we pass pc_pose=ori_pose as the missing argument here:
                            perturbed_points = self.perturber.perturbation(
                                points_np, 
                                self.perturbation_name, 
                                self.intensity,
                                pc_pose=ori_pose
                            )

                    elif self.perturbation_name.endswith("_bbox"):
                        # Ensure we have bounding box data
                        if self.latest_bboxes is None:
                            rospy.logwarn_throttle(2, "Waiting for bounding boxes on /detected_objects_bboxes...")
                            continue
                            
                        # Convert to N x 7 numpy array
                        bbox_np = self.convert_bboxes_to_numpy(self.latest_bboxes)
                        
                        # Only apply if there are actually objects detected
                        if bbox_np.shape[0] > 0:
                            perturbed_points = self.perturber.perturbation(
                                points_np, 
                                self.perturbation_name, 
                                self.intensity,
                                bbox=bbox_np  # Pass the bbox argument
                            )
                        else:
                            # No objects detected, pass clean cloud
                            perturbed_points = points_np.copy()
                    else:
                        perturbed_points = self.perturber.perturbation(
                            points_np, 
                            self.perturbation_name, 
                            self.intensity
                        )
                        
                    if isinstance(perturbed_points, tuple):
                        perturbed_points = perturbed_points[0]

                   # 4. Field Reconstruction for ros_numpy
                    num_cols = perturbed_points.shape[1]
                    
                    # Define the structured data type
                    if num_cols == 5:
                        point_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ring', np.float32)]
                    else:
                        point_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]

                    # Create an empty structured array matching the number of points
                    structured_points = np.zeros(perturbed_points.shape[0], dtype=point_dtype)
                    
                    # Map the raw math columns back to the named fields
                    structured_points['x'] = perturbed_points[:, 0]
                    structured_points['y'] = perturbed_points[:, 1]
                    structured_points['z'] = perturbed_points[:, 2]
                    structured_points['intensity'] = perturbed_points[:, 3]
                    if num_cols == 5:
                        structured_points['ring'] = perturbed_points[:, 4]

                    # 5. Fast Conversion and Publish
                    out_msg = ros_numpy.msgify(PointCloud2, structured_points)
                    
                    # Manually attach the original header
                    out_msg.header.frame_id = msg.header.frame_id

                    # --- TEMPORAL ALIGNMENT FIX ---
                    if self.perturbation_name == "temporal_alignment_noise":
                        # Calculate the delay in seconds (0.05s to 0.25s based on intensity)
                        delay_seconds = [0.05, 0.1, 0.15, 0.2, 0.25][self.intensity]
                        
                        # Subtract the delay from the ROS timestamp to simulate a lagging sensor
                        out_msg.header.stamp = msg.header.stamp - rospy.Duration(delay_seconds)
                    else:
                        out_msg.header.stamp = msg.header.stamp
    

                    self.pub.publish(out_msg)

                except Exception as e:
                    rospy.logerr(f"Error processing point cloud: {traceback.format_exc()}")

            # Sleep for the remainder of the 10Hz cycle
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LidarPerturberNode()
        # Instead of rospy.spin(), we hand control over to our custom loop
        node.run()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3
import rospy
import numpy as np
import traceback
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import ros_numpy
import tf.transformations as tr

from perturbationdrive import LidarPerturbation

class LidarPerturberNode:
    def __init__(self):
        rospy.init_node('lidar_perturbation_node')

        # ==========================================
        #           USER CONFIGURATION
        # ==========================================
        # Below is a list of all available perturbations. You can select one and set the intensity.

        PERTURBATION_NAMES = [
            "lidar_inject_ghost_points", #0
            "pts_motion",#1
            "transform_points",
            "reduce_LiDAR_beamsV2",
            "pointsreducing",
            "simulate_snow_sweep", #insane slow
            "simulate_fog",#slow and too close circles

            "rain_sim",# 7 way too slow
            "snow_sim",# this might take a couple minutes lol. then takes 6 seconds per point cloud like rain sim.
            "fog_sim", # this perturbation is the same as 6 
            "scene_glare_noise",#10
            "lidar_crosstalk_noise",
            "density_dec_global",
            "cutout_local",
            "gaussian_noise", #14
            "uniform_noise",
            "impulse_noise",
            "fov_filter",
            "fulltrajectory_noise",  
            "spatial_alignment_noise", #19
            "temporal_alignment_noise", 
            "fast_rain_sim",
            "fast_fog", #22
            "fast_snow"
        ]
        
        self.perturbation_name = PERTURBATION_NAMES[23] 
        self.intensity = 2
        
        self.input_topic = "/velodyne_points" 
        self.output_topic = "/velodyne_points/perturbed"
        self.odom_topic = "/odom"  # Only needed for dynamic perturbations that require ego-motion data
        
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
        
        rospy.loginfo("LiDAR Perturbation Node Ready.")

    def odom_callback(self, msg):
        """ PRODUCER: Caches the freshest ego-vehicle pose """
        self.latest_odom = msg

    def lidar_callback(self, msg):
        """
        PRODUCER: This simply caches the latest message and exits instantly.
        It never blocks the ROS communication thread.
        """
        self.latest_msg = msg
        self.new_data_available = True

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
                    if self.perturbation_name == "spatial_alignment_noise":
                        # Construct the 4x4 original pose matrix from /odom
                        trans = [self.latest_odom.pose.pose.position.x, self.latest_odom.pose.pose.position.y, self.latest_odom.pose.pose.position.z]
                        rot = [self.latest_odom.pose.pose.orientation.x, self.latest_odom.pose.pose.orientation.y, self.latest_odom.pose.pose.orientation.z, self.latest_odom.pose.pose.orientation.w]
                        
                        ori_pose = tr.quaternion_matrix(rot)
                        ori_pose[0:3, 3] = trans
                        
                        # Get the noisy pose from the 3D_Corruptions_AD function
                        noisy_pose = self.perturber.perturbation(ori_pose, self.perturbation_name, self.intensity)
                        
                        # Calculate the error matrix and apply it to the XYZ points
                        # This physically shifts the point cloud to simulate sensor misalignment
                        error_matrix = np.dot(np.linalg.inv(ori_pose), noisy_pose)
                        
                        # Convert points to homogeneous coordinates (N, 4) to multiply by 4x4 matrix
                        xyz1 = np.ones((points_np.shape[0], 4))
                        xyz1[:, :3] = points_np[:, :3]
                        
                        # Apply the geometric transformation
                        transformed_xyz = np.dot(xyz1, error_matrix.T)
                        perturbed_points = points_np.copy()
                        perturbed_points[:, :3] = transformed_xyz[:, :3]
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
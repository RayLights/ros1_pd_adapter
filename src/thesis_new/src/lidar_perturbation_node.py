#!/usr/bin/env python3
import rospy
import numpy as np
import traceback
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

# Import the LidarPerturbation class from your library
# If this fails, ensure LidarPerturbation is in __init__.py of perturbationdrive
from perturbationdrive import LidarPerturbation

class LidarPerturberNode:
    def __init__(self):
        rospy.init_node('lidar_perturbation_node')

        # ==========================================
        #           USER CONFIGURATION
        # ==========================================
        
        # Select your filter here.
        # Options: "lidar_simulate_adverse_weather", "lidar_point_dropout", etc.
        #lidar_point_dropout,slow
    #lidar_inject_ghost_points, slow
    #lidar_reduce_reflectivity, slow but works
    #lidar_simulate_adverse_weather, fine
    #fast_snow_perturbation, works slow
    #lidar_mc_motion_blur, SLOW
    ##lidar_mc_spatial_misalignment,  slow
    #lidar_mc_beam_reduction, good
    #lidar_mc_random_dropout, okay 
    #lidar_mc_simulate_fog, slow
    #lidar_mc_simulate_snow, slow
    #lidar_mc_simulate_snow_sweep, work
        self.perturbation_name = "lidar_mc_simulate_snow_sweep"
        
        # Set intensity (0 to 4 usually)
        self.intensity = 2
        
        # Topic Names for Velodyne
        self.input_topic = "/velodyne_points" 
        self.output_topic = "/velodyne_points/perturbed"
        
        # ==========================================

        rospy.loginfo(f"Initializing LiDAR Perturbation: {self.perturbation_name} at intensity {self.intensity}")

        # Initialize the Lidar Controller
        # We pass the list of functions to initialize just like the Image class
        self.perturber = LidarPerturbation(
            funcs=[self.perturbation_name]
        )

        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.lidar_callback)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        
        rospy.loginfo("LiDAR Perturbation Node Ready.")

    def lidar_callback(self, msg):
        try:
            # 1. Convert ROS PointCloud2 -> Numpy
            # We explicitly ask for x, y, z, intensity and ring. 
            # Velodyne 'time' too, but we might lose them 
            # if the perturbation function doesn't account for them.
            field_names = ['x', 'y', 'z', 'intensity','ring']
            
            # Reads points into a generator
            point_generator = pc2.read_points(msg, field_names=field_names, skip_nans=True)
            
            # Convert to numpy array (N, 5)
            # This can be slow for very large clouds in python, but works for Velodyne32
            points_list = list(point_generator)
            if not points_list:
                return # Empty cloud
                
            points_np = np.array(points_list, dtype=np.float32)

            # 2. Apply Perturbation
            # The class expects (N, C) numpy array
            perturbed_points = self.perturber.perturbation(
                points_np, 
                self.perturbation_name, 
                self.intensity
            )
            if isinstance(perturbed_points, tuple):
                perturbed_points = perturbed_points[0]
            else:
                perturbed_points = perturbed_points

            # 3. Convert Numpy -> ROS PointCloud2
            # We need to reconstruct the message. 
            # We must define the fields to match the numpy columns.
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
                PointField('ring', 16, PointField.FLOAT32, 1),
            ]

            # Create the header (preserve frame_id and timestamp)
            header = Header()
            header.frame_id = msg.header.frame_id
            header.stamp = msg.header.stamp

            # Create the output message
            #print(f"Published perturbed point cloud with {perturbed_points.shape[0]} points.")
            #print(header)
            #print(fields)
            out_msg = pc2.create_cloud(header, fields, perturbed_points)
            
            
            self.pub.publish(out_msg)

        except Exception as e:
            # Print full trace if something crashes
            rospy.logerr(traceback.format_exc())

if __name__ == '__main__':
    try:
        LidarPerturberNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
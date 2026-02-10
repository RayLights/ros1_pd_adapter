#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
import message_filters
import ros_numpy  # Run: sudo apt-get install ros-noetic-ros-numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import perturbationdrive as pd

# --- ML PROFESSIONAL NOTE: Calibration ---
# Since your bag lacks camera_info, we define these based on your 800x503 resolution.
# Adjust these once you get official specs for the front_narrow camera.
K_INTRINSIC = np.array([
    [1200.0, 0.0, 400.0],
    [0.0, 1200.0, 251.5],
    [0.0, 0.0, 1.0]
])

def process_image(cv_bridge, msg):
    # (Your existing logic kept intact)
    cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if cv_image.dtype == np.float32:
        cv_image = (cv_image * 255).astype(np.uint8)
    if msg.encoding == "rgb8" or msg.encoding == "32FC3":
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # Apply Perturbation
    perturbed_img = pd.perturbationfuncs.gaussian_noise(1, cv_image)
    return perturbed_img.astype(np.uint8)

class ThesisPerturber:
    def __init__(self):
        rospy.init_node('perturbation_node')
        self.bridge = CvBridge()
        
        # 1. Topics
        self.image_topic = "/gmsl_camera/front_narrow/image_raw"
        self.lidar_topic = "/velodyne_points"
        self.output_topic = "/gmsl_camera/front_narrow/perturbed"
        
        # 2. Subscribers with Message Filters
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)
        
        # 3. Synchronizer: Slop=0.1 means messages must be within 0.1s of each other
        # This is critical because your LiDAR is 5Hz and Image is 30Hz
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.unified_callback)
        
        self.img_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        
        rospy.loginfo("Multi-modal Perturbation Node Initialized")

    def unified_callback(self, img_msg, pc_msg):
        try:
            start_time = time.time()
            
            # --- STEP 1: Process & Perturb Image ---
            perturbed_img = process_image(self.bridge, img_msg)
            
            # --- STEP 2: Process & Perturb LiDAR ---
            # Convert ROS PointCloud2 to Numpy Nx3 (XYZ)
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg)
            
            # Example Perturbation: Simulation of sensor jitter/noise
            # perturbed_pc = pd.perturbationfuncs.lidar_jitter(pc_array) # If available
            # Manually adding a tiny amount of noise for now:
            perturbed_pc = pc_array + np.random.normal(0, 0.02, pc_array.shape)

            # --- STEP 3: BEVFUSION INFERENCE ---
            # This is where your mm3d model call goes
            # result = self.run_bev_inference(perturbed_img, perturbed_pc)
            
            # --- STEP 4: Publish Visualization ---
            out_msg = self.bridge.cv2_to_imgmsg(perturbed_img, encoding="bgr8")
            out_msg.header = img_msg.header
            self.img_pub.publish(out_msg)
            
            latency = (time.time() - start_time) * 1000
            rospy.loginfo(f"Fusion Latency: {latency:.2f} ms | Points: {len(perturbed_pc)}")

        except Exception as e:
            rospy.logerr(f"Fusion Callback Error: {e}")

if __name__ == '__main__':
    ThesisPerturber()
    rospy.spin()
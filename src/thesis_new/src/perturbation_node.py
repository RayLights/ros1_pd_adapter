#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import traceback

from perturbationdrive import ImagePerturbation 

class ThesisPerturber:
    def __init__(self):
        rospy.init_node('perturbation_node')
        self.bridge = CvBridge()
        
        # ==========================================
        #           USER CONFIGURATION
        # ==========================================
        
        # Select your filter here.
        # Examples of Static: "static_rain_filter", "gaussian_noise", "fog_filter"
        # Examples of Dynamic: "dynamic_rain_filter", "dynamic_smoke_filter"
        self.perturbation_name = "static_rain_filter" 
        
        # Set intensity (usually 1 to 5)
        self.intensity = 3
        
        # Set your camera resolution (Height, Width)
        # It is CRITICAL this matches your input topic image size for dynamic masks to work.
        self.image_shape = (503, 800) 
        
        # Topic Names
        self.input_topic = "/gmsl_camera/front_narrow/image_raw" 
        self.output_topic = "/gmsl_camera/front_narrow/perturbed"
        
        # ==========================================
        
        rospy.loginfo(f"Initializing Perturbation: {self.perturbation_name} at intensity {self.intensity}")
        
        # Initialize the Controller
        # We pass the chosen function in a list so it only loads necessary assets
        self.perturber = ImagePerturbation(
            funcs=[self.perturbation_name], 
            image_size=self.image_shape
        )
        
        self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        rospy.loginfo("Perturbation Node Ready.")

    def image_callback(self, msg):
        try:
            # 1. Convert Input
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # 2. Fix Float -> Int & Color (if necessary)
            if cv_image.dtype == np.float32:
                cv_image = (cv_image * 255).astype(np.uint8)
            
            # Handle RGB/BGR mismatch common in ROS
            if msg.encoding == "rgb8" or msg.encoding == "32FC3":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            # 3. Apply Perturbation
            # The class handles the logic for static vs dynamic internally
            perturbed_img = self.perturber.perturbation(
                cv_image, 
                self.perturbation_name, 
                self.intensity
            )
            
            # 4. Publish
            # Ensure uint8 output
            perturbed_img = perturbed_img.astype(np.uint8)
            
            out_msg = self.bridge.cv2_to_imgmsg(perturbed_img, encoding="bgr8")
            out_msg.header = msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            rospy.logerr(traceback.format_exc())

if __name__ == '__main__':
    try:
        ThesisPerturber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import traceback

from perturbationdrive.perturbationdrive import ImagePerturbation 

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
        PERTURBATION_NAMES_no_static_no_dynamic = [
            "gaussian_noise","poisson_noise","impulse_noise","defocus_blur","glass_blur","motion_blur", # 6
            "zoom_blur","increase_brightness","contrast","elastic","pixelate","jpeg_filter","shear_image", #13
            "translate_image", "scale_image", "rotate_image", "fog_mapping", "splatter_mapping",#18
            "dotted_lines_mapping", "zigzag_mapping","canny_edges_mapping","speckle_noise_filter",#22
            "false_color_filter","high_pass_filter","low_pass_filter","phase_scrambling",#26
            "histogram_equalisation", "reflection_filter", "white_balance_filter", "sharpen_filter",#30
            "grayscale_filter", "posterize_filter", "cutout_filter", "sample_pairing_filter", "gaussian_blur",#35
            "saturation_filter", "saturation_decrease_filter", "fog_filter", "frost_filter", "snow_filter",#40 snow frost
            "object_overlay","static_lightning_filter", "static_smoke_filter","static_sun_filter","static_rain_filter",#45
            "static_snow_filter", # all works until here 
        ]
        PERTURBATION_NAMES_no_static_no_dynamic = ["new_rain_filter", # 30 frames with intensity 0, 21 at int 1
            "new_dynamic_rain_filter", # 29.9 frames with intensity 0, 18 at int 1
            "dynamic_snow_filter",
    "dynamic_rain_filter",
    "dynamic_raindrop_filter",
    "object_overlay",
    "dynamic_object_overlay",
    "dynamic_sun_filter",
    "dynamic_lightning_filter",
    "dynamic_smoke_filter",
        ]
        #self.perturbation_name = PERTURBATION_NAMES_no_static_no_dynamic[42]
        # Read the index from a ROS parameter, default to 0 if none is provided
        #idx = rospy.get_param('~perturbation_index', 0)
        
        # Add a safety check so it doesn't crash on bad inputs
        #if idx < 0 or idx >= len(PERTURBATION_NAMES_no_static_no_dynamic):
        #    rospy.logwarn(f"Index {idx} is out of bounds. Defaulting to 0.")
        #    idx = 0
            
        self.perturbation_name = PERTURBATION_NAMES_no_static_no_dynamic[2]
        
        # Set intensity (usually 0 to 4)
        self.intensity = 1
        
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
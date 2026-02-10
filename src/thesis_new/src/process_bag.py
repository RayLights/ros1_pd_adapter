#!/usr/bin/env python3
import rosbag
import csv
import os
from cv_bridge import CvBridge
from tqdm import tqdm
from perturbation_node import process_image

# PATHS
INPUT_BAG = os.path.expanduser('~/Downloads/thesis_data.bag')
OUTPUT_BAG = os.path.expanduser('~/Downloads/thesis_data_perturbed.bag')
CSV_LOG = os.path.expanduser('~/Downloads/latency_results.csv')
TARGET_TOPIC = '/gmsl_camera/front_narrow/image_raw'
NEW_TOPIC = '/gmsl_camera/front_narrow/perturbed'

# True for 50-frame test, False for full run
TEST_MODE = True  

print(f"Logging latency to: {CSV_LOG}")
input_bag_obj = rosbag.Bag(INPUT_BAG)
total_msgs = input_bag_obj.get_message_count()
bridge = CvBridge()

with rosbag.Bag(OUTPUT_BAG, 'w') as outbag, open(CSV_LOG, 'w', newline='') as csvfile:
    # Setup CSV writer
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_index', 'latency_ms'])
    
    pbar = tqdm(input_bag_obj.read_messages(), total=total_msgs)
    processed_count = 0

    for topic, msg, t in pbar:
        if topic == TARGET_TOPIC:
            if TEST_MODE and processed_count >= 50:
                continue 

            try:
                # Process and record timing
                out_msg, latency = process_image(bridge, msg)
                outbag.write(NEW_TOPIC, out_msg, t)
                
                # Write to CSV
                csv_writer.writerow([processed_count, latency])
                
                processed_count += 1
                pbar.set_description(f"Last Frame: {latency:.1f}ms")
            except Exception as e:
                print(f"Error: {e}")
        else:
            outbag.write(topic, msg, t)

input_bag_obj.close()
print(f"\nCSV Log saved. You processed {processed_count} frames.")

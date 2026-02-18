#!/bin/bash

# ==========================================
# LiDAR / Camera Perturbation Test Runner
# ==========================================

TOTAL_PERTURBATIONS=26
TEST_DURATION=5  # How many seconds to run each perturbation

echo "Starting Perturbation Test Sequence..."
echo "IMPORTANT: Make sure your rosbag is playing in another terminal so images are publishing!"
echo "--------------------------------------------------------"

for i in $(seq 0 $((TOTAL_PERTURBATIONS - 1))); do
    echo "Testing perturbation index: $i"
    
    # Launch the python script in the background and pass the index as a private ROS parameter
    python3 lidar_perturbations.py _perturbation_index:=$i &
    NODE_PID=$!
    
    # Let it run for the specified duration to process frames
    sleep $TEST_DURATION
    
    # Check if the process is still running
    if ps -p $NODE_PID > /dev/null
    then
       echo -e "\e[32m[SUCCESS]\e[0m Index $i ran without crashing."
       
       # Send SIGINT (Ctrl+C) to gracefully shut down the ROS node
       kill -INT $NODE_PID
       wait $NODE_PID 2>/dev/null
    else
       echo -e "\e[31m[FAILED]\e[0m Index $i crashed during execution!"
    fi
    
    echo "--------------------------------------------------------"
    sleep 1 # Brief pause before spinning up the next one
done

echo "Test sequence complete."
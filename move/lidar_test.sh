#!/bin/bash

# ==========================================
# LiDAR Perturbation Test & Performance Profiler
# ==========================================

TOTAL_PERTURBATIONS=28
MEASUREMENT_TIME=6  # How many seconds to measure the Hz for each test

echo -e "\e[36mStarting LiDAR Perturbation Performance Test Sequence...\e[0m"
echo "IMPORTANT: Make sure your rosbag is playing in another terminal!"
echo "It MUST be publishing /velodyne_points and /detected_objects_bboxes"

for i in $(seq 0 $((TOTAL_PERTURBATIONS - 1))); do
    echo -e "\n========================================================"
    echo -e "\e[33mTesting LiDAR perturbation index: $i\e[0m"
    echo -e "========================================================"
    
    # 1. Launch the python script in the background
    python3 lidar_perturbations.py _perturbation_index:=$i &
    NODE_PID=$!
    
    # 2. Give the node 2 seconds to initialize and start publishing
    sleep 2
    
    # Check if it crashed immediately on startup
    if ! ps -p $NODE_PID > /dev/null; then
       echo -e "\e[31m[FAILED]\e[0m Index $i crashed on startup!"
       continue
    fi

    echo "Measuring publish rate on /velodyne_points/perturbed..."
    echo "--------------------------------------------------------"
    
    # 3. Start rostopic hz in the background to profile performance
    rostopic hz /velodyne_points/perturbed &
    HZ_PID=$!
    
    # 4. Let it measure for the specified duration
    sleep $MEASUREMENT_TIME
    
    # 5. Kill the rostopic hz measurement tool silently
    kill -9 $HZ_PID 2>/dev/null
    wait $HZ_PID 2>/dev/null
    
    echo "--------------------------------------------------------"
    
    # 6. Check if the perturbation node survived the test
    if ps -p $NODE_PID > /dev/null
    then
       echo -e "\e[32m[SUCCESS]\e[0m Index $i ran without crashing."
       
       # Gracefully shut down the ROS node
       kill -INT $NODE_PID
       wait $NODE_PID 2>/dev/null
    else
       echo -e "\e[31m[FAILED]\e[0m Index $i crashed during execution!"
    fi
    
done

echo -e "\n\e[36mLiDAR test sequence complete.\e[0m"
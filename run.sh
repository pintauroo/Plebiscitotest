#!/bin/bash

# Define the start and end index
START_INDEX=1
END_INDEX=30

# Loop over the range
for i in $(seq $START_INDEX $END_INDEX)
do
    # Call the Python script with the current index
    python3 cluster-trace-gpu-v2020/simulator/run_simulator.py $i
done

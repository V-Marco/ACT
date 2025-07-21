#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=nodes
#SBATCH --job-name=monitoring
#SBATCH --output=output/ram_logs.out
#SBATCH --time 0-48:00



START=$(date)

# Define the log file name with a timestamp
LOG_FILE="resource_usage_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting CPU and RAM usage logging to $LOG_FILE. Press Ctrl+C to stop."
echo "---------------------------------------------------" >> "$LOG_FILE"
echo "Timestamp | CPU Usage (%) | RAM Used (MB) | RAM Free (MB)" >> "$LOG_FILE"
echo "---------------------------------------------------" >> "$LOG_FILE"

while true; do
    # Get CPU usage using top in batch mode, grabbing the idle percentage
    # Then calculate actual usage (100 - idle)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | \
                sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
                awk '{print 100 - $1}')

    # Get RAM usage using free -m (in megabytes)
    # total used free shared buff/cache available
    # Mem: 15858      9092      1766        270       4999       6211
    RAM_INFO=$(free -m | grep "Mem:")
    RAM_USED=$(echo "$RAM_INFO" | awk '{print $3}')
    RAM_FREE=$(echo "$RAM_INFO" | awk '{print $4}')

    # Get current timestamp
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Append data to the log file
    echo "$TIMESTAMP | $CPU_USAGE | $RAM_USED | $RAM_FREE" >> "$LOG_FILE"

    # Wait for one second
    sleep 1
done
END=$(date)

printf "Start: $START \nEnd:   $END\n"

echo "Done running model at $(date)"
#!/bin/bash

# List of dates
dates=("06-Sep-2018" "10-May-2018" "11-Aug-2018" "15-May-2018" "17-May-2018")

# Base URL
base_url="https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/mSM60/SpatialDisc/"

# Output directory
output_dir="/home/sachinks/Data/raw/mouse-cshl"

# Loop over the dates and run wget commands
for date in "${dates[@]}"; do
    # Construct the full URL with the current date
    url="${base_url}${date}/BehaviorVideo/"

    # Run wget command
    wget -r -np -nH --cut-dirs=4 --accept-regex '.*\/SVD_Cam.*-Seg.*\.mat' --reject 'index.html.*' "$url" -P "$output_dir"

    echo "Downloaded files for date: $date"
done

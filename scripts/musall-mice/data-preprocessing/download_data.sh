#!/bin/bash

# List of dates
dates=('05-Sep-2018' '06-Jun-2018' '07-Sep-2018' '08-Jun-2018' '11-Sep-2018' '12-Jun-2018')

mouse_id="mSM61"

# Base URL
base_url="https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/${mouse_id}/SpatialDisc"

# Output directory
output_dir="/home/sachinks/Data/raw/mouse-cshl"

# Loop over the dates and run wget commands
for date in "${dates[@]}"; do
    # Construct the full URL with the current date
    url="${base_url}/${date}/BehaviorVideo/"

    # Run wget command
    wget -r -np -nH --cut-dirs=4 --accept-regex '.*\/SVD_Cam.*-Seg.*\.mat' --reject 'index.html.*' "$url" -P "$output_dir"

    echo "Downloaded files for date: $date"
done
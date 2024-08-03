# To download data from the experiment 'Single-trial neural dynamics are dominated by richly varied movements. by Churchland et al. 
# 
# 'https://labshare.cshl.edu/shares/library/repository/38599/


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pandas as pd
import os
import yaml
import argparse
import requests

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--file_list', type=str, required=True,
      help='Path to the csv file containing list of web paths of the data to be downloaded. \
        Get this file from "https://www.dropbox.com/home/MED-draeloslab/Mice_mussal/file_paths.csv"' )
    parser.add_argument( '--download_dir', type=str, required=True,
      help='Path to where the files should be downloaded to. \
        (eg: "/home/<username>/Data/raw/mouse-cshl")' )
    return parser


def find_paths(base_url, pattern_url):
    def is_valid_path(href):
        if href.startswith('/shares/'): # skip looking at parent directory
            return False
        if href.startswith('?'): # irrelevant files
            return False
        return True

    matching_paths = []

    def find_files_recursive(url):
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all links on the page
            links = soup.find_all("a")

            for link in links:
                # Get the URL
                href = link.get("href")

                # Join the URL with the base URL to get the complete path
                file_url = urljoin(url, href)

                if not is_valid_path(href):
                    continue
                
                # Check if the URL matches the specified pattern
                if re.match(pattern_url, file_url):
                    matching_paths.append(file_url)
                    print(f'{len(matching_paths)}: {file_url}')
                    continue

                if href.endswith('/'):
                    find_files_recursive(file_url)
        else:
            print("Failed to retrieve the webpage:", url)

    # Start the recursive search from the base URL
    find_files_recursive(base_url)

    return matching_paths

def main(args):
    base_url = "https://labshare.cshl.edu/shares/library/repository/38599/2pData/Animals/"
    pattern_url = "https://labshare\.cshl\.edu/shares/library/repository/38599/2pData/Animals/.*/SpatialDisc/.*/BehaviorVideo/SVD_Cam.*-Seg.*\.mat"
    file_list = args.file_list
    download_dir = args.download_dir

    if os.path.exists(file_list):
        print("Precomputed values already exist. Loading it...")
        df = pd.read_csv(file_list)
        matching_paths = df.iloc[:, 0].values.tolist()
    else:
        print("Precomputed values do not exist. Searching it...")
        matching_paths = find_paths(base_url, pattern_url)
        df = pd.DataFrame(matching_paths)
        df.to_csv(file_list, index=False, header=False)

    # List of URLs to download
    urls = matching_paths[:2]

    # Loop through the list of URLs and download each one
    for url in urls:
        file_path = url.split("/")
        file_path = [file_path[-7], file_path[-5], file_path[-3], file_path[-2], file_path[-1]]
        file_path = '/'.join(file_path)

        # Create the full path to save the file
        file_path = f'{download_dir}/{file_path}'

        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to a local file
            directory = os.path.dirname(file_path)

            # Check if the directory exists, and if not, create it
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {url}")
        else:
            print(f"Failed to download: {url}")


if __name__ == "__main__":    
    main(create_parser().parse_args())


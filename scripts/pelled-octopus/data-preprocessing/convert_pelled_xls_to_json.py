import argparse
import pandas as pd
import json

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--file_path', type=str, required=True,
      help='Path to the metadata.xlsx file (eg: /home/<username>/Data/raw/octopus/metadata.xlsx)' )
    return parser

def main(args):
    # metadata_xls_path = ""
    metadata_xls_path = args.file_path
    df = pd.read_excel(metadata_xls_path) 
    df2 = df.copy()
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])

    video_data = []

    for index, row in df.iterrows():
        video_item = {
            "id": index,
            "file_name": row['File Name'],
            "experiment_date": str(row['Experiment Date']),
            "stimulation_type": row['Stimulation Type'],
            "movement_classification": int(row['Classification']),
            "stimulation_time": float(row['End (s)']),
        }
        video_data.append(video_item)

    json_data = {"video_data": video_data}
    json_string = json.dumps(json_data, indent=4)
    print(json_string)
    # "fps": 30, // frames per second
    # "duration": 200, // duration of videos in frames
    # "margin": 20, // the number of frames to be kept before the stimulus is applied

if __name__ == "__main__":    
    main(create_parser().parse_args())

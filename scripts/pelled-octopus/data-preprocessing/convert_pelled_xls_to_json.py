import pandas as pd
import json

def main(input_path, output_path):
    df = pd.read_excel(input_path) 
    df2 = df.copy()
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])

    video_data = []

    for index, row in df.iterrows():
        video_item = {
            "id": index,
            "file_name": row['File Name'],
            "experiment_date": str(row['Experiment Date'].date()),
            "stimulation_type": row['Stimulation Type'],
            "movement_classification": int(row['Classification']),
            "stimulation_time": float(row['End (s)']),
        }
        video_data.append(video_item)

    json_data = {
            "fps": 30, # frames per second
            "duration": 200, # duration of videos in frames
            "margin": 20, # the number of frames to be kept before the stimulus is applied
            "video_data": video_data,
        }
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == "__main__":
    metadata_xls_path = "/home/sachinks/Data/raw/octopus/metadata.xlsx"
    output_json_path = "/home/sachinks/Data/raw/octopus/metadata.json"
    main(metadata_xls_path, output_json_path)

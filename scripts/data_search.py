import os


def search_for_video_folder(root_folder):
    """
    Search through all subfolders of the given root folder to find if they contain
    a folder named 'video'. Prints out the path to each folder that contains a 'video' folder.
    """
    print('Beginning search...')
    for root, dirs, files in os.walk(root_folder):
        # Check if 'Video' or 'Videos' is among the directories in the current root
        if 'video' in [d.lower() for d in dirs] or 'videos' in [d.lower() for d in dirs]:
            print(f"'Video' or 'Videos' folder found in: {root}")
    print("Search complete.")

# Example usage
if __name__ == "__main__":
    root_folder = "Z:\Data\Monkeys\Joker"  # Replace this with the path to the top-level folder you want to search
    search_for_video_folder(root_folder)


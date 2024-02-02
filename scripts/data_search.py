import os

def search_for_video_folder(root_folder):
    """
    Search through all subfolders of the given root folder to find if they contain
    a folder named 'video'. Prints out the path to each folder that contains a 'video' folder.
    """
    for root, dirs, files in os.walk(root_folder):
        # Check if 'video' is among the directories in the current root
        if 'Video' in dirs:
            print(f"'video' folder found in: {root}")

# Example usage
if __name__ == "__main__":
    root_folder = "Z:\Data\Monkeys"  # Replace this with the path to the top-level folder you want to search
    search_for_video_folder(root_folder)

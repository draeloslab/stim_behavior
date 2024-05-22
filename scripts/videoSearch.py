import os

def search_files(directory, filename_part):
    """
    Search for files containing a specified substring in their names in all subdirectories of the given directory.
    
    Args:
    directory (str): The root directory from which to start the search.
    filename_part (str): The substring to search for in filenames.
    
    Returns:
    list of str: A list containing the paths to the files where the filenames contain the specified substring.
    """
    matches = []
    # Walk through all directories and files in 'directory'
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the substring is part of the current file's name
            if filename_part in filename:
                matches.append(os.path.join(root, filename))
    return matches

# Example usage
if __name__ == "__main__":
    root_dir = input("Enter the root directory to start search: ")
    part_of_filename = input("Enter the substring to look for in filenames: ")
    found_files = search_files(root_dir, part_of_filename)
    
    if found_files:
        print("Found the following matches:")
        for file_path in found_files:
            print(file_path)
    else:
        print("No files found matching the criteria.")

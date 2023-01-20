import os

def get_files_count(folder_path):
	dirListing = os.listdir(folder_path)
	return len(dirListing)

  
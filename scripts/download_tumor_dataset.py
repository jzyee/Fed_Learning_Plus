import kagglehub

# Define the desired download path
custom_path = "/media/kwanz/New Volume/datasets"

# Download dataset to the specified path
path = kagglehub.dataset_download("fernando2rad/brain-tumor-mri-images-17-classes", path=custom_path)

print("Path to dataset files:", path)
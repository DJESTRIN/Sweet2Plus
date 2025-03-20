import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as gb
import os 
import ipdb
from multiprocessing import Pool
from math import dist as dist
import ipdb


# Step 1: Define the directory where your .npy file is located
input_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\plane0"
# Replace with your desired path
npy_filename = 'F.npy'  # Replace with your .npy file name

# Step 2: Load the .npy file from the specific directory
npy_file_path = os.path.join(input_directory, npy_filename)
npy_data = np.load(npy_file_path)

# Step 3: Convert the NumPy array to a pandas DataFrame (optional but useful)
df = pd.DataFrame(npy_data)
print (df.shape)

#Select a specific row in the dataframe 
y = df.iloc[0,:]
x = pd.Series(range(1, 1001))
#x = np.arange(len(y))
 

#Plot 
plt.plot(x, y, label="Y = 2X")
plt.show()

area = np.trapz(y,x)
print (area)
ipbd.set_trace()


# Step 4: Define the directory where you want to save the CSV file
output_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\plane0"  # Replace with your desired path

# Ensure the output directory exists, if not create it
os.makedirs(output_directory, exist_ok=True)

# Step 5: Specify the full path for the CSV file
csv_filename = 'output_file.csv'  # Name of the output .csv file
csv_file_path = os.path.join(output_directory, csv_filename)

# Step 6: Save the DataFrame to the specified CSV file
df.to_csv(csv_file_path, index=False)

# Optional: Confirm the file was saved successfully
print(f"CSV file has been saved to {csv_file_path}")
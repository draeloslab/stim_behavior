import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the uploaded CSV file
file_path = '/home/jakejoseph/Desktop/Joseph_Code/FESNewCamera-Jake-2024-04-19/dlc-models/iteration-0/FESNewCameraApr19-trainset95shuffle1/train/learning_stats.csv'
data = pd.read_csv(file_path)


# Plotting the loss against the steps
plt.figure(figsize=(10, 5))
plt.plot(data[0], data[1], marker='o', linestyle='-', color='blue')
plt.title('Loss Over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

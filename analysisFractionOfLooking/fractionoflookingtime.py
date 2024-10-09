#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 05:52:20 2024

@author: fatma
"""

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_and_append_csv(folder_path):
    appended_data = []

    # Iterate through each CSV file in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") and "looked_at_counts_subj" in file_name:
            file_path = os.path.join(folder_path, file_name)

            # Extract subject number from the file name using regular expression
            match = re.search(r'subj(\d+)', file_name)
            if match:
                subject_number = int(match.group(1))
            else:
                print(f"Could not extract subject number from file: {file_name}")
                continue

            # Read CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Add a 'Subject_Number' column to the DataFrame
            df['Subject_Number'] = subject_number

            # Append the DataFrame to the list
            appended_data.append(df)

    # Concatenate all DataFrames in the list
    result_df = pd.concat(appended_data, ignore_index=True)

    return result_df

# Replace 'folder_path' with the path to your folder containing CSV files
folder_path ='/Users/fatma/Desktop/PlanningCode/analysisFractionOfLooking/FractionOfLookingAtForeachsubjectAfterextractedFromMeanCode/'
appended_df = read_and_append_csv(folder_path)
# Count occurrences of each 'Looked_At_Counts' value
count_data = appended_df['Looked_At_Counts'].value_counts().sort_index()
total_trials = count_data.sum()

# Calculate the probability for each bin
probability_data = count_data / total_trials

# Set the figure size
plt.figure(figsize=(8,6))

# Create a bar plot to show the probabilities manually
plt.bar(probability_data.index, probability_data.values, color='black', edgecolor='black', width=0.98)

# Set the x-axis limits to range from 0 to 9 and set ticks at every integer
plt.xlim(0,10)  # Limit x-axis from 0 to 9 only
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],fontname='Times New Roman', fontsize=20)  # Set the ticks explicitly to 0-9

# Set the y-axis limits to range from 0 to 1 and set ticks at intervals of 0.1
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 only
plt.yticks([0,0.2,0.4,0.6,0.8], fontsize=24, fontname="Times New Roman")  # Set the ticks explicitly to avoid 1.0

# Set labels with adjusted font sizes
plt.xlabel('Number of Looked-At Items', fontname="Times New Roman", fontsize=24)
plt.ylabel('Fraction of Trials', fontname="Times New Roman", fontsize=24)


# Draw horizontal and vertical lines for cleaner axis look
plt.axhline(y=0, color='black', linewidth=1)
plt.axvline(x=0, color='black', linewidth=1)

# Remove the plot box
plt.box(False)

# Disable grid
plt.grid(False)
# Adjust tick label font size and style
plt.xticks(fontname="Times New Roman", fontsize=24)
plt.yticks(fontname="Times New Roman", fontsize=24)

# Save and show the plot with tight layout to avoid cut-off elements
plt.savefig('llastlookplot.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

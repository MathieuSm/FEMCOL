import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal as sig

# Load data and plot normalized force and displacement curves
Cwd = Path.cwd()
DataPath = Cwd / '02_Data/00_Tests'
FilteredData = [File for File in os.listdir(str(DataPath)) if File.startswith('Filtered')]

Data = pd.read_csv(str(DataPath / FilteredData[0]))

Figure, Axes = plt.subplots(1,1)
Axes.plot(Data['Time [s]'], Data['Force [N]'] / Data['Force [N]'].max(), color=(0,0,1), label='Normalized force')
Axes.plot(Data['Time [s]'], Data['Displacement [mm]'] / Data['Displacement [mm]'].max(), color=(1,0,0), label='Normalized displacement')
Axes.legend()
plt.show()

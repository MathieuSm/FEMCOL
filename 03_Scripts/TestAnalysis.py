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

# Extract signal for stiffness analysis
StartTime = 30
StopTime = 215
StartIndex = np.argmin(np.abs(Data['Time [s]'] - StartTime))
StopIndex = np.argmin(np.abs(Data['Time [s]'] - StopTime))

ForceSig = Data['Force [N]'][StartIndex:StopIndex] / Data['Force [N]'].max()
DisplacementSig = Data['Displacement [mm]'][StartIndex:StopIndex] / Data['Displacement [mm]'].max()

Figure, Axes = plt.subplots(1,1)
Axes.plot(Data['Time [s]'][StartIndex:StopIndex],ForceSig, color=(0,0,1), label='Normalized force')
Axes.plot(Data['Time [s]'][StartIndex:StopIndex],DisplacementSig, color=(1,0,0), label='Normalized displacement')
Axes.legend()
plt.show()

# Find peak positions and heights to compute stiffness
PeaksPos, PeaksH = sig.find_peaks(ForceSig, height=0.25)
TroughsPos, TroughsH = sig.find_peaks(-ForceSig, height=-0.1)
DeltaForces = Data['Force [N]'].max() * (ForceSig[StartIndex + PeaksPos].values -
                                         ForceSig[StartIndex + TroughsPos].values)

PeaksPos, PeaksH = sig.find_peaks(DisplacementSig, height=0.14)
TroughsPos, TroughsH = sig.find_peaks(-DisplacementSig, height=-0.1)
DeltaDisplacements = Data['Displacement [mm]'].max() * (DisplacementSig[StartIndex + PeaksPos].values -
                                                        DisplacementSig[StartIndex + TroughsPos].values)

Stiffness = DeltaForces / DeltaDisplacements

# Plot stiffness obtained
plt.rc('font', **{'size':12})
RandPos = np.random.normal(0,0.02,len(Stiffness))
Figure, Axis = plt.subplots(1,1)
Axis.boxplot(Stiffness, vert=True, widths=0.35,
             showmeans=False,meanline=True,
             capprops=dict(color=(0,0,0)),
             boxprops=dict(color=(0,0,0)),
             whiskerprops=dict(color=(0,0,0),linestyle='--'),
             flierprops=dict(color=(0,0,0)),
             medianprops=dict(color=(0,0,1)),
             meanprops=dict(color=(0,1,0)))
Axis.plot(RandPos - RandPos.mean() + 1,Stiffness, linestyle='none',
          marker='o',fillstyle='none', color=(1,0,0), label='Data')
Axis.plot([],color=(0,0,1), label='Median')
Axis.set_xticks([])
Axis.set_ylabel('Stiffness [N/mm]')
plt.legend()
plt.subplots_adjust(left=0.25, right=0.75)
plt.show()

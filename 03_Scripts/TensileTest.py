from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load data
Cwd = Path.cwd()
DataPath = Cwd / '02_Data/00_Tests'
Data = pd.read_csv(str(DataPath / 'pilot_polymer_failure.csv'),header=2)
Data.columns = ['Time [s]', 'Axial Force [N]',
                'Load cell [N]', 'Axial Displacement [mm]',
                'MTS Displacement [mm]', 'Axial Count [cycles]']

Figure, Axis = plt.subplots(1,1)
Axis.plot(Data['Axial Displacement [mm]'], Data['Axial Force [N]'], color=(1,0,0))
Axis.set_xlabel('Axial Displacement [mm]')
Axis.set_ylabel('Axial Force [N]')
plt.show()
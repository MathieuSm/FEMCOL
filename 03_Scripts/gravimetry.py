# This script loads results from gravimetric analysis and calculates density and the three different weight fractions

# load various packages
import pandas as pd
import os


# specify file path
Cwd = os.getcwd()
data_path = str(os.path.dirname(Cwd) + '/02_Data/03_Scale/Gravimetric_analysis.xlsx')
save_path = str(os.path.dirname(Cwd) + '/04_Results/02_Gravimetry/')

df = pd.read_excel(data_path)
H20_density = 0.99754

# calculate density with density=wet weight * density of H2O / (wet weight - H2O weight)
for row in range(len(df)):
    density = round(df['wet weight'] * H20_density / (df['wet weight'] - df['H2O weight']), 3)
    w_organic = round(df['dry weight'] - df['ash weight'], 3)
    w_mineral = round(df['ash weight'], 3)
    w_water = round(df['wet weight'] - df['dry weight'], 3)
    wf_mineral = round(df['ash weight'] / df['wet weight'], 3)
    wf_organic = round(w_organic / df['wet weight'], 3)
    wf_water = round(1 - wf_organic - wf_mineral, 3)

# add results to dataframe
result_dir = pd.DataFrame()
result_dir['Sample ID'] = df['Sample ID']
result_dir['age'] = df['age']
result_dir['sex'] = df['sex']
result_dir['site'] = df['site']
result_dir['density / g/cm^3'] = density
result_dir['organic weight / g'] = w_organic
result_dir['mineral weight / g'] = w_mineral
result_dir['water weight / g'] = w_water
result_dir['weight fraction of mineral phase / -'] = wf_mineral
result_dir['weight fraction of organic phase / -'] = wf_organic
result_dir['weight fraction of water phase / -'] = wf_water

# save dataframe to csv
result_dir.sort_values(by=['Sample ID'], inplace=True, ascending=True)
result_dir = result_dir.reset_index(drop=True)
result_dir.to_csv(os.path.join(save_path, 'ResultsGravimetry.csv'), index=False)

print(result_dir)

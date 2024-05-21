# This script combines the results of several tests into one file called ResultsOverview.csv

import pandas as pd
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
results_path = str(__location__ + '/04_Results')
data_path = str(__location__ + '/02_Data')
results_mineralized = pd.read_csv(str(results_path + '/00_Mineralized/ResultsElasticTesting.csv'), skiprows=0)
results_demineralized = pd.read_csv(str(results_path + '/01_Demineralized/ResultsFailureTesting.csv'), skiprows=0)
results_gravimetry = pd.read_csv(str(results_path + '/02_Gravimetry/ResultsGravimetry.csv'), skiprows=0)
results_uCT = pd.read_csv(str(results_path + '/03_uCT/ResultsUCT.csv'), skiprows=0)
results_raman = pd.read_csv(str(results_path + '/05_Raman/RamanResults.csv'), skiprows=0)
results_uFE = pd.read_csv(str(results_path + '/06_microFE/uFE_Data_Benjamin.csv'), skiprows=0)
results_segmentation = pd.read_csv(str(data_path + '/04_Segmentation/SegmentationDataMeanStd.csv'), skiprows=0, sep=';')

results_merged = pd.DataFrame()
results_merged['Sample ID'] = results_mineralized['Sample ID']
results_merged['Age / y'] = results_gravimetry['age']
results_merged['Gender'] = results_gravimetry['sex']
results_merged['Site'] = results_gravimetry['site']

results_merged['Stiffness Mineralized / N/mm'] = results_mineralized['Stiffness N/mm']
results_merged['Stiffness Demineralized / N/mm'] = results_demineralized['Stiffness / N/mm']
results_merged['Apparent Modulus Mineralized / GPa'] = results_mineralized['Apparent modulus MPa']/1000
results_merged['Modulus Mineralized / GPa'] = results_mineralized['Modulus Mineralized MPa']/1000

results_merged['Ultimate Force / N'] = results_demineralized['Ultimate Force / N']
results_merged['Ultimate Apparent Stress / MPa'] = results_demineralized['Ultimate stress / MPa']
results_merged['Ultimate Collagen Stress / MPa'] = results_demineralized['Ultimate collagen stress / MPa']
results_merged['Ultimate Stress / MPa'] = results_demineralized['Ultimate stress non-app / MPa']
results_merged['Ultimate Strain / -'] = results_demineralized['Ultimate strain / -']
results_merged['Apparent Modulus Demineralized / MPa'] = results_demineralized['Apparent modulus / MPa']
results_merged['Modulus Demineralized / MPa'] = results_demineralized['Modulus demineralized / MPa']

results_merged['Density / g/' + 'cm\u00B3'] = results_gravimetry['density / g/cm^3']
results_merged['Organic Weight / g'] = results_gravimetry['organic weight / g']
results_merged['Mineral Weight / g'] = results_gravimetry['mineral weight / g']
results_merged['Water Weight / g'] = results_gravimetry['water weight / g']
results_merged['Mineral weight fraction / -'] = results_gravimetry['weight fraction of mineral phase / -']
results_merged['Organic weight fraction / -'] = results_gravimetry['weight fraction of organic phase / -']
results_merged['Water weight fraction / -'] = results_gravimetry['weight fraction of water phase / -']

results_merged['Bone Volume Fraction / -'] = results_uCT['Bone Volume Fraction / -']
results_merged['Bone Mineral Density / mg HA / ' + 'cm\u00B3'] = results_uCT['Bone Mineral Density mg HA / cm3']
results_merged['Tissue Mineral Density / mg HA / ' + 'cm\u00B3'] = results_uCT['Tissue Mineral Density mg HA / cm^3']
results_merged['Bone Mineral Content / mg HA'] = results_uCT['Bone Mineral Content / mg HA']
results_merged['Mean Bone Area Fraction / -'] = results_uCT['Mean Area Fraction / -']
results_merged['Min Bone Area Fraction / -'] = results_uCT['Min Area Fraction / -']

results_merged['Mineral to Matrix Ratio v2/a3 / -'] = results_raman['M2M_ratio_v2/a3_mean']
results_merged['Mineral to Matrix Ratio v1/a1 / -'] = results_raman['M2M_ratio_v1/a1_mean']
results_merged['Crystallinity / -'] = results_raman['crystallinity_mean']
results_merged['Collagen dis/order / -'] = results_raman['coll_dis/order_1670/1640_mean']
results_merged['Matrix maturity / -'] = results_raman['matrix_maturity_1660/1683_mean']
results_merged['Relative Pyridinoline Content / -'] = results_raman['Pyd/matrix_1660/amideI_mean']
results_merged['Relative Proteoglycan Content / -'] = results_raman['PG_amideIII_mean']
results_merged['Relative Lipid Content / -'] = results_raman['lipid_amideIII_mean']

results_merged['Apparent Modulus Mineralized uFE / MPa'] = round(results_uFE['E_uFE_L'], 3)*1000
results_merged['Yield Stress uFE / MPa'] = round(results_uFE['yield_strength_uFE_with_non_broken'], 2)
results_merged['Ultimate Stress uFE / MPa'] = round(results_uFE['strength_uFE_with_non_broken'], 2)

results_merged['Haversian Canals Mean / %'] = round(results_segmentation['Haversian canals mean / %'], 3)
results_merged['Haversian Canals Std / %'] = round(results_segmentation['Haversian canals std / %'], 3)
results_merged['Osteocytes Mean / %'] = round(results_segmentation['Osteocytes mean / %'], 3)
results_merged['Osteocytes Std / %'] = round(results_segmentation['Osteocytes std / %'], 3)
results_merged['Cement Lines Mean / %'] = round(results_segmentation['Cement lines mean / %'], 3)
results_merged['Cement Lines Std / %'] = round(results_segmentation['Cement lines std / %'], 3)

results_merged.to_csv(results_path + '/ResultsOverview.csv', index=False)

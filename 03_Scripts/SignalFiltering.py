from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Load data and plot force displacement curve
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

# Plot force signal in time domain
Figure, Axis = plt.subplots(1,1)
Axis.plot(Data['Time [s]'], Data['Axial Force [N]'], color=(1,0,0))
Axis.set_xlabel('Time [s]')
Axis.set_ylabel('Axial Force [N]')
plt.show()

# # Artificial signal force (for filtering understanding purpose)
# SamplingFrequency = 0.0005
# SamplingInterval = 1/SamplingFrequency
# t = np.linspace(0, 1.0, int(SamplingInterval)+1)
# xlow = np.sin(2 * np.pi * 5 * t)
# plt.plot(xlow)
# plt.show()
# xhigh = np.sin(2 * np.pi * 250 * t)
# plt.plot(xhigh)
# plt.show()
# x = xlow + xhigh
# plt.plot(x)
# plt.show()

# Analyze signal spectrum in frequency domain
Signal = Data['Axial Force [N]']
SamplingInterval = Data['Time [s]'][1] - Data['Time [s]'][0]
SamplingFrequency = 1 / SamplingInterval
NormalizedSpectrum = np.fft.fft(Signal) / len(Signal)
Frequencies = np.fft.fftfreq(Signal.shape[-1], SamplingFrequency)

RealHalfSpectrum = np.abs(NormalizedSpectrum.real[Frequencies >= 0])
HalfFrequencies = Frequencies[Frequencies >= 0]
plt.semilogx(HalfFrequencies, RealHalfSpectrum)
plt.ylim([-0.01,0.3])
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Amplitude (-)')
plt.show()

PeakFrequencies = HalfFrequencies[sig.find_peaks(RealHalfSpectrum, height=1/10)[0]]

# Design filter according to signal frequencies to remove
FilterOrder = 2
CutOffFrequency = PeakFrequencies[-1] * 2E-1
b, a = sig.butter(FilterOrder, CutOffFrequency, 'low', analog=True)
w, h = sig.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(PeakFrequencies[-1], color='green') # cutoff frequency
plt.ylim([-50,5])
plt.show()


# Filter signal and look local filtering effect
SOS = sig.butter(FilterOrder, CutOffFrequency * SamplingFrequency, output='sos')
FilteredSignal = sig.sosfiltfilt(SOS, Signal)

Start, Stop = 1000, 2000
Figure, Axis = plt.subplots(1,1)
Axis.plot(Data['Time [s]'][Start:Stop],Signal[Start:Stop], color=(0,0,0))
Axis.plot(Data['Time [s]'][Start:Stop],FilteredSignal[Start:Stop], color=(1,0,0))
plt.show()

Data['Filtered Axial Force [N]'] = FilteredSignal

# Apply same filter on displacement signal
Signal = Data['Axial Displacement [mm]']
FilteredSignal = sig.sosfiltfilt(SOS, Signal)

Start, Stop = 1000, 2000
Figure, Axis = plt.subplots(1,1)
Axis.plot(Data['Time [s]'][Start:Stop],Signal[Start:Stop], color=(0,0,0))
Axis.plot(Data['Time [s]'][Start:Stop],FilteredSignal[Start:Stop], color=(1,0,0))
plt.show()

Data['Filtered Axial Displacement [mm]'] = FilteredSignal

# Plot filtered force-displacement curve
Figure, Axis = plt.subplots(1,1)
Axis.plot(Data['Axial Displacement [mm]'], Data['Axial Force [N]'], color=(0,0,0), label='Raw signals')
Axis.plot(Data['Filtered Axial Displacement [mm]'], Data['Filtered Axial Force [N]'], color=(1,0,0), label='Filtered signals')
Axis.set_xlabel('Axial Displacement [mm]')
Axis.set_ylabel('Axial Force [N]')
Axis.legend()
plt.show()

# Export filtered signals for further analysis
Data2Export = Data[['Time [s]', 'Filtered Axial Displacement [mm]', 'Filtered Axial Force [N]']]
Data2Export.columns = ['Time [s]', 'Displacement [mm]', 'Force [N]']
DataName = 'FilteredSignals_' + str(FilterOrder) + 'Order_' +\
           str("{:.2e}".format(CutOffFrequency)) + 'CutOff.csv'
Data2Export.to_csv(str(DataPath / DataName), index=False)
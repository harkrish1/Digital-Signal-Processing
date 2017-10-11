from pylab import*
from scipy.io import wavfile
from scipy import signal
import pygame

sampFreq_snd, snd = wavfile.read('speech_file_1.wav')

len_snd = len(snd)

timeArray = arange(0, len_snd, 1)/ float(sampFreq_snd) # equally spaced time values for plotting graph

#Calculating square of signal for power
power = np.array(snd, dtype='int64')**2

plt.plot(timeArray, power, color='k')
ylabel('Power')
xlabel('Time (seconds)')
plt.title('Time-domain Power graph')
plt.show()

# Calculating intensity of sound
row= 1.225 # kg/m^3, Density of air
c = 340  # m/s, speed of sound in air

intensity = power / float(row*c) # Intensity = power / (density * speed)
plt.plot(timeArray, intensity, color='k')
ylabel('Intensity')
xlabel('Time (seconds)')
plt.title('Time-domain Intensity graph')
plt.show()

#Calculating Energy of total signal
#Enenrgy is the sum of signal's power over a time period

tot_energy = sum(power)
# print (tot_energy)
#120270311641

# Calculating Sound Level of signal
# Ratio of signal power to a baseline power, represented as = 10 * log10(p2/p1), here assuming p1=1
sound_level = 10 * log10(power)

plt.plot(timeArray, sound_level, color='k')
ylabel('Sound Level')
xlabel('Time (seconds)')
plt.title('Time-domain Sound Level graph')
plt.show()

#################################################################################
## Noise
#################################################################################

# Reading Noise File
sampFreq_nse, nse = wavfile.read('noise_file.wav')

#Play Audio
pygame.mixer.init(frequency=sampFreq_nse)
pygame.mixer.music.load('noise_file.wav')
# pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue
# Plays audio

timeArray_nse = arange(0, len(nse), 1)/ float(sampFreq_nse)

plt.plot(timeArray_nse, nse, color='k')
ylabel('Amplitude')
xlabel('Time (seconds)')
plt.title('Time-domain Noise Signal')
plt.show()

#Calculating square of noise signal for power
power_nse = np.array(nse, dtype='int64')**2

# Calculating Sound Level of Noise signal
# Ratio of signal power to a baseline power, represented as = 10 * log10(p2/p1), here assuming p1=1
sound_level_nse = 10 * log10(power_nse)

plt.plot(timeArray_nse, sound_level_nse, color='k')
ylabel('Sound Level')
xlabel('Time (seconds)')
plt.title('Time-domain Sound Level graph for Noise')
plt.show()

#######################################################################################
## Mixing Signals
#######################################################################################

#Adding Signals - Noise and speech
mixture = snd + nse
wavfile.write('mixture.wav',sampFreq_nse,mixture)

pygame.mixer.init(frequency=sampFreq_snd)
pygame.mixer.music.load('mixture.wav')
# pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

	
#Calculating fourier transform for signal
len_snd = len(snd)
x = fft(snd)
mag_spec = abs(x/len_snd)
mag_spec1 = mag_spec[:int(len_snd/2+1)]
mag_spec1[1:-1] = 2*mag_spec1[1:-1]
freqs = sampFreq_snd * arange(0,(len_snd/2))/len_snd
phase_spec = angle(x)
phase_spec1 = phase_spec[:int(len_snd/2+1)]


#Calculating fourier transform for noise
len_nse = len(nse)
x_nse = fft(nse)
mag_spec_nse = abs(x_nse/len_nse)
mag_spec1_nse = mag_spec_nse[:int(len_nse/2+1)]
mag_spec1_nse[1:-1] = 2*mag_spec1_nse[1:-1]
freqs_nse = sampFreq_nse * arange(0,(len_nse/2))/len_nse
phase_spec_nse = angle(x_nse)
phase_spec1_nse = phase_spec_nse[:int(len_nse/2+1)]

#Calculating fourier transform for mixture
len_mix = len(mixture)
x_mix = fft(mixture)

mag_spec_mix = abs(x_mix/len_mix)
mag_spec1_mix = mag_spec_mix[:int(len_mix/2+1)]
mag_spec1_mix[1:-1] = 2*mag_spec1_mix[1:-1]
freqs_mix = sampFreq_nse * arange(0,(len_mix/2))/len_mix
phase_spec_mix = angle(x)
phase_spec1_mix = phase_spec_mix[:int(len_snd/2+1)]

#Plotting Graphs
fig, ax = plt.subplots(nrows=3,ncols=2)

# Plotting Signal FFT in graph
plt.subplot(3,2,1)
plt.plot(freqs,mag_spec1) 
plt.title('Frequency-domain Magnitude of Signal') 
plt.ylabel('Magnitude Response, |X|')   

plt.subplot(3,2,2)
plt.plot(freqs,phase_spec1) 
plt.title('Frequency-domain Phase of Signal') 
plt.ylabel('Phase Response') 

# Plotting Noise FFT in graph
plt.subplot(3,2,3)
plt.plot(freqs_nse,mag_spec1_nse) 
plt.title('Frequency-domain Magnitude of Noise') 
plt.ylabel('Magnitude Response, |X|')   

plt.subplot(3,2,4)
plt.plot(freqs_nse,phase_spec1_nse) 
plt.title('Frequency-domain Phase of Noise') 
plt.ylabel('Phase Response') 

# Plotting Mixture FFT in graph
plt.subplot(3,2,5)
plt.plot(freqs_mix,mag_spec1_mix) 
plt.title('Frequency-domain Magnitude Mixture Signal') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Magnitude Response, |X|')   

plt.subplot(3,2,6)
plt.plot(freqs_mix,phase_spec1_mix) 
plt.title('Frequency-domain Phase Mixture Signal') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Phase Response') 

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()

# Calculating Signal to Noise Ratio = 10 * log10(E2/E1)
# E2 = energy of signal, E1 = e ergy of noise
signal_energy = tot_energy
noise_energy = sum(power_nse)
SNR = 10 * log10(signal_energy/noise_energy)
# print (SNR)
#-15.1038767184
###################################################################################################
# Low Pass Filtering for Mixture
###################################################################################################

temp_mix = sampFreq_nse*arange(len_mix/2)/len_mix
# print (len_mix/2,temp_mix, sampFreq_nse)
lpf_mix = temp_mix < 1000
lpf_mix = lpf_mix.astype(float)
plt.plot(temp_mix, lpf_mix) 
plt.title('LowPass Filter on Mixture') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Filter Response')   
plt.show()

# Low Pass Filtering for Signal
temp_snd = sampFreq_snd*arange(0,(len_snd/2))/len_snd
lpf_snd = temp_snd < 1000
lpf_snd = lpf_snd.astype(float)

# Low Pass Filtering for noise
temp_nse = sampFreq_nse*arange(0,(len_nse/2))/len_nse
lpf_nse = temp_nse < 1000
lpf_nse = lpf_nse.astype(float)

######################################################
# Calculating filtered frequency of mixture
######################################################

mag_spec1 = lpf_snd*mag_spec[:int(len_snd/2+1)]
mag_spec1[1:-1] = 2*mag_spec1[1:-1]
freqs = sampFreq_snd * arange(0,(len_snd/2))/len_snd
phase_spec = angle(x)
phase_spec1 = lpf_snd*phase_spec[:int(len_snd/2+1)]

# Calculating filtered fourier transform for noise
mag_spec1_nse = lpf_nse*mag_spec_nse[:int(len_nse/2+1)]
mag_spec1_nse[1:-1] = 2*mag_spec1_nse[1:-1]
freqs_nse = sampFreq_nse * arange(0,(len_nse/2))/len_nse
phase_spec_nse = angle(x_nse)
phase_spec1_nse = lpf_nse*phase_spec_nse[:int(len_nse/2+1)]

#Calculating filtered fourier transform for mixture
mag_spec1_mix = lpf_mix*mag_spec_mix[:int(len_mix/2+1)]
mag_spec1_mix[1:-1] = 2*mag_spec1_mix[1:-1]
freqs_mix = sampFreq_nse * arange(0,(len_mix/2))/len_mix
phase_spec_mix = angle(x)
phase_spec1_mix = lpf_mix*phase_spec_mix[:int(len_snd/2+1)]

#Plotting filtered Graphs
fig, ax = plt.subplots(nrows=3,ncols=2)

# Plotting filtered Signal FFT in graph
plt.subplot(3,2,1)
plt.plot(freqs,mag_spec1) 
plt.title('Magnitude of Signal') 
plt.ylabel('Magnitude Response, |X|')   

plt.subplot(3,2,2)
plt.plot(freqs,phase_spec1) 
plt.title('Phase of Signal') 
plt.ylabel('Phase Response') 

# Plotting filtered Noise FFT in graph
plt.subplot(3,2,3)
plt.plot(freqs_nse,mag_spec1_nse) 
plt.title('Magnitude of Noise') 
plt.ylabel('Magnitude Response, |X|')   

plt.subplot(3,2,4)
plt.plot(freqs_nse,phase_spec1_nse) 
plt.title('Phase of Noise') 
plt.ylabel('Phase Response') 

# Plotting filtered Mixture FFT in graph
plt.subplot(3,2,5)
plt.plot(freqs_mix,mag_spec1_mix) 
plt.title('Magnitude of Mixture') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Magnitude Response, |X|')   

plt.subplot(3,2,6)
plt.plot(freqs_mix,phase_spec1_mix) 
plt.title('Phase of Mixture') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Phase Response') 

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.suptitle('Frequency-domain Plots with Low Pass Filter') 
plt.show()

# Inverse Fourier Transform of filtered signal
# print (x_mix[:int(len_mix/2+1)], '\n\n',x_mix[int(len_mix/2+1):])
# print (flipud(lpf_mix[1:]), '\n\n',lpf_mix[1:])
S_fft = lpf_mix * x_mix[:int(len_mix/2+1)]
S_fft_conj = flipud(lpf_mix[1:]) * x_mix[int(len_mix/2+1):]
s_t = real(ifft(append(S_fft,S_fft_conj)))
# print (ifft(append(S_fft,S_fft_conj)))
s_t = s_t.astype(int16)

timeArray_fil = arange(0, len(s_t), 1)/ float(sampFreq_nse)

plt.plot(timeArray_fil, s_t, color='k')
ylabel('Amplitude')
xlabel('Time (seconds)')
plt.title('Time-domain plot of filtered signal')
plt.show()

wavfile.write('filtered_mixture.wav',sampFreq_nse,s_t)
pygame.mixer.init(frequency=sampFreq_nse)
pygame.mixer.music.load('filtered_mixture.wav')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

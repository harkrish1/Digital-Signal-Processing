from pylab import*
from scipy.io import wavfile
from scipy import signal
import pygame

sampFreq, snd = wavfile.read('speech_file_1.wav')
print (snd, sampFreq)
#[113 149 136 ...,  29  26  20] 16000

print ('Length of data: '+ str(len(snd))+ ' samples' )
#Length of data: 48941 samples

pygame.mixer.init()
pygame.mixer.music.load('speech_file_1.wav')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue
# Plays audio

#Plotting time domain representation of signal

timeArray = arange(0, len(snd), 1)/ float(sampFreq)
# timeArray = timeArray * 1000  #scale to milliseconds, Optional

plt.plot(timeArray, snd, color='k')
ylabel('Amplitude')
xlabel('Time (seconds)')
plt.title('Time-domain Signal')
plt.show()

#Calculating fourier transform
len_snd = len(snd)
x = fft(snd)
print (x[:5])
# [ 7453067.00000000+0.j          
# -804557.34956574-336147.86765385j
#  127401.04129164-492390.42317344j  
# -777734.12498461-827942.35354979j
# -391667.05193138-197883.95313763j ]

#Some Calculations
mag_spec = abs(x/len_snd)
mag_spec1 = mag_spec[:int(len_snd/2+1)]
mag_spec1[1:-1] = 2*mag_spec1[1:-1]
freqs = sampFreq * arange(0,(len_snd/2))/len_snd

phase_spec = angle(x)
phase_spec1 = phase_spec[:int(len_snd/2+1)]
print (phase_spec1)

# Plotting in graph
plt.plot(freqs,mag_spec1) 
plt.title('Frequency-domain Signal') 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Magnitude Response, |X|')   
plt.show()
plt.plot(freqs,phase_spec1) 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Phase Response') 
plt.show()

# Calculating short time fourier transform
win_len = 320
overlap = 160
freq_points = 512

f, t, Zxx = signal.stft(x, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)

temp = 20 * np.log10(np.abs(Zxx)) # putting in decibels
# temp = np.abs(Zxx)
plt.pcolormesh(t, f, temp)
plt.title('Short Time Fourier Transform - Power Spectrum')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.show()

from pylab import*
from scipy.io import wavfile
from scipy import signal
import pygame
# import matlab.engine
# import matlab

sampFreq, sig_1 = wavfile.read('signal_1.wav')
sampFreq, sig_2 = wavfile.read('signal_2.wav')
sampFreq, sig_3 = wavfile.read('signal_3.wav')
sampFreq, sig_4 = wavfile.read('signal_4.wav')


#Plotting time domain representation of signal

timeArray_1 = arange(0, len(sig_1), 1)/ float(sampFreq)
timeArray_2 = arange(0, len(sig_2), 1)/ float(sampFreq)
timeArray_3 = arange(0, len(sig_3), 1)/ float(sampFreq)
timeArray_4 = arange(0, len(sig_4), 1)/ float(sampFreq)

fig, ax = plt.subplots(nrows=2,ncols=2)

# Plotting Signal in graph
# plt.subplot(2,2,1)
# plt.plot(timeArray_1, sig_1, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Signal 1')

# plt.subplot(2,2,2)
# plt.plot(timeArray_2, sig_2, color='k')
# xlabel('Time (seconds)')
# ylabel('Amplitude')
# plt.title('Time-domain Signal 2')

# plt.subplot(2,2,3)
# plt.plot(timeArray_3, sig_3, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Signal 3')

# plt.subplot(2,2,4)
# plt.plot(timeArray_4, sig_4, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Signal 4')

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# plt.show()

########################################################################################
## Defining Masker
########################################################################################

def scaleMasker(target,masker):
	lt = len(target)
	lm = len(masker)
	if lt >= lm:       # equalize the lengths of the two files
		target = target[:lm]
	else:
		masker = masker[:lt]
	# Scale the masker 
	change = 20*log10(std(target)/std(masker))
	masker1 = masker*(10**(change/20)) # scale masker to specified input SNR
	return masker1

	
masker_1 = scaleMasker(sig_1,sig_2); 
masker_2 = scaleMasker(sig_1,sig_3); 
masker_3 = scaleMasker(sig_1,sig_4); 

timeArray_masker_1 = arange(0, len(masker_1), 1)/ float(sampFreq)
timeArray_masker_2 = arange(0, len(masker_2), 1)/ float(sampFreq)
timeArray_masker_3 = arange(0, len(masker_3), 1)/ float(sampFreq)

# Plotting Maskers in graph
# fig, ax = plt.subplots(nrows=2,ncols=2)

# plt.subplot(2,2,1)
# plt.plot(timeArray_1, sig_1, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Signal 1')

# plt.subplot(2,2,2)
# plt.plot(timeArray_masker_1, masker_1, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Masker 1')

# plt.subplot(2,2,3)
# plt.plot(timeArray_masker_2, masker_3, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Masker 2')

# plt.subplot(2,2,4)
# plt.plot(timeArray_masker_3, masker_3, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Masker 3')

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# plt.show()

#####################################################################################
## Two, Three and Four Talker Signals
#####################################################################################
two_talker = sig_1 + masker_1
three_talker = sig_1 + masker_1 + masker_2
four_talker = sig_1 + masker_1 + masker_2 + masker_3

two_talker = two_talker.astype(int16)
three_talker = three_talker.astype(int16)
four_talker = four_talker.astype(int16)

wavfile.write('two_talker.wav',sampFreq,two_talker)
wavfile.write('three_talker.wav',sampFreq, three_talker)
wavfile.write('four_talker.wav',sampFreq,four_talker)

print ('clean speech 1')

pygame.mixer.init(frequency=sampFreq)
pygame.mixer.music.load('signal_1.wav')
# pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue


print ('playing 1')

pygame.mixer.init(frequency=sampFreq)
pygame.mixer.music.load('two_talker.wav')
#pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

print ('playing 2')

pygame.mixer.init(frequency=sampFreq)
pygame.mixer.music.load('three_talker.wav')
#pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

print ('playing 3')

pygame.mixer.init(frequency=sampFreq)
pygame.mixer.music.load('four_talker.wav')
#pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

######################################################################################
## Spectrograms & STFT
######################################################################################

win_len = 320
overlap = 160
freq_points = 512

f_1, t_1, Zxx_1 = signal.stft(sig_1, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_1 = 20 * np.log10(np.abs(Zxx_1))
# t, x = signal.istft(Zxx_1, fs=sampFreq, nperseg=win_len, noverlap=overlap, nfft=freq_points, input_onesided=True)
# scaled = np.int16(x/np.max(np.abs(x)) * 32767)
# wavfile.write('test.wav',sampFreq,scaled)

f_2, t_2, Zxx_2 = signal.stft(two_talker, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_2 = 20 * np.log10(np.abs(Zxx_2))

f_3, t_3, Zxx_3 = signal.stft(three_talker, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_3 = 20 * np.log10(np.abs(Zxx_3))

f_4, t_4, Zxx_4 = signal.stft(four_talker, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_4 = 20 * np.log10(np.abs(Zxx_4))


# Plotting Spectrogram
# fig, ax = plt.subplots(nrows=2,ncols=2)

# plt.subplot(2,2,1)
# plt.pcolormesh(t_1, f_1, temp_1)
# plt.title('STFT - Clean Signal')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(2,2,2)
# plt.pcolormesh(t_2, f_2, temp_2)
# plt.title('STFT - Two Talker Signal')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(2,2,3)
# plt.pcolormesh(t_3, f_3, temp_3)
# plt.title('STFT - Three Talker Signal')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(2,2,4)
# plt.pcolormesh(t_4, f_4, temp_4)
# plt.title('STFT - Four Talker Signal')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# plt.show()

# Spectrogram for Masker 1, 1 + 2, and 1 + 2 + 3

f_5, t_5, Zxx_5 = signal.stft(masker_1, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points, return_onesided = False)
temp_5 = 20 * np.log10(np.abs(Zxx_5))
print()
print(shape(Zxx_5 ))
f_5, t_5, Zxx_5 = signal.stft(masker_1, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_5 = 20 * np.log10(np.abs(Zxx_5))
print ()
print(shape(Zxx_5))



f_6, t_6, Zxx_6 = signal.stft((masker_1 + masker_2), sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_6 = 20 * np.log10(np.abs(Zxx_6))

f_7, t_7, Zxx_7 = signal.stft((masker_1 + masker_2 + masker_3), sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_7 = 20 * np.log10(np.abs(Zxx_7))

# Plotting Spectrogram
# fig, ax = plt.subplots(nrows=3,ncols=1)

# plt.subplot(3,1,1)
# plt.pcolormesh(t_5, f_5, temp_5)
# plt.title('STFT - Masker 1')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(3,1,2)
# plt.pcolormesh(t_6, f_6, temp_6)
# plt.title('STFT - Masker 1 + 2')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(3,1,3)
# plt.pcolormesh(t_7, f_7, temp_7)
# plt.title('STFT - Masker 1 + 2 + 3')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()

#######################################################################################
## SNR and IBM
#######################################################################################

def snr(signal,noise):
	if len(signal) == len(noise):
		temp_1 = []
		for i in range(len(signal)):
			if len(signal[i]) == len(noise[i]):
				temp_2 = []
				for j in range(len(signal[i])):
					temp_2.append(20*log10(abs(signal[i][j])/abs(noise[i][j])))
				temp_1.append(temp_2)
	return temp_1

snr_1 = snr(temp_2, temp_5)
snr_2 = snr(temp_3, temp_6)
snr_3 = snr(temp_4, temp_7)

IBM_m50dB_2talker = [[1 if y>=-50 else 0 for y in x ]for x in snr_1]
IBM_m10dB_2talker = [[1 if y>=-10 else 0 for y in x ]for x in snr_1]
IBM_0dB_2talker = [[1 if y>=0 else 0 for y in x ]for x in snr_1]
IBM_20dB_2talker = [[1 if y>=20 else 0 for y in x ]for x in snr_1]

IBM_m50dB_3talker = [[1 if y>=-50 else 0 for y in x ]for x in snr_2]
IBM_m10dB_3talker = [[1 if y>=-10 else 0 for y in x ]for x in snr_2]
IBM_0dB_3talker = [[1 if y>=0 else 0 for y in x ]for x in snr_2]
IBM_20dB_3talker = [[1 if y>=20 else 0 for y in x ]for x in snr_2]

IBM_m50dB_4talker = [[1 if y>=-50 else 0 for y in x ]for x in snr_3]
IBM_m10dB_4talker = [[1 if y>=-10 else 0 for y in x ]for x in snr_3]
IBM_0dB_4talker = [[1 if y>=0 else 0 for y in x ]for x in snr_3]
IBM_20dB_4talker = [[1 if y>=20 else 0 for y in x ]for x in snr_3]

two_talker_est_m50dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_m50dB_2talker,temp_2)]
two_talker_est_m10dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_m10dB_2talker,temp_2)]
two_talker_est_0dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_0dB_2talker,temp_2)]
two_talker_est_20dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_20dB_2talker,temp_2)]

three_talker_est_m50dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_m50dB_3talker,temp_3)]
three_talker_est_m10dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_m10dB_3talker,temp_3)]
three_talker_est_0dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_0dB_3talker,temp_3)]
three_talker_est_20dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_20dB_3talker,temp_3)]

four_talker_est_m50dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_m50dB_4talker,temp_4)]
four_talker_est_m10dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_m10dB_4talker,temp_4)]
four_talker_est_0dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_0dB_4talker,temp_4)]
four_talker_est_20dB = [[c*d for c,d in zip(a,b)] for a,b in zip(IBM_20dB_4talker,temp_4)]


# Plotting 4 talker IBM Filtered Spectrogram

# fig, ax = plt.subplots(nrows=2,ncols=2)

# plt.subplot(2,2,1)
# plt.pcolormesh(t_4, f_4, four_talker_est_m50dB)
# plt.title('-50 db IBM')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(2,2,2)
# plt.pcolormesh(t_4, f_4, four_talker_est_m10dB)
# plt.title('-10 db IBM')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(2,2,3)
# plt.pcolormesh(t_4, f_4, four_talker_est_0dB)
# plt.title('0 db IBM')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.subplot(2,2,4)
# plt.pcolormesh(t_4, f_4, four_talker_est_20dB)
# plt.title('20 db IBM')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')

# plt.suptitle('STFT with IBM Plots - Four Talker Signal') 
# plt.tight_layout()
# plt.show()

four_talker_est_m50dB = [[10**(b /20) for b in a] for a in four_talker_est_m50dB]
four_talker_est_m10dB = [[10**(b /20) for b in a] for a in four_talker_est_m10dB]
four_talker_est_0dB = [[10**(b /20) for b in a] for a in four_talker_est_0dB]
four_talker_est_20dB = [[10**(b /20) for b in a] for a in four_talker_est_20dB]


t_four_talker_est_m50dB_time, x_four_talker_est_m50dB_time = signal.istft(four_talker_est_m50dB, fs=sampFreq, nperseg=win_len, noverlap=overlap, nfft=freq_points, input_onesided=True)
t_four_talker_est_m10dB_time, x_four_talker_est_m10dB_time = signal.istft(four_talker_est_m10dB, fs=sampFreq,nperseg=win_len, noverlap=overlap, nfft=freq_points, input_onesided=True)
t_four_talker_est_0dB_time, x_four_talker_est_0dB_time = signal.istft(four_talker_est_0dB, fs=sampFreq, nperseg=win_len, noverlap=overlap, nfft=freq_points, input_onesided=True)
t_four_talker_est_20dB_time, x_four_talker_est_20dB_time = signal.istft(four_talker_est_20dB, fs=sampFreq, nperseg=win_len, noverlap=overlap, nfft=freq_points, input_onesided=True)


#Plotting filtered Graphs
# fig, ax = plt.subplots(nrows=2,ncols=2)

# plt.subplot(2,2,1)
# plt.plot(t_four_talker_est_m50dB_time, x_four_talker_est_m50dB_time)
# plt.title('Time [sec]') 
# plt.ylabel('Signal -50DB IBM')

# plt.subplot(2,2,2)
# plt.plot(t_four_talker_est_m10dB_time, x_four_talker_est_m10dB_time)
# plt.title('Time [sec]')
# plt.ylabel('Signal -10DB IBM')

# plt.subplot(2,2,3)
# plt.plot(t_four_talker_est_0dB_time, x_four_talker_est_0dB_time)
# plt.title('Time [sec]')
# plt.ylabel('Signal 0DB IBM')

# plt.subplot(2,2,4)
# plt.plot(t_four_talker_est_20dB_time, x_four_talker_est_20dB_time)
# plt.title('Time [sec]')
# plt.ylabel('Signal 20DB IBM')

# plt.suptitle('Four Talker Signal IBM Time Domain Plots')
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

x_four_talker_est_m50dB_time = np.int16(x_four_talker_est_m50dB_time/np.max(np.abs(x_four_talker_est_m50dB_time)) * 32767)
x_four_talker_est_m10dB_time = np.int16(x_four_talker_est_m10dB_time/np.max(np.abs(x_four_talker_est_m10dB_time)) * 32767)
x_four_talker_est_0dB_time = np.int16(x_four_talker_est_0dB_time/np.max(np.abs(x_four_talker_est_0dB_time)) * 32767)
x_four_talker_est_20dB_time = np.int16(x_four_talker_est_20dB_time/np.max(np.abs(x_four_talker_est_20dB_time)) * 32767)

f_test, t_test, Zxx_test = signal.stft(x_four_talker_est_0dB_time, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_test = 20 * np.log10(np.abs(Zxx_test))

# Plotting Spectrogram
fig, ax = plt.subplots(nrows=1,ncols=1)

plt.subplot(1,1,1)
plt.pcolormesh(t_test, f_test, temp_test)
plt.title('STFT - IBM 20 DB')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.show()

wavfile.write('four_talker_est_m50dB_time.wav',sampFreq,x_four_talker_est_m50dB_time)
wavfile.write('four_talker_est_m10dB_time.wav',sampFreq,x_four_talker_est_m10dB_time)
wavfile.write('four_talker_est_0dB_time.wav',sampFreq,x_four_talker_est_0dB_time)
wavfile.write('four_talker_est_20dB_time.wav',sampFreq,x_four_talker_est_20dB_time)

# pygame.mixer.init(frequency=sampFreq)
# pygame.mixer.music.load('two_talker_est_m50dB_time.wav')
# pygame.mixer.music.play()
# while pygame.mixer.music.get_busy() == True:
    # continue
print(' ------------------------','\n','done','\n', '------------------------')
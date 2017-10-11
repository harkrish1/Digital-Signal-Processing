from pylab import*
from scipy.io import wavfile
from scipy import signal
import pygame

sampFreq, sig_1 = wavfile.read('noisy_speech.wav')

# timeArray = arange(0, len(sig_1), 1)/ float(sampFreq)
# plt.plot(timeArray, sig_1, color='k')
# ylabel('Amplitude')
# xlabel('Time (seconds)')
# plt.title('Time-domain Noisy Signal')
# plt.show()

win_len = 320
overlap = 160
freq_points = 512

f_1, t_1, Zxx_1 = signal.stft(sig_1, sampFreq, nperseg=win_len, noverlap=overlap,nfft=freq_points)
temp_1 = 20 * np.log10(np.abs(Zxx_1))

# plt.pcolormesh(t_1, f_1, temp_1)
# plt.title('STFT - Clean Signal')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (sec)')
# plt.show()

###########################################################################################
## Defining FFTFILT Function
###########################################################################################

def nextpow2(x):
	"""Return the first integer N such that 2**N >= abs(x)"""
	
	return ceil(log2(abs(x)))

def fftfilt(b, x, *n):
	"""Filter the signal x with the FIR filter described by the
	coefficients in b using the overlap-add method. If the FFT
	length n is not specified, it and the overlap-add block length
	are selected so as to minimize the computational cost of
	the filtering operation."""
	
	N_x = len(x)
	N_b = len(b)

    # Determine the FFT length to use:
	if len(n):

		# Use the specified FFT length (rounded up to the nearest
		# power of 2), provided that it is no less than the filter
		# length:
		n = n[0]
		if n != numpy.int(n) or n <= 0:
			raise ValueError('n must be a nonnegative integer')
		if n < N_b:
			n = N_b
		N_fft = 2**nextpow2(n)
	else:

		if N_x > N_b:

			# When the filter length is smaller than the signal,
			# choose the FFT length and block size that minimize the
			# FLOPS cost. Since the cost for a length-N FFT is
			# (N/2)*log2(N) and the filtering operation of each block
			# involves 2 FFT operations and N multiplications, the
			# cost of the overlap-add method for 1 length-N block is
			# N*(1+log2(N)). For the sake of efficiency, only FFT
			# lengths that are powers of 2 are considered:
			N = 2**arange(ceil(log2(N_b)),floor(log2(N_x)))
			cost = ceil(N_x/(N-N_b+1))*N*(log2(N)+1)
			N_fft = N[argmin(cost)]

		else:

			# When the filter length is at least as long as the signal,
			# filter the signal using a single block:
			N_fft = 2**nextpow2(N_b+N_x-1)

	N_fft = int(N_fft)
    
    # Compute the block length:
	L = int(N_fft - N_b + 1)
  

 
    # Compute the transform of the filter:
	H = fft(b,N_fft)
#    plt.plot(H.real)
#    plt.plot(H.imag)
#    plt.show()

	y = zeros(N_x,float)
	i = 0
	while i <= N_x:
		il = min([i+L,N_x])
		k = min([i+N_fft,N_x])
		yt = ifft(fft(x[i:il],N_fft)*H,N_fft) # Overlap..
		y[i:k] = y[i:k] + (yt.real)[:k-i]            # and add
		i += L
	return y


###################################################################################################
## Defining Gammatone and related Function
###################################################################################################
def hz2erb(hz):
	# Convert normal frequency scale in hz to ERB-rate scale.
	# Units are number of Hz and number of ERBs.
	# ERB stands for Equivalent Rectangular Bandwidth.
	return [21.4*log10(4.37e-3*a+1) for a in hz]
def erb2hz(erb):
# Convert ERB-rate scale to normal frequency scale.
# Units are number of ERBs and number of Hz.
# ERB stands for Equivalent Rectangular Bandwidth.
	return [(10**(a/21.4)-1)/4.37e-3 for a in erb]

def loudness(freq):
	# Compute loudness level in Phons on the basis of equal-loudness functions.
	# It accounts a middle ear effect and is used for frequency-dependent gain adjustments.
	# This function uses linear interpolation of a lookup table to compute the loudness level, 
	# in phons, of a pure tone of frequency freq using the reference curve for sound 
	# pressure level dB. The equation is taken from section 4 of BS3383.
	dB=60
	af = [2.347, 	2.19, 	2.05, 	1.879, 	1.724, 	1.579, 	1.512, 	1.466, 	1.426, 	1.394, 	1.372, 	1.344, 	1.304, 	1.256, 	1.203, 	1.135, 	1.062, 	1, 	0.967, 	0.943, 	0.932, 	0.933, 	0.937, 	0.952, 	0.974, 	1.027, 	1.135, 	1.266, 	1.501 ]
	bf = [0.00561, 	0.00527, 	0.00481, 	0.00404, 	0.00383, 	0.00286, 	0.00259, 	0.00257, 	0.00256, 	0.00255, 	0.00254, 	0.00248, 	0.00229, 	0.00201, 	0.00162, 	0.00111, 	0.00052, 	0, 	-0.00039, 	-0.00067, 	-0.00092, 	-0.00105, 	-0.00104, 	-0.00088, 	-0.00055, 	0, 	0.00089, 	0.00211, 	0.00488]
	cf = [74.3, 	65, 	56.3, 	48.4, 	41.7, 	35.5, 	29.8, 	25.1, 	20.7, 	16.8, 	13.8, 	11.2, 	8.9, 	7.2, 	6, 	5, 	4.4, 	4.2, 	3.7, 	2.6, 	1, 	-1.2, 	-3.6, 	-3.9, 	-1.1, 	6.6, 	15.3, 	16.4, 	11.6]
	ff = [20, 	25, 	31.5, 	40, 	50, 	63, 	80, 	100, 	125, 	160, 	200, 	250, 	315, 	400, 	500, 	630, 	800, 	1000, 	1250, 	1600, 	2000, 	2500, 	3150, 	4000, 	5000, 	6300, 	8000, 	10000, 	12500]
	# Stores parameters of equal-loudness functions from BS3383,"Normal equal-loudness level
	# contours for pure tones under free-field listening conditions", table 1.
	# f (or ff) is the tone frequency, af and bf are frequency-dependent coefficients, and
	# tf is the threshold sound pressure level of the tone, in dBs   
	if (freq<20 or freq>12500):
		return 0  # Returning Null Throwing an error, so made it zero
	i=1
	while ff[i]<freq:
		i=i+1
	afy=af[i-1]+(freq-ff[i-1])*(af[i]-af[i-1])/(ff[i]-ff[i-1])
	bfy=bf[i-1]+(freq-ff[i-1])*(bf[i]-bf[i-1])/(ff[i]-ff[i-1])
	cfy=cf[i-1]+(freq-ff[i-1])*(cf[i]-cf[i-1])/(ff[i]-ff[i-1])
	return 4.2+afy*(dB-cfy)/(1+bfy*(dB-cfy))

def gammatone(input, numChan, fRange, fs):
	# Produce an array of filtered responses from a Gammatone filterbank.
	# The first variable is required. 
	# numChan: number of filter channels.
	# fRange: frequency range.
	# fs: sampling frequency.
	# Written by ZZ Jin, adapted by DLW in Jan'07 and JF Woodruff in Nov'08
	if not numChan:
		numChan = 128       # default number of filter channels in filterbank
	if not fRange:
		fRange = [80, 5000] # default frequency range in Hz
	
	if not fs:
		fs = 16000     # default sampling frequency
	
	filterOrder = 4    #filter order
	gL = 2048          # gammatone filter length or 128 ms for 16 kHz sampling rate
	
	sigLength = len(input)     # input signal length
	
	phase = zeros(numChan)        # initial phases
	erb_b = hz2erb(fRange)       # upper and lower bound of ERB
	erb = []
	for i in frange(erb_b[0],erb_b[1],(erb_b[1]-erb_b[0])/(numChan-1)):
		erb.append(i)    # ERB segment does this need to be in floating point?
	cf = erb2hz(erb)       # center frequency array indexed by channel
	b = [1.019*24.7*(4.37*a/1000+1) for a in cf] # rate of decay or bandwidth
	# cf = asarray(cf)
	# b = asarray(b)
	# Generating gammatone impulse responses with middle-ear gain normalization
	
	gt = zeros((numChan,gL))  # Initialization
	tmp_t = arange(0,gL)/fs
	for i in range(numChan):
		gain = 10**((loudness(cf[i])-60)/20)/3*(2*pi*b[i]/fs)**4    # loudness-based gain adjustments
		gt[i] = gain*(fs**3)*tmp_t**(filterOrder-1)*exp(-2*pi*b[i]*tmp_t)*cos(2*pi*cf[i]*tmp_t+phase[i])
	# sig = reshape(input,(sigLength,1))      # convert input to column vector

	# gt = gt.transpose()

	# gammatone filtering using FFTFILT 
	gamma = []
	for i in range(numChan):
		gamma.append(fftfilt(gt[i],input))
	return gamma


gamma_filtered = gammatone(sig_1,64,[50, 8000],sampFreq) 

def cochleagram(r, *winLength, **overlaplen):
	# Generate a cochleagram from responses of a Gammatone filterbank.
	# It gives the log energy of T-F units
	# The first variable is required.
	# winLength: window (frame) length in samples
	# Written by ZZ Jin, and adapted by DLW in Jan'07

	if not winLength:
		winLength = 320      #default window length in sample points which is 20 ms for 16 KHz sampling frequency

	if not overlaplen:
		overlaplen = 160
	winLength = int(winLength[0])

	numChan = shape(r)[0]     # number of channels and input signal length
	sigLength = shape(r)[1]     # number of channels and input signal length
	winShift = winLength - overlaplen #winLength/2;            # frame shift (default is half frame)
	increment = winLength/winShift    # special treatment for first increment-1 frames
	M = int(fix((sigLength-overlaplen)/(winLength-overlaplen))) # Number of time frames

	# calculate energy for each frame in each channel
	a = zeros((numChan,M))
	for m in range(M):      
		for i in range(numChan):
			if m < increment:        # shorter frame lengths for beginning frames
				a[i][m] = sum(r[i][1:m*winShift]**2)
			else:

				startpoint = int(floor((m-increment)*winShift)	)			
				a[i][m] = sum(r[i][startpoint+1:startpoint+winLength]**2)

	return a
coch = cochleagram(gamma_filtered,  320 )
print (shape(coch))
print (shape(gamma_filtered) )

plt.imshow(coch)
plt.show()

power_coch =  20*log10(coch)

plt.imshow(power_coch, origin='lower left', aspect='auto', vmin=0)
plt.show()





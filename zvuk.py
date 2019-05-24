import scipy.io.wavfile as wv
import numpy as np
import os
import random

rate = 44100
msecSamples = rate/1000

def scalePeriodicSample(inputSample, samples):
	sampleSize = len(inputSample)
	if (samples<sampleSize):
		return inputSample[:samples]
	else:
		return np.tile(inputSample,samples/sampleSize)[:samples]
#base signals		
def getNoise(msecs):
	noise = []
	for i in range(msecs*msecSamples):
		noise.append(-1.0+2*random.random())
	return np.asarray(noise,dtype=np.float32)
	
def getSquare(freq,msecs):
	sampleSize = int(rate/freq)
	sq = []
	for i in range(sampleSize/2):
		sq.append(1)
	for i in range(sampleSize/2):
		sq.append(-1)
	sq = np.asarray(sq,dtype=np.float32)
	samples = msecs*msecSamples
	return scalePeriodicSample(sq,samples)
def getSineSample(freq):
	sampleSize = int(rate/freq)
	sn = np.sin(np.linspace(0, 2*np.pi,sampleSize))
	sn = np.asarray(sn,dtype=np.float32)
	return sn
def getSine(freq,msecs):
	sampleSize = int(rate/freq)
	sn = np.sin(np.linspace(0, 2*np.pi,sampleSize))
	sn = np.asarray(sn,dtype=np.float32)
	samples = msecs*msecSamples
	return scalePeriodicSample(sn,samples)


def getSaw(freq,msecs):
	sampleSize = int(rate/freq)
	sw = np.linspace(-1, 1,sampleSize)
	sw = np.asarray(sw,dtype=np.float32)
	samples = msecs*msecSamples
	return scalePeriodicSample(sw,samples)
	
def getTri(freq,msecs):
	sampleSize = int(rate/freq)
	tr = np.concatenate([np.linspace(0, 1.0,sampleSize/4),np.linspace(1.0, -1.0,sampleSize/2),np.linspace(-1.0,0,sampleSize/4)])
	tr = np.asarray(tr,dtype=np.float32)
	samples = msecs*msecSamples
	return scalePeriodicSample(tr,samples)

def silence(msecs):
	return np.zeros(msecSamples*msecs,dtype=np.float32)

#mixer	
def getMix(inputArr):
	maxSize = len(inputArr[0])
	for arr in inputArr:
		maxSize = max(maxSize,len(arr))
	result = np.zeros(maxSize,dtype=np.float32)
	for i in range(len(result)):
		sSum = 0
		sCount = len(inputArr)#0
		for j in range(len(inputArr)):
			if (i<len(inputArr[j])):
				sSum+=inputArr[j][i]
				#sCount+=1
		result[i] = sSum/sCount
	return result
def volume(input, vol):
	return input*vol

#amp env
def applyEnvelop(input,env):
	result = np.copy(input)
	for i in range(len(input)):
		if (i>=len(env)):
			result[i]=0
		else:
			result[i] = min((env[i]+1)/2.0, input[i]) if input[i]>=0 else max((env[i]+1)/-2.0,input[i])
	return result

def concat(inputArray):
	return np.concatenate(inputArray)


def shiftPhase(input, samples):
	return np.roll(np.copy(input),samples)
def lpf(input):
	result = np.zeros(len(input),dtype=np.float32)
	a = 2
	result[0] = a*input[0]
	for i in range(1,len(result)):
		result[i] = result[i-1] + a*(input[i]-result[i-1])
	return result
	
def delayed(input, feedback, time, vol):
	toMix = []
	for i in range(feedback):
		if i==0:
			echoInput = volume(input,vol)
		else:
			echoInput = volume(toMix[len(toMix)-1],vol)
		toMix.append(concat([silence(time),echoInput]))
	return getMix(toMix)
def chorus(input):
	return getMix([input,delayed(input,4,10,0.5),delayed(input,3,15,0.5),delayed(input,2,20,0.5),delayed(input,1,25,0.5),delayed(input,1,24,0.5)])

def delay(input,feedback,time,vol):
	return getMix([input,delayed(input,feedback,time,vol)])
	
def write(filename, data):
	wv.write(filename,rate,data)

def writeAndPlay(filename,data):
	wv.write(filename,rate,data)
	os.system(filename)
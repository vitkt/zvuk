from zvuk import *


import sklearn.preprocessing
def getUnperiodic(msecs):
	Y = []
	count = int(msecs*msecSamples)
	X = np.linspace(-200*np.pi, 200*np.pi,count)
	for i in range(count):
		x = X[i]
		noise = -1.0+0.3*random.random()
		
		y = np.cbrt((np.sin(x**2+x**3+np.abs(x))))*np.sqrt(np.abs(np.sin(x**2+x**3+np.abs(x))))
		y+=noise
		Y.append(y)
	return np.asarray(sklearn.preprocessing.minmax_scale(Y,feature_range=(-1,1)),dtype=np.float32)
import sklearn.neural_network
def nnSine(freq,msecs,iter):
	mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(3,),max_iter=iter)
	sampleSize = int(rate/freq)
	sinex = np.sin(np.linspace(0, 2*np.pi,sampleSize))
	siney = getSineSample(freq)**2
	mlp.fit(sinex.reshape(-1,1),siney.reshape(-1,1))
	eps = 0.3
	result = mlp.predict((sinex).reshape(-1,1))
	samlpes = msecs*msecSamples
	return scalePeriodicSample(result,samlpes)
import matplotlib.pyplot as plt
def fract(lst,start,end):
	third = (end-start)/3
	if end-start<10:
		return
	for i in range(start,start+third):
		lst[i]=1
	for i in range(start+third,start+third*2):
		lst[i]=-1
	for i in range(start+third*2,end):
		lst[i]=1
	
	fract(lst,start,start+third)
	fract(lst,start+third*2,end)
	



def fractFreq(lst,start,end,minSize):
	third = (end-start)/3
	if end-start<minSize:
		for i in range(start,end):
			lst[i]=1
		return
	for i in range(start,start+third):
		lst[i]=1
	for i in range(start+third,start+third*2):
		lst[i]=-1
	for i in range(start+third*2,end):
		lst[i]=1
	
	fractFreq(lst,start,start+third,3+random.random()*minSize)
	fractFreq(lst,start+third*2,end,3+random.random()*minSize)

def getFractalSq(freq,msecs,minS):
	sampleSize = int(rate/freq)
	sq = np.ones(sampleSize)
	fractFreq(sq,0,sampleSize,minS)
	samples = msecs*msecSamples
	return scalePeriodicSample(sq,samples)
	
def trueFractal(freq,msecs):
	sampleSize = int(rate/freq)
	samples = msecs*msecSamples
	fr = list(range(samples))
	fractFreq(fr,0,len(fr),sampleSize)
	return scalePeriodicSample(fr,samples)


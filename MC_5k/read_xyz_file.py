import numpy as np
import glob

Predictlist = glob.glob('Test_PUNET/output/*.xyz')
inputtlist = glob.glob('Test_PUNET/*.xyz')
#inputtlist = glob.glob('GT/duck.xyz')

for i in Predictlist:
    points = np.loadtxt(i)
    print('Predict File :: ',i.split('/')[-1],' Nb. points :: ',len(points))

for i in inputtlist:
    points = np.loadtxt(i)
    print('Input File :: ',i.split('/')[-1],' Nb. points :: ',len(points))
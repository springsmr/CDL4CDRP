import sys
import numpy as np

# one-hot encoding:k classes
def encoding(filename,k):
    data=np.load(filename)
    n,m=data.shape
    encodingData=np.zeros((n,m*k))
    mask=np.zeros((n,m*k))
    for row in range(n):
        for col in range(m):
            r=int(data[row][col])
            if r<0 or r>=k:
                continue
            encodingData[row][col*k+r]=1
            mask[row][col*k:col*k+k]=1
    return encodingData,mask
if __name__=='__main__':
    if len(sys.argv)!=3:
        print('Usage:python encoding.py filename n_classes')
    encoding=encoding(sys.argv[1],int(sys.argv[2]))
    np.save('encoding',encoding)
    print 'Job Done...'

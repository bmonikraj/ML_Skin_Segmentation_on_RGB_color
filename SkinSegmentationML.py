'''
Skin color segmentation based on R,G,B component given for training set.
Here we can use K means clustering because the skin color cluster will be required and all other data not coming under the cluser
can be ignored and classified as non skin

We are trying to find one cluster basically which will denote skin and  rest anything is non skin. So for the purpose of clustering
we are having 2 clusters. One for skin and other for non skin . As soon as we get skin cluster, we just note that
centroid and fond the maximum and minimum range of the point distance under skin clusetr. Can also find average distance

Here I have implemented K-means clustering using two methods
M1 - Self written algorithm using euclidean geometry and error minimization
M2 - Using scipy.cluster module
'''

import pandas
import numpy
import os
import math
from scipy import cluster

#Reading data
datasetPanda = pandas.read_csv(str(os.getcwd())+"\\dataset-Skin_Nonskin.csv")

################################################################################################################
#Using Self created algorithm
#B_array for Blue component of color
#G_array for Green component of color
#R_array for Red component of color
#T_array for Result 1-skin and 0-nonskin
B_array = numpy.array(datasetPanda['B'])
G_array = numpy.array(datasetPanda['G'])
R_array = numpy.array(datasetPanda['R'])
T_array = numpy.array(datasetPanda['T'])
B_array.astype(float)
G_array.astype(float)
R_array.astype(float)

maxIterations = 100

def euclidianDist(b,g,r,cb,cg,cr):
    return math.sqrt(math.pow(b-cb,2)+math.pow(g-cg,2)+math.pow(r-cr,2))

toBeClustered = True
iterationNumber = 0

clusterSkin = [10000]
clusterNoSkin = [150000]
clusterSkinCentroid = [B_array[10000],G_array[10000],R_array[10000]]
clusterNoSkinCentroid = [B_array[150000],G_array[150000],R_array[150000]]

csb = 0
csg = 0
csr = 0
cnsb = 0
cnsg = 0
cnsr = 0

for i in range(0, len(B_array), 1):
    distSkin = euclidianDist(B_array[i], G_array[i], R_array[i], clusterSkinCentroid[0], clusterSkinCentroid[1],clusterSkinCentroid[2])
    distNoSkin = euclidianDist(B_array[i], G_array[i], R_array[i], clusterNoSkinCentroid[0], clusterNoSkinCentroid[1],clusterNoSkinCentroid[2])
    if distNoSkin >= distSkin and i!=10000:
        clusterSkin.append(i)
        csb = csb + B_array[i]
        csg = csg + G_array[i]
        csr = csr + R_array[i]
        clusterSkinCentroid[0] = float(csb) / float(len(clusterSkin))
        clusterSkinCentroid[1] = float(csg) / float(len(clusterSkin))
        clusterSkinCentroid[2] = float(csr) / float(len(clusterSkin))
    if distNoSkin < distSkin and i!=150000:
        clusterNoSkin.append(i)
        cnsb = cnsb + B_array[i]
        cnsg = cnsg + G_array[i]
        cnsr = cnsr + R_array[i]
        clusterNoSkinCentroid[0] = float(cnsb) / float(len(clusterSkin))
        clusterNoSkinCentroid[1] = float(cnsg) / float(len(clusterSkin))
        clusterNoSkinCentroid[2] = float(cnsr) / float(len(clusterSkin))

while toBeClustered and iterationNumber<=100 :
    errors = 0
    j=0
    s=len(clusterSkin)
    while j<s:
        distSkinChk = euclidianDist(B_array[clusterSkin[j]], G_array[clusterSkin[j]], R_array[clusterSkin[j]], clusterSkinCentroid[0], clusterSkinCentroid[1],clusterSkinCentroid[2])
        distNoSkinChk = euclidianDist(B_array[clusterSkin[j]], G_array[clusterSkin[j]], R_array[clusterSkin[j]], clusterNoSkinCentroid[0],clusterNoSkinCentroid[1], clusterNoSkinCentroid[2])
        if distNoSkinChk < distSkinChk:
            errors = errors + 1
            clusterSkinCentroid[0] = clusterSkinCentroid[0]-(float(float(B_array[clusterSkin[j]])/float(len(clusterSkin))))
            clusterSkinCentroid[0] = clusterSkinCentroid[0]*(float(len(clusterSkin)/(len(clusterSkin)-1)))
            clusterSkinCentroid[1] = clusterSkinCentroid[1] - (float(float(G_array[clusterSkin[j]]) / float(len(clusterSkin))))
            clusterSkinCentroid[1] = clusterSkinCentroid[1] * (float(len(clusterSkin) / (len(clusterSkin) - 1)))
            clusterSkinCentroid[2] = clusterSkinCentroid[2] - (float(float(R_array[clusterSkin[j]]) / float(len(clusterSkin))))
            clusterSkinCentroid[2] = clusterSkinCentroid[2] * (float(len(clusterSkin) / (len(clusterSkin) - 1)))
            clusterNoSkin.append(clusterSkin[j])
            clusterNoSkinCentroid[0] = (clusterNoSkinCentroid[0]*float(len(clusterNoSkin)-1)) + B_array[clusterSkin[j]]
            clusterNoSkinCentroid[0] = float(clusterNoSkinCentroid[0]/len(clusterNoSkin))
            clusterNoSkinCentroid[1] = (clusterNoSkinCentroid[1] * float(len(clusterNoSkin) - 1)) + G_array[clusterSkin[j]]
            clusterNoSkinCentroid[1] = float(clusterNoSkinCentroid[1] / len(clusterNoSkin))
            clusterNoSkinCentroid[2] = (clusterNoSkinCentroid[2] * float(len(clusterNoSkin) - 1)) + R_array[clusterSkin[j]]
            clusterNoSkinCentroid[2] = float(clusterNoSkinCentroid[2] / len(clusterNoSkin))
            clusterSkin.pop(j)
        j=j+1
        s=len(clusterSkin)

    k=0
    t=len(clusterNoSkin)
    while k<t:
        distSkinChk = euclidianDist(B_array[clusterNoSkin[k]], G_array[clusterNoSkin[k]], R_array[clusterNoSkin[k]],clusterSkinCentroid[0], clusterSkinCentroid[1], clusterSkinCentroid[2])
        distNoSkinChk = euclidianDist(B_array[clusterNoSkin[k]], G_array[clusterNoSkin[k]], R_array[clusterNoSkin[k]],clusterNoSkinCentroid[0], clusterNoSkinCentroid[1], clusterNoSkinCentroid[2])
        if distSkinChk < distNoSkinChk:
            errors = errors + 1
            clusterNoSkinCentroid[0] = clusterNoSkinCentroid[0] - (float(float(B_array[clusterNoSkin[k]]) / float(len(clusterNoSkin))))
            clusterNoSkinCentroid[0] = clusterNoSkinCentroid[0] * (float(len(clusterNoSkin) / (len(clusterNoSkin) - 1)))
            clusterNoSkinCentroid[1] = clusterNoSkinCentroid[1] - (float(float(G_array[clusterNoSkin[k]]) / float(len(clusterNoSkin))))
            clusterNoSkinCentroid[1] = clusterNoSkinCentroid[1] * (float(len(clusterNoSkin) / (len(clusterNoSkin) - 1)))
            clusterNoSkinCentroid[2] = clusterNoSkinCentroid[2] - (float(float(R_array[clusterNoSkin[k]]) / float(len(clusterNoSkin))))
            clusterNoSkinCentroid[2] = clusterNoSkinCentroid[2] * (float(len(clusterNoSkin) / (len(clusterNoSkin) - 1)))
            clusterSkin.append(clusterNoSkin[k])
            clusterSkinCentroid[0] = (clusterSkinCentroid[0] * float(len(clusterSkin) - 1)) + B_array[clusterNoSkin[k]]
            clusterSkinCentroid[0] = float(clusterSkinCentroid[0] / len(clusterSkin))
            clusterSkinCentroid[1] = (clusterSkinCentroid[1] * float(len(clusterSkin) - 1)) + G_array[clusterNoSkin[k]]
            clusterSkinCentroid[1] = float(clusterSkinCentroid[1] / len(clusterSkin))
            clusterSkinCentroid[2] = (clusterSkinCentroid[2] * float(len(clusterSkin) - 1)) + R_array[clusterNoSkin[k]]
            clusterSkinCentroid[2] = float(clusterSkinCentroid[2] / len(clusterSkin))
            clusterNoSkin.pop(k)
        k=k+1
        t=len(clusterNoSkin)
    print "Errors found:" + str(errors)
    if errors == 0:
        toBeClustered = False
    else:
        toBeClustered = True
    iterationNumber = iterationNumber + 1

print "BGR centroid of skin is : " + str(clusterSkinCentroid)
print "Number of iteration is:"+str(iterationNumber)

DistSkin = []

for p in range(0,len(clusterSkin),1):
    DistSkin.append(euclidianDist(B_array[clusterSkin[p]],G_array[clusterSkin[p]],R_array[clusterSkin[p]],clusterSkinCentroid[0],clusterSkinCentroid[1],clusterSkinCentroid[2]))

maxDist = max(DistSkin)
minDist = min(DistSkin)

print "number of skin:"+str(len(clusterSkin))
print "Range of distance from centroid for skin is ["+str(minDist)+","+str(maxDist)+"]"

##########################################################################################
#Using Scipy.cluster Module for clustering
del datasetPanda['T']

array = numpy.array(datasetPanda)
print array
#array = cluster.vq.whiten(array)
print datasetPanda
print array
array=array.astype(float)
res = cluster.vq.kmeans2(array,2,20)
print res
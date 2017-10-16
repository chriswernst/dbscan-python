## DBSCAN in Python
###### October 2017

###
###

### Overview

###### In this README, we'll walk through the `dbscan.py` example code to show how the algorithm works.

###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59616bad_screen-shot-2017-07-08-at-4.32.22-pm/screen-shot-2017-07-08-at-4.32.22-pm.png)
*Original data on the left and clusters identified by the DBSCAN algorithm on the right. For DBSCAN clusters, large colored points represent core cluster members, small colored points represent cluster edge members, and small black points represent outliers.*

###
###

[**Here is a very cool tool, built by Naftali Harris, for helping to visualize DBSCAN clustering!**](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

###

[**The Wiki Page for DBSCAN may also be useful to provide some background**](https://en.wikipedia.org/wiki/DBSCAN#Algorithm)

###

### DBSCAN Algorithm *(Density-Based Spatial Clustering of Applications with Noise)*
*Sometimes called Euclidean Clustering*

###

DBSCAN is a nice alternative to k-means when you don't know how many clusters to expect in your data, but you do know something about how the points should be clustered in terms of density (distance between points in a cluster).

###



###

DBSCAN datapoints **do not have to be spatial data; they can be color data, intensity values, or other numerical features!** This means we can cluster not only based upon proximity, but we can cluster similarly colored objects!
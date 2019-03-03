#!/usr/bin/env python3

import os
import sys
from math import sqrt
import numpy as np

def read_training(d):
    tr = {}
    for file in os.listdir(d):
        n = int(file.split(".")[0])
        with open(os.path.join(d,file)) as f:
            fc = f.read()
        ss = fc.split("\n\n")
        ls = []
        ns = 0
        for s in ss:
                      
            tts = s.split("\n")
            xs, ys = [],[]
            for t in tts:
                if not t.strip():
                    continue
                x,y = t.strip().split(" ")
                xs.append(float(x))
                ys.append(float(y))
            ls.append((np.array(xs),np.array(ys)))
        tr[n] = ls           
    return tr


def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d

def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results


def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2    
    return sortedLst[index] 


class Traj:
    def __init__(self,xsys):
        xs, ys = xsys
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.xd = np.diff(xs)
        self.yd = np.diff(ys)
        self.dists = np.linalg.norm([self.xd, self.yd],axis=0)
        self.cuts = np.cumsum(self.dists)
        self.d = np.hstack([0,self.cuts])
        
    def getPoints(self, offsets):        
        offdists = offsets * self.cuts[-1]
        ix = np.searchsorted(self.cuts, offdists)        
        offdists -= self.d[ix]
        segoffs = offdists/self.dists[ix]
        x = self.xs[ix] + self.xd[ix]*segoffs
        y = self.ys[ix] + self.yd[ix]*segoffs
        return x,y        

class SampleSet:
    def __init__(self, n, ll):
        # ll is list of tuples [x_array,y_array] for every trajectory in sample
        self.trajs = [Traj(l) for l in ll]
        self.n = n
        self.xp = None
        self.yp = None
        
    def getAvg(self, lims, eps):
        self.endpoints()
        if len(self.trajs) > 3:
            ts, _, _ = self.getfilteredtrajs(lims)
        else:
            ts = self.trajs
        if len(ts) < 2:
            ts = self.trajs
        
        trajLen = median([len(t.xs) for t in ts])
        offs = np.linspace(0,1,trajLen)
        xm = []
        ym = []
        for t in ts:
            xs, ys = t.getPoints(offs)            
            xm.append(xs)
            ym.append(ys)
        
        self.xp, self.yp = zip(*rdp(list(zip(np.mean(xm, axis=0),np.mean(ym, axis=0))), eps))
    
    def endpoints(self):
        cs = np.array([[self.trajs[0].xs[0],self.trajs[0].xs[-1]],
                       [self.trajs[0].ys[0],self.trajs[0].ys[-1]]])
        xs = np.hstack([t.xs[0] for t in self.trajs] + [t.xs[-1] for t in self.trajs])
        ys = np.hstack([t.ys[0] for t in self.trajs] + [t.ys[-1] for t in self.trajs])       
        clabs = []
        oldclabs = []
        for j in range(10):
            for i in range(len(xs)):
                ap = np.array([[xs[i]],[ys[i]]])
                dists = np.linalg.norm(ap - cs, axis=0)
                clabs.append(np.argmin(dists))
            cx = np.array([
                np.mean(xs[np.where(np.array(clabs)==0)]),
                np.mean(xs[np.where(np.array(clabs)==1)])])
            cy = np.array([
                np.mean(ys[np.where(np.array(clabs)==0)]),
                np.mean(ys[np.where(np.array(clabs)==1)])])
            if oldclabs == clabs:
                break
            oldclabs = clabs
            clabs = []
        
        for i,l in enumerate(clabs[:len(clabs)//2]):
            if l == 1:
                oldT = self.trajs[i]                
                reversedTraj = (np.flip(oldT.xs), np.flip(oldT.ys))
                self.trajs[i] = Traj(reversedTraj)
                
    def zdist(self):
        xms = np.array([(t.xs[0]+t.xs[-1])/2 for t in self.trajs])
        yms = np.array([(t.ys[0]+t.ys[-1])/2 for t in self.trajs])
        xm = np.mean(xms)
        ym = np.mean(yms)
        return zscore(np.linalg.norm(np.array([[xm],[ym]]) - np.array([xms,yms]), axis=0))
        
    def zlen(self):
        ls = np.array([t.cuts[-1] for t in self.trajs])
        return zscore(ls)
        
    def getfilteredtrajs(self, lims):
        zl = self.zlen()
        zd = self.zdist()
        lenlim, dislim= lims

        lenout = [self.trajs[i] for i in np.where((zl<lenlim[0])|(zl>lenlim[1]))[0]]
        disout = [self.trajs[i] for i in np.where((zd<dislim[0])|(zd>dislim[1]))[0]]
        
        lenix = (zl>lenlim[0])&(zl<lenlim[1])
        disix = (zd>dislim[0])&(zd<dislim[1])

        filtix = np.where(lenix&disix)[0]

        filtered = [self.trajs[i] for i in filtix]
        return filtered, lenout, disout
        
def zscore(l):
    if len(np.unique(l)) == 1:
        return np.full(len(l),0.)
    return (np.array(l)  - np.mean(l)) / np.std(l)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("use", sys.argv[0], "<data_directory> > solution.txt", file=sys.stderr)
        sys.exit(1)
    ts = read_training(sys.argv[1])
    # limiting Z-scores for length
    lenlim = (-1.3, 1.95)
    # limiting zscores for distance (only cuts the large values)
    dislim = (-8., 1.9)
    limits = (lenlim, dislim)
    for i in sorted(ts.keys()):
        ss = SampleSet(i,ts[i])
        # .01 is eps for Ramer–Douglas–Peucker algorithm
        ss.getAvg(limits, .01)
        for x,y in zip(ss.xp, ss.yp):
            print("%1.7f" % x,"%1.7f" % y)
        print()

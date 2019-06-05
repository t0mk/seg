#!/usr/bin/env python3

import os
import sys
from math import sqrt
import numpy as np
import io


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

class OnlyOnePointError(Exception):
    pass

class Traj:
    def __init__(self,xsys):
        xs, ys = xsys
        a = np.array(xsys).T
        _, filtered = np.unique(a, return_index=True,axis=0)

        if len(filtered) < 2:
            raise OnlyOnePointError()
        self.xs = np.array(xs)[sorted(filtered)]
        self.ys = np.array(ys)[sorted(filtered)]
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
        self.trajs = []
        for l in ll:
            try:
                t = Traj(l)
            except OnlyOnePointError:
                continue
            self.trajs.append(t)
        self.n = n
        self.xp = None
        self.yp = None
        self.err= None
        self.d = None
        self.filtix = None
        self.lenoutix = None
        self.disoutix = None
        self.eps = None

    def getRawAvg(self):
        trajLen = median([len(t.xs) for t in self.trajs])
        offs = np.linspace(0,1,trajLen)
        xm = []
        ym = []
        for t in self.trajs:
            xs, ys = t.getPoints(offs)
            xm.append(xs)
            ym.append(ys)        
        xp, yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        return xp, yp

    def getFiltered(self, dismax, lenlim):
        xa, ya = self.getRawAvg()
        d = zscore(np.array([disterr(t.xs, t.ys, xa, ya) for t in self.trajs]))
        l = self.zlen()
        self.lenoutix = np.where((l<lenlim[0])|(l>lenlim[1]))[0]
        
        lenix = np.where((l>lenlim[0])&(l<lenlim[1]))[0]
        self.disoutix = np.where(d>dismax)[0]
        
        disix = np.where(d<dismax)[0]
        self.d = d
        self.l = l
        self.filtix = np.intersect1d(lenix,disix)
        
    def getAvg(self, dismax, lenlim, eps):
        self.eps = eps
        self.endpoints()        
        self.getFiltered(dismax, lenlim)

        if len(self.filtix)<5:
            nbest = 4
            distrank = np.argsort(self.d)
            self.disoutix = distrank[nbest:]            
            self.lenoutix = []
            self.filtix = distrank[:nbest]
        filtered = [self.trajs[i] for i in self.filtix]
        trajLen = median([len(t.xs) for t in filtered])
        offs = np.linspace(0,1,trajLen*10)
        xm = []
        ym = []
        for t in filtered:
            xs, ys = t.getPoints(offs)            
            xm.append(xs)
            ym.append(ys)
        self.xp, self.yp = zip(*rdp(list(zip(np.mean(xm, axis=0),np.mean(ym, axis=0))), eps))
        #self.err = disterr(self.xp, self.yp, tx,ty)
    
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
                reversedTraj = (np.flip(oldT.xs, axis=0), np.flip(oldT.ys, axis=0))
                self.trajs[i] = Traj(reversedTraj)                
    
    def zlen(self):
        ls = np.array([t.cuts[-1] for t in self.trajs])
        return zscore(ls)
        
    def pax(self, ax):
        
        ax.set_xlim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_ylim(0,1)
                
        for _, t in enumerate(self.trajs):    
            ax.plot(t.xs,t.ys, c="b", marker="o", markersize=2)
        for n, t in enumerate([self.trajs[i] for i in self.disoutix]):    
            ax.plot(t.xs,t.ys, c="g")
        for n, t in enumerate([self.trajs[i] for i in self.lenoutix]):    
            ax.plot(t.xs,t.ys, c="cyan")
        for n, t in enumerate([self.trajs[i] for i in np.intersect1d(self.lenoutix,self.disoutix)]):    
            ax.plot(t.xs,t.ys, c="magenta")
        if self.xp is not None:
            ax.plot(self.xp,self.yp, marker='D', color='r', linewidth=3)            
        tx = truth[self.n][0]
        ty = truth[self.n][1]
        ax.plot(tx,ty, marker="o",color="black",linewidth=3)
            
        
        if self.xp is not None:
            ax.set_xlabel("#%d err:%.3f eps:%.3f,t: %d, f: %d out: %d" % 
                          (self.n, self.err, self.eps, len(self.trajs), len(self.filtix),
                           len(self.disoutix)+len(self.lenoutix)))

def disterr(x1,y1, x2, y2):        
    sd = np.array([x1[0]-x2[0],y1[0]-y2[0]])
    ed = np.array([x1[0]-x2[-1],y1[0]-y2[-1]])
    if np.linalg.norm(sd) > np.linalg.norm(ed):
        x2 = np.flip(x2, axis=0)
        y2 = np.flip(y2, axis=0)
        
    offs = np.linspace(0,1,10)
    xrs1, yrs1 = Traj((x1,y1)).getPoints(offs)
    xrs2, yrs2 = Traj((x2,y2)).getPoints(offs)
    return np.sum(np.linalg.norm([xrs1-xrs2, yrs1-yrs2],axis=0))

def zscore(l):
    if len(np.unique(l)) == 1:
        return np.full(len(l),0.)
    return (np.array(l)  - np.mean(l)) / np.std(l)


def getResults(d, dismax, lenlim, eps):
    datastring = io.StringIO()
    ts = read_training(d)
    for i in sorted(ts.keys()):
        ss = SampleSet(i,ts[i])
        # .01 is eps for Ramer–Douglas–Peucker algorithm
        ss.getAvg(dismax, lenlim, eps)
        for x,y in zip(ss.xp, ss.yp):
            print("%1.7f" % x,"%1.7f" % y, file=datastring)
        print(file=datastring)
    ret = datastring.getvalue()
    datastring.close()
    return ret

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("use", sys.argv[0], "<data_directory> > solution.txt", file=sys.stderr)
        sys.exit(1)
    # limiting Z-scores for length
    lenlim = (-1, 21)
    # limiting zscores for distance (only cuts the large values)

    dismax = 2.13 
    lenlim = (-1.23,1.8129)
    eps =.0755

    r = getResults(sys.argv[1], dismax, lenlim, eps)
    print(r)

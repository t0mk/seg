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
        self.filtered = None
        self.lenout = None
        self.disout = None
        
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
        self.err= None
        
    def getAvg(self, lims, eps):
        self.endpoints()
        if len(self.trajs) > 3:
            self.filtered, self.lenout, self.disout = self.getfilteredtrajs(lims)
        else:
            self.filtered, self.lenout, self.disout = self.trajs, [], []
        #print(len(self.filtered))
        if len(self.filtered)<2:
            nbest = 3
            # a lot of noise in this sample group. Pick 3 closest to each other
            distrank = np.argsort(self.zdist())
            self.lenout = []
            self.disout = [self.trajs[i] for i in distrank[nbest:]]
            self.filtered = [self.trajs[i] for i in distrank[:nbest]]                        
        
        trajLen = median([len(t.xs) for t in self.filtered])
        offs = np.linspace(0,1,trajLen*30)
        xm = []
        ym = []
        for t in self.filtered:
            xs, ys = t.getPoints(offs)            
            xm.append(xs)
            ym.append(ys)
        ym = np.ma.masked_array(ym)
        xm = np.ma.masked_array(xm)
        #self.xp, self.yp = zip(*rdp(list(zip(np.mean(xm, axis=0),np.mean(ym, axis=0))), eps))
        self.xp, self.yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        #tx = truth[self.n][0]
        #ty = truth[self.n][1]
        
        #self.err = disterr(self.xp, self.yp, tx,ty)
        #self.xp, self.yp = np.mean(xm, axis=0),np.mean(ym, axis=0)
    
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
    
    
    def pax(self, ax, lims, eps):
        
        ax.set_xlim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_ylim(0,1)
                
        for _, t in enumerate(self.trajs):    
            ax.plot(t.xs,t.ys, c="b", marker="o", markersize=2)
        
        if len(self.trajs) > 3:    
            for n, t in enumerate(self.lenout):    
                ax.plot(t.xs,t.ys, c="cyan") 
            for n, t in enumerate(self.disout):    
                ax.plot(t.xs,t.ys, c="g")
        if self.xp is not None:
            ax.plot(self.xp,self.yp, marker='D', color='r', linewidth=3)            
        tx = truth[self.n][0]
        ty = truth[self.n][1]
        ax.plot(tx,ty, marker="o",color="black",linewidth=3)
            
        
        if self.xp is not None:
            ax.set_xlabel("#%d err:%.3f eps:%.3f, ns: %d, out: %d" % 
                          (self.n, self.err, eps, len(self.filtered), len(self.lenout)+len(self.disout)))
        #for t in self.trajs:    
        #    ax.plot(t.xs,t.ys, linestyle='--', marker='o', color='grey')
        #if self.xp is not None:
        #    ax.plot(self.xp,self.yp, marker='D', color='r', linewidth=3)
        #ax.set_xlabel(str(self.n))

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


def getResults(d, lenlim, dislim, eps):
    datastring = io.StringIO()
    ts = read_training(d)
    limits = (lenlim, dislim)
    for i in sorted(ts.keys()):
        ss = SampleSet(i,ts[i])
        # .01 is eps for Ramer–Douglas–Peucker algorithm
        ss.getAvg(limits, eps)
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
    dislim = (-8000., .5)
    dislim = (-800., 1.0263)
    lenlim = (-0.5132, .2789)
    eps = .126
    r = getResults(sys.argv[1], lenlim, dislim, eps)
    print(r)

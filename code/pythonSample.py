import os
import sys
from math import sqrt
import numpy as np




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
        self.xd = np.diff(self.xs)
        self.yd = np.diff(self.ys)
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
        
        atleast = 4
        if len(self.filtix) <= atleast:            
            distrank = np.argsort(self.d)
            self.disoutix = distrank[atleast:]
            self.lenoutix = []
            self.filtix = distrank[:atleast]
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
        #self.xp, self.yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        #tx = truth[self.n][0]
        #ty = truth[self.n][1]
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


# function that computes the road segment from a trajectory set
def computeAverageTrajectory(trajectorySet):
    ls = []
    for t in trajectorySet:
        xs = [line["x"] for line in t]
        ys = [line["y"] for line in t]
        ls.append((xs,ys))
    s = SampleSet(0, ls)
    # best params for training set
    # s.getAvg(2.13 ,(-1.23,1.8129), .0755)
    # best params (hopefully) objectively
    s.getAvg(3.0154, (-1.5897,1.9667), .0058)
    ret = []

    for x,y in zip(s.xp, s.yp):
        ret.append({"x": x, "y":y})
    return ret

    
# function reads all the datasets and returns each of them as part of an array
def readAllDatasets(inputDirectory):
    dataSets=[];
    import os;
    for i in range(0,len(os.listdir(inputDirectory))):
        fileName=inputDirectory+"/"+str(i)+".txt";
        if(os.path.isfile(fileName)):
            dataSets.append(readTrajectoryDataset(fileName));
    return dataSets;
    
# reads a set of trajectories from a file
def readTrajectoryDataset(fileName):
    s = open(fileName, 'r').read();
    comp=s.split("\n")
    trajectory=[];
    trajectorySet=[];
    for i in range(0,len(comp)):
        comp[i]=comp[i].split(" ");
        if(len(comp[i])==2):
            # to double??
            point={
                "x":float(comp[i][0]),
                "y":float(comp[i][1])
            }
            trajectory.append(point);
        else:
            trajectorySet.append(trajectory);
            trajectory=[];
    
    return trajectorySet;
    
# function for writing the result to a file
def writeSolution(generatedRoadSegments, outputFile):
    string="";
    for i in range(0,len(generatedRoadSegments)):
        segm=generatedRoadSegments[i];
        for j in range(0,len(segm)):
            string+="{:.7f}".format(segm[j]["x"])+" "+"{:.7f}".format(segm[j]["y"])+"\n";
        string+="\n";
        
    f= open(outputFile,"w+");
    f.write(string);
    f.close(); 
    
# MAIN
inputDirectory="../training_data";
outputFile="solution.txt";

dataSets = readAllDatasets(inputDirectory);

generatedRoadSegments=[];
for i in range(0,len(dataSets)):
    generatedRoadSegments.append(computeAverageTrajectory(dataSets[i]));

writeSolution(generatedRoadSegments, outputFile);

    

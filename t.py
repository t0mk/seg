#!/usr/bin/env python3

import json
import requests
import sys
import seg
import numpy as np


# predicted = [ [{"lat": f, "lon": f}, ..], [], .. ]

def getFile(fn):
    with open(fn) as f:
        return f.read()

def readPredicted(fc):
    r = []
    ss = fc.split("\n\n")
    ls = []
    ns = 0
    for s in ss:
        tts = s.split("\n")
        xs, ys = [],[]
        l = []
        for t in tts:
            if not t.strip():
                continue
            x,y = t.strip().split(" ")
            l.append({"lat": float(x), "lng": float(y)})
        if len(l) > 0:
            r.append(l)
    return r

def test(strPred):
    # returns: SCORE, avg_count
    d = {"request_type": "evaluate", "dataset": "training"}
    d['predicted'] = readPredicted(strPred)

    u = "http://cs.uef.fi/sipu/segments/php/server.php"

    headers={
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "http://cs.uef.fi",
            "Referer": "http://cs.uef.fi/sipu/segments/training.html",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"
    }
    s = json.dumps(d)
    try:
        r = requests.post(u, data={"param": s}, headers=headers)
        j = r.json()
    except:
        print(r)
        sys.exit(0)
    sc =  j["server"]["finalScore"]["csim"]
    co = j["server"]["finalScore"]["av_pred_count"]
    return sc, co

def onerun(d, zmax, lenlim, eps):
    sp = seg.getResults(sys.argv[1], zmax, lenlim, eps)
    sc, co = test(sp)
    print("DM: %.4f, L:%.4f-%.4f E: %.4f, SCORE: %.4f, COUNT: %.4f" %
          (zmax, lenlim[0], lenlim[1], eps, sc, co))


if __name__ == "__main__":
    #arith mean
    #lenlim = (-0.5132, .2789)
    #eps = .126
    #onerun(sys.argv[1], lenlim, dislim, eps)
    #sys.exit(0)
    #dismax = 1.29
    #dismax = 1.4316
    lenlim = (-2, 2)
    for d in np.linspace(.01,.05 ,40): # .04-.08
        #onerun(sys.argv[1],1.7158 , (-1.47,1.87), .0755)
        #onerun(sys.argv[1],1.7158 , (-1.23,1.87), .0755)
        onerun(sys.argv[1], 2.13, (-1.23,1.8129), .0755)




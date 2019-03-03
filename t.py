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

def onerun(d, lenlim, dislim, eps):
    sp = seg.getResults(sys.argv[1], lenlim, dislim, eps)
    sc, co = test(sp)
    print("L: %.4f-%.4f, D: %4.1f-%.4f, E: %.4f, SCORE: %.4f, COUNT: %.4f" %
          (lenlim[0], lenlim[1], dislim[0], dislim[1], eps, sc, co))


if __name__ == "__main__":
    #arith mean
    #dislim = (-800., 1.0263)
    #lenlim = (-0.5132, .2789)
    #eps = .126
    #onerun(sys.argv[1], lenlim, dislim, eps)
    #sys.exit(0)
    for d in np.linspace(.06,.7, 30):
        dislim = (-800., .6724)
        lenlim = (-.5,.52)
        eps = d
        onerun(sys.argv[1], lenlim, dislim, eps)




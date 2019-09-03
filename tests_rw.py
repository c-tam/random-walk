"""
CHARLES TAM
RANDOM WALKS
Tests for random walks

This may take a while to exectue with these parameters
"""

import time
import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
# internal modules
import rw
from rw import walk

def test_var(limit):
    """
    Test variance on distances for a range of walk lengths
    """
    var = np.empty(limit)
    for i in range(1, limit+1):
        # create test walk
        _walk_vtest = walk([0, 0])
        _walk_vtest.random(i)
        var[i-1] = np.var(_walk_vtest.distance()[1])
    return var

def test_msd(limit):
    """
    Test msd for a range of walk lengths
    """
    msd = []
    for i in range(1, limit):
        _walk_mtest = walk([0, 0])
        _walk_mtest.random(i)
        msd = np.append(msd, _walk_mtest.msd())
    return msd

def var_sample(walk_length, tries):
    """
    Try multiple walks to get the one with lowest variance
    """
    # define initial variables
    _walk = walk([0, 0])
    _walk.random(walk_length)
    _var = np.var(_walk.distance()[1])
    _points = _walk.points
    _dist = _walk.dist
    for i in range(tries):
        # candidate variables
        _walk_cand = walk([0, 0])
        _walk_cand.random(walk_length)
        _dist_cand = _walk_cand.distance()
        _var_cand = np.var(_dist_cand[1])
        # update candidates which have smaller variance
        if _var_cand < _var:
            _var = _var_cand
            _points = _walk_cand.points
            _dist = _dist_cand
    return _var, _points, _dist

def main():
    """
    various tests
    """
    # start timer
    start = time.time()

    # make output directory
    if not os.path.exists("output"):
        os.makedirs("output")

    # random walk
    l1 = 250 # length
    w1 = walk([0, 0])
    w1.random(l1)
    w1.distance(filename="output/walk.csv")
    rw.plot_walk(w1.points, title="Random walk, length = %d" % l1, filename="output/walk.pdf")
    rw.plot_hist(w1.dist[1], l1, "output/walkhist.pdf")

    # find random walk with low variance
    print("Finding low variance walk... ", end="", flush=True)
    var_low, walk_low, dist_low = var_sample(l1, 100)
    print("Done")
    rw.plot_walk(walk_low, "Random walk, length = %d" % l1, filename="output/walklow.pdf")
    rw.plot_hist(dist_low[1], l1, "output/walklowhist.pdf")
    # save csv
    data = np.vstack((["steps", "distance"], np.transpose(dist_low)))
    np.savetxt("output/walklowvar.csv", data, delimiter=",", fmt="%s")

    # test how variance scales
    print("Generating data for variance... ", end="", flush=True)
    test_limit_v = 100
    var_walk = test_var(test_limit_v)
    n_v = np.arange(1, test_limit_v+1, 1)
    print("Done")
    var_walk = var_walk[1:]
    n_v = n_v[1:]

    # plot variance
    coef1, covar1 = np.polyfit(np.log10(n_v), np.log10(var_walk), 1, cov=True)
    var_fit = np.poly1d(coef1)(np.log10(n_v))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_v, var_walk, "b,", label="Raw data")
    ax.plot(n_v, 10**var_fit, "r", label="$f = ax+b,$ \n $a=%5.3f,$ \n $b=%5.3f$" % tuple(coef1))
    plt.xscale("log")
    plt.yscale("log")
    ax.legend()
    ax.grid()
    ax.set(xlabel="Walk length", ylabel="Variance",
           title="Variance on distances between points on a random walk")
    plt.savefig("output/variance.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    print("a = {} +/- {}".format(coef1[0], np.sqrt(covar1[0][0])))
    print("b = {} +/- {}".format(coef1[1], np.sqrt(covar1[1][1])))

    # test how msd scales
    test_limit_m = 100
    msd_walk = test_msd(test_limit_m)
    n_m = np.arange(1, test_limit_m, 1)
    # trim first (zero) values
    msd_walk = msd_walk[1:]
    n_m = n_m[1:]

    # plot msd
    coef2, covar2 = np.polyfit(n_m, msd_walk, 1, cov=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_m, msd_walk, "b,", label="Raw data")
    ax.plot(n_m, np.poly1d(coef2)(n_m), "r",
            label="$f = ax+b,$ \n $a=%5.3f,$ \n $b=%5.3f$" % tuple(coef2))
    ax.legend()
    ax.grid()
    ax.set(xlabel="Walk length", ylabel="Mean square displacement",
           title="Mean square displacement of random walks")
    plt.savefig("output/msd.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    print("a = {} +/- {}".format(coef2[0], np.sqrt(covar2[0][0])))
    print("b = {} +/- {}".format(coef2[1], np.sqrt(covar2[1][1])))

    # self avoiding walk
    l2 = 100 # length
    w2 = walk([0, 0])
    p2 = w2.self_avoid(l2)
    rw.plot_walk(p2, title="Self avoiding walk, length = %d" % len(p2), filename="output/saw.pdf")

    # test random walk performance
    length = np.arange(1, 101, 10)
    rw_np = np.empty(len(length))
    rw_loop = np.empty_like(rw_np)
    j = 0
    for i in length:
        rw_np[j] = timeit.timeit("walk([0,0]).random(%d)" %i, "from rw import walk",
                                 number=10)/10
        rw_loop[j] = timeit.timeit("walk([0,0]).random_loop(%d)" %i, "from rw import walk",
                                   number=10)/10
        j += 1

    # plot performance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(length, rw_np, "b", label="Numpy")
    ax.plot(length, rw_loop, "r", label="Loop")
    ax.set(xlabel="Walk length", ylabel="Time/ s", title="Performance of random walk functions")
    ax.legend()
    ax.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("output/rwperf.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    # end timer
    end = time.time()
    print(end - start, "sec to execute")

if __name__ == "__main__":
    main()

"""
CHARLES TAM
RANDOM WALKS
Tests for DLAs

This may take a while to exectue with these parameters
"""

import time
import timeit
import os
import numpy as np
import matplotlib.pyplot as plt
# internal modules
import rw
from rw import dla

def test_gyr(no, multi, prob):
    """
    Get radius of gyration and max radius for a range of walk lengths
    """
    gyr = np.empty(no)
    rad = np.empty_like(gyr)
    n_part = np.empty_like(gyr)
    j = 0
    for i in range(no):
        _gyr, _r_max = rw.gyration([0, 0], dla.spawn_nb(multi*(i+1), 20, 10, prob))
        gyr[i] = _gyr
        rad[i] = _r_max
        n_part[i] = multi*(i+1)
        j += 1
    return gyr, rad, n_part

def main():
    """
    various tests
    """
    # start timer
    start = time.time()

    # make output directory
    if not os.path.exists("output"):
        os.makedirs("output")

    # dla
    length = 500 # length of walk

    # sticking probabilities
    ps1 = 1
    ps2 = 0.3
    ps3 = 0.1

    # generate and plot dlas
    print("Generating data for DLAs... ", end="", flush=True)
    dla1 = dla.spawn_nb(length, 20, 10, ps1)
    dla2 = dla.spawn_nb(length, 20, 10, ps2)
    dla3 = dla.spawn_nb(length, 20, 10, ps3)
    print("Done")
    rw.plot_dla(dla1, ps1, filename="output/dlap1.pdf")
    rw.plot_dla(dla2, ps2, filename="output/dlap03.pdf")
    rw.plot_dla(dla3, ps3, filename="output/dlap01.pdf")

    # box counting dimension
    n_box1, box_1 = rw.box(dla1)
    n_box2, box_2 = rw.box(dla2)
    n_box3, box_3 = rw.box(dla3)

    # fit curves
    coef1, covar1 = np.polyfit(np.log10(1/box_1), np.log10(n_box1), 1, cov=True)
    coef2, covar2 = np.polyfit(np.log10(1/box_2), np.log10(n_box2), 1, cov=True)
    coef3, covar3 = np.polyfit(np.log10(1/box_3), np.log10(n_box3), 1, cov=True)

    # plot box counting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(1/box_1, n_box1, "ko", label=r"$P_{stick} = 1$")
    ax.plot(1/box_2, n_box2, "ks", label=r"$P_{stick} = 0.3$")
    ax.plot(1/box_3, n_box3, "k^", label=r"$P_{stick} = 0.1$")
    ax.plot(1/box_1, 10**np.poly1d(coef1)(np.log10(1/box_1)), "r",
            label="Fitted $P_{stick} = 1$ \n $D=%5.3f$" % coef1[0])
    ax.plot(1/box_2, 10**np.poly1d(coef2)(np.log10(1/box_2)), "g",
            label="Fitted $P_{stick} = 0.3$ \n $D=%5.3f$" % coef2[0])
    ax.plot(1/box_3, 10**np.poly1d(coef3)(np.log10(1/box_3)), "b",
            label="Fitted $P_{stick} = 0.1$ \n $D=%5.3f$" % coef3[0])
    plt.xscale("log")
    plt.yscale("log")
    ax.legend()
    ax.grid()
    ax.set(xlabel=r"Box size$^{-1}$", ylabel="Number of non-empty boxes",
           title=r"Box counting dimension of a DLA with %d particles" % length)
    plt.savefig("output/dlabox.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    # print fitting coefficients
    print("for box counting")
    print("p_stick = 1")
    print("a = {} +/- {}".format(coef1[0], np.sqrt(covar1[0][0])))
    print("b = {} +/- {}".format(coef1[1], np.sqrt(covar1[1][1])))
    print("p_stick = 0.3")
    print("a = {} +/- {}".format(coef2[0], np.sqrt(covar2[0][0])))
    print("b = {} +/- {}".format(coef2[1], np.sqrt(covar2[1][1])))
    print("p_stick = 0.1")
    print("a = {} +/- {}".format(coef3[0], np.sqrt(covar3[0][0])))
    print("b = {} +/- {}".format(coef3[1], np.sqrt(covar3[1][1])))

    # test fractal dimension from radius of gyration and max radius
    no = 10
    multi = 10
    print("Generating data for fractal dimensions... ", end="", flush=True)
    gyr1, rad1, n_part = test_gyr(no, multi, 1) # p = 1
    gyr2, rad2, _ = test_gyr(no, multi, 0.3) # p = 0.3
    gyr3, rad3, _ = test_gyr(no, multi, 0.1) # p = 0.1
    print("Done")

    # fit curves
    coef4, covar4 = np.polyfit(np.log10(rad1), np.log10(n_part), 1, cov=True)
    coef5, covar5 = np.polyfit(np.log10(rad2), np.log10(n_part), 1, cov=True)
    coef6, covar6 = np.polyfit(np.log10(rad3), np.log10(n_part), 1, cov=True)
    coef7, covar7 = np.polyfit(np.log10(gyr1), np.log10(n_part), 1, cov=True)
    coef8, covar8 = np.polyfit(np.log10(gyr2), np.log10(n_part), 1, cov=True)
    coef9, covar9 = np.polyfit(np.log10(gyr3), np.log10(n_part), 1, cov=True)

    # plot radius of gyration
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gyr1, n_part, "ko", label=r"$P_{stick} = 1$")
    ax.plot(gyr2, n_part, "ks", label=r"$P_{stick} = 0.3$")
    ax.plot(gyr3, n_part, "k^", label=r"$P_{stick} = 0.1$")
    ax.plot(gyr1, 10**np.poly1d(coef7)(np.log10(gyr1)), "r",
            label="Fitted $P_{stick} = 1$ \n $D=%5.3f$" % coef7[0])
    ax.plot(gyr2, 10**np.poly1d(coef8)(np.log10(gyr2)), "g",
            label="Fitted $P_{stick} = 0.3$ \n $D=%5.3f$" % coef8[0])
    ax.plot(gyr3, 10**np.poly1d(coef9)(np.log10(gyr3)), "b",
            label="Fitted $P_{stick} = 0.1$ \n $D=%5.3f$" % coef9[0])
    ax.set(xlabel="Radius of gyration", ylabel="Number of particles",
           title=r"Radius of gyration of a DLA")
    plt.xscale("log")
    plt.yscale("log")
    ax.legend()
    ax.grid()
    plt.savefig("output/dlagyr.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    # plot max radius
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rad1, n_part, "ko", label=r"$P_{stick} = 1$")
    ax.plot(rad2, n_part, "ks", label=r"$P_{stick} = 0.3$")
    ax.plot(rad3, n_part, "k^", label=r"$P_{stick} = 0.1$")
    ax.plot(rad1, 10**np.poly1d(coef4)(np.log10(rad1)), "r",
            label="Fitted $P_{stick} = 1$ \n $D=%5.3f$" % coef4[0])
    ax.plot(rad2, 10**np.poly1d(coef5)(np.log10(rad2)), "g",
            label="Fitted $P_{stick} = 0.3$ \n $D=%5.3f$" % coef5[0])
    ax.plot(rad3, 10**np.poly1d(coef6)(np.log10(rad3)), "b",
            label="Fitted $P_{stick} = 0.1$ \n $D=%5.3f$" % coef6[0])
    ax.set(xlabel="Maximum radius", ylabel="Number of particles",
           title=r"Maximum radius of a DLA")
    plt.xscale("log")
    plt.yscale("log")
    ax.legend()
    ax.grid()
    plt.savefig("output/dlarad.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    # print fitting parameters
    print("for radius of gyration")
    print("p_stick = 1")
    print("a = {} +/- {}".format(coef7[0], np.sqrt(covar7[0][0])))
    print("b = {} +/- {}".format(coef7[1], np.sqrt(covar7[1][1])))
    print("p_stick = 0.3")
    print("a = {} +/- {}".format(coef8[0], np.sqrt(covar8[0][0])))
    print("b = {} +/- {}".format(coef8[1], np.sqrt(covar8[1][1])))
    print("p_stick = 0.1")
    print("a = {} +/- {}".format(coef9[0], np.sqrt(covar9[0][0])))
    print("b = {} +/- {}".format(coef9[1], np.sqrt(covar9[1][1])))
    print("")
    print("for max radius")
    print("p_stick = 1")
    print("a = {} +/- {}".format(coef4[0], np.sqrt(covar4[0][0])))
    print("b = {} +/- {}".format(coef4[1], np.sqrt(covar4[1][1])))
    print("p_stick = 0.3")
    print("a = {} +/- {}".format(coef5[0], np.sqrt(covar5[0][0])))
    print("b = {} +/- {}".format(coef5[1], np.sqrt(covar5[1][1])))
    print("p_stick = 0.1")
    print("a = {} +/- {}".format(coef6[0], np.sqrt(covar6[0][0])))
    print("b = {} +/- {}".format(coef6[1], np.sqrt(covar6[1][1])))

    # test dla performance
    print("Testing function performance... ", end="", flush=True)
    particles = np.unique(np.logspace(0, 2, 10, dtype=np.int32))
    _dla_py = np.empty(len(particles))
    _dla_nb = np.empty_like(_dla_py)
    j = 0
    for i in particles:
        _dla_py[j] = timeit.timeit("dla([0,0], 1).spawn(%d, 10, 0)" %i, "from rw import dla",
                                   number=2)/2
        _dla_nb[j] = timeit.timeit("dla.spawn_nb(%d, 10, 0)" %i, "from rw import dla",
                                   number=2)/2
        j += 1
    print("Done")

    # plot performance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(particles, _dla_py, "b", label="Python")
    ax.plot(particles, _dla_nb, "r", label="Numba")
    ax.set(xlabel="Number of particles", ylabel="Time/ s", title="Performance of DLA functions")
    ax.legend()
    ax.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("output/dlaperf.pdf", dpi=100, bbox_inches="tight")
    plt.show()

    # end timer
    end = time.time()
    print(end - start, "sec to execute")

if __name__ == "__main__":
    main()

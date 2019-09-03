"""
CHARLES TAM
RANDOM WALKS

Classes walk and dla
"""
import numpy as np
import numba as nb
from scipy import stats
import matplotlib.pyplot as plt

class walk:
    """
    Random walk
    """
    def __init__(self, start_point):
        """
        Define variables
        """
        self.start_point = np.asarray(start_point)
        self.walk_length = 1
        self.points = None
        self.dist = None

    def random(self, walk_length):
        """
        Find points on a random walk of given length on a square lattice
        """
        walk_length = int(walk_length)
        self.walk_length = walk_length
        # empty variable for points
        points = np.empty((walk_length, 2))
        # populate first column with random numbers
        points[:, 0] = np.random.randint(0, 4, walk_length)
        # map random numbers to a direction
        points[np.where(points[:, 0] == 0)[0]] = [-1, 0]
        points[np.where(points[:, 0] == 1)[0]] = [1, 0]
        points[np.where(points[:, 0] == 2)[0]] = [0, 1]
        points[np.where(points[:, 0] == 3)[0]] = [0, -1]
        # set start point
        points[0] = self.start_point
        # cumulatively sum points
        points_sum = np.cumsum(points, axis=0)
        self.points = points_sum
        return points_sum

    def random_loop(self, walk_length):
        """
        Find points on a random walk of given length using a loop
        Should not use over random()
        """
        self.walk_length = walk_length
        points = np.zeros([walk_length, 2])
        points[0] = self.start_point
        # iterate over walk length
        for i in range(1, walk_length):
            # random direction (right, down, left, up)
            direction = {0:np.array([1, 0]),
                         1:np.array([0, -1]),
                         2:np.array([-1, 0]),
                         3:np.array([0, 1])}.get(np.random.randint(0, 4))
            # append to list
            points[i] = points[i-1] + direction
            self.points = points
        return points

    def random_3d(self, walk_length):
        """
        Find points on a random walk of given length on a simple cubic lattice
        """
        walk_length = int(walk_length)
        # empty variable for points
        points = np.empty((walk_length, 3))
        # populate first column with random numbers
        points[:, 0] = np.random.randint(0, 6, walk_length)
        # map random numbers to a direction
        points[np.where(points[:, 0] == 0)[0]] = [-1, 0, 0]
        points[np.where(points[:, 0] == 1)[0]] = [1, 0, 0]
        points[np.where(points[:, 0] == 2)[0]] = [0, 1, 0]
        points[np.where(points[:, 0] == 3)[0]] = [0, -1, 0]
        points[np.where(points[:, 0] == 4)[0]] = [0, 0, 1]
        points[np.where(points[:, 0] == 5)[0]] = [0, 0, -1]
        # set start point
        points[0] = self.start_point
        # cumulatively sum points
        points_sum = np.cumsum(points, axis=0)
        return points_sum

    def self_avoid(self, walk_length):
        """
        Find points on a self avoiding walk of given length on a square lattice
        using a loop, and terminates walk if it is trapped
        """
        self.walk_length = walk_length
        # empty array for points
        points = [self.start_point]
        dir_dict = {0:[-1, 0], 1:[1, 0], 2:[0, 1], 3:[0, -1]}
        finished = False
        while finished is False:
            # allowed rolls
            ar = [0, 1, 2, 3]
            # delete allowed rolls if curve touches itself within one step
            if any(np.equal(points[:], points[-1] + np.array([-1, 0])).all(1)) is True:
                ar.remove(0)
            if any(np.equal(points[:], points[-1] + np.array([1, 0])).all(1)) is True:
                ar.remove(1)
            if any(np.equal(points[:], points[-1] + np.array([0, 1])).all(1)) is True:
                ar.remove(2)
            if any(np.equal(points[:], points[-1] + np.array([0, -1])).all(1)) is True:
                ar.remove(3)
            # terminate walk if stuck
            if not ar:
                finished = True
            else:
                # roll and map roll to dirction
                direction = dir_dict.get(np.random.choice(ar))
                # append to list of points
                points = np.append(points, [points[-1] + direction], axis=0)
                #terminate loop if desired walk length is reached
                finished = bool(len(points) == walk_length)
        self.points = points
        return points

    def saw_min(self, min_length):
        """
        Find a self avoiding walk with minimum length, and number of attempts
        to do so
        """
        length = False
        attempts = 0
        while length is False:
            saw = self.self_avoid(min_length + 1)
            attempts += 1
            if len(saw) == min_length:
                length = True
        return saw, attempts

    def distance(self, filename=None):
        """
        Find step difference and distance between two points of all points
        in a random walk, using matrices that contain all combination of points
        """
        # arrays of points
        a = np.empty((self.walk_length, self.walk_length, 2))
        a[:] = self.points
        b = np.transpose(a, (1, 0, 2))
        # step difference
        step = np.sum(np.abs(a - b), axis=2).flatten()
        # distance
        dist = np.sqrt(np.sum((a - b)**2, axis=2)).flatten()
        self.dist = np.sort([step, dist])
        # option to save file
        if filename is not None:
            data = np.vstack((["steps", "distance"], np.transpose(self.dist)))
            np.savetxt(filename, data, delimiter=",", fmt="%s")
        return self.dist

    def msd(self):
        """
        Find mean squared displacement of points on a walk
        """
        return np.sum(np.sum((self.start_point - self.points)**2))/self.walk_length

class dla:
    """
    Diffusion limited aggregate
    """
    def __init__(self, agg_start, prob):
        """
        Define intial variables
        """
        self.agg_start = agg_start
        self.points = np.asarray([agg_start])
        self.prob = prob

    def target(self, it, start, bounds):
        """
        Random walk which terminates if reached a desired target, or if it
        leaves the radius of its initial position
        """
        agg = self.points[:it]
        # aggregate expanded by 1 in each direction
        agg_ex = np.unique(np.vstack((agg + [1, 0], agg + [-1, 0],
                                      agg + [0, 1], agg + [0, -1])), axis=0)
        # inital position
        dir_dict = {0:[1, 0], 1:[0, -1], 2:[-1, 0], 3:[0, 1]}
        pos = start
        on_target = False
        while on_target is False:
            # random direction
            direct = np.random.randint(0, 4)
            pos = pos + dir_dict.get(direct)
            dist = np.sqrt(np.sum((pos - self.agg_start)**2))
            # is leaving boundary
            if dist > bounds[1]:
                # restart walk
                pos = start
                on_target = False
            # is leaving boundary
            elif dist < bounds[0]:
                # is point touching aggegate
                if any(np.equal(pos, agg_ex).all(1)):
                    # sticking probability
                    on_target = bool(np.random.choice([True, False],
                                                      p=[self.prob, 1 - self.prob]))
                else:
                    on_target = False
            else:
                on_target = False
        return pos

    def spawn(self, points, d_spawn, d_bound):
        """
        Create start points scattered randomly on a circumference of a circle
        with a given radius, and call target() to get position of stuck particle
        """
        agg = np.empty((points, 2), np.int32)
        agg[0] = self.agg_start
        r_max = 0
        # offset for inner radius of annulus
        d_offset = 2
        for i in range(1, points):
            # random point on circle
            t = 2*np.pi*np.random.random()
            a = r_max + d_spawn
            start = np.array([a*np.cos(t), a*np.sin(t)], dtype=np.int32)
            # get new particle position
            bounds = (r_max + d_offset, a + d_bound)
            agg[i] = self.target(i, start, bounds)
            # update r_max
            r_agg = np.sqrt(np.sum(agg[i]**2))
            if r_agg > r_max:
                r_max = r_agg
            self.points = agg
        return agg

    def spawn_nb(points, d_spawn, d_bound=0, p_stick=1):
        """
        Wrapper function to call Numba functions to create aggregate
        """
        return _spawn(points, d_spawn, d_bound, p_stick)

@nb.jit(nopython=True)
def _get_dir(x):
    """
    Map numbers 0 to 3 to a direction
    """
    if x == 0:
        y = [1, 0]
    if x == 1:
        y = [0, -1]
    if x == 2:
        y = [0, 1]
    if x == 3:
        y = [-1, 0]
    return np.array(y, dtype=np.int32)

@nb.jit(nopython=True)
def _target(start, agg, bounds, p_stick):
    """
    Random walk which terminates if reached a desired target, or if it
    leaves the set boundary
    """
    # inital position
    pos = start
    on_target = False
    while on_target is False:
        # random direction
        direct = _get_dir(np.random.randint(0, 4))
        pos = pos + direct
        pos_abs = np.sqrt(np.sum(pos**2))
        # is point ouside annulus
        if pos_abs > bounds[1]:
            # restart walk
            pos = start
            on_target = False
        # is point inside annulus
        elif pos_abs < bounds[0]:
            # surrounding points
            sur = np.vstack((pos + direct, pos + direct[::-1], pos - direct[::-1]))
            # are surrounding points of particle touching aggregate
            if np.any((agg[:, 0] == sur[0, 0]) & (agg[:, 1] == sur[0, 1])) or \
               np.any((agg[:, 0] == sur[1, 0]) & (agg[:, 1] == sur[1, 1])) or \
               np.any((agg[:, 0] == sur[2, 0]) & (agg[:, 1] == sur[2, 1])):
                # sticking probability
                if p_stick == 1:
                    on_target = True
                elif np.random.random() < p_stick:
                    on_target = True
                else:
                    on_target = False
            else:
                on_target = False
        else:
            on_target = False
    return pos

@nb.jit(nopython=True)
def _spawn(points, d_spawn, d_bound=0, p_stick=1):
    """
    Create start points scattered randomly on a circumference of a circle
    with a given radius, and call target() to get position of stuck particle
    """
    # define initial variables
    agg = np.empty((points, 2), np.int32)
    agg[0] = [0, 0]
    r_max = 0
    # offset for inner radius of annulus
    d_offset = 2
    for i in range(1, points):
        # random point on circle
        t = 2*np.pi*np.random.random()
        a = r_max + d_spawn
        start = np.array([a*np.cos(t), a*np.sin(t)], dtype=np.int32)
        # get new particle position
        bounds = (r_max + d_offset, a + d_bound)
        agg[i] = _target(start, agg[:i], bounds, p_stick)
        # update r_max
        r_agg = np.sqrt(np.sum(agg[i]**2))
        if r_agg > r_max:
            r_max = r_agg
    return agg

def gyration(seed, points, mass=1):
    """
    Find the radius of gyration of the DLA. Assumes the mass of each particle
    is the same, with default value of 1.
    """
    # convert inputs to numpy arrays
    seed = np.asarray(seed)
    points = np.asarray(points)
    # get total mass
    total_mass = len(points)*mass
    # get distances of particles
    distance = np.sum(2*(seed - points)**2)
    r_max = np.amax(np.sqrt(np.sum(points**2, axis=1)))
    return np.sqrt(distance/total_mass), r_max

def box(points):
    """
    Find box counting dimension of fractal
    """
    data = convert_image(points)
    # get shape of data
    shape = np.asarray(data.shape)
    # get ceiling of exponent of data in base 2
    exp = np.ceil(np.log(shape)/np.log(2)).astype(int)
    # pad data with zeros with size equal to 2^exp
    zero_pad = np.zeros(2**exp)
    zero_pad[:data.shape[0], :data.shape[1]] = data
    # create a range of box sizes
    box_size = np.logspace(min(exp), 0, num=min(exp)+1, base=2, dtype="int32")
    n = []
    for i in box_size:
        # split data into i by i boxes
        boxes = zero_pad.reshape(2**exp[0]//i, i, -1, i).swapaxes(1, 2).reshape(-1, i, i)
        # count number of non-zero boxes
        n = np.append(n, sum(np.any(boxes != 0, axis=2).all(1)))
    # get rid of zeros
    idx_nzero = np.where(n != 0)
    n = n[idx_nzero].astype("int32")
    box_size = box_size[idx_nzero]
    return n, box_size

def convert_image(data):
    """
    Convert list of points to matrix of zeros with ones at corresponding points
    """
    # extent of data
    x_extent = int(np.abs(max(data[:, 0]) - min(data[:, 0])))
    y_extent = int(np.abs(max(data[:, 1]) - min(data[:, 1])))
    image = np.zeros((y_extent + 1, x_extent + 1))
    # shift data
    shift = [min(data[:, 0]), min(data[:, 1])]
    data_s = data - shift
    # transform  y coordinate to matrix index j
    data_s[:, 1] = y_extent - data_s[:, 1]
    data_s = data_s.astype(int)
    # set matrix elements to 1 on a point
    image[data_s[:, 1], data_s[:, 0]] = 1
    return image

# plotting functions
def plot_walk(data, title=None, filename=None):
    """
    Plot random walk
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data[:, 0], data[:, 1], "k-")
    ax.set(xlabel="x", ylabel="y", title=title)
    ax.set_aspect("equal", "box")
    # option to save file
    if filename is not None:
        plt.savefig(filename, dpi=400, bbox_inches="tight")
    plt.show()
    return ax

def plot_3d(data, filename=None):
    """
    Plot 3D random walk
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca(projection="3d")
    ax.plot(data[:, 0], data[:, 1], data[:, 2], "k,")
    ax.set(xlabel="x", ylabel="y", zlabel="z",
           title="Random walk, length = %d" %len(data))
    ax.set_aspect("equal", "box")
    if filename is not None:
        plt.savefig(fname=filename, dpi=400, bbox_inches="tight")
    plt.show()
    return ax

def plot_hist(data, length, filename=None):
    """
    Get stats, plot distance data from random walk and fit to pdfs
    """
    # remove zero values
    print("Fitting curves... ", end="", flush=True)
    idx_zero = np.where(data != 0)
    data = data[idx_zero]
    # get stats
    mean = np.mean(data)
    var = np.var(data)
    # histogram plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.histogram_bin_edges(data, int(np.sqrt(len(np.unique(data)))))
    x = np.linspace(0, max(data), 1000)
    ax.hist(data, bins=bins, density=True, edgecolor="white", linewidth=1)

    # fit to various pdfs
    norm1, norm2 = stats.norm.fit(data)
    ray1, ray2 = stats.rayleigh.fit(data)
    lnorm1, lnorm2, lnorm3 = stats.lognorm.fit(data)
    gamma1, gamma2, gamma3 = stats.gamma.fit(data)

    # chi squared test
    chi_n = stats.chisquare(stats.norm.pdf(bins, norm1, norm2), bins)
    chi_r = stats.chisquare(stats.rayleigh.pdf(bins, ray1, ray2), bins)
    chi_l = stats.chisquare(stats.lognorm.pdf(bins, lnorm1, lnorm2, lnorm3), bins)
    chi_g = stats.chisquare(stats.gamma.pdf(bins, gamma1, gamma2, gamma3), bins)
    print("Done")

    # plot pdfs
    ax.plot(x, stats.norm.pdf(x, norm1, norm2), "r",
            label="Normal, \n $\chi^2$ = %5.2f" % chi_n[0])
    ax.plot(x, stats.rayleigh.pdf(x, ray1, ray2), "g",
            label="Rayleigh, \n $\chi^2$ = %5.2f" % chi_r[0])
    ax.plot(x, stats.lognorm.pdf(x, lnorm1, lnorm2, lnorm3), "b",
            label="Log Normal, \n $\chi^2$ = %5.2f" % chi_l[0])
    ax.plot(x, stats.gamma.pdf(x, gamma1, gamma2, gamma3), "k",
            label="Gamma, \n $\chi^2$ = %5.2f" % chi_g[0])
    ax.set(xlabel="Distance", ylabel="Probability",
           title="Distances between points on random walk, length = %d \n $\mu=$%5.3f, $\sigma^2=$%5.3f"
           % (length, mean, var))
    ax.legend()
    # option to save file
    if filename is not None:
        plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.show()
    # print parameters
    print("mean = ", mean)
    print("variance = ", var)
    print("normal chi squared", chi_n[0])
    print("Rayleigh chi squared", chi_r[0])
    print("log normal chi squared", chi_l[0])
    print("gamma chi squared", chi_g[0])
    return ax, mean, var

def plot_dla(data, prob, filename=None):
    """
    Plot diffusion limited aggregate with imshow()
    """
    images = convert_image(data)
    # plot dla
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = (min(data[:, 0]), max(data[:, 0]), min(data[:, 1]), max(data[:, 1]))
    ax.imshow(images, cmap="binary", extent=extent)
    ax.set(xlabel="x", ylabel="y",
           title=r"DLA for %d particles, $P_{stick}$ = %5.2f" % (len(data), prob))
    # option to save file
    if filename is not None:
        plt.savefig(fname=filename, dpi=100, bbox_inches="tight")
    plt.show()
    return ax

def main():
    """
    main
    """
    print("please run one of the test scripts")

if __name__ == "__main__":
    main()

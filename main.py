import os, math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from enum import Enum

def f(x):
    return np.log(x)

def t_life(X, Y):
    return sum([Y[i] * X[i] for i in range(len(X))]) / sum(Y)

def t_0(t_life, k):
    return t_life / (k - 1)

def t_delta(t_0, n=10):
    return t_0 * np.log(n) 


class Values(Enum):
    NTGEN = 0
    TIME = 1


class Graph:
    def __init__(self, path, gtype=Values.NTGEN, k=None):
        self.gtype = gtype
        self.path = os.path.splitext(path)[0]
        with open(path, "r") as file:
            self.X = list()
            self.Y = list()
            for line in file:
                pair = line.split()
                self.X.append(float(pair[0]))

                if self.gtype == Values.NTGEN:
                    self.Y.append(f(float(pair[1])))
                elif self.gtype == Values.TIME:
                    self.Y.append(float(pair[1]))

            if self.gtype == Values.NTGEN:
                self.coeffs = np.polyfit(self.X, self.Y, 1)
                self.Y_fit = np.polyval(self.coeffs, self.X)
                self._k = math.exp(self.coeffs[0])
            elif self.gtype == Values.TIME:
                self._k = k
                self.t_life = t_life(self.X, self.Y)
                self.t_0 = t_0(self.t_life, self._k)
                self.t_delta = t_delta(self.t_0, n=10)

    def plot(self, show=False):
        """
        plot the graph, using matplotlib
        from file with data
        """
        plt.clf()
        plt.plot(self.X, self.Y, 'or')

        if self.gtype == Values.NTGEN:
            plt.plot(self.X, self.Y_fit)
            plt.xlabel("n")
            plt.ylabel("ln(S_n)")
            plt.title("A = " + "%.7f" % self.coeffs[0] + ", k = " + "%.7f" % self._k)
        elif self.gtype == Values.TIME:
            plt.xlabel("T_n_life")
            plt.ylabel("dN_n")
            plt.title("T_life = " + "%.7f" % self.t_life + 
                      ", t_0 = " + "%.7f" % self.t_0 +
                      ", t_delta = " + "%.7f" % self.t_delta)

        self.save()

        if show:
            plt.show()

        return self._k

    def save(self):
        """
        save graph to png picture to provided location
        """
        plt.savefig(self.path + '.png')
    
if __name__ == "__main__":
    # ntgen
    graph1 = Graph(path = "ntgen_LW.dat", gtype=Values.NTGEN)
    graph2 = Graph(path = "ntgen_HW.dat", gtype=Values.NTGEN)
    graph3 = Graph(path = "ntgen_C.dat", gtype=Values.NTGEN)
    k1 = graph1.plot(show=False)
    k2 = graph2.plot(show=False)
    k3 = graph3.plot(show=False)

    # time
    graph4 = Graph(path = "time_n_LW.dat", gtype=Values.TIME, k=k1)
    graph5 = Graph(path = "time_n_HW.dat", gtype=Values.TIME, k=k2)
    graph6 = Graph(path = "time_n_C.dat", gtype=Values.TIME, k=k3)
    graph4.plot(show=False)
    graph5.plot(show=False)
    graph6.plot(show=False)

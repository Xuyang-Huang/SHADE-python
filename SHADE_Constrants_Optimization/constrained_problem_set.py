import numpy as np


class Problem:
    def __init__(self, problem_num):
        assert problem_num != 0, 'Please start from problem 1'
        if problem_num == 4:  # RC04
            self.problem = ReactorNetworkDesign()
        elif problem_num == 11:  # RC11
            self.problem = TwoReactorProblem()
        elif problem_num == 1:  # RC01
            self.problem = HeatExchangerNetworkDesign1()
        elif problem_num == 2:  # RC02
            self.problem = HeatExchangerNetworkDesign2()
        elif problem_num == 5:  # RC05
            self.problem = HaverlysPoolingProblem()
        elif problem_num == 3:  # RC03
            self.problem = OptimalOperationOfAlkylationUnit()
        elif problem_num == 14:  # RC14
            self.problem = MultiProductBatchPlant()
        elif problem_num == 15:  # RC15
            self.problem = WeightMinimizationOfASpeedReducer()
        elif problem_num == 28:  # RC28
            self.problem = RollingElementBearing()
        elif problem_num == 12:  # RC12
            self.problem = ProcessSynthesisProblem()
        elif problem_num == 22:  # RC22
            self.problem = PlanetaryGearTrainDesignOptimizationProblem()

        self.dim = self.problem.dim
        self.h_num = self.problem.h_num
        self.g_num = self.problem.g_num
        self.low_bounds = self.problem.low_bounds
        self.up_bounds = self.problem.up_bounds

    def objective_function(self, x):
        return self.problem.objctive_function(x)

    def constrain_h(self, x):
        return self.problem.constrain_h(x)

    def constrain_g(self, x):
        return self.problem.constrain_g(x)


class ReactorNetworkDesign:  # RC04
    def __init__(self):
        self.dim = 6
        self.h_num = 4
        self.g_num = 1
        self.up_bounds = [1, 1, 1, 1, 16, 16]
        self.low_bounds = [0, 0, 0, 0, 0.00001, 0.00001]
        self.k1 = 0.09755988
        self.k2 = 0.99 * self.k1
        self.k3 = 0.0391908
        self.k4 = 0.9 * self.k3

    def objctive_function(self, x):
        y = - x[:, 3]
        return y

    def constrain_h(self, x):
        h = []
        h.append(self.k1 * x[:, 4] * x[:, 1] + x[:, 0] - 1)
        h.append(self.k3 * x[:, 4] * x[:, 2] + x[:, 2] + x[:, 0] - 1)
        h.append(self.k2 * x[:, 5] * x[:, 1] - x[:, 0] + x[:, 1])
        h.append(self.k4 * x[:, 5] * x[:, 3] + x[:, 1] - x[:, 0] + x[:, 3] - x[:, 2])
        return h

    def constrain_g(self, x):
        g = []
        g.append(x[:, 4] ** 0.5 + x[:, 5] ** 0.5 - 4)
        return g


class TwoReactorProblem:  # RC11
    def __init__(self):
        self.dim = 7
        self.h_num = 4
        self.g_num = 4
        self.up_bounds = [20, 20, 10, 10, 1.49, 1.49, 40]
        self.low_bounds = [0, 0, 0, 0, -0.51, -0.51, 0]

    def objctive_function(self, x):
        x_4 = np.round(x[:, 4])
        x_5 = np.round(x[:, 5])
        y = 7.5 * x_4 + 5.5 * x_5 + 7 * x[:, 2] + 6 * x[:, 3] + 5 * x[:, 6]
        return y

    def constrain_h(self, x):
        x_4 = np.round(x[:, 4])
        x_5 = np.round(x[:, 5])
        z_1 = 0.9 * (1 - np.exp(-0.5 * x[:, 2])) * x[:, 0]
        z_2 = 0.8 * (1 - np.exp(-0.4 * x[:, 3])) * x[:, 1]
        h = []
        h.append(x_4 + x_5 - 1)
        h.append(z_1 + z_2 - 10)
        h.append(x[:, 0] + x[:, 1] - x[:, 6])
        h.append(z_1 * x_4 + z_2 * x_5 - 10)
        return h

    def constrain_g(self, x):
        x_4 = np.round(x[:, 4])
        x_5 = np.round(x[:, 5])
        g = []
        g.append(x[:, 2] - 10 * x_4)
        g.append(x[:, 3] - 10 * x_5)
        g.append(x[:, 0] - 20 * x_4)
        g.append(x[:, 1] - 20 * x_5)
        return g


class HeatExchangerNetworkDesign1:  # RC01
    def __init__(self):
        self.dim = 9
        self.h_num = 8
        self.g_num = 0
        self.up_bounds = [10,200,100,200,2000000,600,600,600,900]
        self.low_bounds = [0,0,0,0,1000,0,100,100,100]

    def objctive_function(self, x):
        y = 35 * x[:, 0] ** 0.6 + 35 * x[:, 1] ** 0.6
        return y

    def constrain_h(self, x):
        h = []
        h.append(200 * x[:, 0] * x[:, 3] - x[:, 2])
        h.append(200 * x[:, 1] * x[:, 5] - x[:, 4])
        h.append(x[:, 2] - 10000 * (x[:, 6] - 100))
        h.append(x[:, 4] - 10000 * (300 - x[:, 6]))
        h.append(x[:, 2] - 10000 * (600 - x[:, 7]))
        h.append(x[:, 4] - 10000 * (900 - x[:, 8]))
        h.append(x[:, 3] * np.log(np.abs(x[:, 7] - 100) + 1e-8) - x[:, 3] * np.log((600 - x[:, 6]) + 1e-8) - x[:, 7] + x[:, 6] + 500)
        h.append(x[:, 5] * np.log(np.abs(x[:, 8] - x[:, 6]) + 1e-8) - x[:, 5] * np.log(600) - x[:, 8] + x[:, 6] + 600)
        return h

    def constrain_g(self, x):
        g = []
        return g


class HaverlysPoolingProblem:  # RC05
    def __init__(self):
        self.dim = 9
        self.h_num = 4
        self.g_num = 2
        self.up_bounds = [100, 200, 100, 100, 100, 100, 200, 100, 200]
        self.low_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def objctive_function(self, x):
        y = -(9*x[:,0]+15*x[:,1]-6*x[:,2]-16*x[:,3]-10*(x[:,4]+x[:,5]))
        return y

    def constrain_h(self, x):
        h = []
        h.append(x[:, 6] + x[:, 7] - x[:, 2] - x[:, 3])
        h.append(x[:, 0] - x[:, 6] - x[:, 4])
        h.append(x[:, 1] - x[:, 7] - x[:, 5])
        h.append(x[:, 8] * x[:, 6] + x[:, 8] * x[:, 7] - 3 * x[:, 2] - x[:, 3])
        return h

    def constrain_g(self, x):
        g = []
        g.append(x[:, 8] * x[:, 6] + 2 * x[:, 4] - 2.5 * x[:, 0])
        g.append(x[:, 8] * x[:, 7] + 2 * x[:, 5] - 1.5 * x[:, 1])
        return g


class HeatExchangerNetworkDesign2:  # RC02
    def __init__(self):
        self.dim = 11
        self.h_num = 9
        self.g_num = 0
        self.up_bounds = [0.819*10**6, 1.131*10**6, 2.05*10**6,0.05074,0.05074,0.05074,200,300,300,300,400]
        self.low_bounds = [10**4,10**4,10**4,0,0,0,100,100,100,100,100]


    def objctive_function(self, x):
        y = (x[:, 0] / (120 * x[:, 3] + 1e-10)) ** 0.6 + (x[:, 1] / (80 * x[:, 4] + 1e-10)) ** 0.6 + (
                    x[:, 2] / (40 * x[:, 5] + 1e-10)) ** 0.6
        return y

    def constrain_h(self, x):
        h = []
        h.append(x[:,0]-1e4*(x[:,6]-100))
        h.append(x[:,1]-1e4*(x[:,7]-x[:,6]))
        h.append(x[:,2]-1e4*(500-x[:,7]))
        h.append(x[:,0]-1e4*(300-x[:,8]))
        h.append(x[:,1]-1e4*(400-x[:,9]))
        h.append(x[:,2]-1e4*(600-x[:,10]))
        h.append(x[:,3]*np.log(np.abs(x[:,8]-100)+1e-8)-x[:,3]*np.log(300-x[:,6]+1e-8)-x[:,8]-x[:,6]+400)
        h.append(x[:,4]*np.log(np.abs(x[:,9]-x[:,6])+1e-8)-x[:,4]*np.log(np.abs(400-x[:,7])+1e-8)-x[:,9]+x[:,6]-x[:,7]+400)
        h.append(x[:,5]*np.log(np.abs(x[:,10]-x[:,7])+1e-8)-x[:,5]*np.log(100)-x[:,10]+x[:,7]+100)
        return h

    def constrain_g(self, x):
        g = []
        return g


class OptimalOperationOfAlkylationUnit:  # RC03
    def __init__(self):
        self.dim = 7
        self.h_num = 0
        self.g_num = 14
        self.up_bounds = [2000,100,4000,100,100,20,200]
        self.low_bounds = [1000,0,2000,0,0,0,0]

    def objctive_function(self, x):
        y = -1.715*x[:,0]-0.035*x[:,0]*x[:,5]-4.0565*x[:,2]-10.0*x[:,1]+0.063*x[:,2]*x[:,4]
        return y

    def constrain_h(self, x):
        h = []
        return h

    def constrain_g(self, x):
        g = []
        g.append(0.0059553571*x[:,5]**2*x[:,0]+0.88392857*x[:,2]-0.1175625*x[:,5]*x[:,0]-x[:,0])
        g.append(1.1088*x[:,0]+0.1303533*x[:,0]*x[:,5]-0.0066033*x[:,0]*x[:,5]**2-x[:,2])
        g.append(6.66173269*x[:,5]**2+172.39878*x[:,4]-56.596669*x[:,3]-191.20592*x[:,5]-10000)
        g.append(1.08702*x[:,5]+0.32175*x[:,3]-0.03762*x[:,5]**2-x[:,4]+56.85075)
        g.append(0.006198*x[:,6]*x[:,3]*x[:,2]+2462.3121*x[:,1]-25.125634*x[:,1]*x[:,3]-x[:,2]*x[:,3])
        g.append(161.18996*x[:,2]*x[:,3]+5000.0*x[:,1]*x[:,3]-489510.0*x[:,1]-x[:,2]*x[:,3]*x[:,6])
        g.append(0.33*x[:,6]-x[:,4]+44.333333)
        g.append(0.022556*x[:,4]-0.007595*x[:,6]-1.0)
        g.append(0.00061*x[:,2]-0.0005*x[:,0]-1.0)
        g.append(0.819672*x[:,0]-x[:,2]+0.819672)
        g.append(24500.0*x[:,1]-250.0*x[:,1]*x[:,3]-x[:,2]*x[:,3])
        g.append(1020.4082*x[:,3]*x[:,1]+1.2244898*x[:,2]*x[:,3]-100000*x[:,1])
        g.append(6.25*x[:,0]*x[:,5]+6.25*x[:,0]-7.625*x[:,2]-100000)
        g.append(1.22*x[:,2]-x[:,5]*x[:,0]-x[:,0]+1.0)
        return g


class MultiProductBatchPlant:  # RC14
    def __init__(self):
        self.dim = 10
        self.h_num = 0
        self.g_num = 10
        self.up_bounds = [3.49,3.49,3.49,2500,2500,2500,20,16,700,450]
        self.low_bounds = [0.51,0.51,0.51,250,250,250,6,4,40,10]
        # constant
        self.S = np.array([[2, 3, 4], [4, 6, 3]])
        self.t = np.array([[8, 20, 8], [16, 4, 4]])
        self.H = 6000
        self.alp = 250
        self.beta = 0.6
        self.Q1 = 40000
        self.Q2 = 20000

    def objctive_function(self, x):
        # decision Variable
        N1 = np.round(x[:, 0])
        N2 = np.round(x[:, 1])
        N3 = np.round(x[:, 2])
        V1 = x[:, 3]
        V2 = x[:, 4]
        V3 = x[:, 5]
        y = self.alp*(N1*V1**self.beta+N2*V2**self.beta+N3*V3**self.beta)
        return y

    def constrain_h(self, x):
        h = []
        return h

    def constrain_g(self, x):
        N1 = np.round(x[:, 0])
        N2 = np.round(x[:, 1])
        N3 = np.round(x[:, 2])
        V1 = x[:, 3]
        V2 = x[:, 4]
        V3 = x[:, 5]
        TL1 = x[:, 6]
        TL2 = x[:, 7]
        B1 = x[:, 8]
        B2 = x[:, 9]
        g = []
        g.append(self.Q1 * TL1 / B1 + self.Q2 * TL2 / B2 - self.H)
        g.append(self.S[0, 0] * B1 + self.S[1, 0] * B2 - V1)
        g.append(self.S[0, 1] * B1 + self.S[1, 1] * B2 - V2)
        g.append(self.S[0, 2] * B1 + self.S[1, 2] * B2 - V3)
        g.append(self.t[0, 0] - N1 * TL1)
        g.append(self.t[0, 1] - N2 * TL1)
        g.append(self.t[0, 2] - N3 * TL1)
        g.append(self.t[1, 0] - N1 * TL2)
        g.append(self.t[1, 1] - N2 * TL2)
        g.append(self.t[1, 2] - N3 * TL2)
        return g


class WeightMinimizationOfASpeedReducer:  # RC15
    def __init__(self):
        self.dim = 7
        self.h_num = 0
        self.g_num = 11
        self.up_bounds = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
        self.low_bounds = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]

    def objctive_function(self, x):
        # decision Variable
        y = 0.7854*x[:,0]*x[:,1]**2*(3.3333*x[:,2]**2+14.9334*x[:,2]-43.0934)-1.508*x[:,0]*(x[:,5]**2+x[:,6]**2)\
            +7.477*(x[:,5]**3+x[:,6]**3)+0.7854*(x[:,3]*x[:,5]**2+x[:,4]*x[:,6]**2)
        return y

    def constrain_h(self, x):
        h = []
        return h

    def constrain_g(self, x):
        g = []
        g.append(-x[:, 0] * x[:, 1] ** 2 * x[:, 2] + 27)
        g.append(-x[:, 0] * x[:, 1] ** 2 * x[:, 2] ** 2 + 397.5)
        g.append(-x[:, 1] * x[:, 5] ** 4 * x[:, 2] * x[:, 3] ** [-3] + 1.93)
        g.append(-x[:, 1] * x[:, 6] ** 4 * x[:, 2] / x[:, 4] ** 3 + 1.93)
        g.append(10 * x[:, 5] ** [-3] * np.sqrt(16.91e6 + (745 * x[:, 3] / (x[:, 1] * x[:, 2])) ** 2) - 1100)
        g.append(10 * x[:, 6] ** [-3] * np.sqrt(157.5e6 + (745 * x[:, 4] / (x[:, 1] * x[:, 2])) ** 2) - 850)
        g.append(x[:, 1] * x[:, 2] - 40)
        g.append(-x[:, 0] / x[:, 1] + 5)
        g.append(x[:, 0] / x[:, 1] - 12)
        g.append(1.5 * x[:, 5] - x[:, 3] + 1.9)
        g.append(1.1 * x[:, 6] - x[:, 4] + 1.9)
        return g


class RollingElementBearing:  # RC28
    def __init__(self):
        self.dim = 10
        self.h_num = 0
        self.g_num = 9
        self.up_bounds = [150,31.5,50.49,0.6,0.6,0.5,0.7,0.4,0.1,0.85]
        self.low_bounds = [125,10.5,4.51,0.515,0.515,0.4,0.6,0.3,0.02,0.6]

    def objctive_function(self, x):
        Dm = x[:, 0]
        Db = x[:, 1]
        Z = np.round(x[:, 2])
        fi = x[:, 3]
        fo = x[:, 4]
        gamma = Db / Dm
        fc = 37.91 * (1 + (1.04 * ((1 - gamma) / (1 + gamma)) ** 1.72 * (fi * (2 * fo - 1) / (fo * (2 * fi - 1))) ** 0.41) ** (10 / 3)) ** (-0.3) * (gamma ** 0.3 * (1 - gamma) ** 1.39 / (1 + gamma) ** (1 / 3)) * (2 * fi / (2 * fi - 1)) ** 0.41
        fc = np.array(fc)
        Z = np.array(Z)
        Db = np.array(Db)
        ind = np.where(Db > 25.4)
        y = fc * Z ** (2 / 3) * Db ** (1.8)
        y[ind] = 3.647 * fc[ind] * Z[ind] ** (2 / 3) * Db[ind] ** 1.4
        return y

    def constrain_h(self, x):
        h = []
        return h

    def constrain_g(self, x):
        Dm = x[:, 0]
        Db = x[:, 1]
        Z = np.round(x[:, 2])
        fi = x[:, 3]
        fo = x[:, 4]
        KDmin = x[:, 5]
        KDmax = x[:, 6]
        eps = x[:, 7]
        e = x[:, 8]
        chi = x[:, 9]
        D = 160
        d = 90
        Bw = 30
        T = D - d - 2 * Db
        phi_o = 2 * np.pi - 2 * np.arccos((((D - d) * 0.5 - 0.75 * T) ** 2 + (0.5 * D - 0.25 * T - Db) ** 2 - (0.5 * d + 0.25 * T) ** 2) / (2 * (0.5 * (D - d) - 0.75 * T) * (0.5 * D - 0.25 * T - Db)))
        Z = np.array(Z)
        Db = np.array(Db)
        g = []
        g.append(Z - 1 - phi_o / (2 * np.arcsin(Db / Dm)))
        g.append(KDmin * (D - d) - 2 * Db)
        g.append(2 * Db - KDmax * (D - d))
        g.append(chi * Bw - Db)
        g.append(0.5 * (D + d) - Dm)
        g.append(Dm - (0.5 + e) * (D + d))
        g.append(eps * Db - 0.5 * (D - Dm - Db))
        g.append(0.515 - fi)
        g.append(0.515 - fo)
        return g


class ProcessSynthesisProblem:  # RC12
    def __init__(self):
        self.dim = 7
        self.h_num = 0
        self.g_num = 9
        self.up_bounds = [100,100,100,1.49,1.49,1.49,1.49]
        self.low_bounds = [0,0,0,-0.51,-0.50,-0.50,-0.50]

    def objctive_function(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        y1 = np.round(x[:,3])
        y2 = np.round(x[:,4])
        y3 = np.round(x[:,5])
        y4 = np.round(x[:,6])
        y = (y1-1)**2 + (y2-1)**2 + (y3-1)**2 - np.log(y4+1+1e-10) + (x1-1)**22 + (x2-2)**2 + (x3-3)**2
        return y

    def constrain_h(self, x):
        h = []
        return h

    def constrain_g(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        y1 = np.round(x[:,3])
        y2 = np.round(x[:,4])
        y3 = np.round(x[:,5])
        y4 = np.round(x[:,6])
        g = []
        g.append(x1 + x2 + x3 + y1 + y2 + y3 - 5)
        g.append(y3**2 + x1**2 + x2**2 + x3**2 - 5.5)
        g.append(x1 + y1 - 1.2)
        g.append(x2 + y2 - 1.8)
        g.append(x3 + y3 - 2.5)
        g.append(x1 + y4 - 1.2)
        g.append(y2**2 + x2**2 - 1.64)
        g.append(y3**2 + x3**2 - 4.25)
        g.append(y2**2 + x3**2 - 4.64)
        return g


class PlanetaryGearTrainDesignOptimizationProblem:  # RC22
    def __init__(self):
        self.dim = 9
        self.h_num = 1
        self.g_num = 10
        self.up_bounds = [96.49,54.49,51.49,46.49,51.49,124.49,3.49,6.49,6.49]
        self.low_bounds = [16.51,13.51,13.51,16.51,13.51,47.51,0.51,0.51,0.51]

    def objctive_function(self, x):
        x = np.round(np.abs(x))
        N1 = x[:, 0]
        N2 = x[:, 1]
        N3 = x[:, 2]
        N4 = x[:, 3]
        N6 = x[:, 5]
        i1 = N6 / N4
        i01 = 3.11
        i2 = N6 * (N1 * N3 + N2 * N4) / ((N1 * N3 * (N6 - N4)))
        i02 = 1.84
        iR = -(N2 * N6 / (N1 * N3))
        i0R = -3.11
        y = np.max([i1-i01,i2-i02,iR-i0R], 0)
        return y

    def constrain_h(self, x):
        Pind = np.array([3, 4, 5])
        N4 = x[:, 3]
        N6 = x[:, 5]
        p = Pind[(x[:, 6]-1).astype(np.int64)].T
        h = []
        h.append(np.remainder(N6-N4, p))
        return h

    def constrain_g(self, x):
        Pind = np.array([3, 4, 5])
        mind = np.array([1.75, 2, 2.25, 2.5, 2.75, 3.0])
        N1 = x[:, 0]
        N2 = x[:, 1]
        N3 = x[:, 2]
        N4 = x[:, 3]
        N5 = x[:, 4]
        N6 = x[:, 5]
        p = Pind[(x[:, 6]-1).astype(np.int64)].T
        m1 = mind[(x[:, 7]-1).astype(np.int64)].T
        m2 = mind[(x[:, 8]-1).astype(np.int64)].T
        Dmax = 220
        dlt22 = 0.5
        dlt33 = 0.5
        dlt55 = 0.5
        dlt35 = 0.5
        dlt34 = 0.5
        dlt56 = 0.5
        beta = np.arccos(((N6 - N3) ** 2 + (N4 + N5) ** 2 - (N3 + N5) ** 2) / (2 * (N6 - N3) * (N4 + N5)))
        g = []
        g.append(m2 * (N6 + 2.5) - Dmax)
        g.append(m1 * (N1 + N2) + m1 * (N2 + 2) - Dmax)
        g.append(m2 * (N4 + N5) + m2 * (N5 + 2) - Dmax)
        g.append(np.abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2)
        g.append(-((N1 + N2) * np.sin(np.pi / p) - N2 - 2 - dlt22))
        g.append(-((N6 - N3) * np.sin(np.pi / p) - N3 - 2 - dlt33))
        g.append(-((N4 + N5) * np.sin(np.pi / p) - N5 - 2 - dlt55))
        if (beta == np.real(beta)).all():
            g.append((N3 + N5 + 2 + dlt35) ** 2 - (
            (N6 - N3) ** 2 + (N4 + N5) ** 2 - 2 * (N6 - N3) * (N4 + N5) * np.cos(2 * np.pi / p - beta)))
        else:
            g.append(1e6)
        g.append(-(N6 - 2 * N3 - N4 - 4 - 2 * dlt34))
        g.append(-(N6 - N4 - 2 * N5 - 4 - 2 * dlt56))
        return g


class RobotGripperProblem:  # RC24
    def __init__(self):
        self.dim = 7
        self.h_num = 0
        self.g_num = 7
        self.up_bounds = [150,150,200,50,150,300,3.14]
        self.low_bounds = [10,10,100,0,10,100,1]

    def objctive_function(self, x):
        a = x[:, 0]
        b = x[:, 1]
        c = x[:, 2]
        e = x[:, 3]
        ff = x[:, 4]
        l = x[:, 5]
        delta = x[:, 6]
        Ymin = 50
        Ymax = 100
        YG = 150
        Zmax = 99.9999
        P = 100
        alpha_0 = np.arccos((a ** 2 + l ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt(l ** 2 + e ** 2))) + np.arctan(e / l)
        beta_0 = np.arccos((b ** 2 + l ** 2 + e ** 2 - a ** 2) / (2 * b * np.sqrt(l ** 2 + e ** 2))) - np.arctan(e / l)
        alpha_m = np.arccos((a ** 2 + (l - Zmax) ** 2 + e ** 2 - b ** 2) / (2 * a * np.sqrt((l - Zmax) ** 2 + e ** 2))) + np.arctan(
            e / (l - Zmax))
        beta_m = np.arccos((b ** 2 + (l - Zmax) ** 2 + e ** 2 - a ** 2) / (2 * b * np.sqrt((l - Zmax) ** 2 + e ** 2))) - np.arctan(
            e / (l - Zmax))

        for i in range(len(x.shape[0])):
            f[i, 1] = -OBJ11[x[i, :], 2] - OBJ11[x[i, :], 1]
        return y

    def constrain_h(self, x):
        h = []
        return h

    def constrain_g(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        y1 = np.round(x[:,3])
        y2 = np.round(x[:,4])
        y3 = np.round(x[:,5])
        y4 = np.round(x[:,6])
        g = []
        g.append(x1 + x2 + x3 + y1 + y2 + y3 - 5)
        g.append(y3**2 + x1**2 + x2**2 + x3**2 - 5.5)
        g.append(x1 + y1 - 1.2)
        g.append(x2 + y2 - 1.8)
        g.append(x3 + y3 - 2.5)
        g.append(x1 + y4 - 1.2)
        g.append(y2**2 + x2**2 - 1.64)
        g.append(y3**2 + x3**2 - 4.25)
        g.append(y2**2 + x3**2 - 4.64)
        return g





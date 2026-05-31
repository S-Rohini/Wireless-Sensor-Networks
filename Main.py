import random
from numpy import matlib
from COA import COA
from DFA import DFA
from Evaluate_Error import evaluat_error
from Global_Vars import Global_Vars
import warnings
from Model_AdaBoost import Model_AdaBoost
from Model_MLP import Model_MLP
from Model_Random_Forest import Model_Random_Forest
from Model_SVM import Model_SVM
from Plot_Results import *
from Proposed import Proposed
from ROA import ROA
from Routing_PlotResults import Routing_PlotResults
from WPA import WPA
from objfun import objfun, obj_fun
warnings.filterwarnings('ignore')


num_of_node = [50, 150, 200]

# Attributes Initialization
an = 0
if an == 1:
    num_of_node = [50, 150, 200]
    p = 0.1
    # Initial Energy
    Eo = 0.3
    # Eelec=Etx=Erx
    ETX = 50 * 1e-09
    ERX = 50 * 1e-09
    # Transmit Amplifier types
    Efs = 10 * 1e-12
    Emp = 0.0013 * 1e-12
    # Data Aggregation Energy
    EDA = 5 * 1e-09
    xm = 100
    ym = 100
    Data = []
    for network in range(len(num_of_node)):
        Total_Packets = 1000
        n = num_of_node[network]
        Packet_loss = np.random.randint(0, 5, size=(n))
        # Percentage of nodes than are advanced
        m = 0.1
        # alpha
        a = 1

        ## Creation of the random Sensor Network
        S_in_xd = np.zeros(n)
        S_in_yd = np.zeros(n)
        XR = np.zeros(n)
        YR = np.zeros(n)
        S_in_G = np.zeros(n)
        S_in_type = []
        S_in_ENERGY = np.zeros(n)
        S_in_E = np.zeros(n)

        for i in range(n):
            S_i_xd = np.random.rand(1) * xm
            S_in_xd[i] = S_i_xd
            XR[i] = S_i_xd
            S_i_yd = (np.random.rand(1, 1) * ym)
            S_in_yd[i] = (S_i_yd)
            YR[i] = (S_i_yd)
            S_i_G = 0

            # initially there are no cluster heads only nodes
            S_i_type = 'N'
            S_in_G[i] = S_i_G
            S_in_type.append(S_i_type)
            temp_rnd0 = i

            # Random Election of Normal Nodes
            if temp_rnd0 >= m * n + 1:
                S_i_E = Eo
                S_i_ENERGY = 0

                S_in_E[i] = S_i_E
                S_in_ENERGY[i] = S_i_ENERGY

            # Random Election of Advanced Nodes
            if temp_rnd0 < m * n + 1:
                S_i_E = Eo * (1 + a)
                S_i_ENERGY = 1

                S_in_E[i] = S_i_E
                S_in_ENERGY[i] = S_i_ENERGY

        Residual_Energy = np.random.uniform(0, 0.018, size=(n, 1))
        Link_Reliability = np.random.uniform(0, 1, size=(n, 1))
        Target = np.concatenate((Residual_Energy, Link_Reliability), axis=1)
        data = [[p] * n, [Eo] * n, [ETX] * n, [ERX] * n, [Efs] * n, [EDA] * n, Packet_loss, [m] * n, [a] * n, S_in_xd,
                XR, S_in_yd, YR, S_in_ENERGY]
        np.save('Data_' + str(network + 1) + '.npy', np.asarray(data).T)
        np.save('Target_' + str(network + 1) + '.npy', Target)

# Optimization for Prediction
an = 0
if an == 1:
    for n in range(len(num_of_node)):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Best_sol = []
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Learning Rate, Activation Function
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = objfun
        max_iter = 50

        print('WPA....')
        [bestfit1, fitness1, bestsol1, Time1] = WPA(initsol, fname, xmin, xmax, max_iter)

        print('DFA....')
        [bestfit2, fitness2, bestsol2, Time2] = DFA(initsol, fname, xmin, xmax, max_iter)

        print('ROA....')
        [bestfit3, fitness3, bestsol3, Time3] = ROA(initsol, fname, xmin, xmax, max_iter)

        print('COA....')
        [bestfit4, fitness4, bestsol4, Time4] = COA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Prediction
an = 0
if an == 1:
    Eval_All = []
    for n in range(len(num_of_node)):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Epochs = [20, 40, 60, 80]
        for act in range(len(Epochs)):
            learnperc = round(Data.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Data[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Data[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 7))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_MLP(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[act],
                                             sol)  # With optimization
            Eval[5, :], pred1 = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epochs[act])
            Eval[6, :], pred2 = Model_AdaBoost(Train_Data, Train_Target, Test_Data,
                                               Test_Target, Epochs[act])
            Eval[7, :], pred3 = Model_Random_Forest(Train_Data, Train_Target, Test_Data, Test_Target, Epochs[act])
            Eval[8, :], pred4 = Model_MLP(Train_Data, Train_Target, Test_Data,
                                          Test_Target, Epochs[act])  # Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_All.append(EVAL)
    np.save('Evaluate_all.npy', Eval_All)  # Save Eval


# Creating an structure
class Structure:
    S_in_G = []
    S_in_type = []
    S_in_E = []
    S_in_ENERGY = []
    S_in_xd = []
    S_in_yd = []


# Optimization for Routing
an = 0
if an == 1:
    mitigation = []
    # Field Dimensions - x and y maximum (in meters)
    xm = 100
    ym = 100
    #  maximum number of rounds
    rmax = 2000
    Global_Vars.rmax = rmax
    #  Optimization paramateres
    no_sol = 10
    dim_sol = 10
    iteration_count = 100
    BestSol_Rout = []
    Fitness_Rout = []
    S_in = Structure()

    # x and y Coordinates of the Sink
    sink_x = 0.5 * xm
    sink_y = 0.5 * ym
    for network in range(len(Global_Vars.num_of_node)):
        # All = Struct(network)
        # Number of Nodes in the field
        n = Global_Vars.num_of_node[network]
        Global_Vars.n = n
        # Optimal Election Probability of a node
        # to become cluster head
        p = 0.1
        Global_inst = Global_Vars()
        Global_Vars.p = p
        # Energy Model (all values in Joules)
        # Initial Energy
        Eo = 0.3
        # Eelec=Etx=Erx
        ETX = 50 * 1e-09
        ERX = 50 * 1e-09
        # Transmit Amplifier types
        Efs = 10 * 1e-12
        Emp = 0.0013 * 1e-12
        # Data Aggregation Energy
        EDA = 5 * 1e-09

        Total_Packets = 1000
        Packet_loss = np.random.randint(0, 5, size=(1, n))

        # Values for Hetereogeneity
        # Percentage of nodes than are advanced
        m = 0.1
        # alpha
        a = 1
        ## Creation of the random Sensor Network
        S_in_xd = np.zeros((n + 1))
        S_in_yd = np.zeros((n + 1))
        XR = np.zeros((n + 1))
        YR = np.zeros((n + 1))
        S_in_G = np.zeros((n + 1))
        S_in_type = []
        S_in_ENERGY = np.zeros((n + 1))
        S_in_E = np.zeros((n + 1))
        for i in range(n + 1):
            S_i_xd = np.random.rand(1) * xm
            S_in_xd[i] = S_i_xd
            XR[i] = S_i_xd
            S_i_yd = (np.random.rand(1, 1) * ym)
            S_in_yd[i] = (S_i_yd)
            YR[i] = (S_i_yd)
            S_i_G = 0

            # initially there are no cluster heads only nodes
            S_i_type = 'N'
            S_in_G[i] = S_i_G
            S_in_type.append(S_i_type)
            temp_rnd0 = i

            # Random Election of Normal Nodes
            if temp_rnd0 >= m * n + 1:
                S_i_E = Eo
                S_i_ENERGY = 0

                S_in_E[i] = S_i_E
                S_in_ENERGY[i] = S_i_ENERGY

            # Random Election of Advanced Nodes
            if temp_rnd0 < m * n + 1:
                S_i_E = Eo * (1 + a)
                S_i_ENERGY = 1

                S_in_E[i] = S_i_E
                S_in_ENERGY[i] = S_i_ENERGY

        S_in_xd[n] = sink_x
        S_in_yd[n] = sink_y
        S_in.S_in_G = S_in_G
        S_in.S_in_type = S_in_type
        S_in.S_in_xd = S_in_xd
        S_in.S_in_yd = S_in_yd
        S_in.S_in_E = S_in_E
        S_in.S_in_ENERGY = S_in_ENERGY
        Global_Vars.S_in = S_in

        n_nodes = Global_Vars.num_of_node[network]
        Npop = 10
        Ch_len = 10
        xmin = np.ones((Npop, Ch_len))
        xmax = np.multiply(n, np.ones((Npop, Ch_len)))
        initsol = np.random.uniform(xmin, xmax)
        max_iter = 100
        fname = obj_fun

        print('WPA....')
        [bestfit1, fitness1, bestsol1, Time1] = WPA(initsol, fname, xmin, xmax, max_iter)

        print('DFA....')
        [bestfit2, fitness2, bestsol2, Time2] = DFA(initsol, fname, xmin, xmax, max_iter)

        print('ROA....')
        [bestfit3, fitness3, bestsol3, Time3] = ROA(initsol, fname, xmin, xmax, max_iter)

        print('COA....')
        [bestfit4, fitness4, bestsol4, Time4] = COA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)

        sols = [bestsol1.ravel(), bestsol2.ravel(), bestsol3.ravel(), bestsol4.ravel(), bestsol5.ravel()]
        fitn = [fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()]
        BestSol_Rout.append(sols)
        Fitness_Rout.append(fitn)
    np.save('BestSol_Rout.npy', np.asarray(BestSol_Rout))
    np.save('Fitness.npy', np.asarray(Fitness_Rout))

# Optmized Routing
an = 0
if an == 1:
    Res = []
    BestSol = np.load('BestSol_Rout.npy', allow_pickle=True)
    for i in range(BestSol.shape[0]):
        n = Global_Vars.n
        S = Global_Vars.S_in
        Source = Global_Vars.Source
        Dest = Global_Vars.Dest
        Packet_loss = Global_Vars.Packet_loss
        Path = np.unique(BestSol)
        short_path = np.array([Source, Path, Dest])
        rmax = Global_Vars.rmax
        for rm in range(rmax):
            r = rm
            v = rm + 1
            # Operation for epoch
            G = []
            cl = []
            p = Global_Vars.p
            if np.mod(r, np.round(1 / p)) == 0:
                for i in range(n + 1):
                    g = 0
                    G.append(g)
                    c = 0
                    cl.append(c)

                S.S_in_G = G
                S.cl = cl
        y2 = [S.S_in_xd, S.S_in_yd, S.S_in_G, S.S_in_type, S.S_in_E, S.S_in_ENERGY, S.cl]
        xd = np.reshape(y2[0], (-1))
        yd = np.reshape(y2[1], (-1))

        # Energy_Consumption
        Ini_energy = 0.2
        Residual_Energy = random.uniform(0, 0.018)
        Energy_Consumption = Ini_energy - Residual_Energy

        # Packet_Delivery_Ratio
        Total_Packets = 1000
        PKT_Loss_Ratio = sum(Packet_loss) / Total_Packets
        packet_delivery_ratio = (Total_Packets - PKT_Loss_Ratio) / Total_Packets
        Trust = Global_Vars.Predcated
        # Throughput
        Throughput = 1 / sum(Packet_loss * 100)
        # Network Lifetime
        node_energies = np.array(S.S_in_ENERGY)  # each node's current energy
        path_nodes = short_path  # nodes on selected path
        consumption_per_packet = Energy_Consumption  # simplification, per-node usage
        node_lifetimes = [node_energies[i] / consumption_per_packet for i in path_nodes]
        Network_Lifetime = min(node_lifetimes)  # bottleneck node decides path lifetime
        k_bits = 1000  # assume 1000-bit packets
        Energy_Efficiency = (packet_delivery_ratio * k_bits) / \
                            (Energy_Consumption * Total_Packets + 1e-9)
        Eval, pred = Model_MLP(Global_Vars.Train_Data, Global_Vars.Train_Target, Global_Vars.Test_Data,
                               Global_Vars.Test_Target, 10)
        Evals = evaluat_error(pred, Global_Vars.Test_Target)
        RMSE = Evals[4]
        # Hop Count
        Hop_Count = len(short_path) - 1  # nodes in path - 1

        # Stability (energy-based)
        node_energies = np.array(S.S_in_ENERGY)
        path_nodes = short_path
        Stability = min(node_energies[i] / max(node_energies) for i in path_nodes)
        fitness = [Network_Lifetime, Energy_Efficiency, Throughput, RMSE]

        np.save('Eenergy_Eff.npy', Energy_Efficiency)
        np.save('Throughput.npy', Throughput)
        np.save('PDR.npy', packet_delivery_ratio)
        np.save('lifetime.npy', Network_Lifetime)
        np.save('Hop_Count.npy', Hop_Count)
        np.save('Stability.npy', Stability)

ConvResults_Pred()
plot_results()
Table()
plotConvResults()
Routing_PlotResults()

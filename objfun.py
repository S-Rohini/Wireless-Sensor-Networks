import random
import numpy as np
from Evaluate_Error import evaluat_error
from Global_Vars import Global_Vars
from Model_MLP import Model_MLP


def objfun(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_MLP(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluat_error(pred, Test_Target)
            Fitn[i] = Eval[5] + Eval[3]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_MLP(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluat_error(pred, Test_Target)
        Fitn = Eval[5] + Eval[3]
        return Fitn


def obj_fun(soln=None):
    n = Global_Vars.n
    S = Global_Vars.S_in
    Source = Global_Vars.Source
    Dest = Global_Vars.Dest
    Packet_loss = Global_Vars.Packet_loss
    Path = np.unique(soln)
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
    Network_Lifetime = np.asarray(min(node_lifetimes))  # bottleneck node decides path lifetime
    k_bits = 1000  # assume 1000-bit packets
    Energy_Efficiency = np.asarray((packet_delivery_ratio * k_bits) / \
                                   (Energy_Consumption * Total_Packets + 1e-9))
    # Hop Count
    Hop_Count = len(short_path) - 1  # nodes in path - 1
    Eval, pred = Model_MLP(Global_Vars.Train_Data, Global_Vars.Train_Target, Global_Vars.Test_Data,
                           Global_Vars.Test_Target, 10)
    Eval = evaluat_error(pred, Global_Vars.Test_Target)
    # Stability (energy-based)
    node_energies = np.array(S.S_in_ENERGY)
    path_nodes = short_path
    Stability = min(node_energies[i] / max(node_energies) for i in path_nodes)

    fitness = (1 / (Network_Lifetime, Energy_Efficiency, Throughput)) + Eval[4]

    return fitness

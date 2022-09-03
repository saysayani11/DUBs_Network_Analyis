import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from collections import Counter

#-- Set Path
os.chdir("/home/sayantoni/Desktop/vidya")

def read_textfile():
    with open('read_list','r') as text:
        textfiles = text.readlines()
        suffix_cif = ".cif"
        pdbids=[]
        for element in textfiles:
            pdbids.append(element.strip())
        pdbids = ["{}{}".format(i,suffix_cif) for i in pdbids]
    return pdbids
cif_files = list(read_textfile())

#--------------------------------------------------------------------
#      BETWEENNESS AND CLOSENESS CENTRALITY ANALYSIS
#--------------------------------------------------------------------


def fetch(file):
   
    atom_type = MMCIF2Dict(file) ['_atom_site.group_PDB']
    atom_site = MMCIF2Dict(file) ['_atom_site.id']
    atom_symbol = MMCIF2Dict(file) ['_atom_site.type_symbol']
    atom_fullsymbol = MMCIF2Dict(file) ['_atom_site.label_atom_id']
    atom_altloc = MMCIF2Dict(file) ['_atom_site.label_alt_id']
    residue_name = MMCIF2Dict(file) ['_atom_site.label_comp_id']
    atom_assym = MMCIF2Dict(file) ['_atom_site.label_asym_id']
    residue_id = MMCIF2Dict(file) ['_atom_site.label_seq_id']
    insertion_code = MMCIF2Dict(file) ['_atom_site.pdbx_PDB_ins_code']
    x_coord = MMCIF2Dict(file) ['_atom_site.Cartn_x']
    y_coord = MMCIF2Dict(file) ['_atom_site.Cartn_y']
    z_coord = MMCIF2Dict(file) ['_atom_site.Cartn_z']
    occupancy = MMCIF2Dict(file) ['_atom_site.occupancy']
    b_factor = MMCIF2Dict(file) ['_atom_site.B_iso_or_equiv']
    model_number = MMCIF2Dict(file) ['_atom_site.pdbx_PDB_model_num']
   
    data_cif = pd.DataFrame(list(zip(atom_type,
                    atom_site,
                    atom_symbol,
                    atom_fullsymbol,
                    atom_altloc,
                    residue_name,
                    atom_assym,
                    residue_id,
                    insertion_code,
                    x_coord,
                    y_coord,
                    z_coord,
                    occupancy,
                    b_factor,
                    model_number)))
   
    #--- remove HETATM, TER
    data_cif = (data_cif[~data_cif[0].isin(['HETATM','TER'])])
   
    #---    remove extra models
    data_cif[14] = data_cif[14].astype(int)
    data_cif = data_cif.drop(data_cif[data_cif[14] >1].index)  
   
    #---   remove altloc
    data_cif = data_cif [~data_cif [4].isin(['B','C','D','E'])]
   
    #---   remove insertion codes
    data_cif = data_cif [~data_cif [8].isin(['B','C','D','E'])]
   
    #--- Pick CA
    pick_CA =  data_cif[data_cif[3] == 'CA']
   
    #--- Pick XYZ coordinates
    xyz_coord = pick_CA[pick_CA.columns[9:12]]
   
    #--- Pick the residue numbers
    res_id =  pick_CA[7]
   
    #--- Pick atomsite IDs  
    atomid = pick_CA[1]
   
    #-- Pick Residues
    res = pick_CA[5]
   
 
   
    temp = pd.concat([res, res_id, xyz_coord, atomid], axis=1)
    return temp
k1 = fetch("5jtvCD.cif")

def _network(file):
    temp = fetch(file)
    res = temp[temp.columns[5]]
    xyz_matrix = temp[temp.columns[2:5]]
    points = np.array(xyz_matrix).astype(float)
    dist_condensed = pdist(points)
   
    prefix = "ATM"
    labels = [prefix + item for item in res]
   
    tuples =[]
    for item in combinations(labels,2):
        tuples.append(item)
       
    source = pd.DataFrame(tuples)[0]
    target = pd.DataFrame(tuples)[1]
       
    edgelist = pd.DataFrame({'source': source,
                              'target': target,
                              'weights': dist_condensed
                            })
   
    cutoff_list = edgelist[edgelist['weights']<8]
     
    #--- set coordinates for network
    chains =  list(set(temp[temp.columns[5]] ))
    count_chain_res = Counter(temp[temp.columns[5]])

    keys = labels
    dict_xyz = (xyz_matrix.astype(float).values.tolist())
    dict_network = dict(list((zip(keys, dict_xyz))))


    #--- Plot the Network
    H = nx.Graph()
    H = nx.from_pandas_edgelist(cutoff_list)
    pos = dict_network
    temp = list(H.edges())

    edge_xyz = [(pos[u], pos[v]) for u, v in H.edges()]
    t = np.asarray(edge_xyz)

    fig = plt.figure(figsize=(50,50))
    ax = fig.add_subplot(222, projection="3d")
       
    xs = np.array(xyz_matrix[9],dtype=float)
    ys = np.array(xyz_matrix[10],dtype=float)
    zs = np.array(xyz_matrix[11],dtype=float)

    ax.scatter(xs,ys,zs, color='black')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.grid(False)
    ax.set_axis_off()
    plt.rcParams['figure.dpi'] = 1200
    for vizedge in t:
        ax.plot(*vizedge.T, linewidth=0.8,color="green")
   
    return cutoff_list
k2 = _network("5jtvCD.cif")

def _BCA(file):
    temp = _network(file)
    H = nx.Graph()
    H = nx.from_pandas_edgelist(temp)
   
    #--- NetworkX normalized betweenness
    bca = nx.betweenness_centrality(H, normalized = True)  
    dpos = list(bca.values())
    temp_net = fetch(file)
    res = temp_net[temp_net.columns[5]]
    xyz_matrix = temp_net[temp_net.columns[2:5]]
    prefix = "ATM"
    labels = [prefix + item for item in res]
    keys = labels
    dict_xyz = (xyz_matrix.astype(float).values.tolist())
    dict_network = dict(list((zip(keys, dict_xyz))))
   
    # --- PLOTTING
    fig = plt.figure(figsize=(20,20))  
    ax = fig.add_subplot(111, projection="3d")      
    xs = np.array(xyz_matrix[9],dtype=float)
    ys = np.array(xyz_matrix[10],dtype=float)
    zs = np.array(xyz_matrix[11],dtype=float)
    t = ax.scatter(xs,ys,zs, s = 70, alpha=1, label='zorder=10', c = dpos,cmap = "Reds", edgecolors='black')  
    # fig,ax = plt.subplots()
    # fig.colorbar(t,ax=ax)
    # ax.remove()
    # plt.savefig('plot_onlycbar.png')
   
    plt.tick_params(labelsize=20)
    ax.set_axis_off()
    # plt.rcParams['figure.dpi'] = 600      
    pos = dict_network
    temp = list(H.edges())
    edge_xyz = [(pos[u], pos[v]) for u, v in H.edges()]
    t = np.asarray(edge_xyz)
    for vizedge in t:
         ax.plot(*vizedge.T, linewidth=0.4,color="black")    
         
    plt.savefig("betweenness.pdf", format="pdf", dpi = 600)
    plt.savefig("betweenness.png", format="png")
    return bca
k3 = _BCA("5jtvCD.cif")

def _CC(file):
    temp = _network(file)
    H = nx.Graph()
    H = nx.from_pandas_edgelist(temp)
   
    #-- NetworkX normalized closeness
    cc = nx.closeness_centrality(H)  
    dpos = list(cc.values())
    temp_net = fetch(file)
    res = temp_net[temp_net.columns[5]]
    xyz_matrix = temp_net[temp_net.columns[2:5]]  
    prefix = "ATM"
    labels = [prefix + item for item in res]
    keys = labels
    dict_xyz = (xyz_matrix.astype(float).values.tolist())
    dict_network = dict(list((zip(keys, dict_xyz))))
   
    # --- PLOTTING
    fig = plt.figure(figsize=(20,20))  
    ax = fig.add_subplot(111, projection="3d")    
    xs = np.array(xyz_matrix[9],dtype=float)
    ys = np.array(xyz_matrix[10],dtype=float)
    zs = np.array(xyz_matrix[11],dtype=float)
    t = ax.scatter(xs,ys,zs, s = 90, c = dpos, cmap = 'seismic')
    fig.colorbar(t, ax=ax)
    ax.set_axis_off()
    plt.rcParams['figure.dpi'] = 600
    plt.title(str(file))          
    pos = dict_network
    temp = list(H.edges())
    edge_xyz = [(pos[u], pos[v]) for u, v in H.edges()]
    t = np.asarray(edge_xyz)
    for vizedge in t:
        ax.plot(*vizedge.T, linewidth=0.4,color="black")
    return cc
k4 = _CC("5jtvCD.cif")

def sort_bet(file):
    bca = _BCA(file)  
    tem1 = pd.DataFrame(bca.items(), columns=['CA_Atoms', 'betweeness_normalized'])
    tem1['residue_name'] = list(fetch(file)[5])
    tem1 = tem1.sort_values(['CA_Atoms'])  
    sorted_betweeness = tem1.sort_values(["betweeness_normalized"], ascending = [False])
    sorted_betweeness['rank'] = list(range(1,len(tem1)+1))
    print(sorted_betweeness)
    filename = str(file) + "_betweeness_data"
    sorted_betweeness.to_excel(filename+".xlsx")
    return sorted_betweeness
k5 = sort_bet("5jtvCD.cif")
   
def sort_cc(file):
    cc = _CC(file)  
    tem1 = pd.DataFrame(cc .items(), columns=['CA_Atoms', 'closeness_normalized'])
    tem1['residue_name'] = list(fetch(file)[5])
    tem1 = tem1.sort_values(['CA_Atoms'])  
    sorted_closeness = tem1.sort_values(["closeness_normalized"], ascending = [False])
    sorted_closeness ['rank'] = list(range(1,len(tem1)+1))
    print(sorted_closeness)
    filename = str(file) + "_closeness_data"
    sorted_closeness.to_excel(filename+".xlsx")
    return sorted_closeness
k6 = sort_cc("5jtvCD.cif")


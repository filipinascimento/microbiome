
from pathlib import Path
import igraph as ig
from tqdm.auto import tqdm
import xnetwork as xn
import pandas as pd
import numpy as np

print("oi")

networkFolder = Path("Networks")
dataFolder = Path("Data")

networks = [
    "expressionAndMicrobiomeCorrelation_CRC0UMAP_Pearson_CLR_Disparity_T0.05",
    "expressionAndMicrobiomeCorrelation_CRC1UMAP_Pearson_CLR_Disparity_T0.05"
]


def calcModularity(g):
    if("community" in g.vertex_attributes()):
        Ci = reindexList(g.vs["community"])
    else:
        return (None,None)
    if("weight" in g.edge_attributes()):
        return None, g.modularity(Ci, weights="weight");
    else:
        return None, g.modularity(Ci, weights=None);



def calcDegree(g):
    results = np.array(g.degree(mode="ALL"))
    return results, np.average(results)


def calcInDegree(g):
    if(not g.is_directed()):
        return (None,None)
    results = np.array(g.indegree())
    return results, np.average(results)

def calcOutDegree(g):
    if(not g.is_directed()):
        return (None,None)
    results = np.array(g.outdegree())
    return results, np.average(results)

def calcStrength(g):
    if("weight" not in g.edge_attributes()):
        return (None,None)
    results = np.array(g.strength(mode="ALL", weights = "weight"))
    return results, np.average(results)

def calcInStrength(g):
    if("weight" not in g.edge_attributes() or not g.is_directed()):
        return (None,None)
    results = np.array(g.strength(mode="IN", weights = "weight"))
    return results, np.average(results)

def calcOutStrength(g):
    if("weight" not in g.edge_attributes() or not g.is_directed()):
        return (None,None)
    results = np.array(g.strength(mode="OUT", weights = "weight"))
    return results, np.average(results)

def calcClusteringCoefficient(g):
    # if("weight" in g.edge_attributes()):
    results = g.transitivity_local_undirected(weights=None)
    # else:
    # 	results = g.transitivity_local_undirected(weights="weight")
    return np.nan_to_num(results,0), np.nanmean(results)

def calcCoreness(g):
    results = np.array(g.coreness(mode="ALL"))
    return results, None

def calcMatchIndex(g):
    degree = np.array(g.degree())
    matchIndex = np.zeros(g.ecount())
    for id,e in enumerate(g.es):
        node1,node2 = e.tuple
        viz1 = g.neighbors(node1)
        viz2 = g.neighbors(node2)
        sharedNei = set(viz1) & set(viz2)
        if ((degree[node1]+degree[node2]) > 2):
            matchIndex[id] = len(sharedNei)/float(degree[node1]+degree[node2]-2)
        else:
            matchIndex[id] = 0
    meanMatchIndex = np.mean(matchIndex)
    return None, meanMatchIndex

def calcBetweenessCentrality(g):
    result = np.array(g.betweenness(directed=g.is_directed()))
    return result,np.average(result)

def calcBetweenessCentralityWeighted(g):
    if("weight" not in g.edge_attributes()):
        return (None,None)
    result = np.array(g.betweenness(weights="weight"))
    return result,np.average(result)

def calcBetweennessCentralization(G):
    vnum = G.vcount()
    if vnum < 3:
        return None,0
    denom = (vnum-1)*(vnum-2)
    temparr = [2*i/denom for i in G.betweenness()]
    max_temparr = max(temparr)
    return None,sum(max_temparr-i for i in temparr)/(vnum-1)

def calcRichClubCoefficient(g, highest=True, scores=None, indices_only=False):
    Trc = richClubPercentage
    degree = np.array(g.degree())
    edges = np.array(g.get_edgelist())
    sourceDegree,targetDegree = degree[edges[:,0]],degree[edges[:,1]]
    dT = int(np.percentile(degree,Trc))
    indNodes = np.nonzero(degree>=dT)[0]
    indEdges = np.nonzero((sourceDegree>=dT)&(targetDegree>=dT))[0]
    if (indNodes.size>1):
        RC = 2.*indEdges.size/(indNodes.size*(indNodes.size-1))
    else:
        RC = 0
    return None,RC

def calcDegreeAssortativity(g):
    return None,g.assortativity_degree(directed=g.is_directed())

def calcDiameter(g):
    if("weight" in g.edge_attributes()):
        return None,g.diameter(directed=g.is_directed(),weights="weight")
    else:
        return None,g.diameter(directed=g.is_directed())

def reindexList(names,returnDict=False):
    d = {ni: indi for indi, ni in enumerate(set(names))}
    numbers = [d[ni] for ni in names]
    if(returnDict):
        return numbers,d
    else:
        return numbers

def getNeighborhoods(g,mode="ALL"):
    if("weight" in g.edge_attributes()):
        return [[(e.target,e["weight"]) if e.target!=i else (e.source,e["weight"]) for e in g.es[g.incident(i,mode=mode)]] for i in range(g.vcount())]
    else:
        return [[(e.target,1) if e.target!=i else (e.source,1) for e in g.es[g.incident(i,mode=mode)]] for i in range(g.vcount())]

def calcModuleDegreeZScore(g,mode="ALL"):
    if("community" in g.vertex_attributes()):
        Ci = reindexList(g.vs["community"])
    else:
        return (None,None)
    neighs = getNeighborhoods(g,mode=mode)
    cneighs = [[(Ci[vertexID],weigth) for vertexID,weigth in neigh] for neigh in neighs]
    kappa = np.zeros(g.vcount())
    kappaSi = [[] for _ in range(max(Ci)+1)]
    
    for i in range(g.vcount()):
        kappa[i] = np.sum([weight for community,weight in cneighs[i] if community==Ci[i]])
        kappaSi[Ci[i]].append(kappa[i])

    avgKappaSi = np.zeros(max(Ci)+1)
    stdKappaSi = np.zeros(max(Ci)+1)

    for ci in range(len(kappaSi)):
        avgKappaSi[ci] = np.average(kappaSi[ci])
        stdKappaSi[ci] = np.std(kappaSi[ci])
    
    zmodule = np.zeros(g.vcount())
    for i in range(g.vcount()):
        ci = Ci[i]
        if(stdKappaSi[ci]>0):
            zmodule[i] = (kappa[i]-avgKappaSi[ci])/stdKappaSi[ci]
    return zmodule,None

def calculateBetweennessAmongCommunity(g,mode="ALL"):
    if("community" in g.vertex_attributes()):
        Ci = reindexList(g.vs["community"])
    else:
        return (None,None)
    uniqueCommunities = set(Ci)
    communityBetweenesses = np.zeros(g.vcount())
    for community in uniqueCommunities:
        communityNodes = np.where(Ci==community)[0]
        subgraph = g.subgraph(communityNodes)
        if("weight" in g.edge_attributes()):
            betweeness = np.array(subgraph.betweenness(weights="weight"))
        else:
            betweeness = np.array(subgraph.betweenness())
        # normalize betweenness by maximum possible value
        betweeness = betweeness/((len(communityNodes)-1)*(len(communityNodes)-2))
        communityBetweenesses[communityNodes] = betweeness
    return communityBetweenesses,None


def calcParticipationCoeff(g,mode="ALL"):
    if("community" in g.vertex_attributes()):
        Ci = reindexList(g.vs["community"])
    else:
        return (None,None)
    neighs = getNeighborhoods(g,mode=mode)
    cneighs = [[(Ci[vertexID],weigth) for vertexID,weigth in neigh] for neigh in neighs]
    
    if("weight" in g.edge_attributes()):
        degrees = np.array(g.strength(mode=mode,weights="weight"))
    else:
        degrees = np.array(g.degree(mode=mode))

    kappasi = np.zeros(g.vcount())
    for i in range(g.vcount()):
        nodeCommunities = set([community for community,weight in cneighs[i]])
        communityDegrees = {community:0 for community in nodeCommunities}
        for community,weight in cneighs[i]:
            communityDegrees[community]+=weight
        kappasi[i] = np.sum(np.power(list(communityDegrees.values()),2))
    
    result = 1.0-kappasi/np.power(degrees,2.0)
    result[degrees==0.0] = 0
    return result,None


for filename in tqdm(networks):
    print(filename)
    networkPath = networkFolder/f"{filename}.xnet"
    g = xn.load(networkPath)
    df = pd.DataFrame()
    # g is igraph object
    # there are two types "M" (Microbiome) and "E" (Expression)
    # We want the genes (E) with highest degree with Microbiome (M)
    # and the microbiome (M) with highest degree with genes (E)
    # We want to consider only conections between E and M
    types = g.vs["Type"]
    # We want to consider only conections between E and M
    connections = g.get_edgelist()
    connections = [c for c in connections if types[c[0]] != types[c[1]]]
    gME = ig.Graph(connections, directed=False)
    # Get the genes with highest degree with microbiome
    df["Label"] = g.vs["Label"]
    df["MEDegrees"] = gME.degree()
    df["Type"] = types
    df["Type_AllTaxa"] = g.vs["Type_AllTaxa"]
    df["Type_Phylum"] = g.vs["Type_Phylum"]
    df["Community"] = g.vs["community"]
    df["Coreness"],_ = calcCoreness(g)
    df["BetweenessCentrality"],_ = calcBetweenessCentrality(g)
    df["ModuleDegreeZScore"],_ = calcModuleDegreeZScore(g)
    df["ParticipationCoeff"],_ = calcParticipationCoeff(g)
    df["BetweenessInCommunity"],_ = calculateBetweennessAmongCommunity(g)
    df.to_csv(dataFolder/f"{filename}_attributes.csv", index=False)
    







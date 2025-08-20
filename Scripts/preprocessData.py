import pandas as pd
import igraph as ig
import xnetwork as xn
import leidenalg as la
from tqdm.auto import tqdm
from mpmath import mp  # pip install mpmath
from scipy import integrate
from functools import partial
import numpy as np
import xnetwork as xn
import umap

mp.dps = 50

useSpearmanCorrelation = False
transformation = "clr"
useDisparityFilter = True
percentiles = [0.01,0.05,0.10,0.25]

def clr_transform(df):
    
    """
    Apply the Centered Log Ratio (CLR) transformation to a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame, where rows are compositions.
    
    Returns:
    - clr_df: DataFrame after applying CLR transformation.
    """
    # replace zeros with 1e-6
    df = df.replace(0, 1e-6)
    # Calculate the geometric mean for each row
    geometric_mean = df.apply(lambda x: np.exp(np.mean(np.log(x[x > 0]))), axis=1)
    
    # Apply the CLR transformation
    clr_df = df.apply(lambda x: np.log(x / geometric_mean), axis=0)
    
    return clr_df

def logit_transform(df):
    """
    Apply the logit transformation to a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame, where each cell is a probability in the interval (0, 1).
    
    Returns:
    - logit_df: DataFrame after applying the logit transformation.
    """
    df = df.replace(0, 1e-6)
    # Apply the logit transformation
    logit_df = df.applymap(lambda x: np.log(x / (1 - x)))
    
    return logit_df


def disparity_filter(g, weights="weight"):
    total_vtx = g.vcount()
    g.es["alpha"] = 1

    for v in range(total_vtx):
        edges = g.incident(v)

        k = len(edges)
        if k > 1:
            sum_w = mp.mpf(sum([g.es[e][weights] for e in edges]))
            for e in edges:
                w = g.es[e][weights]
                p_ij = mp.mpf(w) / sum_w
                alpha_ij = (
                    1
                    - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
                )
                g.es[e]["alpha"] = min(alpha_ij, g.es[e]["alpha"])


def alpha_cut(alpha, g):
    g_copy = g.copy()
    to_delete = g_copy.es.select(alpha_ge=alpha)
    g_copy.delete_edges(to_delete)
    return g_copy


def LeidenModularity(aNetwork, weight=None):
    partition = la.find_partition(
        aNetwork, la.ModularityVertexPartition, weights=weight
    )
    return partition.quality(), partition.membership


def calculateMaxModularity(g, trials=100, weight=None):
    maxModularity = -1
    bestMembership = None
    for _ in range(trials):
        modularity, membership = LeidenModularity(g, weight=weight)
        if modularity > maxModularity:
            maxModularity = modularity
            bestMembership = membership
    return maxModularity, bestMembership


# interpret "na" as not available
df = pd.read_excel("Data/metadata_full_18Nov2014.xlsx")
microbiomeRawDF = pd.read_csv("Data/species_otu_cts_clean.txt", sep="\t")
# create a dictionary of OTU_Name to prevalence
otuNameToPrevalence = {otuName:prevalence for otuName, prevalence in zip(microbiomeRawDF.OTU_Name, microbiomeRawDF.prevalence)}
# drop prevalence column
microbiomeRawDF = microbiomeRawDF.drop(columns=["prevalence"])
microbiomeRawDF.index = microbiomeRawDF.OTU_Name
microbiomeRawDF = microbiomeRawDF.drop(columns=["OTU_Name"])
microbiomeRawDF = microbiomeRawDF.T

# normalize microbiomeDF by the total in each sample
microbiomeRawDF = microbiomeRawDF.div(microbiomeRawDF.sum(axis=1), axis=0)
if transformation == "clr":
    microbiomeRawDF = clr_transform(microbiomeRawDF)
elif transformation == "logit":
    microbiomeRawDF = logit_transform(microbiomeRawDF)

# change intex to int
microbiomeRawDF.index = [int(index) for index in microbiomeRawDF.index]

# create a new version of microbiomeDF for all the df.Patient leave nans if patient not in index
patients = df.Patient.values
microbiomeData = pd.DataFrame(index=np.arange(len(patients)), columns=microbiomeRawDF.columns, dtype=float)
for patientIndex in range(len(patients)):
    patient = patients[patientIndex]
    if patient in microbiomeRawDF.index:
        microbiomeData.loc[patientIndex] = microbiomeRawDF.loc[patient].values
    else:
        microbiomeData.loc[patientIndex] = np.nan

patientClinicalRange = range(1, 28)
patientClinicalData = df.iloc[:, patientClinicalRange].copy()

microbiomeOverallRange = range(28, 32)
microbiomeOverallData = df.iloc[:, microbiomeOverallRange].copy()

expressionRange = range(32, 52)
expressionData = df.iloc[:, expressionRange].copy()

# imputation to replace missing values in the expressions data
expressionData = expressionData.fillna(expressionData.mean())

expressionData.columns = ["E_" + column for column in expressionData.columns]
microbiomeData.columns = ["M_" + column for column in microbiomeData.columns]
patientClinicalData.columns = ["P_" + column for column in patientClinicalData.columns]
microbiomeOverallData.columns = ["MO_" + column for column in microbiomeOverallData.columns]

# Normalize expression by row
# zscore of expression by column
# expressionDataNormalized = (expressionData - expressionData.mean())/expressionData.std()
expressionDataNormalized = expressionData.div(expressionData.sum(axis=1), axis=0)

expressionAndMicrobiomeData = pd.concat([expressionData, microbiomeData], axis=1)

expressionData.index = df.Patient.values
microbiomeData.index = df.Patient.values
patientClinicalData.index = df.Patient.values
microbiomeOverallData.index = df.Patient.values
# rename the columns of each data with prefix: E_, M_, P_, MO_

# merge the data into a single dataframe and save it
mergedData = pd.concat([patientClinicalData, microbiomeOverallData, microbiomeData, expressionData], axis=1)

# save the merged data
mergedData.to_csv("Data/mergedData.csv")

expressionAndMicrobiomeData.index = df.Patient.values
# drop na from microbiomeData and expressionAndMicrobiomeData
originalMicrobiomeData = microbiomeData.copy()
microbiomeData = microbiomeData.dropna()
expressionAndMicrobiomeData = expressionAndMicrobiomeData.dropna()

# calculate correlation among all pairs of expression, microbiome and expressionAndMicrobiomeData
# expressionCorrelationPatients = expressionData.T.corr()
# microbiomeCorrelationPatients = microbiomeData.T.corr()
# expressionAndMicrobiomeCorrelationPatients = expressionAndMicrobiomeData.T.corr()

# expressionCorrelation = expressionData.corr()
# microbiomeCorrelation = microbiomeData.corr()
# expressionAndMicrobiomeCorrelation = expressionAndMicrobiomeData.corr()

# generating network for expressionCorrelationPatients

# PCA plots of patient expressionAndMicrobiomeData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)

def generate_network(
    useAbsoluteCorrelation=False,
    isPatientNetwork=True,
    networkName="expressionCorrelationPatients",
    dataMatrix=None,
    repetitions=10,
    trials=100,
    useDisparity = True,
    percentile = 0.01,
    useSpearman = False,
    useUMAP = False
):
    """
    Generate a network based on correlation data.

    Parameters:
    - useAbsoluteCorrelation (bool): Flag indicating whether to use absolute correlation values. Defaults to False.
    - isPatientNetwork (bool): Flag indicating whether the network represents patient data. Defaults to True.
    - networkName (str): Name of the network. Defaults to "expressionCorrelationPatients".
    - dataMatrix (pd.DataFrame): Data matrix. Defaults to None.
    - repetitions (int): Number of repetitions. Defaults to 10.
    - trials (int): Number of trials. Defaults to 100.
    - thresholdWeight (float): Threshold weight for removing edges. Defaults to 0.6.
    - thresholdBackbone (float): Threshold backbone for alpha cut. Defaults to 1.0.

    Returns:
    - gDisparity (ig.Graph): Generated network.
    """
    # Defaults
    # useAbsoluteCorrelation = False
    # isPatientNetwork = True
    # networkName = "expressionCorrelationPatients"
    # correlationData = expressionCorrelationPatients
    # correlationValues = correlationData.values
    # repetitions = 10
    # trials = 100
    # thresholdWeight = 0.6
    # thresholdBackbone = 1.0

    # Rest of the code...

    # make diagonal elements zero
    if(isPatientNetwork):
        dataMatrix = dataMatrix.T

    if(useUMAP):
        umapData = umap.UMAP(random_state=42).fit_transform(dataMatrix)
        # centralize and multiply by 400
        umapData = (umapData - umapData.mean(axis=0)) * 50
        positions = [tuple(entry) for entry in umapData]
    else: # use PCA
        pca = PCA(n_components=2)
        pcaData = pca.fit_transform(dataMatrix)
        positions = [tuple(entry) for entry in pcaData]




    if(useSpearman):
        correlationData = dataMatrix.corr(method="spearman")
    else:
        correlationData = dataMatrix.corr()

    correlationValues = correlationData.values
    correlationValues[np.diag_indices_from(correlationValues)] = 0
    gOriginal = ig.Graph.Weighted_Adjacency(
        correlationValues, mode="upper", attr="weight"
    )
    gOriginal.vs["ID"] = list(correlationData.columns)
    gOriginal.es["sign"] = np.sign(gOriginal.es["weight"])
    if useAbsoluteCorrelation:
        gOriginal.es["weight"] = np.abs(gOriginal.es["weight"])


    if(useDisparity):
        disparity_filter(gOriginal)
        percentileThreshold = np.quantile(gOriginal.es["alpha"], percentile)
        print("alpha cut: ",percentileThreshold)
        gOriginal.es.select(alpha_gt=percentileThreshold).delete()
    else:
        percentileThreshold = np.quantile(gOriginal.es["weight"], 1.0-percentile)
        # print number of edges before and after
        # edgesBefore = gOriginal.ecount()
        gOriginal.es.select(weight_lt=percentileThreshold).delete()
    # apply disparity filter

    # detect communities
    _, bestMembership = calculateMaxModularity(
        gOriginal, trials=trials, weight="weight"
    )
    gOriginal.vs["community"] = [str(entry) for entry in bestMembership]
    gOriginal.vs["Label"] = [str(entry) for entry in gOriginal.vs["ID"]]
    gOriginal.vs["Position"] = positions
    
    if isPatientNetwork:
        #   add clinical data to the network
        patientIDs = gOriginal.vs["ID"]
        for key in patientClinicalData.keys():
            gOriginal.vs[key] = patientClinicalData[key].loc[patientIDs].values
        # add microbiome data to the network
        for key in microbiomeOverallData.keys():
            gOriginal.vs["Microbiome_" + key] = (
                microbiomeOverallData[key].loc[patientIDs].values
            )
    else:
        # Create Type column based on prefix
        gOriginal.vs["Type"] = [entry.split("_")[0] for entry in gOriginal.vs["ID"]]
        # Create Size attribute 4 if expression and 1 otherwise
        gOriginal.vs["Size"] = [4 if entry.split("_")[0]=="E" else 1 for entry in gOriginal.vs["ID"]]
        # Create Type_Phylum and Type_AllTaxa
        # parse from each Microbiome entry that looks like this M_<Phylum>:<allTaxa>
        # Example:M_F:Anaerococcus_octavius
        # Phylum = F
        # AllTaxa = F:Anaerococcus_octavius
        gOriginal.vs["Type_Phylum"] = ["M_"+entry.split(":")[0].split("_")[1] if entry.split("_")[0]=="M" else entry.split("_")[0] for entry in gOriginal.vs["ID"]]
        gOriginal.vs["Type_AllTaxa"] = ["M_"+entry.split("_")[1] if entry.split("_")[0]=="M" else entry.split("_")[0] for entry in gOriginal.vs["ID"]]

        
    
    # save network
    xn.save(gOriginal, f"Networks/{networkName}.xnet")

expressionCorrelationPatients = expressionData.T.corr()
microbiomeCorrelationPatients = microbiomeData.T.corr()
expressionAndMicrobiomeCorrelationPatients = expressionAndMicrobiomeData.T.corr()

expressionCorrelation = expressionData.corr()
microbiomeCorrelation = microbiomeData.corr()
expressionAndMicrobiomeCorrelation = expressionAndMicrobiomeData.corr()

def filterCRSData(df,crs):
    indices = patientClinicalData[patientClinicalData.P_CRS==crs].index
    # only indices in df.index
    allowedSet = set(df.index)
    indices = [index for index in indices if index in allowedSet]
    return df.loc[indices]

for percentile in percentiles:
    for useUMAP in [True]:
        for useSpearman in [True, False]:
            for useDisparityFilter in [True, False]:
                variantName = "UMAP" if useUMAP else "PCA"
                variantName += "_Spearman" if useSpearman else "_Pearson"
                variantName += "_CLR" if transformation == "clr" else "_Logit" if transformation == "logit" else "_Raw"
                variantName += "_Disparity" if useDisparityFilter else "_Linear"
                variantName += f"_T{percentile}"
                generate_network(
                    useAbsoluteCorrelation=False,
                    isPatientNetwork=True,
                    networkName="expressionCorrelationPatients"+variantName,
                    dataMatrix=expressionData,
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=False,
                    isPatientNetwork=True,
                    networkName="microbiomeCorrelationPatients"+variantName,
                    dataMatrix=microbiomeData,
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=False,
                    isPatientNetwork=True,
                    networkName="expressionAndMicrobiomeCorrelationPatients"+variantName,
                    dataMatrix=expressionAndMicrobiomeData,
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="expressionCorrelation"+variantName,
                    dataMatrix=expressionData,
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="microbiomeCorrelation"+variantName,
                    dataMatrix=microbiomeData,
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )
                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="expressionAndMicrobiomeCorrelation"+variantName,
                    dataMatrix=expressionAndMicrobiomeData,
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )
                


                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="expressionCorrelation_CRC0"+variantName,
                    dataMatrix=filterCRSData(expressionData,0),
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="microbiomeCorrelation_CRC0"+variantName,
                    dataMatrix=filterCRSData(microbiomeData,0),
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )
                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="expressionAndMicrobiomeCorrelation_CRC0"+variantName,
                    dataMatrix=filterCRSData(expressionAndMicrobiomeData,0),
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="expressionCorrelation_CRC1"+variantName,
                    dataMatrix=filterCRSData(expressionData,1),
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )

                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="microbiomeCorrelation_CRC1"+variantName,
                    dataMatrix=filterCRSData(microbiomeData,1),
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )
                generate_network(
                    useAbsoluteCorrelation=True,
                    isPatientNetwork=False,
                    networkName="expressionAndMicrobiomeCorrelation_CRC1"+variantName,
                    dataMatrix=filterCRSData(expressionAndMicrobiomeData,1),
                    repetitions=10,
                    trials=100,
                    useDisparity = useDisparityFilter,
                    percentile = percentile,
                    useSpearman = useSpearman,
                    useUMAP = useUMAP,
                )



generate_network(
    useAbsoluteCorrelation=True,
    isPatientNetwork=False,
    networkName="expressionAndMicrobiomeCorrelation_CRSAll",
    dataMatrix=filterCRSData(expressionAndMicrobiomeData,1),
    repetitions=10,
    trials=100,
    useDisparity = True,
    percentile = 0.05,
    useSpearman = False,
    useUMAP = False,
)
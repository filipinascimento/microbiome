from pathlib import Path
import igraph as ig
from tqdm.auto import tqdm
import xnetwork as xn
import pandas as pd
import RModularity

if __name__ == '__main__':

    filenames = list(Path("Networks").glob("*.xnet"))
    # remove files with Patients in the name
    filenames = [filename for filename in filenames if "Patients" not in filename.stem]
    df = pd.DataFrame(columns=["Filename", "Modularity"])
    for filename in filenames:
        g = xn.load(filename)
        modules = [int(entry) for entry in g.vs["community"]]
        # modules = [int(entry) for entry in g.vs["P_CRS"]]
        modularity = ig.GraphBase.modularity(g, modules)
        majorConnectedComponentSize = max(g.clusters().sizes())/g.vcount()

        Q_rA = RModularity.RModularityFast(
            g.vcount(), # Number of nodes
            g.get_edgelist(), # Edges list
            g.is_directed(), # Directed or not
            )
        print("Q_rA = ", Q_rA)
        Q_diff = RModularity.modularityDifference(
            g.vcount(),
            g.get_edgelist(),
            g.is_directed()
        )
        Q_DL = RModularity.informationModularity(
            g.vcount(),
            g.get_edgelist(),
            g.is_directed()
        )
        print("Q_DL = ", Q_DL)

        df = df.append({
            "Filename": filename.stem,
            "Modularity": modularity,
            "MajorConnectedComponentSize": majorConnectedComponentSize,
            "Q_rA": Q_rA,
            "Q_diff": Q_diff,
            "Q_DL": Q_DL
        }, ignore_index=True)

    # sort by modularity
    df = df.sort_values(by="Modularity", ascending=False)
    # save as csv
    df.to_csv("modularities_genemicrobiome.csv", index=False)


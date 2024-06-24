
# Sensitivity Analysis for OD Algorithms
Jun 2024

Experiments for the paper:
**Impact of the Neighborhood Parameter on Outlier Detection Algorithms**

### PAPER MATERIAL

Results and plots used in the paper are contained in the **material.zip** file.

### FILES IN THIS REPOSITORY

- *material.zip* contains all results as shown in the paper. 
- *dependencies.sh* for installing required Python packages in a clean environment.
- *generate_data.py* creates experimental datasets.
- *outdet.py* runs outlier detection with ABOD, kNN, LOF, LoOP and SDO over the collection of datasets.
- *indices.py* contains functions implementing accuracy indices.
- *explore_results.py* parses results obtained with outlier detection algorithms to create comparison plots and a table with optimal ks.
- *test_kfc.py* rusn KFC tests for finding the optimal k in a collection of datasets. It requires *kfc.py*, which is not included in this repo and must be downloaded from [https://github.com/TimeIsAFriend/KFC/tree/main](https://github.com/TimeIsAFriend/KFC/tree/main). *kfc.py* implements the KFCS and KFCR methods for finding the optimal k as presented in: [paper](https://doi.org/10.1016/j.patrec.2023.08.020).
- *explore_kfc.py* parses results obtained with KFCS and KFCR methods to create latex tables.
- *README.md* this file.

### NOMENCLATURE

The codes to identify datasets and results vary from those shown in the paper. The relationships are shown below:

- **[paper] set of datasets/experiments > [repo]**
- [car] Cardinality > [size]
- [dim] Dimensionality > [dim]
- [gor] Global outliers ratio > [outr]
- [lor] Local outliers ratio > [loc]
- [lay] Layers of density > [mdens]
- [den] Inlier-outlier density ratio > [ddif]
- [zon] Zonification > [clusts]

### REPLICATION OF EXPERIMENTS

#### 0. Clean python installation/environment

1. Create a virtual environment:

        python3 -m venv venv

2. Activate it:

        source venv/bin/activate

3. Install dependencies:

        bash dependencies.sh


#### 1. Generate data

Run:

        python3 generate_data.py datasets plots

It creates the folder **[datasets]** and all datasets in for the experiments. It additionally creates the **[plots]** folder with some selected plots that appear in the paper.


#### 2. Run outlier detection algorithms on data

To run outlier detection algorithms on the generated data:

        python3 outdet.py datasets scores_minmax perf_minmax.csv minmax 1

The **[datasets]** folder must exist. If generated with *Step 1*, it contains datasets in .CSV format (with first row as header and the last column 'y' is the binary label: '1' for outliers). The script will generate the **[scores_minmax]** folder with a file per dataset containing the object-wise outlierness scores outputed by each tested algorithm. It also creates the file **perf_minmax.csv**, with a summary table with the overall performances (various metrics) of all algorithms for all datasets. The **minmax** argument selects the type of normalization applied on the outlierness scores. Argument **1** is just for skipping the first row of the datasets (i.e., the header).

For *proba*-normalization, run:

        python3 outdet.py datasets scores_proba perf_proba.csv gauss 1

Similarly, it will generate scores (**[scores_proba]**) and summaries (**perf_proba.csv**), but for *probability* normalization of scores with the **gauss** argument.

#### 3. Generate comparison plots and the table with optimal ks

Run:

        python3 explore_results.py perf_minmax.csv results

It creates the **[results]** folder with the comparison plots shown in the paper. It also creates a table with the optimal ks according to the evaluation experiments with kNN and ROC-AUC. It is possible to change the algoritm and/or the metric for this table in lines 202 and 203.


#### 3. Obtain KFCS and KFCR estimations

KFCS and KFCR are methods proposed by Yang et al. to estimate the optimal k for outlier detection when using kNN:

        Jiawei Yang, Xu Tan, Sylwan Rahardja, Outlier detection: How to Select k for k-nearest-neighbors-based outlier detectors, 
        Pattern Recognition Letters, Volume 174, 2023, Pages 112-117, ISSN 0167-8655, https://doi.org/10.1016/j.patrec.2023.08.020.

The original paper is accesible [here](https://doi.org/10.1016/j.patrec.2023.08.020). 

To obtain KFCS and KFCR estimates, it is necessary to download the *KFC.py* file from the owners repo: [https://github.com/TimeIsAFriend/KFC/tree/main](https://github.com/TimeIsAFriend/KFC/tree/main). Save it in the same folder as an additional script and rename it as *kfc.py*.

Afterwards, in order to explore datasets and obtain best ks according to KFCS and KFCR, run: 

        python3 test_kfc.py datasets results_kfc.csv 1

It will create the *results_kfc.csv* file. To additionally generate latex summary table with KFCS and KFCR results, run:

        python3 explore_kfc.py results_kfc.csv results

which will output two latex files (*kfc_vals.tex* and *kfc_units.tex*) within the [results] folder.

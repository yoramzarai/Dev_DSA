The files in this folder are the implementation of the DSA framework. The framework contains a data dependednt front-end, and a data independent compute engine.

For example, the front-end for the TCGA data is implemented in TCGA_interface.py, and is based on interface.py. 

Current main notebook: dev_TCGA_survival_analysis.ipynb

Given a new data set (referred to as X here), need to:
1. Create a file called X_interface.py, which imports from interface.py and implements the front-end interface of that dataset (see, for example, TCGA_interface.py for the TCGA dataset)
2. In the main file (currently dev_TCGA_survival_analysis.ipynb, but should change to survival_analysis.ipynb), import X_interface.py and instantiate `Mut`, `Dsa_data`, and `Dsa_compute` from it (see dev_TCGA_survival_analysis.ipynb).

TBD:
- Consider moving DSA_Mutation_ID to DSA_data (i.e., make DSA_Mutation_ID a variable in DSA_data).

# pylint: disable=line-too-long,invalid-name,pointless-string-statement,too-many-arguments,too-many-locals
# pyright: reportReturnType=false
""" 
DSA Interface.

For each dataset, create its interface based on this protocol. 
See, e.g., TCGA_interface.ipynb for implementation of the interface for the GDC-TCGA data.

Notes:
- Consider moving DSA_Mutation_ID to DSA_data.
"""
from typing import Protocol
import pandas as pd


class DSA_Mutation_ID(Protocol):
    """Template for mutation ID."""
    mut_id_cols: list[str]
    mut_id_sep: str

    def get_gene_name(self, mid: str) -> str:
        """Returns the gene name."""

    def get_start_pos(self, mid: str) -> int:
        """Returns the mutation start position"""

    def get_end_pos(self, mid: str) -> int:
        """Retuens the mutation end position"""

    def get_ref_allele(self, mid: str) -> str:
        """Returns the reference allele."""

    def get_var_allele(self, mid: str) -> str:
        """Returns the variant allele."""

    def get_chromosome(self, mid: str) -> str:
        """Returns the chromosome."""

    def mid2dict(self, mid: str) -> dict[str, str]:
        """Converts mutation ID to a dict with mut_id_cols keys."""

    def add_mid_col(self, df: pd.DataFrame, mut_id_col: str = 'mut_id') -> pd.DataFrame:
        """Adds a column to the dataframe with mutation ID values."""

    def get_df_mut(self, df_muts: pd.DataFrame, mid: str) -> pd.DataFrame:
        """Returns a subset of the mutations dataframe corresponding to mutation ID mid."""


class DSA_data(Protocol):
    """Template for mutation and clinical data for DSA."""
    df_clinical_therapies_dict: dict  # key is a therapy, drug, or subsets of drugs/therapies, value is the corresponding clinical dataframe

    cancer_col: str  # cancer column name in both mutation and clinical data
    patient_ID_col: str  # patients ID column name in both mutation and clinical data
    event_name: str = 'event'  # survival event column name in df_clinical_therapies_dict[key]
    duration_name: str = 'duration'  # survival duration column name in df_clinical_therapies_dict[key]
    # keys in df_clinical_therapies_dict
    All_name: str = 'ALL_patients'  # key for clinical data of all patients
    None_therapy_name: str = 'NO_therapy'  # key for clinical data of all patients that did not receive any therapy/drug
    All_therapies_name: str = 'ALL_therapies'  # key for clinical data of all patients that receive any kind of therapy/drug

    # @property
    # def All_name(self):
    #     return self.__All_name

    def get_gene_mutations(self, gene: str) -> pd.DataFrame:
        """Given a gene name, returns its mutation dataframe."""

    def get_filtered_gene_mutations(self, gene: str, filter_data: dict) -> tuple[list, pd.DataFrame, pd.DataFrame]:
        """Given a gene name, returns its filtered mutation list, gene dataframe with filtered mutations, 
        and gene (all) mutation dataframe based on filter_data. filter_data is specific to each dataset interface."""

    def get_clinical_data(self) -> pd.DataFrame:
        """Returns the clinical dataframe."""

    def cancer_name_mutation2clinical(self, cancer: str) -> str:
        """Converts the cancer name taken from the mutation data to the corresponding name in the clinical data."""

    def cancer_name_clinical2mutation(self, cancer: str) -> str:
        """Converts the cancer name taken from the clinical data to the corresponding name in the mutation data."""


class DSA_compute(Protocol):
    """Template for DSA computation."""
    P_survival_label: str = 'p_survival'  # survival p-value label
    P_val_threshold: float = 0.05  # largest p value to be declared as significant

    Enable_sampling_analysis: bool = True
    Sampling_num_rands: int = 500  # number of randomizations
    Sampling_p_val_method: str = 'logrank'   # use 'cox' or 'logrank'
    Sampling_average_method: str = 'median'  # use 'median' or 'mean'

    # Logrank statistical test parameters (see https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html#lifelines.statistics.logrank_test)
    Logrank_weightings: str = '' #None # None or '' disable weightings
    Logrank_p: float = 0.0
    Logrank_q: float = 0.0

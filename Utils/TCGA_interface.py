# pylint: disable=line-too-long,invalid-name,pointless-string-statement,too-many-arguments,too-many-locals
""" 
TCGA DSA interface based on the template interface.

We inherit from the protocols in interface.py, as we need to inheret variables (could have used ABC as well).

TBD:
- Changing the cancer name in the mutation data from TCGA_X to X? See __mutation_rename_cols in TCGA_data.
"""
import pathlib
import pickle
import pandas as pd

import GDC_dataset_feb2021_utils as gdcu
import data_utils as dtut

from Utils.interface import DSA_Mutation_ID,DSA_data,DSA_compute


MID_fields: list[str] = ['Gene_name', 'Chromosome', 'Variant_Classification', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2']
MID_sep: str = ':'
MID_INS_DEL_label: str = '-'  # allele lable used to identify insertion or deletion

class TCGA_Mutation_ID(DSA_Mutation_ID):
    """TCGA mutation ID"""
    mut_id_cols: list[str] = MID_fields
    mut_id_sep: str = MID_sep

    def mid2dict(self, mid: str) -> dict:
        """Returns the dictionary of mutation ID (with mut_id_cols as keys)."""
        return dict(zip(self.mut_id_cols, mid.split(self.mut_id_sep)))

    def get_gene_name(self, mid: str) -> str:
        return self.mid2dict(mid)['Gene_name']

    def get_start_pos(self, mid: str) -> int:
        return int(self.mid2dict(mid)['Start_Position'])

    def get_end_pos(self, mid: str) -> int:
        try:
            return int(self.mid2dict(mid)['End_Position'])
        except KeyError:
            ref, start_pos = self.get_ref_allele(mid), self.get_start_pos(mid)
            return start_pos + len(ref) - 1 if ref != MID_INS_DEL_label else start_pos + 1

    def get_ref_allele(self, mid: str) -> str:
        return self.mid2dict(mid)['Reference_Allele']

    def get_var_allele(self, mid: str) -> str:
        return self.mid2dict(mid)['Tumor_Seq_Allele2']

    def get_chromosome(self, mid: str) -> str:
        return self.mid2dict(mid)['Chromosome']

    def add_mid_col(self, df: pd.DataFrame, mut_id_col: str = 'mut_id') -> pd.DataFrame:
        """Adds a mutation ID column to the dataframe"""
        if not all([x in df.columns for x in self.mut_id_cols]):
            raise ValueError(f"Not all {self.mut_id_cols} are in mutation df columns !!")

        if mut_id_col in df.columns:
            print(f"{mut_id_col} column already in mutation df !!")
            return df

        df[mut_id_col] = df.apply(dtut.mutation_id_from_series, cols=self.mut_id_cols, sep=self.mut_id_sep, axis=1)
        return df

    def get_df_mut(self, df_muts: pd.DataFrame, mid: str) -> pd.DataFrame:
        return dtut.df_with_mut_no_end(df_muts, self.mid2dict(mid))


class TCGA_data(DSA_data):
    """TCGA data. """
    __mutation_folder: pathlib.Path

    __clinical_data: pd.DataFrame
    __df_clinical_drugs_dict_file_name: pathlib.Path

    cancer_col: str = gdcu.Proj_name_col
    patient_ID_col: str = gdcu.Mutation_case_id_col

    def __init__(self, mutation_folder: pathlib.Path, clinical_file: pathlib.Path, df_clinical_drugs_dict_file_name: pathlib.Path):
        self.__mutation_folder = mutation_folder
        self.__clinical_data = pd.read_pickle(clinical_file)
        self.__df_clinical_drugs_dict_file_name = df_clinical_drugs_dict_file_name
        self.df_clinical_therapies_dict = self.__generate_clinical_therapies_dict()

    def __mutation_rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames the patientdf ID and cancer columns in the mutation data."""

        # need also to change TCGA_X to X in the self.cancer_col column, as the cancer name value in the mutation data is TCGA_X
        # (whereas in the clinical data it is X). However, if we do it here, we need to update in get_filtered_gene_mutations
        # where we select only TCGA projects (as then it will contain X and not TCGA_X) and also update the
        # methods cancer_name_mutation2clinical and cancer_name_clinical2mutation. Thus, for now we are keeping
        # it TCGA_X in the mutation data.

        return df.rename(columns={gdcu.Mutation_case_id_col: self.patient_ID_col, gdcu.Proj_name_col: self.cancer_col})

    def __clinical_rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames the patient ID and cancer columns in the clinical data."""
        return df.rename(columns={gdcu.Clinical_case_id_col: self.patient_ID_col, gdcu.Proj_name_col: self.cancer_col})

    def get_gene_mutations(self, gene: str) -> pd.DataFrame:
        """Returns the mutation data of a gene."""
        if (df := gdcu.load_gene_mutation_data(gene, main_path=self.__mutation_folder)) is not None:
            return self.__mutation_rename_cols(df)
        return pd.DataFrame()
    
    def get_filtered_gene_mutations(self, gene: str, filter_data: dict) -> tuple[list, pd.DataFrame, pd.DataFrame]:
        """
        Returns the filtered mutation data based on filter_data.
        
        filter_data: dict. keys: values are:
        - 'mut_id_col_to_add': str, containing the column name to add to the mutation data containing mutation ID.
        - 'filter_silent': bool, True [False] implies enable [disable] filtering out non-silent mutations
        - 'filter_TCGA_projects': bool, True [False] implies enamble [disable] filtering out non-TCGA projects.
        - 'mutation_min_affected_number_of_patients': int, filter out mutations affecting less than this number of patients
        """
        if (df_gene := self.get_gene_mutations(gene)) is None:
            return [], pd.DataFrame(), pd.DataFrame()
        
        if df_gene.empty:
            print(f"Mutation data for {gene=} is empty !!")
            return [], pd.DataFrame(), pd.DataFrame()

        org_df_gene = df_gene.copy()

        # add mutation ID column
        mut_id_col_to_add = filter_data['mut_id_col_to_add']
        assert mut_id_col_to_add not in df_gene.columns, f"{mut_id_col_to_add} column exists in {gene} mutation data !!"
        df_gene[mut_id_col_to_add] = df_gene.apply(dtut.mutation_id_from_series, cols=MID_fields, sep=MID_sep, axis=1)
        
        if filter_data['filter_silent']:
            # select only silent mutations (defined by Silent_muts in Utils/data_utils.py)
            df_gene = df_gene.query(f"Variant_Classification in {dtut.Silent_muts}").reset_index(drop=True).copy()

        if filter_data['filter_TCGA_projects']:
            # select only TCGA projects
            df_gene = df_gene.loc[df_gene[self.cancer_col].str.contains('TCGA_')].reset_index(drop=True).copy()

        # remove mutations with less than mutation_min_affected_number_of_patients patients (using sort_values to retrun a sorted list)
        filtered_muts = df_gene.groupby(by=mut_id_col_to_add).size().to_frame(name='num_p').sort_values(by='num_p', ascending=False).query(f"num_p >= {filter_data['mutation_min_affected_number_of_patients']}").reset_index()[mut_id_col_to_add].tolist()

        return filtered_muts, df_gene.query(f"{mut_id_col_to_add} in {filtered_muts}").copy(), org_df_gene
    
    def get_clinical_data(self) -> pd.DataFrame:
        """Returns the clinical data."""
        return self.__clinical_rename_cols(self.__clinical_data)

    def get_subset_clinical_data(self, patients_set: set):
        """Returns the clinical data of a sub-set of patients."""
        return self.__clinical_rename_cols(self.__clinical_data.query(f"{gdcu.Clinical_case_id_col} in {patients_set}"))
    
    def __generate_clinical_therapies_dict(self) -> dict:
        df_clinical_dict = dtut.prepare_clinical_dfs(self.__clinical_data, gdcu.Therapies, self.event_name, self.duration_name,
                                                    clinical_case_id_col=gdcu.Clinical_case_id_col,
                                                    all_name=self.All_name, none_therapy_name=self.None_therapy_name,
                                                    all_therapies_name=self.All_therapies_name)
        with open(self.__df_clinical_drugs_dict_file_name, 'rb') as fp:
            df_clinical_dict |= pickle.load(fp)

        return {k: self.__clinical_rename_cols(v) for k, v in df_clinical_dict.items()}
    
    def cancer_name_mutation2clinical(self, cancer: str) -> str:
        # if we change the cancer value in the mutation data from TCGA_X to X, then this function should simply return cancer.
        return gdcu.TCGA_proj_full2name(cancer)

    def cancer_name_clinical2mutation(self, cancer: str) -> str:
        # if we change the cancer value in the clinical data from X to TCGA_X, then this function should simply return cancer.
        return gdcu.TCGA_proj_name2full(cancer)


class TCGA_compute(DSA_compute):
    """TCGA DSA compute."""


# pylint: disable=line-too-long,invalid-name,pointless-string-statement,too-many-arguments,too-many-locals
""" 
DSA utils.
"""
import re
import pickle
from typing import Callable
import datetime
import pathlib
from enum import Enum, auto
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

import stats_utils as stut

from Utils.interface import DSA_Mutation_ID, DSA_data, DSA_compute


# Curves' report title and description pages parameters
report_curves_title_page_params: dict = {
    'x_start': 0.5,
    'y_start': 0.5,
    'color': 'blue',
    'fontsize': 20,
    'fontstyle': 'italic', 
    'fontweight': 'bold',
    'verticalalignment': 'center',
    'horizontalalignment': 'center',
}

report_curves_description_pages_params: dict = {
    'x_start': 0.03,
    'y_start': 0.97,
    'color': 'black',
    'fontsize': 12,
    'fontstyle': 'normal', 
    'fontweight': 'normal',
    'verticalalignment': 'top',
    'horizontalalignment': 'left',
}


class Status(Enum):
    """Differential survival (DS) analysis status."""
    Ok = 0  # no errors (currently not used)
    Size_small = auto()  # not enought data (patients) to perform DS analysis
    DS_significant = auto()  # significant DS
    DS_insignificant = auto()  # insignificant DS


@dataclass
class DS_test_group_cnfg:
    """
    Configuration of a DS test (or DS tests), and its groups.

    A single DS test defines two groups, and the DS analysis compares the survaival between the two groups.
    """
    Group_cnfg_typ = tuple[bool, bool, bool]  # the type of a test's group

    class Field_index(Enum):
        """Index in Group_cnfg_typ (mutation indicator, cancer indicator, therapy indicator)."""
        Mut = 0
        Cancer = 1
        Therapy = 2

    @staticmethod
    def group_config2name(group_config: Group_cnfg_typ) -> str:
        """
        Names a test's group configuration.

        Capital letter [small letter] indicates with [without] the corresponding field: M/m for mutation, C/c for cancer, T/t for therapy.
        A small "c" indicates PAN-cancer analysis.

        For example, mCp implies all patients without the (certain) mutation, with a (certain) cancer type and without a (certain) therapy.
        """
        m_t = 'M' if group_config[DS_test_group_cnfg.Field_index.Mut.value] else 'm'
        c_t = 'C' if group_config[DS_test_group_cnfg.Field_index.Cancer.value] else 'c'
        p_t = 'T' if group_config[DS_test_group_cnfg.Field_index.Therapy.value] else 't'
        return f"{m_t}{c_t}{p_t}"

    @staticmethod
    def name2group_config(group_name: str) -> Group_cnfg_typ:
        """
        Converts a group name (e.g. 'MCt') to its configuration (e.g., (True, True, False)).
        """
        return (
            group_name[DS_test_group_cnfg.Field_index.Mut.value] == 'M',
            group_name[DS_test_group_cnfg.Field_index.Cancer.value] == 'C',
            group_name[DS_test_group_cnfg.Field_index.Therapy.value] == 'T'
        )

    @staticmethod
    def DS_name2config(test_name: str, tests_split: str = ',', groups_split: str = '-') -> list:
        """Converts a DS test name (e.g. MCT-mCT,MCt-mCt) to its configuration."""
        return [[DS_test_group_cnfg.name2group_config(x) for x in test.split(groups_split)] for test in test_name.split(tests_split)]
    
    @staticmethod
    def DS_config2name(tests_config: list, tests_split: str = ',', groups_split: str = '-') -> str:
        """
        Converts a DS configuration (e.g. [[(True, True, True), (True, True, False)], [(False, True, True), (False, True, False)]])
        to its name (e.g. MCT-MCt,mCT-mCt).
        """
        return tests_split.join(
            [
                groups_split.join([DS_test_group_cnfg.group_config2name(group_config) for group_config in test_config])
                for test_config in tests_config
            ]
        )

    @staticmethod
    def test_name2indication_name(test_name: str, tests_split: str = ',', groups_split: str = '-') -> str:
        """Converts a test name (e.g. MCT-MCt,mCT-mCt') to an indicator name (e.g. '111-110_011-010')."""
        return tests_split.join(
            [
                groups_split.join([''.join((map(str, map(int, DS_test_group_cnfg.name2group_config(grp)))))  for grp in test.split(groups_split)]) 
                for test in test_name.split(tests_split)
            ]
        )

# this is redundent. Should simply use DS_test_group_cnfg.DS_name2config in the code.
DS_test_name2config: Callable[[str], list] = DS_test_group_cnfg.DS_name2config



def prepare_clinical_for_KM(patients_list: list[str], df_clinical: pd.DataFrame, Dsa_data: DSA_data,
                            convert_days_to_years: bool = True) -> pd.DataFrame:
    """
    Queries df_clinical with patients_list and sorts by duration_col_label.

    Specifically, given a list of patients, and a clinical dataframe that contains the columns event_col_label, duration_col_label, and clinical_patient_id,
    the function returns the clinical data of the patients in patients_list with the columns clinical_patient_id, event_col_label, duration_col_label, 
    sorted by duration_col_label. This is used for Kaplan-Meier survival analysis.

    UPDATE: instead of getting df_clinical (which must be one of the values in Dsa_data.df_clinical_therapies_dict), we get the therapy which is a key 
    in Dsa_data.df_clinical_therapies_dict

    therapy: str - a key in Dsa_data.df_clinical_therapies_dict
    """
    df = df_clinical.query(f"{Dsa_data.patient_ID_col} in {patients_list}")[[Dsa_data.patient_ID_col, Dsa_data.event_name, Dsa_data.duration_name]].sort_values(by=Dsa_data.duration_name).copy().reset_index(drop=True)

    # convert days to years
    if convert_days_to_years:
        for i in df.index:
            df.at[i, Dsa_data.duration_name] /= 365.0
    return df


def find_XX_interesting_results(df_sgnf_r: pd.DataFrame,
                                query_test: str,
                                test_cnfg_name: str = 'test_cnfg_name', cancer: str = 'cancer',
                                mut_id: str = 'mut_id', therapy: str = 'therapy',
                                debug: bool = True) -> pd.DataFrame | None:
    """
    Searches for a significant test (i.e. a row in df_sgnf) with X in both groups, that does not have a matching test (i.e. a different row
    with the same mutation, cancer and therapy, where at least one group don't have X). X can be 'M', 'm', 'C', 'c', 'T', or 't'.
    This is relevant for a DS test that contains multiple tests. 

    query_test: str - one of the 'M', 'm', 'C', 'c', 'T', or 't'.

    Returns a dataframe containing these cases, or None if no such cases exist.
    """
    gather_interesting_tests: list = []
    for (m, c, p), df_mcp in df_sgnf_r.groupby(by=[mut_id, cancer, therapy]):
        # check if df_mcp contains only a single query_test-query_test test
        if (df_mcp.shape[0] == 1) and (len([i.start() for i in re.finditer(query_test, df_mcp.iloc[0][test_cnfg_name])]) == 2):
            if debug:
                print(f"\tFound: {df_mcp.iloc[0][test_cnfg_name]}|{m}|{c}|{p} has no matching different significant test.")
            gather_interesting_tests.append(df_mcp.iloc[0])
    return pd.DataFrame(gather_interesting_tests) if gather_interesting_tests else None


def concat_batch_DS_result_files(batch_indexes: list, base_file_template: str,
                                 index_place_holder_str: str = '?') -> tuple[pd.DataFrame, dict, bool]:
    """
    Concatenates (both the excel and the pickle files) the different batch results of a specific test. 

    The assumption is that the batches contain non-overlapping results for a test_name|mut_id|cancer|therapy case. Note that 
    the batches can have the same gene name key, however the cases of the same gene name in the different batches are 
    non-overlapping. This occurs, e.g., when the batches operate on non-overlapping mutations.

    batch_indexes: list - a list of batch indexes, e.g. [1,2,3,4]
    base_file_template: str - batch file without extension, where the batch index is replaced by index_place_holder_str.
                              For example, str(results_path / f"survival_sgnf_splice_list_10_15_M_P_w_wo_batch_?_4"), where
                              index_place_holder_str = '?'.

    Returns a tuple containing:
    1. the excels concatenated dataframe
    2. the pickles concatenated dictionary
    3. return status: True implies concatenation ok, False implies that the number of test_name|mut_id|cancer|therapy cases 
       in the concatenated dictionary does not match the number of rows in the concatenated dataframe.
    """
    # concatenating the excel files
    concat_df = pd.concat([pd.read_excel(base_file_template.replace(index_place_holder_str, str(x)) + '.xlsx') for x in batch_indexes]).reset_index(drop=True)

    # concatenating the pickle files (each contains a dictionary d, where d[gene_name] is a dictionary with test_name|mut_id|cancer|therapy keys)
    concat_d: dict = {}
    for x in batch_indexes:
        with open(base_file_template.replace(index_place_holder_str, str(x)) + '.pkl', 'rb') as fp:
            cur_d = pickle.load(fp)
            ''' 
            if a key in cur_d already exists in all_d, need to add its content (which is a dictionary) and remove it from cur_d.
            The keys of cur_d are gene names. Since the batches are based on different mutations, a key in cur_d can already exist 
            in all_d. Now, cur_d[gene_name] is a dictionary with the keys of the form test_name|mut_id|cancer|therapy
            (e.g. 'mCT-mCt|PIK3CA:chr3:Intron:179203524:G:C|TCGA_BRCA|chemotherapy'). These can NOT exist in all_d[gene_name]
            (since the batches contain non-overlaping mutations), so we can simply add them to all_d[gene_name].
            '''
            remove_keys: list = []
            for g_name, g_name_dict in cur_d.items():
                if g_name in concat_d:
                    concat_d[g_name] |= g_name_dict  # these are non-overlapping
                    remove_keys.append(g_name)
            for x in remove_keys:
                del cur_d[x]
            concat_d |= cur_d  # now cur_d doesn't have any keys that already exist in concat_d
    # total number of cases in the concatenated dictionary
    a = np.sum([len(v.keys()) for v in concat_d.values()])
    # assert a == concat_df.shape[0], f"the number of test|mut_id|cancer|therapy cases in the concatenated dictionary does not match df_concat shape !!"
    return concat_df, concat_d, a == concat_df.shape[0]


def get_data_of_cnfg_group(group_cnfg: DS_test_group_cnfg.Group_cnfg_typ,
                           patients_with_mut: list[str], patients_without_mut: list[str], Dsa_data: DSA_data, #df_clinical_therapies_dict: dict,
                           cancer: str, therapy: str) -> tuple[str, list, pd.DataFrame]:
    """
    Given a test's group configuration, the function returns a tuple containing:
    1. group configuration name.
    2. group list of patients.
    3. group clinical dataframe.

    Inputs:
    group_cnfg: Group_cnfg_typ -  a group configuration. E.g. (True, True, False)
    patients_with_mut: list - a list of patients (corresponding to patients with a mutation, over all cancer types and all therapies) 
    patients_without_mut: list - a list of patients (corresponding to patients without a mutation, over all cancer types and all therapies)
    """
    # based on mutation
    test_p = patients_with_mut if group_cnfg[DS_test_group_cnfg.Field_index.Mut.value] else patients_without_mut

    # based on cancer
    if group_cnfg[DS_test_group_cnfg.Field_index.Cancer.value]:
        # pick only patients with cancer cancer type
        all_cancer_patients = set(Dsa_data.df_clinical_therapies_dict[Dsa_data.All_name].query(f"{Dsa_data.cancer_col} == '{cancer}'")[Dsa_data.patient_ID_col].unique().tolist())
        #all_cancer_patients = set(df_clinical_therapies_dict[all_name].query(f"{proj_name_col} == '{cancer}'")[clinical_case_id_col].unique().tolist())
        test_p = list(all_cancer_patients.intersection(set(test_p)))
    else:
        # no update to test_p as in this case we are using all patients regardless of cancer type (i.e. PAN-cancer analysis)
        pass

    # based on therapy (which also depends on cancer type)
    cur_clinical_df = Dsa_data.df_clinical_therapies_dict[therapy] if group_cnfg[DS_test_group_cnfg.Field_index.Therapy.value] else Dsa_data.df_clinical_therapies_dict[Dsa_data.None_therapy_name]
    # update based on group_cnfg[DS_test_group_cnfg.Field_index.Cancer.value]
    test_df = cur_clinical_df.query(f"{Dsa_data.cancer_col} == '{cancer}'").query(f"{Dsa_data.patient_ID_col} in {test_p}") if group_cnfg[DS_test_group_cnfg.Field_index.Cancer.value] else cur_clinical_df.query(f"{Dsa_data.patient_ID_col} in {test_p}")
    test_p = test_df[Dsa_data.patient_ID_col].unique().tolist()

    return DS_test_group_cnfg.group_config2name(group_cnfg), test_p, test_df


# def differential_survival_two_groups(patients_group1: list, patients_group2: list, clinical_df_group1: pd.DataFrame, clinical_df_group2: pd.DataFrame,
#                                      duration_col_label: str, event_col_label: str, clinical_case_id_col: str, p_survival_label: str,
#                                      min_group_size: int, p_val_threshold: float, trim_by_last_event: bool,
#                                      logrank_kwargs: dict,
#                                      sampling_p_val_method: str, sampling_average_method: str, sampling_num_rands: int,
#                                      enable_sampling_analysis: bool = True) -> tuple[dict, Enum]:

def differential_survival_two_groups(patients_group1: list, patients_group2: list, clinical_df_group1: pd.DataFrame, clinical_df_group2: pd.DataFrame,
                                     Dsa_data: DSA_data, Dsa_compute: DSA_compute, min_group_size: int, logrank_kwargs: dict) -> tuple[dict, Enum]:
    """
    Performs differential survival analysis on two groups of patients.  
    """
    # filter to patients with clinical data (and with valid survival values), and sort by duration
    df_group1 = prepare_clinical_for_KM(patients_group1, clinical_df_group1, Dsa_data)
    df_group2 = prepare_clinical_for_KM(patients_group2, clinical_df_group2, Dsa_data)

    ret = {'df_group1': df_group1, 'df_group2': df_group2, 'Group1_size': df_group1.shape[0], 'Group2_size': df_group2.shape[0]}

    if df_group1.shape[0] < min_group_size or df_group2.shape[0] < min_group_size:
        return ret, Status.Size_small

    # KM
    Km_group1 = stut.Kaplan_Meier_survival(df_group1, Dsa_data.duration_name, Dsa_data.event_name, p_survival_str=Dsa_compute.P_survival_label, trim_by_last_event=False)
    Km_group2 = stut.Kaplan_Meier_survival(df_group2, Dsa_data.duration_name, Dsa_data.event_name, p_survival_str=Dsa_compute.P_survival_label, trim_by_last_event=False)

    # compute p values
    cox_p_value, _ =   stut.cox_regression_p_value(Km_group1.df_km, Km_group2.df_km, duration_col_label=Dsa_data.duration_name, event_col_label=Dsa_data.event_name)
    logrank_p_value, _ = stut.logrank_test_p_value(Km_group1.df_km, Km_group2.df_km, duration_col_label=Dsa_data.duration_name, event_col_label=Dsa_data.event_name, args=logrank_kwargs)

    ret |= {
        'Km_group1': Km_group1,
        'Km_group2': Km_group2,
        'cox_p_value': cox_p_value,
        'logrank_p_value': logrank_p_value,
        'AUC1': Km_group1.auc,
        'AUC2': Km_group2.auc,
    }

    status = Status.DS_insignificant
    s_pv, s_auc = -1.0, -1.0
    # in case of error or non-conversion, the cox / logrank test returns -1 as the p value. Thus we also check for 0 <=
    if (0 <= cox_p_value <= Dsa_compute.P_val_threshold) or (0 <= logrank_p_value <= Dsa_compute.P_val_threshold):
        status = Status.DS_significant
        if Dsa_compute.Enable_sampling_analysis:
            # consider adding a check if both groups are large "enough", in which case
            # probably no need to perform sampling.

            # the function samples from the second input a size of the first input
            in1, in2 = (df_group1, df_group2) if df_group1.shape[0] <= df_group2.shape[0] else (df_group2, df_group1)
            s_pv, s_auc, _, _ = stut.sample_survivals_pval(
                in1, in2, Dsa_compute.Sampling_num_rands, Dsa_data.duration_name, Dsa_data.event_name, Dsa_compute.P_survival_label,
                p_val_method=Dsa_compute.Sampling_p_val_method, average_method=Dsa_compute.Sampling_average_method, trim_by_last_event=False,
                logrank_args=logrank_kwargs)

    return ret | {'Sample_p_value': s_pv, 'Sample_AUC': s_auc}, status


# def process_gene_muts_survival_tests(tests_cnfg: list, df_gene: pd.DataFrame, gene_muts: list[str], df_clinical_therapies_dict: dict,
#                                      ignore_therapies: set, ignore_cancers: set,
#                                      duration_col_label: str, event_col_label: str,
#                                      proj_name_col: str, mutation_case_id_col: str, clinical_case_id_col: str, p_survival_label: str,
#                                      min_group_size: int, p_val_threshold: float, trim_by_last_event: bool,
#                                      logrank_kwargs: dict,
#                                      sampling_p_val_method: str, sampling_average_method: str, sampling_num_rands: int,
#                                      mut_id_cols: list[str], mut_id_sep: str,
#                                      debug_level: int = 0, enable_sampling_analysis: bool = True, main_test_index: int = 1,
#                                      all_name: str = 'ALL_patients', none_therapy_name: str = 'NO_therapy') -> dict:
    
def process_gene_muts_survival_tests(tests_cnfg: list, df_gene: pd.DataFrame, gene_muts: list[str],
                                     Dsa_mutation_ID: DSA_Mutation_ID, Dsa_data: DSA_data, Dsa_compute: DSA_compute,
                                     ignore_therapies: set, ignore_cancers: set,
                                     min_group_size: int, logrank_kwargs: dict,
                                     debug_level: int = 0, main_test_index: int = 1) -> dict:
    """
    Performs differential survival analysis of a given list of a gene mutations, based on tests configuration.

    test_cnfg: list - a list of lists. Each sub-list contains a test configuration. A test configuration is defined by two tuples, each contains 3 booleans.
    df_gene: pd.DataFrame - a dataframe of all gene's mutations.
    gene_muts: list -  a list of mutation IDs (with mut_id_cols fields separated by mut_id_sep).
    df_clinical_therapies_dict: dict - a dictionary with key as therapy and value as the corresponding clinical data (see prepare_clinical_dfs in data_utils.py).
    debug_level: bool - 0 (no prints), 1 (significant prints), 2 (most prints), 3 (all prints).
    main_test_index: int - relevant only for multiple tests in tests_cnfg. If, for a given mutation, the main_test_index test
    does not yield any significant result (for at least once case of cancer and therapy), then the following tests are not executed for that mutation.
    Set to len(tests_cnfg) to disable.

    The returned dictionary contains only cases with significant results. Key is mut_id|cancer|therapy. Value is a dictionary with significant results.
    """
    if (debug_level > 1) and (len(tests_cnfg) > 1):
        print(f"Running {len(tests_cnfg)} tests. Main test index = {main_test_index}.")

    results = {}
    DS_name = DS_test_group_cnfg.DS_config2name(tests_cnfg)  # used in the key of results
    # therapies to process (note that, e.g., the none_therapy_name "therapy" group is used to select patients in case of
    # patients without the therapy, regardless if it is in ignore_therapies)
    therapy_space = list(set(Dsa_data.df_clinical_therapies_dict.keys()).difference(ignore_therapies))
    for mut_id in gene_muts:
        #mut_info = Dsa_mutation_ID.mid2dict(mut_id)

        # gene's dataframe with mut_id mutation
        #df_mut_id = dtut.df_with_mut_no_end(df_gene, mut_info)[[Dsa_data.cancer_col, Dsa_data.patient_ID_col]].drop_duplicates()  # this is done to remove [proj_name, case_id] duplicates, which do exists (due to other columns)
        df_mut_id = Dsa_mutation_ID.get_df_mut(df_gene, mut_id)[[Dsa_data.cancer_col, Dsa_data.patient_ID_col]].drop_duplicates()  # this is done to remove [proj_name, case_id] duplicates, which do exists (due to other columns)

        # the following is not relevant for tests that use all cancer types (i.e. PAN-cancer) (i.e. when DS_test_group_cnfg.Field_index.Cancer.value == 0)
        # dataframe with mut_id per cancer type
        df_mut_id_grp_cancer = df_mut_id.groupby(by=Dsa_data.cancer_col)
        if (df_mut_id_cancer := df_mut_id_grp_cancer.size().to_frame(name='num_p').sort_values(by='num_p', ascending=False).query(f"num_p >= {min_group_size}").reset_index()).empty:
            # no patients (of any cancer type) with at least min_group_size patients
            if debug_level > 2:
                print(f"\tNot enough cancer patients (separately for each cancer type) affected with {mut_id}.")
            # continue  # commented-out this in case of PAN-cancer analysis. If not PAN-cancer, then df_mut_id_cancer[proj_name_col].unique().tolist() is an empty list and the "for cancer.." below is not executed
        else:
            if debug_level > 1:
                print(f"\n{mut_id=} relevant cancer types (i.e. with at least {min_group_size} patients):")
                print(df_mut_id_cancer)
        
        # regardless of cancer type and of therapy
        ptnt_with_mutid = df_mut_id[Dsa_data.patient_ID_col].unique().tolist()
        ptnt_without_mutid = list({x for x in Dsa_data.df_clinical_therapies_dict[Dsa_data.All_name][Dsa_data.patient_ID_col].unique().tolist() if x not in ptnt_with_mutid})

        # main_test_signf_therapies[cancer] = list of therapies that resulted in significant DS in main_test_index in cancer cancer
        main_test_signf_therapies = defaultdict(list)

        for test_index, test_cnfg in enumerate(tests_cnfg, start=1):
            #cancer_space = ['PAN-cancer'] if all([not x[DS_test_group_cnfg.Field_index.Cancer.value] for x in test_cnfg]) else list(set(df_mut_id_cancer[proj_name_col].unique().tolist()).difference(ignore_cancers))
            cancer_space = ['PAN-cancer'] if all(not x[DS_test_group_cnfg.Field_index.Cancer.value] for x in test_cnfg) else list(set(df_mut_id_cancer[Dsa_data.cancer_col].unique().tolist()).difference(ignore_cancers))
            for cancer in cancer_space:
                cancer_name_in_clinical = Dsa_data.cancer_name_mutation2clinical(cancer)

                # indicates if in this test, both groups are without therapy
                both_groups_no_therapy = all(not x[DS_test_group_cnfg.Field_index.Therapy.value] for x in test_cnfg)

                # if both_groups_no_therapy, then for any therapy, the non-therapy is the
                # none_therapy_name group. Thus, no need to run all therapies in therapy_space, can just run the an arbitrary test (e.g. the first one).
                # Below, if in this case we get a significant result, we produce the same significant result for all therapies.

                if test_index > main_test_index:
                    sgnf_main_therapies = list(set(main_test_signf_therapies.get(cancer, [])))
                    cur_therapy_space = sgnf_main_therapies[:1] if both_groups_no_therapy else sgnf_main_therapies  # if is empty, then cur_therapy_space will also be empty
                    if debug_level > 2:
                        print(f"\tTest {test_index}: running only {cur_therapy_space} therapies for {cancer=} ({sgnf_main_therapies=}, {both_groups_no_therapy=}), as these were the only therapies found to be significant for test {main_test_index}.")
                else:
                    cur_therapy_space = therapy_space[:1] if both_groups_no_therapy else therapy_space
                
                for therapy in cur_therapy_space:
                    # test_cnfg in a tuple of two configuration tuples, first for groupA and second for groupB
                    groupA_name, groupA_p, groupA_clinical_df = get_data_of_cnfg_group(test_cnfg[0], ptnt_with_mutid, ptnt_without_mutid, Dsa_data, cancer_name_in_clinical, therapy)
                    groupB_name, groupB_p, groupB_clinical_df = get_data_of_cnfg_group(test_cnfg[1], ptnt_with_mutid, ptnt_without_mutid, Dsa_data, cancer_name_in_clinical, therapy)

                    test_cnfg_name = f"{groupA_name}-{groupB_name}"
                    #test_cnfg_name = f"{groupA_name}<>{groupB_name}"

                    result, ds_status = differential_survival_two_groups(
                        groupA_p, groupB_p, groupA_clinical_df, groupB_clinical_df,
                        Dsa_data, Dsa_compute, min_group_size, logrank_kwargs)

                    cur_settings = f"{test_cnfg_name}, {mut_id}, {cancer}, {therapy}"

                    match ds_status:
                        case Status.Size_small:
                            if debug_level > 2:
                                print(f"\tNot enough patients for {cur_settings}. There are {result['Group1_size']:,} and {result['Group2_size']:,} patients in each group ...")
                        case Status.DS_insignificant:
                            if debug_level > 2:
                                print(f"\tDS insignificant for {cur_settings}: cox_p_value={result['cox_p_value']:.05f}, logrank_p_value={result['logrank_p_value']:.05f}")
                        case Status.DS_significant:
                            # if both_groups_no_therapy, then this significant result applies to all therapies in therapy_space
                            # (since in this case, cur_therapy_space has only one therapy from therapy_space). We copy this significant
                            # result to all therapies.
                            loop_therapies = therapy_space if both_groups_no_therapy else [therapy]

                            if test_index == main_test_index:
                                main_test_signf_therapies[cancer].extend(loop_therapies)  # saving the therapy that resulted in significant DS (for cancer cancer)

                            if (debug_level > 2) and (both_groups_no_therapy):
                                print(f"\t{cur_settings} produced significant result (i.e. a test where both groups are without therapy). Thus copy significant result to all therapies ...")

                            for thrp in loop_therapies:
                                c_settings = f"{test_cnfg_name}, {mut_id}, {cancer}, {thrp}"
                                if debug_level > 0:
                                    print(f"*****[{c_settings} | G1_size={result['Group1_size']:,}, G2_size={result['Group2_size']:,}, pv_cox={result['cox_p_value']:.06f}, pv_logrank={result['logrank_p_value']:.06f}, AUC1={result['AUC1']:.02f}, AUC2={result['AUC2']:.02f} | pv_samp={result['Sample_p_value']:.06f}, AUC_samp={result['Sample_AUC']:.02f}]*****")
                                
                                # note that the key here is unique over the 4 for loops (mut_id, test_cnfg, cancer, and thrp)
                                results[f"{DS_name}|{test_cnfg_name}|{mut_id}|{cancer}|{thrp}"] = {
                                    'pre_group1_size': len(groupA_p),  # number of affected patients from all therapies (regradless if they have any valid survival parameters (e.g. duration and even values))
                                    'pre_group2_size': len(groupB_p),  # same for un-affected patients
                                } | result
                        case _:
                            if debug_level > 0:
                                print(f"Received an unknown {ds_status=} .....")
    return results


def parse_gene_muts_survival_results(results: dict, pop_fields: list) -> pd.DataFrame:
    """Parses the dictionary result from process_gene_muts_survival_tests function."""
    all_res: list[dict] = []
    for k, v in results.items():
        s = dict(zip(['DSA_type','test_cnfg_name', 'mut_id', 'cancer', 'therapy'], k.split('|'))) | v
        for pop_field in pop_fields:
            s.pop(pop_field, None)
        all_res.append(s)

    return pd.DataFrame(all_res)


def create_survival_results_dataframe(results: dict) ->pd.DataFrame:
    """
    Creates a dataframe containing informative values from results. 
    
    results: dict - process_muts_survival_tests_wrapper output.
    """
    pop_fields: list[str] = [
        'df_group1', 'df_group2',
        'Km_group1', 'Km_group2', 
        'pre_group1_size', 'pre_group2_size'
        ]
    all_dfs = []
    for gn, gn_result in results.items():
        df_gn = parse_gene_muts_survival_results(gn_result, pop_fields=pop_fields)
        df_gn.insert(2, 'gene', gn)
        all_dfs.append(df_gn)
    return pd.concat(all_dfs).sort_values(by=['mut_id', 'cancer', 'therapy']).reset_index(drop=True) if all_dfs else pd.DataFrame()


def plot_gene_muts_survival_result(result_v: dict, title_str: str,
                                   group1_label: str, group2_label: str,
                                   group1_color: str = 'red', group2_color: str = 'blue',
                                   p_value_txt_pos: tuple = (0.4, 0.75), figsize: tuple = (12,6), linewidth: int = 3, fontsize: int = 16):
    """Plots the two survival groups and p values of a result from process_gene_muts_survival function."""
    cox_p_value = result_v['cox_p_value']
    logrank_p_value = result_v['logrank_p_value']
    Km_group1 = result_v['Km_group1']
    Km_group2 = result_v['Km_group2']
    Group1_size = result_v['Group1_size']
    Group2_size = result_v['Group2_size']
    Sample_p_value = result_v['Sample_p_value']
    Sample_AUC = result_v['Sample_AUC']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=True)

    Km_group1.plot_curve(ax, linewidth=linewidth, color=group1_color)
    Km_group2.plot_curve(ax, linewidth=linewidth, color=group2_color)
    ax.legend([f'{group1_label} ({Group1_size:,}, {Km_group1.auc:.02f})', f'{group2_label} ({Group2_size:,}, {Km_group2.auc:.02f})'], fontsize=fontsize-1)

    ax.set_title(title_str, fontsize=fontsize)
    ax.text(p_value_txt_pos[0], p_value_txt_pos[1], f"cox_$p_v={cox_p_value:.05f}$, logrank_$p_v={logrank_p_value:.05f}$", transform=ax.transAxes, fontsize=fontsize-1)
    # writes median results for the sample analysis
    if Sample_p_value is not None:
        ax.text(p_value_txt_pos[0], p_value_txt_pos[1]-0.07, f"Sampling: $p_v={Sample_p_value:.05f}$, AUC={Sample_AUC:.02f}", transform=ax.transAxes, fontsize=fontsize-1)
    ax.set_xlabel('$t$ [years]', fontsize=fontsize-1)
    ax.set_ylabel('$P_s$', fontsize=fontsize-1)
    ax.set_ylim(bottom=0, top=1.1)
    ax.grid()

    return fig, ax


def plot_results_KM_curves_to_pdf(pdf_file: pathlib.Path, gene_results: dict,
                                  exclude_genes: tuple = (),
                                  title_page_text: str = '',
                                  title_page_params = None, #report_curves_title_page_params,
                                  description_pages_text_list = None,
                                  description_pages_params = None, #report_curves_description_pages_params,
                                  figure_title_base: str = 'KM curve for',
                                  figsize: tuple = (12, 6,),
                                  linewidth: int = 3,
                                  fontsize: int = 14,
                                  p_value_txt_pos: tuple = (0.4, 0.75,),
                                  group1_color = 'red',
                                  group2_color = 'blue') -> None:
    """
    Generates all results' KM curves to a pdf file. 
    Supports a title page and (possibly) multiple description pages preceeding the KM curves."""
    if description_pages_text_list is None:
        description_pages_text_list = []
    if title_page_params is None:
        title_page_params = report_curves_title_page_params
    if description_pages_params is None:
        description_pages_params = report_curves_description_pages_params

    with PdfPages(str(pdf_file)) as fp:
        # title page
        if title_page_text:
            first_page = plt.figure(figsize=figsize)
            first_page.clf()
            first_page.text(title_page_params['x_start'], title_page_params['y_start'], title_page_text, 
                            color=title_page_params['color'], horizontalalignment=title_page_params['horizontalalignment'],
                            verticalalignment=title_page_params['verticalalignment'], fontsize=title_page_params['fontsize'],
                            fontstyle=title_page_params['fontstyle'], fontweight=title_page_params['fontweight'])
            fp.savefig(first_page)
            plt.close()

        # description pages
        if description_pages_text_list:
            for page_str in description_pages_text_list:
                page = plt.figure(figsize=figsize)
                page.clf()
                page.text(description_pages_params['x_start'], description_pages_params['y_start'], page_str, 
                          color=description_pages_params['color'], horizontalalignment=description_pages_params['horizontalalignment'],
                          verticalalignment=description_pages_params['verticalalignment'], fontsize=description_pages_params['fontsize'],
                          fontstyle=description_pages_params['fontstyle'], fontweight=description_pages_params['fontweight'])
                fp.savefig(page)
                plt.close()

        # KM curves
        for gene, gene_v in gene_results.items():
            if gene in exclude_genes:
                continue
            for info in gene_v.keys():
                group1_lbl, group2_lbl = info.split('|')[1].split('-')
                fig, _ = plot_gene_muts_survival_result(gene_results[gene][info], f"{figure_title_base} {info}",
                                                        group1_lbl, group2_lbl,
                                                        group1_color=group1_color, group2_color=group2_color,
                                                        figsize=figsize, p_value_txt_pos=p_value_txt_pos,
                                                        linewidth=linewidth, fontsize=fontsize)
                fp.savefig(fig)
                plt.close()


def process_muts_survival_tests_wrapper(tests_cnfg: list, gene_muts_dict: dict[str, list],
                                        Dsa_mutation_ID: DSA_Mutation_ID, Dsa_data: DSA_data, Dsa_compute: DSA_compute,
                                        ignore_therapies: set, ignore_cancers: set,
                                        min_group_size: int, logrank_kwargs: dict,
                                        debug_level: int = 0, main_test_index: int = 1) -> dict:
    """"
    Process arbitrary (i.e. from different genes) mutations effect on survival. 
    
    This is simply a wrapper for process_gene_muts_survival_tests while loading the gene mutation data.

    Inputs:
    tests_cnfg: list - see process_gene_muts_survival_tests
    gene_muts_dict: dict - gene_muts_dict[gene] is a list of mutation IDs of the gene gene.

    Output: a dictionary d, where d[gene] contains the survival analysis result dictionary (output of process_gene_muts_survival) for gene gene.

    See the function process_gene_muts_survival.
    """
    gene_results = {}
    for i, (gene, gene_muts) in enumerate(gene_muts_dict.items(), start=1):
        if (df_gene := Dsa_data.get_gene_mutations(gene)) is None:
            continue

        if df_gene.empty:
            if debug_level > 0:
                print(f"{i}. Mutation data for {gene=} is empty...")
            continue

        if debug_level > 0:
            print(f"{i}. Processing {gene=} ...")

        results = process_gene_muts_survival_tests(tests_cnfg, df_gene, gene_muts,
                                                   Dsa_mutation_ID, Dsa_data, Dsa_compute,
                                                   ignore_therapies, ignore_cancers,
                                                   min_group_size, logrank_kwargs,
                                                   debug_level=debug_level, main_test_index=main_test_index)
        
        if results:
            # indicates that significant results found for gene gene
            if debug_level > 1:
                print(f"\t****Found significant results for {gene=}.****")
            gene_results[gene] = results
    return gene_results


def process_gene_all_muts_survival_tests_wrapper(tests_cnfg: list, gene: str, filter_data: dict,
                                                 Dsa_mutation_ID: DSA_Mutation_ID, Dsa_data: DSA_data, Dsa_compute: DSA_compute,
                                                 ignore_therapies: set, ignore_cancers: set,
                                                 min_group_size: int, logrank_kwargs: dict,
                                                 debug_level: int = 0, main_test_index: int = 1) -> tuple[dict, list, pd.DataFrame]:
    """
    Given a gene, loads its mutation data, filter it, and performs survival analysis on all the filtered mutations according to tests_cnfg.
    This is basically a wrapper to two main functions: get_GDC_filtered_gene_mutation_data (from GDC_dataset_feb2021_utils) and process_gene_muts_survival_tests.

    Specifically, given a gene, we:
    1. Load all gene's mutations, and filter it (based on user input).
    2. Evaluate each mutation effect on survival.

    See details in get_GDC_filtered_gene_mutation_data and process_gene_muts_survival_tests functions.

    Returns a tuple with:
    1. significant results
    2. the gene filtered mutations that were analyzed
    3. the gene unfiltered mutation dataframe
    """
    gene_muts, _, df_gene = Dsa_data.get_filtered_gene_mutations(gene, filter_data)

    result = process_gene_muts_survival_tests(tests_cnfg, df_gene, gene_muts,
                                              Dsa_mutation_ID, Dsa_data, Dsa_compute,
                                              ignore_therapies, ignore_cancers,
                                              min_group_size, logrank_kwargs,
                                              debug_level=debug_level, main_test_index=main_test_index)
    
    return result, gene_muts, df_gene


def create_gene_mutations_dict(mut_ids: list[str], Mut_id: DSA_Mutation_ID) -> dict:
    """Given a list of mutation, creates a dictionary D where D[gene] is a list of all gene's mutations."""
    gene_muts = defaultdict(list)
    for mut_id in mut_ids:
        gene_muts[Mut_id.get_gene_name(mut_id)].append(mut_id)
    return dict(gene_muts)
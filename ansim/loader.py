import pandas as pd
import glob
import re
from matplotlib import pyplot as plt

from . import utils
from . import properties



# This function removes all runs that are less than 100 MB
def clean_runs(limit = 100): # in MB
    runs_paths = glob.glob(properties.ANSIM_RUNS_FOLDER_PATH+"*txt")
    print("Current number of runs is: ", len(runs_paths))
    for run_path in runs_paths:
        utils.move_file(run_path, limit, move_to_folder=properties.ANSIM_DISMISSED_RUNS_PATH)
    print("Number of runs after removing small files is: ", len(runs_paths))


# HELPER FUNCTION that is used in the function get_description_file below
def _add_col_id(row):
    row['id'] = 'CH' + row.Channel.split('_')[1]
    row['Description'] = row['Description'] if type(row['Description']) ==type('') and len(row['Description'])>0 else ('signal'+'_'+'_'.join(row.Signal.lower().split(' ')))
    return row

# This function reads the description file, ANSIM_DESCRIPTION_FILE_PATH, and returns a dataframe with the following data:
# 	--> Channel, Signal, Description, Device, Device-Info
def get_description_file(verbose=1):
    if verbose > 0:
        print("NOTE: Channel IDs do not matter. What is important here is Signal and Description.")
    with open(properties.ANSIM_DESCRIPTION_FILE_PATH, 'r+') as file:
        _str = file.read()
        _str = re.sub('\t+', '\t', _str)
        _str = re.sub('\n+', '\n', _str)
        file.seek(0)

        file.write(_str)
    df = pd.read_csv(properties.ANSIM_DESCRIPTION_FILE_PATH, sep='\t', header=0,error_bad_lines=False, warn_bad_lines=True,
            skipinitialspace=True, skip_blank_lines=True)
    df.dropna(axis=0, how ="all", inplace=True)
    df = df.apply(_add_col_id, axis=1)
    df.set_index('id', inplace=True)
    return df


        
# loop over all the experiments and get the id, subject, run, signal source, scsp
# return 1 dataframe where each experiment is 1 row.
# You can optionally save the file by setting save to True and the output_file_name
def get_runs_overview(save=False, output_file_name = "runs_overview.csv"):
    instances = []
    runs_paths = glob.glob(properties.ANSIM_RUNS_FOLDER_PATH+"*txt")

    for fname in runs_paths:
        with open(fname, 'r+') as file:

            row = {}
            ## add attributes
            # NZP2DE_0_001_0AH_SC1CSP1_BIO1E_2018-11-21T09_33_46
            name = fname.split('\\')[-1]
            row['id'] = name
            row['subject'] = name.split('_')[3]
            row['run'] = name.split('_')[2] # usually a run with assistant or no assistant0,,
            row['signal_source'] = name.split('_')[5]
            row['scsp'] = name.split('_')[4]

            instances.append(row)   
    overview_df = pd.DataFrame(instances)
    if save:
        utils.create_folder(properties.OUTPUT_PATH)
        overview_df.to_csv(properties.OUTPUT_PATH + output_file_name)
        print('The runs overview dataframe was saved as csv in {}'.format(properties.OUTPUT_PATH + output_file_name))

    return overview_df

# print general info from function get_runs_overview
def print_runs_overview(overview_df = None):
    if overview_df is None:
        overview_df = get_runs_overview()
    print('overall stats:')
    print('# of files: ', len(overview_df))
    print('# of unique subjects: ', len(overview_df.subject.unique()))
    print('# of unique runs: ', len(overview_df.run.unique()))
    print('# of unique signal source: ', len(overview_df.signal_source.unique()))
    print('# of unique scsp: ', len(overview_df.scsp.unique()))


# Get the info of 1 run and plot it
# For Information: BIO1E is BIOPAC System for Scientific medicine data
# and FIN1E is Finapress system for continuous finger blood pressure
# you have the option to plot and/or save it
def get_single_run_data(run_path, plot= False, save=False):
    col_metric_mapping = {}
    desc_df = get_description_file(verbose=0)
    with open(run_path, 'r+') as file:
        _str = file.read()
        _str = re.sub('\t+', '\t', _str)
        _str = re.sub('\n+', '\n', _str)
        file.seek(0)

        file.write(_str)
        file.seek(0)

        ##### columns
        lines =  file.readlines()
        channels = int(lines[2].split(' ')[0])
        desc_limit = 3+(channels*2)
        for l in range(3, (3+(channels*2)), 2):
            col_metric_mapping[utils.clean_excessive_spaces(lines[l], remove_all=True)] = utils.clean_excessive_spaces(lines[l + 1], remove_all=True)
        
        col_names=[]
        for k in col_metric_mapping.keys():
            col_names.append(desc_df[desc_df.Signal.apply(lambda x: k.startswith(x))].Description.values[0])
            
        time = utils.clean_excessive_spaces(lines[desc_limit]).split('\t')[0]
        col_names = [time] + col_names
        
        ####
        _df = pd.read_csv(run_path, sep='\t', names=col_names, error_bad_lines=False, warn_bad_lines=True, 
                skipinitialspace=True, skip_blank_lines=True, skiprows=(desc_limit+1), index_col=False)
        _df.dropna(axis=0, how ="all", inplace=True)
        _df.set_index(time, inplace=True)
        
        ## add attributes
        # example name: NZP2DE_0_001_0AH_SC1CSP1_BIO1E_2018-11-21T09_33_46
        splitted = run_path.split('/')
        splitted = run_path.split('\\') if len(splitted) < 2 else splitted
        name = splitted[-1]
        _df['id'] = name
        _df['subject'] = name.split('_')[3]
        _df['run'] = name.split('_')[2] # usually a run with assistant or no assistant0,,
        _df['signal_source'] = name.split('_')[5]
        _df['scsp'] = name.split('_')[4]
        
        if plot:
            _df.plot.line(subplots=True, figsize=(7, 10))
            plt.show()
            
        _df.reset_index(inplace=True)
        
        if save:
            utils.create_folder(properties.OUTPUT_RUN_CSV_PATH)
            saved_path = properties.OUTPUT_RUN_CSV_PATH + '{}.csv'.format(name.split('.')[0])
            _df.to_csv(saved_path)
            print('The run was saved as csv under: ', saved_path)
        
        col_metric_mapping['name'] = name
        return _df, col_metric_mapping


# returns a dataframe containing all the metrics of the run
# this is just to check that the metrics are consistent across all runs
def get_all_runs_metrics(save= False, output_file_name="metrics.csv"):

    metrics = []
    
    runs_paths = glob.glob(properties.ANSIM_RUNS_FOLDER_PATH+"*txt")

    for f in runs_paths:
        df, c = get_single_run_data(f, save=False)
        metrics.append(c)

    metrics_df =   pd.DataFrame(metrics)  
    if save:
        utils.create_folder(properties.OUTPUT_PATH)
        saved_path = properties.OUTPUT_PATH + '{}'.format(output_file_name)
        metrics_df.to_csv(saved_path)
        print('The metrics overview of all runs is saved here: ', saved_path)
    return metrics_df


# Converts all files to csv
def convert_runs_to_csv():
    
    runs_paths = glob.glob(properties.ANSIM_RUNS_FOLDER_PATH+"*txt")
    for f in runs_paths:
        _, _= get_single_run_data(f, save=True)

    return

def get_run_csv_paths(path=properties.OUTPUT_RUN_CSV_PATH):
    runs_csv_paths = glob.glob(path+"*.csv")
    if len(runs_csv_paths) == 0:
        print('The runs were not converted to csvs. Please run ansim.loader.convert_runs_to_csv() and then continue')
    return runs_csv_paths


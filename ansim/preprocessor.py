
import pandas as pd
import numpy as np
import random
import os

from . import window_dataset
from . import loader

class Preprocessor:
    SPLIT_BY_RUN = "RUN"
    SPLIT_BY_SUBJECT = "SUBJECT"



    ### HELPERS defs
    @staticmethod
    def _get_baseline_run_match(run_path):
        splitted = run_path.split('/')
        splitted = run_path.split('\\') if len(splitted) < 2 else splitted
        name = splitted[-1]

        match = name[0:17]
        return match

    @staticmethod
    def _get_run_path_subject(run_path):
        splitted = run_path.split('/')
        splitted = run_path.split('\\') if len(splitted) < 2 else splitted
        name = splitted[-1]

        subject = name.split('_')[3]
        return subject


    # class init
    def __init__(self, x_columns=None, y_columns=None,
                 omit_baseline = False,
                 train_split= 0.8, split_by= SPLIT_BY_RUN,
                 windowDataset= window_dataset.WindowDataset(),
                 data_root_path = ''):

        self.windowDataset = windowDataset
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.omit_baseline = omit_baseline
        self.train_split = train_split
        self.split_by = split_by
        self.data_root_path = data_root_path


    def get_run_data(self, run_path, max_gz, tolerance, is_baseline):
        run = pd.read_csv(run_path)
        run = run.sort_values(by=['min'])
        run.reset_index(inplace=True)
        run.dropna(how='any', inplace=True)
        run['max_gz'] = max_gz
        run['tolerance'] = tolerance
        run['is_baseline'] = 1 if is_baseline else 0
        return run


    def get_alternate_run(self, run_path):
        runs_csv_paths = loader.get_run_csv_paths('{}data/sahc_csv/*'.format(self.data_root_path))
        matching = [x for x in runs_csv_paths \
                    if Preprocessor._get_baseline_run_match(x)  == Preprocessor._get_baseline_run_match(run_path) and \
                    x != run_path]
        if len(matching) >0 :
            print(matching)
        return matching[0]




    def validate_run(self, run_path):
        is_valid = True
        is_baseline = False
        err_message = ""
        run = pd.read_csv(run_path)
        print('*--------------------------*')
        # dismiss invalid subject
        if run['subject'].values[0] == '0AA':
            err_message = "Invalid Subject : 0AA"
            is_valid = False

        # dismiss when data has no min column
        if 'min' in run.columns:
            run = run.sort_values(by=['min'])
            run.reset_index(inplace=True)
        else:
            is_valid = False
            err_message = "Data has no valid time column (min)"

        # it is a baseline if g-force is less than or equal to 0.75
        run.dropna(how='any', inplace=True)

        max_g_force = round(run['G-force_z_axis'].max(), 2)

        # baseline
        if max_g_force <= 0.75 and max_g_force > 0.65: # baseline
            is_baseline = True

        # low gforce
        if max_g_force <= 0.65:
            is_valid = False
            err_message = "max g z force is too small:" + str(max_g_force)

        # Here, round the g-force to 2 if it is less that 1, otherwise round
        run['gz_rounded'] = np.where(run['G-force_z_axis'] < 1, round(run['G-force_z_axis'], 2),
                                     round(run['G-force_z_axis']))

        max_gforce_z_df = run[run['gz_rounded'] == run['gz_rounded'].max()]

        max_gz = run['gz_rounded'].max()
        tolerance = round((max_gforce_z_df['min'].max() - max_gforce_z_df['min'].min()), 2)

        # in case the tolerance is so low - re-adjusts
        if is_valid and max_gz > 1:
            decrease = 0.6
            while tolerance < 0.1:
                print('resetting tolerance:', tolerance)
                new_max_gz = max_gz - decrease
                max_gz = max_gz - 1 if ((decrease * 10 % 6 == 0) or ((decrease - 0.6) * 10 % 9 == 0)) else max_gz
                run['gz_rounded'] = np.where(round(run['G-force_z_axis'], 1) >= new_max_gz, max_gz, run['gz_rounded'])
                max_gforce_z_df = run[run['gz_rounded'] == max_gz]
                tolerance = round((max_gforce_z_df['min'].max() - max_gforce_z_df['min'].min()), 2)
                decrease = decrease + 0.1

        if is_baseline:
            actual_run_path = self.get_alternate_run(run_path)
            actual_run_data = self.validate_run( actual_run_path)
            tolerance  = actual_run_data['tolerance']
            max_gz = actual_run_data['max_gz']
            print('setting baseline tolerance and max gz to: ', tolerance, ', ', max_gz)


        return  {"valid": is_valid,
                     "message": err_message,
                     'file_name': run_path,
                     'is_baseline': is_baseline,
                     'tolerance': tolerance,
                     'max_gz': max_gz}

    def get_all_valid_invalid_runs(self, verbose=0):  # 0 no comments, 1 overall comments, and 2 detailed comments

        maxgforce_tolerance_path = '{}data/experiment_runid_maxgz_tolerance.csv'.format(self.data_root_path)

        runs_csv_paths = loader.get_run_csv_paths('{}data/sahc_csv/*'.format(self.data_root_path))
        print('there are {} csv files.'.format(str(len(runs_csv_paths))))

        if verbose > 0 :
            print("an overview of  runs will be saved here: ", maxgforce_tolerance_path)


        # check if we already saved the list of invalid runs
        if os.path.isfile(maxgforce_tolerance_path) :
            data = pd.read_csv(maxgforce_tolerance_path)
        else:
            data_arr = []
            if verbose == 1: print('fetching invalid files')

            for csv in runs_csv_paths:
                if verbose == 2: print(csv)

                validity = self.validate_run(csv)

                data_arr.append(validity)

            if verbose in (1, 2):
                print('saving the  runs with tolerance, max gz, validity reasons as csv in: ', maxgforce_tolerance_path)
            data = pd.DataFrame(data_arr)
            data.to_csv(maxgforce_tolerance_path)

        return data



    def get_train_test_split(self, runs_paths):
        training_paths, test_paths = [], []
        random.seed(1) # keep this to avoid random splitting each time
        if self.split_by == Preprocessor.SPLIT_BY_SUBJECT: #run:
            print('Splitting training and test sets by subjects')
            subjects = []
            for run_path in runs_paths:
                run_subject =Preprocessor._get_run_path_subject(run_path)
                subjects.append(run_subject)
            subjects = list(set(subjects))

            # split the subjects to train and test

            train_size = round(len(subjects) * self.train_split)
            training_subjects = random.sample(subjects, k=train_size)
            print('there are {} subjects in the training set out of {} subjects.'.format(str(len(training_subjects)),
                                                                                             str(len(subjects))))
            print('subjects are: ', str(subjects))

            training_paths = [x for x in runs_paths if Preprocessor._get_run_path_subject(x) in training_subjects]
            test_paths = [x for x in runs_paths if Preprocessor._get_run_path_subject(x) not in training_subjects]

        elif self.split_by == Preprocessor.SPLIT_BY_RUN: #run:
            # split the subjects to train and test
            print('Splitting training and test sets by runs')

            train_size = round(len(runs_paths) * self.train_split)
            training_runs = random.sample(runs_paths, k=train_size)

            training_paths = [x for x in runs_paths if x in training_runs]
            test_paths = [x for x in runs_paths if x not in training_runs]
        else:
            print('Please set the "split_by" to ',  Preprocessor.SPLIT_BY_RUN, ' or ', Preprocessor.SPLIT_BY_SUBJECT)
        return training_paths, test_paths

    def get_windowed_data(self):

        # CHECK VALID CSVS
        runs_maxgz_tolerance_val_df = self.get_all_valid_invalid_runs(verbose=1)
        valid_runs = runs_maxgz_tolerance_val_df[(runs_maxgz_tolerance_val_df['valid'] == True )]
        valid_runs = valid_runs if self.omit_baseline == False else \
            (valid_runs[valid_runs['is_baseline'] == False ])

        # get train and test sets
        training_paths, test_paths = self.get_train_test_split(valid_runs['file_name'].values.tolist())
        dataset_train = None
        dataset_test = None

        for csv in training_paths:
            run_vals = runs_maxgz_tolerance_val_df[runs_maxgz_tolerance_val_df['file_name'] == csv]
            max_gz = run_vals['max_gz'].values.tolist()[0]
            tolerance = run_vals['tolerance'].values.tolist()[0]
            is_baseline = bool(run_vals['is_baseline'].values.tolist()[0])
            run = self.get_run_data(csv, max_gz, tolerance, is_baseline)

            self.windowDataset.y_df = run[self.y_columns]
            self.windowDataset.X_df = run[self.x_columns]
            windowed = self.windowDataset.window_dataset()
            dataset_train = windowed if dataset_train is None else dataset_train.concatenate(windowed)

        for csv in test_paths:
            run_vals = runs_maxgz_tolerance_val_df[runs_maxgz_tolerance_val_df['file_name'] == csv]
            max_gz = run_vals['max_gz'].values.tolist()[0]
            tolerance = run_vals['tolerance'].values.tolist()[0]
            is_baseline = bool(run_vals['is_baseline'].values.tolist()[0])
            run = self.get_run_data(csv, max_gz, tolerance, is_baseline)

            self.windowDataset.y_df = run[self.y_columns]
            self.windowDataset.X_df = run[self.x_columns]
            windowed = self.windowDataset.window_dataset()
            dataset_test = windowed if dataset_test is None else dataset_test.concatenate(windowed)

        return dataset_train, dataset_test

    def prepare_baseline_data(self):
        # CHECK VALID CSVS
        runs_maxgz_tolerance_val_df = self.get_all_valid_invalid_runs(verbose=1)
        print('there are ', len(runs_maxgz_tolerance_val_df), ' files')
        valid_runs = runs_maxgz_tolerance_val_df[(runs_maxgz_tolerance_val_df['valid'] == True)]
        print('there are ', len(valid_runs), ' valid runs')
        valid_runs = valid_runs if self.omit_baseline == False else \
            (valid_runs[valid_runs['is_baseline'] == False])
        print('omit_baseline is ', self.omit_baseline, ' -  valid runs: ', len(valid_runs))

        # get train and test sets
        training_paths, test_paths = self.get_train_test_split(valid_runs['file_name'].values.tolist())


        X_training, y_training, X_test, y_test = None, None, None, None
        for csv in training_paths:
            run_vals = runs_maxgz_tolerance_val_df[runs_maxgz_tolerance_val_df['file_name'] == csv]
            max_gz = run_vals['max_gz'].values.tolist()[0]
            tolerance = run_vals['tolerance'].values.tolist()[0]
            is_baseline = bool(run_vals['is_baseline'].values.tolist()[0])
            run = self.get_run_data(csv, max_gz, tolerance, is_baseline)

            y_training = run[self.y_columns] if y_training is None else pd.concat([y_training, run[self.y_columns]])
            X_training = run[self.x_columns] if X_training is None else pd.concat([X_training, run[self.x_columns]])

        for csv in test_paths:
            run_vals = runs_maxgz_tolerance_val_df[runs_maxgz_tolerance_val_df['file_name'] == csv]
            max_gz = run_vals['max_gz'].values.tolist()[0]
            tolerance = run_vals['tolerance'].values.tolist()[0]
            is_baseline = bool(run_vals['is_baseline'].values.tolist()[0])
            run = self.get_run_data(csv, max_gz, tolerance, is_baseline)

            y_test = run[self.y_columns] if y_test is None else pd.concat([y_test, run[self.y_columns]])
            X_test = run[self.x_columns] if X_test is None else pd.concat([X_test, run[self.x_columns]])

        return X_training, y_training, X_test, y_test
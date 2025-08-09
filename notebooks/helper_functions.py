import ast
from cmath import phase
import os
from xml.etree.ElementInclude import include
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

hyperparams_dir = '../psychopy/hyperparameter_and_stimuli/hyperparameters.csv'

stimuli_dir = '../psychopy/hyperparameter_and_stimuli/2025-06-15_13-07_all_balanced_stimuli.csv'
stimuli_group_1_dir = '../psychopy/hyperparameter_and_stimuli/2025-06-17_18-17_group1_balanced_stimuli.csv'
stimuli_group_2_dir = '../psychopy/hyperparameter_and_stimuli/2025-06-17_18-17_group2_balanced_stimuli.csv'

data_dir = '../psychopy/experiment_data'

# outlier subject IDs = [13]

def load_hyperparameters(file_path: str = hyperparams_dir) -> pd.DataFrame:
    """
    Load hyperparameters from a CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the CSV file containing hyperparameters.
    
    Returns:
        pd.DataFrame: The loaded hyperparameters as a DataFrame.
    """
    try:
        hyperparams = pd.read_csv(file_path, header=0)
        return hyperparams
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
    
def load_stimuli(group: int = 0) -> pd.DataFrame:
    """
    Load stimuli from a CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the CSV file containing stimuli.
    
    Returns:
        pd.DataFrame: The loaded stimuli as a DataFrame.
    """
    try:
        if group == 1:
            stimuli = pd.read_csv(stimuli_group_1_dir, header=0)
        elif group == 2:
            stimuli = pd.read_csv(stimuli_group_2_dir, header=0)
        else:
            stimuli = pd.read_csv(stimuli_dir, header=0)
        return stimuli
    except Exception as e:
        print(f"Error loading stimuli: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def load_data(file_path: str, exclude_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame and add subject_id and group_id columns
    based on the filename. The filename is expected to follow the pattern:
    YYYYMMDD_HHMMSS_<proband_nr>_group_<group_nr>_[...].csv

    Parameters:
        file_path (str): The path to the CSV file.
        exclude_ids (List[int], optional): List of subject IDs to exclude from the DataFrame.

    Returns:
        pd.DataFrame: The loaded data with added 'subject_id' and 'group_id' columns.
    """
    try:
        data = pd.read_csv(file_path, header=0)
        file_name = os.path.basename(file_path)
        pattern: str = r'^\d{8}_\d{6}_(\d+)_group_(\d+)_experiment_results.csv$'
        match = re.match(pattern, file_name)
        if match:
            subject_id, group_id = match.groups()
            print(f"Loading data from file: {file_name}")
            print("Subject ID:", subject_id, "Group ID:", group_id)
            data['subject_id'] = int(subject_id)
            data['group_id'] = int(group_id)
        else:
            print(f"Filename {file_name} does not match expected pattern.")
        if exclude_ids:
            data = data[~data['subject_id'].isin(exclude_ids)]
            print(f"Excluding subject IDs: {exclude_ids} because they are outliers.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
    
# load all data files in the data directory
def load_all_data_as_df(directory: str, exclude_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load all CSV files in a specified directory.

    Depending on the 'as_dict' flag, either returns a dictionary mapping file names to
    DataFrames or a single DataFrame obtained by vertically concatenating all CSV files.

    Parameters:
        directory (str): The path to the directory containing CSV files.
        as_dict (bool): If True, return a dictionary of DataFrames; otherwise, return a concatenated DataFrame.
                        Default is False (concatenated DataFrame).

    Returns:
        Union[Dict[str, pd.DataFrame], pd.DataFrame]: A dict of DataFrames if as_dict is True,
                                                        otherwise a concatenated DataFrame.
    """
    
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    data_frames = [load_data(os.path.join(directory, filename), exclude_ids=exclude_ids) for filename in files]
    full_df = pd.concat(data_frames, ignore_index=True)
    
    return full_df



### calculations ###

def compute_likelihood(observed_vars: Tuple[str], observed_values: Tuple[float], observed_E: int, hyperparams: pd.DataFrame) -> float:
    """
    Computes the probability distribution for a given mean and standard deviation.
    
    Parameters:
        observed_vars (List[str]): List of observed variables within one trial (always 1 in the training phase and up to 3 in the test phase).
        observed_values (List[float]): Corresponding observed values for each of the observed variables.
        observed_E (int): The observed E value for the trial.
        hyperparams (pd.DataFrame): DataFrame containing hyperparameters for each variable.
    
    Returns:
        float: The computed likelihood for the observed variables and values given the hyperparameters.
    """
    likelihood = 1.0
    
    for V, value_V in zip(observed_vars, observed_values):
        # Print computing likelihood for variable: V, observed value: value_V, observed E: observed_E
        var_hyperparams = hyperparams[hyperparams['Variable'] == V].iloc[0]
        mu_V = var_hyperparams['intercept'] + var_hyperparams['coef'] * observed_E
        sigma_V = var_hyperparams['sigma']
        
        # Print using hyperparameters: mu=mu_V (type mu_V), sigma=sigma_V (type sigma_V)
        # Print likelihood for V (type V) with value value_V (type value_V):
        
        likelihood *= 1 / np.sqrt(2 * np.pi * sigma_V**2) * np.exp(-0.5 * ((value_V - mu_V) / sigma_V) ** 2) / (sigma_V * np.sqrt(2 * np.pi))
        
    # Print total likelihood: likelihood
    return likelihood


def compute_optimal_posterior(exp_data: pd.DataFrame, hyperparams: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the optimal posterior distribution for each trial in the experiment data.
    
    Parameters:
        exp_data (pd.DataFrame): DataFrame containing the experimental data. Columns: [phase,block,trial_id,trial,E,cue,cue_value,estimate,rt]
        hyperparams (pd.DataFrame): DataFrame containing the hyperparameters for each variable.
        
    Returns:
        pd.DataFrame: Experimental Data DataFrame with the computed posterior distributions as a new column.
    """
    posteriors = []
    
    for index, row in exp_data.iterrows():
        if index % 1000 == 0:        # type: ignore
            print(f"Processing trial {index} of {len(exp_data)}")
        trial = row['trial']
        E = row['E']
        observed_vars_per_trial = row['cue']
        cue_values_per_trial = row['cue_value']
        
        
        # Compute the likelihood
        likelihood = compute_likelihood(observed_vars=observed_vars_per_trial, observed_values=cue_values_per_trial, observed_E=E, hyperparams=hyperparams)
        denominator = likelihood + compute_likelihood(observed_vars=observed_vars_per_trial, observed_values=cue_values_per_trial, observed_E=1-E, hyperparams=hyperparams)
        
        # Compute the posterior (assuming uniform prior for simplicity)
        posterior = np.round(likelihood / denominator, 4)
        posteriors.append(posterior)
    print("Finished processing all trials.")
    exp_data['bayes_optimal_posterior'] = posteriors
    return exp_data


def compute_deviations(exp_data: pd.DataFrame, metric: int = 1) -> pd.DataFrame:
    """
    Computes the deviations of the estimated posterior from the optimal posterior.
    
    Parameters:
        exp_data (pd.DataFrame): DataFrame containing the experimental data with posterior distributions.
        metric (int): The metric to use for deviation calculation. Default is 2 (Euclidean distance).
        
    Returns:
        pd.DataFrame: DataFrame with an additional column for deviations.
    """
    exp_data['deviation'] = np.round(np.abs(exp_data['estimate'] - exp_data['bayes_optimal_posterior']) ** metric, 4)
    return exp_data


def preprocess_data(path_to_data_files: str = data_dir, hyperparams: pd.DataFrame = load_hyperparameters(), exclude: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Preprocesses the experimental data by converting string representations of lists into actual lists.
    
    Parameters:
        exp_data (pd.DataFrame): DataFrame containing the experimental data.
        hyperparams (pd.DataFrame): DataFrame containing the hyperparameters for each variable.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'cue' and 'cue_value' columns converted to lists.
    """
    # Determine the number of files in the target directory
    num_files = len([f for f in os.listdir(path_to_data_files) if f.endswith('.csv')])
    
    # If there is a file with the name 'num_files_full_data.csv', load it and return it
    existing_filename = f'{num_files}_full_data_exclude_{ "_".join(str(exclude_id) for exclude_id in exclude) }.csv' if exclude else f'{num_files}_full_data.csv'
    
    if os.path.exists(existing_filename):
        full_data_df = pd.read_csv(existing_filename)
        # Convert "cue" column to a tuple of strings
        full_data_df['cue'] = full_data_df['cue'].apply(lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else tuple(x))        # Convert "cue_value" column to a float value from a tuple

        # Example for a pandas Series column
        full_data_df['cue_value'] = full_data_df['cue_value'].apply(
            lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else x
        )        
        return full_data_df
    
    else:
        full_data_df = load_all_data_as_df(directory=path_to_data_files, exclude_ids=exclude)
        
    data = full_data_df.copy()

    # Exclude outlier subject_id = [13]
    # data = data[data['subject_id'] != 13].reset_index(drop=True)  
    
    # Convert "cue" column to a tuple of strings
    data['cue'] = data['cue'].apply(func=lambda x: tuple(s.strip(" '") for s in x.strip("[]").split(",")))

    # Convert "cue_value" column to a tuple of floats
    data['cue_value'] = data['cue_value'].apply(lambda x: tuple(float(s.strip(" '")) for s in x.strip("[]").split(", ")))
    
    # Set the 'trial' column to 1 for all training trials, since it should indicate the number of shown variables (which is always 1 in the training phase)
    data.loc[data['phase'] == 'train', 'trial'] = 1
    
    # Set rt to 10 everywhere where rt is -1
    data['rt'] = data['rt'].replace(-1, 10)
    
    # Compute sum of all rt values per subject_id
    data['rt_sum_per_subject'] = data.groupby('subject_id')['rt'].transform('sum')/ 60  # Convert to minutes
    
    # Create a mapping from (subject_id, trial_id, first value of cue in test) to block
    test_mapping = data[data['phase'] == 'test'].apply(
        lambda row: ((row['subject_id'], row['trial_id'], row['cue'][0]), row['block']),
        axis=1
    ).to_dict()

    # Assign train_trials_index in the 'train' phase based on the mapping
    data['train_trials_index'] = 301  # Initialize with NaN
    data.loc[data['phase'] == 'train', 'train_trials_index'] = data[data['phase'] == 'train'].apply(
        lambda row: test_mapping.get((row['subject_id'], row['trial_id'], row['cue'][0]), np.nan),
        axis=1
    )
    # Remove all demo trials where 'block' is -1
    data = data[data['block'] != -1].reset_index(drop=True)
    
    # Filter out trials with missing estimation values (-1)
    data = data[data['estimate'] != -1].reset_index(drop=True)
    
    # Convert estimation values to floats
    data['estimate'] = np.round(data['estimate'] / 100.0, 2)
     
    # Set the correct estimation values for 'E' == 0 trials
    data['estimate'] = data.apply(lambda row: 1 - row['estimate'] if row['E'] == 0 else row['estimate'], axis='columns')
    
    # Convert subject_id to int
    data['subject_id'] = data['subject_id'].astype(int)

    # Extend data with a new column for the posterior
    data = compute_optimal_posterior(exp_data=data, hyperparams=hyperparams)
    
    # Extend data with a new column for the deviations
    data = compute_deviations(exp_data=data, metric=1)
        
    # Compute mean and std estimate per cue and trial_id for each phase (train and test) separately and assign the values to the same columns
    data['mean_estimate'] = data.groupby(['phase', 'cue', 'trial_id'])['estimate'].transform('mean').round(4)
    data['std_estimate'] = data.groupby(['phase', 'cue', 'trial_id'])['estimate'].transform('std').round(4)
    
    # Compute mean and std deviation over time per cue and block for each phase separately and assign the values to the same columns
    data['mean_deviation'] = data.groupby(['phase', 'cue', 'block'])['deviation'].transform('mean').round(4)
    data['std_deviation'] = data.groupby(['phase', 'cue', 'block'])['deviation'].transform('std').round(4)
    
    # Set number of subjects per cue combination
    data['num_subjects'] = data.groupby(['phase', 'cue', 'trial_id'])['subject_id'].transform('nunique')
    
    # New column for number of shown cues per trial and place it after the block column
    data.insert(data.columns.get_loc('trial_id'), 'num_cues', data['cue'].apply(lambda x: len(x) if isinstance(x, tuple) else -1)) # type: ignore
    # data = data[['phase', 'block', 'num_cues', 'trial_id', 'E', 'cue', 'cue_value', 'estimate', 'rt', 'bayes_optimal_posterior', 'mean_estimate', 'std_estimate', 'mean_deviation', 'std_deviation', 'deviation', 'num_subjects']]

    
    # Save preprocessed data to a CSV file
    filename = f'{num_files}_full_data_exclude_{ "_".join(str(exclude_id) for exclude_id in exclude) }.csv' if exclude else f'{num_files}_full_data.csv'
    data.to_csv(filename, index=False)

    # Extend data with a new column for the empirical accuracy to be able to compare it with the theoretical values (missing)
    return data


def create_subset(full_dataset: pd.DataFrame, cue: tuple, phase: str) -> pd.DataFrame:
    """
    Creates a subset of the full dataset based on the specified cue and phase.
    
    Args:
        full_dataset (pd.DataFrame): The complete dataset.
        cue (str): The cue to filter by.
        phase (str): The phase to filter by ('train' or 'test').
    
    Returns:
        pd.DataFrame: The filtered subset of the dataset.
    """
    subset = full_dataset[
        (full_dataset['phase'] == phase) &
        (full_dataset['cue'] == cue)
        ].drop_duplicates('trial_id')[['trial_id', 'E', 'cue', 'mean_estimate', 'std_estimate', 'bayes_optimal_posterior', 'mean_deviation', 'std_deviation', 'num_subjects']].sort_values('bayes_optimal_posterior').reset_index(drop=True)

    return subset

def get_var_colors() -> List[List[str]]:
    """
    Returns a list of color pairs for each variable.
    
    Returns:
        List[List[str]]: A list of color pairs for each variable.
    """
    return [['#11875d', '#0bf77d'], ['#f7022a', '#ff9408'], ["#1a31ff", "#5db9ff"]]


def get_n_cue_colors() -> List[List[List[str]]]:
    """
    Returns a list of color pairs for each number of cues.
    
    Returns:
        List[List[List[str]]]: A list of color pairs for each number of cues.
    """
    colors_1_cue = get_var_colors()
    return [[colors_1_cue[0], ['#154406', '#74a662'], ['#373e02', '#acbf69']],
            [colors_1_cue[1], ['#c44240', '#ffb16d'], ['#840000', '#d8863b']],
            [colors_1_cue[2], ["#00459F", "#5ECEEA"], ["#2C3B7E", "#94B2FF"]]]


# A function that takes a plot function as input and saves the resulting plot as an SVG file in a specified directory
def save_plot_as_svg(
    plot_func,
    file_name: str,
    file_path: str = "plots",
    *args,
    **kwargs
):
    """
    Saves a plot generated by the given plot function as an SVG file.

    The plot function may either
      a) return the Figure it created, or
      b) draw directly on plt, in which case we grab the current figure.

    Parameters:
    -----------
    plot_func : callable
        The plotting function itself (not its return value). 
        Should accept *args, **kwargs and either return a matplotlib.figure.Figure
        or draw on plt.
    file_name : str
        The desired filename (without ".svg").
    file_path : str
        Directory where the file will be saved (created if needed).
    *args, **kwargs :
        Passed through to plot_func.
    """
    if not callable(plot_func):
        raise TypeError(f"Expected a callable plotting function, got {type(plot_func)}")

    # 1) Generate the plot and try to get the Figure it created
    fig = plot_func(*args, **kwargs)
    
    # 2) If plot_func returned nothing, grab the current figure
    if not isinstance(fig, Figure):
        fig = plt.gcf()

    # 3) Ensure the target directory exists
    os.makedirs(file_path, exist_ok=True)
    
    # 4) Construct the complete path
    svg_path = os.path.join(file_path, file_name + ".svg")
    
    # 5) Save only this Figure
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    print(f"Plot saved as SVG at: {svg_path}")

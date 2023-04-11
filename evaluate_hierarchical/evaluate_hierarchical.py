import os
import sys
import json
from config.log import logger
from utils import file_utils, evaluation_utils

from datetime import datetime

def add_args(group):
    """ Description of possible arguments for the evaluation of hierarchical models
    """
    group.add_argument(
        '-top', '--top_level_folder',
        metavar='',
        dest='top_level_folder',
        required=True,
        type=str,
        help='Specify the folder in which the results of the evaluation of the top-level classifier lies'
    )
    group.add_argument(
        '-lift', '--lifting_folder',
        metavar='',
        dest='lifting_folder',
        required=True,
        type=str,
        help='Specify the folder in which the results of the evaluation of the lifting classifier lies'
    )
    group.add_argument(
        '-walk', '--walking_folder',
        metavar='',
        dest='walking_folder',
        required=True,
        type=str,
        help='Specify the folder in which the results of the evaluation of the walking classifier lies'
    )
    group.add_argument(
        '-n', '--nr_obs_per_activity',
        metavar='',
        dest='nr_obs_per_activity',
        type=int,
        default=4718,
        help='Specify the amount of observations to be seen in the joined predictions.csv, default: %(default)s'
    )
    group.add_argument(
        '-o', '--output_date',
        metavar='',
        dest='output_date',
        type=str,
        help='Specify the date of the folder in which to save the files with the features, format YYYY-MM-DDThh:mm'
    )


def check_table_args(table):
    """ Sets the default values if not given by user
    """
    if "dryrun" not in table:
        table["dryrun"] = False
    if "output_date" not in table:
        table["output_date"] = None
    return table


def evaluate_hierarchical(table: dict):
    """ Joins the evaluation of three classifiers for a joined evaluation
    """
    error = evaluation_utils.valid_table_hierarachical(table)
    if error:
        return

   # For better readability the variables are created internally from the table
    data_path = table.get("data_path")
    store_local = not table.get("dryrun")
    top_level_folder = table.get("top_level_folder")
    lifting_folder = table.get("lifting_folder")
    walking_folder = table.get("walking_folder")
    nr_obs_per_activity = table.get("nr_obs_per_activity")
    output_date = table.get("output_date")
    service = "evaluate_hierarchical"


    # When the results are being stored and the respective folders do not exist yet, create them at <dp>/evaluate_model/<network_type>/<saving_time>
    saving_time = datetime.now().strftime(
        '%Y-%m-%dT%H-%M') if output_date is None else output_date
    with open(os.path.join(top_level_folder, "log.json")) as f:
        model_log_file = json.load(f)
    network_type = model_log_file.get("train_model").get("type")
    saving_folder = os.path.join(
        data_path, "evaluate_model", network_type, saving_time)
    table["saving_folder"] = saving_folder
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)



    # Initialize a log file in which the progress of evaluation
    log_file, error = file_utils.initialize_log_file(os.path.join(saving_folder, "log.json"), os.path.join(
        top_level_folder, "log.json"), service, table, os.path.join(lifting_folder, "log.json"), os.path.join(walking_folder, "log.json"))
    if error:
        logger.error(f"Initializing log file failed: {error}")
        return


    # Read the individual predictions.csv files to find the individual probabilities of classifying correct /incorrect 
    singular_probs, error = evaluation_utils.create_all_singular_probabilities(top_level_folder, lifting_folder, walking_folder)
    if error:
        return

    # Create a joined Confusion matrix containing the (conditional) probabilities. Coditional because the probaility of classifiying lifting or walking activities correct is 
    # dependent on the correct classification of the top-lecel classifier
    classes = ["lifting", "dropping", "holding", "walking\nstraight", "walking\nup\nstairs", "walking\ndown\nstairs", "resting"]
    df_prob, error = evaluation_utils.create_df_prob(singular_probs, classes)
    if error:
        return

    # Create a df containing concrete oberservations of the respective activities in a balanced way, i.e. each activity is observed equally often
    pred_df, error = evaluation_utils.create_pred_df(df_prob, nr_obs_per_activity)
    if error:
        return

    # Create a file containing the predictions of the joined model
    predictions_df = evaluation_utils.create_joined_predictions(pred_df, nr_obs_per_activity)

    if store_local:
        predictions_df.to_csv(os.path.join(
            saving_folder, "predictions.csv"), index=False)
        error = file_utils.update_log_file(
            log_file, table, saving_folder, service)
        if error:
            return error


    confusion_matrix, accuracy, error = evaluation_utils.create_confusion_matrix(predictions_df, store_local, saving_folder, "top", False)
    if error:
        return

    resulting_metrics, cohen_k_score, matthews_corr_coeff, error = evaluation_utils.calculate_metrics(
        predictions_df, saving_folder)
    if error:
        return
    log_file[service]["cohen_k_score"] = cohen_k_score
    log_file[service]["matthews_corr_coeff"] = matthews_corr_coeff

    if store_local:
        predictions_df.to_csv(os.path.join(
            saving_folder, "predictions.csv"), index=False)
        resulting_metrics.to_csv(os.path.join(
            saving_folder, "resulting_metrics.csv"))

    if store_local:
        if len(log_file[service]["occurred_errors"]) == 0:
            log_file[service]["successfully_processed_files"] = "all"

        error = file_utils.update_log_file(
            log_file, table, saving_folder, service)
        if error:
            return error
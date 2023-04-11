import os
import json
import warnings
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef


from config.log import logger
from utils import file_utils, time_utils, train_model_utils, plot_utils, data_utils


def valid_table(table: dict):
    """Checks if all the given parameters in the table are valid for the evaluation of the model
    """
    if not os.path.exists(table.get("test_data")):
        logger.error(f"The given test_data folder does not exist")
        return "invalid_test_data_given"

    if table.get("dryrun"):
        logger.warning(f"Storing results is highly recommended")

    if not isinstance(table.get("dryrun"), bool):
        logger.error(f"Got dryrun of nonbool dtype")
        return "got_dryrun_of_nonbool_dtype"

    if not os.path.exists(table.get("model_folder")):
        logger.error(f"The given model_folder does not exist")
        return "invalid_model_folder_given"

    model_files = []
    json_files, error = file_utils.get_files(table.get("model_folder"), "json")
    if error:
        return error

    if "SVM" not in table.get("model_folder"):
        pt_files, error = file_utils.get_files(table.get("model_folder"), "pt")
        if error:
            return error
        model_files.extend(pt_files)

    model_files.extend(json_files)

    if not model_files:
        logger.error(f"The given model_folder does not contain a model file")
        return "invalid_model_folder_given"

    if not os.path.exists(table.get("test_data")):
        logger.error(f"The given test_data does not exist")
        return "invalid_test_data_given"

    test_files, error = file_utils.get_files(table.get("test_data"), "csv")
    if error:
        logger.error(f"The test_data folder does not contain any csv files")
        return error

    if not os.path.exists(os.path.join(table.get("test_data"), "log.json")):
        logger.error(
            f"test_data folder does not contain a log file with the prior treatments")
        return "test_data_log_file_not_found"

    if table.get("test_data") == table.get("model_folder"):
        logger.warning(
            f"Got same folder for the storage of model and the test data")
        return "received_identical_model_and_data_folder"

    if table.get("output_date") is not None:
        error = time_utils.valid_time_string(table.get("output_date"))
        if error:
            return error

    return None


def valid_preprocessing_of_data_for_model(model_folder: str, test_data: str):
    """Uses the log file to test if the preprocessing steps of the test data are suited for the given model
    """
    if not os.path.exists(model_folder):
        logger.error("Given model folder does not exist")
        return "given_model_folder_does_not_exist"

    if not os.path.exists(os.path.join(model_folder, "log.json")):
        logger.error("Given model folder does not contain a log file")
        return "given_model_folder_does_not_contain_log_file"

    if not os.path.exists(test_data):
        logger.error("Given test_data folder does not exist")
        return "given_test_data_folder_does_not_exist"

    if not os.path.exists(os.path.join(test_data, "log.json")):
        logger.error("Given test_data folder does not contain a log file")
        return "given_test_data_folder_does_not_contain_log_file"

    with open(os.path.join(test_data, "log.json")) as f:
        test_data_log_file = json.load(f)

    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)

    # As the log.json of the SVM folder only contains which level the seen SVM classifies, no info regarding prior processing exists
    if "SVM" in model_folder:
        return None

    for preprocessing_step in test_data_log_file.keys():
        if preprocessing_step not in model_log_file.keys():
            logger.error(
                f"The Preprocessing step {preprocessing_step} was executed on the test-data, but not on the model")
            return "preprocessing_steps_differ"

    if "create_windows" not in test_data_log_file.keys():
        logger.error(
            f"Evaluating models for models without data in featured format not yet implemented")
        return "received_unwindowed_data_not_yet_implemented"

    test_data_window_length = test_data_log_file.get(
        "create_windows").get("window_length")
    model_window_length = model_log_file.get(
        "create_windows").get("window_length")
    if test_data_window_length != model_window_length:
        logger.error(
            f"The model was trained on windows of length {model_window_length}, whereas the length seen in the test data is {test_data_window_length}")
        return "differing_window_lengths_found"

    if test_data_log_file.get("create_windows").get("flatten") != model_log_file.get("create_windows").get("flatten"):
        logger.error(
            f"The model was trained on  flattened windows: {model_log_file.get('create_windows').get('flatten')}, whereas the test data is flattened: {test_data_log_file.get('create_windows').get('flatten')}")
        return "differing_flatten_values_found"

    if test_data_log_file.get("create_windows").get("method") != model_log_file.get("create_windows").get("method"):
        logger.error(
            f"The model data was filled via : {model_log_file.get('create_windows').get('method')}, whereas the test data was filled via: {test_data_log_file.get('create_windows').get('method')}")
        return "differing_filling_methods_values_found"

    if test_data_log_file.get("balancing_over").get("granularity") != model_log_file.get("balancing_over").get("granularity"):
        logger.error(
            f"The model data was balanced based on granularity : {model_log_file.get('balancing_over').get('granularity')}, whereas the test data was balanced based on granularity: {test_data_log_file.get('balancing_over').get('granularity')}")
        return "differing_balancing_granularities_found"

    if test_data_log_file.get("prepare_dataset").get("convolutional") != model_log_file.get("prepare_dataset").get("convolutional"):
        logger.error(
            f"The model data contains convolutional data: {model_log_file.get('prepare_dataset').get('convolutional')}, whereas the test data contains convolutional data: {test_data_log_file.get('prepare_dataset').get('convolutional')}")
        return "differing_convolutional_values_found"

    list_of_used_folders = [model_log_file.get("train_model").get(
        "training_data"), model_log_file.get("train_model").get("validation_data")]
    if test_data in list_of_used_folders:
        logger.error(
            f"Evaluating the model on data which was used for training is not allowed due to data leakage problems")
        return "training_folder_cannot_be_used_for_evaluation"

    return None


def load_trained_model(model_folder: str, log_file: dict):
    """returns the loaded model from the folder
    """
    network_type, error = detect_network_type(model_folder)
    if error:
        return None, error

    if os.path.exists( os.path.join(model_folder, "best_val_model.pt")):
        return train_model_utils.load_model(model_folder, network_type, log_file, "best_val_model.pt")

    return train_model_utils.load_model(model_folder, network_type, log_file)


def detect_network_type(model_folder: str):
    """reads in the log file of the given folder to return the seen model type
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)

    if "SVM" in model_folder:
        if model_log_file.get("train_model").get("type") != "SVM":
            logger.error(
                f"received a model in a SVM folder which is not a SVM : {model_log_file.get('train_model').get('type')}")
            return None, "found_non_svm_model_in_svm_folder"

        if model_log_file.get("train_model").get("hierarchical_model") not in ["top", "walking", "lifting"]:
            logger.error(
                f"received a SVM for separation of unknown classes: {model_log_file.get('train_model').get('hierarchical_model')}")
            return None, "found_svm_for_separation_of_unknown_classes"

        return "SVM", None

    if "balancing_over" not in model_log_file.keys():
        logger.error(
            f"Log file of model folder does not contain information regarding prior oversampling")
        return None, "no_oversampling_detected"

    if "prepare_dataset" not in model_log_file.keys():
        logger.error(
            f"Log file of model folder does not contain information regarding preparing the dataset before training")
        return None, "no_prepare_dataset_detected"

    if "generate_features" in model_log_file.keys() and "reduce_data" in model_log_file.keys():
        return "FFNN", None

    elif model_log_file.get("create_windows").get("flatten") and "generate_features" not in model_log_file.keys() and not model_log_file.get("prepare_dataset").get("convolutional"):
        return "FFNN", None

    elif model_log_file.get("create_windows").get("flatten") and "generate_features" not in model_log_file.keys() and model_log_file.get("prepare_dataset").get("convolutional") and model_log_file.get("train_model").get("type") == "CNN":
        return "CNN", None

    elif model_log_file.get("create_windows").get("flatten") and "generate_features" not in model_log_file.keys() and model_log_file.get("prepare_dataset").get("convolutional") and model_log_file.get("train_model").get("type") == "RNN":
        return "RNN", None

    logger.error(
        "Could not detect the correct model type based on the log file")
    return None, "detecting_model_type_ failed"


def create_predictions(model: nn.Module, test_data_folder: str, model_folder: str, store_local: bool, saving_folder: str, verbose: bool, frac=1.0):
    """does the heavy lifting for the evaluation, e.g. returns classification for each valid datapoint in the test data and returns a df with the predictions
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_folder_log_file = json.load(f)
    hierarchical_model = model_folder_log_file.get(
        "train_model").get("hierarchical_model")

    all_files, error = file_utils.get_files(test_data_folder, "csv")
    if error:
        return None, None, None, error

    all_files.remove(os.path.join(test_data_folder, "labels.csv"))

    all_files, error = train_model_utils.if_hierarchical_data_remove_unneeded_files(
        all_files, test_data_folder, hierarchical_model)
    if error:
        return None, None, None, error

    joined_filesize = file_utils.count_filesizes_of_list(all_files)
    file_utils.estimate_needed_time(
        len(all_files), "evaluate_model", os.cpu_count(), joined_filesize)

    data_in_channel_format = model_folder_log_file.get(
        "train_model").get("type") in ["CNN", "RNN"]
    if "RNN" in model_folder:
        model = model.double()

    processing_neural_network = model_folder_log_file.get(
        "train_model").get("type") in ["FFNN", "CNN", "RNN"]
    if processing_neural_network:
        model = model.eval()
    if not processing_neural_network:
        # As the baseline Method was fitted with featurenames, a Userwarning is being given for each window, therefore those are suppressed
        warnings.filterwarnings("ignore", category=UserWarning)

    hierarchical_data_handeled = "generate_features" in model_folder_log_file.keys()
    standardizing_df, standardization_type, error = data_utils.prepare_standardization(
        model_folder, hierarchical_data_handeled, hierarchical_model)
    if error:
        return None, None, None, error

    prediction_time = []

    for file_nr, file in enumerate(all_files):
        if (file_nr/len(all_files)) > frac:
            break

        if file_nr == 0 or file_nr % 2000 == 0:
            logger.info(f"[{file_nr}/{len(all_files)}] files classified")

        loader, y_transformed = create_loader_for_file(
            file, data_in_channel_format, hierarchical_data_handeled, standardizing_df, standardization_type, hierarchical_model)

        if file_nr == 0:
            all_true_labels = y_transformed
        else:
            all_true_labels = torch.cat((all_true_labels, y_transformed), 0)

        with torch.no_grad():
            for minibatch_nr, minibatch in enumerate(loader):

                start = time.time()
                y_pred = model.predict(minibatch[0])
                prediction_time.append(time.time()-start)

                if isinstance(y_pred, np.ndarray):
                    y_pred = transform_svm_output_to_valid_tensor(y_pred, 3)
                # transform the ground truth for this batch into an appropriate format for confusion matrix
                y_pred_transformed = train_model_utils.transform_predictions_for_confusion_matrix(
                    y_pred)
                if minibatch_nr == 0 and file_nr == 0:
                    y_pred_joined = y_pred_transformed
                else:
                    y_pred_joined = torch.cat(
                        (y_pred_joined, y_pred_transformed), 0)

    # Create the final confusion matrix of the data
    conf_mat = ConfusionMatrix(len(torch.unique(all_true_labels)))

    confusion_matrix = conf_mat(y_pred_joined, all_true_labels)

    accuracy = round((torch.diagonal(confusion_matrix, 0).sum(
    ).item()*100)/(torch.sum(confusion_matrix)).item(), 2)

    if verbose:
        logger.info(
            f"This is the Confusion Matrix of the test-data after Training: (n = {torch.sum(confusion_matrix).item()})\n{confusion_matrix}")
    logger.info(f"The final test-accuracy is {accuracy}%")

    plot_utils.plot_confusion_matrix(
        confusion_matrix, store_local, saving_folder, "test_data", hierarchical_model)

    results_df = pd.DataFrame(
        {"file": all_files, "true_label": all_true_labels, "prediction": y_pred_joined})

    # Reverse the label, so that the seen label in the file name has the same meaning as the label in the respective columns
    results_df["true_label"] = results_df["true_label"].to_numpy()+1
    results_df["prediction"] = results_df["prediction"].to_numpy()+1

    return confusion_matrix, accuracy, results_df, pd.Series(prediction_time), None


def create_loader_for_file(file: str, data_in_channel_format: bool, hierarchical_data_handeled: bool, standardizing_df: pd.DataFrame, standardization_type: str, hierarchical_model: str):
    """reads in a single file and returns a loader and y_transformed 
    """
    joined_df, label = train_model_utils.read_prepared_data_file(file)
    joined_df = joined_df.T

    if hierarchical_data_handeled:
        joined_df = data_utils.standardize(
            joined_df, standardization_type, standardizing_df)

    # create Tensors from the objects
    x = np.stack([joined_df[col].to_numpy()
                 for col in joined_df.columns], 1).astype(np.float64)
    x = torch.tensor(x, dtype=torch.float).reshape(1, -1)
    y = torch.LongTensor([label])

    # When data_in_channel_format is being processed, there might me a reshaping of x needed to bring it to [nr_windows(1),  nr_channels, resampled_timepoints_per_window]
    if data_in_channel_format:
        nr_channels = len(pd.read_csv(
            file, usecols=[0], dtype=np.float64, header=None))
        x = train_model_utils.reshape_tensor_to_first_identical_dim(
            x, y, nr_channels)

    Dataset = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    del x

    loader = DataLoader(Dataset, batch_size=1, shuffle=False)

    y_transformed = train_model_utils.bring_to_correct_format(y)
    del y

    return loader, y_transformed


def calculate_metrics(results_df: pd.DataFrame, model_folder: str):
    """calculates a df of performance metrics, as well as matthews correlation coeff and cohens kappa
    """
    if "true_label" not in results_df.columns:
        logger.error(
            f"Results_df does not contain a column with the true label, further metrics cannot be calculated")
        return pd.DataFrame(), 0.0, 0.0,  "true_label_col_not_found"

    if "prediction" not in results_df.columns:
        logger.error(
            f"Results_df does not contain a column with the prediction, further metrics cannot be calculated")
        return pd.DataFrame(), 0.0, 0.0, "prediction_col_not_found"

    metrics_df = calculate_precision_recall_f1(results_df, model_folder)
    logger.info(
        f"These are the resulting metrics for the the classes:\n\n{metrics_df.head(15)}")

    cohen_k_score = calculate_cohen_kappa_score(results_df)
    interpret_cohen_kappa_score(cohen_k_score)

    mathhews_corr_coeff = calculate_matthews_corr_coeff(results_df)
    interpret_matthews_corrcoef(mathhews_corr_coeff)

    return metrics_df, cohen_k_score, mathhews_corr_coeff, None


def calculate_precision_recall_f1(results_df: pd.DataFrame, model_folder: str):
    """returns a df containing the precision, recall and f1 score for every class, as well as the micro and macro avg per class
    """
    with open(os.path.join(model_folder, "log.json")) as f:
        model_log_file = json.load(f)
    hierarchical_model = model_log_file.get(
        "train_model").get("hierarchical_model")

    differences = list(
        set(results_df["true_label"]) - set(results_df["prediction"]))
    if len(differences) > 0:
        logger.warning(f"These labels were never predicted: {differences}")

    result_overview = classification_report(
        results_df["true_label"], results_df["prediction"], output_dict=True)
    overview_df = pd.DataFrame(result_overview)

    # Replace the numerical class names with the meaningful, human-readable ones
    label_names = plot_utils.get_class_names_from_len(
        len(results_df["true_label"].unique()), hierarchical_model)
    len_diff = len(overview_df.columns) - len(label_names)
    label_names.extend(overview_df.columns[-len_diff:])
    overview_df.columns = label_names
    overview_df = overview_df.T

    return overview_df


def calculate_cohen_kappa_score(df: pd.DataFrame):
    """ returns the cohen-kappa-score, see: https://en.wikipedia.org/wiki/Cohen%27s_kappa
    """
    return cohen_kappa_score(df["true_label"], df["prediction"])


def interpret_cohen_kappa_score(score: float):
    """ logs an interpretation of the cohen_kappa_score
    """

    # the interpretation of these values is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/

    interpretation = "not better than a random assignment of labels."
    if score > 0.2:
        interpretation = "minimal better than a random assignment of labels."
    if score > 0.4:
        interpretation = "weakly better than a random assignment of labels."
    if score > 0.6:
        interpretation = "moderately better than a random assignment of labels."
    if score > 0.8:
        interpretation = "strongly better than a random assignment of labels."
    if score > 0.9:
        interpretation = "almost perfect"

    logger.info(
        f"With a cohen-kappa-score of {round(score,2)} this score assigns that the model is {interpretation}")


def calculate_matthews_corr_coeff(df: pd.DataFrame):
    """ returns matthew's correlation coefficient
    """
    return matthews_corrcoef(df["true_label"], df["prediction"])


def interpret_matthews_corrcoef(coeff: float):
    """ logs an interpretation of matthews correlation coefficient (a discrete version of pearson's corr coeff)
    """

    # the interpretation of these values is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/

    interpretation = "are independent of the true label"
    if coeff > 0.2:
        interpretation = "correlate hardly with the true label."
    if coeff > 0.4:
        interpretation = "correlate weakly with the true label."
    if coeff > 0.6:
        interpretation = "correlate moderately with the true label."
    if coeff > 0.8:
        interpretation = "correlate strongly with the true label."
    if coeff > 0.9:
        interpretation = "correlate almost perfectly with the true label"

    logger.info(
        f"With a matthews correlation coefficient  of {round(coeff,2)} this coeff assigns that the model predictions {interpretation}")


def transform_svm_output_to_valid_tensor(input_ndarray: np.ndarray, classes_to_separate=3):
    """transforms the prediction of the SVM into the same format as the one used by the NN
    """
    prediction_tensor = torch.from_numpy(input_ndarray)

    # The NN return their predictions encoded in a one-hot encoder, therefore the format is adjusted
    return torch.nn.functional.one_hot(prediction_tensor.to(torch.int64), num_classes=classes_to_separate)


def valid_table_hierarachical(table:dict):
    """Checks if all the given parameters in the table are valid for the evaluation of hierarchical models
    """
    for classifier in ["top_level", "lifting", "walking"]:
        if not os.path.exists(table.get(f"{classifier}_folder")):
            logger.error(f"the given folder for the {classifier} classifier does not exist: {table.get(f'{classifier}_folder')}")
            return f"{classifier}_folder_not_found"
    
        log_file_loc = os.path.join(table.get(f"{classifier}_folder"), "log.json")
        if not os.path.exists(log_file_loc):
            logger.error(f"the given folder for the {classifier} classifier does not contain a log file: {table.get(f'{classifier}_folder')}")
            return f"{classifier}_folder_does_not_contain_a_log_file"
        
        # Test if the model within the folder was evaluated on data in the format of features
        with open(log_file_loc) as f:
            log_file: dict = json.load(f)
        if "evaluate_model_second_location" not in log_file.keys():
            logger.error(f"The folder for the given {classifier} classifier does not contain evaluated data")
            return "received _model_not_evaluated_prior"
        if "generate_features" not in log_file.get("evaluate_model_second_location").keys():
            logger.error(f"The given {classifier} classifier was not evaluated on data in the format of features")
            return "received _model_not_evaluated_on_featured_data"

        # test if the predictions.csv exists
        predictions_loc = os.path.join(table.get(f"{classifier}_folder"), "predictions.csv")
        if not os.path.exists(predictions_loc):
            logger.error(f"the given folder for the {classifier} classifier does not contain a predictions file: {table.get(f'{classifier}_folder')}")
            return f"{classifier}_folder_does_not_contain_a_predictions_file"
        predictions_df = pd.read_csv(predictions_loc, usecols=["true_label"])
        if len(predictions_df["true_label"].unique())>3:
            logger.error(f"Folder for {classifier} classifier contains more than three true labels")
            return "too_much_true_labels_within_predictions_df"
    
    if table.get("nr_obs_per_activity") <1:
        logger.error(f"The given number of observations per activity is too small")
        return "received _too_small_number_of_activities"
    if table.get("nr_obs_per_activity") <1000:
        logger.warning("received  small amount of observations for each activity, Using values > 1000 is recommended")

    if table.get("output_date") is not None:
        error = time_utils.valid_time_string(table.get("output_date"))
        if error:
            return error
    
    if len(set([table.get("top_level_folder"), table.get("lifting_folder"), table.get("walking_folder")])) <3:
        logger.error(f"The given folders are not three disctinct folders")
        return "received _identical_folders"

    return None


def create_dict_joined_prob(singular_probs:tuple):
    """create a confusion matrix containing the (conditional) probs for events: p(lifting predicted and true_label = lifting)
    """
    top_0_0, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, top_2_0, top_2_1, top_2_2, lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, walking_0_0, walking_0_1, walking_0_2, walking_1_0, walking_1_1, walking_1_2, walking_2_0, walking_2_1, walking_2_2 = singular_probs

    dicto = {}

    # Make all the cols for the true class lifting
    dicto["lifting_lifting"] = top_0_0 * lifting_0_0
    dicto["lifting_dropping"] = top_0_0 *lifting_0_1
    dicto["lifting_holding"] = top_0_0 * lifting_0_2
    dicto["lifting_walking\nstraight"] = top_0_1/9
    dicto["lifting_walking\nup\nstairs"] = top_0_1/9
    dicto["lifting_walking\ndown\nstairs"] = top_0_1/9
    dicto["lifting_resting"] = top_0_2 / 3

    # Make all the cols for the true class dropping
    dicto["dropping_lifting"] = top_0_0 * lifting_1_0
    dicto["dropping_dropping"] = top_0_0 *lifting_1_1
    dicto["dropping_holding"] = top_0_0 * lifting_1_2
    dicto["dropping_walking\nstraight"] = top_0_1/9
    dicto["dropping_walking\nup\nstairs"] = top_0_1/9
    dicto["dropping_walking\ndown\nstairs"] = top_0_1/9
    dicto["dropping_resting"] = top_0_2 / 3

    # Make all the cols for the true class holding
    dicto["holding_lifting"] = top_0_0 * lifting_2_0
    dicto["holding_dropping"] = top_0_0 *lifting_2_1
    dicto["holding_holding"] = top_0_0 * lifting_2_2
    dicto["holding_walking\nstraight"] = top_0_1/9
    dicto["holding_walking\nup\nstairs"] = top_0_1/9
    dicto["holding_walking\ndown\nstairs"] = top_0_1/9
    dicto["holding_resting"] = top_0_2 / 3


    # Make all the cols for the true class walking_normal
    dicto["walking\nstraight_lifting"] = top_1_0 /9
    dicto["walking\nstraight_dropping"] = top_1_0 /9
    dicto["walking\nstraight_holding"] = top_1_0 /9
    dicto["walking\nstraight_walking\nstraight"] = top_1_1 * walking_0_0
    dicto["walking\nstraight_walking\nup\nstairs"] = top_1_1 * walking_0_1
    dicto["walking\nstraight_walking\ndown\nstairs"] = top_1_1 * walking_0_2
    dicto["walking\nstraight_resting"] = top_1_2 / 3


    # Make all the cols for the true class walking_upstairs
    dicto["walking\nup\nstairs_lifting"] = top_1_0 /9
    dicto["walking\nup\nstairs_dropping"] = top_1_0 /9
    dicto["walking\nup\nstairs_holding"] = top_1_0 /9
    dicto["walking\nup\nstairs_walking\nstraight"] = top_1_1 * walking_1_0
    dicto["walking\nup\nstairs_walking\nup\nstairs"] = top_1_1 * walking_1_1
    dicto["walking\nup\nstairs_walking\ndown\nstairs"] = top_1_1 * walking_1_2
    dicto["walking\nup\nstairs_resting"] = top_1_2 / 3


    # Make all the cols for the true class walking_downstairs
    dicto["walking\ndown\nstairs_lifting"] = top_1_0 /9
    dicto["walking\ndown\nstairs_dropping"] = top_1_0 /9
    dicto["walking\ndown\nstairs_holding"] = top_1_0 /9
    dicto["walking\ndown\nstairs_walking\nstraight"] = top_1_1 * walking_2_0
    dicto["walking\ndown\nstairs_walking\nup\nstairs"] = top_1_1 * walking_2_1
    dicto["walking\ndown\nstairs_walking\ndown\nstairs"] = top_1_1 * walking_2_2
    dicto["walking\ndown\nstairs_resting"] = top_1_2 / 3


    # Make all the cols for the true class resting
    dicto["resting_lifting"] = top_2_0 /3
    dicto["resting_dropping"] = top_2_0 /3
    dicto["resting_holding"] = top_2_0 /3
    dicto["resting_walking\nstraight"] = top_2_1 /3
    dicto["resting_walking\nup\nstairs"] = top_2_1 /3
    dicto["resting_walking\ndown\nstairs"] = top_2_1 /3
    dicto["resting_resting"] = top_2_2
    
    sum_prob = sum([dicto[key] for key in dicto.keys()])
    if 0.999 > sum_prob or 1.001<sum_prob:
        logger.error(f"The sum of the probabilities {sum_prob} is not within acceptable boundaries")
        return {}, "incorrect_sum_of_prob"

    return dicto, None


def make_df_joined_prob(joined_prob_dict:dict, classes:list):
    """Creates a df containing the joined probabilities
    """
    if len(classes)<2:
        logger.error(f"received  a list of classes shorter than 2")
        return pd.DataFrame(), "classes_list_to_short"
    
    # Create a Df of zeros with col and index names
    df = pd.DataFrame()
    for i in classes:
        df[i]= [0]*len(classes)
    df.index = classes

    for true_label_nr, true_label in enumerate(df.columns):
        row = []
        for pred_nr, prediction in enumerate(df.index):
            row.append(joined_prob_dict[true_label + "_" + prediction])
        df.iloc[true_label_nr] = row
    
    return df, None


def correct_new_values(new_values:pd.Series, wanted_n_per_activity:int, probs:pd.Series):
    """If the amount of drawn values is too big / too small, adjust to correct number
    """
    probs = probs*(1/probs.sum())
    np.random.seed(42)

    if new_values.sum()<wanted_n_per_activity:
        while new_values.sum()<wanted_n_per_activity:

            drawn_value = np.random.multinomial(n = 1, pvals=probs)
            new_values =  pd.Series(new_values.to_numpy() + np.array(drawn_value)).astype("int16")
        return new_values

    if new_values.sum()>wanted_n_per_activity:
        while new_values.sum()>wanted_n_per_activity:

            drawn_value = np.random.multinomial(n = 1, pvals=probs)
            new_values =  pd.Series(new_values.to_numpy() - np.array(drawn_value)).astype("int16")
        return new_values

    return new_values


def create_pred_df(df_prob:pd.DataFrame, n:int):
    """Returns a Df containing the wanted number of observations per activity
    """
    df_prob = df_prob.T

    for col in df_prob.columns:
        new_values = round(df_prob[col]*n).astype("int16")
        if new_values.sum()!= n:
            new_values = correct_new_values(new_values, n, df_prob[col])
        df_prob[col] = new_values.to_numpy()

    for col in df_prob.columns:
        if df_prob[col].sum()!=n:
            logger.error(f"The col {col} does not contain the correct of obersavtions ({n}): {df_prob[col].sum()}")
            return pd.DataFrame(), "wrong_amount_of_observations_in_row"

    return df_prob.T, None


def create_joined_predictions(pred_df:pd.DataFrame, n:int):
    """returns the predictions df, i.e. a df containing two cols: predictions and true_label
    """
    predictions = []
    for true_label_nr, true_label in enumerate(pred_df.index):
        for pred_nr, pred_label in enumerate(pred_df.index):
            predictions.extend([pred_nr]*pred_df.iloc[true_label_nr,pred_nr])

    true_labels = []
    for true_label_nr, true_label in enumerate(pred_df.index):
        true_labels.extend([true_label_nr]*n)
    
    # Create the predictions df with the same label encoding as seen in evaluate_model
    predictions_df = pd.DataFrame({"true_label":true_labels,"prediction":predictions})
    predictions_df["prediction"] = predictions_df["prediction"].to_numpy()+1
    predictions_df["true_label"] = predictions_df["true_label"].to_numpy()+1

    return predictions_df


def create_probabilites(model_folder:str, classifier:str):
    """returns the probabilities for observing each event for the respective classifier, i.e. p(predicted =0 and true_label = 0). reweights the top-classifier
    """
    predictions_df = pd.read_csv(os.path.join(model_folder, "predictions.csv"), usecols=["true_label", "prediction"])
    # The entries of the top-level classifier get reweighted to account for lifting and walking being observed more often
    reweighting_factor = 3 if classifier =="top" else 1
    n = len(predictions_df)
    new_n = int(n/3*7) if classifier =="top" else n

    field_0_0 = len(predictions_df[(predictions_df["true_label"] == 1) & (predictions_df["prediction"] == 1)])*reweighting_factor /new_n
    field_0_1 = len(predictions_df[(predictions_df["true_label"] == 1) & (predictions_df["prediction"] == 2)])*reweighting_factor /new_n
    field_0_2 = len(predictions_df[(predictions_df["true_label"] == 1) & (predictions_df["prediction"] == 3)])*reweighting_factor /new_n
    field_1_0 = len(predictions_df[(predictions_df["true_label"] == 2) & (predictions_df["prediction"] == 1)])*reweighting_factor /new_n
    field_1_1 = len(predictions_df[(predictions_df["true_label"] == 2) & (predictions_df["prediction"] == 2)])*reweighting_factor /new_n
    field_1_2 = len(predictions_df[(predictions_df["true_label"] == 2) & (predictions_df["prediction"] == 3)])*reweighting_factor /new_n
    # The third row does contain the resting data for the top level daata, which does not need rewighting
    field_2_0 = len(predictions_df[(predictions_df["true_label"] == 3) & (predictions_df["prediction"] == 1)]) /new_n
    field_2_1 = len(predictions_df[(predictions_df["true_label"] == 3) & (predictions_df["prediction"] == 2)]) /new_n
    field_2_2 = len(predictions_df[(predictions_df["true_label"] == 3) & (predictions_df["prediction"] == 3)]) /new_n

    sum_prob = sum([field_0_0, field_0_1, field_0_2, field_1_0, field_1_1, field_1_2, field_2_0, field_2_1, field_2_2])

    if 0.999 > sum_prob or 1.001<sum_prob:
        logger.error(f"The sum of the probabilities {sum_prob} is not within acceptable boundaries")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, "incorrect_sum_of_prob"

    return field_0_0, field_0_1, field_0_2, field_1_0, field_1_1, field_1_2, field_2_0, field_2_1, field_2_2, None


def create_all_singular_probabilities(top_level_folder:str, lifting_folder:str, walking_folder:str) : 
    """returns a tuple of the probabilites of the observing the respectivbe field within the confusion matrix 
    """
    top_0_0, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, top_2_0, top_2_1, top_2_2, error = create_probabilites(top_level_folder, "top")
    if error:
        return (), error

    lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, error = create_probabilites(lifting_folder, "lifting")
    if error:
        return (), error
    
    walking_0_0, walking_0_1, walking_0_2, walking_1_0, walking_1_1, walking_1_2, walking_2_0, walking_2_1, walking_2_2, error = create_probabilites(walking_folder, "walking")
    if error:
        return (), error
    

    return (top_0_0, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, top_2_0, top_2_1, top_2_2, lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, walking_0_0, walking_0_1, walking_0_2, walking_1_0, walking_1_1, walking_1_2, walking_2_0, walking_2_1, walking_2_2), None


def create_df_prob(singular_probs:tuple, classes:list):
    """uses the singular probabailities to create a joined df
    """
    joined_prob_dict, error = create_dict_joined_prob(singular_probs)
    if error:
        return pd.DataFrame(), error
    
    return make_df_joined_prob(joined_prob_dict, classes)


def create_confusion_matrix(predictions_df:pd.DataFrame, store_local:bool, saving_folder:str, hierarchical_model:str, verbose:bool):
    """creates a joined confusion matrix from the predictions df
    """
    all_true_labels = torch.from_numpy(predictions_df["true_label"].to_numpy()-1)
    y_pred_joined = torch.from_numpy(predictions_df["prediction"].to_numpy()-1)

    # Create the final confusion matrix of the data
    conf_mat = ConfusionMatrix(len(torch.unique(all_true_labels)))

    confusion_matrix = conf_mat(y_pred_joined, all_true_labels)

    accuracy = round((torch.diagonal(confusion_matrix, 0).sum(
    ).item()*100)/(torch.sum(confusion_matrix)).item(), 2)

    if verbose:
        logger.info(
            f"This is the Confusion Matrix of the joined Model: (n = {torch.sum(confusion_matrix).item()})\n{confusion_matrix}")
    logger.info(f"The final Accuracy is {accuracy}%")

    plot_utils.plot_confusion_matrix(
        confusion_matrix, store_local, saving_folder, "test_data", hierarchical_model)

    return confusion_matrix, accuracy, None

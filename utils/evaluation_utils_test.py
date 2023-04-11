import unittest
import os
import sys
import shutil
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import evaluation_utils, file_utils, train_model_utils


class TestEvaluationUtils(unittest.TestCase):
    def test_valid_table(self):
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_valid_table"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        test_data_folder = os.path.join(folder, "test_data")
        model_folder = os.path.join(folder, "model_data")

        # test if the test folder does not exist
        got = evaluation_utils.valid_table({"test_data": test_data_folder})
        self.assertEqual(got, "invalid_test_data_given")

        os.makedirs(test_data_folder)

        # test if dryrun has an invalid datatype
        got = evaluation_utils.valid_table(
            {"test_data": test_data_folder, "dryrun": 14, "data_path": folder})
        self.assertEqual(got, "got_dryrun_of_nonbool_dtype")

        # Test if the model folder does not exist
        got = evaluation_utils.valid_table(
            {"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
        self.assertEqual(got, "invalid_model_folder_given")

        # Test if the model folder does not contain a log.json
        os.makedirs(model_folder)
        got = evaluation_utils.valid_table(
            {"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
        self.assertEqual(got, 'no_data_in_dir')
        pd.DataFrame({"some_col": [1, 2]}).to_csv(
            os.path.join(model_folder, "log.json"))
        pd.DataFrame({"some_col": [1, 2]}).to_csv(
            os.path.join(model_folder, "model.pt"))

        # test if the test data folder does not contain a log file
        got = evaluation_utils.valid_table(
            {"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
        self.assertEqual(got, 'no_data_in_dir')
        pd.DataFrame({"some_col": [1, 2]}).to_csv(
            os.path.join(test_data_folder, "log.json"))
        pd.DataFrame({"some_col": [1, 2]}).to_csv(
            os.path.join(test_data_folder, "file.csv"))
        pd.DataFrame({"some_col": [1, 2]}).to_csv(
            os.path.join(model_folder, "file.csv"))

        # test if both given folders are identical
        got = evaluation_utils.valid_table(
            {"test_data": model_folder, "dryrun": True, "model_folder": model_folder, "data_path": folder})
        self.assertEqual(got, "received_identical_model_and_data_folder")

        # Test if everything is valid
        got = evaluation_utils.valid_table(
            {"test_data": test_data_folder, "dryrun": True, "model_folder": model_folder, "plot": False, "data_path": folder})
        self.assertIsNone(got)

    def test_valid_preprocessing_of_data_for_model(self):
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_valid_preprocessing_of_data_for_model"
        if os.path.exists(folder):
            shutil.rmtree(folder)

        # test if the model folder does not exist
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            folder, "")
        self.assertEqual(got, "given_model_folder_does_not_exist")

        model_folder = os.path.join(folder, "model")
        os.makedirs(model_folder)

        # Test if the log file is missing
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, "")
        self.assertEqual(got, "given_model_folder_does_not_contain_log_file")
        error = file_utils.save_json(
            {}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)

        # test if the test folder does not exist
        test_data_folder = os.path.join(folder, "test_data")
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(got, "given_test_data_folder_does_not_exist")
        os.makedirs(test_data_folder)

        # Test if the log file is missing
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(
            got, "given_test_data_folder_does_not_contain_log_file")
        error = file_utils.save_json(
            {}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)

        # Test if the preprocessing steps differ
        error = file_utils.save_json({"create_windows": True, "balancing_over": True,
                                     "train_model": True}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)
        error = file_utils.save_json({"create_windows": {"window_length": 1000}, "generate_features": True,
                                     "balancing_over": True}, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(got, "preprocessing_steps_differ")

        # Test if the window_lengths differ
        error = file_utils.save_json({"create_windows": {"window_length": 400}, "generate_features": True,
                                     "balancing_over": True}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(got, "differing_window_lengths_found")

        # test if the flatten values differ
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True},
                                     "generate_features": True, "balancing_over": True}, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": False},
                                     "generate_features": True, "balancing_over": True}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(got, "differing_flatten_values_found")

        # test if the filling method differs
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"},
                                     "generate_features": True, "balancing_over": True}, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something_else"},
                                     "generate_features": True, "balancing_over": True}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(got, "differing_filling_methods_values_found")

        # test if the classification granularity differs
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"},
                                     "generate_features": True, "balancing_over": {"granularity": "top"}}, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"},
                                     "generate_features": True, "balancing_over": {"granularity": "mid"}}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertEqual(got, "differing_balancing_granularities_found")

        # Test when one of the sources contains data for CNNs
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {
                                     "granularity": "top"}, "prepare_dataset": {"convolutional": True}}, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {
                                     "granularity": "top"}, "prepare_dataset": {"convolutional": False}}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)

        # Test when everything is valid
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {"granularity": "top"}, "prepare_dataset": {
                                     "convolutional": True}, "train_model": {"training_data": "some_data", "validation_data": "some_other_data", }}, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        error = file_utils.save_json({"create_windows": {"window_length": 1000, "flatten": True, "method": "something"}, "generate_features": True, "balancing_over": {"granularity": "top"}, "prepare_dataset": {
                                     "convolutional": True}, "train_model": {"training_data": "some_data", "validation_data": "some_other_data", }}, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)
        got = evaluation_utils.valid_preprocessing_of_data_for_model(
            model_folder, test_data_folder)
        self.assertIsNone(got)

    def test_detect_network_type(self):
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_detect_network_type"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        # test correct detection for SVMs
        svm_folder = os.path.join(folder, "SVM")
        os.makedirs(svm_folder)

        # Test when a FFN Model lies wihin a folder for the SVMs
        error = file_utils.save_json(
            {"train_model": {"type": "FFNN"}}, os.path.join(svm_folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(svm_folder)
        self.assertEqual(got_error, "found_non_svm_model_in_svm_folder")

        # Test when an invalid hierarchical model was given
        error = file_utils.save_json({"train_model": {
                                     "type": "SVM", "hierarchical_model": "something_else"}}, os.path.join(svm_folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(svm_folder)
        self.assertEqual(
            got_error, "found_svm_for_separation_of_unknown_classes")

        # Test if everything works as intended for SVM
        error = file_utils.save_json({"train_model": {
                                     "type": "SVM", "hierarchical_model": "walking"}}, os.path.join(svm_folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(svm_folder)
        self.assertIsNone(got_error)
        self.assertEqual(got_type, "SVM")

        # No balancing for NN
        error = file_utils.save_json({"create_windows": {
                                     "flatten": True}, "train_model": True}, os.path.join(folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(folder)
        self.assertEqual(got_error, "no_oversampling_detected")

        # Not the correct format
        error = file_utils.save_json({"create_windows": True, "balancing_over": True,
                                     "train_model": True, "balancing_over": True}, os.path.join(folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(folder)
        self.assertEqual(got_error, "no_prepare_dataset_detected")

        # FFNN
        error = file_utils.save_json({"create_windows": True, "balancing_over": True, "train_model": True,
                                     "prepare_dataset": True, "generate_features": True, "reduce_data": True}, os.path.join(folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(folder)
        self.assertIsNone(got_error)
        self.assertEqual(got_type, "FFNN")

        error = file_utils.save_json({"create_windows": {"flatten": True}, "balancing_over": True, "train_model": True, "prepare_dataset": {
                                     "convolutional": False}}, os.path.join(folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(folder)
        self.assertIsNone(got_error)
        self.assertEqual(got_type, "FFNN")

        # CNN
        error = file_utils.save_json({"create_windows": {"flatten": True}, "balancing_over": True, "train_model": {
                                     "type": "CNN"}, "prepare_dataset": {"convolutional": True}}, os.path.join(folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(folder)
        self.assertIsNone(got_error)
        self.assertEqual(got_type, "CNN")

        # cannot be detected
        error = file_utils.save_json({"create_windows": {"flatten": False}, "balancing_over": True, "train_model": {
                                     "type": "RNN"}, "prepare_dataset": {"convolutional": False}}, os.path.join(folder, "log.json"))
        self.assertIsNone(error)
        got_type, got_error = evaluation_utils.detect_network_type(folder)
        self.assertIsNone(got_type)
        self.assertEqual(got_error, "detecting_model_type_ failed")

    def test_load_trained_model(self):
        # test if everything works as intended
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_load_trained_model"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        log_file = {"train_model": {"seed": 42, "batch_normalization": True, "n_features": 15, "out_szs": 4, "dropout_rate": 0.1, "layer_structure": "80f30"},
                    "generate_features": True, "reduce_data": True, "balancing_over": True, "prepare_dataset": True}
        error = file_utils.save_json(
            log_file, os.path.join(folder, "log.json"))
        self.assertIsNone(error)

        # Save a model in the folder be loaded
        layers, error = train_model_utils.build_layers_from_string(
            log_file.get("train_model").get("layer_structure"), "FFNN")
        self.assertIsNone(error)
        Model = train_model_utils.build_nn_model("FFNN", log_file.get("train_model").get(
            "batch_normalization"), log_file.get("train_model").get("seed"), log_file)
        model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get(
            "out_szs"), layers, log_file.get("train_model").get("dropout_rate"), log_file.get("train_model").get("batch_normalization"))

        error = train_model_utils.save_model(model, folder)
        self.assertIsNone(error)

        got_model, got_error = evaluation_utils.load_trained_model(
            folder, log_file)
        self.assertIsNone(got_error)
        self.assertEqual(str(type(
            got_model)), "<class 'utils.train_model_utils.create_ffnn_model.<locals>.FeedforwardNetwork'>")

    def test_create_loader_for_file(self):
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_create_loader_for_file"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        # Test if everything works as intended
        pd.DataFrame([[1.256, 4.365125, 2.45788]]).to_csv(
            os.path.join(folder, "11_some_identifier.csv"), index=False)
        got_loader, got_y_transformed = evaluation_utils.create_loader_for_file(os.path.join(
            folder, "11_some_identifier.csv"), False, False, pd.DataFrame(), None, "top")
        self.assertTrue(torch.equal(
            got_y_transformed, torch.zeros((1)).long()))
        self.assertTrue(isinstance(got_loader, DataLoader))

    def test_create_predictions(self):
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_create_predictions"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        model_folder = os.path.join(folder, "model_folder")
        os.makedirs(model_folder)

        log_file = {"train_model": {"seed": 42, "batch_normalization": False, "n_features": 3, "out_szs": 4, "dropout_rate": 0.1, "layer_structure": "10", "hierarchical_model": "top", "type": "FFNN"},
                    "generate_features": True, "reduce_data": True, "balancing_over": {"label_depth": 2}, "prepare_dataset": True}
        error = file_utils.save_json(
            log_file, os.path.join(model_folder, "log.json"))
        self.assertIsNone(error)

        test_data_folder = os.path.join(folder, "test_data")
        os.makedirs(test_data_folder)
        error = file_utils.save_json(
            log_file, os.path.join(test_data_folder, "log.json"))
        self.assertIsNone(error)
        # create the training data
        pd.DataFrame([[1.648], [6.245], [3.1455]]).to_csv(os.path.join(
            test_data_folder, "1_some_ident.csv"), index=False, header=False)
        pd.DataFrame([[0.648], [6.245], [3.1455]]).to_csv(os.path.join(
            test_data_folder, "2_some_ident.csv"), index=False, header=False)

        pd.DataFrame([[1.648], [6.245], [3.1455]]).to_csv(
            os.path.join(test_data_folder, "labels.csv"))
        # create a file for the standardization
        pd.DataFrame({"mean": [0, 0, 0], "scale": [1, 1, 1], "var": [1, 1, 1]}).T.to_csv(
            os.path.join(model_folder, "standardization_std.csv"))

        layers, error = train_model_utils.build_layers_from_string(
            log_file.get("train_model").get("layer_structure"), "FFNN")
        self.assertIsNone(error)
        Model = train_model_utils.build_nn_model("FFNN", log_file.get("train_model").get(
            "batch_normalization"), log_file.get("train_model").get("seed"), log_file)
        model = Model(log_file.get("train_model").get("n_features"), log_file.get("train_model").get("out_szs"), layers, log_file.get("train_model").get("dropout_rate"),
                      log_file.get("train_model").get("batch_normalization"))
        error = train_model_utils.save_model(model, model_folder)
        self.assertIsNone(error)
        model = model.eval()

        got_confusion_matrix, got_accuracy, got_results_df, prediction_time, error = evaluation_utils.create_predictions(
            model, test_data_folder, model_folder, True, folder, False, 1.)
        self.assertIsNone(error)

        want_df = pd.DataFrame({"file": [os.path.join(test_data_folder, "1_some_ident.csv"), os.path.join(
            test_data_folder, "2_some_ident.csv")], "true_label": [1, 2], "prediction": [1, 1]})
        want_df["prediction"] = want_df["prediction"].astype(np.int32)
        if "win" in sys.platform:
            want_df["file"] = [file.replace("\\","/") for file in want_df["file"]]

        assert_frame_equal(got_results_df, want_df)
        self.assertEqual(got_accuracy, 50.)
        self.assertEqual(str(type(got_confusion_matrix)),
                         "<class 'torch.Tensor'>")

    def test_calculate_precision_recall_f1(self):
        # Test if everything works as intended
        folder = "/tmp/trainings_pipeline/evaluation_utils_test/test_calculate_precision_recall_f1"
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)

        log_file = {"train_model": {"hierarchical_model": "top"}}
        error = file_utils.save_json(
            log_file, os.path.join(folder, "log.json"))
        self.assertIsNone(error)

        results_df = pd.DataFrame(
            {"true_label": [0, 0, 1, 1, 2, 2], "prediction": [0, 0, 1, 2, 1, 1]})
        got = evaluation_utils.calculate_precision_recall_f1(
            results_df, folder)

        want = pd.DataFrame({"precision": [1.0, 1/3, 0., 0.5, 0.444444444444, 0.444444444],
                             "recall": [1.0, 0.5, 0., 0.5, 0.5, 0.5],
                             "f1-score": [1.0, 0.4, 0., 0.5, 0.466667, 0.466667],
                             "support": [2.0, 2.0, 2., 0.5, 6.0, 6.0], })
        want.index = ['lifting', 'walking', 'resting',
                      'accuracy', 'macro avg', 'weighted avg']

        assert_frame_equal(got, want)

    def test_calculate_cohen_kappa_score(self):
        # Test if everything works as intended
        got = evaluation_utils.calculate_cohen_kappa_score(
            pd.DataFrame({"true_label": [0, 1, 2], "prediction": [2, 1, 0]}))
        self.assertEqual(got, 0.0)

    def test_calculate_matthews_corr_coeff(self):
        # Test if everything works as intended
        got = evaluation_utils.calculate_matthews_corr_coeff(
            pd.DataFrame({"true_label": [0, 1, 2], "prediction": [2, 1, 0]}))
        self.assertEqual(got, 0.0)

    def test_calculate_metrics(self):
        # Test if the true_label col is missing
        got_df, got_cohen, got_matthew, got_error = evaluation_utils.calculate_metrics(
            pd.DataFrame({"wrong_label": [0, 1, 2], "prediction": [2, 1, 0]}), "somewhere")
        self.assertTrue(got_df.empty)

        # Test if the prediction col is missing
        got_df, got_cohen, got_matthew, got_error = evaluation_utils.calculate_metrics(
            pd.DataFrame({"true_label": [0, 1, 2], "guesses": [2, 1, 0]}), "somewhere")
        self.assertTrue(got_df.empty)

    def test_transform_svm_output_to_valid_tensor(self):
        input_ndarray = np.arange(1)
        want = torch.tensor([[1, 0, 0]])
        got = evaluation_utils.transform_svm_output_to_valid_tensor(
            input_ndarray, 3)
        self.assertTrue(torch.equal(got, want))

    # new funstions
    def test_valid_table_hierarachical(self):
        # test if one of the given folders does not exist
        non_existing_folder = "/some_non_existing_folder/non_existing_subfolder"
        if os.path.exists(non_existing_folder):
            shutil.rmtree(non_existing_folder)
        
        table = {"top_level_folder":non_existing_folder}
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "top_level_folder_not_found")

        # Tets if the folder does no contain a log file
        os.makedirs(non_existing_folder)
        existing_folder = non_existing_folder
        table = {"top_level_folder":existing_folder}
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "top_level_folder_does_not_contain_a_log_file")

        # test if the log file belongs to a folder which is not an evaluation folder
        log_file_loc = os.path.join(existing_folder, "log.json")
        error = file_utils.save_json({"some_preprocessing":{"some_param":12}}, log_file_loc)
        self.assertIsNone(error)
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "received _model_not_evaluated_prior")

        # Test if the log file belongs to a model not using featured data
        error = file_utils.save_json({"evaluate_model_second_location":{"some_param":12}}, log_file_loc)
        self.assertIsNone(error)
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "received _model_not_evaluated_on_featured_data")

        # Test if the folder does not contain a predictions.csv
        error = file_utils.save_json({"evaluate_model_second_location":{"generate_features":12}}, log_file_loc)
        self.assertIsNone(error)
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "top_level_folder_does_not_contain_a_predictions_file")

        # test if the predictions_csv contains too much activity levels
        pd.DataFrame({"true_label":[x for x in range(10)]}).to_csv(os.path.join(existing_folder, "predictions.csv"))
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "too_much_true_labels_within_predictions_df")

        # test if nr_obs is too small
        pd.DataFrame({"true_label":[x for x in range(2)]}).to_csv(os.path.join(existing_folder, "predictions.csv"))
        table = {"top_level_folder":existing_folder, "lifting_folder":existing_folder, "walking_folder":existing_folder, "nr_obs_per_activity":-100}
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "received _too_small_number_of_activities")

        # Tets if the given folders are ikdentical
        table = {"top_level_folder":existing_folder, "lifting_folder":existing_folder, "walking_folder":existing_folder, "nr_obs_per_activity":10000}
        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertEqual(got, "received _identical_folders")

        # test if everything works as intended
        existing_folder_2 = "/some_non_existing_folder/existing_subfolder_2"
        if not os.path.exists(existing_folder_2):
            os.makedirs(existing_folder_2)
        pd.DataFrame({"true_label":[x for x in range(2)]}).to_csv(os.path.join(existing_folder_2, "predictions.csv"))
        log_file_loc = os.path.join(existing_folder_2, "log.json")
        error = file_utils.save_json({"evaluate_model_second_location":{"generate_features":12}}, log_file_loc)
        self.assertIsNone(error)
    
        existing_folder_3 = "/some_non_existing_folder/existing_subfolder_3"
        if not os.path.exists(existing_folder_3):
            os.makedirs(existing_folder_3)
        pd.DataFrame({"true_label":[x for x in range(2)]}).to_csv(os.path.join(existing_folder_3, "predictions.csv"))
        log_file_loc = os.path.join(existing_folder_3, "log.json")
        error = file_utils.save_json({"evaluate_model_second_location":{"generate_features":12}}, log_file_loc)
        self.assertIsNone(error)

        table = {"top_level_folder":non_existing_folder, "lifting_folder":existing_folder_2, "walking_folder":existing_folder_3, "nr_obs_per_activity":10000}

        got = evaluation_utils.valid_table_hierarachical(table)
        self.assertIsNone(got)
        for folder in [non_existing_folder, existing_folder, existing_folder_2, existing_folder_3]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            
    def test_create_dict_joined_prob(self):
        # Test if everything works as intended
        top_0_0, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, top_2_0, top_2_1, top_2_2 = (1/9 for x in range(9))
        lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2  = (1/9 for x in range(9))
        walking_0_0, walking_0_1, walking_0_2, walking_1_0, walking_1_1, walking_1_2, walking_2_0, walking_2_1, walking_2_2 = (1/9 for x in range(9))

        singular_probs = top_0_0, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, top_2_0, top_2_1, top_2_2, lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, walking_0_0, walking_0_1, walking_0_2, walking_1_0, walking_1_1, walking_1_2, walking_2_0, walking_2_1, walking_2_2

        got_dicto, error = evaluation_utils.create_dict_joined_prob(singular_probs)
        self.assertIsNone(error)
        self.assertEqual(got_dicto["resting_resting"], 1/9)
        self.assertEqual(got_dicto["lifting_lifting"], 1/81)

        # Test if the proibabilities are wrong
        singular_probs = 250, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, 190, top_2_1, top_2_2, lifting_0_0, lifting_0_1, lifting_0_2, 3000, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, walking_0_0, walking_0_1, walking_0_2, walking_1_0, walking_1_1, walking_1_2, walking_2_0, walking_2_1, walking_2_2
        got_dicto, error = evaluation_utils.create_dict_joined_prob(singular_probs)
        self.assertEqual(error, "incorrect_sum_of_prob")

    def test_make_df_joined_prob(self):
        # Test fi the classes given are invalid
        joined_prob_dict = {"one_one":0.25,"one_two":0.1,"two_one":0.2, "two_two":0.45}
        classes = []
        got, error = evaluation_utils.make_df_joined_prob(joined_prob_dict, classes)
        self.assertEqual(error, "classes_list_to_short")

        # Test if everything works as intended
        joined_prob_dict = {"one_one":0.25,"one_two":0.1,"two_one":0.2, "two_two":0.45}
        classes = ["one", "two"]
        got, error = evaluation_utils.make_df_joined_prob(joined_prob_dict, classes)
        self.assertIsNone(error)
        want = pd.DataFrame({"one":[0.25,0.2], "two":[0.1,0.45]})
        want.index = classes
        assert_frame_equal(got, want)
    
    def test_correct_new_values(self):
        # Test if the amount of n_per_activity is too small
        new_values = pd.Series([10,20,30])
        wanted_n_per_activity = 61
        probs = pd.Series([0.1,0.2,0.3])
        got = evaluation_utils.correct_new_values(new_values, wanted_n_per_activity, probs)
        want = [10,21,30]
        self.assertListEqual([x for x in got], want)

        # Test if the amount of n_per_activity is too big
        new_values = pd.Series([10,20,30])
        wanted_n_per_activity = 59
        probs = pd.Series([0.1,0.2,0.3])
        got = evaluation_utils.correct_new_values(new_values, wanted_n_per_activity, probs)
        want = [10,19,30]
        self.assertListEqual([x for x in got], want)

        # Test if the new_values are correct
        new_values = pd.Series([10,20,30])
        wanted_n_per_activity = 60
        probs = pd.Series([0.1,0.2,0.3])
        got = evaluation_utils.correct_new_values(new_values, wanted_n_per_activity, probs)
        want = [10,20,30]
        self.assertListEqual([x for x in got], want)
    
    def test_create_pred_df(self):
        # Test if everything is valid
        df_prob = pd.DataFrame({"one":[0.2,0.1], "two":[0.1,0.6]})
        n = 100
        got, error = evaluation_utils.create_pred_df(df_prob, n)
        self.assertIsNone(error)
        want = pd.DataFrame({"one":[70,13], "two":[30,87]}).astype("int16")
        assert_frame_equal(got, want)
    
    def test_create_joined_predictions(self):
        # test if everything works as intended
        pred_df = pd.DataFrame({"one":[70,13], "two":[30,87]}).astype("int16")
        pred_df.index = pred_df.columns
        n = 100
        got = evaluation_utils.create_joined_predictions(pred_df, n)
    
        self.assertListEqual([x for x in got["true_label"].value_counts()], [n,n])
        self.assertListEqual([x for x in got["prediction"].value_counts()], [117,83])

    def test_create_probabilites(self):
        folder = "/evaluation_utils_test/test_create_probabilites"
        if not os.path.exists(folder):
            os.makedirs(folder)

        pd.DataFrame({"true_label":[1,1,1,2,2,2,3,3,3],"prediction":[1,1,1,2,3,3,1,2,3]}).to_csv(os.path.join(folder, "predictions.csv"))
        # Test if the data comes from lifting / walking
        field_0_0, field_0_1, field_0_2, field_1_0, field_1_1, field_1_2, field_2_0, field_2_1, field_2_2, error = evaluation_utils.create_probabilites(folder, "lifting")
        self.assertIsNone(error)

        self.assertEqual(field_0_0, 1/3)
        self.assertEqual(field_0_1, 0)
        self.assertEqual(field_0_2, 0)
        self.assertEqual(field_1_0, 0)
        self.assertEqual(field_1_1, 1/9)
        self.assertEqual(field_1_2, 2/9)
        self.assertEqual(field_2_0, 1/9)
        self.assertEqual(field_2_1, 1/9)
        self.assertEqual(field_2_2, 1/9)

        # test if the data comes from tp_level classifier
        field_0_0, field_0_1, field_0_2, field_1_0, field_1_1, field_1_2, field_2_0, field_2_1, field_2_2, error = evaluation_utils.create_probabilites(folder, "top")
        self.assertIsNone(error)

        self.assertEqual(field_0_0, 3/7)
        self.assertEqual(field_0_1, 0)
        self.assertEqual(field_0_2, 0)
        self.assertEqual(field_1_0, 0)
        self.assertEqual(field_1_1, 1/7)
        self.assertEqual(field_1_2, 2/7)
        self.assertEqual(field_2_0, 1/21)
        self.assertEqual(field_2_1, 1/21)
        self.assertEqual(field_2_2, 1/21)

        shutil.rmtree(folder)

    def test_create_all_singular_probabilities(self):
        # Test if everything works as intended
        folder = "/evaluation_utils_test/test_create_all_singular_probabilities"
        if not os.path.exists(folder):
            os.makedirs(folder)

        pd.DataFrame({"true_label":[1,1,1,2,2,2,3,3,3],"prediction":[1,1,1,2,3,3,1,2,3]}).to_csv(os.path.join(folder, "predictions.csv"))
        got_tuple, got_error = evaluation_utils.create_all_singular_probabilities(folder, folder, folder)
        self.assertIsNone(got_error)


        top_0_0 =  3/7
        top_0_1 =  0
        top_0_2 =  0
        top_1_0 =  0
        top_1_1 =  1/7
        top_1_2 =  2/7
        top_2_0 =  1/21
        top_2_1 =  1/21
        top_2_2 =  1/21

        lifting_0_0 =  1/3
        lifting_0_1 =  0
        lifting_0_2 =  0
        lifting_1_0 =  0
        lifting_1_1 =  1/9
        lifting_1_2 =  2/9
        lifting_2_0 =  1/9
        lifting_2_1 =  1/9
        lifting_2_2 =  1/9

        want_tuple = top_0_0, top_0_1, top_0_2, top_1_0, top_1_1, top_1_2, top_2_0, top_2_1, top_2_2, lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, lifting_0_0, lifting_0_1, lifting_0_2, lifting_1_0, lifting_1_1, lifting_1_2, lifting_2_0, lifting_2_1, lifting_2_2, 
        self.assertTupleEqual(got_tuple, want_tuple)
    
    def test_create_confusion_matrix(self):
        # Test if everything works as intended
        predictions_df = pd.DataFrame({"true_label":[1,1,2,2], "prediction":[1,1,1,2]})
        got_matrix, got_acc, got_error = evaluation_utils.create_confusion_matrix(predictions_df, False, "", "top", False)
        self.assertIsNone(got_error)
        self.assertEqual(got_acc, 75.0)
        self.assertEqual(str(type(got_matrix)), "<class 'torch.Tensor'>")
        self.assertEqual(got_matrix[0,0], 2)
        self.assertEqual(got_matrix[1,1], 1)
        self.assertEqual(got_matrix[0,1], 0)
        self.assertEqual(got_matrix[1,0], 1)


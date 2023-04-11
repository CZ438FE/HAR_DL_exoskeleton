import unittest
import os
import shutil
import torch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils import plot_utils, train_model_utils, file_utils


class TestPlotUtils(unittest.TestCase):
    def test_return_dict_nr_to_label(self):
        # Test if an an invalid level was given
        invalid_level = "invalid_level"
        got_dict, got_error = plot_utils.return_dict_nr_to_label(invalid_level)
        self.assertDictEqual(got_dict, {})
        self.assertEqual(got_error,  "invalid_granularity_given")

    def test_change_label_nr_to_human_readable(self):
        # Test if {} gets returned when no valid dict containing data gets inserted
        invalid_dict = {}
        valid_dict_nr_to_label = {
            1: "activity one", 2: "activity two", 4: "activity four", 5: "activity five"}
        got = plot_utils.change_label_nr_to_human_readable(
            invalid_dict, valid_dict_nr_to_label)
        self.assertDictEqual(got, {})
        # Test if {} gets returned the dict containing the mapping information is empty
        valid_dict = {4: 200, 2: 250, 5: 300, 1: 800}
        invalid_dict_nr_to_label = {}
        got = plot_utils.change_label_nr_to_human_readable(
            valid_dict, invalid_dict_nr_to_label)
        self.assertDictEqual(got, {})
        # Test if the dict containing the mapping information is empty
        invalid_dict_nr_to_label = {}
        got = plot_utils.change_label_nr_to_human_readable(
            valid_dict, invalid_dict_nr_to_label)
        self.assertDictEqual(got, {})
        # Test if the conversion works with valid inputs
        want = {"activity\none": 800, "activity\ntwo": 250,
                "activity\nfour": 200, "activity\nfive": 300, }
        got = plot_utils.change_label_nr_to_human_readable(
            valid_dict, valid_dict_nr_to_label)
        self.assertDictEqual(got, want)

    def test_show_label_distribution(self):
        # test a wrong granularity is given
        invalid_granularity = "invalid_gran"
        valid_labels_distribution = {"1": 20, "2": 25, "3": 34, "4": 0}
        got = plot_utils.show_label_distribution(
            invalid_granularity, valid_labels_distribution, True)
        self.assertIsNone(got)
        # Test if no labels_distribution_dict was given
        got = plot_utils.show_label_distribution("mid", {}, True)
        self.assertIsNone(got)
        # Test if everything works as intended
        got = plot_utils.show_label_distribution(
            "top", valid_labels_distribution, True)

    def test_infer_level(self):
        # Test a label depth of 3
        got_level, got_error = plot_utils.infer_level("211")
        self.assertEqual(got_level, "low")
        self.assertIsNone(got_error)

        # Test a label depth of 2
        got_level, got_error = plot_utils.infer_level("21")
        self.assertEqual(got_level, "mid")
        self.assertIsNone(got_error)

        # Test a label depth of 1
        got_level, got_error = plot_utils.infer_level("2")
        self.assertEqual(got_level, "top")
        self.assertIsNone(got_error)

        # Test if it finds the correct depth with a list of depth 3
        got_level, got_error = plot_utils.infer_level(["211", "15", "31"])
        self.assertEqual(got_level, "low")
        self.assertIsNone(got_error)

        # Test if it finds the correct depth with a list of depth 2
        got_level, got_error = plot_utils.infer_level(["21", "15", "31"])
        self.assertEqual(got_level, "mid")
        self.assertIsNone(got_error)

        # Test if it finds the correct depth with a list of depth 1
        got_level, got_error = plot_utils.infer_level(["1", "3", "3"])
        self.assertEqual(got_level, "top")
        self.assertIsNone(got_error)

        # test if an invalid depth is given
        got_level, got_error = plot_utils.infer_level("251651651")
        self.assertEqual(got_error, "level_not_defined_for_maxlen_9")

        # test if an invalid datatype is given
        got_level, got_error = plot_utils.infer_level(
            pd.DataFrame({"251651651": [1, 1, 5]}))
        self.assertEqual(got_error, "invalid_dtype_given_for_inferring_level")

    def test_replace_list_labels_with_human_readable_form(self):
        # Test if it correctly replaces with a depth of "mid"
        list_of_labels_mid = [11, 31, 24]
        got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(
            list_of_labels_mid)
        want = ["Lifting", "Resting", "Walking Sidesteps"]
        self.assertListEqual(got_levels, want)

        # Test if it correctly replaces with a depth of "top"
        list_of_labels_mid = [2, 3, 1]
        got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(
            list_of_labels_mid)
        want = ["Walking", "Resting", "Lifting"]
        self.assertListEqual(got_levels, want)

        # Test if it correctly replaces with a depth of "low"
        list_of_labels_mid = [221, 32, 15, 222]
        got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(
            list_of_labels_mid)
        want = ["Upstairs Free", "Sitting", "Holding", "Upstairs Carrying"]
        self.assertListEqual(got_levels, want)

        # Test if an empty list gets returned when an invalid label type is given
        got_levels, got_error = plot_utils.replace_list_labels_with_human_readable_form(
            pd.DataFrame())
        want = []
        self.assertListEqual(got_levels, want)

    def test_plot_original_samples_on_data_after_resampling(self):
        saving_folder = "/tmp/trainings_pipeline/plot_utils_test/test_plot_original_samples_on_data_after_resampling"
        if os.path.exists(saving_folder):
            shutil.rmtree(saving_folder)
        os.makedirs(saving_folder)

        # Test if None is returned when creating the list of human-readable levels fails
        invalid_prop_original_data_dict = {
            "class": pd.DataFrame(), "prop_existing_after_balancing": [None]}
        got = plot_utils.plot_original_samples_on_data_after_resampling(
            invalid_prop_original_data_dict, saving_folder, False)
        self.assertEqual(got, "invalid_dtype_given_for_inferring_level")

        # Test if None is returned when the given dict is malformed
        invalid_prop_original_data_dict = {"class": pd.DataFrame()}
        got = plot_utils.plot_original_samples_on_data_after_resampling(
            invalid_prop_original_data_dict, saving_folder, False)
        self.assertEqual(got, "malformed_prop_original_data_dict_given")

        # test if it correctly displays when everything is valid
        valid_prop_original_data_dict = {
            "class": [1, 2, 3], "prop_existing_after_balancing": [0.8, 0.7, 0.95]}
        got = plot_utils.plot_original_samples_on_data_after_resampling(
            valid_prop_original_data_dict, saving_folder, True)
        self.assertIsNone(got)
        self.assertTrue(os.path.exists(os.path.join(
            saving_folder, "proportion_of_original_data_on_data_after_resampling.png")))

    def test_plot_accuracies_and_loss_over_batches(self):
        # test if False gets returned, when the input is empty
        got = plot_utils.plot_accuracies_and_loss_over_batches(
            [], [], [], [], False, "", 100)
        self.assertFalse(got)

        # test if False gets returned, when the input is invalid
        got = plot_utils.plot_accuracies_and_loss_over_batches(
            [200, 200, 200], [], [], [], False, "", 100)
        self.assertTrue(got)

        # test if true gets returned, when the input has errors, which may be fixed
        got = plot_utils.plot_accuracies_and_loss_over_batches(
            [0.1, 200, 200], [], [], [], False, "", 100)
        self.assertTrue(got)

        # test if true gets returned, when the input has errors, which may be fixed
        got = plot_utils.plot_accuracies_and_loss_over_batches(
            [], [], [0.1, 200, 200], [], False, "", 100)
        self.assertTrue(got)

        # Test if the output gets saved correctly
        saving_folder = "/tmp/trainings_pipeline/plot_utils_test/test_plot_accuracies_and_loss_over_batches"
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        if os.path.exists(os.path.join(saving_folder, "losses.png")):
            os.remove(os.path.join(saving_folder, "losses.png"))
        got = plot_utils.plot_accuracies_and_loss_over_batches(
            [], [], [0.1, 200, 200], [], True, saving_folder, 100)
        self.assertTrue(got)
        exists = os.path.exists(os.path.join(
            saving_folder, "Training_minibatch.png"))
        self.assertTrue(exists)

    def test_get_class_names_from_len(self):
        # test if a strange len was given
        got = plot_utils.get_class_names_from_len(8, "top")
        self.assertListEqual(got, [x for x in range(8)])

        # Test if the lne is 3
        got = plot_utils.get_class_names_from_len(3, "top")
        self.assertListEqual(got, ["lifting", "walking", "resting"])

        # Test if the len is 7
        got = plot_utils.get_class_names_from_len(7, "top")
        self.assertListEqual(got, ["lifting", "dropping", "holding", "walking\nstraight",
                             "walking\nupstairs", "walking\ndownstairs", "resting"])

    def test_plot_confusion_matrix(self):
        # test if everything works as intended
        minibatch_size = 1
        dataset = "test_set"
        folder = "/tmp/trainings_pipeline/plot_utils_test/test_plot_confusion_matrix"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        file_names = [os.path.join(folder, "1_some_file.csv"), os.path.join(folder, "1_some_file_1.csv"), os.path.join(
            folder, "2_some_file.csv"), os.path.join(folder, "2_some_file_1.csv")]
        pd.DataFrame({0: [1]}).to_csv(file_names[0], index=False, header=False)
        pd.DataFrame({0: [1]}).to_csv(file_names[1], index=False, header=False)
        pd.DataFrame({0: [2]}).to_csv(file_names[2], index=False, header=False)
        pd.DataFrame({0: [2]}).to_csv(file_names[3], index=False, header=False)
        pd.DataFrame({"filename": file_names, "label": [1, 1, 2, 2]}).to_csv(
            os.path.join(folder, "labels.csv"), index=False)

        log_file = {"balancing_over": {"label_depth": 2}}
        error = file_utils.save_json(
            log_file, os.path.join(folder, "log.json"))
        self.assertIsNone(error)

        loader, error = train_model_utils.create_loader(
            folder, minibatch_size, dataset, 1.0, "top")
        self.assertIsNone(error)
        FFN_Model = train_model_utils.create_ffnn_model(False, 42)
        model = FFN_Model(1, 2, [1], 0.0, False)
        n_features = 1
        got_confusion_matr, got_accuracy, got_y_pred_joined = train_model_utils.display_confusion_matrix(
            loader, model, dataset, True, folder)
        self.assertTrue(torch.equal(got_confusion_matr,
                        torch.tensor([[0, 2], [0, 2]], dtype=torch.long)))
        self.assertEqual(got_accuracy, 50.00)

        got = plot_utils.plot_confusion_matrix(
            got_confusion_matr, True, folder, "Test_utils", "top")
        want = pd.DataFrame([[0, 2], [0, 2]]).astype(np.int32)

        assert_frame_equal(got, want)
        self.assertTrue(os.path.exists(os.path.join(
            folder, "Test_utils_confusion_matrix.png")))

    def test_plot_course(self):
        # test if everything works as intended
        folder = "/tmp/trainings_pipeline/plot_utils_test/test_plot_course"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        success = plot_utils.plot_course(
            [1, 2], "meter", "kmh", "testing", True, folder)
        self.assertTrue(success)

        self.assertTrue(os.path.exists(
            os.path.join(folder, "testing_meter.png")))

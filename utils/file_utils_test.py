import unittest
import os
import sys
import shutil
import json
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal

from utils import file_utils


class TestFileUtils(unittest.TestCase):
    def test_get_files(self):
        file_type = "csv"
        non_existing_folder = "/some_non_existing_folder/non_existing_subfolder"
        if os.path.exists(non_existing_folder):
            os.remove(non_existing_folder)

        # Test if the folder_does not exist
        got_file_list, got_error = file_utils.get_files(
            non_existing_folder, file_type)
        self.assertListEqual(got_file_list, [])
        self.assertEqual(got_error, "given_dir_does_not_exist")

        # Test if the folder does exist, but there is no data in there
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_get_files"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        got_file_list, got_error = file_utils.get_files(folder_path, file_type)
        self.assertListEqual(got_file_list, [])
        self.assertEqual(got_error, "no_data_in_dir")

        # test if there are files in the folder
        df = pd.DataFrame({"label": [11, 11], "feature_col": [21, 25]})
        full_filepath = os.path.join(folder_path, "file.csv")
        df.to_csv(full_filepath, index=False)
        got_file_list, got_error = file_utils.get_files(folder_path, file_type)
        self.assertListEqual(got_file_list, [full_filepath])
        self.assertIsNone(got_error)
        os.remove(full_filepath)

    def test_gather_sessions_in_dict(self):
        # test if the gathering works on valid inputs
        example_list = ["home/blub/something(2)_1.csv", "home/blub/something(3)_1.csv",
                        "home/blub/something_else(4)_1.csv", "home/blub/yet_another_thing(3)_1.csv", ]
        wanted_dict = {
            "something": ["home/blub/something(2)_1.csv",  "home/blub/something(3)_1.csv"],
            "something_else": ["home/blub/something_else(4)_1.csv"],
            "yet_another_thing": ["home/blub/yet_another_thing(3)_1.csv"]
        }
        self.assertDictEqual(
            wanted_dict, file_utils.gather_sessions_in_dict(example_list))
        # Test if the list_of_window_files is empty
        got = file_utils.gather_sessions_in_dict([])
        self.assertDictEqual({}, got)

    def test_contains_files_with_various_levels(self):
        test_folder = "/tmp/trainings_pipeline/file_utils_test/test_contains_files_with_various_levels"
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        # create files with a column which contains only one level
        file_1 = pd.DataFrame({"some_col": [1, 1], "another_col": [1, 5]})
        file_1.to_csv(os.path.join(test_folder, "file_1.csv"), index=False)
        file_2 = pd.DataFrame({"some_col": [8, 8], "another_col": [1, 5]})
        file_2.to_csv(os.path.join(test_folder, "file_2.csv"), index=False)

        # Test if it correctly returns False when all the files contain only one entry in a given column
        got_bool, got_error = file_utils.contains_files_with_various_levels(
            test_folder, "some_col")
        self.assertFalse(got_bool)
        self.assertIsNone(got_error)

        # Test if False, error is returned when the file does not contain the colname in question
        got_bool, got_error = file_utils.contains_files_with_various_levels(
            test_folder, "not_existing_col")
        self.assertFalse(got_bool)
        self.assertEqual(
            got_error, f"column_not_existing_col_missing_from_file_{os.path.join(test_folder, 'file_1.csv')}")

        # Test if it correctly returns True when the files do contain various levels
        file_3 = pd.DataFrame(
            {"some_col": [1, 151165161], "another_col": [1, 5]})
        file_3.to_csv(os.path.join(test_folder, "file_3.csv"), index=False)
        got_bool, got_error = file_utils.contains_files_with_various_levels(
            test_folder, "some_col")
        os.remove(os.path.join(test_folder, "file_3.csv"))
        self.assertTrue(got_bool)
        self.assertIsNone(got_error)

    def test_debundle_files(self):
        # test if it correctly does nothing when a file has only identical entries in specified column
        test_folder = "/tmp/trainings_pipeline/file_utils_test/test_debundle"
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        original_df = pd.DataFrame(
            {"some_col": [1, 1, 1], "another_col": [8, 6, 4]})
        original_df.to_csv(os.path.join(test_folder, "file.csv").replace("\\","/"), index=False)
        result = file_utils.debundle_files(test_folder, "some_col")
        self.assertIsNone(result)

        # Test if the original file still exists
        exists = os.path.exists(os.path.join(test_folder, "file.csv").replace("\\","/"))
        self.assertTrue(exists)
        # test if the df is unchanged
        df = pd.read_csv(os.path.join(test_folder, "file.csv").replace("\\","/"))
        assert_frame_equal(original_df, df)

        # Test if the function returns an error when the searched-for column does not exist
        got = file_utils.debundle_files(test_folder, "not_existing_column")
        self.assertEqual(
            got, f"column_not_existing_column_missing_from_file_{os.path.join(test_folder, 'file.csv')}".replace('\\','/'))

        # test if the function correctly splits when the column has multiple values
        result = file_utils.debundle_files(test_folder, "another_col")
        self.assertIsNone(result)
        # Check if the files exist
        file_8_exists = os.path.exists(os.path.join(test_folder, "8_file.csv").replace("\\","/"))
        self.assertTrue(file_8_exists)
        file_6_exists = os.path.exists(os.path.join(test_folder, "6_file.csv").replace("\\","/"))
        self.assertTrue(file_6_exists)
        file_4_exists = os.path.exists(os.path.join(test_folder, "4_file.csv").replace("\\","/"))
        self.assertTrue(file_4_exists)
        # Check if the original file got deleted
        original_exists = os.path.exists(os.path.join(test_folder, "file.csv").replace("\\","/"))
        self.assertFalse(original_exists)

        # Check if the files got cut correctly
        df_8 = pd.read_csv(os.path.join(test_folder, "8_file.csv").replace("\\","/"))
        assert_frame_equal(df_8, pd.DataFrame(
            {"some_col": [1], "another_col": [8]}))
        df_6 = pd.read_csv(os.path.join(test_folder, "6_file.csv").replace("\\","/"))
        assert_frame_equal(df_6, pd.DataFrame(
            {"some_col": [1], "another_col": [6]}))
        df_4 = pd.read_csv(os.path.join(test_folder, "4_file.csv").replace("\\","/"))
        assert_frame_equal(df_4, pd.DataFrame(
            {"some_col": [1], "another_col": [4]}))

    def test_save_df_with_key(self):
        file = "somewhere/some_file.csv"
        saving_folder = "/tmp/trainings_pipeline/file_utils_test/test_save_as_csv"
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        invalid_df = 17
        # If the resulting file does already exist from another test-run, delete the files
        if os.path.exists(os.path.join(saving_folder, "some_file.csv")):
            os.remove(os.path.join(saving_folder, "some_file.csv"))

        # Test if it correctly does not save if a wrong datatype was given for the df
        result = file_utils.save_df_with_key(file, saving_folder, invalid_df)
        self.assertEqual(result, "got_df_of_non_pdDataFrame_type")
        exists = os.path.exists(os.path.join(saving_folder, "some_file.csv"))
        self.assertFalse(exists)

        # Test if correctly saves if everything is valid
        if os.path.exists(os.path.join(saving_folder, "some_file.csv")):
            os.remove(os.path.join(saving_folder, "some_file.csv"))
        result = file_utils.save_df_with_key(
            file, saving_folder, pd.DataFrame({"some_col": [1, 2, 3]}))
        self.assertIsNone(result)
        exists = os.path.exists(os.path.join(saving_folder, "some_file.csv"))
        self.assertTrue(exists)

    def test_save_json(self):
        saving_folder = "/tmp/trainings_pipeline/file_utils_test/test_save_json"
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        # Test if everything runs as intended:
        valid_json = {"some_key": "some_val", "another_key": [1, 1, 2]}
        full_filename = os.path.join(saving_folder, "dict.json")
        if os.path.exists(full_filename):
            os.remove(full_filename)

        got = file_utils.save_json(valid_json, full_filename)
        self.assertIsNone(got)

    def test_estimate_needed_time(self):

        # test if an invalid service was given
        got_dur, got_error = file_utils.estimate_needed_time(
            {}, "not_implemented_service", 8)
        self.assertEqual(got_dur, "")
        self.assertEqual(got_error, "invalid_service_or_method_given")

        # Test if create_windows was used as a service
        got_dur, got_error = file_utils.estimate_needed_time(
            2000, "create_windows", 8)
        self.assertEqual(got_dur, 17)
        self.assertIsNone(got_error)

        # Test if generate_features was used as a service
        got_dur, got_error = file_utils.estimate_needed_time(
            1200, "generate_features", 8)
        self.assertEqual(got_dur, 2)
        self.assertIsNone(got_error)

    def test_reduce_files_to_handle(self):
        # test if nothing is being done, when the log file is not being found
        target_location = "/tmp/trainings_pipeline/file_utils_test/test_reduce_files_to_handle"
        if os.path.exists(target_location):
            shutil.rmtree(target_location)
        got_list, got_error = file_utils.reduce_files_to_handle(
            [], target_location, "test_service")
        self.assertListEqual(got_list, [])
        self.assertEqual(got_error, "no_log_in_target_location")

        # test if nothing gets returned when a malformed log file is given
        malformed_log_file = {"wrong_key": 42}
        os.makedirs(target_location)
        file_utils.save_json(malformed_log_file, os.path.join(
            target_location, "log.json"))
        got_list, got_error = file_utils.reduce_files_to_handle(
            [], target_location, "test_service")
        self.assertListEqual(got_list, [])
        self.assertEqual(
            got_error, "log_file_does_not_contain_test_service_as_key")

        # Test if correctly stops executing, when a file was already processed (according to log) which is not in the files to handle
        log_file = {"test_service": {
            "successfully_processed_files": ["some_file.csv"]}}
        file_utils.save_json(log_file, os.path.join(
            target_location, "log.json"))
        got_list, got_error = file_utils.reduce_files_to_handle(
            [], target_location, "test_service")
        self.assertListEqual(got_list, [])
        self.assertEqual(
            got_error, "successfully_processed_file_which_is_not_in_the_files_to_handle")

        # test if works as intended, when everything given is valid
        file_utils.save_json(log_file, os.path.join(
            target_location, "log.json"))
        got_list, got_error = file_utils.reduce_files_to_handle(
            ["some_file.csv", "another_file.csv"], target_location, "test_service")
        self.assertListEqual(got_list, ["another_file.csv"])

    def test_gather_filesizes_in_df(self):
        # test if no files were given
        got = file_utils.gather_filesizes_in_df([])
        assert_frame_equal(got, pd.DataFrame({"file": [], "size": []}))

        # Test if everything is valid
        data_folder = "/tmp/trainings_pipeline/file_utils_test/test_gather_filesizes_in_df"
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
        os.makedirs(data_folder)
        df_1 = pd.DataFrame({"some_col": [x for x in range(1000)], "string_col": [
                            str(x) + "b" for x in range(1000)]})
        df_1.to_csv(os.path.join(data_folder, "file_1.csv"), index=False)
        df_1 = pd.DataFrame({"some_col": [x for x in range(100)], "string_col": [
                            str(x) + "b" for x in range(100)]})
        df_1.to_csv(os.path.join(data_folder, "file_2.csv"), index=False)
        input_list = [os.path.join(data_folder, "file_1.csv").replace("\\","/"), os.path.join(
            data_folder, "file_2.csv").replace("\\","/")]
        want_df = pd.DataFrame({"file": [os.path.join(data_folder, "file_1.csv").replace("\\","/"), os.path.join(
            data_folder, "file_2.csv").replace("\\","/")], "size": [8.59375, 0.68359375]})
        if "win" in sys.platform:
            want_df["size"] =[9.5712890625, 0.7822265625]
        got_df = file_utils.gather_filesizes_in_df(input_list)
        assert_frame_equal(got_df, want_df)

    def test_sort_by_filesizes(self):
        # test when no files are given
        got = file_utils.sort_by_filesizes([])
        self.assertListEqual(got, [])

        # Test if everything is valid when order == ascending is being used
        data_folder = "/tmp/trainings_pipeline/file_utils_test/test_sort_by_filesizes"
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
        os.makedirs(data_folder)
        df_1 = pd.DataFrame({"some_col": [x for x in range(1000)], "string_col": [
                            str(x) + "b" for x in range(1000)]})
        df_1.to_csv(os.path.join(data_folder, "file_1.csv"), index=False)
        df_1 = pd.DataFrame({"some_col": [x for x in range(100)], "string_col": [
                            str(x) + "b" for x in range(100)]})
        df_1.to_csv(os.path.join(data_folder, "file_2.csv"), index=False)
        input_list = [os.path.join(data_folder, "file_1.csv"), os.path.join(
            data_folder, "file_2.csv")]
        want = input_list[::-1]
        got = file_utils.sort_by_filesizes(input_list)
        self.assertListEqual(got, want)

        # Test when order descending is being used: Are already in the correct order
        got = file_utils.sort_by_filesizes(input_list, False)
        self.assertListEqual(got, input_list)

    def test_detect_dataset(self):
        # test when trainings-data was seen
        data_folder = "/tmp/trainings_pipeline/file_utils_test/test_detect_dataset"

        file_name_of_trainings_date = os.path.join(
            data_folder, "2022-01-01T12-00-00_2022-01-01T13-00.csv").replace("\\","/")
        file_name_of_potential_test_date_date = os.path.join(
            data_folder, "2022-11-01T12-00-00_2022-11-01T13-00.csv").replace("\\","/")
        file_name_of_potential_holdout_date_date = os.path.join(
            data_folder, "2023-04-10T12-00-00_2022-11-01T13-00.csv").replace("\\","/")

        got = file_utils.detect_dataset(
            [file_name_of_potential_test_date_date, file_name_of_potential_test_date_date, file_name_of_trainings_date])
        self.assertEqual(got, "training_data")

        # test when validation data was seen
        got = file_utils.detect_dataset([file_name_of_potential_test_date_date,
                                        file_name_of_potential_test_date_date, file_name_of_potential_test_date_date])
        self.assertEqual(got, "validation_data")

        # test when test data was seen
        got = file_utils.detect_dataset([file_name_of_potential_holdout_date_date,
                                        file_name_of_potential_holdout_date_date, file_name_of_potential_holdout_date_date])
        self.assertEqual(got, "test_data")

    def test_create_unix_time_from_filename(self):
        # Test if everything works as intended
        got = file_utils.create_unix_time_from_filename(
            "/somewhere/somefile/2023-01-15T12_05_46_some_ident.csv")
        want = 1673737200
        self.assertEqual(got, want)

    def test_initialize_log_file(self):
        # Test if the log file exists, and the parameters of the table match the parameters in the log file
        future_location = "/tmp/trainings_pipeline/file_utils_test/test_initialize_log_file/future_location"
        if os.path.exists(future_location):
            shutil.rmtree(future_location)
        os.makedirs(future_location)
        log_file_template = {"test_service": {"successfully_processed_files": [
            "somewhere/some_file.csv"], "occurred_errors": {}, "saved_files": 15, "store_local": True, "some_param": 42}}
        table = {"store_local": True, "some_param": 42}
        file_utils.save_json(log_file_template, os.path.join(
            future_location, "log.json"))
        got_log, got_error = file_utils.initialize_log_file(
            os.path.join(future_location, "log.json"), "", "test_service", table)
        self.assertDictEqual(got_log, log_file_template)
        self.assertIsNone(got_error)

        # Test if the log file exists in future locations, but parameters are not identical
        table = {"store_local": False, "some_param": 130}
        file_utils.save_json(log_file_template, os.path.join(
            future_location, "log.json"))
        got_log, got_error = file_utils.initialize_log_file(
            os.path.join(future_location, "log.json"), "", "test_service", table)
        self.assertDictEqual(got_log, {})
        self.assertEqual(
            got_error, "parameters_of_current_iteration_not_identical_to_previous")

        # test if the future location does not contain a log file, but the prior does: initializing new log in future
        if os.path.exists(future_location):
            shutil.rmtree(future_location)
        prior_location = "/tmp/trainings_pipeline/file_utils_test/test_initialize_log_file/prior_location"
        if os.path.exists(prior_location):
            shutil.rmtree(prior_location)
        os.makedirs(prior_location)
        log_file_template = {"previous_service": {"successfully_processed_files": [
            "somewhere/some_file.csv"], "occurred_errors": {}, "saved_files": 15, "store_local": True, "some_param": 42}}

        file_utils.save_json(log_file_template, os.path.join(
            prior_location, "log.json"))
        got_log, got_error = file_utils.initialize_log_file(os.path.join(
            future_location, "log.json"), os.path.join(prior_location, "log.json"), "test_service", table)
        want_log = {"previous_service": {"successfully_processed_files": ["somewhere/some_file.csv"], "occurred_errors": {}, "saved_files": 15, "store_local": True, "some_param": 42},
                    "test_service": {"successfully_processed_files": [], "occurred_errors": {}, "saved_files": 0, "store_local": False, "some_param": 130}}
        self.assertDictEqual(got_log, want_log)
        self.assertIsNone(got_error)

        # Test if the service is create_windows
        if os.path.exists(prior_location):
            shutil.rmtree(prior_location)
        if os.path.exists(future_location):
            shutil.rmtree(future_location)
        got_log, got_error = file_utils.initialize_log_file(os.path.join(
            future_location, "log.json"), os.path.join(future_location, "log.json"), "create_windows", table)
        want_log = {"create_windows": {"successfully_processed_files": [
        ], "occurred_errors": {}, "saved_files": 0, "store_local": False, "some_param": 130}}
        self.assertDictEqual(got_log, want_log)
        self.assertIsNone(got_error)

        # test if the file is neither in the prior nor the future loc and service != create_windows
        if os.path.exists(future_location):
            shutil.rmtree(future_location)
        if os.path.exists(prior_location):
            shutil.rmtree(prior_location)

        got_log, got_error = file_utils.initialize_log_file(os.path.join(
            future_location, "log.json"), prior_location, "test_service", table)
        want_log = {"create_windows": {
            "successfully_processed_files": [], "occurred_errors": {}, "saved_files": 0}}
        self.assertDictEqual(got_log, {})
        self.assertEqual(
            got_error, "log_file_neither_in_target_nor_prior_location")

    def test_create_log_from_result_strings(self):
        # Test if everything works as intended
        path_to_existing_log_file = "/tmp/trainings_pipeline/file_utils_test/test_create_log_from_result_string"
        if os.path.exists(path_to_existing_log_file):
            shutil.rmtree(path_to_existing_log_file)
        os.makedirs(path_to_existing_log_file)
        current_log = {"test_service": {
            "successfully_processed_files": [], "occurred_errors": {}, "saved_files": 0}}
        file_utils.save_json(current_log, os.path.join(
            path_to_existing_log_file, "log.json"))
        list_of_result_strings = ["/somewhere/some_file.csv+++True+++None+++20",
                                  "/somewhere/another_file.csv+++False+++no_time_col_found+++0"]
        want = {"test_service": {
            "successfully_processed_files": ["/somewhere/some_file.csv"],
            "occurred_errors": {"no_time_col_found": ["/somewhere/another_file.csv"]},
                "saved_files": 20}}
        got = file_utils.create_log_from_result_strings(list_of_result_strings, os.path.join(
            path_to_existing_log_file, "log.json"), "test_service")
        self.assertDictEqual(got, want)

    def test_processing_flattened_data(self):
        # Test if create_win not in the keys
        got_bool, got_error = file_utils.processing_flattened_data({})
        self.assertFalse(got_bool)
        self.assertEqual(got_error, "create_windows _not_found_in_the_keys")

        # test if flatten not in the keys
        got_bool, got_error = file_utils.processing_flattened_data(
            {"create_windows": {"another_param": 42}})
        self.assertFalse(got_bool)
        self.assertEqual(
            got_error, "key_flatten_not_found_in_key_create_windows")

        # Test if everything works as intended
        got_bool, got_error = file_utils.processing_flattened_data(
            {"create_windows": {"flatten": True}})
        self.assertTrue(got_bool)
        self.assertIsNone(got_error)

    def test_processing_featured_data(self):
        # Test for data without features
        got_bool = file_utils.processing_featured_data(
            {"create_windows": {"flatten": True}})
        self.assertFalse(got_bool)
        # test for featured_data
        self.assertTrue(file_utils.processing_featured_data(
            {"generate_features": {"flatten": True}}))

    def test_read_csv_safely(self):
        folder = "/tmp/trainings_pipeline/file_utils_test/test_read_csv_safely"
        if not os.path.exists(folder):
            os.makedirs(folder)
        df = pd.DataFrame({"some_col": [1., 2., 3.], "label": [2, 2, 2]})
        df["label"] = df["label"].astype(np.int16)
        df.to_csv(os.path.join(folder, "df.csv"), index=False)
        got = file_utils.read_csv_safely(os.path.join(folder, "df.csv"))
        assert_frame_equal(got, df)

    def test_count_filesizes_of_dir(self):
        folder = "/tmp/trainings_pipeline/file_utils_test/test_count_filesizes_of_dir"
        if not os.path.exists(folder):
            os.makedirs(folder)
        df = pd.DataFrame({"some_col": [1, 2, 3]*20, "label": [2, 2, 2]*20})
        df.to_csv(os.path.join(folder, "df.csv"), index=False)
        df = pd.DataFrame({"some_col": [1, 2, 3]*60, "label": [2, 2, 2]*60})
        df.to_csv(os.path.join(folder, "df_2.csv"), index=False)

        # Test if the folder does not exist
        non_existing_folder = "/some_non_existing_folder/non_existing_subfolder"
        if os.path.exists(non_existing_folder):
            os.remove(non_existing_folder)
        got_size, got_error = file_utils.count_filesizes_of_dir(
            non_existing_folder)
        self.assertEqual(got_size, 1000.)
        self.assertEqual(got_error, "given_dir_does_not_exist")

        # Tets if the folder does exist
        got_size, got_error = file_utils.count_filesizes_of_dir(folder)
        self.assertEqual(got_size, 0.0)
        self.assertIsNone(got_error)

    def test_save_csv_from_dict(self):
        # Test if df is missing from keys
        self.assertEqual(file_utils.save_csv_from_dict({}),
                         "no_df_for_saving_given")

        # test if df is not a pd.DataFrame
        self.assertEqual(file_utils.save_csv_from_dict(
            {"df": 12}), "got_df_of_non_pdDataframe_type")

        # Test if everything works as intended
        folder = "/tmp/trainings_pipeline/file_utils_test/test_save_csv_from_dict"
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.assertIsNone(file_utils.save_csv_from_dict({"df": pd.DataFrame({"some_col": [
                          1, 2, 3], "label": [1, 2, 3]}), "filename": os.path.join(folder, "df.csv")}))
        exists = os.path.exists(os.path.join(folder, "df.csv"))
        self.assertTrue(exists)
        df = pd.read_csv(os.path.join(folder, "df.csv"), header=None)
        assert_frame_equal(df, pd.DataFrame([[1], [2], [3]]))

    def test_save_csvs_labelwise(self):
        # Test if everything works as intended
        df = pd.DataFrame({"some_col": [0, 1, 2], "label": [0, 1, 2]})
        folder = "/tmp/trainings_pipeline/file_utils_test/test_save_csv_labelwise"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        got_error = file_utils.save_csvs_labelwise(df, folder)
        self.assertIsNone(got_error)

        # Find the filenames of the existing files
        filenames, error = file_utils.get_files(folder, "csv")
        self.assertIsNone(error)
        self.assertEqual(len(filenames), 3)

        # test if the resulting files contain the correct content
        for number, filename in enumerate(filenames):
            if f"/{number}_prepared_" in filename:
                assert_frame_equal(pd.read_csv(
                    filename, header=None), pd.DataFrame([[number]]))

    def test_remove_outdated_processed_files(self):
        # Test when not all files from create_windows were processed
        folder = "/tmp/trainings_pipeline/file_utils_test/test_remove_outdated_processed_files"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(os.path.join(folder, "labeled_data"))
        df = pd.DataFrame({"some_col": [1, 2, 3], "label": [1, 2, 3]})
        df.to_csv(os.path.join(folder, "labeled_data", "df.csv"), index=False)

        log_file = {"create_windows": {
            "successfully_processed_files": [], "data_path": folder}}
        got_log, got_error = file_utils.remove_outdated_processed_files(
            log_file, "generate_features")
        self.assertDictEqual(got_log, log_file)
        self.assertIsNone(got_error)

        # Test when all the files from create_windows were processed
        df = pd.DataFrame({"some_col": [1, 2, 3], "label": [1, 2, 3]})
        df.to_csv(os.path.join(folder, "labeled_data", "df.csv"))

        log_file = {"create_windows": {"successfully_processed_files": [
            os.path.join(folder, "labeled_data", "df.csv")], "data_path": folder}}
        got_log, got_error = file_utils.remove_outdated_processed_files(
            log_file, "generate_features")
        want = {"create_windows": {
            "successfully_processed_files": "all", "data_path": folder}}
        self.assertDictEqual(got_log, want)

        # Test when all the data from gen_features was processed
        log_file = {"create_windows": {"successfully_processed_files": "all", "data_path": folder,
                                       "output_date": "1990-01-01T12-00"}, "generate_features": {"successfully_processed_files": [1]}}
        if os.path.exists(os.path.join(folder, "windows",  "1990-01-01T12-00")):
            shutil.rmtree(os.path.join(folder, "windows",  "1990-01-01T12-00"))
        os.makedirs(os.path.join(folder, "windows",  "1990-01-01T12-00"))
        df.to_csv(os.path.join(folder, "windows",
                  "1990-01-01T12-00", "df.csv"))
        got_log, got_error = file_utils.remove_outdated_processed_files(
            log_file, "reduce_data")
        want = {"create_windows": {"successfully_processed_files": "all", "data_path": folder, "output_date": "1990-01-01T12-00"},
                "generate_features": {"successfully_processed_files": "all"}}
        self.assertDictEqual(got_log, want)
        self.assertIsNone(got_error)

        # test when all the files from reduce_data_were processed
        gen_features_saving_dir = os.path.join(
            folder, "features", "1990-01-01T12-00")
        if os.path.exists(gen_features_saving_dir):
            shutil.rmtree(gen_features_saving_dir)
        os.makedirs(gen_features_saving_dir)
        df.to_csv(os.path.join(gen_features_saving_dir, "df.csv"))

        log_file = {"create_windows": {"successfully_processed_files": "all", "data_path": folder},
                    "generate_features": {"successfully_processed_files": "all", "saving_folder": gen_features_saving_dir},
                    "reduce_data": {"successfully_processed_files": [os.path.join(gen_features_saving_dir, "df.csv")]}}
        got_log, got_error = file_utils.remove_outdated_processed_files(
            log_file, "balancing_over")
        want = {"create_windows": {"successfully_processed_files": "all", "data_path": folder},
                "generate_features": {"successfully_processed_files": "all", "saving_folder": gen_features_saving_dir},
                "reduce_data": {"successfully_processed_files": "all"}}

        self.assertDictEqual(got_log, want)
        self.assertIsNone(got_error)

    def test_update_log_file(self):
        # Test when everything is valid and gets saved correctly
        folder = "/tmp/trainings_pipeline/file_utils_test/test_update_log_file"
        gen_features_saving_dir = os.path.join(
            folder, "features", "1990-01-01T12-00")
        if os.path.exists(gen_features_saving_dir):
            shutil.rmtree(gen_features_saving_dir)
        os.makedirs(gen_features_saving_dir)
        log_file = {"create_windows": {"successfully_processed_files": "all", "data_path": folder},
                    "generate_features": {"successfully_processed_files": "all", "saving_folder": gen_features_saving_dir},
                    "reduce_data": {"successfully_processed_files": [os.path.join(gen_features_saving_dir, "df.csv")]}}
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        df.to_csv(os.path.join(gen_features_saving_dir, "df.csv"), index=False)
        table = {"some_param": 10, "another_param": 42}
        processing_step = "reduce_data"
        got_error = file_utils.update_log_file(
            log_file, table, folder, processing_step)
        self.assertIsNone(got_error)

        # Test if the resulting file exists
        self.assertTrue(os.path.join(folder, "log.json"))
        # test if it contains the correct data
        want = {"create_windows": {"successfully_processed_files": "all", "data_path": folder},
                "generate_features": {"successfully_processed_files": "all", "saving_folder": gen_features_saving_dir},
                "reduce_data": {"successfully_processed_files": [os.path.join(gen_features_saving_dir, "df.csv")], "some_param": 10, "another_param": 42}}
        with open(os.path.join(folder, "log.json")) as f:
            got = json.load(f)
        self.maxDiff = None
        self.assertDictEqual(got, want)

    def test_processing_windowed_data(self):
        # Test when the data exists in windowed form
        got = file_utils.processing_windowed_data({"create_windows": True})
        self.assertTrue(got)

        # test when the data does not exists in windowed form
        got = file_utils.processing_windowed_data({})
        self.assertFalse(got)

    def test_get_unique_levels_from_folder(self):
        # test if it correctly returns an error wen the folder does not contain any files
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_get_unique_levels_from_folder"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        got_levels, got_error = file_utils.get_unique_levels_from_folder(
            folder_path, "label", np.int16, "csv")
        self.assertIsNone(got_levels)
        self.assertEqual(got_error, "no_data_in_dir")

        # test if everything works as intended
        pd.DataFrame({"label": [1, 1, 1, ]}).to_csv(
            os.path.join(folder_path, "df_1.csv"))
        pd.DataFrame({"label": [2, 2, 123]}).to_csv(
            os.path.join(folder_path, "df_2.csv"))
        got_levels, got_error = file_utils.get_unique_levels_from_folder(
            folder_path, "label", np.int16, "csv")
        assert_array_equal(got_levels, np.array([1, 2, 123]).reshape(3,))
        self.assertIsNone(got_error)

    def test_count_filesizes_of_list(self):
        # Test if everything works as intended
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_count_filesizes_of_list"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        # Test when only a really small file exists in there
        pd.DataFrame({"label": [x for x in range(1000)], "some_col": [
                     str(x) for x in range(1000)], }).to_csv(os.path.join(folder_path, "df.csv"))
        all_files, _ = file_utils.get_files(folder_path, "csv")

        got = file_utils.count_filesizes_of_list(all_files)
        want = 0.00
        self.assertEqual(got, want)

        # Test when a bigger file is in the list
        pd.DataFrame({"label": [x for x in range(1000000)], "some_col": [str(
            x) for x in range(1000000)], }).to_csv(os.path.join(folder_path, "df_2.csv"))
        all_files, _ = file_utils.get_files(folder_path, "csv")

        got = file_utils.count_filesizes_of_list(all_files)
        want = 0.02
        self.assertEqual(got, want)

    def test_list_of_maxsized_filechunks(self):
        # Test when all the files fit into a single chunk
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_list_of_maxsized_filechunks"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        # Create a list of small files
        for i in range(10):
            pd.DataFrame({"label": [x for x in range(1000000)], "some_col": [str(
                x) for x in range(1000000)], }).to_csv(os.path.join(folder_path, f"df_{i}.csv").replace("\\","/"))
        all_files, _ = file_utils.get_files(folder_path, "csv")
        all_files = [file.replace("\\","/") for file in all_files]

        got_list, got_error = file_utils.list_of_maxsized_filechunks(
            all_files, 99.85654)
        want_list = [all_files]
        self.assertListEqual(got_list, want_list)
        self.assertIsNone(got_error)

        # Test when the files need to be split into chunks
        got_list, got_error = file_utils.list_of_maxsized_filechunks(
            all_files, 0.02)
        want_list = [[os.path.join(folder_path, f"df_{x}.csv").replace("\\","/")] for x in [
            6, 4, 2, 9, 8, 5, 0, 1, 3, 7]]
        if "win" in sys.platform:
            want_list = [[os.path.join(folder_path, f"df_{x}.csv").replace("\\","/")] for x in [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.assertListEqual(got_list, want_list)
    
        self.assertIsNone(got_error)

    def test_processing_oversampled_data(self):
        # test if the data was oversampled
        self.assertTrue(file_utils.processing_oversampled_data(
            {"balancing_over": 12}))
        # test if not
        self.assertFalse(file_utils.processing_oversampled_data({}))

    def test_flatten_create_generator(self):
        got = file_utils.flatten_create_generator([[1, 2, 3], 4, 5, [6, 8, 9]])
        self.assertEqual(str(type(got)), "<class 'generator'>")

    def test_flatten_nested_lists(self):
        # Test if everything works as intended
        nested_list = [[1, 2, 3], 4, 5, [6, 8, 9]]
        got = file_utils.flatten_nested_lists(nested_list)
        want = [1, 2, 3, 4, 5, 6, 8, 9]
        self.assertListEqual(got, want)

    def test_unpack_list_of_result_strings_if_needed(self):
        # test when the list contains result-strings
        got = file_utils.unpack_list_of_result_strings_if_needed(
            ["success", "fail", "success"])
        self.assertListEqual(got, ["success", "fail", "success"])

        # test when list contains nested lists
        got = file_utils.unpack_list_of_result_strings_if_needed(
            [["success", "fail"], ["success", "fail", "success"]])
        self.assertListEqual(
            got, ["success", "fail", "success", "fail", "success"])

    def test_read_and_check_existing_log_file(self):
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_read_and_check_existing_log_file"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        # test if the parameters of the previous iteration do not match the current ones
        previous_log_file = {"some_service": {"some_param": 15}}
        error = file_utils.save_json(
            previous_log_file, os.path.join(folder_path, "log.json"))
        self.assertIsNone(error)

        table = {"some_param": 13}
        service = "some_service"

        got_log_file, got_error = file_utils.read_and_check_existing_log_file(
            os.path.join(folder_path, "log.json"), table, service)
        self.assertEqual(
            got_error, "parameters_of_current_iteration_not_identical_to_previous")

        # test if everything works as intended
        table = {"some_param": 13, "new_param": 42}

        previous_log_file = {"some_service": {
            "some_param": 13}, "another_service": {"param": "something"}}
        error = file_utils.save_json(
            previous_log_file, os.path.join(folder_path, "log.json"))
        self.assertIsNone(error)

        got_log_file, got_error = file_utils.read_and_check_existing_log_file(
            os.path.join(folder_path, "log.json"), table, service)
        self.assertIsNone(got_error)
        want_log_file = {"some_service": {
            "some_param": 13, "new_param": 42}, "another_service": {"param": "something"}}
        self.assertDictEqual(got_log_file, want_log_file)

    def test_initialize_empty_log_file(self):
        # Test if everything works as intended
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_initialize_empty_log_file"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        service = "some_service"
        table = {"some_param": 13, "new_param": 42}
        future_location = os.path.join(folder_path, "log.json")

        got_log_file, got_error = file_utils.initialize_empty_log_file(
            service, table, future_location)
        self.assertIsNone(got_error)

        want_log_file = {'some_service': {'successfully_processed_files': [
        ], 'occurred_errors': {}, 'saved_files': 0, "some_param": 13, "new_param": 42}}
        self.assertDictEqual(got_log_file, want_log_file)

        self.assertTrue(os.path.exists(os.path.join(folder_path, "log.json")))
        with open(os.path.join(folder_path, "log.json")) as f:
            read_log_file = json.load(f)

        self.assertDictEqual(read_log_file, want_log_file)

    def test_bring_prior_existing_log_file_to_future_location(self):
        # Test if everything works as intended
        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_initialize_empty_log_file"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        future_folder = os.path.join(folder_path, "future")
        os.makedirs(future_folder)

        log_file = {"some_service": {"some_param": 13, "new_param": 42, "successfully_processed_files": [],
                                     "occurred_errors": {},
                                     "saved_files": 0}}
        error = file_utils.save_json(
            log_file, os.path.join(folder_path, "log.json"))
        self.assertIsNone(error)

        table = {"unseen_param": 69}
        service = "new_service"

        got_log_file, got_error = file_utils.bring_prior_existing_log_file_to_future_location(
            os.path.join(folder_path, "log.json"), service, table, os.path.join(future_folder, "log.json"))
        self.assertIsNone(got_error)

        want_log_file = {"some_service": {"some_param": 13, "new_param": 42, "successfully_processed_files": [],
                                          "occurred_errors": {}, "saved_files": 0},
                         "new_service": {"unseen_param": 69, "successfully_processed_files": [],
                                         "occurred_errors": {}, "saved_files": 0}}
        self.assertDictEqual(got_log_file, want_log_file)

        self.assertTrue(os.path.join(future_folder, "log.json"))
        with open(os.path.join(future_folder, "log.json")) as f:
            read_log_file = json.load(f)

        self.assertDictEqual(read_log_file, want_log_file)

    def test_append_second_location(self):
        # test if everything works as intended
        log_file = {"creative_service": {"param": 45}}
        service = "creative_service"

        folder_path = "/tmp/trainings_pipeline/file_utils_test/test_append_second_location"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        second_prior_location = os.path.join(folder_path, "log.json")
        second_location_log_file = {"some_service": {"some_param": 13, "new_param": 42, "successfully_processed_files": [],
                                    "occurred_errors": {}, "saved_files": 0},
                                    "new_service": {"unseen_param": 69, "successfully_processed_files": [],
                                                    "occurred_errors": {}, "saved_files": 0}}

        error = file_utils.save_json(
            second_location_log_file, second_prior_location)
        self.assertIsNone(error)

        got = file_utils.append_second_location(
            log_file, second_prior_location, service)

        want = {"creative_service": {"param": 45},
                "creative_service_second_location":
                    {"some_service": {"some_param": 13, "new_param": 42, "successfully_processed_files": [],
                                      "occurred_errors": {}, "saved_files": 0},
                     "new_service": {"unseen_param": 69, "successfully_processed_files": [],
                                     "occurred_errors": {}, "saved_files": 0}}}
        self.assertDictEqual(got, want)

    def test_valid_prior_preprocessing_steps_seen(self):
        # Test when the data has not been balanced
        service = "prepare_dataset"
        log_file = {}
        table = {}
        got_log, got_error = file_utils.valid_prior_preprocessing_steps_seen(
            service, log_file, table)
        self.assertEqual(
            got_error, "tried_preparing_dataset_without_balancing before")

        # test when the data has been oversampled at a different granularity
        log_file = {"balancing_over": {"granularity": 15}}
        table = {"granularity": 14}
        got_log, got_error = file_utils.valid_prior_preprocessing_steps_seen(
            service, log_file, table)
        self.assertEqual(
            got_error, f"tried_preparing_dataset_without_differing_granularity_level_chose_{log_file.get('balancing_over').get('granularity')}_for_balancing_received_{table.get('granularity')}")

        # test if everything works as intended
        table = {"granularity": 15}
        got_log, got_error = file_utils.valid_prior_preprocessing_steps_seen(
            service, log_file, table)
        self.assertIsNone(got_error)
        self.assertDictEqual(got_log, log_file)

    def test_valid_anonymize_file_found(self):
        # test if the folder does not exist
        folder = "/tmp/trainings_pipeline/file_utils_test/test_valid_anonymize_file_found"
        if os.path.exists(folder):
            shutil.rmtree(folder)

        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(got, "datapath_does_not_exist")

        # no json in there
        os.makedirs(folder)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(got, "anonymization_file.json_does_not_exist")

        # not all required kexs existing
        error = file_utils.save_json(
            {"create_windows": 15}, os.path.join(folder, "anonymization_file.json"))
        self.assertIsNone(error)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(got, "not_all_required_keys_found")

        # create_windows does not have all needed subkeys
        error = file_utils.save_json({"create_windows": {}, "svm_features": {
        }, "feature_utils": {}}, os.path.join(folder, "anonymization_file.json"))
        self.assertIsNone(error)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(
            got, "create_windows_does_not_contain_all_necessary_subkeys")

        # svm_features does not have all needed subkeys
        error = file_utils.save_json({"create_windows": {"needed_columns": [], "data_columns": [
        ]}, "svm_features": {}, "feature_utils": {}}, os.path.join(folder, "anonymization_file.json"))
        self.assertIsNone(error)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(
            got, "svm_features_does_not_contain_all_necessary_subkeys")

        # feature_utils does not have all needed subkeys
        error = file_utils.save_json({"create_windows": {"needed_columns": [], "data_columns": []},
                                     "svm_features": {"top": [], "lifting": [], "walking": []}, "feature_utils": {}}, os.path.join(folder, "anonymization_file.json"))
        self.assertIsNone(error)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(
            got, "feature_utils_does_not_contain_all_necessary_subkeys")

        # Non-list-object does not have all needed subkeys
        error = file_utils.save_json({"create_windows": {"needed_columns": [], "data_columns": []},
                                     "svm_features": {"top": [], "lifting": [], "walking": []},
                                      "feature_utils": {"accel_cols": 12, "differences_cols": [], "needed_cols": [], "on_single_column": [], "magn_feat_name": []}},
                                     os.path.join(folder, "anonymization_file.json"))
        self.assertIsNone(error)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertEqual(got, "non_list_object_found")

        # everything valid
        error = file_utils.save_json({"create_windows": {"needed_columns": [], "data_columns": []},
                                     "svm_features": {"top": [], "lifting": [], "walking": []},
                                      "feature_utils": {"accel_cols": [], "differences_cols": [], "needed_cols": [], "on_single_column": [], "magn_feat_name": []}},
                                     os.path.join(folder, "anonymization_file.json"))
        self.assertIsNone(error)
        got = file_utils.valid_anonymize_file_found(folder)
        self.assertIsNone(got)

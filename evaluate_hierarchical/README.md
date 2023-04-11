## Evaluate_Hierarchical

Provides all the necessary utilities to evaluate a hierarchically classifying model by aggregating the performance of three classifiers: The top-level classifier, as well as the lifting and walking classifier.


### Required Arguments:

-top **top_level_folder**: Specify the path to a directory, in which the program searches for the evaluated top-level classifier

-lift **lifting_folder**: Specify the path to a directory, in which the program searches for the evaluated lifting classifier

-walk **walking_folder**: Specify the path to a directory, in which the program searches for the evaluated walking classifier



### Optional Arguments:

-n **nr_obs_per_activity**: Specify the amount of observations for the joined classification, default is 4718

-o **Output_date**: Specify the output_date, meaning the name of the directory in which the results of the evaluation shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm


If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/evaluate_model/<type>/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 

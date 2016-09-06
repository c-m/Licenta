class Options(object):
    GRADES_ONLY = 'grades_only'
    AGG_GRADES_ONLY = 'agg_grades_only'
    ALL_FEATURES = 'all_features'
    ALL_FEATURES_AGG = 'all_features_agg'
    ALL_LOGS = 'all_logs'

    def __init__(self):
        pass

students_data = load_data()

option = Options.ALL_FEATURES_AGG
data = get_data(option, students_dataset)
data = preprocess_data(data, poly_features=False)

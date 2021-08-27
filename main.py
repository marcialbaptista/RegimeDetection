# This is a sample Python script.
from Program import Program

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    preprocess_data = False
    plot_random_flight = False
    create_taxi_df = True
    run_rf_classifier = False
    run_lstm_classifier = False
    program = Program(debug=True, error=True)
    if create_taxi_df or preprocess_data:
        #program.create(filename='data//flights_dir_new.pickle')
        program.load(filename='data//flights_dir.pickle')
    if plot_random_flight:
        program.plot_random_flights(col_index=8, no_flights=50)
    if create_taxi_df:
        program.create_df(filename='data//taxi_data_fast_flight_stage_detection.pickle')
    if run_rf_classifier:
        program.run_rf_multi_classification(filename='data//taxi_data_fast_flight_stage_detection.pickle')
    if run_lstm_classifier:
        program.run_lstm_multi_classification(filename='data//taxi_data_fast_flight_stage_detection.pickle')




# def other():
#     print_hi('PyCharm')
#     create_taxi_df = False
#     dirs = AircraftDirs(debug=False, error=True)
#     if dirs.is_created():
#         dirs.load()
#     elif create_taxi_df:
#         dirs.create()
#         dirs.find_duplicates()
#
#     #data.plot_flights(percentage_flights_per_aircraft_plot=0.1, col_index=8)
#     #data.plot_histogram_len_flights()
#     #data.plot_aircraft(save_plot=True, col_index=14)
#     #data.find_outliers(col_index=14)
#     #data.plot_taxi_aircraft(col_index=8)
#
#     if create_taxi_df:
#         dirs.load()
#         data = AllData(debug=True, error=True)
#         data.create_taxi_df()
#         data.save_taxi_df()
#
#     lstm = LSTMModelClassification()
#     lstm.cross_validate()
#     lstm.evaluate_classification()
#
#     rf = RFModel()
#     rf.cross_validate()
#     rf.evaluate()
#
#
#
#     print('Finished')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
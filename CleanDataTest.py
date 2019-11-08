import unittest

import numpy as np
import pandas as pd

import CleanData

pd.set_option('display.max_columns', None, 'display.max_rows', None)

input_df = pd.read_csv("data/train.csv", header=0)
df = CleanData.clean_data(input_df.head())
output, y_train = CleanData.convert_data(input_df.head(100))


def test_clean_raw_data():
    np.testing.assert_array_equal(df["VisitorTeamAbbr"].head(), ["KC", "KC", "KC", "KC", "KC"])


def test_add_play_direction():
    np.testing.assert_array_equal(df["ToLeft"].head(), [True, True, True, True, True])


def test_add_ball_carrier():
    np.testing.assert_array_equal(df["IsBallCarrier"].head(), [False, False, False, False, False])


def test_add_yards_from_own_goal():
    np.testing.assert_array_equal(df["YardsFromOwnGoal"].head(), [35, 35, 35, 35, 35])


def test_add_side():
    np.testing.assert_array_equal(df["side"].head(), ['defense', 'defense', 'defense', 'defense', 'defense'])


def test_get_all_pids():
    np.testing.assert_array_equal(CleanData.get_all_pids(df), [20170907000118])


def test_make_x_go_to_left():
    np.testing.assert_allclose(df["X"].head(), [36.09, 35.33, 36., 38.54, 40.68])


def test_make_y_go_to_left():
    np.testing.assert_allclose(df["Y"].head(), [18.46, 20.66, 20.1, 25.6, 17.88])


def test_convert_orientation_to_radians():
    np.testing.assert_allclose(df["Orientation_new"].head(), [6.143384, 5.194274, 4.764923, 4.708375, 4.932824],
                               rtol=1e-06)


def test_convert_dir_to_radians():
    np.testing.assert_allclose(df["Dir_new"].head(), [1.521578, 1.897173, 1.96751, 0.272969, 1.296954], rtol=1e-05)


def test_play_direction():
    np.testing.assert_array_equal(df["PlayDirection"].head(), ["right", "right", "right", "right", "right"])


def test_yard_line():
    np.testing.assert_array_equal(df["YardLine"].head(), [35, 35, 35, 35, 35])


def test_quarter():
    np.testing.assert_array_equal(df["Quarter"].head(), [1, 1, 1, 1, 1])


def test_game_clock():
    np.testing.assert_array_equal(df["GameClock"].head(),
                                  ["14:14:00", "14:14:00", "14:14:00", "14:14:00", "14:14:00"])


def test_down():
    np.testing.assert_array_equal(df["Down"].head(), [3, 3, 3, 3, 3])


def test_distance():
    np.testing.assert_array_equal(df["Distance"].head(), [2, 2, 2, 2, 2])


def test_pid_works():
    # print(f'output: {output}')
    expected = [1, 2.02E+13, 5.194274387, 4.408352625, 4.764923391, 6.143384434, 3.810751889, 4.708374723,
                6.029588967, 4.932824065, 4.162959332, 4.66404336, 4.445353605, 1.897172897, 3.213849285,
                1.967509666, 1.521578042, 0.087440996, 0.272969495, 1.759990018, 1.296954167, 3.142290785,
                4.059461307, 5.67773059, 4.593310353, 4.880256141, 5.448981556, 6.480871855, 7.500466652,
                7.820038363, 9.902019996, 10.62247617, 12.96859283, 14.64451092, 22.41587161, 0.477278702,
                0.340542898, 0.51208955, 0.727539316, 1.056445942, 0.370295381, 0.993997736, 0.478383384,
                1.143020142, 1.248397964, 0.1626551, 0.063138631, -0.694243143, 0.140446896, 1.184640417,
                0.571781257, 0.418012381, 3.279161389, 1.243543167, -0.515199569, -1.049916468, 0.186998554,
                -0.415227062, -0.190857168, -1.211888885, -1.205291285, 0.83256603, 0.040812369, -3.154298747,
                -1.328909475, -1.127904874, -0.360243544, 0.180642024, 1.78337743, 2.043431488, 2.242573556,
                1.385093294, 2.043606021, 1.550201442, 0.750666111, 0.384321501, 1.084547597, 1.552819435,
                0.364075682, 2.317099115, 3.09481783, 2.653773128, 2.891486972, 2.197369528, 1.894554903,
                1.834166511, 3.290294706, 3.388556743, 1.449724112, 3.815337469, 4.364057745, 4.401931394,
                4.651075145, 4.789916492, 6.11478536, 6.775160515, 12.04151569, 13.29683421, 2.341561388,
                0.515244964, 0.314504304, 0.326060873, 0.458839954, 0.593792927, 1.085937561, 1.227573816,
                1.227772386, 1.348619768, -0.332279565, -0.343511046, -1.739928545, -0.686682726, -1.290382832,
                -0.056044231, 0.952888917, 4.23121523, -0.991510982, -1.039875878, 0.771485768, -1.460137035,
                -0.657456203, -0.726957243, -1.106757492, -1.709081345, -0.998199735, -2.935867448, -1.851190421,
                -2.05150144, 2017090700, 'home', 31.25, 22.77, 31.25, 22.77, 3.75, 3.63, 3.35, 0.38, 161.98, 245.74,
                2543773, 'James White', 28, 2017, 35.00, 1, '14:14:00', 'NE', 3, 2, 'NE', 0, 0, 2543773, 'SHOTGUN',
                "1 RB, 1 TE, 3 WR", 6, "2 DL, 3 LB, 6 DB", 'right', '2017-09-08T00:44:06.000Z',
                '2017-09-08T00:44:05.000Z', 8, '10-May', 205, '2/3/1992', 'Wisconsin', 'RB', 'NE', 'KC', 1,
                'Gillette Stadium', "Foxborough, MA", 'Outdoor', 'Field Turf', 'Clear and warm', 63, 77, 8, 'SW',
                True, True, 35, 'offense', 1.256287996, 2.718175777]
    expected_headers = ["", 'PlayId', 'def_Orientation_new1', 'def_Orientation_new2', 'def_Orientation_new3',
                        'def_Orientation_new4', 'def_Orientation_new5', 'def_Orientation_new6',
                        'def_Orientation_new7', 'def_Orientation_new8', 'def_Orientation_new9',
                        'def_Orientation_new10', 'def_Orientation_new11', 'def_Dir_new1', 'def_Dir_new2',
                        'def_Dir_new3', 'def_Dir_new4', 'def_Dir_new5', 'def_Dir_new6', 'def_Dir_new7',
                        'def_Dir_new8', 'def_Dir_new9', 'def_Dir_new10', 'def_Dir_new11', 'def_dist_from_RB1',
                        'def_dist_from_RB2', 'def_dist_from_RB3', 'def_dist_from_RB4', 'def_dist_from_RB5',
                        'def_dist_from_RB6', 'def_dist_from_RB7', 'def_dist_from_RB8', 'def_dist_from_RB9',
                        'def_dist_from_RB10', 'def_dist_from_RB11', 'def_ang_from_RB1', 'def_ang_from_RB2',
                        'def_ang_from_RB3', 'def_ang_from_RB4', 'def_ang_from_RB5', 'def_ang_from_RB6',
                        'def_ang_from_RB7', 'def_ang_from_RB8', 'def_ang_from_RB9', 'def_ang_from_RB10',
                        'def_ang_from_RB11', 'def_radial_speed1', 'def_radial_speed2', 'def_radial_speed3',
                        'def_radial_speed4', 'def_radial_speed5', 'def_radial_speed6', 'def_radial_speed7',
                        'def_radial_speed8', 'def_radial_speed9', 'def_radial_speed10', 'def_radial_speed11',
                        'def_tangential_speed1', 'def_tangential_speed2', 'def_tangential_speed3',
                        'def_tangential_speed4', 'def_tangential_speed5', 'def_tangential_speed6',
                        'def_tangential_speed7', 'def_tangential_speed8', 'def_tangential_speed9',
                        'def_tangential_speed10', 'def_tangential_speed11', 'off_Orientation_new1',
                        'off_Orientation_new2', 'off_Orientation_new3', 'off_Orientation_new4',
                        'off_Orientation_new5', 'off_Orientation_new6', 'off_Orientation_new7',
                        'off_Orientation_new8', 'off_Orientation_new9', 'off_Orientation_new10', 'off_Dir_new1',
                        'off_Dir_new2', 'off_Dir_new3', 'off_Dir_new4', 'off_Dir_new5', 'off_Dir_new6',
                        'off_Dir_new7', 'off_Dir_new8', 'off_Dir_new9', 'off_Dir_new10', 'off_dist_from_RB1',
                        'off_dist_from_RB2', 'off_dist_from_RB3', 'off_dist_from_RB4', 'off_dist_from_RB5',
                        'off_dist_from_RB6', 'off_dist_from_RB7', 'off_dist_from_RB8', 'off_dist_from_RB9',
                        'off_dist_from_RB10', 'off_ang_from_RB1', 'off_ang_from_RB2', 'off_ang_from_RB3',
                        'off_ang_from_RB4', 'off_ang_from_RB5', 'off_ang_from_RB6', 'off_ang_from_RB7',
                        'off_ang_from_RB8', 'off_ang_from_RB9', 'off_ang_from_RB10', 'off_radial_speed1',
                        'off_radial_speed2', 'off_radial_speed3', 'off_radial_speed4', 'off_radial_speed5',
                        'off_radial_speed6', 'off_radial_speed7', 'off_radial_speed8', 'off_radial_speed9',
                        'off_radial_speed10', 'off_tangential_speed1', 'off_tangential_speed2',
                        'off_tangential_speed3', 'off_tangential_speed4', 'off_tangential_speed5',
                        'off_tangential_speed6', 'off_tangential_speed7', 'off_tangential_speed8',
                        'off_tangential_speed9', 'off_tangential_speed10', 'GameId', 'Team', 'X', 'Y', 'X_new',
                        'Y_new', 'RB_Dis_YL', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'NflId', 'DisplayName',
                        'JerseyNumber', 'Season', 'YardLine', 'Quarter', 'GameClock', 'PossessionTeam', 'Down',
                        'Distance', 'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'NflIdRusher',
                        'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel',
                        'PlayDirection', 'TimeHandoff', 'TimeSnap', 'Yards', 'PlayerHeight', 'PlayerWeight',
                        'PlayerBirthDate', 'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
                        'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather', 'Temperature',
                        'Humidity', 'WindSpeed', 'WindDirection', 'ToLeft', 'IsBallCarrier', 'YardsFromOwnGoal',
                        'side', 'Orientation_new', 'Dir_new']
    expected_df = pd.Series(expected, expected_headers)
    np.testing.assert_approx_equal(output.iloc[0]["def_Orientation_new1"], expected_df["def_Orientation_new1"])
    np.testing.assert_approx_equal(output.iloc[0]["def_Dir_new1"], expected_df.loc["def_Dir_new1"])
    np.testing.assert_approx_equal(output.iloc[0]["def_dist_from_RB1"], expected_df.loc["def_dist_from_RB1"])
    np.testing.assert_approx_equal(output.iloc[0]["def_ang_from_RB1"], expected_df.loc["def_ang_from_RB1"])
    np.testing.assert_approx_equal(output.iloc[0]["def_radial_speed1"], expected_df.loc["def_radial_speed1"])
    np.testing.assert_approx_equal(output.iloc[0]["def_tangential_speed1"], expected_df.loc["def_tangential_speed1"])
    np.testing.assert_approx_equal(output.iloc[0]["X_new"], expected_df.loc["X_new"])
    np.testing.assert_approx_equal(output.iloc[0]["Y_new"], expected_df.loc["Y_new"])
    np.testing.assert_approx_equal(output.iloc[0]["RB_Dis_YL"], expected_df.loc["RB_Dis_YL"])
    np.testing.assert_approx_equal(output.iloc[0]["YardLine"], expected_df.loc["YardLine"])
    np.testing.assert_approx_equal(output.iloc[0]["Quarter"], expected_df.loc["Quarter"])
    np.testing.assert_approx_equal(output.iloc[0]["Down"], expected_df.loc["Down"])
    np.testing.assert_approx_equal(output.iloc[0]["Distance"], expected_df.loc["Distance"])


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import pandas as pd

import CleanData

pd.set_option('display.max_columns', None, 'display.max_rows', None)

input_df = pd.read_csv("data/train.csv", header=0)
df = CleanData.clean_data(input_df.head(22 * 10))
output, y_train = CleanData.convert_data(df)
y_train = y_train.values


# def convert_to_rect(R, theta, dx, dy):
#     x = []
#     y = []
#     for r, t in zip(R, theta):
#         x.append(r * math.cos(t) + dx)
#         y.append(r * math.sin(t) + dy)
#     return x, y
#
#
# def test_graph():
#     for i in range(10):
#         raw = df.iloc[22 * i: 22 * (i + 1), ]
#         converted = pd.DataFrame(output.iloc[i]).transpose()
#         print(f'Play: {i}')
#         print(f'converted.iloc[0]: {converted.iloc[0]}')
#         dx = converted.iloc[0]["X_new"]
#         dy = converted.iloc[0]["Y_new"]
#         x = raw["X"].to_list()
#         y = raw["Y"].to_list()
#         plt.scatter(x, y, label='Actual')
#         plt.xlim(0, 100)
#         plt.ylim(0, 160 / 3)
#         def_r = []
#         off_r = []
#         def_theta = []
#         off_theta = []
#         for c in converted.columns:
#             if "def_dist_from_RB" in c:
#                 def_r.append(converted.iloc[0][c])
#             if "off_dist_from_RB" in c:
#                 off_r.append(converted.iloc[0][c])
#             if "def_ang_from_RB" in c:
#                 def_theta.append(converted.iloc[0][c])
#             if "off_ang_from_RB" in c:
#                 off_theta.append(converted.iloc[0][c])
#         x_new, y_new = convert_to_rect(def_r, def_theta, dx, dy)
#         print(f'x_new: {x_new}')
#         print(f'y_new: {y_new}')
#         plt.scatter(x_new, y_new, label='Converted Defense', marker='o')
#         x_new, y_new = convert_to_rect(off_r, off_theta, dx, dy)
#         plt.scatter(x_new, y_new, label='Converted Offense', marker='s')
#         plt.plot([50, 50], [0, 160 / 3])
#         plt.title(f'play {i}')
#         plt.legend()
#         plt.show()
#         print('\n\n\n')


def test_clean_raw_data():
    np.testing.assert_array_equal(df["VisitorTeamAbbr"].head(), ["KC", "KC", "KC", "KC", "KC"])


def test_add_play_direction():
    np.testing.assert_array_equal(df["ToLeft"].head(), [True, True, True, True, True])


def test_add_team_on_offense():
    np.testing.assert_array_equal(df["TeamOnOffense"].head(), ["home", "home", "home", "home", "home"])


def test_is_on_offense():
    np.testing.assert_array_equal(df["IsOnOffense"].head(), [False, False, False, False, False])


def test_add_ball_carrier():
    np.testing.assert_array_equal(df["IsBallCarrier"].head(), [False, False, False, False, False])


def test_add_yards_to_end_zone():
    np.testing.assert_array_equal(df["YardsToEndZone"].head(), [65, 65, 65, 65, 65])


def test_add_yards_from_own_goal():
    np.testing.assert_array_equal(df["YardsFromOwnGoal"].head(), [35, 35, 35, 35, 35])


def test_x_std():
    np.testing.assert_almost_equal(df["X_std"].head(), [36.09, 35.33, 36., 38.54, 40.68])


def test_y_std():
    np.testing.assert_almost_equal(df["Y_std"].head(), [18.4933333, 20.6933333, 20.1333333, 25.6333333, 17.9133333])


def test_dir_std_1():
    np.testing.assert_almost_equal(df["Dir_std_1"].head(), [177.18, 198.7, 202.73, 105.64, 164.31])


def test_dir_std_2():
    np.testing.assert_almost_equal(df["Dir_std_2"].head(), [-2.82, 18.7, 22.73, -74.36, -15.69])


def test_x_std_end():
    np.testing.assert_almost_equal(df["X_std_end"].head(), [36.0068547, 35.4646575, 36.4713946, 38.1355507, 40.187813])


def test_y_std_end():
    np.testing.assert_almost_equal(df["Y_std_end"].head(), [20.1812868, 21.0911616, 21.2585831, 25.746562, 19.6655182])


def test_add_side():
    np.testing.assert_array_equal(df["side"].head(), ['defense', 'defense', 'defense', 'defense', 'defense'])


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


def test_y_train():
    np.testing.assert_equal(len(y_train[0]), 115)
    np.testing.assert_equal(y_train[0][22], 0)
    np.testing.assert_equal(y_train[0][23], 1)


def test_pid_works():
    print(f'output: {output}')
    np.testing.assert_approx_equal(output.iloc[0]["def_dist_from_RB1"], 4.593310)
    np.testing.assert_approx_equal(output.iloc[0]["def_ang_from_RB1"], -0.4772787023711037)
    np.testing.assert_approx_equal(output.iloc[0]["def_radial_speed1"], 0.41522706231045375)
    np.testing.assert_approx_equal(output.iloc[0]["def_tangential_speed1"], 0.06313863100377297)
    np.testing.assert_approx_equal(output.iloc[0]["X_new"], 34.5594359549917)
    np.testing.assert_approx_equal(output.iloc[0]["Y_new"], 24.294820396571957)
    np.testing.assert_approx_equal(output.iloc[0]["RB_Dis_YL"], 0.44056404500830126)
    np.testing.assert_approx_equal(output.iloc[0]["YardLine"], 35)
    np.testing.assert_approx_equal(output.iloc[0]["Quarter"], 1)
    np.testing.assert_approx_equal(output.iloc[0]["Down"], 3)
    np.testing.assert_approx_equal(output.iloc[0]["Distance"], 2)
    np.testing.assert_approx_equal(output.iloc[0]["YardsToEndZone"], 65)


if __name__ == '__main__':
    unittest.main()

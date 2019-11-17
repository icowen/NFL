import math
import numpy as np
import random

import pandas as pd

import CleanData

pd.set_option('display.max_columns', None, 'display.max_rows', None)

input_df = pd.read_csv("data/train.csv", header=0).head(22 * 20)
df = CleanData.clean_data(input_df)
output, y_train, cumsum = CleanData.convert_data(df)
y_train = y_train.values


def logit(p):
    return math.log(p / (1 - p))


def inverse_logit(x):
    return math.pow(math.e, x) / (1 + math.pow(math.e, x))


def test_get_yard_dist():
    print(f'cumsum: {cumsum}')
    np.testing.assert_equal(len(cumsum), 115)
    # cumsum = df["Yards"].value_counts(normalize=True).sort_index().cumsum()
    # for i in range(-15, 100):
    #     j = i
    #     while j not in cumsum.index and j > -15:
    #         j -= 1
    #     if i == -15:
    #         cumsum[i] = 0
    #     cumsum[i] = cumsum[j]
    # cumsum = cumsum.sort_index()
    o = []
    fake_output = [random.random() for _ in range(115)]
    for i in range(115):
        if fake_output[i] == 0:
            inside = cumsum[i - 15] + float('-inf')
        else:
            inside = cumsum[i - 15] + logit(fake_output[i])
        o.append(inverse_logit(inside))

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
#         plt.scatter(x_new, y_new, label='Converted Defense', marker='o')
#         x_new, y_new = convert_to_rect(off_r, off_theta, dx, dy)
#         plt.scatter(x_new, y_new, label='Converted Offense', marker='s')
#         plt.grid(b=True)
#         plt.title(f'play {i}')
#         plt.legend()
#         plt.show()
#
#
# def test_clean_raw_data():
#     np.testing.assert_array_equal(df["VisitorTeamAbbr"].head(), ["KC", "KC", "KC", "KC", "KC"])
#
#
# def test_add_play_direction():
#     np.testing.assert_array_equal(df["ToLeft"].head(), [True, True, True, True, True])
#
#
# def test_add_team_on_offense():
#     np.testing.assert_array_equal(df["TeamOnOffense"].head(), ["home", "home", "home", "home", "home"])
#
#
# def test_is_on_offense():
#     np.testing.assert_array_equal(df["IsOnOffense"].head(), [False, False, False, False, False])
#
#
# def test_add_ball_carrier():
#     np.testing.assert_array_equal(df["IsBallCarrier"].head(), [False, False, False, False, False])
#
#
# def test_add_yards_to_end_zone():
#     np.testing.assert_array_equal(df["YardsToEndZone"].head(), [65, 65, 65, 65, 65])
#
#
# def test_add_yards_from_own_goal():
#     np.testing.assert_array_equal(df["YardsFromOwnGoal"].head(), [35, 35, 35, 35, 35])
#
#
# def test_x_std():
#     np.testing.assert_almost_equal(df["X_std"].head(), [36.09, 35.33, 36., 38.54, 40.68])
#
#
# def test_y_std():
#     np.testing.assert_almost_equal(df["Y_std"].head(), [18.4933333, 20.6933333, 20.1333333, 25.6333333, 17.9133333])
#
#
# def test_dir_std_1():
#     np.testing.assert_almost_equal(df["Dir_std_1"].head(), [177.18, 198.7, 202.73, 105.64, 164.31])
#
#
# def test_dir_std_2():
#     np.testing.assert_almost_equal(df["Dir_std_2"].head(), [-2.82, 18.7, 22.73, -74.36, -15.69])
#
#
# def test_x_std_end():
#     np.testing.assert_almost_equal(df["X_std_end"].head(), [36.0068547, 35.4646575, 36.4713946, 38.1355507, 40.187813])
#
#
# def test_y_std_end():
#     np.testing.assert_almost_equal(df["Y_std_end"].head(), [20.1812868, 21.0911616, 21.2585831, 25.746562, 19.6655182])
#
#
# def test_add_side():
#     np.testing.assert_array_equal(df["side"].head(), ['defense', 'defense', 'defense', 'defense', 'defense'])
#
#
# def test_yard_line():
#     np.testing.assert_array_equal(df["YardLine"].head(), [35, 35, 35, 35, 35])
#
#
# def test_quarter():
#     np.testing.assert_array_equal(df["Quarter"].head(), [1, 1, 1, 1, 1])
#
#
# def test_game_clock():
#     np.testing.assert_array_equal(df["GameClock"].head(),
#                                   ["14:14:00", "14:14:00", "14:14:00", "14:14:00", "14:14:00"])
#
#
# def test_down():
#     np.testing.assert_array_equal(df["Down"].head(), [3, 3, 3, 3, 3])
#
#
# def test_distance():
#     np.testing.assert_array_equal(df["Distance"].head(), [2, 2, 2, 2, 2])
#
#
# def test_y_train():
#     np.testing.assert_equal(len(y_train[0]), 115)
#     np.testing.assert_equal(y_train[0][22], 0)
#     np.testing.assert_equal(y_train[0][23], 1)
#
#
# def test_pid_works():
#     print(f'output: {output}')
#     np.testing.assert_approx_equal(output.iloc[0]["def_dist_from_RB1"], 4.593310)
#     np.testing.assert_approx_equal(output.iloc[0]["def_ang_from_RB1"], -0.4772787023711037)
#     np.testing.assert_approx_equal(output.iloc[0]["def_radial_speed1"], 0.41522706231045375)
#     np.testing.assert_approx_equal(output.iloc[0]["def_tangential_speed1"], 0.0631386310037747)
#     np.testing.assert_approx_equal(output.iloc[0]["X_new"], 31.25)
#     np.testing.assert_approx_equal(output.iloc[0]["Y_new"], 22.803333333333335)
#     np.testing.assert_approx_equal(output.iloc[0]["RB_Dis_YL"], 3.75)
#     np.testing.assert_approx_equal(output.iloc[0]["YardLine"], 35)
#     np.testing.assert_approx_equal(output.iloc[0]["Quarter"], 1)
#     np.testing.assert_approx_equal(output.iloc[0]["Down"], 3)
#     np.testing.assert_approx_equal(output.iloc[0]["Distance"], 2)
#     np.testing.assert_approx_equal(output.iloc[0]["YardsToEndZone"], 65)

#
# if __name__ == '__main__':
#     unittest.main()

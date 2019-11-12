import math

import pandas as pd


def clean_team_names(x):
    if x == "ARI": return "ARZ"
    if x == "BAL": return "BLT"
    if x == "CLE": return "CLV"
    if x == "HOU": return "HST"
    return x


def get_side(x):
    if x["PossessionTeam"] == x["HomeTeamAbbr"] and x["Team"] == "home":
        return "offense"
    if x["PossessionTeam"] == x["HomeTeamAbbr"] and x["Team"] == "away":
        return "defense"
    if x["PossessionTeam"] != x["HomeTeamAbbr"] and x["Team"] == "home":
        return "defense"
    if x["PossessionTeam"] != x["HomeTeamAbbr"] and x["Team"] == "away":
        return "offense"


def get_all_pids(df):
    return df["PlayId"].unique()


def combine_by_pid(pid, df):
    df_pid = df.loc[df["PlayId"] == pid]
    running_back = df_pid.loc[df_pid["NflId"] == df_pid["NflIdRusher"]]
    running_back_coords = [running_back.iloc[0]["X_std"], running_back.iloc[0]["Y_std"]]

    df_pid = df_pid.loc[df_pid["NflId"] != df_pid["NflIdRusher"]]
    df_pid.loc[:, "dist_from_RB"] = df_pid.apply(
        lambda x: math.sqrt((x["X_std"] - running_back_coords[0]) ** 2 +
                            (x["Y_std"] - running_back_coords[1]) ** 2), axis=1)
    df_pid.loc[:, "ang_from_RB"] = df_pid.apply(
        lambda x: math.acos((x["X_std"] - running_back_coords[0]) / x["dist_from_RB"])
        if x["Y_std"] > running_back_coords[1]
        else -math.acos((x["X_std"] - running_back_coords[0]) / x["dist_from_RB"]),
        axis=1)
    df_pid.loc[:, "radial_speed"] = df_pid.apply(
        lambda x: x["S"] * math.cos(x["ang_from_RB"] - x["Dir_std_2"]), axis=1)
    df_pid.loc[:, "tangential_speed"] = df_pid.apply(
        lambda x: x["S"] * math.sin(x["ang_from_RB"] - x["Dir_std_2"]), axis=1)

    df_def = df_pid.loc[df_pid["side"] == "defense"]
    df_off = df_pid.loc[df_pid["side"] == "offense"]

    off_sorted = df_off.sort_values(by=["dist_from_RB"])
    def_sorted = df_def.sort_values(by=["dist_from_RB"])
    off_data = off_sorted[
        ["Orientation", "Dir_std_2", "dist_from_RB", "ang_from_RB", "radial_speed", "tangential_speed"]]
    def_data = def_sorted[
        ["Orientation", "Dir_std_2", "dist_from_RB", "ang_from_RB", "radial_speed", "tangential_speed"]]
    return off_data, def_data


def get_team_on_offense(x):
    return "home" if x["PossessionTeam"] == x["HomeTeamAbbr"] else "away"


def get_is_on_offense(x):
    return x["Team"] == x["TeamOnOffense"]


def get_x_std(x):
    return 120 - x["X"] - 10 if x["ToLeft"] else x["X"] - 10


def get_y_std(x):
    return 160 / 3 - x["Y"] if x["ToLeft"] else x["Y"]


def get_dir_std_1(x):
    return x["Dir"] + 360 if x["ToLeft"] and x["Dir"] < 90 else x["Dir"]


def get_dir_std_1_part_two(x):
    return x["Dir"] - 360 if (not x["ToLeft"]) and x["Dir"] > 270 else x["Dir_std_1"]


def get_dir_std_2(x):
    return x["Dir_std_1"] - 180 if x["ToLeft"] else x["Dir_std_1"]


def get_x_std_end(x):
    return x["S"] * math.cos((90 - x["Dir_std_2"]) * math.pi / 180) + x["X_std"]


def get_y_std_end(x):
    return x["S"] * math.sin((90 - x["Dir_std_2"]) * math.pi / 180) + x["Y_std"]


def clean_data(df):
    df.loc[:, "VisitorTeamAbbr"] = df["VisitorTeamAbbr"].map(clean_team_names)
    df.loc[:, "HomeTeamAbbr"] = df["HomeTeamAbbr"].map(clean_team_names)
    df.loc[:, "ToLeft"] = df.apply(lambda x: x["PlayDirection"] == "left", axis=1)
    df.loc[:, "IsBallCarrier"] = df.apply(lambda x: x["NflId"] == x["NflIdRusher"], axis=1)
    df.loc[:, "YardsFromOwnGoal"] = df.apply(
        lambda x: x["YardLine"]
        if x["FieldPosition"] == x["PossessionTeam"]
        else 50 - x["YardLine"], axis=1)
    df.loc[:, "YardsFromOwnGoal"] = df.apply(
        lambda x: 50
        if x["YardLine"] == 50
        else x["YardsFromOwnGoal"], axis=1)
    df.loc[:, "side"] = df.apply(lambda x: get_side(x), axis=1)
    df.loc[:, "TeamOnOffense"] = df.apply(lambda x: get_team_on_offense(x), axis=1)
    df.loc[:, "IsOnOffense"] = df.apply(lambda x: get_is_on_offense(x), axis=1)
    df.loc[:, "X_std"] = df.apply(lambda x: get_x_std(x), axis=1)
    df.loc[:, "Y_std"] = df.apply(lambda x: get_y_std(x), axis=1)
    df.loc[:, "Dir_std_1"] = df.apply(lambda x: get_dir_std_1(x), axis=1)
    df.loc[:, "Dir_std_1"] = df.apply(lambda x: get_dir_std_1_part_two(x), axis=1)
    df.loc[:, "Dir_std_2"] = df.apply(lambda x: get_dir_std_2(x), axis=1)
    df.loc[:, "X_std_end"] = df.apply(lambda x: get_x_std_end(x), axis=1)
    df.loc[:, "Y_std_end"] = df.apply(lambda x: get_y_std_end(x), axis=1)
    df = df.dropna(subset=['Dir'])
    df = df.dropna(subset=['Orientation'])

    # df.loc[:, "Orientation_new"] = df["Orientation"] + 90
    # df.loc[:, "Dir_new"] = df["Dir"] + 90
    # df.loc[df["PlayDirection"] == "left", "X"] = 120 - df.loc[df["PlayDirection"] == "left", "X"] - 10
    # df.loc[df["PlayDirection"] == "left", "Y"] = 160/3 - df.loc[df["PlayDirection"] == "left", "Y"]
    # df.loc[df["PlayDirection"] == "left", "Dir_new"] = df.loc[df["PlayDirection"] == "left", "Dir_new"] + 180
    # df.loc[df["Dir_new"] > 360, "Dir_new"] = df.loc[df["Dir_new"] > 360, "Dir_new"] - 360
    # df.loc[df["PlayDirection"] == "left", "Orientation_new"] = df.loc[df[
    #                                                                       "PlayDirection"] == "left", "Orientation_new"] + 180
    # df.loc[df["Orientation_new"] > 360, "Orientation_new"] = df.loc[
    #                                                              df["Orientation_new"] > 360, "Orientation_new"] - 360
    # df.loc[:, "Orientation_new"] = df["Orientation_new"] / 180 * math.pi
    # df.loc[:, "Dir_new"] = df["Dir_new"] / 180 * math.pi
    # df.loc[df["PlayDirection"] == "left", "PlayDirection"] = "right"
    return df


def get_output_data(df):
    pids = get_all_pids(df)
    data_by_game = pd.DataFrame()
    for pid in pids:
        off_data, def_data = combine_by_pid(pid, df)
        for col in off_data.columns:
            off_data = off_data.rename(columns={col: "off_" + col})
            def_data = def_data.rename(columns={col: "def_" + col})
        off_data = off_data.unstack().to_frame().reset_index(level=1, drop=True).T
        def_data = def_data.unstack().to_frame().reset_index(level=1, drop=True).T
        new_cols = []
        for col in off_data.columns.unique():
            for i in range(1, 11):
                new_cols.append(col + str(i))
        try:
            off_data.columns = new_cols
            new_cols = []
            for col in def_data.columns.unique():
                for i in range(1, 12):
                    new_cols.append(col + str(i))
            def_data.columns = new_cols
            play = pd.concat([off_data, def_data], axis=1)
            rusher_data = df.loc[(df["PlayId"] == pid) & (df["NflId"] == df["NflIdRusher"])].reset_index()
            rusher_data.loc[:, "X_new"] = rusher_data.apply(lambda x: 100 - x["X"] if x["X"] > 50 else x["X"], axis=1)
            rusher_data.loc[:, "Y_new"] = rusher_data.apply(lambda x: 160/3 - x["Y"] if x["X"] > 50 else x["Y"], axis=1)
            rusher_data.loc[:, "RB_Dis_YL"] = rusher_data.apply(lambda x: abs(x["X_new"] - x["YardLine"]), axis=1)
            play = pd.concat([play, rusher_data], axis=1)
            data_by_game = data_by_game.append(play)
        except ValueError:
            continue
    return data_by_game


def convert_to_training_values(data_by_game):
    keep = ['dist_from_RB', 'ang_from_RB', 'X_new', 'Y_new',
            'RB_Dis_YL', 'tangential_speed', 'radial_speed',
            "YardLine", "Quarter", "Down", "Distance"]
    yards_gained = data_by_game["Yards"]
    y_train = []
    for play in yards_gained.values:
        y = [0 for i in range(199)]
        for i in range(99 + play, 199):
            y[i] = 1
        y_train.append(y)
    for c in data_by_game.columns:
        if all(k not in c for k in keep):
            data_by_game = data_by_game.drop(c, axis=1)
    return data_by_game, pd.DataFrame(y_train)


def convert_data(df):
    df = clean_data(df)
    df = get_output_data(df)
    x_train, y_train = convert_to_training_values(df)
    # x_train = pd.DataFrame(x_train)
    # x_train.to_csv(r'x_train.csv', index=None)
    # y_train = pd.DataFrame(y_train)
    # y_train.to_csv(r'y_train.csv', index=None)
    return x_train, y_train

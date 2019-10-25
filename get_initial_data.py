import csv
import pandas as pd


d = pd.read_csv('peter_data2.csv')
d = d.drop(d.columns[0], axis=1)
keep = ['dist_from_RB', 'ang_from_RB', 'X_new', 'Y_new',
        'RB_Dis_YL', 'tangential_speed', 'radial_speed']
yards_gained = d["Yards"]
with open('dist_ang_radial_tang_x_y_disfromyl_yards.csv', 'w', newline='') as f:
    c = csv.writer(f, delimiter=',')
    for play in yards_gained.values:
        y = [0 for i in range(199)]
        for i in range(99 + play, 199):
            y[i] = 1
        c.writerow(y)
# for c in d.columns:
#     if all(k not in c for k in keep):
#         d = d.drop(c, axis=1)
# data = d.values
# with open('dist_ang_radial_tang_x_y_disfromyl.csv', 'w', newline='') as f:
#     c = csv.writer(f, delimiter=',')
#     for play in data:
#         c.writerow(play)

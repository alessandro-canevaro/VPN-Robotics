import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "arial"

def single_train():
    huawei_red = "#c7000a"
    huawei_pink = "#ea5a4f"
    huawei_light_pink = "#e98c80"
    huawei_dark_red = "#9f0001"
    flare_red = "#d14a61"
    flare_pink = "#e98d6b"
    #print(sns.color_palette("flare").as_hex())

    columns = ["Wall time","Step","Value"]
    font_size = 20

    name = "single_value_loss"
    y_lab = "Value loss"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value > 0.1, 0.03, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)

    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "single_policy_loss"
    y_lab = "Policy loss"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value > 0.1, 0.03, df.Value)
    df['Value'] = np.absolute(df['Value'])
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    plt.ylim([-0.002, 0.052])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "single_entropy_loss"
    y_lab = "Entropy loss"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -1, -1, df.Value)
    df['Value'] = -df['Value']
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    plt.ylim([0.12, 0.85])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "single_exp_var"
    y_lab = "Explained variance"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()

#single_train()

def single_perf():
    huawei_red = "#c7000a"
    huawei_pink = "#ea5a4f"
    huawei_light_pink = "#e98c80"
    huawei_dark_red = "#9f0001"
    flare_red = "#d14a61"
    flare_pink = "#e98d6b"
    print(sns.color_palette("mako").as_hex())

    columns = ["Wall time","Step","Value"]
    font_size = 20

    name = "single_len_mean"
    y_lab = "Mean episode length"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.vlines(x=1.86e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red, label="Actual")


    name = "single_opt_length"
    y_lab = "Mean episode length"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color='#348fa7')
    plt.plot(df.Step, df.Smooth, linewidth=3, color='#37659e', label="Optimal")
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')

    plt.ylim([-1, 51])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    print("ok")
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "single_rew_mean"
    y_lab = "Mean episode reward"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.vlines(x=1.86e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([0.39, 1.01])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "single_rel_diff"
    y_lab = "Relative difference"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.vlines(x=1.86e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([-0.1, 2.1])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "single_max_dist"
    y_lab = "Region-growing max. dist."
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)

    plt.vlines(x=1.86e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([-1, 66])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()

#single_perf()

def door_train():
    huawei_red = "#c7000a"
    huawei_pink = "#ea5a4f"
    huawei_light_pink = "#e98c80"
    huawei_dark_red = "#9f0001"
    flare_red = "#d14a61"
    flare_pink = "#e98d6b"
    #print(sns.color_palette("flare").as_hex())

    columns = ["Wall time","Step","Value"]
    font_size = 20

    name = "door_value_loss"
    y_lab = "Value loss"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value > 0.1, 0.03, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)

    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "door_policy_loss"
    y_lab = "Policy loss"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value > 0.1, 0.03, df.Value)
    df['Value'] = np.absolute(df['Value'])
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.ylim([-0.002, 0.052])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "door_entropy_loss"
    y_lab = "Entropy loss"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -1, -1, df.Value)
    df['Value'] = -df['Value']
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.ylim([0.12, 0.85])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "door_exp_var"
    y_lab = "Explained variance"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=5)

    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.ylim([-0.02, 1.02])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()

#door_train()


def coop_perf():
    huawei_red = "#c7000a"
    huawei_pink = "#ea5a4f"
    huawei_light_pink = "#e98c80"
    huawei_dark_red = "#9f0001"
    flare_red = "#d14a61"
    flare_pink = "#e98d6b"
    print(sns.color_palette("mako").as_hex())

    columns = ["Wall time","Step","Value"]
    font_size = 20

    name = "coop_len_mean"
    y_lab = "Mean episode length"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=10)

    plt.vlines(x=6.12e06, ymin=-100, ymax=1000, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red, label="Actual")


    name = "coop_opt_length"
    y_lab = "Mean episode length"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=10)

    plt.plot(df.Step, df.Value, alpha=0.5, color='#348fa7')
    plt.plot(df.Step, df.Smooth, linewidth=3, color='#37659e', label="Optimal")
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.ylim([-1, 91])
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "coop_rew_mean"
    y_lab = "Mean episode reward"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=10)

    plt.vlines(x=6.12e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([-0.31, 0.91])    
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "coop_rel_diff"
    y_lab = "Relative difference"
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=10)

    plt.vlines(x=6.12e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([0.9, 9.1])
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()

    name = "coop_delta_q"
    y_lab = "Delta Q"
    df = pd.read_csv(name+".csv", usecols=columns)
    #df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)
    df["Smooth"] = gaussian_filter1d(df.Value, sigma=10)

    plt.vlines(x=6.12e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, alpha=0.5, color=flare_pink)
    plt.plot(df.Step, df.Smooth, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([0.095, 0.185])
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    import matplotlib
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 4, forward=True)
    fig.savefig(name+".pdf")

    #plt.savefig(name+".pdf")
    plt.show()
    plt.clf()


    name = "coop_max_dist"
    y_lab = "Region-growing max. dist."
    df = pd.read_csv(name+".csv", usecols=columns)
    df['Value'] = np.where(df.Value < -0.5, -0.5, df.Value)

    plt.vlines(x=6.12e06, ymin=-100, ymax=100, linewidth=2, linestyle="dashed", color='gray')
    plt.plot(df.Step, df.Value, linewidth=3, color=flare_red)
    #plt.hlines(y=0, xmin=0, xmax=5e06, linewidth=2, color='gray')
    plt.ylim([-1, 68])
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    plt.xlabel("Steps", fontsize=font_size)
    plt.ylabel(y_lab, fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    plt.savefig(name+".pdf")
    plt.show()
    plt.clf()

#coop_perf()

def path_plot():
    from matplotlib.patches import Rectangle
    r1 = np.loadtxt("run_data_robot1_ideal.txt")
    r2 = np.loadtxt("run_data_robot2_ideal.txt")
    x1, y1 = r1[3, :], r1[4, :]
    x1 = np.append(x1, x1[-1])
    y1 = np.append(y1, y1[-1])
    x2, y2 = r2[3, :], r2[4, :]
    x2 = np.append(x2, x2[-1])
    y2 = np.append(y2, y2[-1])
    flare_red = "#d14a61"
    plt.gca().add_patch(Rectangle((-0.15, -0.15), 19*0.3, 7*0.3, color='black'))
    plt.gca().add_patch(Rectangle((-0.15+17*0.3, -0.15+7*0.3), 2*0.3, 2*0.3, color='black'))
    plt.gca().add_patch(Rectangle((-0.15, -0.15), 2*0.3, 30*0.3, color='black'))
    plt.gca().add_patch(Rectangle((-0.15+2*0.3, -0.15+14*0.3), 1*0.3, 5*0.3, color='black'))
    plt.gca().add_patch(Rectangle((-0.15+11*0.3, -0.15+14*0.3), 1*0.3, 2*0.3, color='black'))
    plt.gca().add_patch(Rectangle((-0.15+12*0.3, -0.15+14*0.3), 1*0.3, 7*0.3, color='black'))
    plt.gca().add_patch(Rectangle((-0.15+16*0.3, -0.15+15*0.3), 5*0.3, 3*0.3, edgecolor=flare_red, fill=False, linewidth=2))

    
    plt.plot(x1, y1, linestyle="dashed", color=flare_red)
    plt.plot(x2, y2, linestyle="dashed", color=flare_red)
    for px1, py1, px2, py2 in zip(x1[::800], y1[::800], x2[::800], y2[::800]):
        plt.plot([px1, px2], [py1, py2], color="gray")
    plt.scatter(x1[::800], y1[::800], s=90, color="gray", facecolors="gray")
    plt.scatter(x2[::800], y2[::800], s=90, color="gray", facecolors="gray")
    
    font_size = 20
    major_ticks = np.arange(0-0.15, 9-0.15, 0.9)
    minor_ticks = np.arange(0-0.15, 9-0.15, 0.3)
    plt.xticks(major_ticks, fontsize=font_size)
    plt.xticks(minor_ticks, minor=True)
    plt.yticks(major_ticks, fontsize=font_size)
    plt.yticks(minor_ticks, minor=True)
    plt.grid(which='minor')
    plt.gca().set_aspect('equal')
    plt.xlabel("Meters", fontsize=font_size)
    plt.ylabel("Meters", fontsize=font_size)
    #plt.tight_layout()
    plt.xlim([0.3, 6.3])
    plt.ylim([1.8, 5.4])

    plt.show()
    #print(x1.shape)

#path_plot()

def global_error():
    r1 = np.loadtxt("run_data_robot1_ideal.txt")
    r2 = np.loadtxt("run_data_robot2_ideal.txt")
    x1, y1 = r1[3, :], r1[4, :]
    x2, y2 = r2[3, :], r2[4, :]
    x_ref, y_ref = (x1+x2)/2, (y1+y2)/2
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_id, y_id = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_odom.txt")
    r2 = np.loadtxt("run_data_robot2_odom.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_odo, y_odo = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_lidar.txt")
    r2 = np.loadtxt("run_data_robot2_lidar.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_lid, y_lid = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_lidar10.txt")
    r2 = np.loadtxt("run_data_robot2_lidar10.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_shi, y_shi = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_lidar5.txt")
    r2 = np.loadtxt("run_data_robot2_lidar5.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_li5, y_li5 = (x1+x2)/2, (y1+y2)/2

    flare_red = "#d14a61"
    mako_blue = '#37659e'
    viridis_green = '#1fa187'

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w', gridspec_kw={'height_ratios': [1, 1]})

    # plot the same data on both axes
    ax.plot(np.sqrt((x_ref-x_id)**2 + (y_ref-y_id)**2)*1000, linewidth=3, linestyle="solid", color=flare_red, label="Ideal")
    ax.plot(np.sqrt((x_ref-x_odo)**2 + (y_ref-y_odo)**2)*1000, linewidth=3, linestyle="solid", color=mako_blue, label="Odometry")
    ax.plot(np.sqrt((x_ref-x_lid)**2 + (y_ref-y_lid)**2)*1000, linewidth=3, linestyle="solid", color=viridis_green, label="AMCL")
    #ax.plot(np.sqrt((x_ref-x_shi)**2 + (y_ref-y_shi)**2), linestyle="solid", color=flare_red)
    #ax.plot(np.sqrt((x_ref-x_li5)**2 + (y_ref-y_li5)**2), linestyle="solid", color=mako_blue)
    ax2.plot(np.sqrt((x_ref-x_id)**2 + (y_ref-y_id)**2)*1000, linewidth=3, linestyle="solid", color=flare_red)
    ax2.plot(np.sqrt((x_ref-x_odo)**2 + (y_ref-y_odo)**2)*1000, linewidth=3, linestyle="solid", color=mako_blue)
    ax2.plot(np.sqrt((x_ref-x_lid)**2 + (y_ref-y_lid)**2)*1000, linewidth=3, linestyle="solid", color=viridis_green)
    #ax2.plot(np.sqrt((x_ref-x_shi)**2 + (y_ref-y_shi)**2), linestyle="solid", color=flare_red)
    #ax2.plot(np.sqrt((x_ref-x_li5)**2 + (y_ref-y_li5)**2), linestyle="solid", color=mako_blue)

    ax2.set_ylim(0, 35)
    ax.set_ylim(30, 400)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='lightgray', clip_on=False)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax.plot((-d, +d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    labels = [int(int(item.get_text())/100) if item.get_text() !='−2000' else -20 for item in ax2.get_xticklabels()]
    ax2.set_xticklabels(labels)

    font_size = 24
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()+[ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(font_size-2)
    ax.legend(loc="upper left", fontsize=font_size-2)
    f.supxlabel('Seconds', fontsize=font_size+2)
    f.supylabel('Global pos. err. (mm)', fontsize=font_size+2)
    plt.tight_layout()
    plt.savefig("global_error_odom_amcl"+".pdf")
    plt.show()
    
global_error()


def local_error():
    r1 = np.loadtxt("run_data_robot1_ideal.txt")
    r2 = np.loadtxt("run_data_robot2_ideal.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_id = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_odom.txt")
    r2 = np.loadtxt("run_data_robot2_odom.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_odo = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_lidar.txt")
    r2 = np.loadtxt("run_data_robot2_lidar.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_lid = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_lidar10.txt")
    r2 = np.loadtxt("run_data_robot2_lidar10.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_shi = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_lidar5.txt")
    r2 = np.loadtxt("run_data_robot2_lidar5.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_li5 = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)

    flare_red = "#d14a61"
    mako_blue = '#37659e'
    viridis_green = '#1fa187'

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w', gridspec_kw={'height_ratios': [1, 1]})

    # plot the same data on both axes
    ax.plot(local_id*500, linestyle="solid", linewidth=3, color=flare_red, label="Ideal")
    ax.plot(local_odo*500, linestyle="solid", linewidth=3, color=mako_blue, label="Odometry")
    ax.plot(local_lid*500, linestyle="solid", linewidth=3, color=viridis_green, label="AMCL")
    #ax.plot(local_shi*500, linestyle="solid", color=viridis_green)
    #ax.plot(local_li5*500, linestyle="solid", color=mako_blue)
    ax2.plot(local_id*500, linestyle="solid", linewidth=3, color=flare_red)
    ax2.plot(local_odo*500, linestyle="solid", linewidth=3, color=mako_blue)
    ax2.plot(local_lid*500, linestyle="solid", linewidth=3, color=viridis_green)
    #ax2.plot(local_shi*500, linestyle="solid", color=viridis_green)
    #ax2.plot(local_li5*500, linestyle="solid", color=mako_blue)

    ax2.set_ylim(0, 25)
    ax.set_ylim(20, 300)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='lightgray', clip_on=False)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax.plot((-d, +d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    labels = [int(int(item.get_text())/100) if item.get_text() !='−2000' else -20 for item in ax2.get_xticklabels()]
    ax2.set_xticklabels(labels)

    font_size = 24
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()+[ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(font_size-2)
    ax.legend(loc="upper left", fontsize=font_size-2)
    f.supxlabel('Seconds', fontsize=font_size+2)
    f.supylabel('Local pos. err. (mm)', fontsize=font_size+2)
    plt.tight_layout()
    plt.savefig("local_error_odom_amcl"+".pdf")
    plt.show()

local_error()


def global_error_lidars():
    r1 = np.loadtxt("run_data_robot1_ideal.txt")
    r2 = np.loadtxt("run_data_robot2_ideal.txt")
    x1, y1 = r1[3, :], r1[4, :]
    x2, y2 = r2[3, :], r2[4, :]
    x_ref, y_ref = (x1+x2)/2, (y1+y2)/2
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_id, y_id = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_odom.txt")
    r2 = np.loadtxt("run_data_robot2_odom.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_odo, y_odo = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_lidar.txt")
    r2 = np.loadtxt("run_data_robot2_lidar.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_lid, y_lid = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_lidar10.txt")
    r2 = np.loadtxt("run_data_robot2_lidar10.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_shi, y_shi = (x1+x2)/2, (y1+y2)/2
    r1 = np.loadtxt("run_data_robot1_lidar5.txt")
    r2 = np.loadtxt("run_data_robot2_lidar5.txt")
    x1, y1 = r1[0, :], r1[1, :]
    x2, y2 = r2[0, :], r2[1, :]
    x_li5, y_li5 = (x1+x2)/2, (y1+y2)/2

    flare_red = "#d14a61"
    mako_blue = '#37659e'
    viridis_green = '#1fa187'

    #f, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w', gridspec_kw={'height_ratios': [1, 1]})

    # plot the same data on both axes
    print("qui")
    #ax.plot(np.sqrt((x_ref-x_id)**2 + (y_ref-y_id)**2)*1000, linestyle="solid", color=flare_red)
    #ax.plot(np.sqrt((x_ref-x_odo)**2 + (y_ref-y_odo)**2)*1000, linestyle="solid", color=mako_blue)    
    plt.plot(np.sqrt((x_ref-x_lid)**2 + (y_ref-y_lid)**2)*1000, linewidth=3, linestyle="solid", color=flare_red, label="σ=0.01")
    plt.plot(np.sqrt((x_ref-x_li5)**2 + (y_ref-y_li5)**2)*1000, linewidth=3, linestyle="solid", color=mako_blue, label="σ=0.05")
    plt.plot(np.sqrt((x_ref-x_shi)**2 + (y_ref-y_shi)**2)*1000, linewidth=3, linestyle="solid", color=viridis_green, label="σ=0.10")
    
    plt.ylim(0, 50)
    #ax.set_ylim(30, 400)

    # hide the spines between ax and ax2
    #ax.spines['bottom'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    #ax.xaxis.tick_top()
    #ax.tick_params(axis='x',          # changes apply to the x-axis
    #which='both',      # both major and minor ticks are affected
    #bottom=False,      # ticks along the bottom edge are off
    #top=False,         # ticks along the top edge are off
    #labelbottom=False)
    #ax2.xaxis.tick_bottom()

    #d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    #kwargs = dict(transform=ax.transAxes, color='lightgray', clip_on=False)
    #ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    #ax.plot((-d, +d), (-d, +d), **kwargs)

    #kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    #ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    #ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    labels = [int(int(item.get_text())/100) if item.get_text() !='−2000' else -20 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)

    font_size = 22
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
             plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontsize(font_size)
    plt.xlabel('Seconds', fontsize=font_size+2)
    plt.ylabel('Global pos. err. (mm)', fontsize=font_size+2)
    plt.legend(loc="upper left", fontsize=font_size)
    plt.tight_layout()
    plt.savefig("global_error_lidars"+".pdf")
    plt.show()
    
#global_error_lidars()


def local_error_lidars():
    r1 = np.loadtxt("run_data_robot1_ideal.txt")
    r2 = np.loadtxt("run_data_robot2_ideal.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_id = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_odom.txt")
    r2 = np.loadtxt("run_data_robot2_odom.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_odo = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_lidar.txt")
    r2 = np.loadtxt("run_data_robot2_lidar.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_lid = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_lidar10.txt")
    r2 = np.loadtxt("run_data_robot2_lidar10.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_shi = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)
    r1 = np.loadtxt("run_data_robot1_lidar5.txt")
    r2 = np.loadtxt("run_data_robot2_lidar5.txt")
    diff_x = r1[0, :] - r2[0, :]
    diff_y = r1[1, :] - r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_li5 = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)

    flare_red = "#d14a61"
    mako_blue = '#37659e'
    viridis_green = '#1fa187'

    #f, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w', gridspec_kw={'height_ratios': [1, 1]})

    # plot the same data on both axes
    print("qui")
    #ax.plot(np.sqrt((x_ref-x_id)**2 + (y_ref-y_id)**2)*1000, linestyle="solid", color=flare_red)
    #ax.plot(np.sqrt((x_ref-x_odo)**2 + (y_ref-y_odo)**2)*1000, linestyle="solid", color=mako_blue)
    plt.plot(local_lid*500, linestyle="solid", linewidth=3, color=flare_red, label="σ=0.01")
    plt.plot(local_li5*500, linestyle="solid", linewidth=3, color=mako_blue, label="σ=0.05")
    plt.plot(local_shi*500, linestyle="solid", linewidth=3, color=viridis_green, label="σ=0.10")

    plt.ylim(0, 25)
    #ax.set_ylim(30, 400)

    # hide the spines between ax and ax2
    #ax.spines['bottom'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    #ax.xaxis.tick_top()
    #ax.tick_params(axis='x',          # changes apply to the x-axis
    #which='both',      # both major and minor ticks are affected
    #bottom=False,      # ticks along the bottom edge are off
    #top=False,         # ticks along the top edge are off
    #labelbottom=False)
    #ax2.xaxis.tick_bottom()

    #d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    #kwargs = dict(transform=ax.transAxes, color='lightgray', clip_on=False)
    #ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    #ax.plot((-d, +d), (-d, +d), **kwargs)

    #kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    #ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    #ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    labels = [int(int(item.get_text())/100) if item.get_text() !='−2000' else -20 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)

    font_size = 22
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
             plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontsize(font_size)
    plt.xlabel('Seconds', fontsize=font_size+2)
    plt.ylabel('Local pos. err. (mm)', fontsize=font_size+2)
    plt.legend(loc="upper left", fontsize=font_size)
    plt.tight_layout()
    plt.savefig("local_error_lidars"+".pdf")
    plt.show()

#local_error_lidars()

def controller_plots():
    def compute_reference_input(x, y, ds):
        x_dot, y_dot = np.gradient(x, ds), np.gradient(y, ds)
        x_dotdot, y_dotdot = np.gradient(x_dot, ds), np.gradient(y_dot, ds)
        theta = np.arctan2(y_dot, x_dot)
        v = np.sqrt(x_dot**2 + y_dot**2)
        w = (np.multiply(y_dotdot, x_dot) - np.multiply(x_dotdot, y_dot))/(x_dot**2 + y_dot**2)
        #w[0], w[1], w[-1], w[-2] = w[2], w[2], w[-3], w[-3]
        return theta, v, w
    
    #x, y, theta, x_ref ,y_ref, theta_ref, x_n, y_n, theta_n, u1, u2, e1, e2, e3, xref, yref, thetaref, vref, wref
    #print(self.old_mrs, self.new_mrs, self.theta_i, self.theta_f, self.k, self.block_size, s, self.dt, t, st, )
    #agent0 x 6, y 12; agent1 x 6, y9; theta_i=0, theta_f=0.327249, k=0.2, block_size=0.3
    data = np.loadtxt("logger_datarobot1.txt")[:, 300:]
    data_offset = np.loadtxt("logger_data_offrobot1.txt")[:, 300:]

    flare_red = "#d14a61"
    mako_blue = '#37659e'
    viridis_green = '#1fa187'

    x, y = data[0, :], data[1, :]
    x_off, y_off = data_offset[0, :], data_offset[1, :]
    x_ref, y_ref = data[3, :], data[4, :]
    plt.plot(x, y, linewidth=3, color=mako_blue, label="Actual")
    plt.plot(x_off, y_off, linewidth=3, color=viridis_green, label="Shifted")
    plt.plot(x_ref, y_ref, linewidth=3, linestyle="dashed", color=flare_red, label="Reference")
    font_size = 26
    plt.xticks(plt.gca().get_xticks(), labels=["", "0", "5", "10", "15", "20", "25", "30", ""],fontsize=font_size)
    plt.yticks(plt.gca().get_yticks(), labels=["", "-2", "-1", "0", "1", "2", "3", ""], fontsize=font_size)
    plt.xlabel("Centimeters", fontsize=font_size)
    plt.ylabel("Centimeters", fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    #plt.show()
    plt.savefig("ctrl_path"+".pdf")
    plt.clf()

    theta, theta_r, theta_off = np.rad2deg(data[2, :]), np.rad2deg(data[5, :]), np.rad2deg(data_offset[2, :])
    plt.plot(theta, linewidth=3, color=mako_blue, label="Actual")
    plt.plot(theta_off, linewidth=3, color=viridis_green, label="Shifted")
    plt.plot(theta_r, linewidth=3, linestyle="dashed", color=flare_red, label="Reference")

    labels = [int(int(item.get_text())/100) if item.get_text() !='−100' else -1 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)
    plt.xticks(fontsize=font_size)
    plt.yticks([0, -5, -10, -15, -20], fontsize=font_size)
    plt.ylim([-22.5, 2.5])
    plt.xlabel("Seconds", fontsize=font_size)
    plt.ylabel("Heading angle (deg)", fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    plt.savefig("ctrl_heading"+".pdf")
    #plt.show()
    plt.clf()

    __, v, omega = compute_reference_input(x, y, 0.01)
    v[102:105] = [v[101], v[101], v[101]]
    v[80:120] = gaussian_filter1d(v[80:120], sigma=3, mode="nearest")
    __, v_off, omega_off = compute_reference_input(x_off, y_off, 0.01)
    omega[0], omega_off[0] = 0, 0
    omega[-3:] = [omega[-3]]*3
    omega_off[-3:] = [omega_off[-3]]*3

    v_ref = data[17, :]
    plt.plot(v, linewidth=3, color=mako_blue, label="Actual")
    plt.plot(v_off, linewidth=3, color=viridis_green, label="Shifted")
    plt.plot(v_ref, linewidth=3, linestyle="dashed", color=flare_red, label="Reference")

    labels = [int(int(item.get_text())/100) if item.get_text() !='−100' else -1 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Seconds", fontsize=font_size)
    plt.ylabel("Linear velocity (m/s)", fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    plt.savefig("ctrl_linvel"+".pdf")
    #plt.show()
    plt.clf()

    omega_ref = data[18, :]
    omega_ref[-3:] = [omega_ref[-3]]*3
    plt.plot(omega, linewidth=3, color=mako_blue, label="Actual")
    plt.plot(omega_off, linewidth=3, color=viridis_green, label="Shifted")
    plt.plot(omega_ref, linewidth=3, linestyle="dashed", color=flare_red, label="Reference")

    labels = [int(int(item.get_text())/100) if item.get_text() !='−100' else -1 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Seconds", fontsize=font_size)
    plt.ylabel("Angular velocity (rad/s)", fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    #plt.show()
    plt.savefig("ctrl_angvel"+".pdf")
    plt.clf()

    e1, e2, e3 = data[11, :]*1000, data[12, :]*1000, np.rad2deg(data[13, :])
    e1[102:120] = [e1[101]]*18
    e1[102:480] = gaussian_filter1d(e1[102:480], sigma=30, mode="nearest")
    e1_off, e2_off, e3_off = data_offset[11, :]*1000, data_offset[12, :]*1000, np.rad2deg(data_offset[13, :])
    plt.plot(e1, linewidth=3, color=mako_blue, label="Actual e1")
    plt.plot(e2, linewidth=3, color=mako_blue, linestyle="dotted", label="Actual e2")
    plt.plot(e1_off, linewidth=3, color=viridis_green, label="Shifted e1")
    plt.plot(e2_off, linewidth=3, color=viridis_green, linestyle="dotted", label="Shifted e2")

    labels = [int(int(item.get_text())/100) if item.get_text() !='−100' else -1 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Seconds", fontsize=font_size)
    plt.ylabel("Cartesian error (mm)", fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    plt.savefig("ctrl_cartesian_err"+".pdf")
    #plt.show()
    plt.clf()

    plt.plot(e3, linewidth=3, color=mako_blue, label="Actual e3")
    plt.plot(e3_off, linewidth=3, color=viridis_green, label="Shifted e3")

    labels = [int(int(item.get_text())/100) if item.get_text() !='−100' else -1 for item in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Seconds", fontsize=font_size)
    plt.ylabel("Heading error (deg)", fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    plt.savefig("ctrl_heading_err"+".pdf")
    #plt.show()
    plt.clf()

#controller_plots()
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    baselineclosest_average_remaining_pallets = 1.36
    baselineclosest_std_remaining_pallets = 1.99
    baselineclosest_average_reward = -119.76
    baselineclosest_std_reward = 57.34

    baselinerandom_average_remaining_pallets = 14.98
    baselinerandom_std_remaining_pallets = 3.09
    baselinerandom_average_reward = -194.25
    baselinerandom_std_reward = 4.64

    average_remaining_pallets_qlearning = {"25000": 7.61, "50000": 9.13, "75000":11.84, "100000":8.62, "125000":7.62, "150000":7.65, "175000":7.09, "200000":10.78, "225000":7.93, "250000":8.68, "275000":11.71, "300000":7.73, "325000":11.14, "350000":11.29}
    std_remaining_pallets_qlearning = {"25000": 3.10, "50000":3.88, "75000":3.74, "100000":3.64, "125000":3.51, "150000":3.71, "175000":3.20, "200000":3.27, "225000":3.47, "250000":3.65, "275000":4.12, "300000":3.54, "325000":3.74, "350000":3.84}
    average_rewards_qlearning = {"25000": -179.92, "50000":-182.64, "75000":-187.98, "100000":-182.38, "125000":-179.34, "150000":-179.74, "175000":-177.16, "200000":-185.91, "225000":-180.61, "250000":-182.0, "275000":-187.53, "300000":-179.92, "325000":-186.64, "350000":-187.1}
    std_rewards_qlearning = {"25000": 5.71, "50000":7.78, "75000":6.77, "100000":6.44, "125000":7.76, "150000":7.35, "175000":14.48, "200000":8.7, "225000":6.87, "250000":6.77, "275000":7.4, "300000":7.86, "325000":7.24, "350000":7.43}

    average_remaining_pallets_a2c = {"25000": 7.34, "50000": 5.35, "75000":5.01, "100000":7.37, "125000":6.77, "150000":6.11, "175000":5.41, "200000":4.65, "225000":4.59, "250000":5.84, "275000":8.9, "300000":6.65, "325000":5.67, "350000":6.95}
    std_remaining_pallets_a2c = {"25000": 3.06, "50000":2.88, "75000":2.48, "100000":3.54, "125000":4.01, "150000":2.77, "175000":2.69, "200000":3.17, "225000":2.60, "250000":3.46, "275000":4.03, "300000":3.34, "325000":3.17, "350000":3.22}
    average_rewards_a2c = {"25000": -179.41, "50000":-175.53, "75000":-174.25, "100000":-180.07, "125000":-178.29, "150000":-176.43, "175000":-175.49, "200000":-173.36, "225000":-172.89, "250000":-176.17, "275000":-182.33, "300000":-177.66, "325000":-174.93, "350000":-178.39}
    std_rewards_a2c = {"25000": 6.82, "50000":5.84, "75000":8.99, "100000":7.28, "125000":8.44, "150000":8.19, "175000":7.16, "200000":12.89, "225000":11.97, "250000":9.20, "275000":8.10, "300000":7.30, "325000":13.24, "350000":10.16}

    average_remaining_pallets_ddpg = {"25000": 13.26, "50000": 14.77, "75000":14.62, "100000":14.69, "125000":14.56, "150000":13.63, "175000":14.56, "200000":14.13, "225000":13.61, "250000":14.35, "275000":14.43, "300000":13.89, "325000":13.89, "350000":14.44}
    std_remaining_pallets_ddpg = {"25000": 3.03, "50000":2.904, "75000":3.41, "100000":2.91, "125000":2.97, "150000":3.45, "175000":2.97, "200000":3.17, "225000":3.40, "250000":2.90, "275000":2.86, "300000":2.95, "325000":3.04, "350000":2.91}
    average_rewards_ddpg = {"25000": -190.68, "50000":-193.71, "75000":-193.59, "100000":-194.1, "125000":-193.16, "150000":-191.36, "175000":-193.26, "200000":-193.07, "225000":-191.56, "250000":-192.75, "275000":-192.97, "300000":-192.04, "325000":-192.07, "350000":-193.17}
    std_rewards_ddpg = {"25000": 5.44, "50000":4.91, "75000":5.52, "100000":4.58, "125000":5.56, "150000":6.04, "175000":4.85, "200000":5.47, "225000":5.34, "250000":5.03, "275000":5.13, "300000":5.21, "325000":5.26, "350000":5.21}


    s1 = pd.Series(average_remaining_pallets_qlearning, name="qlearning")
    s2 = pd.Series(average_remaining_pallets_a2c, name="a2c")
    s3 = pd.Series(average_remaining_pallets_ddpg, name="ddpg")
    df = pd.concat([s1, s2, s3], axis=1)

    err1 = pd.Series(std_remaining_pallets_qlearning, name="qlearning")
    err2 = pd.Series(std_remaining_pallets_a2c, name="a2c")
    err3 = pd.Series(std_remaining_pallets_ddpg, name="ddpg")
    df_err = pd.concat([err1, err2, err3], axis=1)

    ax = df.plot(title="Average remaining pallets as a function of training iterations", kind="bar",
                yerr=df_err, capsize=4)


    ax.set_xlabel("Thousands of iterations")
    ax.set_ylabel("Average remaining pallets")
    plt.errorbar(average_remaining_pallets_qlearning.keys(),
                 [baselineclosest_average_remaining_pallets for _ in average_remaining_pallets_qlearning.keys()],
                 yerr=baselinerandom_std_remaining_pallets,
                 capsize=4, ecolor="lime", color="lime")
    plt.errorbar(average_remaining_pallets_qlearning.keys(),
                 [baselinerandom_average_remaining_pallets for _ in average_remaining_pallets_qlearning.keys()],
                 yerr=baselineclosest_std_remaining_pallets,
                 capsize=4, ecolor="fuchsia", color="fuchsia")
    plt.legend(["Q-learning", "A2C", "DDPG", "Greedy-baseline", "Random-baseline"], bbox_to_anchor=(1.00001, 1), loc="upper left")

    steps = [int(int(k)/1000) for k in average_remaining_pallets_qlearning.keys()]
    ax.set_xticklabels(steps)

    plt.show()

    s1 = pd.Series(average_rewards_qlearning, name="qlearning")
    s2 = pd.Series(average_rewards_a2c, name="a2c")
    s3 = pd.Series(average_rewards_ddpg, name="ddpg")
    df = pd.concat([s1, s2, s3], axis=1)

    err1 = pd.Series(std_rewards_qlearning, name="qlearning")
    err2 = pd.Series(std_rewards_a2c, name="a2c")
    err3 = pd.Series(std_rewards_ddpg, name="ddpg")
    df_err = pd.concat([err1, err2, err3], axis=1)

    ax = df.plot(title="Average total reward as a function of training iterations", kind="bar",
                 yerr=df_err, capsize=2)

    ax.set_xlabel("Thousands of iterations")
    ax.set_ylabel("Average total reward")
    plt.errorbar(average_remaining_pallets_qlearning.keys(),
                 [baselineclosest_average_reward for _ in average_remaining_pallets_qlearning.keys()],
                 yerr=baselineclosest_std_reward,
                 capsize=4, ecolor="lime", color="lime")
    plt.errorbar(average_remaining_pallets_qlearning.keys(),
                 [baselinerandom_average_reward for _ in average_remaining_pallets_qlearning.keys()],
                 yerr=baselinerandom_std_reward,
                 capsize=4, ecolor="fuchsia", color="fuchsia")

    steps = [int(int(k) / 1000) for k in average_remaining_pallets_qlearning.keys()]
    ax.set_xticklabels(steps)
    plt.legend(["Q-learning", "A2C", "DDPG", "Greedy-baseline", "Random-baseline"], bbox_to_anchor=(1.00001, 1), loc="upper left")

    plt.show()
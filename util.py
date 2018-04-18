import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def log_run(agent, file_name):
    labels = ['time',
              'x', 'y', 'z',
              'phi', 'theta', 'psi',
              'x_velocity', 'y_velocity', 'z_velocity',
              'x_accel', 'y_accel', 'z_accel',
              'phi_velocity', 'theta_velocity', 'psi_velocity', 'rotor_speed1',
              'rotor_speed2', 'rotor_speed3', 'rotor_speed4',
              'reward']
    results = {x: [] for x in labels}

    # Run the simulation, and save the results.
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        state = agent.reset_episode()
        while True:
            rotor_speeds = agent.act(state)
            state, reward, done = agent.task.step(rotor_speeds)

            to_write = [agent.task.sim.time]
            to_write += list(agent.task.sim.pose)
            to_write += list(agent.task.sim.v)
            to_write += list(agent.task.sim.linear_accel)
            to_write += list(agent.task.sim.angular_v)
            to_write += list(agent.task.sim.rotor_speeds)
            to_write += [reward]

            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break
    return results


def load_log(file_path):
    return pd.read_csv(file_path)


def plot_log(file_path):
    results = load_log(file_path)
    plot_run(results)


def normalize_angle(angles):
    # Adjust angles to range -pi to pi
    norm_angles = np.copy(angles)
    for i in range(len(norm_angles)):
        while norm_angles[i] > np.pi:
            norm_angles[i] -= 2 * np.pi
    return norm_angles


def plot_run(results, standalone=True):
    if standalone:
        plt.subplots(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.title('Position')
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Position')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 2)
    plt.title('Velocity')
    plt.plot(results['time'], results['x_velocity'], label='x')
    plt.plot(results['time'], results['y_velocity'], label='y')
    plt.plot(results['time'], results['z_velocity'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Velocity')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 3)
    plt.title('Orientation')
    plt.plot(results['time'], normalize_angle(results['phi']), label='phi')
    plt.plot(results['time'], normalize_angle(results['theta']), label='theta')
    plt.plot(results['time'], normalize_angle(results['psi']), label='psi')
    plt.xlabel('time, seconds')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 4)
    plt.title('Angular Velocity')
    plt.plot(results['time'], results['phi_velocity'], label='phi')
    plt.plot(results['time'], results['theta_velocity'], label='theta')
    plt.plot(results['time'], results['psi_velocity'], label='psi')
    plt.xlabel('time, seconds')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 5)
    plt.title('Rotor Speed')
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4')
    plt.xlabel('time, seconds')
    plt.ylabel('Rotor Speed, revolutions / second')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 6)
    plt.title('Reward')
    plt.plot(results['time'], results['reward'], label='Reward')
    plt.xlabel('time, seconds')
    plt.ylabel('Reward')
    if standalone:
        plt.legend(loc=3)
    ax2 = plt.twinx()
    ax2.plot(results['time'], np.cumsum(results['reward']), color='xkcd:red', label='Accum. Reward')
    ax2.set_ylabel('Accumulated Reward')
    if standalone:
        ax2.legend(loc=4)
    plt.grid(True)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_data(ax, time, y_data, data_labels):
    for data, label in zip(y_data, data_labels):
        ax.plot(time, data, label=label)
        if label is not None:
            ax.legend(loc=0)


def subplot_constructor(rows, cols, titles, x_label, y_labels, subplot_size=(5,3)):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*subplot_size[0], rows*subplot_size[1]))

    for ax, title, y_label in zip(axes.ravel(), titles, y_labels):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)
    plt.tight_layout()
    return fig, axes


def grade(agent, file_name, trials=10):
    titles = ['Position', 'Velocity', 'Acceleration', 'Rotor Speed', 'Reward', 'Accumulated Reward']

    data_labels = [[None],
                   [None],
                   [None],
                   [None],
                   [None],
                   [None]]

    axis_labels = ['Distance, m',
                   'Velocity, m/s',
                   'Acceleration, m/s^2',
                   'Rotor Speed, RPM',
                   'Reward/Step',
                   'Accumulated Reward']

    fig, axes = subplot_constructor(2, 3, titles, 'Time, seconds', axis_labels, (5, 4))
    rewards = []
    for i in range(trials):
        if trials > 10:
            label = [None]
        else:
            label = ["Run {}".format(i)]
        results = log_run(agent, file_name)
        rewards.append(np.sum(results['reward']))
        x_data = results['time']
        y_data = [[results['z']],
                  [results['z_velocity']],
                  [results['z_accel']],
                  [results['rotor_speed1']],
                  [results['reward']],
                  [np.cumsum(results['reward'])]]
        for ax, y in zip(axes.ravel(), y_data):
            plot_data(ax, x_data, y, label)
    plt.show()

    avg_reward = np.mean(rewards)
    max_reward = len(results['time'])
    grade = 100*avg_reward/max_reward
    print("Average accumulated reward over the last {} runs is {:.3f} out of {:.0f} possible or {:.3f}%".format(trials,
                                                                                                                avg_reward,
                                                                                                                max_reward,
                                                                                                                grade))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def log_run(agent, file_name):
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4',
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


def plot_run(results):
    plt.subplots(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.title('Position')
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.title('Velocity')
    plt.plot(results['time'], results['x_velocity'], label='x')
    plt.plot(results['time'], results['y_velocity'], label='y')
    plt.plot(results['time'], results['z_velocity'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Velocity')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.title('Orientation')
    plt.plot(results['time'], normalize_angle(results['phi']), label='phi')
    plt.plot(results['time'], normalize_angle(results['theta']), label='theta')
    plt.plot(results['time'], normalize_angle(results['psi']), label='psi')
    plt.xlabel('time, seconds')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.title('Angular Velocity')
    plt.plot(results['time'], results['phi_velocity'], label='phi')
    plt.plot(results['time'], results['theta_velocity'], label='theta')
    plt.plot(results['time'], results['psi_velocity'], label='psi')
    plt.xlabel('time, seconds')
    plt.grid(True)
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
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.title('Reward')
    plt.plot(results['time'], results['reward'], label='Reward')
    plt.xlabel('time, seconds')
    plt.ylabel('Reward')
    plt.legend(loc=3)
    plt.twinx()
    plt.plot(results['time'], np.cumsum(results['reward']), color='xkcd:red', label='Accum. Reward')
    plt.ylabel('Accumulated Reward')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def get_episodes_from_memory(memory):
#     episodes = []
#
#     while
#
# def play_memory(memory):
#     while
#     episode
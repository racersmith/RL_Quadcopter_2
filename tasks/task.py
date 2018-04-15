import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,
                 target_pos,
                 init_pose=None,
                 init_velocities=None,
                 init_angle_velocities=None,
                 runtime=5.,
                 pos_noise=None,
                 ang_noise=None,
                 vel_noise=None,
                 ang_vel_noise=None):

        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        self.target_pos = target_pos
        self.pos_noise = pos_noise
        self.ang_noise = ang_noise
        self.vel_noise = vel_noise
        self.ang_vel_noise = ang_vel_noise


        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        # self.action_repeat = 3

        # self.state_size = self.action_repeat * 6
        self.state_size = len(self.get_state())
        hover = 403.929915
        self.action_low = 0.97 * hover  # Avoid a div0 error in physic sim
        self.action_high = 1.02 * hover
        self.action_size = 4

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # Position Reward
        # reward = 1.0/(np.linalg.norm(self.sim.pose[:3] - self.target_pos) + 1.0)

        # Don't hit the ground
        # reward = min(0.0, np.log(max(0.3679, self.sim.pose[2])))

        # Reward positve velocity
        # reward += self.sim.v[2]/10.0

        # Penalty for non-goal related velocity
        # reward -= min(0.5, 0.1*np.log(np.linalg.norm(self.sim.v[:2])+1.0))

        # Penalty for angular velocity
        # reward -= min(0.5, 0.1*np.log(np.linalg.norm(self.sim.angular_v)+1.0))

        # reward for position
        # reward = []
        # reward = 0.333 / (np.log(np.linalg.norm(self.sim.pose[:3] - self.target_pos) + 1) + 1)
        # reward.append(1.0 * self.reward_func(self.sim.pose[:3] - self.target_pos))
        # reward.append(self.reward_func(self.sim.v))
        # reward.append(0.25*self.reward_func(self.sim.angular_v))
        # reward.append(1.0 * self.reward_func(self.normalize_angles(self.sim.pose[3:])))
        # return np.prod(reward)

        # Positional error
        penalty = np.linalg.norm((self.sim.pose[:3] - self.target_pos))
        # reward = 1/(1+penalty)

        return self.reward_from_huber_loss(penalty, delta=0.15, max_reward=1, min_reward=0)

        # reward = 1 - 0.5*penalty

        # return np.clip(reward, -1, 1)
        # penalty += 0.1 * np.linalg.norm(self.normalize_angles(self.sim.pose[3:]))

        # Heading error
        # penalty += 0.1 * abs(self.normalize_angles(self.sim.pose[-1:]))**2

        # Angular velocity
        # penalty += 0.1 * np.linalg.norm((self.sim.angular_v))**2
        # return 1/(penalty + 1)  # turn the penalty into a reward

        # Penalty for velocity
        # reward += 0.333 / (np.log(np.linalg.norm(self.sim.v) + 1) + 1)

        # Penalty for velocity
        # reward += 0.333 / (np.log(np.linalg.norm(self.sim.angular_v) + 1) + 1)

        # return reward

    def reward_from_huber_loss(self, x, delta, max_reward=1, min_reward=0):
        return np.maximum(max_reward - delta * delta * (np.sqrt(1 + (x / delta) ** 2) - 1), min_reward)

    def reward_func(self, x):
        # return 1.0/(np.linalg.norm(x) + 1.0)
        return 1.0 / (np.log((np.linalg.norm(x) + 1.0)) + 1.0)


    def normalize_angles(self, angles):
        # Normalize angles to +/- 1
        norm_angles = np.copy(angles)
        for i in range(len(norm_angles)):
            while norm_angles[i] > np.pi:
                norm_angles[i] -= 2 * np.pi
        return norm_angles

    def get_state(self):
        pos_error = (self.sim.pose[:3] - self.target_pos)
        # return np.array([pos_error[-1], self.sim.v[-1]])
        orientation = self.normalize_angles(self.sim.pose[3:])
        state_list = list()
        state_list.append(pos_error)
        state_list.append(orientation)
        state_list.append(self.sim.v)
        state_list.append(self.sim.angular_v)
        # state_list.append([self.sim.time])
        return np.concatenate(state_list)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        # update sim

        # Single action for vertical only
        # done = self.sim.next_timestep(rotor_speeds*np.ones(4))

        # Full action space
        done = self.sim.next_timestep(rotor_speeds)

        # Create state values
        next_state = self.get_state()

        # Grab reward
        reward = self.get_reward()

        # Give some clear end of episode signals on altitude
        # if done:
            # # Lost massive altitude
            # if next_state[2] < -8.0:
            #     reward -= 5
            #
            # # Gained substantial altitude
            # if next_state[2] > 8.0:
            #     reward -= 2
            #
            # # Still at target altitude'ish
            # if abs(next_state[2]) < 8.0:
            #     reward += 5.0

            # if np.linalg.norm(self.sim.pose[:3] - self.target_pos) < 5:
            #     reward += 10
            # else:
            #     reward -= 10

            # reward += 10/(np.linalg.norm(self.sim.pose[:3] - self.target_pos))

        # tack on the penalty resulting from ending early
        # if done and self.sim.time < self.sim.runtime:
        #     reward += reward * (self.sim.runtime - self.sim.time)/self.sim.dt

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()

        # Randomize the start pose
        if self.pos_noise is not None or self.ang_noise is not None:
            rand_pose = np.copy(self.sim.init_pose)
            if self.pos_noise is not None and self.pos_noise > 0:
                rand_pose[:3] += np.random.normal(0.0, self.pos_noise, 3)
            if self.ang_noise is not None and self.ang_noise > 0:
                rand_pose[3:] += np.random.normal(0.0, self.ang_noise, 3)

            self.sim.pose = np.copy(rand_pose)

        # Randomize starting velocity
        if self.vel_noise is not None:
            rand_vel = np.copy(self.sim.init_velocities)
            rand_vel += np.random.normal(0.0, self.vel_noise, 3)
            self.sim.v = np.copy(rand_vel)

        # Randomize starting angular velocity
        if self.ang_vel_noise is not None:
            rand_ang_vel = np.copy(self.sim.init_angle_velocities)
            rand_ang_vel += np.random.normal(0.0, self.ang_vel_noise, 3)
            self.sim.angular_v = np.copy(rand_ang_vel)

        return self.get_state()

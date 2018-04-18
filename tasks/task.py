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

        # self.state_size = self.action_repeat * 6
        self.state_size = len(self.get_state())
        hover = 403.929915
        self.action_low = 0.95 * hover  # Avoid a div0 error in physic sim
        self.action_high = 1.1 * hover
        self.action_size = 1

        # Conversion constants for tanh activation
        self.action_m = (self.action_high-self.action_low)/2.0
        self.action_b = (self.action_high+self.action_low)/2.0

    def get_reward(self):
        # Positional error
        # loss = np.linalg.norm((self.sim.pose[:3] - self.target_pos))

        # reward = 1/(1+penalty)
        # return np.clip(1-(self.sim.pose[2]-self.target_pos[2])**2, 0, 1)

        # return self.reward_from_huber_loss(penalty, delta=1, max_reward=1, min_reward=0)
        loss = (self.sim.pose[2]-self.target_pos[2])**2
        # loss += 0.1*self.sim.linear_accel[2]**2
        reward = self.reward_from_huber_loss(loss, delta=0.5)
        return reward

    def reward_from_huber_loss(self, x, delta, max_reward=1, min_reward=0):
        return np.maximum(max_reward - delta * delta * (np.sqrt(1 + (x / delta) ** 2) - 1), min_reward)

    def normalize_angles(self, angles):
        # Normalize angles to +/- 1
        norm_angles = np.copy(angles)
        for i in range(len(norm_angles)):
            while norm_angles[i] > np.pi:
                norm_angles[i] -= 2 * np.pi
        return norm_angles

    def get_state(self):
        pos_error = (self.sim.pose[:3] - self.target_pos)

        # Simple linear Z-axis state
        return np.array([pos_error[2],
                         self.sim.v[2],
                         self.sim.linear_accel[2]
                         ])

        # Full State
        # orientation = self.normalize_angles(self.sim.pose[3:])
        # state_list = list()
        # state_list.append(pos_error)
        # state_list.append(orientation)
        # state_list.append(self.sim.v)
        # state_list.append(self.sim.linear_accel)
        # state_list.append(self.sim.angular_v)
        # state_list.append(self.sim.angular_accels)
        # state_list.append([self.sim.time])
        # return np.concatenate(state_list)

    def convert_action(self, action):
        return action*self.action_m + self.action_b

    def step(self, action):
        """Uses action to obtain next state, reward, done."""

        rotor_speeds = self.convert_action(action)

        # update sim
        # Single action for vertical only
        done = self.sim.next_timestep(rotor_speeds*np.ones(4))

        # Full action space
        # done = self.sim.next_timestep(rotor_speeds)

        # Create state values
        next_state = self.get_state()

        # Grab reward
        reward = self.get_reward()

        if reward <= 0:
            done = True

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()

        if self.action_size == 1:
            # Randomize the start pose
            if self.pos_noise is not None or self.ang_noise is not None:
                rand_pose = np.copy(self.sim.init_pose)
                if self.pos_noise is not None and self.pos_noise > 0:
                    rand_pose[2] += np.random.normal(0.0, self.pos_noise, 1)

                self.sim.pose = np.copy(rand_pose)

            # Randomize starting velocity
            if self.vel_noise is not None:
                rand_vel = np.copy(self.sim.init_velocities)
                rand_vel[2] += np.random.normal(0.0, self.vel_noise, 1)
                self.sim.v = np.copy(rand_vel)
            return self.get_state()

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

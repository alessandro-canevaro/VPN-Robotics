import math
import os
from collections import deque
from random import getrandbits, randrange

import gym
from gym import spaces
from scipy.signal import convolve2d

from src.environment.arena_maps.MapGenerator import MapGenerator

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import numpy as np
import pygame

colors = {
    "red": (199, 0, 11),
    "yellow": (248, 181, 60),
    "light_gray": (221, 221, 221),
    "white": (255, 255, 255),
    "dark_gray": (89, 87, 87),
}


class MyMapGenerator(MapGenerator):
    def __init__(self, **kwargs):
        pass

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 5}

    def __init__(
        self,
        size,
        obstacles_prob,
        max_dist,
        render_mode=None,
        render_fps=4,
        is_train=True,
        map_mode="arena",
        step_reward = -0.01,
        min_dist=1,
    ):
        self.size = size  # The size of the square grid
        self.obstacles_prob = obstacles_prob
        self.max_dist = max_dist
        self.min_dist = min_dist
        if is_train:
            self.curr_dist = self.min_dist
        else:
            self.curr_dist = self.max_dist

        self.render_fps = render_fps
        self.window_size = 800#512  # The size of the PyGame window

        self.map_mode = map_mode
        if self.map_mode == "arena":
            self.mg = MyMapGenerator()

        self.step_reward = step_reward
        self._render_trace = []

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, self.size, self.size), dtype=np.float32
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            6: np.array([1, 0]),
            4: np.array([0, 1]),
            1: np.array([-1, 0]),
            3: np.array([0, -1]),
            7: np.array([1, 1]),
            5: np.array([1, -1]),
            0: np.array([-1, -1]),
            2: np.array([-1, 1]),
        }
 
        assert render_mode is None or render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.episode_counter = 0

    def increase_curriculum(self):
        if self.curr_dist < self.max_dist:
            self.curr_dist += 1
        return self.curr_dist

    def _get_obs(self):
        state = np.zeros((4, self.size, self.size))
        state[0, :, :] = self._map
        a_row, a_col = self._agent
        state[1, a_row, a_col] = 1
        t_row, t_col = self._target
        #state[3,t_row, t_col] = 1
        state[3, t_row-1:t_row+2, t_col-2:t_col+3] = 1
        return state
    
    def _get_distance_matrix(self, starting_point=None):     
        # A data structure for queue used in BFS
        class queueNode:
            def __init__(self, row, col, dist: int):
                self.row = row # The coordinates of the cell
                self.col = col
                self.dist = dist # Cell's distance from the source

        # Check whether given cell(row,col)
        # is a valid cell or not
        def isValid(row: int, col: int):
            return 0 <= row < self.size and 0 <= col < self.size

        visited = np.full_like(self._map, False)
        
        # Mark the source cell as visited
        a_row, a_col = starting_point
        visited[a_row, a_col] = True
        
        # Create a queue for BFS
        q = deque()
        
        # Distance of source cell is 0        
        s = queueNode(a_row, a_col, 0)
        q.append(s) # Enqueue source cell

        dist_mat = -np.ones_like(self._map, dtype=np.float32)
        dist_mat[a_row, a_col] = 0
        
        # Do a BFS starting from source cell
        while q:
            curr = q.popleft() # Dequeue the front cell
            
            r, c = curr.row, curr.col
            
            # Otherwise enqueue its adjacent cell
            for d_r, d_c in self._action_to_direction.values():
                row = r + d_r
                col = c + d_c
                
                # if adjacent cell is valid, has path
                # and not visited yet, enqueue it.
                if (isValid(row, col) and
                self._map[row, col] == 0 and
                    not visited[row, col]):
                    visited[row, col] = True
                    Adjcell = queueNode(row, col,
                                        curr.dist+1)
                    q.append(Adjcell)
                    dist_mat[row, col] = curr.dist+1        
        return dist_mat

    def _generate_map(self):
        if self.map_mode == "random":
            self._walls = np.random.choice([0, 1], size=((self.size-2)**2,), p=[1-self.obstacles_prob, self.obstacles_prob]).reshape((self.size-2, self.size-2))
            self._walls = np.pad(self._walls, pad_width=1, mode='constant', constant_values=1)
            self._obstacles = np.zeros_like(self._walls)
        elif self.map_mode == "arena":
            self._walls = self.mg.create_indoor_map(height=self.size, width=self.size, corridor_radius=2, iterations=self.np_random.integers(10, 100))
            self._obstacles = np.random.choice([0, 1], size=((self.size-2)**2,), p=[1-self.obstacles_prob, self.obstacles_prob]).reshape((self.size-2, self.size-2))
            self._obstacles = np.pad(self._obstacles, pad_width=1, mode='constant', constant_values=1)
            self._obstacles = np.bitwise_and(np.bitwise_not(self._walls), self._obstacles)
        self._map = np.bitwise_or(self._obstacles, self._walls)
            
    def _spawn_agent(self):
        rows, cols = np.where((self._map == 0))
        if rows.size > 0:
            idx = self.np_random.integers(0, rows.size)
            self._agent = np.array([rows[idx], cols[idx]])
        else:
            raise AssertionError("Agent cannot be spawned")
        
    def _spawn_target(self, goal_shape=(1, 1)):
        dist_mat = self._get_distance_matrix(starting_point=self._agent)
        if goal_shape==(1, 1):
            rows, cols = np.where((dist_mat <= self.curr_dist) & (dist_mat >= self.min_dist))
        else:
            occluded_cells = self._map.copy()
            ax, ay = self._agent
            occluded_cells[ax, ay] = 1
            filter1 = np.ones(goal_shape)
            out1 = convolve2d(occluded_cells, filter1, mode="same") == 0
            out1 = np.bitwise_and(out1, (dist_mat <= self.curr_dist) & (dist_mat >= self.min_dist))
            rows, cols = np.where((out1 == 1))
        if rows.size > 0:
            idx = self.np_random.integers(0, rows.size)
            t_row, t_col = rows[idx], cols[idx]
            self._target = np.array([t_row, t_col])
            self.optimal_episode_length = dist_mat[t_row, t_col]
        else:
            raise AssertionError("Target cannot be spawned") 

    def reset(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        while True:
            #Generate walls
            self._generate_map()
            try:
                #Spawn Agent
                self._spawn_agent()
            except AssertionError:
                continue
            try:
                #Spawn Goal
                self._spawn_target()
            except AssertionError:
                continue          
            break
       
        self.episode_counter += 1

        if self.render_mode == "human":
            self._render_frame()
            self._render_trace = []

        return self._get_obs()
        
    def step(self, action):
        direction = self._action_to_direction[action]
        n_row, n_col = self._agent + direction

        if self._map[n_row, n_col] != 1: #wall not hit
            self._agent = np.array([n_row, n_col])  
        
         # An episode is done iff the agent has reached the target
        x_t, y_t = self._target
        terminated = abs(x_t-n_row) <=1 and abs(y_t-n_col) <= 2
        #terminated = np.array_equal(self._agent, self._target)
        reward = 1 if terminated else self.step_reward * np.linalg.norm(direction, ord=2) # Binary sparse rewards
               
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, {"o": self.optimal_episode_length}

    def render(self, mode, **kwargs):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_map(self, canvas, pix_square_size):
        for index, x in np.ndenumerate(self._walls):
            if x == 1:
                pygame.draw.rect(
                canvas,
                colors["dark_gray"],
                pygame.Rect(
                    pix_square_size * np.array(index),
                    (pix_square_size, pix_square_size),
                ),
                )

        for index, x in np.ndenumerate(self._obstacles):
            if x == 1:
                pygame.draw.circle(
                canvas,
                colors["dark_gray"],
                pix_square_size * (np.array(index)+0.5),
                pix_square_size / 3,
                )

    def _draw_target(self, canvas, pix_square_size):
        pygame.draw.rect(
            canvas,
            colors["red"],
            pygame.Rect(
                pix_square_size * self._target,
                (pix_square_size, pix_square_size),
            ),
            width=4,
        )

    def _draw_agent(self, canvas, pix_square_size):
        pygame.draw.circle(
            canvas,
            colors["yellow"],
            (self._agent + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        #render agent's trace
        for pos in self._render_trace:
            pygame.draw.circle(
                canvas,
                colors["yellow"],
                (pos + 0.5) * pix_square_size,
                pix_square_size / 5,
            )
        self._render_trace.append(self._agent)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(colors["light_gray"])
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels


        self._draw_map(canvas, pix_square_size)
        # First we draw the target
        self._draw_target(canvas, pix_square_size)
        # Now we draw the agent
        self._draw_agent(canvas, pix_square_size)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                colors["white"],
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                colors["white"],
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            canvas = pygame.transform.flip(canvas, flip_x=True, flip_y=False)
            canvas = pygame.transform.rotate(canvas, 90)
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.array(pygame.surfarray.pixels3d(canvas))
            #return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
      

class Rod:
    def __init__(self, pos=np.array([0.0, 0.0]), lin_vel=np.array([0.0, 0.0]), angle=0.0, omega=0.0, mass=1.0, length=1.0, inertia=1.0, force=np.array([0.0, 0.0]), torque=0) -> None:
        self.pos = pos
        self.angle = angle
        self.lin_vel = lin_vel
        self.omega = omega
        self.mass = mass
        self.length = length
        self.inertia = inertia #1/12 * self.mass * self.length**2
        self.force = force
        self.torque = torque

    def move(self, f1, f2, dt=1.0):
        force = f1 + f2

        theta = math.radians(self.angle)
        offset = np.array([math.cos(theta), math.sin(theta)]) * (self.length/2)
        r1, r2 = -offset, offset

        tau1 = r1[0]*f1[1] - r1[1]*f1[0]
        tau2 = r2[0]*f2[1] - r2[1]*f2[0]
        torque = tau1 + tau2

        lin_acc = force / self.mass
        lin_vel = lin_acc * dt #+ self.lin_vel 
        pos = self.pos + lin_vel * dt

        ang_acc = torque / self.inertia
        omega = ang_acc * dt #+ self.omega
        angle = self.angle + omega * dt

        return Rod(pos, lin_vel, angle, omega, self.mass, self.length, self.inertia, force, torque)

    def get_tips_positions(self):
        angle = math.radians(self.angle)
        offset = np.array([math.cos(angle), math.sin(angle)]) * (self.length/2)
        return self.pos - offset, self.pos + offset


class CooperativeGridWorldEnv(GridWorldEnv):
    def __init__(self, size, obstacles_prob, max_dist, render_mode=None, render_fps=4, is_train=True, map_mode="random", step_reward=-0.01, min_dist=1):
        super().__init__(size, obstacles_prob, max_dist, render_mode, render_fps, is_train, map_mode, step_reward, min_dist)
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, self.size, self.size), dtype=np.float32
        )
    
    def _get_obs(self):
        state = np.zeros((4, self.size, self.size))
        state[0, :, :] = self._map

        a1x, a1y = self._agent1
        state[1, a1x, a1y] = 1
        a2x, a2y = self._agent2
        state[2, a2x, a2y] = 1
        #c2x, c2y = self._center
        #state[3, c2x, c2y] = 1

        tx, ty = self._target
        state[3, tx, ty] = 1
        #state[4, tx-1:tx+2, ty-2:ty+3] = 1

        return state
     
    def _spawn_agent(self):       
        filter1 = np.ones((1, 4))
        out1 = convolve2d(self._map, filter1, mode="same")
        rows, cols = np.where((out1 == 0))

        if rows.size > 0:
            idx = self.np_random.integers(0, rows.size)
            self._agent1 = np.array([rows[idx], cols[idx]-2])
            self._agent2 = np.array([rows[idx], cols[idx]+1])
            if bool(getrandbits(1)):
                self._agent1[1], self._agent2[1] = self._agent2[1], self._agent1[1]

            xd, yd = self._agent2-self._agent1
            center = (self._agent1+self._agent2)/2
            angle = math.atan2(yd, xd) * 180 / math.pi
            self.rod = Rod(pos=center, angle=angle, mass=2, length=3, inertia=0.08)
            self._center = np.rint(center).astype(int)

            self._render_trace1 = [self._agent1]
            self._render_trace2 = [self._agent2]
        else:
            raise AssertionError("Agent cannot be spawned")
    
    def _spawn_target(self):
        dist_mat = self._get_distance_matrix(starting_point=self._center)
        occluded_cells = self._map.copy()
        
        a1x, a1y = self._agent1
        occluded_cells[a1x, a1y] = 1
        a2x, a2y = self._agent2
        occluded_cells[a2x, a2y] = 1
        
        filter1 = np.ones((3, 5))
        out1 = convolve2d(occluded_cells, filter1, mode="same") == 0
        out1 = np.bitwise_and(out1, (dist_mat <= self.curr_dist) & (dist_mat >= self.min_dist))
        rows, cols = np.where((out1 == 1))
        if rows.size > 0:
            idx = self.np_random.integers(0, rows.size)
            t_row, t_col = rows[idx], cols[idx]
            self._target = np.array([t_row, t_col])
            self.optimal_episode_length = dist_mat[t_row, t_col]
        else:
            raise AssertionError("Target cannot be spawned") 

    def move_obstacles(self, moving_prob=0.3):
        obstructed = self._walls.copy()
        tx, ty = self._target
        obstructed[tx-1: tx+2, ty-2: ty+3] = 1
        a1x, a1y = self._agent1
        obstructed[a1x, a1y] = 1
        a2x, a2y = self._agent2
        obstructed[a2x, a2y] = 1
        cx, cy = self._center
        obstructed[cx, cy] = 1

        rows, cols = np.where(self._obstacles == 1)
        for r, c in zip(rows, cols):
            if self.np_random.random() < moving_prob:
                dx, dy = self.np_random.choice(list(self._action_to_direction.values()))
                nx, ny = r + dx,  c+ dy
                curr_obstructed = np.bitwise_or(obstructed, self._obstacles)
                if curr_obstructed[nx, ny] == 0:
                    self._obstacles[r, c] = 0
                    self._obstacles[nx, ny] = 1

        self._map = np.bitwise_or(self._obstacles, self._walls)

    def step(self, action):
        #old position
        self._old_agent1 = self._agent1
        self._old_agent2 = self._agent2
        
        a1, a2 = action

        d1 = self._action_to_direction[a1]
        d2 = self._action_to_direction[a2]

        new_rod = self.rod.move(d1, d2)
    
        p1, p2 = new_rod.get_tips_positions()
        #discretize:
        x1_int, y1_int = np.rint(p1).astype(int)
        x2_int, y2_int = np.rint(p2).astype(int)

        xc_int, yc_int = np.rint(new_rod.pos).astype(int)

        #check r1 and r2 are inside the maze and not on walls
        cond = 0 <= x1_int < self.size and 0 <= y1_int < self.size
        cond = cond and 0 <= x2_int < self.size and 0 <= y2_int < self.size
        cond = cond and self._map[x1_int, y1_int] != 1 #not on wall
        cond = cond and self._map[x2_int, y2_int] != 1 #not on wall
        #cond = cond and self._map[xc_int, yc_int] != 1 #not on wall
        if cond:
            self.rod=new_rod
            self._agent1 = np.array([x1_int, y1_int])
            self._agent2 = np.array([x2_int, y2_int])
            self._center = np.array([xc_int, yc_int])

        # An episode is done iff the agent has reached the target
        x_t, y_t = self._target
        terminated = abs(x_t-x1_int) <=1 and abs(y_t-y1_int) <= 2 and abs(x_t-x2_int) <=1 and abs(y_t-y2_int) <= 2
        sys_dir = np.linalg.norm(d1, ord=2) #+ np.linalg.norm(self._agent2-self._old_agent2)
        reward = 1 if terminated else self.step_reward * sys_dir # Binary sparse rewards

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, {"o": self.optimal_episode_length, "reward2": 1 if terminated else self.step_reward * np.linalg.norm(d2, ord=2)}
   
    def _draw_agent(self, canvas, pix_square_size):
        p1, p2 = self.rod.get_tips_positions()
        pygame.draw.circle(
            canvas,
            colors["yellow"],
            (p1 + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        pygame.draw.circle(
            canvas,
            colors["red"],
            (p2 + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        pygame.draw.line(
            canvas,
            colors["dark_gray"],
            (p1 + 0.5) * pix_square_size, 
            (p2 + 0.5) * pix_square_size,
            width=16
        )
        
        pygame.draw.rect(
            canvas,
            colors["yellow"],
            pygame.Rect(
                pix_square_size * self._agent1,
                (pix_square_size, pix_square_size),
            ),
            width=4,
        )
        pygame.draw.rect(
            canvas,
            colors["red"],
            pygame.Rect(
                pix_square_size * self._agent2,
                (pix_square_size, pix_square_size),
            ),
            width=4,
        )
        pygame.draw.rect(
            canvas,
            colors["dark_gray"],
            pygame.Rect(
                pix_square_size * self._center,
                (pix_square_size, pix_square_size),
            ),
            width=4,
        )

        #render agent's trace
        self._render_trace1.append(self._agent1)
        for pos_from, pos_to in zip(self._render_trace1, self._render_trace1[1:]):
            pygame.draw.line(
                canvas,
                colors["yellow"],
                (pos_from+0.5)*pix_square_size,
                (pos_to+0.5)*pix_square_size,
                width=3,
            )
        
        self._render_trace2.append(self._agent2)
        for pos_from, pos_to in zip(self._render_trace2, self._render_trace2[1:]):
            pygame.draw.line(
                canvas,
                colors["red"],
                (pos_from+0.5)*pix_square_size,
                (pos_to+0.5)*pix_square_size,
                width=3,
            )
        
    def _draw_target(self, canvas, pix_square_size):
        pygame.draw.rect(
            canvas,
            colors["red"],
            pygame.Rect(
                pix_square_size * (np.array(self._target)-np.array([1, 2])),
                (pix_square_size*3, pix_square_size*5),
            ),
            width=4,
        )
 


class MRS: #multi robot system class
    def __init__(self, agents, pos=np.array([0.0, 0.0]), angle=0.0, mass=1.0, inertia=1.0) -> None:
        self.agents = agents #nx2 agents[i] = r[i], angle[i] (polar coordinates)
        self.pos = pos
        self.angle = angle
        self.mass = mass
        self.inertia = inertia #1/12 * self.mass * self.length**2

    def move(self, forces, dt=1.0):
        force = np.sum(forces, axis=0)
        torque = 0.0
        assert len(forces) == len(self.agents), "Num agents !- num forces"
        for r, f in zip(self.agents, forces):
            theta = math.radians(self.angle+r[1])
            offset = np.array([math.cos(theta), math.sin(theta)]) * r[0]
            torque += offset[0]*f[1] - offset[1]*f[0]
       
        lin_acc = force / self.mass
        lin_vel = lin_acc * dt #+ self.lin_vel 
        pos = self.pos + lin_vel * dt

        ang_acc = torque / self.inertia
        omega = ang_acc * dt #+ self.omega
        angle = self.angle + omega * dt

        return MRS(self.agents, pos, angle, self.mass, self.inertia)

    def get_agents_positions(self):
        agents_pos = np.zeros_like(self.agents, dtype=np.float32) 
        for i, r in enumerate(self.agents):
            theta = math.radians(self.angle+r[1])
            offset = np.array([math.cos(theta), math.sin(theta)]) * r[0]
            agents_pos[i] = self.pos + offset
        return agents_pos
    
    def __str__(self):
        output = f"{len(self.agents)}-agent(s) MRS: \n"
        for i, (x, y) in enumerate(self.get_agents_positions()):
            output += f"agent {i}: x: {x}, y: {y}\n"
        return output


class GridWorldEnvV2(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 5}

    def __init__(self, agents, size, obstacles_prob, move_obstacles_prob, goal_shape, spawn_max_dist, spawn_min_dist, spawn_starting_dist, extra_obs_agents, map_mode, step_reward, render_mode, render_fps) -> None:
        self.agents = agents
        self.mass = len(self.agents)
        self.inertia = 0.08
        self.MRS = MRS(self.agents, mass=self.mass, inertia=self.inertia)
        self.discrete_loc = np.rint(self.MRS.get_agents_positions())

        self.goal_shape=goal_shape
        
        self.size = size  # The size of the square grid
        self.obstacles_prob = obstacles_prob
        self.move_obstacles_prob = move_obstacles_prob

        self.max_dist = spawn_max_dist
        self.min_dist = spawn_min_dist
        self.curr_dist = spawn_starting_dist

        self.render_fps = render_fps
        self.window_size = 800 #512  # The size of the PyGame window

        self.map_mode = map_mode
        if self.map_mode == "arena":
            self.mg = MyMapGenerator()

        self.step_reward = step_reward

        self.extra_obs_agents = extra_obs_agents
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.agents)+2+self.extra_obs_agents, self.size, self.size), dtype=np.float32
        )

        # We have 8 actions, corresponding to "right", "up", "left", "down", ...
        self._action_to_direction = {
            0: np.array([-1, -1]),
            1: np.array([-1, 0]), #up
            2: np.array([-1, 1]),            
            3: np.array([0, -1]), #left
            4: np.array([0, 1]), #right
            5: np.array([1, -1]),
            6: np.array([1, 0]), #down
            7: np.array([1, 1]),           
        }
        self.action_space = spaces.Discrete(len(self._action_to_direction))
 
        assert render_mode is None or render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode
        self._render_trace = []

        self.window = None
        self.clock = None

        self.episode_counter = 0

    def increase_curriculum(self):
        if self.curr_dist < self.max_dist:
            self.curr_dist += 1
        return self.curr_dist

    def _get_distance_matrix(self, visited):     
        # A data structure for queue used in BFS
        class queueNode:
            def __init__(self, row, col, dist: int):
                self.row = row # The coordinates of the cell
                self.col = col
                self.dist = dist # Cell's distance from the source
        
        # Create a queue for BFS
        q = deque()
        
        dist_mat = -np.ones_like(self._map, dtype=np.float32)
        # Distance of source cell is 0  
        for r, c in zip(*np.where(visited==True)):   
            s = queueNode(r, c, 0)
            q.append(s) # Enqueue source cell
            dist_mat[r, c] = 0
        
        # Do a BFS starting from source cell
        while q:
            curr = q.popleft() # Dequeue the front cell
            r, c = curr.row, curr.col
            
            # Otherwise enqueue its adjacent cell
            for d_r, d_c in self._action_to_direction.values():
                nr = r + d_r
                nc = c + d_c
                # if adjacent cell is valid, has path
                # and not visited yet, enqueue it.
                if (self._map[nr, nc] == 0 and
                    not visited[nr, nc]):
                    visited[nr, nc] = True
                    adj_cell = queueNode(nr, nc,
                                        curr.dist+1)
                    q.append(adj_cell)
                    dist_mat[nr, nc] = curr.dist+1        
        return dist_mat

    def get_plot_data(self):
        data = {}
        data["map"] = self._map
        data["walls"] = self._walls
        data["obstacles"] = self._obstacles
        data["agents"] = np.sum(self._agents_mat, axis=0)
        data["target"] = self._target_mat
        data["bfs"] = self.dist_mat
        data["agents_spawn"] = self.agents_spawn_mat == 1
        data["target_spawn"] = self.target_spawn_mat == 0
        return data

    def _generate_map(self):
        if self.map_mode == "random":
            self._walls = np.random.choice([0, 1], size=((self.size-2)**2,), p=[1-self.obstacles_prob, self.obstacles_prob]).reshape((self.size-2, self.size-2))
            self._walls = np.pad(self._walls, pad_width=1, mode='constant', constant_values=1)
            self._obstacles = np.zeros_like(self._walls)
        elif self.map_mode == "arena":
            self._walls = self.mg.create_indoor_map(height=self.size, width=self.size, corridor_radius=2, iterations=self.np_random.integers(10, 100))
            self._obstacles = np.random.choice([0, 1], size=((self.size-2)**2,), p=[1-self.obstacles_prob, self.obstacles_prob]).reshape((self.size-2, self.size-2))
            self._obstacles = np.pad(self._obstacles, pad_width=1, mode='constant', constant_values=1)
            self._obstacles = np.bitwise_and(np.bitwise_not(self._walls), self._obstacles)
        self._map = np.bitwise_or(self._obstacles, self._walls)

    def _spawn_target(self):
        self.target_spawn_mat = self._map.copy()

        if self.goal_shape != (1, 1):
            filter = np.ones(self.goal_shape)
            out = convolve2d(self.target_spawn_mat, filter, mode="same") > 0
            self.target_spawn_mat = np.bitwise_or(self.target_spawn_mat, out)
            
        rows, cols = np.where(self.target_spawn_mat == 0)
        if rows.size > 0:
            idx = self.np_random.integers(0, rows.size)
            t_row, t_col = rows[idx], cols[idx]
            self._target = np.array([t_row, t_col])

            self._target_mat = np.zeros_like(self._map)
            self._target_mat[t_row-self.goal_shape[0]//2:t_row+self.goal_shape[0]//2+1, t_col-self.goal_shape[1]//2:t_col+self.goal_shape[1]//2+1] = 1
        else:
            raise AssertionError("Target cannot be spawned") 

    def _spawn_agent(self): 
        agents_cart = np.empty_like(self.agents)
        for i, (r, a) in enumerate(self.agents):
            agents_cart[i] = r*np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        xmin, xmax = np.min(agents_cart[:, 0]), np.max(agents_cart[:, 0])
        ymin, ymax = np.min(agents_cart[:, 1]), np.max(agents_cart[:, 1])
        shape = (int(xmax-xmin)+1, int(ymax-ymin)+1)
        agents_shape = np.ones(shape)
        x_offset, y_offset = -0.5 if shape[0] % 2 == 0 else 0.0, -0.5 if shape[1] % 2 == 0 else 0.0

        t_row, t_col = self._target

        visited = np.array(self._target_mat, dtype=bool)
        self.dist_mat = self._get_distance_matrix(visited)

        self.agents_spawn_mat = convolve2d(np.bitwise_or(self._map, self._target_mat), agents_shape, mode="same") == 0
        self.agents_spawn_mat = np.bitwise_and(self.agents_spawn_mat, (self.dist_mat <= self.curr_dist) & (self.dist_mat >= self.min_dist))
        
        rows, cols = np.where((self.agents_spawn_mat == 1))

        if rows.size > 0:
            idx = self.np_random.integers(0, rows.size)
            np.random.shuffle(self.agents)
            self.MRS = MRS(self.agents, np.array([rows[idx]+x_offset, cols[idx]+y_offset]), mass=self.mass, inertia=self.inertia)

            self._agents_mat = np.zeros((len(self.agents), self.size, self.size))
            for i, (x, y) in enumerate(np.rint(self.MRS.get_agents_positions()).astype(int)):
                self._agents_mat[i, x, y] = 1

            self.optimal_episode_length = self.dist_mat[rows[idx], cols[idx]]
        else:
            raise AssertionError("Agent cannot be spawned")

    def reset(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        while True:
            #Generate walls
            self._generate_map()
            try:
                #Spawn Goal
                self._spawn_target()
            except AssertionError:
                continue 

            try:
                #Spawn Agent
                self._spawn_agent()
            except AssertionError:
                continue
                     
            break
       
        self.episode_counter += 1

        if self.render_mode == "human":           
            self._render_trace = []
            self._render_frame()


        obs = np.concatenate((np.expand_dims(self._map, axis=0), self._agents_mat, np.zeros((self.extra_obs_agents, self.size, self.size)), np.expand_dims(self._target_mat, axis=0)), axis=0)
        return obs

    def step(self, actions):   
        if type(actions) == int or type(actions) == np.int64:
            actions = [int(actions)]

        directions = np.empty((len(actions), 2))
        for i, a in enumerate(actions):
            directions[i] = self._action_to_direction[a]
        new_mrs = self.MRS.move(directions)

        new_agent_pos_disc = np.rint(new_mrs.get_agents_positions()).astype(int)

        #check r1 and r2 are inside the maze and not on walls
        cond = True
        for x, y in new_agent_pos_disc:
            cond = cond and 0 <= x < self.size and 0 <= y < self.size
            cond = cond and self._map[x, y] != 1 #not on wall

        if cond:
            self.MRS = new_mrs

            self._agents_mat = np.zeros((len(self.agents), self.size, self.size))
            for i, (x, y) in enumerate(new_agent_pos_disc):
                self._agents_mat[i, x, y] = 1

        if self.render_mode == "human":
            self._render_frame()
        if self.move_obstacles_prob > 0.0:
            self._move_obstacles(self.move_obstacles_prob)

        # An episode is done iff the agent has reached the target
        terminated = False if -1 in self._target_mat - np.sum(self._agents_mat, axis=0) else True

        rewards = []
        for d in directions:
            rewards.append(1 if terminated else self.step_reward * np.linalg.norm(d, ord=2)) # Binary sparse rewards

        obs = np.concatenate((np.expand_dims(self._map, axis=0), self._agents_mat, np.zeros((self.extra_obs_agents, self.size, self.size)), np.expand_dims(self._target_mat, axis=0)), axis=0)

        return obs, rewards[0], terminated, {"o": self.optimal_episode_length, "reward2": rewards[1] if len(rewards) > 1 else 0.0}
    
    def _move_obstacles(self, moving_prob):
        obstructed = self._walls.copy()
        obstructed += self._target_mat
        obstructed += np.sum(self._agents_mat, axis=0).astype(int)

        rows, cols = np.where(self._obstacles == 1)
        for r, c in zip(rows, cols):
            if self.np_random.random() < moving_prob:
                dx, dy = self.np_random.choice(list(self._action_to_direction.values()))
                nx, ny = r + dx,  c+ dy
                curr_obstructed = np.bitwise_or(obstructed, self._obstacles)
                if curr_obstructed[nx, ny] == 0:
                    self._obstacles[r, c] = 0
                    self._obstacles[nx, ny] = 1

        self._map = np.bitwise_or(self._obstacles, self._walls)

    def render(self, mode, **kwargs):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_map(self, canvas, pix_square_size):
        for index, x in np.ndenumerate(self._walls):
            if x == 1:
                pygame.draw.rect(
                canvas,
                colors["dark_gray"],
                pygame.Rect(
                    pix_square_size * np.array(index),
                    (pix_square_size, pix_square_size),
                ),
                )

        for index, x in np.ndenumerate(self._obstacles):
            if x == 1:
                pygame.draw.circle(
                canvas,
                colors["dark_gray"],
                pix_square_size * (np.array(index)+0.5),
                pix_square_size / 3,
                )

    def _draw_target(self, canvas, pix_square_size):
        pygame.draw.rect(
            canvas,
            colors["red"],
            pygame.Rect(
                pix_square_size * (np.array(self._target)-(np.array(self.goal_shape)//2)),
                pix_square_size * np.array(self.goal_shape),
            ),
            width=4,
        )

    def _draw_agents(self, canvas, pix_square_size):
        agents = self.MRS.get_agents_positions()
        for pos in agents:
            pygame.draw.circle(
                canvas,
                colors["yellow"],
                (pos + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            pygame.draw.rect(
                canvas,
                colors["yellow"],
                pygame.Rect(
                    pix_square_size * np.rint(pos),
                    (pix_square_size, pix_square_size),
                ),
                width=4,
            )

        old_pos = agents[0]
        for pos in agents[1:]:
            pygame.draw.line(
                canvas,
                colors["dark_gray"],
                (old_pos + 0.5) * pix_square_size, 
                (pos + 0.5) * pix_square_size,
                width=8,
            )
            old_pos = pos
        pygame.draw.line(
                canvas,
                colors["dark_gray"],
                (old_pos + 0.5) * pix_square_size, 
                (agents[0] + 0.5) * pix_square_size,
                width=8,
        )
        
        #render agent's trace
        self._render_trace.append(agents)
        for pos_from, pos_to in zip(self._render_trace[:-1], self._render_trace[1:]):
            for i, j in zip(pos_from, pos_to):
                pygame.draw.line(
                    canvas,
                    colors["yellow"],
                    (i+0.5)*pix_square_size,
                    (j+0.5)*pix_square_size,
                    width=3,
                )
      
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(colors["light_gray"])
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels


        self._draw_map(canvas, pix_square_size)
        # First we draw the target
        self._draw_target(canvas, pix_square_size)
        # Now we draw the agent
        self._draw_agents(canvas, pix_square_size)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                colors["white"],
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                colors["white"],
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            canvas = pygame.transform.flip(canvas, flip_x=True, flip_y=False)
            canvas = pygame.transform.rotate(canvas, 90)
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.array(pygame.surfarray.pixels3d(canvas))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
      
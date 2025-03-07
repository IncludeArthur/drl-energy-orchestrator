import numpy as np
import math
import perlin

import gymnasium as gym
from gymnasium import spaces

#from env_utils import TrafficGen
from env_utils import Vnf

#import torch
#import torch.nn as nn

"""
The environment models a single host running multiple VNFs and handling a varying amount of incoming traffic
"""

class NfvEnv(gym.Env):

    def __init__(self, max_instances=100, max_traffic=8000, idle_power=2000, periodicity = 100, traffic_gen='sin', verbose=False):
        
        self._max_instances = max_instances
        self._max_traffic = max_traffic #used only for generating traffic requests, nfv instrances can handle more
        self._idle_power = idle_power

        print('created NFV scaling environment')
        print('with max traffic: ', max_traffic)
        print('with max instances: ', max_instances)
        
        traffic_variance = 0
        traffic_periodicity = periodicity

        if traffic_gen == 'sin':
            traffic_gen = self.new_traffic_sin
        elif traffic_gen == 'step':
            traffic_gen = self.new_traffic_step
        elif traffic_gen == 'static':
            traffic_gen = self.new_traffic
        elif traffic_gen == 'mix':
            traffic_gen = self.new_traffic_mix
        elif traffic_gen == 'perlin':
            traffic_gen = self.new_traffic_perlin
            self._p = perlin.Perlin(1234)
        elif traffic_gen == 'sinperlin':
            traffic_gen = self.new_traffic_sinperlin
            self._p = perlin.Perlin(1234)
        else:
            raise ValueError('traffic generator not')
            
            
        self.traffic_gen = traffic_gen
        self._traffic_variance = traffic_variance
        self._traffic_periodicity = traffic_periodicity

        #self._total_power=0
        self._traffic=0

        self.running_vnfs = []
        self._instance_counter = 0

        self._step_count = 0

        # The observation space is a dictionary of variables
        # We need to use gym.wrappers.FlattenObservation to use it in learning code
        self.observation_space = spaces.Dict(
            {
                "power": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "traffic": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "active_instances":  spaces.Box(low=0, high=self._max_instances, shape=(1,), dtype=np.float32)
            }
        )
        
        # We have 3 actions, corresponding to "scale up", "maintain", "scale down"
        self.action_space = spaces.Discrete(3)

    # private method that translates the environment’s state into an observation NORMALIZED
    def _get_obs(self):
        
        #TODO get this data from Vnf class
        # what is different vnfs types?
        vnf_idle_power = 500
        vnf_power_coef = 20
        vnf_max_traffic = 200
        
        max_power = vnf_power_coef * vnf_max_traffic * self._max_instances + vnf_power_coef * vnf_idle_power + self._idle_power

        return {"power": self._measure_power()/10_000, "traffic": self._traffic/100, "active_instances": self._active_instances()}
        #return {"power": self._measure_power()/max_power, "traffic": self._traffic/self._max_traffic, "active_instances": self._active_instances()/self._max_instances}

    # private method that translates the environment’s state into an observation
    def _get_obs_raw(self):
        return {"power": self._measure_power(), "traffic": self._traffic, "active_instances": self._active_instances()}
        
    def _get_info(self):
        traffic_vnfs = []
        for vnf in self.running_vnfs:
            traffic_vnfs.append(vnf.traffic)
        return {'step_count': self._step_count,'instance_counter' : self._instance_counter,'traffic_vnfs': traffic_vnfs}

    def _measure_power(self):
        power = self._idle_power 
        for vnf in self.running_vnfs:
            power += vnf._idle_power + vnf._power_coef * self._traffic
        return power

    def new_traffic_sin(self):

        #variation = self._max_traffic * (math.sin(self._step_count/100) ** 2)
        new_traffic = self._max_traffic * (math.sin(self._step_count/self._traffic_periodicity) ** 2)
        #noise = self.np_random.integers( 0, self._traffic_variance)             
        return new_traffic

    def new_traffic_step(self):
  
        new_traffic = self._step_count // 200 % 2 * self._max_traffic
        #if self._step_count > 50: new_traffic = 200
        return new_traffic

    def new_traffic(self):
        
        new_traffic = 1000
        #if self._step_count > 50: new_traffic = 200
        return new_traffic

    def new_traffic_perlin(self):
        n = self._p.one(self._step_count)
        new_traffic = self._max_traffic/2 + n*(self._max_traffic/100)
        return new_traffic

    def new_traffic_sinperlin(self):
        pert = self._p.one(self._step_count)*(self._max_traffic/200)
        sint = self._max_traffic * (math.sin(self._step_count/self._traffic_periodicity) ** 2) / 2
        new_traffic = pert + sint + self._max_traffic/4
        return new_traffic

    def new_traffic_mix(self):
        x = self._step_count/self._traffic_periodicity
        new_traffic = (math.sin (2 * x) + math.sin(math.pi * x)) * (self._max_traffic / 4) + (self._max_traffic / 2)
        return new_traffic

    def _active_instances(self):
        #return len(self.running_vnfs)
        return self._instance_counter
        
    def _add_instance(self):
        if self._active_instances() >= self._max_instances:
            return 1
        self.running_vnfs.append(Vnf())
        self._instance_counter += 1
        return 0

    def _remove_instance(self):
        if self._active_instances() == 0:
            return 1
        vnf = self.running_vnfs.pop()
        del vnf
        self._instance_counter -= 1
        return 0

    # return the maximum bandwitdh that the env can handle with the current number of vnf instances
    def total_bandwidth(self):
        total =0
        for vnf in self.running_vnfs:
            total += vnf.max_traffic
        return total
        
    def request_traffic(self, traffic):

        # check if current instances can handle the requested traffic
        # if not return extra traffic value -> violation of SLA
        if traffic > self.total_bandwidth():
            for vnf in self.running_vnfs:
                vnf.set_traffic(vnf.max_traffic)
            self._traffic = self.total_bandwidth() 
            return traffic - self.total_bandwidth()

        # For now the traffic is equally split among all vnf instances
        # in the future the load balancing can be controlled by the RL algorithm itself
        
        if self._active_instances()>0:
            split_traffic = traffic/self._active_instances()

        for vnf in self.running_vnfs:
            vnf.set_traffic(split_traffic)
                
        self._traffic = traffic

        return 0

    def optimal_instances(self, traffic):
        return math.ceil(traffic/200) #200 is Vnf max traffic !dirty

    def action_masks(self):
        mask = np.ones(3, dtype=bool)
        if self._instance_counter == 0:
            mask[0] = False
        if self._instance_counter == self._max_instances:
            mask[2] == False
        return mask

    def step(self, action):

        terminated = False
        truncated = False
        reward = 0
        
        # first react to agent action 
        if action == 2: 
            allocation_error = self._add_instance()
            if allocation_error: 
                reward = -20
            else: reward = -0.01

        elif action == 1: 
            reward = 0 
                
        elif action == 0: 
            allocation_error = self._remove_instance()
            if allocation_error: 
                reward = -20
            else: reward = 0
        
        # then update the environment
        traffic_req = self.traffic_gen()
        #print('requesting ', traffic_req)
        sla_error = self.request_traffic(traffic_req)

        # terminates when SLA is violated
        if sla_error > 0: 
            #terminated = True
            reward = reward - (10*sla_error/(traffic_req + 10)) #10 is "safety param"

        # calculate the reward based on the total power consumption
        self._power = self._measure_power()
        reward = reward -self._power/(traffic_req*1_000 + 10) #TODO find function that maps power consumption reward to positive numbers

        info = self._get_info()
        info['req_traffic']=traffic_req
        info['optimal_instances']= self.optimal_instances(traffic=traffic_req)

        self._step_count +=1
        
        return self._get_obs_raw(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # initialize random number generator that is used by the environment to a deterministic state
        self._seed = seed
        super().reset(seed=seed, options=options)
        
        #remove all instantiated vnf if any
        del self.running_vnfs[:]
        self._instance_counter = 0
        
        self._traffic = 0
        self._step_count = 0
            
        observation = self._get_obs_raw()
        
        # np.array([self.agent_pos]).astype(np.float32), {}
        return observation, self._get_info()

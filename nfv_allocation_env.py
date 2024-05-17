import numpy as np
import math
import json
import heapq

import gymnasium as gym
from gymnasium import spaces

from env_utils import CloudHost, EdgeHost, PDUSession

class NfvAllocEnv(gym.Env):

    ''' #### ENVIRONMENT FUNCTIONS #### '''

    def __init__(self, config_file, dt_scale, duration_mean, duration_scale, qi_dict):

        self.time = 0
        self.qos_breach = 0
        self.latency_error = 0
        
        self.node_features = 4 # norm Traffic, norm max Traffic, norm Power, node latency
        self.session_features = 2 #qi, throughput. duration?

        cloud_latency = 200
        power_coef = 10

        self.autogen_sessions = True

        f = open(config_file)
        host_config = json.load(f)
        
        self.cloud_node = CloudHost(power_coef=host_config['cloud']['power_coef'], latency=host_config['cloud']['latency'], cpu_coef=None)

        edge_hosts_config = host_config['edge']
        self.n_edges = len(edge_hosts_config)
        self.edge_nodes = []
        for node in edge_hosts_config:
            self.edge_nodes.append(EdgeHost(power_model=node['power_model'], latency=node['latency'], max_traffic=node['max_traffic'], cpu_model=None))

        # RNG paramters
        self.dt_scale = dt_scale
        self.duration_mean = duration_mean 
        self.duration_scale = duration_scale
        self.qi_dict = qi_dict

        f = open('5qi_table.json')
        self.qi_table = json.load(f)

        # normalization parameters
        #self.traffic_normalization = 10e2
        #self.power_normalization = 10e1
        #self.latency_normalization = 10e2
        self.reward_normalization = 1 #10e2 
        self.reward_latency_normalization = 0.01 #0.01
        
        self.rng = None
        self.request = None
        self.dt = 0

        # state space of dimension (|E|+1)×m, where E is the set of edge nodes and m is the number of features per node
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_edges + 1, self.node_features + self.session_features + 1), dtype=np.float32)

        # action space = discrete |E|+1 
        self.action_space = spaces.Discrete(self.n_edges + 1) #0 is the cloud node

        print(self.cloud_node)
        print(self.edge_nodes)
        
        return

    def step(self, action):

        #advance time to next request
        self.time += self.dt

        allocation_error = 0
        latency_error = 0
        reward = 0

        # take action
        if action == 0:
            #assign PDU session to cloud
            self.cloud_node.add_session(self.request)
            latency_error = (self.cloud_node.get_latency()-self.request.latency)#/self.cloud_node.get_latency()
        else:
            node = action -1
            allocation_error = self.edge_nodes[node].add_session(self.request)
            latency_error = (self.edge_nodes[node].get_latency()-self.request.latency)#/self.edge_nodes[node].get_latency()

        # calculate reward
        total_power = self.measure_power()
        total_traffic = self.measure_traffic()
        if total_traffic != 0:
            reward = -total_power/(total_traffic*self.reward_normalization)

        if allocation_error >0: 
            #this should never happen with MaskablePPO
            reward -= 20
            self.qos_breach += 1
            #assign PDU session to cloud
            self.cloud_node.add_session(self.request)
            latency_error = self.cloud_node.get_latency()-self.request.latency

        self.latency_error = latency_error

        if latency_error > 0 : 
           reward -= latency_error * self.reward_latency_normalization * (10/self.request.priority)
        
        # generate new session request
        if self.autogen_sessions:
            session, dt = self.get_new_session()
            self.request = session
            self.dt = dt

        #remove expired sessions!
        self.cloud_node.remove_expired_sessions(self.time)
        for edge in self.edge_nodes:
            edge.remove_expired_sessions(self.time)

        # gather return values
        terminated = False
        truncated = False

        observations = self._get_obs()
        info = self._get_info()

        info['latency_error']=max(latency_error,0)
        info['allocation_error']=allocation_error
        info['total_power']=self.measure_power()
        info['total_traffic']=self.measure_traffic()
        info['reward'] = reward
        
        return observations, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        
        # initialize random number generator that is used by the environment to a deterministic state
        if seed is not None: #otherwise seed is reset to None by the timelimit wrapper. No way of passing the seed?
            self._seed = seed
        super().reset(seed=self._seed, options=options)

        self.time = 0
        self.qos_breach = 0
        reward = 0

        self.rng = np.random.default_rng(self._seed)

        self.cloud_node.reset()
        for edge in self.edge_nodes:
            edge.reset()

        # generate new session request
        session, dt = self.get_new_session()
        self.request = session
        self.dt = dt
        
        return self._get_obs(), self._get_info()

    def action_masks(self):
        #action 0 is the cloud host -> always available
        mask = np.ones(self.n_edges + 1, dtype=bool)
        for i, edge in enumerate(self.edge_nodes):
            if edge.is_full(self.request.throughput):
                mask[i+1] = False
        return mask

    ''' #### CUSTOM FUNCTIONS #### '''
    def get_obs(self):
        return self._get_obs()

    def _get_obs(self):
        
        #edge_nodes[norm_traffic, norm_power, node_latency] + request[latency, band]
        
        cloud_observations = np.array([self.cloud_node.get_traffic(), 
                                       0.0, 
                                       self.cloud_node.get_power_consumption(), 
                                       self.cloud_node.get_latency()]).reshape((1,4))
        edge_obs = []
        
        for node in self.edge_nodes:
            traffic = node.get_traffic()
            max_traffic = node.max_traffic
            power = node.get_power_consumption()
            latency = node.get_latency()

            node_obs = [traffic,
                        max_traffic,
                        power,
                        latency]
            edge_obs.append(node_obs)

        edge_observations = np.array(edge_obs)

        request_observations = np.tile(
            np.array([self.request.qos, 
                      self.request.throughput, #add also request.duration
                      self.dt]), 
            (self.n_edges+1,1))

        observations = np.concatenate((cloud_observations, edge_observations), axis=0)
        observations = np.concatenate((observations, request_observations), axis=1)
            
        return observations

    def _get_info(self):

        info = {
                'qos_breach': self.qos_breach,
                'power_per_mbit': self.power_per_mbit()
               }
        
        return info

    def power_per_mbit(self):
        if self.measure_traffic() == 0:
            power_per_mbit = 0    
        else:
            power_per_mbit = self.measure_power() / self.measure_traffic()
        return power_per_mbit

    def measure_power(self):

        total_power = self.cloud_node.get_power_consumption()
        
        for edge in self.edge_nodes:
            total_power += edge.get_power_consumption()

        return total_power

    def measure_traffic(self):

        total_traffic = self.cloud_node.get_traffic()
        
        for edge in self.edge_nodes:
            total_traffic += edge.get_traffic()

        return total_traffic

    def get_new_session(self):

        qi_list = list(self.qi_dict.keys())
        qi_prob = list(self.qi_dict.values())

        qi = self.rng.choice(qi_list, p = qi_prob)
        duration = self.rng.normal(loc=self.duration_mean, scale=self.duration_scale)
        throughput = self.rng.uniform(low=10, high=100)
        latency = self.qi_table[str(qi)]["delay"]
        priority = self.qi_table[str(qi)]["priority"]
        
        session = PDUSession(qi, latency, duration, priority, throughput, self.time)
        dt = self.rng.exponential(self.dt_scale)
        
        return session, dt

    def set_session(self, session, dt):
        self.session = session
        self.dt = dt

    def set_dt_scale(self, scale: float):
        self.dt_scale = scale

    def autogen_session(self, value: bool):
        self.autogen_sessions = value

    def print_env(self):
        print('current time: ', self.time)
        print('env action mask: ', self.action_masks())
        print('-- Cloud host --')
        print('power coefficient: ', self.cloud_node.power_coef)
        print('base latency: ', self.cloud_node.base_latency)
        print('current traffic: ', self.cloud_node.traffic)
        print('current power consumption: ', self.cloud_node.get_power_consumption())
        print('active sessions: ', len(self.cloud_node.active_sessions))
        qis = []
        for session in self.cloud_node.active_sessions:
            qis.append(session[1].qos)
        print(qis)
            
        
        for edge in self.edge_nodes:
            print('-- Edge host --')
            print('power model: ', edge.power_model)
            print('base latency: ', edge.base_latency)
            print('max traffic: ', edge.max_traffic)
            print('current traffic: ', edge.traffic)
            print('current power consumption: ', edge.get_power_consumption())
            print('is on: ', edge.is_on)
            print('turned off times: ', edge.on_off_counter)
            print('active sessions: ', len(edge.active_sessions))
            qis = []
            exp = []
            for session in edge.active_sessions:
                qis.append(session[1].qos)
                exp.append(session[0])
            print('qos indicator: ',qis)
            #print('end times: ',exp)

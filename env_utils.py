import math
import heapq
import json

class Vnf:
    
    def __init__(self, idle_power = 500, power_coef = 20, max_traffic = 200):

        self._idle_power = idle_power
        self._power_coef = power_coef
        self.max_traffic = max_traffic

        # Initialize to zero traffic
        self.traffic = 0

    def set_traffic(self, req_traffic):
        
        if req_traffic > self.max_traffic:
            self.traffic = self.max_traffic
            return (req_traffic - self.max_traffic)
        else:
            self.traffic = req_traffic
            return 0

class EdgeHost:

    def __init__(self, power_model, cpu_model, latency, max_traffic):

        self.power_model = power_model #list of 11 values representing the consumption at [idle, 10%max_traffic, ... ,100%max_traffic]
        self.cpu_model = cpu_model
        self.base_latency = latency
        self.max_traffic = max_traffic
        self.traffic = 0
        self.active_sessions = []
        self.is_on = False
        self.on_off_counter = 0

    def is_full(self, requested_traffic):
        return (self.traffic + requested_traffic)>self.max_traffic

    #dont use, use self.traffic instead
    def get_traffic(self):
        traffic = 0
        for session in self.active_sessions:
            traffic += session[1].throughput
        return traffic

    def get_power_consumption(self):
        
        if self.is_on == False:
            return 0
            
        traffic = self.traffic
        t = traffic / self.max_traffic
        lp = math.floor(t*10)
        hp = math.ceil(t*10)
        power = self.power_model[lp] + (self.power_model[hp]-self.power_model[lp])*(t*10-lp)
        return power

    def get_cpu_consumption(self):
        
        if self.is_on == False:
            return 0
            
        traffic = self.traffic
        t = traffic / self.max_traffic
        lp = math.floor(t*10)
        hp = math.ceil(t*10)
        cpu = self.cpu_model[lp] + (self.cpu_model[hp]-self.cpu_model[lp])*(t*10-lp)
        return cpu

    def get_latency(self):
        penalty = 0
        if self.is_on == False:
            penalty = 40
        return self.base_latency + penalty

    def add_session(self, session):
        
        if (session.throughput + self.traffic) > self.max_traffic:
            return session.throughput + self.traffic - self.max_traffic

        if self.is_on == False:
            self.is_on =True
            self.on_off_counter += 1

        self.traffic += session.throughput
        heapq.heappush(self.active_sessions, (session.end_time, session))
        return 0
        
    def remove_expired_sessions(self, current_time):

        if not self.active_sessions:
            self.is_on = False
            return 1 #list is empty
            
        next_end_time, _ = self.active_sessions[0]
        while next_end_time <= current_time:
            _, session = heapq.heappop(self.active_sessions)
            self.traffic -= session.throughput
            if not self.active_sessions:
                self.is_on = False
                return 0
            next_end_time, _ = self.active_sessions[0]
            
        return 0

    def reset(self):
        #self.cpu = 0
        self.traffic = 0
        self.active_sessions.clear()
        self.is_on = True

class CloudHost:
    
    def __init__(self, power_coef, cpu_coef, latency):

        self.power_coef = power_coef
        self.cpu_coef = cpu_coef
        self.base_latency = latency
        self.traffic = 0
        self.active_sessions = []

    def get_traffic(self):
        traffic = 0
        for session in self.active_sessions:
            traffic += session[1].throughput
        return traffic

    def get_power_consumption(self):
        power = self.traffic * self.power_coef
        return power

    def get_cpu_consumption(self):
        cpu = self.traffic * self.cpu_coef
        return cpu

    def get_latency(self):
        return self.base_latency

    def add_session(self, session):
        self.traffic += session.throughput
        heapq.heappush(self.active_sessions, (session.end_time, session))
        return 0
        
    def remove_expired_sessions(self, current_time):
        
        if not self.active_sessions: return 1 #list is empty
            
        next_end_time, _ = self.active_sessions[0]
        while next_end_time <= current_time:
            _, session = heapq.heappop(self.active_sessions)
            self.traffic -= session.throughput
            if not self.active_sessions: return 0
            next_end_time, _ = self.active_sessions[0]
        return 0    
    
    def reset(self):
        self.traffic = 0
        self.active_sessions.clear()

class PDUSession:

    #latencies = [30,40,300]

    def __init__(self, qi, latency, priority ,duration, throughput, start_time):
        
        self.qos = qi
        self.latency = latency
        self.priority = priority
        self.duration = duration
        self.throughput = throughput
        self.start_time = start_time
        self.end_time = start_time + duration

    def latency(self):
        return self.latency   

class TrafficGen:
    
    def traffic_sin(self, step, maximum, periodicity, shift=0):

        new_traffic = maximum * (math.sin((step + shift)/periodicity) ** 2)     
        return new_traffic

    def traffic_step(self, step, maximum, periodicity, shift=0):
  
        new_traffic = (step + shift) // periodicity % 2 * self.max_traffic
        return new_traffic

    def traffic_static(self, level=1000):
        
        new_traffic = level
        return new_traffic
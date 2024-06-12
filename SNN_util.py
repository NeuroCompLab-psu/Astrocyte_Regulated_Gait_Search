import numpy as np

class LIF:
    def __init__(self, 
                 _size, 
                 resting_potential, 
                 threshold, 
                 membrane_potential_decay_tau, 
                 refractory_period):
        '''
        _size could be an integer or a tuple
        
        '''
        # parameters
        self._size = _size
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.mem_tau = membrane_potential_decay_tau
        self.refractory_period = refractory_period
        
        # variables
        self.v = np.zeros(_size)
        self.refractory = np.zeros(_size)
        self.potential_change_next_step = np.zeros(_size)

    def step(self, dt):
        self.active = self.refractory <= 0
        self.refractory -= dt
        self.refractory[(self.refractory < 0)] = 0
        self.v[self.active] += self.potential_change_next_step[self.active]
        self.v *= np.exp(-dt / self.mem_tau)
        self.potential_change_next_step = np.zeros(self._size)
        self.AP = self.v > self.threshold
        self.v[self.AP] = self.resting_potential
        self.refractory[self.AP] = self.refractory_period
        return self.AP
    
    def reset(self):
        self.v = np.zeros(self._size)
        self.refractory = np.zeros(self._size)
        self.potential_change_next_step = np.zeros(self._size)
    
    def inject_potential(self, potential_change):
        self.potential_change_next_step += potential_change
        
        
class Stochastic_LIF(LIF):
    def __init__(self, 
                 _size, 
                 resting_potential, 
                 threshold, 
                 membrane_potential_decay_tau, 
                 refractory_period, 
                 AP_trigger_range):
        
        super().__init__(_size, 
                       resting_potential, 
                       threshold, 
                       membrane_potential_decay_tau, 
                       refractory_period)
        self.AP_trigger_range = AP_trigger_range
        
    def step(self, dt):
        self.active = self.refractory <= 0
        self.refractory -= dt
        self.refractory[(self.refractory < 0)] = 0
        self.v[self.active] += self.potential_change_next_step[self.active]
        self.v *= np.exp(-dt / self.mem_tau)
        self.potential_change_next_step = np.zeros(self._size)
        if type(self._size) is tuple:
            self.AP = np.random.rand(*self._size) < 1 / (1 + np.exp(-2 / self.AP_trigger_range * (self.v - self.threshold)))
        else:
            self.AP = np.random.rand(self._size) < 1 / (1 + np.exp(-2 / self.AP_trigger_range * (self.v - self.threshold)))
        self.v[self.AP] = self.resting_potential
        self.refractory[self.AP] = self.refractory_period
        return self.AP
    
    def reset(self):
        self.v = np.zeros(self._size)
        self.refractory = np.zeros(self._size)
        self.potential_change_next_step = np.zeros(self._size)
    
    
class Pacemaker_Stochastic_LIF(Stochastic_LIF):
    def __init__(self, 
                 _size, 
                 resting_potential, 
                 threshold, 
                 membrane_potential_decay_tau, 
                 refractory_period, 
                 AP_trigger_range, 
                 Ca_concentration_increment_per_AP, 
                 Ca_tau, 
                 Ca_dependent_K_channel_threshold, 
                 Ca_dependent_K_channel_sensitivity, 
                 K_potential_change_coef):
        
        super().__init__(_size, 
                       resting_potential, 
                       threshold, 
                       membrane_potential_decay_tau, 
                       refractory_period, 
                       AP_trigger_range)
        # parameters
        self.Ca_concentration_increment_per_AP = Ca_concentration_increment_per_AP
        self.Ca_tau = Ca_tau
        self.Ca_dependent_K_channel_threshold = Ca_dependent_K_channel_threshold
        self.Ca_dependent_K_channel_sensitivity = Ca_dependent_K_channel_sensitivity
        self.K_potential_change_coef = K_potential_change_coef
        
        # variables
        self.Ca_concentration = np.zeros(self._size)
        
    def step(self, dt):
        self.active = self.refractory <= 0
        self.refractory -= dt
        self.refractory[(self.refractory < 0)] = 0
        self.v[self.active] += self.potential_change_next_step[self.active]
        self.v *= np.exp(-dt / self.mem_tau)
        self.potential_change_next_step = np.zeros(self._size)
        if type(self._size) is tuple:
            self.AP = np.random.rand(*self._size) < 1 / (1 + np.exp(-2 / self.AP_trigger_range * (self.v - self.threshold)))
        else:
            self.AP = np.random.rand(self._size) < 1 / (1 + np.exp(-2 / self.AP_trigger_range * (self.v - self.threshold)))
        self.v[self.AP] = self.resting_potential
        self.refractory[self.AP] = self.refractory_period
        self.Ca_concentration[self.AP] += self.Ca_concentration_increment_per_AP
        self.Ca_concentration *= np.exp(-dt / self.Ca_tau)
        self.v -= dt * self.K_potential_change_coef * 1 / (1 + np.exp(-self.Ca_dependent_K_channel_sensitivity * (self.Ca_concentration - self.Ca_dependent_K_channel_threshold)))
        return self.AP
    
    def reset(self):
        self.v = np.zeros(self._size)
        self.refractory = np.zeros(self._size)
        self.potential_change_next_step = np.zeros(self._size)
        self.Ca_concentration = np.zeros(self._size)
        
def synapse_generating_rbf(distance, rbf_decay_coef):
    return np.exp(-distance * rbf_decay_coef)

def motor_pool_internal_connection_generator(num_R, strength_at_0_dist, rbf_decay_coef):
    '''
    All the rythmic generating neurons will distribute in the x[0,1] * y[0,1] * z[0,1] cube
    '''
    
    connection = np.zeros((num_R, num_R))
    locations = np.random.rand(num_R, 3)
    for i in range(num_R):
        for j in range(num_R):
            if i == j: continue
            distance_i_j = np.linalg.norm(locations[i] - locations[j])
            connection[i][j] = strength_at_0_dist * synapse_generating_rbf(distance_i_j, rbf_decay_coef)

    return connection






















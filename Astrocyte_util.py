class Li_Rinzel:
    def __init__(self, 
                IP3_star = 0.16,
                r_IP3 = 0.5,
                tau_IP3 = 7,
                a2 = 0.2,
                r_C = 6,
                r_L = 0.11,
                C0 = 2,
                c1 = 0.185,
                v_ER = 0.8,
                k_ER = 0.1,
                d1 = 0.13,
                d2 = 1.049,
                d3 = 0.9434,
                d5 = 0.08234):
        self.IP3_star = IP3_star
        self.r_IP3 = r_IP3
        self.tau_IP3 = tau_IP3
        self.a2 = a2
        self.r_C = r_C
        self.r_L = r_L
        self.C0 = C0
        self.c1 = c1
        self.v_ER = v_ER
        self.k_ER = k_ER
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d5 = d5
        
        self.Ca2 = 0.071006
        self.h = 0.7791
        self.IP3 = 0.16
        self.Q2 = self.d2 * (self.IP3 + self.d1) / (self.IP3 + self.d3)
        self.h_inf = self.Q2 / (self.Q2 + self.Ca2)
        self.m_inf = self.IP3 / (self.IP3 + self.d1)
        self.n_inf = self.Ca2 / (self.Ca2 + self.d5)
        self.tau_h = 1 / (self.a2 * (self.Q2 + self.Ca2))
        self.J_chan = self.r_C * (self.m_inf * self.n_inf * self.h)**3 * (self.C0 - (1+self.c1) * self.Ca2)
        self.J_pump = self.v_ER * self.Ca2**2 / (self.k_ER**2 + self.Ca2**2) 
        self.J_leak = self.r_L * (self.C0 - (1+self.c1) * self.Ca2)
        
        
        
    def step(self, dt, AG):
        self.Ca2 += (self.J_chan + self.J_leak - self.J_pump) * dt
        self.h += (self.h_inf - self.h) / self.tau_h * dt
        self.IP3 += ((self.IP3_star - self.IP3) / self.tau_IP3 + self.r_IP3 * AG) * dt
        
        self.Q2 = self.d2 * (self.IP3 + self.d1) / (self.IP3 + self.d3)
        self.m_inf = self.IP3 / (self.IP3 + self.d1)
        self.n_inf = self.Ca2 / (self.Ca2 + self.d5)
        self.h_inf = self.Q2 / (self.Q2 + self.Ca2)
        self.tau_h = 1 / (self.a2 * (self.Q2 + self.Ca2))
        self.J_chan = self.r_C * (self.m_inf * self.n_inf * self.h)**3 * (self.C0 - (1+self.c1) * self.Ca2)
        self.J_pump = self.v_ER * self.Ca2**2 / (self.k_ER**2 + self.Ca2**2) 
        self.J_leak = self.r_L * (self.C0 - (1+self.c1) * self.Ca2)
        
    def reset(self):
        self.Ca2 = 0.071006
        self.h = 0.7791
        self.IP3 = 0.16
        self.Q2 = self.d2 * (self.IP3 + self.d1) / (self.IP3 + self.d3)
        self.h_inf = self.Q2 / (self.Q2 + self.Ca2)
        self.m_inf = self.IP3 / (self.IP3 + self.d1)
        self.n_inf = self.Ca2 / (self.Ca2 + self.d5)
        self.tau_h = 1 / (self.a2 * (self.Q2 + self.Ca2))
        self.J_chan = self.r_C * (self.m_inf * self.n_inf * self.h)**3 * (self.C0 - (1+self.c1) * self.Ca2)
        self.J_pump = self.v_ER * self.Ca2**2 / (self.k_ER**2 + self.Ca2**2) 
        self.J_leak = self.r_L * (self.C0 - (1+self.c1) * self.Ca2)
            
            
                 
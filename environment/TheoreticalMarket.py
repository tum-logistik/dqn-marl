import numpy as np




class TheoreticalMarket():
    def __init__(self,beta0 = 1, beta1 = -2, beta2 = -3, a = 0.03, ref_p = 1.5):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = -(beta1 + beta2)
        self.a = a
        self.c_n = -beta2 * ref_p
        self.ref_p = ref_p
    
    def demand_func(self, x):
        candidate_demand = self.beta0 + self.beta1*x + self.beta2*x + self.c_n
        # return candidate_demand
        return np.max([0.0, candidate_demand])
    
    def exp1minus_func(self, x):
        return 1 - self.a * x
    
    def win_prob(self, x_n, x_other_arr):
        numerator = np.exp(1 - self.a*x_n) 
        denom_vals = np.vectorize(self.exp1minus_func)(x_other_arr)
        denom = np.sum(np.clip(denom_vals, 0, 1))

        if numerator >= 0 and denom > 0:
            return numerator / denom
        else:
            return 0

    def get_c1(self, x):
        return (self.gamma / self.demand_func(x)) - (1 / x)
        
    
    def get_c2(self, x):
        return self.gamma / (self.demand_func(x) * x )
    
    def compute_opt_dev(self, x_n):
        c1 = self.get_c1(x_n)
        c2 = self.get_c2(x_n)
        square_term = np.max([0, np.power(c1, 2) - c1 + 4*(c2 -1)*c2 - 2*c2])
        opt_dev = np.sqrt(square_term) / 2*c2
        return opt_dev

    def compute_profit(self, x_n, prices):
        return self.win_prob(x_n, prices) * x_n
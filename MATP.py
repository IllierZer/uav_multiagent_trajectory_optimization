import numpy as np
from params import *
from cvxopt import matrix, solvers
from opt_mat import opt_mat
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

class mission :
    def __init__ (self, num_drones, trajectory_duration, initial_positions, final_positions) :

        self.N = num_drones
        self.T = trajectory_duration
        self.h = 0.05 # discretization factor
        self.K = int(self.T / self.h) # + 1 is needed when indices (k) start from 1
        self.om = opt_mat(self.N, self.K, self.h, p_max, p_min, v_max, v_min, a_max, a_min, j_max, j_min, R)
        self.initial_positions = np.array(initial_positions)
        self.final_positions = np.array(final_positions)
        self.positions = []
        self.with_collision = False # for the first iteration of SCP, collision inequality constraints are not added
        self.P = np.eye(3*self.N*self.K)
        self.q = np.tile(np.array([0., 0., 2*g]), self.N*self.K)
        self.epsilon = 1e-2
        self.max_iterations = 1000
        '''
        initial_positions and final_positions are matrices of size (N X 3), with each 
        row corresponding to a drone and each column corresponding to x, y and z axes respectively 
        '''

    def compute_objective(self, accelerations) :
        objective_value = np.dot(accelerations.T, np.matmul(self.P, accelerations)) + np.dot(self.q, accelerations)
        return objective_value[0][0]

    def update_positions(self, accelerations) :
        '''
        accelerations is an array of size 3NK
        each drone has K accelerations corresponding to K time steps
        '''        
        accelerations = accelerations.reshape(-1, 3) # converting accelerations to a matrix of size (NK X 3)
        positions = [] # positions of each drone at a given time step (NK X 3)
        for i in range (self.N) : # loop over drones
            pos_i = [] # positions of drone i at each time step
            acc_i = accelerations[i*self.K : (i+1)*self.K] # accelerations of drone i at each time step
            for k in range (self.K) : # loop over time steps
                val = np.zeros(3)
                if k == 0 :
                    pass 
                else :
                    for j in range (k) :
                        val += (2 * (k + 1) - (2 * (j + 1) + 1)) * acc_i[j]

                p_i_k = initial_positions[i] + (0.5 * self.h**2) * val
                pos_i.append(p_i_k)
            positions.append(pos_i)
        
        self.positions = np.vstack(positions).flatten() 
        # self.positions is an array of size 3NK 
 
    def get_collision_constraint_matrices(self) : # issue
        As = []
        bs = []
        for k in range(1, self.K) : 
            for i in range(self.N - 1) :
                for j in range(i + 1, self.N) :
                    positions = self.positions.reshape(-1, 3)
                    A, b = self.om.collision_ineq_sd(positions[i*self.K + k], self.initial_positions[i], positions[j*self.K + k], self.initial_positions[j], i, j, k)
                    As.append(A)
                    bs.append(b)
        A_in_collision = np.vstack(As) 
        b_in_collision = np.hstack(bs)
        return A_in_collision, b_in_collision

    def equality_constraint_matrices(self) : # construct A_eq and convert it into type matrix()
        A_blocks = []
        b_blocks = []
        for i in range (self.N) : # loop over drones
            A_eq_fp, b_eq_fp = self.om.f_pos_eq_sd(self.initial_positions[i], self.final_positions[i])
            A_eq_fv, b_eq_fv = self.om.f_vel_eq_sd()
            A_eq_fa, b_eq_fa = self.om.f_acc_eq_sd()
            A_eq_ia, b_eq_ia = self.om.i_acc_eq_sd()
            A_eq_i = np.vstack((A_eq_fp, A_eq_fv, A_eq_fa, A_eq_ia))
            b_eq_i = np.hstack((b_eq_fp, b_eq_fv, b_eq_fa, b_eq_ia))
            A_blocks.append(A_eq_i)
            b_blocks.append(b_eq_i)

        self.A_eq = block_diag(*A_blocks)
        self.b_eq = np.hstack(b_blocks)
                     
    def inequality_constraint_matrices(self) : # construct b_in and A_in and convert it to type matrix()
        A_blocks = []
        b_blocks = []
        for i in range (self.N) : # loop over drones
            A_ineq_pos, b_ineq_pos = self.om.pos_ineq_sd(self.initial_positions[i])
            A_ineq_vel, b_ineq_vel = self.om.vel_ineq_sd()
            A_ineq_acc, b_ineq_acc = self.om.acc_ineq_sd()
            A_ineq_jerk, b_ineq_jerk = self.om.jerk_ineq_sd()
            A_ineq = np.vstack((A_ineq_pos, A_ineq_vel, A_ineq_acc, A_ineq_jerk))
            b_ineq = np.hstack((b_ineq_pos, b_ineq_vel, b_ineq_acc, b_ineq_jerk))

            A_blocks.append(A_ineq)
            b_blocks.append(b_ineq)
        
        A = block_diag(*A_blocks)
        b = np.hstack(b_blocks)

        if self.with_collision :
            A_in_collision, b_in_collision = self.get_collision_constraint_matrices()
            self.A_in = np.vstack((A, A_in_collision))
            self.b_in = np.hstack((b, b_in_collision))
        else :
            self.A_in = A
            self.b_in = b
            self.with_collision = True


    def solve_qp(self) :
        # solve the quadratic program
        P = matrix(self.P, tc = 'd') 
        q = matrix(self.q, tc = 'd')
        G = matrix(self.A_in, tc = 'd')
        h = matrix(self.b_in, tc = 'd')
        A = matrix(self.A_eq, tc = 'd')
        b = matrix(self.b_eq, tc = 'd')
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        return sol['x']
    
    def optimize(self) :
        self.equality_constraint_matrices()
        self.inequality_constraint_matrices()
        accelerations = self.solve_qp()
        self.update_positions(np.array(accelerations))
        return self.compute_objective(np.array(accelerations))

    def plan_trajectories(self) :
        # first iteration gives chi_0
        self.objectives = []
        self.steps = []
        f_prev = self.optimize()
        num_iterations = 1
        self.objectives.append(f_prev)
        self.steps.append(num_iterations)
        while num_iterations <= self.max_iterations :
            num_iterations += 1
            f_curr = self.optimize()
            if (abs(f_curr - f_prev) < self.epsilon) :
                break
            else :
                f_prev = f_curr
            
            self.objectives.append(f_curr)
            self.steps.append(num_iterations)
            
        self.positions = self.positions.reshape(-1, 3)

    def plot(self) :
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')            
        for i in range(self.N) :
            ax.plot(self.initial_positions[i][0], self.initial_positions[i][1], self.initial_positions[i][2], marker = 'v')
            ax.plot(self.final_positions[i][0], self.final_positions[i][1], self.final_positions[i][2], marker = 'o')
            ax.plot([self.initial_positions[i][0], self.final_positions[i][0]], [self.initial_positions[i][1], self.final_positions[i][1]], [self.initial_positions[i][2], self.final_positions[i][2]], linestyle = ':', color = "black")
            ax.plot(self.positions[i*self.K : (i+1)*self.K, 0], self.positions[i*self.K : (i+1)*self.K, 1], self.positions[i*self.K : (i+1)*self.K, 2], label = 'drone ' + str(i+1))
        ax.legend()

        plt.subplots(1)
        plt.plot(self.steps, self.objectives)
        plt.xlabel('iterations')
        plt.ylabel('objective')
        plt.legend()

        plt.show()


if __name__ == '__main__' :
    num_drones = 4
    trajectory_duration = 5
    initial_positions = np.array([[0, 2, 0], 
                                  [0, 0, 0],
                                  [0, 1, 0],
                                  [0, 3, 0]])
    final_positions = np.array([[2, 0, 1], 
                                [2, 3, 0.5],
                                [2, 1, 0.5],
                                [2, 2, 0.5]])
    m = mission(num_drones, trajectory_duration, initial_positions, final_positions)
    m.plan_trajectories()
    m.plot()

    # define specific edge cases (all possible trajectory permutations)
    # vary h and observe behaviour of these edge cases
    # check repeatability of the results


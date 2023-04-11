import numpy as np
from params import *

class opt_mat :
    '''
    Optimization matrices for a single drone
    '''
    def __init__(self, N, K, h, p_max, p_min, v_max, v_min, a_max, a_min, j_max, j_min, R) :
        self.N = N
        self.K = K
        self.h = h
        self.p_max = p_max
        self.p_min = p_min
        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max
        self.a_min = a_min
        self.j_max = j_max
        self.j_min = j_min
        self.R = R
        
    # inequality matrices

    def pos_ineq_sd(self, p_i_initial) : # tested
        A_in_l = np.zeros((3*(self.K-1), 3*self.K)) # less than constraint matrix
        A_in_g = np.zeros((3*(self.K-1), 3*self.K)) # greater than constraint matrix
        b_in_l = np.tile((2 * (self.p_max - p_i_initial) / self.h**2), self.K-1) # less than constraint affine term
        b_in_g = np.tile((2 * (-self.p_min + p_i_initial) / self.h**2), self.K-1) # greater than constraint affine term
        for j in range(self.K - 1) : # loop for (K-1) times
            for l in range(3*(j+1)) :
                m = l % 3
                n = (l // 3) + 1
                A_in_l[3*j + m][l] = 2*(j+2)  - (2*n + 1)
                A_in_g[3*j + m][l] = - 2*(j+2)  + (2*n + 1)
        A_in = np.vstack((A_in_l, A_in_g))
        b_in = np.hstack((b_in_l, b_in_g)) 
        return A_in, b_in

    def vel_ineq_sd(self) : # tested
        A_in_l = np.zeros((3*(self.K-1), 3*self.K)) # less than constraint matrix
        A_in_g = np.zeros((3*(self.K-1), 3*self.K)) # greater than constraint matrix
        b_in_l = np.tile((self.v_max / self.h), self.K-1) # less than constraint affine term
        b_in_g = np.tile((-self.v_min / self.h), self.K-1) # greater than constraint affine term
        for j in range(self.K - 1) : # loop for (K-1) times
            for l in range(3*(j+1)) :
                m = l % 3
                A_in_l[3*j + m][l] = 1
                A_in_g[3*j + m][l] = - 1
        A_in = np.vstack((A_in_l, A_in_g))
        b_in = np.hstack((b_in_l, b_in_g))
        return A_in, b_in
    
    def acc_ineq_sd(self) : # tested
        A_in_l = np.hstack((np.eye(3*(self.K-1)), np.zeros((3*(self.K-1), 3)))) # less than constraint matrix
        A_in_g = np.hstack((-1*np.eye(3*(self.K-1)), np.zeros((3*(self.K-1), 3)))) # greater than constraint matrix
        b_in_l = np.tile(self.a_max , self.K-1) # less than constraint affine term
        b_in_g = np.tile(-self.a_min , self.K-1) # greater than constraint affine term
        A_in = np.vstack((A_in_l, A_in_g))
        b_in = np.hstack((b_in_l, b_in_g))
        return A_in, b_in
    
    def jerk_ineq_sd(self) : # tested
        A_in_l = np.zeros((3*(self.K-1), 3*self.K)) # less than constraint matrix
        A_in_g = np.zeros((3*(self.K-1), 3*self.K)) # greater than constraint matrix
        b_in_l = np.tile(self.j_max * self.h , self.K-1) # less than constraint affine term
        b_in_g = np.tile(-self.j_min * self.h , self.K-1) # greater than constraint affine term
        for j in range(self.K - 1) : # loop for (K-1) times
            for l in range(6) :
                m = l % 3
                n = l // 3
                A_in_l[3*j + m][3*j + l] = (-1)**(n+1)
                A_in_g[3*j + m][3*j + l] = (-1)**n
        A_in = np.vstack((A_in_l, A_in_g))
        b_in = np.hstack((b_in_l, b_in_g))
        return A_in, b_in
    
    # construct collision constraint matrix
    def collision_ineq_sd(self, p_i_k, p_i_initial, p_j_k, p_j_initial, i, j, k) : # tested
        '''
        - make sure i < j since we are counting only once
        - k is constrained to go from 1 to K-1 
        - p_i_k, p_j_k are 3x1 vectors representing the position of the i-th and j-th drone at time k
        '''
        if i >= j :
            print("i should be less than j")
            return
        
        alpha_k = p_i_k - p_j_k
        b_col_ineq_ij = 2 * (-np.linalg.norm(alpha_k) * self.R - np.dot(alpha_k, p_j_initial - p_i_initial)) / self.h**2
        A_j = np.zeros((3, 3*self.K))
        A_i = np.zeros((3, 3*self.K))
        for l in range(3*(self.K-1)) :
            m = l % 3
            n = (l // 3) + 1
            val = 2*(k + 1)  - (2*n + 1)
            if val < 0 :
                val = 0
            A_j[m][l] = val
            A_i[m][l] =  -val

        A_ij = np.hstack((np.zeros((3, 3*i*self.K)), A_i, np.zeros((3, 3*(j-i-1)*self.K)), A_j, np.zeros((3, 3*(self.N-j-1)*self.K))))
        A_col_ineq_ij = np.dot(alpha_k, A_ij)
        return A_col_ineq_ij, b_col_ineq_ij
    
    # equality matrices

    def f_pos_eq_sd(self, p_i_initial, p_i_final) : # tested
        A_eq = np.zeros((3, 3*self.K))
        b_eq = 2 * (p_i_final - p_i_initial) / self.h**2
        for l in range(3*(self.K-1)) :
            m = l % 3
            n = (l // 3) + 1
            A_eq[m][l] = 2*self.K  - (2*n + 1)
        return A_eq, b_eq
    
    def f_vel_eq_sd(self) : # tested
        A_eq = np.zeros((3, 3*self.K))
        b_eq = np.zeros(3)
        for l in range(3*(self.K-1)) :
            m = l % 3
            A_eq[m][l] = 1
        return A_eq, b_eq
    
    def f_acc_eq_sd(self) : # tested
        A_eq = np.hstack((np.zeros((3, 3*(self.K-1))), np.eye(3)))
        b_eq = np.zeros(3)
        return A_eq, b_eq
    
    def i_acc_eq_sd(self) : # tested
        A_eq = np.hstack((np.eye(3), np.zeros((3, 3*(self.K-1)))))
        b_eq = np.zeros(3)  
        return A_eq, b_eq


if __name__ == '__main__' :
    N = 4
    K = 5
    h = 0.2
    p_initial = np.array([0, 0, 0])
    p_final = np.array([1, 1, 1])
    om = opt_mat(N, K, h, p_max, p_min, v_max, v_min, a_max, a_min, j_max, j_min, R)
    A, b = om.collision_ineq_sd(np.array([2, 1, 4]), np.array([5, 0, 0]), np.array([4, 2, 1]), np.array([1, 0, 0]), 1, 3, 4)
    print (np.shape(A))
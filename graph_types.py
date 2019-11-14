# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:30:13 2018

@author: Max
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix , identity
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.sparse.linalg import expm_multiply as exp_mult
import operator as op
from functools import reduce
import numpy.random as rnd
import pandas as pd
import sys


'''
Before the different graph types are defined as classes, I define a few
different functions that I use both within the functions of the classes and 
in associated scripts where these classes are implemented.
'''
#%% COMBINATION FUNCTION, USED IN HYPERCUBE.
def ncr(n, r):    
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom
    
#%% Calculate the Hamming weight of a number (will round down if a float is
                                             # entered).

def hamming_weight(number):
    n = int(number)
    nb = bin(n)
    weight = 0
    for i in range(2,len(nb)):
        if nb[i] == '1':
            weight += 1
    return weight

#%% Calculate ideal value of gamma for the Grover search problem as defined in 
# Goldstone and Childs. Note that the value in the paper is HALF that
# calculated here due to differences in Hamiltonian definitions.

def calc_gamma(dimension):
    sigma = 0
    n = dimension
    for i in range(1,n+1):
        sigma += ncr(n,i)/i*1.
    return 2.**(-(n+1))*sigma
#%% Create the n n dimensional Pauli gate operators in the basis state of the 
# n dimensional hypercube. Stores gates in sparse matrix format.
def pauli_gates(dimension):
    gates = []
    gates_row_col = []
    for i in range(int(2**(dimension))):
        gates_row_col.append(i)
        
    for i in range(dimension):
        data = []
        entry = 1
        control = 2**(i)
        data.append(entry)
        for j in range(1,2**(dimension)):
            if j%control == 0:
                entry = -entry
            data.append(entry)
        gates.append(csr_matrix((data,(gates_row_col,gates_row_col))))
        
    return gates
    
#%% Spin-Glass Hamiltonian creation function.

def spin_glass(dimension,mean,sigma,coeffs='random'):
    n = dimension
    tot = 2**n
    mu = mean
    s = sigma
    gates = pauli_gates(n)
    if coeffs == 'random':
        J = rnd.normal(mu,s,size = [n,n])
        h = rnd.normal(mu,s,size = [n])
    if type(coeffs) == list:
        J = coeffs[0]
        h = coeffs[1]
        
    temp = np.zeros(tot)
    row_col  = np.arange(tot)
    H_sk = csr_matrix((temp,(row_col,row_col)),dtype = np.complex128)
    for i in range(n):
        H_sk -= h[i]*gates[i]
        for j in range(n):
            if i != j:
                H_sk -= 0.5*J[i][j]* gates[i].dot(gates[j])
                
 #   H_sk = np.complex128(H_sk)
    return H_sk
#%% Oracle Hamiltonian for the Grover search problem. Hamiltonian returned in a
# sparse matrix format.

def oracle(dimension,entry):
    tot = 2**dimension
    temp = csr_matrix(([-1],([entry],[entry])),[tot,tot])
    return temp

#%%
    
def interpolate_complex(x_y,step):
    re = np.real(x_y)
    im = np.imag(x_y)
    re_spline = interp1d(step,re)
    im_spline = interp1d(step,im)
    true_re = re_spline(0)
    true_im = im_spline(0)
    return true_re, true_im
    
#%%
    
def neville(T_k0):
    T = np.zeros(T_k0.shape[0],T_k0.shape[0])
    T[:,0] = T_k0
    k = T_k0.shape[0]
    n = np.arange(2,2(k+2),2)
    for i in range(1,T.shape[0]):
        for j in range(1,T.shape[1]):
            T[i][j] = T[i][j-1] + (T[i][j-1]-T[i-1][j-1])/((n[i]/n[i-j])**2-1.)
            
    return np.linalg.norm(T[k][k]-T[k][k-1])
    

#%% LINE GRAPH CLASS
class line_graph:
    ''' Generate a line graph for simulation.
    
    Keyword arguments:
        
    length -- define the length of the graph to be simulated, i.e. the number
              of vertices of the line.
              
    quantum -- boolean
    
               Set quantum == True:
                   Variables for the sparse matrix Hamiltonian and wave
                   functions are initialised for later use.
                   
               Set quantum == False:
                   These variables are not initialised.
    
    '''

#==============================================================================
    
    def __init__(self,length=1000,quantum=False):
        '''Define variables for the length of the line and the probabilities of
        finding the walker on each point of the line. If quantum set to true,
        define variables as stated above, with matrix M in csr sparse matrix
        format.'''
        
        self.__l = length
        self.__line = np.zeros(self.__l)
        self.__line[int(self.__l/2)] = 1
        self.__line_points = np.arange(self.__l)
        if quantum == True:
            self.__psi = np.zeros(self.__l,dtype=np.complex_) #Wavefunction
                                                              # variable.
            self.__psi[int(self.__l/2)] = np.complex128(1)
            
            #Set the non zero row and column entries of M.
            self.__sparse_row = [0,0,1]               
            self.__sparse_col = [0,1,0]
            self.__sparse_data = [1,-1,-1]
            for row in range(1,self.__l-1):
                for col in range(1,self.__l-1):
                    if col == row+1 or col == row-1:
                        self.__sparse_row.append(row)
                        self.__sparse_col.append(col)
                        self.__sparse_data.append(-1)
                    if col == row:  
                        self.__sparse_row.append(row)
                        self.__sparse_col.append(col)
                        self.__sparse_data.append(2)
           
            self.__sparse_row.append(self.__l-2)
            self.__sparse_row.append(self.__l-1)
            self.__sparse_row.append(self.__l-1)
            self.__sparse_col.append(self.__l-1)
            self.__sparse_col.append(self.__l-2)
            self.__sparse_col.append(self.__l-1)
            self.__sparse_data.append(-1)
            self.__sparse_data.append(-1)
            self.__sparse_data.append(1)
            
            #Formally define M. Strictly, M is -L where L is the Laplacian.
            self.__M = csr_matrix((self.__sparse_data,
                                  (self.__sparse_row,self.__sparse_col)))
             
#==============================================================================                        
                    

    def next_frame_classical(self):
        '''Increment the classical discrete random walker simulation by one.
        Used in animation, not necessary for general iteration to produce a 
        graph. See below if this is desired.'''
        
        new_line = np.zeros(self.__l)
        for j in range(1,self.__line.shape[0]-1):
            if self.__line[j] != 0:
                new_line[j+1] += self.__line[j]/2.
                new_line[j-1] += self.__line[j]/2.

        self.__line = new_line

#==============================================================================        
        
    def iterate_classical(self,iterations,show_graph):
        '''Iterate the classical random  walk for a fixed number of intervals.
        
        Keyword arguments:
            
            iterations -- the number of iterations the simulation should run
                          for. Note this value must be an INTEGER.
                          
            show_graph -- boolean
            
                          Set show_graph == True:
                              A graph showing the final result of the
                              simulation will be displayed.
                              
                          Set show_graph == False:
                              A graph will not be displayed.
        
        '''
        self.__iter = iterations
        self.__inc = 0
        while self.__inc<self.__iter:
            new_line = np.zeros(self.__l)              #Same process as the 
            print(self.__inc)                          # next_frame function,
            for i in range(1,self.__line.shape[0]-1):  # just for a fixed
                if self.__line[i] != 0:                # number of increments.
                    new_line[i+1] += self.__line[i]/2.
                    new_line[i-1] += self.__line[i]/2.
            self.__line = new_line
            self.__inc+=1
            
        if show_graph == True:
            #Clear any existing plots.
            plt.cla()                                     
            plt.plot(np.arange(self.__l),self.__line,'.')
            plt.xlabel('position')
            plt.ylabel('probability')
            plt.title('Classical, discrete random walk over '+
                      str(iterations)+' iterations')
            plt.show()
            
#==============================================================================
#%%    
    def iterate_quantum_rk(self,iterations,step_size,show_graph=False):
        '''Iterate the explicit Runge-Kutta 4 method for initial value problem 
        ODEs for a fixed number of iterations in defined time step 
        increments to simulate the time evolution of a wavefunction on a line. 
        
        Keyword arguments:
            
            iterations -- The number of iterations the simulation should run
                          for.
                          
            step_size -- The time step that the simulation is advanced by
                         between each iteration.
                         
            show_graph -- boolean
            
                          Set show_graph == True:
                              A graph showing the final result of the
                              simulation will be displayed.
                              
                          Set show_graph == False:
                              A graph will not be displayed.
        '''
        
        half = np.complex128(0.5)           #Define a complex half (1+0j)
        self.__t = np.complex128(step_size) #Init variable for time step.
        for q in range(iterations):
            if q%1000==0:                   #Print command to display progress;
                print(q)                    # can be ommitted if not desired.
                
            #Implement the RK4 method using sucessive k values.
            self.__k_1 = self.__t*self.quant_line_diff(self.__psi)     
            self.__k_2 = self.__t*self.quant_line_diff(self.__psi+\
                                                       half*self.__k_1)
            self.__k_3 = self.__t*self.quant_line_diff(self.__psi+\
                                                       half*self.__k_2)
            self.__k_4 = self.__t*self.quant_line_diff(self.__psi+self.__k_3)
            self.__psi += np.complex128(1/6)*(self.__k_1+np.complex128(2)*\
                                              self.__k_2+np.complex128(2)*\
                                              self.__k_3+self.__k_4)
        self.__line = np.absolute(self.__psi)**2#Take mod sqrd of psi for prob.
        if show_graph == True:  #Show graph if desired.
            plt.cla()
            plt.plot(np.arange(self.__l),self.__line,'-')
            plt.xlabel('position')
            plt.ylabel('probability')
            plt.title('Quantum, continuous random walk over '+str(iterations)+
                      ' iterations')
            plt.show()  
            
#==============================================================================
#%%
    def next_frame_quantum_rk(self,step_size):
        
        '''Increment the continuous quantum random walker simulation by one,
        using the Runge-Kutta 4 method. Used in animation, not necessary for 
        general iteration to produce a graph. See above if this is desired.
        
        Keyword arguments:
    
            step_size -- The time step that the simulation is advanced by
                         between each iteration.
        '''
        
        half = np.complex128(0.5)
        self.__t = np.complex128(step_size)
        self.__k_1 = self.__t*self.quant_line_diff(self.__psi)     
        self.__k_2 = self.__t*self.quant_line_diff(self.__psi+\
                                                   half*self.__k_1)
        self.__k_3 = self.__t*self.quant_line_diff(self.__psi+\
                                                   half*self.__k_2)
        self.__k_4 = self.__t*self.quant_line_diff(self.__psi+self.__k_3)
        self.__psi += np.complex128(1/6)*(self.__k_1+np.complex128(2)*\
                                          self.__k_2+np.complex128(2)*\
                                          self.__k_3+self.__k_4)
        self.__line = np.absolute(self.__psi)**2
                
#==============================================================================
#%%       
    def quant_line_diff(self,psi):
        '''Differentiate the quantum system according to the TDSE.'''
        diff = -1j*self.__M.dot(self.__psi)
        return diff
            
#==============================================================================        
#%%        
    def length(self):
        '''Return the length of line on which the simulation takes place.'''
        
        return self.__l
        
#==============================================================================        
        
    def line(self):
        '''Return the probability of finding a walker at each vertex of the
        line in array form.'''
        
        return self.__line
        
#==============================================================================        
        
    def sparse_row(self):
        '''Return the rows used to define the entries of sparse matrix M.'''
        return self.__sparse_row
        
#==============================================================================        
        
    def sparse_col(self):
        '''Return the columns used to define the entries of sparse matrix M.'''
        return self.__sparse_col
        
#==============================================================================        
        
    def sparse_data(self):
        '''Return the data points used in sparse matrix M.'''
        return self.__sparse_data
        
#==============================================================================        
        
    def psi(self):
        '''Return the complex wavefunction psi in array form.'''
        return self.__psi
                
#%% DEFINITION OF HYPERCUBE AS AN OBJECT.

class hypercube:
    ''' Generate an n dimensional hypercube for simulation.
    
    Keyword arguments:
        
        dimension -- The dimension of the hypercube. 2=square, 3=cube etc.
        
        gamma -- The weighting of the problem Hamiltonian wrt the Laplacian. 
                 Note that there is an equation in
                 this script to calculate the optimal value of gamma for a
                 given dimension of hypercube for the Grover search problem.
                 
        H_problem -- Where the problem Hamiltonian is defined. 
        
                     If set to 'none' then this will not be accounted for and
                     the Hamiltonian of the system will simply be the negative 
                     Laplacian.
                     
                     If desired, the H_problem is entered in a csr sparse
                     matrix format. 
    '''
    
    def __init__(self,dimension,gamma=1,H_problem='none',full_propagator=False):
        '''Initialise the hypercube graph by defining the adjacency matrix,
        Laplacian and full Hamiltonan of the hypercube. '''
        self.__n  = dimension        #Dimension of hypercube, or no of qubits.
        self.__tot_v = 2**self.__n   #Total number of vertices in hypercube.
        self.__cube = np.arange(self.__tot_v,dtype = 'int') #List of vertex 
                                                            # labels.
        self.__adjacent = np.zeros((self.__tot_v,self.__n)) #List of vertices
                                                            # adjacent to given
                                                            # vertex.
        self.__psi = np.zeros(self.__tot_v,dtype = np.complex128) #Vector for
        self.__psi[0] = 1                                         # wavefunc.
        
        #As adjacency matrix is defined as a csr sparse matrix, the data, rows
        # and columns for an entry must be explicitly defined. In this case, 
        # all entries have the value of 1 so only rows and columns are of
        # interest.
        self.__adj_row = []
        self.__adj_col = []
        self.__no_weights = []  #Define list to store hamming weight of each
                                # vertex.
        self.__hamming_max = hamming_weight(self.__tot_v-1) #Define max hamming
                                                            # weight on graph.
                            
        #Define array to store number of entries with a hamming weight equal
        # to the index of the entry. The hamming weight = 0 case is explicitly
        # defined as it will not arise in the assignment operations.
        self.__hamming_line = np.zeros(self.__hamming_max+1,dtype = 'float')
        self.__hamming_line[0] = 1
        self.__H_prob = H_problem
        self.__min_vec = 0
        self.__t = 0
        
        #Sort vertex entries based on their Hamming Weights
        for i in range(self.__hamming_max+1):
            self.__no_weights.append([])
            
        for i in range(self.__cube.shape[0]):
            self.__no_weights[hamming_weight(self.__cube[i])].append(
                              self.__cube[i])
        
        #Find vertices adjacent to a given vertex by performing a logical XOR 
        # with ascending powers of two (one at a time) and storing this
        # as these are the vertices who's binary representations differ by only
        # one bit from the vertex being considered.
        for i in range(self.__n):
            flip = 2**i
            self.__adjacent[:,i]=self.__cube^flip
        for i in range(self.__adjacent.shape[0]):
            for j in range(self.__adjacent.shape[1]):
                self.__adj_row.append(i)
                self.__adj_col.append(int(self.__adjacent[i][j]))
                
        #All data will be 1s, so just create the corresponding number of 1s as
        # there are non 0 entries in the adjacency matrix. Then create a sparse
        # matrix to store these.
        self.__adj_data = np.ones(len(self.__adj_row),dtype=np.complex128)
        
        self.__A = csr_matrix((self.__adj_data,(self.__adj_row,
                                                self.__adj_col)))
        
        #Create a matrix who's entries are the degree of a vertex (on the
        # diagonal). Then define the Laplacian L as A-D.
        diag_row_col = []
        diag_data = []
        for i in range(self.__tot_v):
            diag_row_col.append(i)
            diag_data.append(self.__n)
            
        self.__D = csr_matrix((diag_data,(diag_row_col,diag_row_col)))        
        self.__L = self.__A - self.__D
        
        #Define full Hamiltonian based on problem Hamiltonian and Laplacian.
        if type(H_problem) == str:
            self.__H = -self.__L
            
        else:
            self.__H = -gamma*self.__L + self.__H_prob

#==============================================================================
#%%        
    def init_full_prop(self,step_size):
        '''Define operation to perform a full propagation using exponentiated
        dense full Hamiltonian. Note that this becomes unfeasibly slow for 
        systems >~ 10 qubits.
        
        Keyword Arguments:
            
            step_size -- the desired step size which the wavefunction should be
                         propagated by in time.
        '''
        self.__prop = expm(-1j*step_size*self.H_dense())

#==============================================================================  
#%%
    def init_sg_animation(self,gamma_0,t_0,power):
        '''Put the wavefunction in the ground state of the Laplacian, define
        the ramp end time and define ramping functions.
        
        Keyword Arguments:
            
            gamma_0 -- The long time value of the hopping rate at the end of
                       any ramping.
                       
            t_0 -- The time at which the ramping of Hamiltonians ends.
            
            power -- The polynomial power used for the curve of the ramp.
        '''
        self.ground_state()
        self.__t_0 = t_0
        scale = (1./self.__t_0)
        self.aqc_gamma = lambda t : gamma_0*(1+(scale*t-1)**power)
        self.aqc_coeff = lambda t : (1-(scale*t-1)**power)
 
#%%        
    def A(self):
        '''Return the adjacency matrix of the hypercube in sparse format.'''
        return self.__A
       
#==============================================================================        
#%%        
    def A_dense(self):
        '''Return the adjacency matrix of the hypercube in dense format.'''
        return self.__A.toarray()
       
#==============================================================================        
#%%        
    def L(self):
        '''Return the Laplacian of the hypercube in sparse format.'''
        return self.__L

#==============================================================================
#%%        
    def L_dense(self):
        '''Return the Laplacian of the hypercube in dense format.'''
        return self.__L.toarray()

#==============================================================================
#%%        
    def quant_line_diff(self,psi):
        '''Return the numerical evaluation of the time derivative of psi in 
        vector form.
        
        Keyword Arguments:
            
            psi -- The wavefunction who's time derivative is to be evaluated.
        '''
        diff = -1j*self.__H.dot(self.__psi)
        return diff
        
#==============================================================================
#%%        
    def H_dense(self):
        '''Return the full Hamiltonian of the hypercube in dense format.'''
        return self.__H.toarray()
        
#==============================================================================
#%%
    def psi(self):
        '''Return the complex wavefunction of the hypercube in vector form.'''
        return self.__psi

#==============================================================================              
#%%            
    def hamming_line(self):
        '''Return the array storing the number of vertices with hamming 
        weights corresponding to the index of a given vertex.'''
        return self.__hamming_line
        
#==============================================================================   
#%%        
    def hamming_max(self):
        '''Return the maximum hamming weight of any vertex on the graph.'''
        return self.__hamming_max
        
#============================================================================== 
#%%
    def vertex_probability(self,vertex):
        '''Return the mod squared value of psi at a given vertex. I.e. return
        the probability of measuring the wavefunction at a given vertex.
        
        Keyword Arguments:
            
            vertex- The vertex at which the probability is desired.
        '''
        return np.absolute(self.__psi[vertex])**2

#==============================================================================
#%%
    def prop(self):
        '''Return the time propagator used in simulation.'''
        return self.__prop
        
#==============================================================================
#%%
    def dimension(self):
        '''Return the dimension of the hypercube (number of qubits in 
        register).'''
        return self.__n

#==============================================================================
#%%
    def abs_sqr_psi(self):
        '''Assign the cube variable to be a vector storing the probability of 
        measuring the wavefunction at the vertex corresponding to the index
        of that value.'''
        self.__cube = np.absolute(self.__psi)**2

#==============================================================================
#%%
    def cube(self):
        '''Return the 'cube' variable. Note that this starts simply as an array
        to sequentially store the vertex numbers but later changes to a 
        probability vector.'''
        return self.__cube

#==============================================================================
#%%
    def psi_array(self):
        '''Return an array of calculated psi vectors at a range of times in 
        a simulation.'''
        return self.__psi_array

#==============================================================================
#%%
    def H(self):
        '''Return the full Hamiltonian in sparse format.'''
        return self.__H

#==============================================================================
#%%        
    def rk4(self,step_size):
        '''Perform a Runge-Kutta-4 time propagation.
        
        Keyword Arguments:
            
            step_size -- the time step which should be used to propagate the 
                         wavefunction forward.
        '''
        half = np.complex128(0.5)
        step = np.complex128(step_size)
        self.__k_1 = step*self.quant_line_diff(self.__psi)     
        self.__k_2 = step*self.quant_line_diff(self.__psi+\
                                                   half*self.__k_1)
        self.__k_3 = step*self.quant_line_diff(self.__psi+\
                                                   half*self.__k_2)
        self.__k_4 = step*self.quant_line_diff(self.__psi+self.__k_3)
        self.__psi += np.complex128(1/6)*(self.__k_1+np.complex128(2)*\
                                          self.__k_2+np.complex128(2)*\
                                          self.__k_3+self.__k_4)
        
#==============================================================================        
#%%
    def full_prop(self):  
        '''Perform a matrix multiplication of the time propagator with the 
        wavevector. Note: when using a dense matrix propagator, this becomes
        inefficient for large systems'''
        
        self.__psi = self.__prop.dot(self.__psi)
        
#============================================================================== 
#%%
    def trunc_prop(self,step_size,order):
        '''Perform a truncated time propagation using a Taylor expanded form of
        the time propagator.
        
        Keyword Arguments:
            
            step_size -- The time step to move the simulation forward by.
            
            order -- The order of the Taylor expansion used to create the 
                     truncated propagator.
        '''
        step_size = np.complex128(step_size)
        new_psi = self.__psi
        temp = self.__psi
       # print(step_size)
       # print(self.__H)
        div = 1
        for i in range(1,order+1):
            div = div*i
       #     print(temp)
            temp = self.__H.dot(-1j*step_size*temp)
            
            new_psi += (temp)/div
        self.__psi = new_psi
        
#==============================================================================
#%%
    def exp_prop(self,step_size,iteration_time):
        '''Perform a time propagation with the ability to use large time steps
        ~1 without a loss of accuracy using the SciPy expm_multiply (renamed
        exp_mult here) function.
        
        Keyword Arguments:
            
            step_size -- The time step to move the simulation forward by.
            
            iteration_time -- The total time the simulation should run for.
        '''
        
        iter_no = int(iteration_time/step_size)
        
        self.__psi_array = exp_mult(-1j*self.__H,self.__psi,start=0,
                                    stop=iteration_time,num=iter_no)
         
        self.__psi = self.__psi_array[-1]

        return self.__psi_array

#==============================================================================
#%%        
    def trunc_prop_matrix(self,step_size,order):
        '''Perform a single time propagation using a truncated propagator for a
        defined step size.
        
        Keyword Arguments:
            
            step_size -- The time step to move the simulation forward by.
            
            order -- The order of the Taylor expansion used to create the 
                     truncated propagator.  
        '''
        self.__prop = identity(2**(self.__n))
        temp = -1j*self.__H*step_size
        for i in range(1,order+1):
            self.__prop += (temp**i)/np.math.factorial(i)
        
#==============================================================================
#%%
    def sum_hamming(self):
        '''Assign a variable to sum the probabilities of vertices depending on 
        their hamming weights.'''
        self.__hamming_line = np.zeros(self.__hamming_max+1,dtype = 'float')
        for i in range(len(self.__no_weights)):
            for j in range(len(self.__no_weights[i])):
                self.__hamming_line[i] += self.__cube[self.__no_weights[i][j]]

#==============================================================================
#%%
    def ave_hamming(self):
        '''Assign a variable to average the probabilities of vertices depending
        on their hamming weights.'''
        self.__hamming_line = np.zeros(self.__hamming_max+1,dtype = 'float')
        for i in range(len(self.__no_weights)):
            for j in range(len(self.__no_weights[i])):
                self.__hamming_line[i] += self.__cube[self.__no_weights[i][j]]
        for i in range(len(self.__no_weights)):
            self.__hamming_line[i] = self.__hamming_line[i]/ \
                                     len(self.__no_weights[i])
            
#============================================================================== 
#%%       
    
    def iterate_quantum_rk(self,iterations,step_size,display='average',
                           show_graph=False,iter_type = 'rk'):
        if iter_type == 'full_prop':
            self.__H_dense = self.__H.toarray()
            if np.max(np.abs(self.__H_dense - np.transpose(self.__H_dense))) > 1E-10:
                raise Exception('Not Hermitian')
            
            self.__prop = expm(-1j*step_size*self.__H_dense)
           # print(self.__prop)
            for q in range(iterations):
                if q%1000==0:
                    sys.stdout.write('\r %s '%q + 'of %s iterations' %iterations+ ' norm = %s ' %np.sum(np.absolute(self.__psi)**2))
                self.full_prop()
                
        if iter_type == 'rk':
            for q in range(iterations):
                if q%1000==0:
                    sys.stdout.write('\r %s '%q + 'of %s iterations' %iterations+ ' norm = %s ' %np.sum(np.absolute(self.__psi)**2))
                self.rk4(step_size)

        if type(iter_type) == list:
            self.init_truncated_prop(iter_type[0],iter_type[1])
            for q in range(iterations):
                if q%1000==0:
                    sys.stdout.write('\r %s '%q + 'of %s iterations' %iterations+ ' norm = %s ' %np.sum(np.absolute(self.__psi)**2))
                self.full_prop()        
                
        self.__cube = np.absolute(self.__psi)**2
        print(np.sum(self.__cube))
        if display == 'average':
            self.ave_hamming()
        if display == 'sum':
            self.sum_hamming()
        if show_graph == True:
            plt.figure()
            plt.plot(np.arange(self.__hamming_max+1),self.__hamming_line,'-')
            plt.xlabel('position')
            plt.ylabel('probability')
            plt.title('Quantum, continuous random walk over '+str(iterations)+
            ' iterations on a'+ str(self.__n)+'dimensional hypercube')
            plt.show()
            
#==============================================================================
#%%
    def next_frame_quantum_rk(self,step_size=0.0001,display='average'):
        self.rk4(step_size)
        self.__cube = np.absolute(self.__psi)**2
        if display == 'average':
            self.ave_hamming()
        if display == 'sum':
            self.sum_hamming()
            
#============================================================================== 
#%%
    def next_frame_quantum_full_prop(self,step_size = 0.0001,display = 'average'):
        self.full_prop()
        self.__cube = np.absolute(self.__psi)**2
        if display == 'average':
            self.ave_hamming()
        if display == 'sum':
            self.sum_hamming()

#==============================================================================
#%%
    def next_frame_quantum_trunc_prop(self,step_size = 0.0001,order= 4,display = 'average'):
        self.trunc_prop(step_size,order)
        self.__cube = np.absolute(self.__psi)**2
        if display == 'average':
            self.ave_hamming()
        if display == 'sum':
            self.sum_hamming()

#==============================================================================
#%%
    def next_frame_quantum_sg(self,step_size = 0.0001,order= 4):
        
        if self.__t <= self.__t_0:
            gamma = self.aqc_gamma(self.__t)
            coeff = self.aqc_coeff(self.__t)
            self.change_H(gamma,coeff)
        self.trunc_prop(step_size,order)
        self.__cube = self.abs_sqr_psi()
        self.__t += step_size

#==============================================================================            
#%%
    def iterate_vertex(self,iterations,time_step,vertex,show_graph=False,
                       prop_order = 'none'):
        T = iterations*time_step
        prob_list = np.zeros(iterations)
        for i in range(iterations):
            if i%10000 == 0:
                sys.stdout.write('\r %s '%i + 'of %s iterations ' %iterations+
                                 'vertex prob = %s ' %self.vertex_probability(vertex))
            if prop_order == 'none':
                self.rk4(time_step)
            
            if type(prop_order) == int:
                
                self.trunc_prop(time_step,prop_order)
            
            prob_list[i] = self.vertex_probability(vertex)
        if show_graph == True:
            plt.figure()
            plt.plot(np.arange(0,T,time_step),prob_list)
            plt.xlabel('Time')
            plt.ylabel('Probability')
            plt.show()
        return prob_list
        
#==============================================================================
#%%
    def iterate_vertex_tag(self,iterations,time_step,vertex,show_graph=False,
                       prop_order = 'none'):
        T = iterations*time_step
        prob_list = np.zeros(iterations)
        self.exp_prop(self.__dt,(iterations)*self.__dt)
        self.__vertex_list = self.__psi_array[:,vertex]
        prob_list = np.absolute(self.__vertex_list)**2
        if show_graph == True:
            plt.figure()
            plt.plot(np.arange(0,T,time_step),prob_list)
            plt.xlabel('Time')
            plt.ylabel('Probability')
            plt.show()
        return prob_list
        
    
#%%
    def ground_state(self):
        self.__psi = np.ones(self.__tot_v,dtype = np.complex128)* \
                     np.complex128(1./(np.sqrt(self.__tot_v)))
#==============================================================================                     
#%%        
    def change_H(self,gamma,problem_coefficient):
        self.__H = -gamma*self.__L + problem_coefficient*self.__H_prob
        
#==============================================================================
#%%
    def energy_expectation(self):
        temp = self.__H.dot(self.__psi)
        expect = np.conj(self.__psi).dot(temp)
        return expect

#==============================================================================
#%%
    def energy_floor(self,return_vec=False):
        self.__min_val, self.__min_vec = eigsh(self.__H,k=1,which='SA')
        self.__min_val = self.__min_val[0]
        return self.__min_val, self.__min_vec

#==============================================================================
#%%
    def ground_overlap(self):
        if type(self.__min_vec) == int:
            raise Exception('The ground eigenstate must be found \
                            before an overlap can be calculated.')
            
  #      print(self.__psi,self.__min_vec)
        overlap = np.conj(self.__psi).dot(self.__min_vec)
        overlap = np.absolute(overlap)**2
        return overlap

#==============================================================================
#%%        
    def iterate_aqc(self,run_time,time_step,gamma_0,
                    time_constant,prop_order='none'):
        aqc_gamma = lambda t : gamma_0*(1-np.exp(-time_constant*t))
        self.__t = 0
        self.__dt = time_step
        self.__T = run_time
        iterations = int(self.__T/self.__dt)
        self.__H = -self.__L
        old_gamma = 0
        if type(prop_order) == int:
            for i in range(iterations):
                self.__t = self.__dt*i
                gamma = aqc_gamma(self.__t)
                if gamma - old_gamma > 1E-6:
                    self.change_H(gamma)
                self.trunc_prop(self.__dt,prop_order)
                old_gamma = gamma
                
        if prop_order == 'none':
            for i in range(iterations):
                self.__t += self.__dt*i
                gamma = aqc_gamma(self.__t)
                if gamma - old_gamma > 1E-6:
                    self.change_H(gamma)
                    self.rk4(self.__dt)
                old_gamma = gamma
                
#==============================================================================                
#%%
    def optimal_chi(self,time_step,start_chi,end_chi,
                    interval,gamma_0,prop_order='none'):
        
        aqc_gamma = lambda t,chi : gamma_0*(1+np.exp(-chi*t))
        problem_coeff = lambda t,chi : (1-np.exp(-chi*t))
        iterations = int((end_chi-start_chi)/interval)
        chi_list = []
        expectation_list = []
        for i in range(iterations):
            self.ground_state()
            test_chi = start_chi + i*interval
            chi_list.append(test_chi)
            self.__t = 0
            self.__dt = np.min([(test_chi)**-1/1E5,0.0001])
            self.__H = -self.__L
            gamma = 2*gamma_0
            old_gamma = np.inf
            prob_coeff = 1
            inc = 1
            while ((gamma/gamma_0)-1 > 1E-6):
                self.__t = self.__dt*inc
                old_gamma = gamma
                gamma = aqc_gamma(self.__t,test_chi)
               #
               #print(old_gamma - gamma)
                prob_coeff = problem_coeff(self.__t,test_chi)
                self.change_H(gamma,prob_coeff)
                
                if type(prop_order):
                    self.trunc_prop(self.__dt,prop_order)
                    
                if prop_order == 'none':
                    self.rk4(self.__dt)
                
                inc+=1
            expectation_list.append(self.energy_expectation())
            sys.stdout.write('\r Test Chi = %s.' %test_chi + 
                             'Final Chi = %s' %end_chi+
                 ' norm = %s ' %np.sum(np.absolute(self.__psi)**2)+
                 'final exp = %s' %(gamma/gamma_0-1))
            
        plt.figure()
        plt.plot(chi_list,expectation_list)
        plt.xlabel('Time constant chi')
        plt.ylabel('<H>')
        plt.show()
        
#%%
    def optimal_tau(self,max_time_step,start_tau,end_tau,
                    interval,gamma_0,prop_order='none'):
        
        aqc_gamma = lambda t,tau : gamma_0*(1+np.exp(-t/tau))
        problem_coeff = lambda t,tau : (1-np.exp(-t/tau))
        iterations = int((end_tau-start_tau)/interval)
        tau_list = []
        expectation_list = []
        floor_val_list = []
        floor_val = self.energy_floor(False)[0]
        for i in range(iterations):
            floor_val_list.append(floor_val)
        floor_val_points = np.arange(start_tau,end_tau,interval)
        for i in range(iterations):
            self.ground_state()
            test_tau = start_tau + i*interval
            tau_list.append(test_tau)
            self.__t = 0
            self.__t_0 = 3*test_tau*np.log(10)
            self.__dt = np.min([self.__t_0/1E5,max_time_step])
            
            self.__H = -self.__L
            gamma = 2*gamma_0
            prob_coeff = 1
            inc = 1
            
            while (self.__t <= self.__t_0):
                if inc % 10000 == 0:
                    print(self.__t,self.__t_0)
                self.__t = self.__dt*inc
                gamma = aqc_gamma(self.__t,test_tau)
                prob_coeff = problem_coeff(self.__t,test_tau)
                self.change_H(gamma,prob_coeff)
                
                if type(prop_order):
                    self.trunc_prop(self.__dt,prop_order)
                    
                if prop_order == 'none':
                    self.rk4(self.__dt)
                
                inc+=1
            expectation_list.append(self.energy_expectation())
            sys.stdout.write('\r Test Tau = %s.' %test_tau + 
                             'Final Tau = %s' %end_tau+
                 ' norm = %s ' %np.sum(np.absolute(self.__psi)**2)+
                 ' end time = %s' %(self.__t)+
                 ' t_0 = %s' %(self.__dt))
            
        plt.figure()
        plt.plot(tau_list,expectation_list)
        plt.plot(floor_val_points, floor_val_list, '--')
        plt.xlabel('Time constant tau')
        plt.ylabel('<H>')
        plt.show()
        
        return tau_list,
        
#%%     
    def optimal_polynomial(self,max_time_step,start_power,end_power,
                           scaling_start,scaling_end,scaling_interval,gamma_0,
                           prop_order='none'):
        
        aqc_gamma = lambda t,power,scale : gamma_0*(1+(scale*t-1)**power)
        problem_coeff = lambda t,power,scale : (1-(scale*t-1)**power)
        if start_power%2 != 0 or end_power%2 != 0:
            raise Exception('Both start and end powers must be even integers.')
            
        power_iterations = int(((end_power+2)-start_power)/2)
        scaling_iterations = int((scaling_end+scaling_interval-
                                  scaling_start)/scaling_interval)
        power_list = []
        scaling_list = []
        expectation_list = []
        overlap_list = []
        self.__H = -gamma_0*self.__L + self.__H_prob
        self.energy_floor()
     #   print(self.__min_vec)
#==============================================================================
#         floor_val_list = []
#         floor_val = self.energy_floor(False)[0]
#         for i in range(iterations):
#             floor_val_list.append(floor_val)
#         floor_val_points = np.arange(start_tau,end_tau,interval)
#==============================================================================
        for i in range(power_iterations):
            test_power = start_power + i*2
        #    print(test_power)
            for j in range(scaling_iterations):
                self.ground_state()
                test_scaling = scaling_start + j*scaling_interval
                self.__t = 0
                self.__t_0 = 1./test_scaling
                self.__dt = np.min([self.__t_0/1E5,max_time_step])
                
                self.__H = -self.__L
                gamma = 2*gamma_0
                prob_coeff = 1
                inc = 1
                
                while (self.__t <= self.__t_0):
                    self.__t = self.__dt*inc
                    gamma = aqc_gamma(self.__t,test_power,test_scaling)
                    prob_coeff = problem_coeff(self.__t,test_power,
                                               test_scaling)
                    if inc % 10000 == 0:
                        print(self.__t,self.__t_0)
              #      print(gamma,prob_coeff)
                    self.change_H(gamma,prob_coeff)
                    
                    if type(prop_order):
                        self.trunc_prop(self.__dt,prop_order)
                        
                    if prop_order == 'none':
                        self.rk4(self.__dt)
        
                    inc+=1
                    
                expectation_list.append(self.energy_expectation())
                overlap_list.append(self.ground_overlap())
                scaling_list.append(self.__t_0)
                power_list.append(test_power)
                sys.stdout.write('\r Test Power = %s.' %test_power + 
                                 'Test Scaling = %s' %test_scaling+
                     ' norm = %s ' %np.sum(np.absolute(self.__psi)**2)+
                     ' expectation_energy = %s ' %expectation_list[-1]+
                     ' t_0 = %s ' %self.__t_0 )
        
        X, Y = power_list, scaling_list 
        fig = plt.figure()
        if end_power-start_power > 0 and scaling_end-scaling_start > 0:
            ax = fig.gca(projection='3d')   
            ax.scatter(X, Y, expectation_list)
            ax.set_xlabel('Polynomial Power')
            ax.set_ylabel('Scaling Constant')
            ax.set_zlabel('<H>')
            plt.show()
            
        if scaling_end-scaling_start > 0:
            plt.plot(scaling_list,expectation_list)
            plt.xlabel('Scaling Constant')
            plt.ylabel('<H>')
            plt.title('Polynomial ramp. Power = %s' %X[0])
            plt.show()
            
        if end_power-start_power > 0:
            plt.plot(power_list,expectation_list)
            plt.xlabel('Polynomial Power')
            plt.ylabel('<H>')
            plt.title('Polynomial ramp. Scaling Constant = %s' %Y[0])
            plt.show()
            
        
        return(X,Y,expectation_list,overlap_list)
            
#%%        
    def compare_ramp(self,polynomial_power,t_0,
                     gamma_0,max_time_step,tolerance_order=3,prop_order=4):
        poly_gamma = lambda t,power,scale : gamma_0*(1+(scale*t-1)**power)
        poly_coeff = lambda t,power,scale : (1-(scale*t-1)**power)        
    
        exp_gamma = lambda t,tau : gamma_0*(1+np.exp(-t/tau))
        exp_coeff = lambda t,tau : (1-np.exp(-t/tau))
        
        floor_val = self.energy_floor(False)[0]
        self.__t_0 = t_0
        poly_scale = (1./self.__t_0)*(1-(10**(-tolerance_order))**(1/polynomial_power))
        exp_tau = self.__t_0/(tolerance_order*np.log(10))
        self.__dt = np.min([self.__t_0/1E5,max_time_step])
        iterations = int(self.__t_0/self.__dt)
        self.__t = 0
        self.ground_state()
        self.__H = -self.__L
        for i in range(1,iterations+1):
            
            self.__t = self.__dt*i
            #print(iterations)
            gamma = poly_gamma(self.__t,polynomial_power,poly_scale)
            prob_coeff = poly_coeff(self.__t,polynomial_power,poly_scale)
            self.change_H(gamma,prob_coeff)
            if i % 10000== 0:
                sys.stdout.write('\r %s ' %i+ 
                                 ' %s '%iterations+
                                 ' poly %s'  %t_0)
            if type(prop_order):
                self.trunc_prop(self.__dt,prop_order)
                
            if prop_order == 'none':
                self.rk4(self.__dt)
            
        self.__poly_energy = self.energy_expectation()
        
        self.__t = 0
        self.ground_state()
        self.__H = -self.__L
        for i in range(1,iterations+1):
            if i % 10000 == 0:
                sys.stdout.write('\r %s ' %i+ 
                                 ' %s '%iterations+
                                 ' exp %s'  %t_0)
            self.__t = self.__dt*i
            gamma = exp_gamma(self.__t,exp_tau)
            prob_coeff = exp_coeff(self.__t,exp_tau)
            self.change_H(gamma,prob_coeff)
            
            if type(prop_order):
                self.trunc_prop(self.__dt,prop_order)
                
            if prop_order == 'none':
                self.rk4(self.__dt)
            
        self.__exp_energy = self.energy_expectation()
        
        return self.__poly_energy, self.__exp_energy, floor_val
        
#%% 
    def iterate_vertex_ramp(self, ramp = ['poly',2,1],t_0=1,gamma_0=2,
                            max_time_step = 0.0005,zero_time_step=0.00005,
                            tolerance_order=3,prop_order=4,iteration_time=10,
                            vertex=0):
    
        self.__t_0 = t_0
        if self.__t_0 == 0:
            self.__dt = zero_time_step
        else:
            self.__dt = np.min([self.__t_0/1E4,max_time_step])
        ramp_iterations = int(self.__t_0/self.__dt)
        tot_iterations = int((self.__t_0+iteration_time)/self.__dt)     
        self.__vertex_list = np.zeros(tot_iterations,dtype=np.complex128)
        self.__long_time_prob = np.zeros(tot_iterations-ramp_iterations,dtype=np.complex128)
        
        self.__v = vertex
        prop = prop_order
        if t_0 == 0:
            self.ground_state()
            self.__H = -gamma_0*self.__L + self.__H_prob
            
            self.__vertex_list = self.iterate_vertex(tot_iterations, self.__dt,
                                                     vertex=self.__v,
                                                     show_graph=False,
                                                     prop_order = prop)
            self.__long_time_prob = self.__vertex_list
            time_list = np.arange(0,iteration_time,self.__dt)
            expect_energy = self.energy_expectation()
            floor_val, floor_vec = self.energy_floor()
            energy_diff = expect_energy - floor_val 
            overlap = self.ground_overlap()
        else:
            if ramp[0] == 'poly':
                power = ramp[1]
                scale = (1./self.__t_0)#*(1-(10**(-tolerance_order))**(1/power))
                aqc_gamma = lambda t : gamma_0*(1+(scale*t-1)**power)
                aqc_coeff = lambda t : (1-(scale*t-1)**power)
            
            if ramp[0] == 'exp':
                tau = self.__t_0/(tolerance_order*np.log(10))
                aqc_gamma = lambda t : gamma_0*(1+np.exp(-t/tau))
                aqc_coeff = lambda t : (1-np.exp(-t/tau))
            
            self.ground_state()
            self.__t = 0
            
            
            self.__H = -self.__L
            gamma = 2*gamma_0
            prob_coeff = 1
            inc = 0
            
            for i in range(ramp_iterations):
                if i % 10000 == 0:
                    print(self.__t,self.__t_0)
                self.__t = self.__dt*i
                gamma = aqc_gamma(self.__t)
                prob_coeff = aqc_coeff(self.__t)
                self.change_H(gamma,prob_coeff)
                
                self.trunc_prop(self.__dt,prop_order)
                    
                self.__vertex_list[i] = self.__psi[vertex]
            
            expect_energy = self.energy_expectation()
            floor_val, floor_vec = self.energy_floor()
            energy_diff = expect_energy - floor_val 
            
            #self.trunc_prop_matrix(self.__dt,prop_order)    
                
            for i in range(ramp_iterations,tot_iterations):
                if i % 10000 == 0:
                    print(i,tot_iterations,np.sum(np.absolute(self.__psi)**2))
                
                self.trunc_prop(self.__dt,prop_order)
                self.__vertex_list[i] = self.__psi[vertex]
                self.__long_time_prob[i-ramp_iterations] = self.__psi[vertex]
                
            self.__vertex_list = np.absolute(self.__vertex_list)**2
            self.__long_time_prob = np.absolute(self.__long_time_prob)**2
            time_list = np.zeros(self.__vertex_list.shape[0])
            for i in range(self.__vertex_list.shape[0]):
                time_list[i] = i*self.__dt

            overlap = self.ground_overlap()
        

        return time_list, self.__vertex_list, self.__long_time_prob,\
        energy_diff, overlap
     
#%%
    def iterate_vertex_ramp_tag(self, ramp = ['poly',2,1],t_0=1,gamma_0=2,
                            max_ramp_step = 0.0005,long_time_step=0.00005,
                            tolerance_order=3,prop_order=4,iteration_time=10,
                            vertex=0):
    
        self.__t_0 = t_0
        if self.__t_0 == 0:
            self.__dt = long_time_step
        else:
            self.__dt = np.min([self.__t_0/1E4,max_ramp_step])
        ramp_iterations = int(self.__t_0/self.__dt)
        tot_iterations = int((self.__t_0)/self.__dt+iteration_time/long_time_step)     
        self.__vertex_list = np.zeros(tot_iterations,dtype=np.complex128)
        self.__long_time_prob = np.zeros(tot_iterations-ramp_iterations,dtype=np.complex128)
        
        self.__v = vertex
        prop = prop_order
        if t_0 == 0:
            self.ground_state()
            self.__H = -gamma_0*self.__L + self.__H_prob
            
            self.__vertex_list = self.iterate_vertex_tag(tot_iterations, self.__dt,
                                                     vertex=self.__v,
                                                     show_graph=False,
                                                     prop_order = prop)
            self.__long_time_prob = self.__vertex_list
            time_list = np.arange(0,iteration_time,self.__dt)
            expect_energy = self.energy_expectation()
            floor_val, floor_vec = self.energy_floor()
            energy_diff = expect_energy - floor_val 
            overlap = self.ground_overlap()
        else:
            if ramp[0] == 'poly':
                power = ramp[1]
                scale = (1./self.__t_0)#*(1-(10**(-tolerance_order))**(1/power))
                aqc_gamma = lambda t : gamma_0*(1+(scale*t-1)**power)
                aqc_coeff = lambda t : (1-(scale*t-1)**power)
            
            if ramp[0] == 'exp':
                tau = self.__t_0/(tolerance_order*np.log(10))
                aqc_gamma = lambda t : gamma_0*(1+np.exp(-t/tau))
                aqc_coeff = lambda t : (1-np.exp(-t/tau))
            
            self.ground_state()
            self.__t = 0
            
            
            self.__H = -self.__L
            gamma = 2*gamma_0
            prob_coeff = 1
            inc = 0
            self.__ramp_times = np.arange(0,self.__t_0,self.__dt)
         #   print(self.__ramp_times.shape[0],ramp_iterations)
            self.__gamma_vals = aqc_gamma(self.__ramp_times)
            self.__coeff_vals = aqc_coeff(self.__ramp_times)
            
            for i in range(ramp_iterations):
                if i % 10000 == 0:
                    print(self.__t,self.__t_0)
                self.__t = self.__dt*i
                gamma = self.__gamma_vals[i]
                prob_coeff = self.__coeff_vals[i]
                self.change_H(gamma,prob_coeff)
                
                self.trunc_prop(self.__dt,prop_order)
                    
                self.__vertex_list[i] = self.__psi[vertex]
            
            expect_energy = self.energy_expectation()
            floor_val, floor_vec = self.energy_floor()
            energy_diff = expect_energy - floor_val 
            self.__dt = long_time_step
            self.exp_prop(self.__dt,(tot_iterations-ramp_iterations)*self.__dt)
            self.__vertex_list[ramp_iterations:tot_iterations] = self.__psi_array[:,vertex]
            self.__long_time_prob = self.__psi_array[:,vertex]
# =============================================================================
#             
#             for i in range(ramp_iterations,tot_iterations):
#                 if i % 10000 == 0:
#                     print(i,tot_iterations,np.sum(np.absolute(self.__psi)**2))
#                 
#                 self.trunc_prop(self.__dt,prop_order)
#                 self.__vertex_list[i] = self.__psi[vertex]
# =============================================================================
               # self.__long_time_prob[i-ramp_iterations] = self.__psi[vertex]
                
            self.__vertex_list = np.absolute(self.__vertex_list)**2
            self.__long_time_prob = np.absolute(self.__long_time_prob)**2
            time_list = np.zeros(self.__vertex_list.shape[0])
            for i in range(self.__vertex_list.shape[0]):
                time_list[i] = i*self.__dt

            overlap = self.ground_overlap()
        

        return time_list, self.__vertex_list, self.__long_time_prob,\
        energy_diff, overlap
        
#%%
    def ground_overlap_ramp(self, ramp = ['poly',2,1],t_0=1,gamma_0=2,
                            max_time_step = 0.0005,tolerance_order=3,
                            prop_order=4,iteration_time=10,vertex=0):
        
        
        self.__t_0 = t_0
        self.__dt = np.min([self.__t_0/1E4,max_time_step])
        ramp_iterations = int(self.__t_0/self.__dt)
        self.__overlap_list = []
        self.__vertex_list = []
        if ramp[0] == 'poly':
                power = ramp[1]
                scale = (1./self.__t_0)#*(1-(10**(-tolerance_order))**(1/power))
                aqc_gamma = lambda t : gamma_0*(1+(scale*t-1)**power)
                aqc_coeff = lambda t : (1-(scale*t-1)**power)
            
        if ramp[0] == 'exp':
            tau = self.__t_0/(tolerance_order*np.log(10))
            aqc_gamma = lambda t : gamma_0*(1+np.exp(-t/tau))
            aqc_coeff = lambda t : (1-np.exp(-t/tau))
        
        self.ground_state()
        self.__t = 0
        
        
        self.__H = -self.__L
        gamma = 2*gamma_0
        prob_coeff = 1
        inc = 0
        
        for i in range(ramp_iterations):
            if i % 500 == 0:
                print(self.__t,self.__t_0,gamma_0 - gamma,1 - prob_coeff)
            self.__t = self.__dt*i
            gamma = aqc_gamma(self.__t)
            prob_coeff = aqc_coeff(self.__t)
            self.change_H(gamma,prob_coeff)
            
            self.trunc_prop(self.__dt,prop_order)
                
            expect_energy = self.energy_expectation()
            floor_val, floor_vec = self.energy_floor()
            overlap = self.ground_overlap()
            self.__overlap_list.append(overlap)
            self.__vertex_list.append(self.__psi[vertex])

        self.__vertex_list = np.absolute(self.__vertex_list)**2
        time_list = np.zeros(len(self.__overlap_list))
        for i in range(len(self.__overlap_list)):
            time_list[i] = i*self.__dt

        return time_list, self.__overlap_list, self.__vertex_list
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
        
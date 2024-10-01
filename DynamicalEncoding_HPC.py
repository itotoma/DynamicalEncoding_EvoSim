import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score 
from sklearn.metrics import mutual_info_score
import random
import pickle
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import copy

Duration = int(sys.argv[1])
Outputnoise = float(sys.argv[2])
Filename = sys.argv[3]

warnings.simplefilter('ignore')

Num_Net = 20
Num_Gene = 5         
Num_Receptor = 2 # The first {Num_Receptor} genes are receptors.
HubGeneID = Num_Gene-1 # The last gene is output.
Num_NoisePattern = 10
Num_InputPattern= 4

G_max = 100
dt = 1        
t_end=100         
times=np.arange(0,t_end,dt) 
ReferenceTime = [0, int(Duration/dt)]



print(f"Decoder references from {ReferenceTime[0]} to {ReferenceTime[1]} step of Hub dynamics.")


#input_filter = [1,1,0,0,0]
zeros_input = np.zeros((len(times), 5))
true_labels = np.repeat(list(range(Num_InputPattern)), Num_NoisePattern)
MAX_FIT = mutual_info_score(true_labels, true_labels)
with open(f'perturb_inputs.pickle', mode='rb') as f:
    perturb_inputs = pickle.load(f)

print(MAX_FIT)

#Define indivisdual instance
class Cell:

    def __init__(self, id):
        self.adjmatrix = np.random.normal(0, 1, (Num_Gene, Num_Gene))
        self.fitness = 0
        #self.drate = np.random.normal(1, 0.01, (Num_Gene))
        self.id = id

    #def get_network(self):
    #    return self.adjmatrix

    #def get_drate(self):
    #    return self.drate

    #def get_fitness(self):
    #    return self.fitness

    #def get_theta(self):
    #    return self.theta

    def set_network(self, adjmatrix):
        self.adjmatrix = adjmatrix

    def set_fitness(self, fitness):
        self.fitness = fitness

    #def set_drate(self, drate):
    #    drate[drate < 0] = 0
    #    self.drate = drate

    #def set_theta(self):
    #    self.theta = theta





n_hill = 2
Ka = 0.5
def get_dx(x, W, d_rate, x_input):
    z = np.dot(W, x) + x_input
    dx = z**n_hill / (Ka + z**n_hill) - d_rate * x
    return dx

def solve_dx(W, d_rate, x_input_arr, x0, t_end):
    times = np.arange(0,t_end,dt) 
    x_values = np.zeros((len(times), len(x0)))
    x_values[0] = x0 + x_input_arr[0]
    x_values[0][x_values[0] < 0] = 0
    for i in range(1, len(times)):
        t = times[i-1]
        x = x_values[i-1]
        k1 = dt * get_dx(x, W, d_rate, x_input_arr[i])
        k2 = dt * get_dx(x + 0.5 * k1, W, d_rate, x_input_arr[i])
        k3 = dt * get_dx(x + 0.5 * k2, W, d_rate, x_input_arr[i])
        k4 = dt * get_dx(x + k3, W, d_rate, x_input_arr[i])
        x_values[i] = x + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_values[i][x_values[i] < 0] = 0
    return x_values
    




def get_encoder_responses(inputs_list, Outputnoise, cell, Duration):

    W = cell.adjmatrix
    #d_rate = cell.get_drate()
    d_rate = 1.0
    steady_state = solve_dx(W, d_rate, zeros_input, np.zeros(Num_Gene), 50)[-1]
    pattern1_out = np.array([solve_dx(W, d_rate, noisy_input, steady_state, Duration) for noisy_input in inputs_list[0]])
    pattern2_out = np.array([solve_dx(W, d_rate, noisy_input, steady_state, Duration) for noisy_input in inputs_list[1]])
    pattern3_out = np.array([solve_dx(W, d_rate, noisy_input, steady_state, Duration) for noisy_input in inputs_list[2]])
    pattern4_out = np.array([solve_dx(W, d_rate, noisy_input, steady_state, Duration) for noisy_input in inputs_list[3]])
    pattern1_hubout = pattern1_out[0:,0:,HubGeneID] + np.random.normal(0, Outputnoise, (Num_NoisePattern, Duration))
    pattern2_hubout = pattern2_out[0:,0:,HubGeneID] + np.random.normal(0, Outputnoise, (Num_NoisePattern, Duration))
    pattern3_hubout = pattern3_out[0:,0:,HubGeneID] + np.random.normal(0, Outputnoise, (Num_NoisePattern, Duration))
    pattern4_hubout = pattern4_out[0:,0:,HubGeneID] + np.random.normal(0, Outputnoise, (Num_NoisePattern, Duration))
    pattern1_hubout[pattern1_hubout < 0] = 0
    pattern2_hubout[pattern2_hubout < 0] = 0
    pattern3_hubout[pattern3_hubout < 0] = 0
    pattern4_hubout[pattern4_hubout < 0] = 0
    return np.vstack((pattern1_hubout, pattern2_hubout, pattern3_hubout, pattern4_hubout))

def get_decoder_responses(restricted_encoder_dynamics):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(restricted_encoder_dynamics)
    kmeans = KMeans(n_clusters = Num_InputPattern, random_state=42)
    predicted_labels = kmeans.fit_predict(scaled)
    return predicted_labels

def evaluation(cell, inputs_list, true_labels, Outputnoise, Duration):
    ami_arr = [0]*20
    for a in range(20):
        encoder_responses = get_encoder_responses(inputs_list, Outputnoise, cell, Duration)
        restricted_encoder_responses = encoder_responses[:,0:Duration]
        predicted_labels = get_decoder_responses(restricted_encoder_responses)
        ami_arr[a] = mutual_info_score(true_labels, predicted_labels)/MAX_FIT
    return np.mean(ami_arr)

def mutate(Cell):
    mutation_rate = 0.2
    bitmask = np.where(np.random.uniform(0, 1, (Num_Gene, Num_Gene))  < mutation_rate, 1, 0)
    noise_matrix = np.random.normal(0, 0.1, (Num_Gene, Num_Gene))
    mutated_net = Cell.adjmatrix + noise_matrix*bitmask
   
    #bitmask = np.where(np.random.uniform(0, 1, Num_Gene)  < mutation_rate, 1, 0)
    #noise = np.random.normal(0, 0.1, Num_Gene)
    #mutated_drate = Cell.get_drate() + noise*bitmask
    Cell.set_network(mutated_net)
    #Cell.set_drate(mutated_drate)
    return Cell


def select(population, tournament_size):    #Tournament selection
    tournament_groups = [random.sample(population, tournament_size) for i in range(Num_Net)]
    selected = [sorted(tournament_group, reverse=True, key=lambda u: u.fitness)[0] for tournament_group in tournament_groups]
    return selected


def evolution(inputs_list, Duration, Outputnoise):  
    
    # create cell instances
    Cell_group = [Cell(_id) for _id in range(Num_Net)]
    [cell.set_fitness(evaluation(cell, inputs_list, true_labels, Outputnoise, Duration)) for cell in Cell_group]

    for g in np.arange(G_max):
        
        # calculate fitness for each cell
    
        ave_fit = np.mean([cell.fitness for cell in Cell_group])
        print(f"Gen{g}, average fitness:{ave_fit}")
        print([cell.id for cell in Cell_group])
        print([np.round(cell.fitness,3) for cell in Cell_group])
        #print([np.sum(cell.adjmatrix) for cell in Cell_group])
        print("===================")
        
        most_fitted = sorted(Cell_group, reverse=True, key=lambda u: u.fitness)[0]
        
        
        

        if ave_fit > 0.90:
            print("saturated")
            return most_fitted

        # increase
        DCell_group = copy.deepcopy(Cell_group)
        #mutation
        DCell_group = [mutate(cell) for cell in DCell_group]
        [cell.set_fitness(evaluation(cell, inputs_list, true_labels, Outputnoise, Duration)) for cell in DCell_group]
        #print(np.mean([cell.get_fitness() for cell in DCell_group]))

        # selection
        Mixed_group = Cell_group + DCell_group
        Selected_group = sorted(Mixed_group, reverse=True, key=lambda u: u.fitness)[0:Num_Net]
        #Cell_group = select(Mixed_group, 4)
        Cell_group = copy.deepcopy(Selected_group)

    print("No saturated")




typical_adapted_network_list = list()
for run in range(10):
    #input_pattern1 = [[[0,1,0,0,0] + np.random.normal(0,0.01, 5)*input_filter for _ in range(len(times))] for _ in range(Num_NoisePattern)]
    #input_pattern2 = [[[1,1,0,0,0] + np.random.normal(0,0.01, 5)*input_filter for _ in range(len(times))] for _ in range(Num_NoisePattern)]
    #input_pattern3 = [[[1,0,0,0,0] + np.random.normal(0,0.01, 5)*input_filter for _ in range(len(times))] for _ in range(Num_NoisePattern)]
    #input_pattern4 = [[[0,0,0,0,0] + np.random.normal(0,0.01, 5)*input_filter for _ in range(len(times))] for _ in range(Num_NoisePattern)]
    #perturb_inputs = [input_pattern1, input_pattern2, input_pattern3, input_pattern4]
    typical_adapted_network = evolution(perturb_inputs, Duration, Outputnoise)
    typical_adapted_network_list.append(typical_adapted_network)


with open(f'{Filename}.pickle', mode='wb') as f:
    pickle.dump(typical_adapted_network_list, f)

import numpy as np
from read_scale import load_image
from representation import DiscreteChordRepresentation, point_on_inscribed_circle

def tmp_gen_array(chords, array_shape):
    empty_array = np.zeros(array_shape)

    for i in range(chords.shape[0]):
        for j in range(chords.shape[1]):
            x, y = point_on_inscribed_circle(array_shape[1], array_shape[0], i * 2 * np.pi / chords.shape[0])
            empty_array[int(y), int(x)] = 1

def objective_function(arr1, arr2):
    return np.sum((arr1 - arr2)**2)

def perturb(params):
    return params + np.random.normal(0, 0.1, params.shape)

def initiate(nrad):
    chord_adj = np.random.rand(nrad, nrad)
    chord_adj = (chord_adj + chord_adj.T) / 2  # Make the matrix symmetric
    for i in range(nrad):
        chord_adj[i, i] = 0
    
    return chord_adj

def simulated_annealing(target_file, nrad, initial_temperature=1.0, cooling_rate=0.95):
    rep = DiscreteChordRepresentation(img_path=target_file, num_pegs=nrad)
    target_array = rep.img

    current_chords = initiate(nrad)
    current_array = tmp_gen_array(current_chords, target_array.shape)
    current_loss = objective_function(target_array, current_array)
    temperature = initial_temperature

    while temperature > 0.1:  # Adjust the stopping criterion as needed
        new_chords = perturb(current_chords)  # Generate new parameters by perturbing the current parameters
        current_array = tmp_gen_array(current_chords, target_array.shape)
        new_loss = objective_function(target_array, current_array)
        difference = current_loss - new_loss

        if difference < 0:
            current_chords = new_chords
            current_loss = new_loss
        else:
            acceptance_probability = np.exp(-difference / temperature)
            if np.random.rand() < acceptance_probability:
                current_chords = new_chords
                current_loss = new_loss

        temperature *= cooling_rate

    return current_chords

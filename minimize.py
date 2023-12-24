import numpy as np
from representation import DiscreteChordRepresentation, point_on_inscribed_circle

def tmp_gen_array(chords, array_shape):
    empty_array = np.zeros(array_shape)

    for i in range(chords.shape[0]):
        for j in range(chords.shape[1]):
            x, y = point_on_inscribed_circle(array_shape[1], array_shape[0], i * 2 * np.pi / chords.shape[0])
            empty_array[int(y), int(x)] = 1

def objective_function(arr1, arr2):
    return np.sum((arr1 - arr2)**2)

def perturb(temperature, params):
    mask = np.random.choice([0, 1], size=params.shape, p=[0.9, 0.1])
    tmp = params + temperature*np.random.normal(0, 0.1, params.shape)*mask
    tmp[tmp < 0] = 0
    tmp[tmp > 1] = 1
    return tmp

def initiate(nrad):
    chord_adj = np.random.rand(nrad, nrad)
    chord_adj = (chord_adj + chord_adj.T) / 2  # Make the matrix symmetric
    for i in range(nrad):
        chord_adj[i, i] = 0
    
    return chord_adj

def simulated_annealing(target_file, nrad, initial_temperature=10.0, cooling_rate=0.99):
    rep = DiscreteChordRepresentation(img_path=target_file, num_pegs=nrad)
    target_array = rep.img

    rep.set_chords(initiate(nrad))
    current_array = rep.get_updated_image()

    current_loss = objective_function(target_array, current_array)
    temperature = initial_temperature

    tmp = DiscreteChordRepresentation(img_path=target_file, num_pegs=nrad)
    while temperature > 0.01:  # Adjust the stopping criterion as needed
        print(temperature)
        new_chords = perturb(temperature, rep.chords)  # Generate new parameters by perturbing the current parameters
        tmp.set_chords(new_chords)
        new_array = tmp.get_updated_image()
        new_loss = objective_function(target_array, new_array)
        difference = current_loss - new_loss

        if difference > 0 or np.exp(difference / temperature) > np.random.rand():
            rep.set_chords(new_chords)
            current_loss = new_loss

        temperature *= cooling_rate  # Decrease the temperature

    return rep

if __name__ == "__main__":
    dcr = simulated_annealing("triangle.jpeg", 32)
    dcr.visualize_chords()
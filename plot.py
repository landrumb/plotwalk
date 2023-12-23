def plot():
    import matplotlib.pyplot as plt
    import numpy as np

    # Number of points
    num_points = 10

    # Define the circle
    theta = np.linspace(0, 2.*np.pi, num_points, endpoint=False)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Plot the circle
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.set_aspect('equal')

    # Plot the chords
    for i in range(num_points):
        for j in range(i+1, num_points):
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=np.random.random())

    plt.show()
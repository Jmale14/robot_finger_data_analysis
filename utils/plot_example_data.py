import matplotlib.pyplot as plt

def plt_dataset(fold):
    plt.subplot(3, 1, 1)
    plt.plot(fold[:,0], label=f'accx')
    plt.plot(fold[:,1], label=f'accy')
    plt.plot(fold[:,2], label=f'accz')
    plt.xlabel('Time')
    plt.ylabel('Acc')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(fold[:,3], label=f'gyx')
    plt.plot(fold[:,4], label=f'gyy')
    plt.plot(fold[:,5], label=f'gyz')
    plt.xlabel('Time')
    plt.ylabel('Gyro')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(fold[:,6], label=f'pressure')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plots
    plt.show()

def plot_example_data(normalised_folds):
    for i in range(0, 10):
        plt_dataset(normalised_folds[0][0][i])
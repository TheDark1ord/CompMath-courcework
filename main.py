import numpy as np
from scipy.optimize import brentq, fminbound
from scipy.integrate import quad
import matplotlib.pyplot as plt

def getFirstA():
    equation = lambda z: z - np.cos(np.divide(0.7854 - z * np.sqrt(1 - np.square(z)), 1 + 2 * np.square(z)))
    return 1.7893 * brentq(equation, 0, 1)

def getSecondA():
    integralFunction = lambda x : np.divide(np.sin(x), np.square(np.cos(x)))
    regularFunction = lambda x: np.sqrt(x) - np.cos(x)

    x = fminbound(regularFunction, 5, 8)
    y = quad(integralFunction, 0, np.pi / 4)[0]

    return x * y

def heatEquation(phi, a, l, t_end, dx, dt):
    N, K = int(l / dx) + 1, int(t_end / dt) + 1
    result = np.ndarray(shape=(K, N))

    length = np.arange(dx, l, dx)
    result[0] = np.array([0, *list(map(phi, length)), 0])

    A = a / dx**2
    B = 2*a / dx**2 + 1/dt

    for k in range(1, K):
        alpha = np.zeros(N)
        beta = np.zeros(N)

        result[k][0] = 0
        result[k][-1] = 0

        for n in range(1, N):
            alpha[n] = (A / (B - A * alpha[n-1]))
            beta[n] = ((A * beta[n-1] + result[k-1][n]/dt) / (B - A * alpha[n-1]))

        for n in range(2, N):
            result[k][N - n] = alpha[N - n] * result[k][N - n + 1] + beta[N - n]

    return result


def main():
    l, t_end, l_step, t_step = 1, 3, 0.01, 0.05

    a1 = getFirstA()
    a2 = getSecondA()

    print(a1, a2)

    phi = lambda x: 100
    result1 = heatEquation(phi, a1, l, t_end, l_step, t_step)
    result2 = heatEquation(phi, a2, l, t_end, l_step, t_step)


    plt.grid()
    plt.xlabel("Length (cm)")
    plt.ylabel("Temperature (C°)")

    plt.plot(np.arange(0, l + l_step, l_step), result1[0])
    plt.savefig("Initial distribution.png")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(np.arange(0, l + l_step, l_step), result1[60])
    ax1.grid()
    ax1.set_xlabel("Length (cm)")
    ax1.set_ylabel("Temperature (C°)")
    ax1.title.set_text("a = " + str(round(a1, 4)))

    ax2.plot(np.arange(0, l + l_step, l_step), result2[60])
    ax2.grid()
    #ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel("Length (cm)")
    ax2.set_ylabel("Temperature (C°)")
    ax2.title.set_text("a = " + str(round(a2, 4)))

    fig.suptitle('Step = 60', fontsize=16)
    fig.tight_layout()
    plt.savefig("First step.png")




if __name__ == "__main__":
    main()
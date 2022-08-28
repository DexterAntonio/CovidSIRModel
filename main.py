import matplotlib.pyplot as plt
from src.pandemic import Pandemic

def run_pandemic():
    pandemic = Pandemic()
    pandemic.run(100)
    plt.figure(1, dpi=150)
    t = pandemic.t
    plt.plot(t, pandemic.S, label='S')
    plt.plot(t, pandemic.I, label='I')
    plt.plot(t, pandemic.R, label='R')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_pandemic()


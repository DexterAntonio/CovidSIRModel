import matplotlib.pyplot as plt
from pandemic import Pandemic

def test_Pandemic():
    pandemic = Pandemic()
    pandemic.run()
    plt.figure(1, dpi=150)
    t = pandemic.t
    plt.plot(t, pandemic.S, label='S')
    plt.plot(t, pandemic.I, label='I')
    plt.plot(t, pandemic.R, label='R')
    plt.legend()


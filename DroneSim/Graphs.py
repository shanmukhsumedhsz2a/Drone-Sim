import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.animation import FuncAnimation

style.use('ggplot')

fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

fig2, axe = plt.subplots(2, 1, figsize=(10, 8))
ax12 = axe[0]
ax22 = axe[1]


def animate(i):
    with open('GraphingStuff.pickle', 'rb') as f:
        xs, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = pickle.load(f)

    #z = xs[-100:]
    #xs = z

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.set_title('Best Fitness', loc='left')
    ax2.set_title('Top 10 Average', loc='left')
    ax3.set_title('Average Fitness', loc='left')
    ax1.plot(xs, y1)
    ax2.plot(xs, y2)
    ax3.plot(xs, y3)

    ax12.clear()
    ax22.clear()
    ax12.set_title('Maximums',loc='left')
    ax22.set_title('Averages', loc='left')
    ax12.plot(xs,y4,label='Mutant')
    ax12.plot(xs,y5,label='Cross')
    ax12.plot(xs,y6,label='Elite')
    ax12.plot(xs,y7,label='New')
    ax12.legend(loc='upper left')

    ax22.plot(xs, y8, label='Mutant')
    ax22.plot(xs, y9, label='Cross')
    ax22.plot(xs, y10, label='Elite')
    ax22.plot(xs, y11, label='New')
    ax22.legend(loc='upper left')


def PlotGraphs():
    ani = FuncAnimation(fig, animate, interval=10)
    an2 = FuncAnimation(fig2, animate, interval=10)

    plt.show()


PlotGraphs()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_RB(rb, acceleration=False, save=False, path=''):

    red = 1.0
    green = 0
    blue = 0
    step = 1.0 / (len(rb) / 2)

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    for exp in rb:

        if acceleration is True:
            x_pos, y_pos = exp[0][0]
            x_vel, y_vel = exp[4][0]
        else:
            x_pos, y_pos = exp[0]
            x_vel, y_vel = exp[4]

        x_vel -= x_pos
        y_vel -= y_pos

        color = matplotlib.colors.to_hex((red, green, blue))

        plt.plot(x_pos, y_pos, '.', color=color)
        #plt.arrow(x_pos, y_pos, x_vel, y_vel, color="blue", width=0.001, head_width=0.008)

        if green < 1.0:
            green = min(green + step, 1.0)
        else:
            red = max(red - step, 0)

    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, angle=0.0, facecolor='none', edgecolor='blue', linewidth=1))

    if save:
        plt.savefig(path + "/rb.png")

    plt.show()



def visualize_Q_arrow(q_values, save=False, path=''):

    qs = q_values[:,0]
    qs -= np.min(qs)
    qs /= np.max(qs)
    states = q_values[:,1:]

    red = 1.0
    green = 1.0
    blue = 0

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    for q,state in zip(qs,states):

        x_pos , y_pos = state[0],state[1]
        x_vel , y_vel = state[2]*0.1,state[3]*0.1

        color = matplotlib.colors.to_hex((red*(1-q), green*q, blue))
        plt.plot(x_pos, y_pos, '.', color=color)
        plt.arrow(x_pos, y_pos, x_vel, y_vel, color=color, width=0.001, head_width=0.008)

    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, angle=0.0, facecolor='none', edgecolor='blue', linewidth=1))

    if save:
        plt.savefig(path + "/Q_arrow.png")

    plt.show()



def visualize_Q_contour(q_values, save=False, path=''):

    qs = q_values[:,0]
    states = q_values[:,1:]

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    colorset = plt.tricontourf(states[:,0], states[:,1], qs)
    colorbar = plt.colorbar(colorset)
    colorbar.ax.set_ylabel('Q values')

    plt.title(r'$Q(s, \pi(s))$')
    plt.xlabel('x dimension')
    plt.ylabel('y dimension')
    plt.xticks(np.arange(-1, 1, step=0.1))
    plt.yticks(np.arange(-1, 1, step=0.1))

    if save:
        plt.savefig(path + '/Q_contour.png')

    plt.show()


def visualize_Q_contour_time(all_q_values, save=False, path=''):

    fig = plt.figure()

    plt.set_cmap('RdYlGn')

    def animate(i):
        q_values = all_q_values[i]
        qs = q_values[:,0]
        states = q_values[:,1:]
        return plt.tricontourf(states[:,0], states[:,1], qs)

    anim = animation.FuncAnimation(fig, animate, interval=200, frames=len(all_q_values))

    if save:
        anim.save(path + "/Q_contour_time.gif", writer='imagemagick', fps=4)





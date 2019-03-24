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



def visualize_Pi(pi_values, save=False, path='', inline=True):
    states = pi_values[:,:2]
    pis = pi_values[:,2:]

    fig, ax = plt.subplots(1)

    for pi,state in zip(pis,states):

        x_pos , y_pos = state[0],state[1]
        x_pi , y_pi = pi[0]*0.1,pi[1]*0.1

        plt.plot(x_pos, y_pos, '.', color="black")
        plt.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.001, head_width=0.008)

    plt.title(r'$Pi(s)$')

    if save:
        plt.savefig(path + "/Pi_arrow.png")
    
    if inline is True:
        plt.show()

    else:
        return fig


def visualize_Pi_time(all_pi_values, save=False, path='',eval_freq=1):

    fig, ax = plt.subplots(1)

    def animate(i):
        fig.clear()
        ax = fig.add_subplot(111)
        pi_values = all_pi_values[i]

        states = pi_values[:,:2]
        pis = pi_values[:,2:]

        for pi,state in zip(pis,states):

            x_pos , y_pos = state[0],state[1]
            x_pi , y_pi = pi[0]*0.1,pi[1]*0.1

            ax.plot(x_pos, y_pos, '.', color="black")
            ax.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.001, head_width=0.008)

        ax.set_title(r'$Pi(s)$ timestep : {}'.format(i*eval_freq))
        ax.set_xlabel('x dimension')
        ax.set_ylabel('y dimension')
        ax.set_xticks(np.arange(-1, 1, step=0.1))
        ax.set_yticks(np.arange(-1, 1, step=0.1))

    anim = animation.FuncAnimation(fig, animate, interval=200, frames=len(all_pi_values))

    if save:
        anim.save(path + "/Pi_arrow_time.gif", writer='imagemagick', fps=4)



def visualize_Q(q_values, save=False, path='', inline=True):

    qs = q_values[:,0]
    states = q_values[:,1:]

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    colorset = ax.tricontourf(states[:,0], states[:,1], qs)
    colorbar = fig.colorbar(colorset)
    colorbar.ax.set_ylabel('Q values')

    ax.set_title(r'$Q(s, \pi(s))$')
    ax.set_xlabel('x dimension')
    ax.set_ylabel('y dimension')
    ax.set_xticks(np.arange(-1, 1, step=0.1))
    ax.set_yticks(np.arange(-1, 1, step=0.1))

    if save:
        fig.savefig(path + '/Q_contour.png')

    if inline is True:
        plt.show()

    else:
        return fig


def visualize_Q_time(all_q_values, save=False, path='',eval_freq=1):

    fig, ax = plt.subplots(1)

    plt.set_cmap('RdYlGn')

    def animate(i):
        fig.clear()
        ax = fig.add_subplot(111)
        q_values = all_q_values[i]
        qs = q_values[:,0]
        states = q_values[:,1:]
        colorset = ax.tricontourf(states[:,0], states[:,1], qs)
        colorbar = plt.colorbar(colorset, aspect=20, format="%.4f")
        colorbar.ax.set_ylabel('Q values')
        colorbar.ax.tick_params(labelsize=10)
        ax.set_title(r'$Q(s, \pi(s))$ timestep : {}'.format(i*eval_freq))
        ax.set_xlabel('x dimension')
        ax.set_ylabel('y dimension')
        ax.set_xticks(np.arange(-1, 1, step=0.1))
        ax.set_yticks(np.arange(-1, 1, step=0.1))

    anim = animation.FuncAnimation(fig, animate, interval=200, frames=len(all_q_values))

    if save:
        anim.save(path + "/Q_contour_time.gif", writer='imagemagick', fps=4)

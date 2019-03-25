import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_RB(rb, acceleration=False, save=False, path=''):

    red = 1.0
    green = 0
    blue = 0
    step = 1.0 / (len(rb) / 2)

    fig, ax = plt.subplots(1)
    divider = make_axes_locatable(ax)

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

    plt.title("Replay buffer states")
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.xticks(np.arange(-1, 1, step=0.2))
    plt.yticks(np.arange(-1, 1, step=0.2))

    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    colorbar = matplotlib.colorbar.ColorbarBase(cax, norm=norm, orientation='vertical')
    colorbar.set_ticks([0, 1])
    colorbar.ax.set_yticklabels(['Old', 'New'])


    if save:
        plt.savefig(path + "/rb.png")

    plt.show()



def visualize_Pi(pi_values, save=False, name="Pi_arrow.png", title=r'$\pi(s)$', path='', inline=True):
    states = pi_values[:,:2]
    pis = pi_values[:,2:]

    fig, ax = plt.subplots(1)

    for pi,state in zip(pis,states):

        x_pos , y_pos = state[0],state[1]
        x_pi , y_pi = pi[0]*0.1,pi[1]*0.1

        plt.plot(x_pos, y_pos, '.', color="black")
        plt.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.001, head_width=0.02)

    plt.title(title)
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.xticks(np.arange(-1, 1, step=0.2))
    plt.yticks(np.arange(-1, 1, step=0.2))

    if save:
        plt.savefig(path + "/"+name)
    
    if inline is True:
        plt.show()

    else:
        return fig


def visualize_Pi_time(all_pi_values, save=False, name="Pi_arrow_time.gif", title=r'$\pi(s)$', path='', steps_name="", steps=None, fps=4):

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
            ax.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.001, head_width=0.02)

        ax.set_title(title + r' ; {}'.format(steps_name) + ' {}'.format(int(steps[i])))
        ax.set_xlabel('1st dimension')
        ax.set_ylabel('2nd dimension')
        ax.set_xticks(np.arange(-1, 1, step=0.2))
        ax.set_yticks(np.arange(-1, 1, step=0.2))

    anim = animation.FuncAnimation(fig, animate, interval=200, frames=len(steps))

    if save:
        anim.save(path + "/"+name, writer='imagemagick', fps=fps)



def visualize_Q(q_values, save=False, name='Q_contour.png', title=r'$Q(s,\pi(s))$', path='', inline=True):

    qs = q_values[:,0]
    states = q_values[:,1:]

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    colorset = ax.tricontourf(states[:,0], states[:,1], qs)
    colorbar = fig.colorbar(colorset)
    colorbar.ax.set_ylabel('Q values')

    ax.set_title(title)
    ax.set_xlabel('1st dimension')
    ax.set_ylabel('2nd dimension')
    ax.set_xticks(np.arange(-1, 1, step=0.2))
    ax.set_yticks(np.arange(-1, 1, step=0.2))

    if save:
        fig.savefig(path + '/'+name)

    if inline is True:
        plt.show()

    else:
        return fig


def visualize_Q_time(all_q_values, save=False, name="Q_contour_time.gif", title=r'$Q(s,\pi(s))$', path='', steps_name="", steps=None, fps=4):

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

        ax.set_title(title + r' ; {}'.format(steps_name) + ' {}'.format(int(steps[i])))
        ax.set_xlabel('1st dimension')
        ax.set_ylabel('2nd dimension')
        ax.set_xticks(np.arange(-1, 1, step=0.2))
        ax.set_yticks(np.arange(-1, 1, step=0.2))

    anim = animation.FuncAnimation(fig, animate, interval=200, frames=len(steps))

    if save:
        anim.save(path + "/"+name, writer='imagemagick', fps=fps)

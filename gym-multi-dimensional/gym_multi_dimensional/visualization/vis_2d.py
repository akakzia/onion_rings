import matplotlib
import matplotlib.pyplot as plt

def visualize(rb):

    red = 1.0
    green = 0
    blue = 0
    step = 1.0 / (len(rb) / 2)

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    for exp in rb:
        x_pos , y_pos = exp[0][0]
        x_vel , y_vel = exp[4][0]
        x_vel -= x_pos
        y_vel -= y_pos

        color = matplotlib.colors.to_hex((red, green, blue))
        plt.plot(x_pos, y_pos, '.', color=color)
        plt.arrow(x_pos, y_pos, x_vel, y_vel, color="blue", width=0.001, head_width=0.008)

        if green < 1.0:
            green = min(green + step, 1.0)
        else:
            red = max(red - step, 0)


    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, angle=0.0, facecolor='none', edgecolor='blue', linewidth=1))
    plt.show()

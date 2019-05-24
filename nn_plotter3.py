import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

[test, transformed_data, col, loss, acc] = np.load('nn_trans_evol3.npy')

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 0), colspan=1), aspect='equal', autoscale_on=False,
                      xlim=(-0.1, 1.1), ylim=(-0.2, 1))
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 1), colspan=1), aspect='equal', autoscale_on=False,
                      projection='3d',
                      xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

ax3 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 2), colspan=1), aspect='equal', autoscale_on=False,
                      xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax3.set_xticks([])
ax3.set_yticks([])

ax4 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 3), colspan=1), aspect='equal', autoscale_on=False,
                      xlim=(0, 1), ylim=(0, 1))
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((1, 0), colspan=4), autoscale_on=False,
                      xlim=(0, 40), ylim=(0, 1))


ax1.set_title('Test Data')
ax2.set_title('Layer 1')
ax3.set_title('Layer 2')
ax4.set_title('Soft-max Layer')
ax5.set_title('Loss')

ax1.scatter(test[:, 0], test[:, 1], 50, c=col)

points1 = ax2.scatter(transformed_data[0][-3][:, 0],
                      transformed_data[0][-3][:, 1],
                      transformed_data[0][-3][:, 2], c=col)


points2 = ax3.scatter(transformed_data[0][-2][:, 0], transformed_data[0][-2][:, 1], s=50)


points3 = ax4.scatter(transformed_data[0][-1][:, 0], transformed_data[0][-1][:, 1], s=50)

line, = ax5.plot([], [])

loss_text = ax5.text(0.8, 0.9, '', transform=ax5.transAxes)
acc_text = ax5.text(0.8, 0.8, '', transform=ax5.transAxes)


def animate(i):
    points1._offsets3d = (transformed_data[i][-3][:, 0],
                          transformed_data[i][-3][:, 1],
                          transformed_data[i][-3][:, 2])
    points1.set_array(col)

    points2.set_offsets(transformed_data[i][-2])
    points2.set_array(col)

    points3.set_offsets(transformed_data[i][-1])
    points3.set_array(col)
    line.set_data(range(i+1), loss[0:i+1])
    loss_text.set_text('{:14} {:2.2f}'.format('Loss:', loss[i]))
    acc_text.set_text('{:10} {:2.2f} %'.format('Accuracy:', acc[i] * 100))
    return points1, points2, points3, line, loss_text, acc_text,


anim = animation.FuncAnimation(fig, animate,
                               frames=40, interval=400, blit=False)


anim.save('3d_doughnut.gif', writer='imagemagick', fps=5)
plt.show()

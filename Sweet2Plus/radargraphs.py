import matplotlib.pyplot as plt
import numpy as np
import ipdb

def radar_plot(labels,values,title,filename_path,single_neuron=True,Grouping=[]):
    """
    Citation: https://www.pythoncharts.com/matplotlib/radar-charts/
    """
    if not Grouping:
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        angles += angles[:1]
        labels += labels[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        if single_neuron:
            values += values[:1]
            ax.plot(angles, values, color='#1aaf6c', linewidth=1)
            ax.fill(angles, values, color='#1aaf6c', alpha=0.1)
        else:
            for value in values:
                value += value[:1]
                arr=np.asarray(value)
                if value[3]==arr.max():
                    ax.plot(angles, value, linewidth=2,color='red')
                else:
                    ax.plot(angles, value, linewidth=2,color='black')
                #ax.fill(angles, value, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), labels)

        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        #ax.set_ylim(0, 100)
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])
        ax.set_rlabel_position(180 / num_vars)

        # Add some custom styling.
        # Change the color of the tick labels.
        ax.tick_params(colors='#222222')
        # Make the y-axis (0-100) labels smaller.
        ax.tick_params(axis='y', labelsize=8)
        # Change the color of the circular gridlines.
        ax.grid(color='#AAAAAA')
        # Change the color of the outermost gridline (the spine).
        ax.spines['polar'].set_color('#222222')
        # Change the background color inside the circle itself.
        ax.set_facecolor('#FAFAFA')
        ax.set_title(title, y=1.08)
        plt.savefig(filename_path)
    else:
        Grouping=Grouping[0]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        angles += angles[:1]
        labels += labels[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        if single_neuron:
            values += values[:1]
            ax.plot(angles, values, color='#1aaf6c', linewidth=1)
            ax.fill(angles, values, color='#1aaf6c', alpha=0.1)
        else:
            for value,group in zip(values,Grouping):
                value = np.append(value,value[0])
                arr=np.asarray(value)
                if group==0:
                    ax.plot(angles, value, linewidth=2,color='black')
                else:
                    ax.plot(angles, value, linewidth=2,color='red')
                #ax.fill(angles, value, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), labels)

        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        #ax.set_ylim(0, 100)
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])
        ax.set_rlabel_position(180 / num_vars)

        # Add some custom styling.
        # Change the color of the tick labels.
        ax.tick_params(colors='#222222')
        # Make the y-axis (0-100) labels smaller.
        ax.tick_params(axis='y', labelsize=8)
        # Change the color of the circular gridlines.
        ax.grid(color='#AAAAAA')
        # Change the color of the outermost gridline (the spine).
        ax.spines['polar'].set_color('#222222')
        # Change the background color inside the circle itself.
        ax.set_facecolor('#FAFAFA')
        ax.set_title(title, y=1.08)
        plt.savefig(filename_path)
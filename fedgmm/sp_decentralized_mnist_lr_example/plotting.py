import numpy as np
import matplotlib.pyplot as plt
class PlotElement:
    def __init__(self, x, y, title, normalize=False):
        self.x = x
        self.y = y
        self.title = title
        self.normalize = normalize

    def plot(self, ax=None, save_path=None):
        if ax is None:
            fig, ax = plt.subplots()

        if self.normalize:
            x_data = self.x / self.x.max()
            y_data = self.y / self.y.max()
        else:
            x_data = self.x
            y_data = self.y

        # Use different markers for each plot
        marker = {'FedDeepGMM-SGDA': 'o',  
                  'Actual Causal Effect': 's',  # Triangle Down
                  'DeepGMM-OAdam':'v',
                  'DeepGMM-SMDA':'p',
                  'FedDeepGMM-SMDA':'x',
                  'DeepGMM-SGDA': 'd'}  # Square

        # Plot data with specified styles
        ax.plot(x_data, y_data, label=self.title, marker=marker[self.title],
                markersize=1, linewidth=1)  # Increased markersize and linewidth
        # ax.set_title("Mse vs No. of Communication Rounds")
        ax.legend(loc='lower right', fontsize=12, framealpha=0.8, handlelength=2)  # Increase the numeric value as needed  # Adjusted fontsize

        if save_path:
            # plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            # plt.savefig(f"{save_path}.pdf")
            plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight', format='pdf')
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', format='png')


        return ax



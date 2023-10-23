import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.linalg as linalg
from scipy.linalg import hankel
from scipy.linalg import svd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 11, 4

class SSA(object):
    '''Singular Spectrum Analysis object'''
    def __init__(self, time_series, L, criteria = 0.9997): # The time_series should be in pandas dataframe
        """
        Perform Singular Spectrum Analysis on a time series.

        Parameters:
            series (array-like): The input time series.
            L (int): The window size for embedding the time series.
            criteria (float): The contribution bar

        """        
        self.ts = pd.DataFrame(time_series)
        self.N = len(self.ts)
        self.L = L
        self.K = self.N - self.L + 1
        self.criteria = criteria
        if not 2 <= self.L <= self.N/2 + 0.05:
            print(self.N/2 + 0.05 - self.L)
            raise ValueError("The window length must be in the interval [2, N/2].")
    
    def decomposition(self):
        """
        Perform Embedding and SVD.

        Return:
            U
            Sigma: the obtained Singular Values
            Vt
            rank: the  rank of the matrix XXt
        """
        # Embed the time series into a Hankel matrix
        hankel_mat = hankel(self.ts, np.zeros(self.L))
        hankel_mat = hankel_mat[:-self.L + 1, :]
        # print(hankel_mat.shape)
        
        # Perform Singular Value Decomposition (SVD) on the Hankel matrix
        self.U, self.Sigma, self.Vt = svd(hankel_mat)
        self.rank = np.linalg.matrix_rank(hankel_mat)

        return self.U, self.Sigma, self.Vt, self.rank
    
    def calculate_component_contribution(self):
        """
        Calculate the contribution of each component based on their singular values.

        Parameters:
            singular_values (array-like): The singular values obtained from Singular Spectrum Analysis.

        Returns:
            contributions (list): List of contributions corresponding to each component.
        """
        singular_values = self.Sigma
        # Calculate the total variance (sum of squares of all singular values)
        total_variance = sum(sv ** 2 for sv in singular_values)

        # Calculate the contribution of each component
        self.contributions = [(sv ** 2) / total_variance for sv in singular_values]

        # print(self.contributions)
        
        # Calculate how many components should remain
        sum_of_contributions = 0
        index = 0
        for i in self.contributions:
            sum_of_contributions += i
            index += 1
            if sum_of_contributions > self.criteria:
                break
        # print(index)
        self.num_components = index
        return index

    def reconstruction(self):
        # Truncate the matrices to retain only the desired components
        U_trunc = self.U[:, :self.num_components]
        S_trunc = np.diag(self.Sigma[:self.num_components])
        Vt_trunc = self.Vt[:self.num_components, :]

        components = np.zeros((self.N, self.num_components))
        for i in range(self.num_components):
            X_elem = S_trunc[i][i] * np.outer(U_trunc[:, i], Vt_trunc[i, :])
            X_rev = X_elem[::-1]
            components[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
        self.components = components
        return components
    
    def plot_components(self, start = 0):
        for i in self.components.T[start:]:
            plt.plot(i)
        plt.show()
    
    def obtain_correlation_matrix(self):
        df = pd.DataFrame(self.components)
        # print(df.shape)
        correlation_matrix = df.corr()
        # Create a heatmap using Seaborn
        plt.figure(figsize=(10, 8))  # Set the size of the plot
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 5})

        # Add plot title
        plt.title("Correlation Matrix Heatmap")
        plt.show()
        return correlation_matrix
    
    def grouping(self, group_number = 2):
        # As the maintained series don't show great correlation, perform the K-means Clustering to group the series into 2 or less series
        desired_num_clusters = min(group_number, self.components.shape[1])
        # if (self.contributions[0] >= criteria):
        #     return 'You should reduce the group number'
        
        # print('The number of newly created series is:', str(desired_num_clusters))
        flatten_timeseries = self.components.T
        kmeans = KMeans(n_clusters=desired_num_clusters, random_state=2023)
        cluster_labels = kmeans.fit_predict(flatten_timeseries)
        combined_series = np.zeros((desired_num_clusters, flatten_timeseries.shape[1]))
        for cluster_idx in range(desired_num_clusters):
            combined_series[cluster_idx] = np.sum(flatten_timeseries[cluster_labels == cluster_idx], axis=0)

        self.combined_series = combined_series
        return combined_series

    def plot_combined(self):
        # this method will combine all the series we obtained, and make a comparison with the origian series
        reconstructed_series = np.sum(self.combined_series, axis = 0)
        reconstructed_series = pd.DataFrame(reconstructed_series, index = self.ts.index)
        plt.figure(figsize=(10, 6))
        plt.plot(self.ts, label='Original Time Series', linewidth=2)
        plt.plot(reconstructed_series, label='Reconstructed Time Series', linestyle='dashed', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Singular Spectrum Analysis')
        plt.legend()
        plt.show()
    
if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    import scipy.linalg as linalg
    from scipy.linalg import hankel
    from scipy.linalg import svd
    from sklearn.cluster import KMeans
    import warnings
    warnings.filterwarnings("ignore")

    path = './Data/Grouped/'
    group = ['Consumer and Service Sectors', 'Financial, Healthcare, Technology, and Utilities Sectors', 'Industrial and Infrastructure Sectors']
    def read_stock(file_name, directory):
        stock = pd.read_csv(directory + file_name, header = [0])
        index = pd.to_datetime(stock['Date'])
        stock.drop('Date', axis = 1)
        stock.index = index
        stock = stock[['Adj Close']]
        return stock

    stock_example = read_stock('ALL.AX.csv', path + group[0] + '/')

    ssa = SSA(stock_example, 20, criteria= 0.9995)
    U, Sigma, Vt, rank = ssa.decomposition()
    index = ssa.calculate_component_contribution()
    components = ssa.reconstruction()
    ssa.plot_components()
    ssa.obtain_correlation_matrix()
    combined_series = ssa.grouping()
    ssa.plot_combined()
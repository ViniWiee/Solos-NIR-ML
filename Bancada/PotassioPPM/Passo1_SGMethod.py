import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


## lendo e preparando o dataset

Dados = pd.read_excel('Dados.xlsx',sheet_name='Bancada')
Dados.info()

WL = list(Dados.columns.values)[2:553]
y = np.asarray(Dados.iloc[:,557], dtype= 'int')
spectrum = np.asarray(Dados.iloc[:,2:553])

"""
A seguir, iremos utilizar o método Savitzky-Golay com segunda derivada. 
A Função foi encontrada em  https://nirpyresearch.com/savitzky-golay-smoothing-method/

"""

w = 17
p = 2

savitskyGolaySpectrum= savgol_filter(spectrum, w, polyorder=p, deriv=2)


with plt.style.context(('ggplot')):
    ax1 = plt.subplot(211)
    plt.plot(WL, spectrum.T)
    plt.ylabel('Reflectancia NIR')


    ax2 = plt.subplot(212)
    plt.plot(WL, savitskyGolaySpectrum.T)
    plt.xlabel('Comprimento de Onda ($cm^{-1}$)')
    plt.show()

pd.DataFrame(savitskyGolaySpectrum).to_csv("CSV/savitskygolayspectrum.csv", header=None, index=None)
pd.DataFrame(y).to_csv("CSV/y.csv", header=None, index=None)

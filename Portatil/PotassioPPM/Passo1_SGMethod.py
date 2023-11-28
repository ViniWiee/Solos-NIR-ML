import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


## lendo e preparando o dataset

Dados = pd.read_excel('Dados.xlsx',sheet_name='Bancada')
Dados.info()


y = np.asarray(Dados.iloc[:,262])
y = np.asarray(y,dtype='int')
spectrum = np.asarray(Dados.iloc[:,2:258])

"""
A seguir, iremos utilizar o método Savitzky-Golay com segunda derivada. 
A Função foi encontrada em  https://nirpyresearch.com/savitzky-golay-smoothing-method/

"""

w = 17
p = 2

savitskyGolaySpectrum= savgol_filter(spectrum, w, polyorder=p, deriv=2)

fig, axs = plt.subplots(2, 1)  # initialize a figure with 2 axis
axs[0].plot(spectrum)  # plot the data
axs[0].set_xlabel('wavenumber [$\mathregular{cm^{-1}]}$')  # matplotlib can interpret this LateX expression
axs[0].set_ylabel('Absorbancia')
axs[0].grid(True)  # show a grid to help guide the viewer's eye
axs[0].set_title("Espectro Bruto Original")  # set a title from the filename of the input data
axs[1].plot(savitskyGolaySpectrum)
axs[1].set_xlabel('wavenumber [$\mathregular{cm^{-1}]}$')
axs[1].set_title("Espectro pós SAVGOLFilter")
axs[1].grid(True)
fig.tight_layout()  # a standard option for matplotlib
plt.show()

pd.DataFrame(savitskyGolaySpectrum).to_csv("CSV/savitskygolayspectrum.csv", header=None, index=None)
pd.DataFrame(y).to_csv("CSV/y.csv", header=None, index=None)

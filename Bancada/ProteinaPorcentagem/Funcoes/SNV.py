import numpy as np


def snv(input_data):
    """
        :snv: A correction technique which is done on each
        individual spectrum, a reference spectrum is not
        required
        :param input_data: Array of spectral data
        :type input_data: DataFrame

        :returns: data_snv (ndarray): Scatter corrected spectra
    """

    input_data = np.asarray(input_data)

    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    for i in range(data_snv.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])

    return data_snv

"""
Este arquivo é baseado no método de PCA escrito pelo professor Jefferson Tales de Oliva.

O código apresentado neste arquivo é um código modificado, atualizado para funcionar com o meu código de pré processamento
de um espectro NIR.

"""


def apply_PCA(table, n_components):
    from sklearn.decomposition import PCA
    #from paje import feature_file_processor


    # standardize features: PCA is sensible to the measure scale
    # x = StandardScaler().fit_transform(table)

    # apply PCA
    pca = PCA(n_components = n_components)
    pc = pca.fit_transform(table)

    # generate a feature table and return it
    return pc
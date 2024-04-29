from adnmtf import NMF, NTF # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pkg_resources  # part of setuptools
import sys
import sklearn.decomposition

version = pkg_resources.require("adilsm")[0].version
print("adilsm version="+version)

def format_loadings(h, list_columns):
    # Format loadings
    df_h = pd.DataFrame(data=h)
    n_comp = len(df_h.columns)
    df_h.columns = ['theme_' + str(i) for i in range(1, n_comp + 1)]
    df_h.insert(loc=0, column='label', value=(list_columns))

    # Add description index
    df_h['description'] = df_h['label']
  
    return df_h

def generate_h_sparse(h, q_ism, n_items, n_comp, n_scores, sparsity_coeff=.8):
    # Calculate hhii of each h column and generate sparse loadings
    hhii = np.zeros(n_comp, dtype=int)
    h_threshold = np.zeros(n_comp)

    if q_ism is not None:
        i1 = 0
        for i_score in range(0,n_scores):
            i2 = i1+n_items[i_score]
            h[i1:i2,:] *= q_ism[i_score,:]
            i1 = i2

    for i in range(0,n_comp):
        # calculate inverse hhi
        if np.max(h[:,i]) > 0:
            hhii[i] = int(round(np.sum(h[:, i])**2 / np.sum(h[:, i]**2)))
        
        # sort the dataframe by score in descending order
        h_threshold[i] = np.sort(h[:, i], axis=0)[::-1][hhii[i]-1] * sparsity_coeff
        

    h_sparse = np.where(h < h_threshold[None,:], 0, h)
    
    return h_sparse, hhii

# def tensorize_scores(m0, m0_nan_0, m0_weight, n_items, n_scores, n_themes,
#                      hv_sparse, max_iter_mult=200, fast_mult_rules=True):
#     EPSILON = np.finfo(np.float32).eps

#     i1 = 0
#     for i_score in range(n_scores):
#         i2 = i1+n_items[i_score]
#         #  use only rows with non-missing view for the current view
#         non_missing_rows = np.where(np.sum(m0_weight[:, i1:i2], axis=1) > 0)[0]
#         w_score = np.zeros((m0.shape[0], n_themes))
#         w_score_non_missing = np.ones((len(non_missing_rows), n_themes))
#         h_score = hv_sparse[i_score].copy()
#         m0_score = m0_nan_0[non_missing_rows, i1:i2]
#         m0_weight_score = m0_weight[non_missing_rows, i1:i2]
#         i1=i2

#         if i_score == 0:
#             h_updated_sparse_0 = h_score
#         else:
#             h_updated_sparse_0 = np.vstack((h_updated_sparse_0, h_score))
 
#         for _ in range(0, max_iter_mult):
#             # Weighted multiplicative rules
#             m0_score_est = w_score_non_missing @ h_score.T
#             w_score_non_missing *= (m0_score @ h_score / ((m0_weight_score*m0_score_est) @ h_score + EPSILON))

#         w_score[non_missing_rows,:] = w_score_non_missing

#         # Generate embedding tensor and  initialize h_updated
#         if i_score == 0:
#             tensor_score = w_score
#             h_updated = h_score
#         else:
#             tensor_score = np.hstack((tensor_score, w_score))
#             h_updated = np.vstack((h_updated, h_score))

def integrate_scores(m0_nan_0, m0_weight, h_sparse, w_ism, h_ism, q_ism, n_scores, n_items, n_themes, update_w_ism=True, update_h_ism=True,
                     max_iter_mult=200, fast_mult_rules=True, sparsity_coeff=.8, use_scikit_NMF=False):
    
    EPSILON = np.finfo(np.float32).eps

    # Generate w for each score, based on sparse loadings and create tensor_score
    if use_scikit_NMF:
        my_nmfmodel = sklearn.decomposition.NMF(n_components=n_themes, init='custom', solver='mu', beta_loss='frobenius', max_iter=max_iter_mult, random_state=0)

    # Extract score-related items
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_items[i_score]

        m0_score = m0_nan_0[:, i1:i2]
        m0_weight_score = m0_weight[:, i1:i2]
        w_score = w_ism.copy().astype(m0_score.dtype)
        h_score = h_sparse[i1:i2, :].copy().astype(m0_score.dtype)

        i1 = i2

        # Apply multiplicative updates to preserve h sparsity
        if use_scikit_NMF:
            w_score = my_nmfmodel.fit_transform(m0_score.copy(), W=w_score, H=h_score.T)
            h_score = my_nmfmodel.components_.T
            w_score = np.nan_to_num(w_score)
            h_score = np.nan_to_num(h_score)
        else:
            for _ in range(0, max_iter_mult):
                # Weighted multiplicative rules
                if fast_mult_rules:
                    m0_score_est = (w_score @ h_score.T)*m0_weight_score
                    h_score *= ((w_score.T @ m0_score) / (w_score.T @ m0_score_est + EPSILON)).T
                    w_score *= (m0_score @ h_score / (m0_score_est @ h_score + EPSILON))
                else:
                    h_score *= ((w_score.T @ m0_score) / (w_score.T @ ((w_score @ h_score.T)*m0_weight_score) + EPSILON)).T
                    w_score *= (m0_score @ h_score / ((m0_weight_score*(w_score @ h_score.T)) @ h_score + EPSILON))

        # Normalize w_score by max column and update h_score
        max_values = np.max(w_score, axis=0)
        # Replace maximum values equal to 0 with 1
        w_score = np.divide(w_score, np.where(max_values == 0, 1, max_values))
        h_score = np.multiply(h_score, max_values)

        # Generate embedding tensor and  initialize h_updated
        if i_score == 0:
            tensor_score = w_score
            h_updated = h_score
        else:
            tensor_score = np.hstack((tensor_score, w_score))
            h_updated = np.vstack((h_updated, h_score))

    # Apply NTF with prescribed number of themes and update themes
    my_ntfmodel = NTF(n_components=n_themes, leverage=None, init_type=2, max_iter=200, tol=1e-6, verbose=-1, random_state=0)

    if q_ism is None:
        if update_w_ism:
            estimator_ = my_ntfmodel.fit_transform(tensor_score, n_blocks=n_scores)
        else:
            estimator_ = my_ntfmodel.fit_transform(tensor_score, w=w_ism, update_w=False, n_blocks=n_scores)
    else:
        estimator_ = my_ntfmodel.fit_transform(tensor_score, w=w_ism, h=h_ism, q=q_ism, update_w=update_w_ism, update_h=update_h_ism, n_blocks=n_scores)

    w_ism = estimator_.w
    h_ism = estimator_.h
    q_ism = estimator_.q

    # Update loadings based on h_updated (initialized by multiplicative updates)
    h_updated = h_updated @ h_ism
    h_updated_sparse, hhii_updated = generate_h_sparse(h_updated, q_ism, n_items, n_themes, n_scores, sparsity_coeff=sparsity_coeff)

    return h_updated, h_updated_sparse, hhii_updated, w_ism, h_ism, q_ism, tensor_score

def data_prep(Xs:list[np.array], norm_columns:int = 2):
    Xs_concat = Xs[0].copy()
    for X in Xs[1:]:
        if X.shape[0] != Xs[0].shape[0]:
            sys.exit("All the input array dimensions must match exactly along dimension 0.")
        else:    
            Xs_concat = np.hstack((Xs_concat, X))

    m0 = Xs_concat

    n_items = [Xs[i].shape[1] for i in range(len(Xs))]
    n_scores = len(n_items)

    m0_nan_0 = m0.copy()

    # create m0_weight with ones and zeros if not_missing/missing value
    m0_weight = np.where(np.isnan(m0), 0, 1)
    m0_nan_0[np.isnan(m0_nan_0)]=0

    if norm_columns == 1 or norm_columns == 2:
        if norm_columns == 2:
            # Remove min column values
            min_values = np.min(m0_nan_0, axis=0)
            m0 -= min_values
            m0_nan_0 -= min_values
        # Scale each column of m0
        max_values = np.max(m0_nan_0, axis=0)
        # Replace maximum values equal to 0 with 1
        m0 = np.divide(m0, np.where(max_values == 0, 1, max_values))
        m0_nan_0 = np.divide(m0_nan_0, np.where(max_values == 0, 1, max_values))
        Xs_norm = []
        i1 = 0
        for i_score in range(n_scores):
            i2 = i1+n_items[i_score]
            Xs_norm.append(m0[:,i1:i2])
            i1 = i2
    else:
        Xs_norm = Xs

    return m0, m0_nan_0, m0_weight, n_items, n_scores, Xs_norm

def ism(Xs:list[np.array], n_embedding:int, n_themes:int, norm_columns:int = 2, max_iter:int=200, tol:float=1.e-6, verbose:int=-1, random_state:int=0, 
        max_iter_integrate:int=20, max_iter_mult:int=200, fast_mult_rules:bool=True, w_ism:np.array=None, update_w_ism:bool=True,
        hv_init_mask:list[np.array]=None, update_h_ism:bool=True, sparsity_coeff:float=.8, use_scikit_NMF:bool=False):
    """Estimate ISM model

    Parameters
    ----------
    Xs: List of NDArray
        List of matrices of views.
    n_embedding: integer
        Dimension of the embedding space.
    n_themes: integer
        Dimension of the latent space.
    norm_columns: int, default: 2
        =1: Scale each column of the concatenated matrix
        =2: Substract min and scale each column of the concatenated matrix
    max_iter: integer, default: 200
        Maximum number of iterations.    
    tol: float, default: 1e-6
        Tolerance of the stopping condition.
    verbose: integer, default: 0
        The verbosity level (0/1).
    random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_iter_integrate: integer, default: 20
        Max number of iterations during the straightening process.
    max_iter_mult: integer, default: 200
        Max number of iterations of NMF multiplicative updates during the embedding process.
    fast_mult_rules: boolean, default True
        Use common matrix estimate in w and h updates
    w_ism: NDArray, default: None
        Initialized meta-scores
    hv_init_mask: List of NDArray, default: None
        view-mapping mask (to enforce zero attributes)
    update_w_ism: boolean, default: True
        Update or not initialized meta-scores
    update_h_ism: boolean, default True
        Update or not the NTF factoring matrix H*.
    sparsity_coeff: float, default: .8
        Enhance H sparsity by a multiplicative factor applied to the inverse HHI.
    use_scikit_NMF: boolean, default: False
        Use scikit-learn NMF instead of adnmtf (only if no missing values)
    ism_kwargs: dict
        Additional keyword arguments for ISM

    Returns
    -------
    Dictionary
    ilsm_result['Hv']: View-mapping
    ilsm_result['Hv_sparse']: Sparse view-mapping
    ilsm_result['HHII']: Number of non-negligable values by Hv component
    ilsm_result['W']: ISM meta-scores
    ilsm_result['H']: NTF loadings in latent space
    ilsm_result['Q']: NTF view loadings
    ilsm_result['EMBEDDING']: Embedded views (concatenated)
    ilsm_result['NORMED_VIEWS']: Normed views (concatenated)

    Example
    -------
    >>> import ILSM_functions
    >>> n_embedding, n_themes = [9,10]
    >>> h_updated, h_updated_sparse, w_ism, h_ism, q_ism, tensor_score = ism(m0, n_embedding, n_themes, n_scores, n_items, update_h_ism=True,
                                                                        max_iter_mult=200, sparsity_coeff=.8)

    References
    ----------
    Fogel, P., Boldina, G., Augé, F., Geissler, C., & Luta, G. (2024).
    ISM: A New Space-Learning Model for Heterogenous Multi-view Data Reduction, Visualization and Clustering.
    Preprints. https://doi.org/10.20944/preprints202402.1001.v4
    """
    
    if w_ism is not None and n_embedding != n_themes:
        print("Warning: n_embedding != n_themes: Initial w_ism will be ignored.")
        w_ism = None
        update_w_ism = True

    m0, m0_nan_0, m0_weight, n_items, n_scores, Xs_norm = data_prep(Xs, norm_columns=norm_columns)
    
    if hv_init_mask is not None:
        h_mask = np.ones((m0.shape[1], n_themes))   
        i1 = 0
        for i_score in range(n_scores):
            i2 = i1+n_items[i_score]
            h_mask[i1:i2,:] = hv_init_mask[i_score]
            i1 = i2
    else:
        h_mask = None
    
    if w_ism is None:
        # Initial Embedding
        if use_scikit_NMF:
            print('Using scikit-learn NMF...')
            my_nmfmodel = sklearn.decomposition.NMF(n_components=n_embedding, init='nndsvda', solver='cd', beta_loss='frobenius', max_iter=max_iter, random_state=0)
            w = my_nmfmodel.fit_transform(m0.copy())
            h = my_nmfmodel.components_.T
        else:
            my_nmfmodel = NMF(n_components=n_embedding, leverage=None, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state)
            estimator_ = my_nmfmodel.fit_transform(m0.copy(), h=h_mask)
            w = estimator_.w
            h = estimator_.h

        if h_mask is not None:
            h *= h_mask
    
        h_sparse, hhii = generate_h_sparse(h, None, n_items, n_embedding, n_scores, sparsity_coeff=sparsity_coeff)
    
        # Embed using scores w found in preliminary NMF and initialize themes through NTF       
        h_updated, h_updated_sparse, hhii_updated, w_ism, h_ism, q_ism, tensor_score = \
            integrate_scores(m0_nan_0, m0_weight, h_sparse, w, None, None, n_scores, n_items, n_themes, update_w_ism=update_w_ism, update_h_ism=True,
                            max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff,
                            use_scikit_NMF=use_scikit_NMF)
        
        error = np.linalg.norm(m0 -  w_ism @ h_updated_sparse.T) / np.linalg.norm(m0)
        print('error ism before straightening: ',round(error, 2))
    else:
        q_ism = np.ones((n_scores, n_themes))
        
        if h_mask is None:
            h_updated_sparse = np.ones((m0.shape[1], n_themes)) 
        else:
            h_updated_sparse = h_mask
        
        hhii_updated = np.ones(n_themes)

    # Iterate embedding with themes subtensor until sparsity becomes stable 
    flag = 0

    print('Straightening:')
    for iter_integrate in range(0, max_iter_integrate):
        print('iteration ' + str(iter_integrate) +'...')      
        hhii_updated_0 = hhii_updated.copy()

        if iter_integrate == 0:               
            h_updated, h_updated_sparse, hhii_updated, w_ism, h_ism, q_ism, tensor_score = \
                integrate_scores(m0_nan_0, m0_weight, h_updated_sparse, w_ism, np.identity(n_themes), q_ism, n_scores, n_items, n_themes, update_w_ism=update_w_ism, update_h_ism=update_h_ism,
                                 max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff,
                                 use_scikit_NMF=use_scikit_NMF)      
        else:
            h_updated, h_updated_sparse, hhii_updated, w_ism, h_ism, q_ism, tensor_score = \
                integrate_scores(m0_nan_0, m0_weight, h_updated_sparse, w_ism, h_ism, q_ism, n_scores, n_items, n_themes, update_w_ism=update_w_ism, update_h_ism=update_h_ism,
                                 max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff,
                                 use_scikit_NMF=use_scikit_NMF)    
                
        if (hhii_updated == hhii_updated_0).all():
            flag+=1
        else:
            flag=0
        
        if flag==3:
            break

    Xs_emb = []
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_embedding
        Xs_emb.append(tensor_score[:,i1:i2])
        i1 = i2

    hv = []
    hv_sparse = []
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_items[i_score]
        hv.append(h_updated[i1:i2,:])
        hv_sparse.append(h_updated_sparse[i1:i2,:])
        i1 = i2

    error = np.linalg.norm(m0 -  w_ism @ h_updated_sparse.T) / np.linalg.norm(m0)
    print('error ism after straightening: ',round(error, 2))
    ilsm_result = {}
    ilsm_result['HV'] = hv
    ilsm_result['HV_SPARSE'] = hv_sparse
    ilsm_result['HHII'] = hhii_updated
    ilsm_result['W'] = w_ism
    ilsm_result['H'] = h_ism
    ilsm_result['Q'] = q_ism
    ilsm_result['EMBEDDING'] = Xs_emb
    ilsm_result['NORMED_VIEWS'] = Xs_norm

    return ilsm_result

def ilsm(Xs:list[np.array], n_embedding:int, n_themes:int, norm_columns:int = 2, max_iter:int=200, tol:float=1.e-6, verbose:int=-1, random_state:int=0, 
        max_iter_integrate:int=20, max_iter_mult:int=200, fast_mult_rules:bool=True, update_h_ism:bool=True, sparsity_coeff:float=.8, use_scikit_NMF:bool=False):
    """Estimate ISM model on separate NMF views

    Parameters
    ----------
    Xs: List of NDArray
        List of matrices of views.
    n_embedding: integer
        Dimension of the embedding space.
    n_themes: integer
        Dimension of the latent space.
    norm_columns: int, default: 2
        =1: Scale each column of the concatenated matrix
        =2: Substract min and scale each column of the concatenated matrix
    max_iter: integer, default: 200
        Maximum number of iterations.    
    tol: float, default: 1e-6
        Tolerance of the stopping condition.
    verbose: integer, default: 0
        The verbosity level (0/1).
    random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_iter_mult: integer, default: 200
        Max number of iterations of NMF multiplicative updates during the embedding process.
    fast_mult_rules: boolean, default True
        Use common matrix estimate in w and h updates
    update_h_ism: boolean, default True
        Update or not the NTF factoring matrix H*.
    sparsity_coeff: float, default: .8
        Enhance H sparsity by a multiplicative factor applied to the inverse HHI.
    use_scikit_NMF: boolean, default: False
        Use scikit-learn NMF instead of adnmtf (only if no missing values)
    ism_kwargs: dict
        Additional keyword arguments for ISM

    Returns
    -------
    Dictionary
    ilsm_result['Hv']: View-mapping
    ilsm_result['Hv_sparse']: Sparse view-mapping
    ilsm_result['HHII']: Number of non-negligable values by Hv component
    ilsm_result['W']: ISM meta-scores
    ilsm_result['H']: NTF loadings in latent space
    ilsm_result['Q']: NTF view loadings
    ilsm_result['EMBEDDING']: Embedded views (concatenated)
    ilsm_result['NORMED_VIEWS']: Normed views (concatenated)

    Example
    -------
    >>> import ILSM_functions
    >>> n_embedding, n_themes = [9,10]
    >>> h_updated, h_updated_sparse, w_ism, h_ism, q_ism, tensor_score = ism(m0, n_embedding, n_themes, n_scores, n_items, update_h_ism=True,
                                                                        max_iter_mult=200, sparsity_coeff=.8)

    References
    ----------
    Fogel, P., Boldina, G., Augé, F., Geissler, C., & Luta, G. (2024).
    ISM: A New Space-Learning Model for Heterogenous Multi-view Data Reduction, Visualization and Clustering.
    Preprints. https://doi.org/10.20944/preprints202402.1001.v4
    """

    m0, _, _, _, n_scores, Xs_norm = data_prep(Xs, norm_columns=norm_columns)
   
    # Initial Embedding
    Ws = []
    Hs = []
    if use_scikit_NMF:
        print('Using scikit-learn NMF...')
        my_nmfmodel = sklearn.decomposition.NMF(n_components=n_embedding, init='nndsvda', solver='cd', beta_loss='frobenius', max_iter=max_iter, random_state=0)
        for i in range(n_scores):
            w = my_nmfmodel.fit_transform(Xs[i].copy())
            h = my_nmfmodel.components_.T
            Ws.append(w)
            Hs.append(h)
    else:
        my_nmfmodel = NMF(n_components=n_embedding, leverage=None, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state)
        for i in range(n_scores):
            estimator_ = my_nmfmodel.fit_transform(Xs[i].copy())
            w = estimator_.w
            h = estimator_.h
            Ws.append(w)
            Hs.append(h)
        
    ilsm_result = ism(Ws, n_embedding, n_themes, norm_columns=False, update_h_ism=update_h_ism, max_iter_integrate=max_iter_integrate,
                                        max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff)
    hv = ilsm_result['HV']
    hv_sparse = ilsm_result['HV_SPARSE']
    hhii = ilsm_result['HHII']
    w_ism = ilsm_result['W']
    h_ism = ilsm_result['H']
    q_ism = ilsm_result['Q']
    Xs_emb = ilsm_result['EMBEDDING']
    Xs_norm = ilsm_result['NORMED_VIEWS']

    # Chain-multiplication to retrieve view-mapping to original matrix
    hv_ilsm = []
    hv_sparse_ilsm = []
    hv_ilsm_concat = np.empty((0, n_themes))
    for i in range(n_scores):
        hv_ilsm.append(Hs[i] @ hv[i])
        hv_sparse_ilsm.append(Hs[i] @ hv_sparse[i])
        hv_ilsm_concat = np.vstack((hv_ilsm_concat, hv_ilsm[i]))
    
    error = np.linalg.norm(m0 -  w_ism @ hv_ilsm_concat.T) / np.linalg.norm(m0)
    print('error ilsm: ',round(error, 2))
  
    ilsm_result = {}
    ilsm_result['HV'] = hv_ilsm
    ilsm_result['HV_SPARSE'] = hv_sparse_ilsm
    ilsm_result['HHII'] = hhii
    ilsm_result['W'] = w_ism
    ilsm_result['H'] = h_ism
    ilsm_result['Q'] = q_ism
    ilsm_result['EMBEDDING'] = Xs_emb
    ilsm_result['NORMED_VIEWS'] = Xs_norm

    return ilsm_result 
    
def ism_predict_w(Xs:list[np.array], hv_sparse:list[np.array], h_ism:np.array, q_ism:np.array,
                  n_themes:int, norm_columns:int = 2, max_iter:int=200, tol:float=1.e-6,
                  verbose:int=-1, random_state:int=0, max_iter_mult:int=200):
    """Expand meta-scores to new observations

    Parameters
    ----------
    Xs: List of NDArray
        List of matrices of views.
    hv_sparse: List of NDArray
        List of view-mapping matrix.
    h_ism: NDArray
        Factoring matrix H*.
    q_ism: NDArray
        View loading Q.
    n_themes:
        Dimension of the latent space.
    leverage:  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.
    norm_columns: int, default: 2
        =1: Scale each column of the concatenated matrix
        =2: Substract min and scale each column of the concatenated matrix
    max_iter: integer, default: 200
        Maximum number of iterations.    
    tol: float, default: 1e-6
        Tolerance of the stopping condition.
    verbose: integer, default: 0
        The verbosity level (0/1).
    random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_iter_mult: integer, default: 200
        Max number of iterations of NMF multiplicative updates during the embedding process.
    ntf_kwargs: dict
        Additional keyword arguments for NTF

    Returns
    -------
    Dictionary
    ilsm_result['Hv']: View-mapping
    ilsm_result['Hv_sparse']: Sparse view-mapping
    ilsm_result['HHII']: Number of non-negligable values by Hv component
    ilsm_result['W']: ISM meta-scores
    ilsm_result['H']: NTF loadings in latent space
    ilsm_result['Q']: NTF view loadings
    ilsm_result['EMBEDDING']: Embedded views (concatenated)
    ilsm_result['NORMED_VIEWS']: Normed views (concatenated)

    Example
    -------
    >>> import ILSM_functions
    >>> n_embedding, n_themes = [9,10]
    >>> h_updated_sparse, w_ism, h_ism, q_ism, tensor_score = ism(m0, n_embedding, n_themes, n_scores, n_items, update_h_ism=True,
                                                                        max_iter_mult=200, sparsity_coeff=.8)

    References
    ----------
    Fogel, P., Boldina, G., Augé, F., Geissler, C., & Luta, G. (2024).
    ISM: A New Space-Learning Model for Heterogenous Multi-view Data Reduction, Visualization and Clustering.
    Preprints. https://doi.org/10.20944/preprints202402.1001.v1
    """
    EPSILON = np.finfo(np.float32).eps

    m0, m0_nan_0, m0_weight, n_items, n_scores, Xs_norm = data_prep(Xs, norm_columns=norm_columns)
   
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_items[i_score]
        #  use only rows with non-missing view for the current view
        non_missing_rows = np.where(np.sum(m0_weight[:, i1:i2], axis=1) > 0)[0]
        w_score = np.zeros((m0.shape[0], n_themes))
        w_score_non_missing = np.ones((len(non_missing_rows), n_themes))
        h_score = hv_sparse[i_score].copy()
        m0_score = m0_nan_0[non_missing_rows, i1:i2]
        m0_weight_score = m0_weight[non_missing_rows, i1:i2]
        i1=i2

        if i_score == 0:
            h_updated_sparse_0 = h_score
        else:
            h_updated_sparse_0 = np.vstack((h_updated_sparse_0, h_score))
 
        for _ in range(0, max_iter_mult):
            # Weighted multiplicative rules
            m0_score_est = w_score_non_missing @ h_score.T
            w_score_non_missing *= (m0_score @ h_score / ((m0_weight_score*m0_score_est) @ h_score + EPSILON))

        w_score[non_missing_rows,:] = w_score_non_missing

        # Generate embedding tensor and  initialize h_updated
        if i_score == 0:
            tensor_score = w_score
            h_updated = h_score
        else:
            tensor_score = np.hstack((tensor_score, w_score))
            h_updated = np.vstack((h_updated, h_score))
   
    # Impute rows with missing views
    temp = np.where(tensor_score > 0, 1, 0) / n_themes # will be used to find the number of non-missing views by rows

    # Normalize q_ism by the mean weight of each component across all views    
    q_ism_norm = q_ism / np.mean([q_ism[i_score,:] for i_score in range(n_scores)], axis=0)
    
    for i_score in range(n_scores):
        i1 = i_score*n_themes
        i2 = (i_score+1)*n_themes
        include_mask = np.logical_or(np.arange(tensor_score.shape[1]) < i1, np.arange(tensor_score.shape[1]) >= i2)
        n_scores_non_missing = np.sum(temp[:, include_mask], axis=1)
        missing_rows = np.where(np.sum(tensor_score[:, i1:i2], axis=1) == 0)[0]
        if len(missing_rows) > 0:
            # Estimate missing view using the weighted average other non-missing views, which is an estimate of the meta-score, where the weights are derived from view-loadings.
            for j_score in range(n_scores):
                if j_score != i_score:
                    j1 = j_score*n_themes
                    j2 = (j_score+1)*n_themes
                    tensor_score[missing_rows, i1:i2] += q_ism_norm[j_score, :] * tensor_score[missing_rows, j1:j2]
                    # tensor_score[missing_rows, i1:i2] += q_ism[j_score, :] * tensor_score[missing_rows, j1:j2] / (q_ism[i_score, :] + EPSILON)
 
            tensor_score[missing_rows, i1:i2] /= np.repeat(n_scores_non_missing[missing_rows,np.newaxis], n_themes, axis=1)
            tensor_score[missing_rows, i1:i2] *= np.where(q_ism[i_score, :] > 0, 1, 0)

    # Apply NTF with prescribed number of themes and update themes
    my_ntfmodel = NTF(n_components=n_themes, leverage=None, init_type=2, max_iter=max_iter,
                      tol=tol, verbose=verbose, random_state=random_state)
    estimator_ = my_ntfmodel.fit_transform(tensor_score, h=h_ism, q=q_ism, update_h=False, update_q=False, n_blocks=n_scores)

    w_ism = estimator_.w
    h_ism = estimator_.h
    q_ism = estimator_.q

    # Update loadings based on h_updated (initialized by multiplicative updates)
    h_updated = h_updated @ h_ism
    # sparsity_coeff is set to 0 to only integrate q_ism in view-mapping
    h_updated_sparse, hhii_updated = generate_h_sparse(h_updated, q_ism, n_items, n_themes, n_scores, sparsity_coeff=0)

    # NTF normalizes h_ism and q_ism, affecting the scale of w_ism and h_updated_sparse
    # Rescale to make training and test comparable
    scale = np.linalg.norm(h_updated_sparse_0, axis=0) / np.linalg.norm(h_updated_sparse, axis=0)
    h_updated_sparse *= scale
    w_ism /= scale
    # print(scale)

    Xs_emb = []
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_themes
        Xs_emb.append(tensor_score[:,i1:i2])
        i1 = i2

    hv = []
    hv_sparse = []
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_items[i_score]
        hv.append(h_updated[i1:i2,:])
        hv_sparse.append(h_updated_sparse[i1:i2,:])
        i1 = i2

    error = np.linalg.norm(m0 -  w_ism @ h_updated_sparse.T) / np.linalg.norm(m0)
    print('error ism_predict_w: ',round(error, 2))
    ilsm_result = {}
    ilsm_result['HV'] = hv
    ilsm_result['HV_SPARSE'] = hv_sparse
    ilsm_result['HHII'] = hhii_updated
    ilsm_result['W'] = w_ism
    ilsm_result['H'] = h_ism
    ilsm_result['Q'] = q_ism
    ilsm_result['EMBEDDING'] = Xs_emb
    ilsm_result['NORMED_VIEWS'] = Xs_norm

    return ilsm_result

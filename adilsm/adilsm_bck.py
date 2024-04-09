from adnmtf import NMF, NTF
import pandas as pd
import numpy as np

print("coucou")

def format_loadings(h4, list_columns):
    # Format loadings
    df_h4 = pd.DataFrame(data=h4)
    n_comp = len(df_h4.columns)
    df_h4.columns = ['theme_' + str(i) for i in range(1, n_comp + 1)]
    df_h4.insert(loc=0, column='label', value=(list_columns))

    # Add description index
    df_h4['description'] = df_h4['label']
  
    return df_h4

def generate_h4_sparse(h4, q4_ism, n_items, n_comp, n_scores, sparsity_coeff=.8):
    # Calculate hhii of each h column and generate sparse loadings
    hhii = np.zeros(n_comp, dtype=int)
    h_threshold = np.zeros(n_comp)

    if q4_ism is not None:
        i1 = 0
        for i_score in range(0,n_scores):
            i2 = i1+n_items[i_score]
            h4[i1:i2,:] *= q4_ism[i_score,:]
            i1 = i2

    for i in range(0,n_comp):
        # calculate inverse hhi
        if np.max(h4[:,i]) > 0:
            hhii[i] = int(round(np.sum(h4[:, i])**2 / np.sum(h4[:, i]**2)))
            # hhii[i] = np.count_nonzero(h4[:, i])
        
        # sort the dataframe by score in descending order
        h_threshold[i] = np.sort(h4[:, i], axis=0)[::-1][hhii[i]-1] * sparsity_coeff
        

    h4_sparse = np.where(h4 < h_threshold[None,:], 0, h4)
    
    return h4_sparse, hhii

def integrate_scores(m0_nan_0, m0_weight, h4_sparse, w4_ism, h4_ism, q4_ism, n_scores, n_items, n_themes, update_h4_ism=False,
                     max_iter_mult=200, fast_mult_rules=True, sparsity_coeff=.8):
    
    EPSILON = np.finfo(np.float32).eps

    # Generate w for each score, based on sparse loadings and create tensor_score

    # Extract score-related items
    i1 = 0
    for i_score in range(n_scores):
        i2 = i1+n_items[i_score]
        w4_score = w4_ism.copy()
        h4_score = h4_sparse[i1:i2, :].copy()
        m0_score = m0_nan_0[:, i1:i2]
        m0_weight_score = m0_weight[:, i1:i2]
        i1=i2

        # Apply multiplicative updates to preserve h sparsity   
        for _ in range(0, max_iter_mult):
            # Weighted multiplicative rules
            if fast_mult_rules:
                m0_score_est = (w4_score @ h4_score.T)*m0_weight_score
                h4_score *= ((w4_score.T @ m0_score) / (w4_score.T @ m0_score_est + EPSILON)).T
                w4_score *= (m0_score @ h4_score / (m0_score_est @ h4_score + EPSILON))
            else:
                h4_score *= ((w4_score.T @ m0_score) / (w4_score.T @ ((w4_score @ h4_score.T)*m0_weight_score) + EPSILON)).T
                w4_score *= (m0_score @ h4_score / ((m0_weight_score*(w4_score @ h4_score.T)) @ h4_score + EPSILON))

        # Normalize w4_score by max column and update h4_score
        max_values = np.max(w4_score, axis=0)
        # Replace maximum values equal to 0 with 1
        w4_score = np.divide(w4_score, np.where(max_values == 0, 1, max_values))
        h4_score = np.multiply(h4_score, max_values)

        # Generate embedding tensor and  initialize h4_updated
        if i_score == 0:
            tensor_score = w4_score
            h4_updated = h4_score
        else:
            tensor_score = np.hstack((tensor_score, w4_score))
            h4_updated = np.vstack((h4_updated, h4_score))

    # Apply NTF with prescribed number of themes and update themes
    my_ntfmodel = NTF(n_components=n_themes, leverage=None, init_type=2, max_iter=200, tol=1e-6, verbose=-1, random_state=0)

    if q4_ism is None:
        estimator_ = my_ntfmodel.fit_transform(tensor_score, n_blocks=n_scores)
    else:
        estimator_ = my_ntfmodel.fit_transform(tensor_score, w=w4_ism, h=h4_ism, q=q4_ism, update_h=update_h4_ism, n_blocks=n_scores)

    w4_ism = estimator_.w
    h4_ism = estimator_.h
    q4_ism = estimator_.q

    # Update loadings based on h4_updated (initialized by multiplicative updates)
    h4_updated = h4_updated @ h4_ism
    h4_updated_sparse, hhii_updated = generate_h4_sparse(h4_updated, q4_ism, n_items, n_themes, n_scores, sparsity_coeff=sparsity_coeff)

    return h4_updated, h4_updated_sparse, hhii_updated, w4_ism, h4_ism, q4_ism, tensor_score

def ism(m0:np.array, n_embedding:int, n_themes:int, n_scores:int, n_items:list[int], norm_m0:bool = True, max_iter:int=200, tol:float=1.e-6, verbose:int=-1, random_state:int=0, 
        max_iter_integrate:int=20, max_iter_mult:int=200, fast_mult_rules:bool=True, update_h4_ism:bool=False, sparsity_coeff:float=.8):
    """Estimate ISM model

    Parameters
    ----------
    m0: NDArray
        Matrix of views, concatenated horizontally.
    n_embedding: integer
        Dimension of the embedding space.
    n_themes: integer
        Dimension of the latent space.
    n_scores: integer
        Number of views.
    n_items: integer
        List of numbers of attributes (features) per view
    norm_m0: boolean
        Scale each column of the concatenated matrix
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
    update_h4_ism: boolean, default False
        Update or not the NTF factoring matrix H*.
    sparsity_coeff:
        Enhance H sparsity by a multiplicative factor applied to the inverse HHI.
    ntf_kwargs: dict
        Additional keyword arguments for NTF

    Returns
    -------
    ISM decomposition W, H*, Q and view mapping matrix H

    Example
    -------
    >>> import ILSM_functions
    >>> n_embedding, n_themes = [9,10]
    >>> h4_updated, h4_updated_sparse, w4_ism, h4_ism, q4_ism, tensor_score = ism(m0, n_embedding, n_themes, n_scores, n_items, update_h4_ism=True,
                                                                        max_iter_mult=200, sparsity_coeff=.8)

    References
    ----------
    Fogel, P., Boldina, G., Augé, F., Geissler, C., & Luta, G. (2024).
    ISM: A New Space-Learning Model for Heterogenous Multi-view Data Reduction, Visualization and Clustering.
    Preprints. https://doi.org/10.20944/preprints202402.1001.v1
    """
     
    m0_nan_0 = m0.copy()

    # create m0_weight with ones and zeros if not_missing/missing value
    m0_weight = np.where(np.isnan(m0), 0, 1)
    m0_nan_0[np.isnan(m0_nan_0)]=0

    if norm_m0 is True:
        #Scale each column of m0
        max_values = np.max(m0_nan_0, axis=0)
        # # Replace maximum values equal to 0 with 1
        m0 = np.divide(m0, np.where(max_values == 0, 1, max_values))
        m0_nan_0 = np.divide(m0_nan_0, np.where(max_values == 0, 1, max_values))

    # Initial Embedding
    my_nmfmodel = NMF(n_components=n_embedding, leverage=None, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state)
    estimator_ = my_nmfmodel.fit_transform(m0.copy())

    w4 = estimator_.w
    h4 = estimator_.h

    error = np.linalg.norm(m0 -  w4 @ h4.T) / np.linalg.norm(m0)
    # print('error nmf: ',round(error, 2))
  
    h4_sparse, hhii = generate_h4_sparse(h4, None, n_items, n_embedding, n_scores, sparsity_coeff=sparsity_coeff)

    # Embed using scores w4 found in preliminary NMF and initialize themes through NTF       
    h4_updated, h4_updated_sparse, hhii_updated, w4_ism, h4_ism, q4_ism, tensor_score = \
        integrate_scores(m0_nan_0, m0_weight, h4_sparse, w4, None, None, n_scores, n_items, n_themes, update_h4_ism=True,
                         max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff)
    
    error = np.linalg.norm(m0 -  w4_ism @ h4_updated_sparse.T) / np.linalg.norm(m0)
    print('error ism before straightening: ',round(error, 2))

    # Iterate embedding with themes subtensor until sparsity becomes stable 
    flag = 0

    for iter_integrate in range(0, max_iter_integrate):      
        hhii_updated_0 = hhii_updated.copy()

        if iter_integrate == 0:               
            h4_updated, h4_updated_sparse, hhii_updated, w4_ism, h4_ism, q4_ism, tensor_score = \
                integrate_scores(m0_nan_0, m0_weight, h4_updated_sparse, w4_ism, np.identity(n_themes), q4_ism, n_scores, n_items, n_themes, update_h4_ism=update_h4_ism,
                                 max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff)      
        else:
            h4_updated, h4_updated_sparse, hhii_updated, w4_ism, h4_ism, q4_ism, tensor_score = \
                integrate_scores(m0_nan_0, m0_weight, h4_updated_sparse, w4_ism, h4_ism, q4_ism, n_scores, n_items, n_themes, update_h4_ism=update_h4_ism,
                                 max_iter_mult=max_iter_mult, fast_mult_rules=fast_mult_rules, sparsity_coeff=sparsity_coeff)    
                
        if (hhii_updated == hhii_updated_0).all():
            flag+=1
        else:
            flag=0
        
        if flag==3:
            break

    error = np.linalg.norm(m0 -  w4_ism @ h4_updated_sparse.T) / np.linalg.norm(m0)
    print('error ism after straightening: ',round(error, 2))

    return h4_updated, h4_updated_sparse, hhii_updated, w4_ism, h4_ism, q4_ism, tensor_score, m0


def ism_expand(m0, h4_sparse, h4_ism, q4_ism, n_themes, n_scores, n_items, max_iter=200, tol=1.e-6, verbose=-1, random_state=0, 
       max_iter_mult=200):
    """Expand meta-scores to new observations

    Parameters
    ----------
    m0: float
        Matrix of views, concatenated horizontally.
    h4_sparse: float
        View-mapping matrix H.
    h4_ism: float
        Factoring matrix H*.
    q4_ism: float
        View loading Q.
    n_themes:
        Dimension of the latent space.
    n_scores: integer
        Number of views.
    n_items: integer
        List of numbers of attributes (features) per view
    leverage:  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.
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
    Expanded meta-scores

    Example
    -------
    >>> import ILSM_functions
    >>> n_embedding, n_themes = [9,10]
    >>> h4_updated_sparse, w4_ism, h4_ism, q4_ism, tensor_score = ism(m0, n_embedding, n_themes, n_scores, n_items, update_h4_ism=True,
                                                                        max_iter_mult=200, sparsity_coeff=.8)

    References
    ----------
    Fogel, P., Boldina, G., Augé, F., Geissler, C., & Luta, G. (2024).
    ISM: A New Space-Learning Model for Heterogenous Multi-view Data Reduction, Visualization and Clustering.
    Preprints. https://doi.org/10.20944/preprints202402.1001.v1
    """
    EPSILON = np.finfo(np.float32).eps
    #Scale each column of m0
    m0_nan_0 = m0.copy()

    # create m0_weight with ones and zeros if not_missing/missing value
    m0_weight = np.where(np.isnan(m0), 0, 1)
    m0_nan_0[np.isnan(m0_nan_0)]=0

    max_values = np.max(m0_nan_0, axis=0)
    # Replace maximum values equal to 0 with 1
    m0 = np.divide(m0, np.where(max_values == 0, 1, max_values))
    m0_nan_0 = np.divide(m0_nan_0, np.where(max_values == 0, 1, max_values))
    
    i1 = 0

    for i_score in range(n_scores):
        i2 = i1+n_items[i_score]
        non_missing_rows = np.where(np.sum(m0_weight[:, i1:i2], axis=1) > 0)[0]
        w4_score = np.zeros((m0.shape[0], n_themes))
        w4_score_non_missing = np.ones((len(non_missing_rows), n_themes))
        h4_score = h4_sparse[i1:i2, :].copy()
        m0_score = m0_nan_0[non_missing_rows, i1:i2]
        m0_weight_score = m0_weight[non_missing_rows, i1:i2]
        i1=i2

        # Apply multiplicative updates to preserve h sparsity   
        for _ in range(0, max_iter_mult):
            # Weighted multiplicative rules
            m0_score_est = w4_score_non_missing @ h4_score.T
            w4_score_non_missing *= (m0_score @ h4_score / ((m0_weight_score*m0_score_est) @ h4_score + EPSILON))

        w4_score[non_missing_rows,:] = w4_score_non_missing

        # Generate embedding tensor and  initialize h4_updated
        if i_score == 0:
            tensor_score = w4_score
        else:
            tensor_score = np.hstack((tensor_score, w4_score))

    # Impute rows with missing views
    temp = np.where(tensor_score > 0, 1, 0) / n_themes # will be used to find the number of non-missing views by rows

    # Normalize q4_ism by the mean weight of each component across all views    
    q4_ism_norm = q4_ism / np.mean([q4_ism[i_score,:] for i_score in range(n_scores)], axis=0)
    
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
                    tensor_score[missing_rows, i1:i2] += q4_ism_norm[j_score, :] * tensor_score[missing_rows, j1:j2]
                    # tensor_score[missing_rows, i1:i2] += q4_ism[j_score, :] * tensor_score[missing_rows, j1:j2] / (q4_ism[i_score, :] + EPSILON)
 
            tensor_score[missing_rows, i1:i2] /= np.repeat(n_scores_non_missing[missing_rows,np.newaxis], n_themes, axis=1)
            tensor_score[missing_rows, i1:i2] *= np.where(q4_ism[i_score, :] > 0, 1, 0)

    # Apply NTF with prescribed number of themes and update themes
    my_ntfmodel = NTF(n_components=n_themes, leverage=None, init_type=2, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state)
    estimator_ = my_ntfmodel.fit_transform(tensor_score, h=h4_ism, q=q4_ism, update_h=False, update_q=True, n_blocks=n_scores)
    w4_ism = estimator_.w

    return w4_ism
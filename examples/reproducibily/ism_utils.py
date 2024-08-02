import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def specificity_score(df_w):
    n_themes = (df_w.shape[1]-1)
    k = np.unique(df_w['list_cell_codes']).shape[0]
    means_within = np.zeros(n_themes)
    means_outside = np.zeros(n_themes)
    specificity = np.zeros((k,n_themes))
    group_means = df_w.iloc[:, 0:n_themes+1].groupby('list_cell_codes').mean()
    for i in range(k):
        for j in range(n_themes):
            means_within[j] = group_means.iloc[i,j]
            means_outside[j] = np.mean(np.hstack((group_means.iloc[:i-1,j],group_means.iloc[i+1:,j])))

        specificity[i,:] = (means_within - means_outside) / np.max(means_within)
        # specificity[specificity<0] = 0
    
        means_within_outside = np.hstack((means_within[:,np.newaxis],
                                      means_outside[:,np.newaxis], specificity[i,:][:,np.newaxis]))

        if i==0:
            df_means_within_outside = pd.DataFrame(means_within_outside.copy())
            df_means_within_outside.insert(0, 'cell_type', i)
            df_means_within_outside.insert(0, 'id', range(n_themes))
        else:
            df_means_within_outside_2 = pd.DataFrame(means_within_outside.copy())
            df_means_within_outside_2.insert(0, 'cell_type', i)
            df_means_within_outside_2.insert(0, 'id', range(n_themes))
            df_means_within_outside = pd.concat([df_means_within_outside, df_means_within_outside_2], ignore_index=True)

    df_means_within_outside.columns = ['id', 'cell_type', 'within', 'outside', 'specificity']
    df_means_within_outside['cell_type'] = df_means_within_outside['cell_type'].astype('category')
    df_means_within_outside['id'] = df_means_within_outside['id'].astype('category')
    
    return specificity, df_means_within_outside

def specificity_plot(df_means_within_outside):
    x = df_means_within_outside['outside']
    y = df_means_within_outside['within']
    z = df_means_within_outside['cell_type']
    ids = df_means_within_outside['id']

    # Create subplots for each category of z
    unique_categories = np.unique(z)
    num_categories = len(unique_categories)

    fig, axs = plt.subplots(1, num_categories, figsize=(14, 2), sharex=False, sharey=False)

    for i, category in enumerate(unique_categories):
        mask = (z == category)
        min_val = .95*min(x[mask].min(), y[mask].min())
        min_val = 0
        max_val = 1.05*max(x[mask].max(), y[mask].max())
        axs[i].scatter(x[mask], y[mask], label=f'Category {category}')
        axs[i].plot([min_val, max_val], [min_val, max_val], ls='--', c='r')  # Add the diagonal line (identity line)
        # for ii, row in enumerate(ids[mask]):
        #     axs[i].text(x[ii], y[ii], row, fontsize=8)
        axs[i].set_xlim(min_val, max_val)
        axs[i].set_ylim(min_val, max_val)
        axs[i].set_title(f'Cell type ({category})')
        axs[i].set_xlabel('Outside')
        axs[i].set_ylabel('Within')

    plt.tight_layout()
    plt.show()

import os
import numpy as np
import pandas as pd
from lenskit import topn, crossfold
from lenskit.algorithms.als import ImplicitMF
from lenskit.metrics import ndcg_score, recall_score

# Create working directory
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Load data
data = pd.read_csv('u.data', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])

# Prepare the dataset
ratings = data[['user', 'item', 'rating']]

# Experiment storage
experiment_data = {'MovieLens100K': {'metrics': {'ndcg': [], 'recall': []}, 'predictions': [], 'ground_truth': []}} 

seed_list = [42, 123, 999]

for seed in seed_list:
    # Split data into train and test
    folds = crossfold.Stratified(random_state=seed)
    for train_set, test_set in folds.split(ratings):
        # Train model
        model = ImplicitMF(random_state=seed)
        model.fit(ratings.iloc[train_set])

        # Make predictions
        preds = model.rank_items(test_set['item'].values)
        preds_df = pd.DataFrame({'user': test_set['user'], 'item': preds})
        experiment_data['MovieLens100K']['predictions'].append(preds_df)
        experiment_data['MovieLens100K']['ground_truth'].append(test_set[['user', 'item']])

        # Evaluate metrics
        ndcg_score_value = ndcg_score(preds_df['item'], test_set['item'])
        recall_score_value = recall_score(preds_df['item'], test_set['item'])
        experiment_data['MovieLens100K']['metrics']['ndcg'].append(ndcg_score_value)
        experiment_data['MovieLens100K']['metrics']['recall'].append(recall_score_value)

        print(f'Seed: {seed}, NDCG: {ndcg_score_value:.4f}, Recall: {recall_score_value:.4f}') 

# Save experiment data
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
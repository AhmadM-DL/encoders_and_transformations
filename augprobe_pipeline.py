import numpy as np
import random, os, json
from metrics import top_k_augmentations_recall, augmentations_rank
from encoders import get_features

def probe(encoder, dataset, transformation, n_augmentations=10, sample_size=500, encoder_target_dim=768, random_state=42, chkpt_path="./chkpt", verbose=True):
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Create checkpoint
    if verbose: print("Checking path ...")
    if not os.path.exists(chkpt_path):
        os.mkdir(chkpt_path)
    
    # Take a random subset
    if verbose: print(f"Sampling {sample_size} images ...")
    sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    sample_data = [dataset[i] for i in sample_indices]
    
    # Apply transformations on each image in the sample
    if verbose: print("Applying transformations ...")
    all_images = []
    image_ids = []
    
    for idx, (image, label) in enumerate(sample_data):
        # Original image
        all_images.append(image)
        image_ids.append(idx)
        # Generate augmentations
        augmented_images = transformation([image]*n_augmentations)
        all_images.extend(augmented_images)
        image_ids.extend([idx]*n_augmentations)  # Same ID for original and its augmentations

    # Get the features of each image and augmentations
    if verbose: print("Getting images embeddings ...")
    features = []
    for image in all_images:
        feature = get_features(encoder, image, encoder_target_dim, "cuda")
        features.append(feature)
    features = np.vstack(features).astype('float32')
    
    # Compute metrics
    if verbose: print({"Computing metrics ..."})
    top_k_aug_recall_scores = top_k_augmentations_recall(features, image_ids, n_augmentations, n_augmentations)
    aug_avg_rank_scores, aug_min_rank_scores, aug_max_rank_scores = augmentations_rank(features, image_ids)
    
    # Store the metrics in checkpoint format
    if verbose: print("Saving to chekpoint ...")
    encoder_name = getattr(encoder, '__name__', str(encoder))
    dataset_name = getattr(dataset, 'dataset_name', 'unknown_dataset')
    transformation_name = getattr(transformation, '__name__', str(transformation))
    
    config = {
        'n_augmentations': n_augmentations,
        'sample_size': sample_size,
        'encoder_target_dim': encoder_target_dim,
        'random_state': random_state,
    }
    
    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'transformation': transformation_name,
        'config': config,
        'metrics': {
            'top_k_recall': top_k_aug_recall_scores,
            'average_rank': aug_avg_rank_scores,
            'min_rank': aug_min_rank_scores,
            'max_rank': aug_max_rank_scores
        }
    }

    # Write to checkpoint
    chkpt_file = os.path.join(chkpt_path, "checkpoint.json")
    if os.path.exists(chkpt_file):
        chkpt = json.load(open(chkpt_file, "r"))
    else:
        chkpt = []
    chkpt.append(results)
    json.dump(chkpt, open(chkpt_file, "w"), ensure_ascii=True, indent=4)
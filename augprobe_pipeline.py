import numpy as np
import random, os, json, gc
from metrics import *
from metrics import TOP_K_RECALL_METRIC, RANK_METRIC, RBF_CKA_METRIC, LINEAR_CKA_METRIC
from encoders import get_features, get_encoder
from datasets import get_dataset
from multiprocessing import Process, Queue, shared_memory

def _get_sample(dataset_name, split, processor, random_state, sample_size, shared_mem_name, shape, dtype):
    print("Starting worker ...")
    dataset = get_dataset(dataset_name, split, processor= processor)
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    # Take a random subset
    sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    shm = shared_memory.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    for i, idx in enumerate(sample_indices):
        image, label = dataset[idx]
        image = np.asarray(image.resize((shape[1], shape[2]))) / 255.0
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        shared_array[i] = image  # shape is (sample_size, H, W, C)

    print("Worker finished ...")
    shm.close()
    del dataset
    gc.collect()

def probe(encoder_name, dataset_name, transformation, transformation_name, metrics= [TOP_K_RECALL_METRIC, RANK_METRIC, RBF_CKA_METRIC, LINEAR_CKA_METRIC], image_size= 224, n_augmentations=10, sample_size=500, encoder_target_dim=768, random_state=42, chkpt_path="./chkpt", chkpt_name="checkpoint",  verbose=True):
    
    # Create checkpoint
    if verbose: print("Checking path ...")
    if not os.path.exists(chkpt_path):
        os.mkdir(chkpt_path)

    # Loading encoder
    encoder, processor = get_encoder(encoder_name)

    # Getting a sample - run dataset loading in a seprate process
    if verbose: print(f"Sampling {sample_size} images (launching a worker) ...")
    # Create shared memory
    shape = (sample_size, image_size, image_size, 3)
    dtype = np.float32
    shm = shared_memory.SharedMemory(create=True, size=np.prod(shape) * np.dtype(dtype).itemsize)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    p = Process(
        target=_get_sample,
        args=(dataset_name, "train", None, random_state, sample_size, shm.name, shape, dtype)
    )
    p.start()
    p.join()

    # Read images from shared memory
    images = shared_array
    shm.close()
    shm.unlink()

    # Apply transformations on each image in the sample
    if verbose: print("Applying transformations ...")
    all_images = []
    image_ids = []
    
    for idx in range(sample_size):
        # Original image
        image = images[idx]
        all_images.append(image)
        image_ids.append(idx)
        # Generate augmentations
        augmented_images = transformation([image]*n_augmentations)
        all_images.extend(augmented_images)
        image_ids.extend([idx]*n_augmentations)  # Same ID for original and its augmentations

    if verbose: print("Clearing sample from memory ...")
    del sample_data
    gc.collect()

    # Get the features of each image and augmentations
    if verbose: print("Getting images embeddings ...")
    features = []
    batch_size = 128  
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i+batch_size]
        batch_processed = processor(batch_images, return_tensors='pt')['pixel_values']
        batch_features = get_features(encoder, batch_processed, encoder_target_dim, "cuda")
        batch_features = batch_features.cpu().numpy()
        features.append(batch_features)
    features = np.vstack(features).astype('float32')

    if verbose: print("Clearing images from memory ...")
    del all_images
    gc.collect()

    if verbose: print("Clearing model from memory ...")
    del encoder
    gc.collect()
    
    # Compute metrics
    if verbose: print("Computing metrics ...")
    
    if TOP_K_RECALL_METRIC in metrics:
        top_k_aug_recall_scores = top_k_augmentations_recall(features, image_ids, n_augmentations, n_augmentations)
    else:
        top_k_aug_recall_scores = []

    if RANK_METRIC in metrics:
        aug_avg_rank_scores, aug_min_rank_scores, aug_max_rank_scores = augmentations_rank(features, image_ids)
    else:
        aug_avg_rank_scores= []
        aug_min_rank_scores= []
        aug_max_rank_scores= []

    if RBF_CKA_METRIC in metrics:
        rbf_cka_score = rbf_cka(features, image_ids, n_augmentations)
    else: 
        rbf_cka_score = None
    
    if LINEAR_CKA_METRIC in metrics:
        linear_cka_score = linear_cka(features, image_ids, n_augmentations)
    else:
        linear_cka_score = None

    if verbose: print("Clearing embeddings from memory ...")
    del features
    gc.collect()

    # Store the metrics in checkpoint format
    if verbose: print("Saving to chekpoint ...")
    config = {
        'n_augmentations': n_augmentations,
        'sample_size': sample_size,
        'encoder_target_dim': encoder_target_dim,
        'image_size': image_size,
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
            'max_rank': aug_max_rank_scores,
            'rbf_cka': rbf_cka_score,
            'linear_cka': linear_cka_score
        }
    }

    # Write to checkpoint
    chkpt_file = os.path.join(chkpt_path, f"{chkpt_name}.json")
    if os.path.exists(chkpt_file):
        chkpt = json.load(open(chkpt_file, "r"))
    else:
        chkpt = []
    chkpt.append(results)
    json.dump(chkpt, open(chkpt_file, "w"), ensure_ascii=True, indent=4)
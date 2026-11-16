
import os
import sys
import numpy as np
import json
import joblib
import logging
from os.path import join, dirname
import pickle
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATA_PATH = '/path/to/data'

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val# / (val.sum() + 1e-10)

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.

    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.

    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    # print "Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window)

    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)

    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)),
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata

def downsample_word_vectors(stories, word_vectors, wordseqs):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}
        wordseqs: Dictionary of {story: <object>}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times,
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

def stories_not_in_subj(raw_stories, sub_filename):
	"""Identify stories that are missing from a subject's data directory.

    Args:
        raw_stories: Dictionary of {story: <object>} containing all available stories.
        sub_filename: Name of the subject's data directory.

    Returns:
        Set of story names that are in raw_stories but missing from the subject's directory.
    """
	available_files = set(f.replace('.npy', '') for f in os.listdir(DATA_PATH+'/'+sub_filename))
	missing = set(raw_stories.keys()) - available_files

	return missing

def split_train_test(raw_stories,test_size=0.2,random_state=42):
	"""Split stories into training and test sets, excluding stories missing from any subject.

    Filters out stories that are not available for all subjects before splitting
    to ensure consistency across subjects.

    Args:
        raw_stories: Dictionary of {story: <object>} containing all available stories.

    Returns:
        train_stories: List of story names for training.
        test_stories: List of story names for testing.
    """
	missing_stories = stories_not_in_subj(raw_stories, 'subject2')
	all_stories = set(raw_stories.keys())
	filtered_stories = list(all_stories - missing_stories)

	train_stories, test_stories = train_test_split(filtered_stories, test_size=test_size, random_state=random_state)

	return train_stories, test_stories

def preprocessing_pipeline(list_stories,word_vectors,raw_stories):
    """Downsample, trim, and add delays to word vectors for a list of stories.

    Applies the full preprocessing pipeline to convert word-level embedding
    vectors into TR-level feature matrices ready for use in a linear model.
    Specifically:
        1. Downsamples word vectors from word timestamps to TR timestamps via Lanczos interpolation.
        2. Trims the first 5 and last 10 TRs to account for hemodynamic response delay.
        3. Creates lagged versions of the features to model the hemodynamic response.

    Args:
        raw_stories: Dictionary of {story: <object>} containing timing information.
        word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}
        list_stories: List of story names to process.

    Returns:
        X: Dictionary of {story: <float32>[num_TRs, vector_size * num_delays]}
           containing the final feature matrices ready for model training.
    """
    downsampled = downsample_word_vectors(list_stories, word_vectors, raw_stories)
    # Trimming the first 5 & last 10 seconds
    # each story tr_times == len(rows) in .npy files after trim
    trimmed = {}
    for story, matrix in downsampled.items():
        trimmed[story] = matrix[5:-10, :]
    # 1 through 4 second (inclusive) delays
    # (mimicks brain delay in word processing)
    delays = [1, 2, 3, 4]
    X = {}
    for story, matrix in trimmed.items():
        X[story] = make_delayed(matrix, delays)

    return X

# data_dir = os.environ.get('DATA_DIR', '/path/to/project/data')

# if __name__ == "__main__":
#     raw_stories = pickle.load(open(DATA_PATH+'/raw_text.pkl', 'rb'))

#     # TEST = True

#     all_stories = list(raw_stories.keys())
#     # if TEST:
#     #      all_stories = all_stories[:5]

#     # TEMP, switch out for real!
#     word_vectors = {}
#     for story in all_stories:
#         n_words = len(raw_stories[story].data)
#         word_vectors[story] = np.random.randn(n_words, 50)

#     downsampled = downsample_word_vectors(all_stories, word_vectors, raw_stories)

#     # Trimming the first 5 & last 10 seconds
#     # each story tr_times == len(rows) in .npy files after trim
#     trimmed = {}
#     for story, matrix in downsampled.items():
#          trimmed[story] = matrix[5:-10, :]

#     # 1 through 4 second (inclusive) delays
#     # (mimicks brain delay in word processing)
#     delays = [1, 2, 3, 4]
#     X = {}
#     for story, matrix in trimmed.items():
#          X[story] = make_delayed(matrix, delays)

#     # checking final trimmed/downsampled dimensions
#     for story, mat in X.items():
#          print(f"X shape for {story}:", mat.shape)

#     # export final processed pickle file to relative home directory
#     output_path = os.path.join(os.path.expanduser('~'), 'X_processed.pkl')
#     pickle.dump(X, open(output_path, 'wb'))
#     print(f"Saved to: {output_path}")

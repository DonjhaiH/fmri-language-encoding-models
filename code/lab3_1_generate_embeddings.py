import numpy as np
import io
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.models import KeyedVectors
import gensim.downloader as api
import pickle
import os
import warnings
from datasets import Dataset
from huggingface_hub import HfApi
from preprocessing import *
warnings.filterwarnings(action='ignore')
hfapi = HfApi(token=os.environ["HF_TOKEN"])

TEST = False
DATA_PATH = '/path/to/data'

# Bag-of-Words
def bow(stories, raw_stories, vectorizer=None):
    """Generate Bag-of-Words embedding vectors for a list of stories.

        If no vectorizer is provided, fits a new CountVectorizer on the provided
        stories. If a Vectorizer is provided, uses it directly to transform the
        stories, which allows fitting on training stories only to avoid data
        leakage, while reusing the same vocabulary for test stories.

        Args:
            stories: List of story names to transform.
            raw_stories: Dictionary of {story: <object>} containing the raw text
                         data for each story, where raw_stories[story].data is a
                         list of individual words.
            vectorizer: Optional pre-fit CountVectorizer. If None, a new one is
                        fit on the provided stories. Default is None.

        Returns:
            transf_vectors: Dictionary of {story: <float32>[num_story_words, vocab_size]}
                            containing one sparse embedding vector per word per story.
            vectorizer: The fit CountVectorizer, to be reused for transforming test stories.
        """
    # training
    if vectorizer is None:
        vectorizer = CountVectorizer()
        all_train_words = []
        for story in stories:
            all_train_words.extend(raw_stories[story].data)
        vectorizer.fit(all_train_words)

    transf_vectors = {}
    for story in stories:
        transf_vectors[story] = vectorizer.transform(raw_stories[story].data).toarray().astype(np.float32)

    return transf_vectors, vectorizer

# glove and word2vec
def pretrained_model(text, model):
    """Extract pretrained word vectors for each word in a story.

        For each word in the story, retrieves its vector from the pretrained model.
        Words not found in the model vocabulary are replaced with zero vectors to
        maintain alignment with word timestamps for Lanczos downsampling.

        Args:
            text: DataSequence object containing the story data, where text.data
                  is a list of individual words.
            model: Pretrained word vector model (e.g. Word2Vec, GloVe) with a
                   vocabulary and vector_size attribute.

        Returns:
            vectors: <float32>[num_story_words, vector_size] array containing
                     one embedding vector per word.
        """
    # assuming <class 'ridge_utils.DataSequence.DataSequence'>
    words = text.data
 	# Extract vectors for words that exist in the model vocabulary
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
        else:
            vectors.append(np.zeros(model.vector_size))

    return np.array(vectors, dtype=np.float32)

if __name__ == "__main__":

    print("Loading raw stories") # checkpoint
    raw_stories = pickle.load(open(DATA_PATH+'/raw_text.pkl', 'rb'))
	# there are more stories in the raw file than in subjects
    missing_stories = stories_not_in_subj(raw_stories, 'subject2')
    all_stories = list(raw_stories.keys())
    filtered_stories = list(set(all_stories) - missing_stories)
    if TEST:
        filtered_stories = filtered_stories[:2]
        raw_stories = dict([(k,v) for k,v in raw_stories.items() if k in filtered_stories])

    # split for training and testing
    print("Train and test splits") # checkpoint
    train_stories, test_stories = split_train_test(raw_stories)

    ## Embeddings ##
    # # bag of words embedding
    # print("BOW embeddings") # checkpoint
    # train_vectors_bow, vectorizer = bow(train_stories, raw_stories)
    # test_vectors_bow, _ = bow(test_stories, raw_stories, vectorizer=vectorizer)

    # # downsampling and trimming pipeline
    # train_preproc_bow = preprocessing_pipeline(train_stories,train_vectors_bow,raw_stories)
    # test_preproc_bow = preprocessing_pipeline(test_stories,test_vectors_bow,raw_stories)
    # print(f"BOW done, shape: {list(train_preproc_bow.values())[0].shape}") # checkpoint

    # wor2vec
    print("w2v embeddings") # checkpoint
    w2v_model = api.load("word2vec-google-news-300")

    # create train & test embeddings for word to vec
    # w2v_embeddings = dict([(story,pretrained_model(raw_stories[story], w2v_model)) for story in filtered_stories])
    train_w2v = dict([(story, pretrained_model(raw_stories[story], w2v_model)) for story in train_stories])
    test_w2v  = dict([(story, pretrained_model(raw_stories[story], w2v_model)) for story in test_stories])

    # downsampling and trimming pipeline
    # w2v_preproc = preprocessing_pipeline(filtered_stories,w2v_embeddings,raw_stories)
    train_preproc_w2v = preprocessing_pipeline(train_stories, train_w2v, raw_stories)
    test_preproc_w2v  = preprocessing_pipeline(test_stories,  test_w2v,  raw_stories)
    print(f"""w2v done, train shape: {list(train_preproc_w2v.values())[0].shape}\n
          test shape: {list(test_preproc_w2v.values())[0].shape}""") # checkpoint

    # glove
    print("GloVe embeddings") # checkpoint
    glv_model = api.load("glove-wiki-gigaword-300")

    # create train & test embeddings for word to vec
    # glv_embeddings = dict([(story,pretrained_model(raw_stories[story], glv_model)) for story in filtered_stories])
    train_glv = dict([(story, pretrained_model(raw_stories[story], glv_model)) for story in train_stories])
    test_glv  = dict([(story, pretrained_model(raw_stories[story], glv_model)) for story in test_stories])

    # downsampling and trimming pipeline
    # glv_preproc = preprocessing_pipeline(filtered_stories,glv_embeddings,raw_stories)
    train_preproc_glv = preprocessing_pipeline(train_stories, train_glv, raw_stories)
    test_preproc_glv  = preprocessing_pipeline(test_stories,  test_glv,  raw_stories)
    # print(f"GloVe done, shape: {list(glv_preproc.values())[0].shape}") # checkpoint
    print(f"""GloVe done, train shape: {list(train_preproc_glv.values())[0].shape}\n
          test shape: {list(test_preproc_glv.values())[0].shape}""") # checkpoint

    final_embeddings = {'train_w2v':   train_preproc_w2v,
                        'test_w2v':    test_preproc_w2v,
                        'train_glove': train_preproc_glv,
                        'test_glove':  test_preproc_glv}

    # export final processed pickle files to hugging face
    for k,v in final_embeddings.items():
        # filename = 'testingX_' + k + '.pkl'
        # output_path = os.path.join(os.path.expanduser('~'), filename)
        # pickle.dump(v, open(output_path, 'wb'))
        filename = f'X_{k}.pkl'
        buffer = io.BytesIO()
        pickle.dump(v, buffer)
        buffer.seek(0)

        hfapi.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=filename,
            repo_id="Donjhaih/lab3_stat214_files",
            repo_type="dataset"
        )
        print(f"Uploaded: {filename}")

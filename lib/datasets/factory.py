from datasets.text import ctw1500

def get_imdb(dataset):
    imdb = ctw1500(dataset)
    return imdb

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

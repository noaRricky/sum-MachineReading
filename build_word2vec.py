from gensim.models import Word2Vec
from jieba_token import text_gen

WINDOW_SIZE = 6
EMBEDDING_SIZE = 124


def build_word2vec(file_path: str, size: int, window=5, min_count=5, workers=4) -> Word2Vec:
    """bulld word2vec model

    Arguments:
        file_path {str} -- data file path
        size {int} -- vocabulary size

    Keyword Arguments:
        window {int} -- window size (default: {5})
        min_count {int} -- min count number (default: {1})
        worker {int} -- process to use (default: {4})

    Returns:
        Word2Vec -- word2vec model
    """

    # init word2vec model
    model = Word2Vec(size=size, window=window,
                     min_count=min_count, workers=workers)

    gen = text_gen(file_path)
    for sentence in gen:
        model.train([sentence])

    return model


if __name__ == '__main__':
    file_path = './data/data.json'
    save_path = './data/word2vec.kv'
    model = build_word2vec(file_path, EMBEDDING_SIZE, WINDOW_SIZE)
    model.wv.save(save_path)

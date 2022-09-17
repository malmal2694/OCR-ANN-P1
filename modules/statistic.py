import nltk
from numpy import ndarray


def char_error_rate(model_sent: ndarray, valid_sent: ndarray) -> float:
    """
    Parameters
    ----------
    model_sent (np.ndarray): The sentence model created. model_sent is a vector
    with the length of the sentence.
    valid_sent (np.ndarray): The sentence we want to reach ideally. valid_sent
    is a vector with the length of the sentence.

    Returns
    -------
    Calculated CER; A value between ``0`` and ``1``. Smaller value means smaller error.
    """
    return nltk.edit_distance(model_sent, valid_sent) / float(len(valid_sent))


def word_error_rate(
    model_sent: ndarray, vslid_sent: ndarray, ws_index: int
) -> float:  # sent: list of ids, whitespace index
    """
    Parameters
    ----------
    model_sent (np.ndarray): The sentence model created. model_sent is a vector
    with the length of the sentence.
    valid_sent (np.ndarray): The sentence we want to reach ideally. valid_sent
    is a vector with the length of the sentence.
    ws (int): Index of whitespace (The character that is between words of sentence).

    Returns
    -------
    Calculated WER; A value between ``0`` and ``1``. Smaller value means smaller error.
    """
    model_sent = [str(e) for e in model_sent]
    vslid_sent = [str(e) for e in vslid_sent]
    rec_words = "".join(model_sent).split(str(ws_index))
    ref_words = "".join(vslid_sent).split(str(ws_index))
    return nltk.edit_distance(rec_words, ref_words) / float(len(ref_words))

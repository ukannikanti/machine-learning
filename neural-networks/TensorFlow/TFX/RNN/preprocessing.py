
"""Preprocessing file.

Best Practices when creating pre-processing function.

1. Feature names matter
        The naming of the output features of the preprocessing is important. As you will see in the following TFT implementations,
        we reuse the name of the input feature and append _xf.
        Also, the names of the input nodes of the TensorFlow models need to match the names of the output features from the
        preprocessing_fn function.

2. Consider the data types
        TFT limits the data types of the output features. It exports all preprocessed features as either tf.string, tf.float32,
        or tf.int64 values.
        This is important in case your model can’t consume these data types. S
        ome models from TensorFlow Hub require inputs to be presented as tf.int32 values (e.g., BERT models).
        We can avoid that situation if we cast the inputs to the correct data types inside our models or
        if we convert the data types in the estimator input functions.

3. Preprocessing happens in batches
        When you write preprocessing functions, you might think of it as processing one data row at a time. In fact, 
        TFT performs the operations in batches.
        This is why we will need to reshape the output of the preprocessing_fn() function to a Tensor or SparseTensor 
        when we use it in the context of our Transform component.
"""

import tensorflow as tf
import tensorflow_transform as tft

_FEATURE_KEY = 'text'
_LABEL_KEY = 'label'

_VOCAB_SIZE = 8000
_MAX_LEN = 400


def _transformed_name(key, is_input=False):
    return key + ('_xf_input' if is_input else '_xf')


def _tokenize_review(review):
    """Tokenize the reviews by spliting the reviews.
      Constructing a vocabulary. Map the words to their frequency index in the
      vocabulary.
      Args:
        review: tensors containing the reviews. (batch_size/None, 1)
      Returns:
        Tokenized and padded review tensors. (batch_size/None, _MAX_LEN)
      """
    review_sparse = tf.strings.split(tf.reshape(review, [-1])).to_sparse()
    # tft.apply_vocabulary doesn't reserve 0 for oov words. In order to comply
    # with convention and use mask_zero in keras.embedding layer, set oov value
    # to _VOCAB_SIZE and padding value to -1. Then add 1 to all the tokens.
    review_indices = tft.compute_and_apply_vocabulary(
        review_sparse, default_value=_VOCAB_SIZE, top_k=_VOCAB_SIZE)
    dense = tf.sparse.to_dense(review_indices, default_value=-1)
    # TFX transform expects the transform result to be FixedLenFeature.
    padding_config = [[0, 0], [0, _MAX_LEN]]
    dense = tf.pad(dense, padding_config, 'CONSTANT', -1)
    padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
    padded += 1
    return padded


# TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
      Args:
        inputs: map from feature keys to raw not-yet-transformed features.
      Returns:
        Map from string feature key to transformed feature operations.
      """
    return {
          _transformed_name(_LABEL_KEY):
              inputs[_LABEL_KEY],
          _transformed_name(_FEATURE_KEY, True):
              _tokenize_review(inputs[_FEATURE_KEY])
      }

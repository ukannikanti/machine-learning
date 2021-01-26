"""Preprocessing file.
"""
import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURE_KEYS = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income']

CATEGORICAL_FEATURE_KEYS = ['ocean_proximity']

LABEL_KEY = 'median_house_value'

def transformed_name(key):
    return key + '_xf'


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
      Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
      Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
          in the second dimension.
      Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
      """
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
          tf.sparse.to_dense(
              tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
              default_value),
          axis=1)

def preprocessing_fn(inputs):
    """Preprocessing function. TFX calls this function. 
        For simplicity there's not much preprocessing here.."""

    outputs = {}

    # Scale numerical features
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))
        
    # Generate vocabularies and maps categorical features
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
            x=_fill_in_missing(inputs[key]), num_oov_buckets=1, vocab_filename=key)
        
    # Add label key.    
    outputs[transformed_name(LABEL_KEY)] = _fill_in_missing(inputs[LABEL_KEY])
        
    return outputs

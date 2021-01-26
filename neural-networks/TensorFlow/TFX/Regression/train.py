
"""Training file.
"""
import tensorflow as tf
import tensorflow_transform as tft
import os

def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.
      Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch
      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of label indices.
      """
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
          file_pattern=file_pattern,
          batch_size=batch_size,
          features=transformed_feature_spec,
          reader=_gzip_reader_fn,
          label_key='median_house_value_xf') # median_income_xf => y value.

    return dataset

def _build_keras_model(tf_transform_output, hidden_units, learning_rate):
    """Creates a DNN Keras model.
      Args:
        hidden_units: [int], the layer sizes of the DNN (input layer first).
      Returns:
        A keras Model.
      """
    _FEATURE_KEYS = ['longitude_xf', 'latitude_xf', 'housing_median_age_xf', 'total_rooms_xf', 'median_income_xf',
                        'total_bedrooms_xf', 'population_xf', 'households_xf', 'ocean_proximity_xf']
    
    inputs = [
          tf.keras.layers.Input(shape=(1,), name=f)
          for f in _FEATURE_KEYS
      ]
    
    # shape=(None, 8)
    d = tf.keras.layers.concatenate(inputs)
    d = tf.keras.layers.Dense(64, activation='relu')(d)
    d = tf.keras.layers.Dense(32, activation='relu')(d)
    d = tf.keras.layers.Dense(16, activation='relu')(d)
    outputs = tf.keras.layers.Dense(1, activation='linear')(d)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    
    print(model.summary())

    return model

# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
    """Build the estimator using the high level API.

       Args:
        trainer_fn_args: Holds args used to train the model as name/value pairs.
        schema: Holds the schema of the training examples.

      Returns:
        A dict of the following:

          - estimator: The estimator that will be used for training and eval.
          - train_spec: Spec for training.
          - eval_spec: Spec for eval.
          - eval_input_receiver_fn: Input function for eval.
      """
    # Get the previous step transform output. Which contains transformed data file locations etc.
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64

    tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

    # Create datasets for training and validation.
    train_dataset = _input_fn(trainer_fn_args.train_files, tf_transform_output, TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(trainer_fn_args.eval_files, tf_transform_output, EVAL_BATCH_SIZE)
   
    # Build the model
    model = _build_keras_model(
        tf_transform_output=tf_transform_output,
        hidden_units=64,
        learning_rate=0.002
    )

    log_dir = os.path.join(os.path.dirname(trainer_fn_args.serving_model_dir), 'logs')
    print(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    callbacks = [
        tensorboard_callback
    ]

    # Fit model.
    model.fit(
        train_dataset,
        steps_per_epoch=trainer_fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=trainer_fn_args.eval_steps,
        callbacks=callbacks)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }

    model.save(trainer_fn_args.serving_model_dir, save_format='tf', signatures=signatures)
    

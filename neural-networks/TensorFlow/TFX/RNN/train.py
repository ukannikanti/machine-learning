"""Training file"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
import kerastuner
from tfx.components.tuner.component import TunerFnResult


_DROPOUT_RATE = 0.2
_EMBEDDING_UNITS = 64
_EVAL_BATCH_SIZE = 5
_HIDDEN_UNITS = 64
_LEARNING_RATE = 1e-4
_LSTM_UNITS = 32
_VOCAB_SIZE = 8000
_MAX_LEN = 400
_TRAIN_BATCH_SIZE = 10

def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.
      Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch.
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
          label_key='label_xf')

    return dataset


def _build_keras_model(hparams: kerastuner.HyperParameters) -> keras.Model:
    """Creates a LSTM Keras model for classifying imdb data.
      Reference: https://www.tensorflow.org/tutorials/text/text_classification_rnn
      Returns:
        A Keras Model.
      """
    # The model below is built with Sequential API, please refer to
    # https://www.tensorflow.org/guide/keras/sequential_model
    
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(_VOCAB_SIZE + 2, _EMBEDDING_UNITS),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(_LSTM_UNITS,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(_LSTM_UNITS)),
    tf.keras.layers.Dense(_HIDDEN_UNITS, activation='relu'),
    tf.keras.layers.Dropout(_DROPOUT_RATE),
    tf.keras.layers.Dense(1)])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(hparams.get('learning_rate')),
              metrics=['accuracy', tf.keras.metrics.BinaryAccuracy()])


    print(model.summary())
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        print(feature_spec)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _get_hyperparameters() -> kerastuner.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = kerastuner.HyperParameters()
    # Defines search space.
    hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2], default=1e-2)
    hp.Int('num_layers', 1, 4, default=2)
    return hp


# TFX Tuner will call this function.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.
      Args:
        fn_args: Holds args as name/value pairs.
          - working_dir: working dir for tuning.
          - train_files: List of file paths containing training tf.Example data.
          - eval_files: List of file paths containing eval tf.Example data.
          - train_steps: number of train steps.
          - eval_steps: number of eval steps.
          - schema_path: optional schema of the input data.
          - transform_graph_path: optional transform graph produced by TFT.
      Returns:
        A namedtuple contains the following:
          - tuner: A BaseTuner that will be used for tuning.
          - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                        model , e.g., the training and validation dataset. Required
                        args depend on the above tuner's implementation.
      """
    # RandomSearch is a subclass of kerastuner.Tuner which inherits from
    tuner = kerastuner.RandomSearch(
      _build_keras_model,
      max_trials=3,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=kerastuner.Objective('val_loss', 'min'),
      directory=fn_args.working_dir,
      project_name='imdb')
    
    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
   
    train_dataset = _input_fn(
        fn_args.train_files,
      fn_args.data_accessor,
      transform_graph,
      batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      batch_size=_EVAL_BATCH_SIZE)

    return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
        Args:
        fn_args: Holds args used to train the model as name/value pairs.
      """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    print(fn_args.train_files)

    train_dataset = _input_fn(
          fn_args.train_files, tf_transform_output, batch_size=_TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(
          fn_args.eval_files, tf_transform_output, batch_size=_EVAL_BATCH_SIZE)

    if fn_args.hyperparameters:
        hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is removed
        # from the pipeline. User can also inline the hyperparameters directly in
        # _build_keras_model.
        hparams = _get_hyperparameters()
    print('HyperParameters for training : %s' % hparams.get_config())

    model = _build_keras_model(hparams)

    # In distributed training, it is common to use num_steps instead of num_epochs
    # to control training.
    # Reference: https://stackoverflow.com/questions/45989971/
    # /distributed-training-with-tf-estimator-resulting-in-more-training-steps
    # In this example not using distributed training. TODO: WIll add later.

    model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

    # Model validation.
    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
              tf.TensorSpec(shape=[None],
                            dtype=tf.string,
                            name='examples')),
      }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)



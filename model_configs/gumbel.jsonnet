// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == "1" then false else true);
// model class specific variable
local cnn_kernel_size = 5;
local max_seq_len = 500;
local seq2vec_hidden_size = 300;
local dropout = 0.2;
local concatenate_mention_rep = true;
local deltabox_delta_temp = std.parseJson(std.extVar('deltabox_delta_temp'));
local volume_temp = std.parseJson(std.extVar('volume_temp'));
local gumbel_beta = std.parseJson(std.extVar('gumbel_beta'));
local box_space_dim = std.parseJson(std.extVar('box_space_dim'));
local ff_hidden= std.parseJson(std.extVar('ff_hidden'));
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_activation = std.parseJson(std.extVar('ff_activation'));
// local gumbel_beta = 0.1;
// local box_space_dim = 10;
// local ff_hidden= 5;
// local ff_dropout = 0.2;
// local ff_activation = 'tanh';
// local volume_temp = 0.2;
{
  dataset_reader: {
    type: 'toy-data-1',
    //lazy: true,
    //cache_directory: root_dir + '/.data_cache',
  },

    validation_dataset_reader: {
    type: 'toy-data-1',
    //cache_directory: root_dir + '/.data_cache',
  },

  train_data_path: data_dir + '/toy_data_deep_hierarchy/train.json',
  validation_data_path: data_dir + '/toy_data_deep_hierarchy/dev.json',
  test_data_path: data_dir + '/toy_data_deep_hierarchy/test.json',


  model: {
    type: 'toy-box-model',
    debug_level: 0,
    feedforward: {
      input_dim: 2,
      num_layers: 2,
      hidden_dims: [ff_hidden, box_space_dim * 2],
      activations: [ff_activation, 'linear'],
      dropout: dropout,
    },
    vec2box: {
      type: 'vec2box',
      box_factory: {
        type: 'box_factory',
        name: 'mindelta_from_vector',
        kwargs_dict: { beta: deltabox_delta_temp },
      },
    },
    intersect: { type: 'gumbel', beta: gumbel_beta, approximation_mode: 'clipping' },
    volume: { type: 'bessel-approx', log_scale: true, beta: volume_temp, gumbel_beta: gumbel_beta },
    label_embeddings: {
      type: 'box_embedding',
      embedding_dim: box_space_dim,
      vocab_namespace: 'labels',
      box_factory: {
        type: 'box_factory',
        name: 'mindelta_from_vector',
        kwargs_dict: { beta: deltabox_delta_temp },
      },
     box_initializer: {
        type: 'uniform-box-initializer',
        vocab_namespace: 'labels',
        dimensions: box_space_dim,
        minimum: -1,
        maximum: 1,
        delta_min: 0.5,
        delta_max: 0.7,
        box_type_factory: {
          type: 'box_factory',
          name: 'mindelta_from_vector',
          kwargs_dict: { beta: deltabox_delta_temp },
        },
      },
    },
    dropout: dropout,
    initializer: {
      regexes: [
        ['.*linear_layers.*weight', { type: 'xavier_uniform' }],
        ['.*linear_layers.*bias', { type: 'zero' }],
      ],
    },

  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 16,
    },
    //num_workers: 1,
    pin_memory: false,
  },
  trainer: {
    num_epochs: if test == '1' then 20 else 40,
    grad_norm: 2.0,
    patience: 3,
    validation_metric: '+MAP',
    cuda_device: std.parseInt(cuda_device),
    tensorboard_writer: { should_log_learning_rate: true },
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 0,
      verbose: true,
    },
    optimizer: {
      lr: 0.001,
      type: 'adam',
    },
    checkpointer: {
      num_serialized_models_to_keep: 1,
    },
     epoch_callbacks: ['track_epoch_callback'] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}


// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == "1" then false else true);

// model and data specific variable
// dataset variables
local dataset_name = 'bibtex';
local dataset_metadata = (import '../components/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables

//local gumbel_beta = std.parseJson(std.extVar('gumbel_beta'));
local gumbel_beta = 0.1;
//local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_dropout = 0.0;
local volume_temp = 1;
local box_space_dim = 150;
local ff_hidden= 300;
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation = 'relu';
local reg_weight = 0.001;
local deltabox_delta_temp=20;
//local ff_weight_decay =  std.parseJson(std.extVar('ff_weight_decay'));
local ff_weight_decay = 0;
local gain=if ff_activation == 'tanh' then 5/3 else 1;
{
  evaluate_on_test: true,
  dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  validation_dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),

  model: {
    type: 'toy-box-model',
    debug_level: 0,
    feedforward: {
      input_dim: num_input_features,
      num_layers: 2,
      hidden_dims: [ff_hidden, box_space_dim * 2],
      activations: [ff_activation, 'linear'],
      dropout: [ff_dropout, 0.0],
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
    volume: { type: 'bessel-approx', log_scale: true, beta: volume_temp, gumbel_beta: gumbel_beta    },
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
        minimum: 0,
        maximum: 1,
        delta_min: 0.1,
        delta_max: 0.11,
        //delta_min: 0.5,
        //delta_max: 0.7,
        box_type_factory: {
          type: 'box_factory',
          name: 'mindelta_from_vector',
          kwargs_dict: { beta: deltabox_delta_temp },
        },
      },
    },
    initializer: {
      regexes: [
         [@'.*_feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else {type: 'kaiming_uniform', nonlinearity: 'relu'})],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },

  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 64,
    },
    //num_workers: 1,
    pin_memory: false,
  },
  trainer: {
    num_epochs: if test == '1' then 20 else 40,
    //grad_norm: 10.0,
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
      parameter_groups: [[[@'.*_feedforward._linear_layers.*weight', @'.*linear_layers.*bias'], { weight_decay: ff_weight_decay}],],
 //[[@'_feedforward\._linear_layers\.0\.bias'], {requires_grad: false}]],
      lr: 0.001,
      weight_decay: 0,
      type: 'adamw',
    },
    checkpointer: {
      num_serialized_models_to_keep: 1,
    },
     epoch_callbacks: ['track_epoch_callback'] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}

// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == "1" then false else true);
// model class specific variable
local cnn_kernel_size = 5;
local use_tc_in_training = false;
local use_tc_in_val_test = true;
local position_emb_dim = 10;
local seq2vec_hidden_size = 300;
local dropout = 0.2;
local concatenate_mention_rep = true;

local dataset_name = std.parseJson(std.extVar('dataset_name'));
local dataset_metadata = (import '../model_configs/components/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

local label_space_dim = std.parseJson(std.extVar('label_space_dim'));
local ff_hidden= std.parseJson(std.extVar('ff_hidden'));
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));

// local label_space_dim = 10;
// local ff_hidden= 5;
// local ff_dropout = 0.2;
// local ff_activation = 'tanh';

{
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
    type: 'toy-vector',
    feedforward: {
      input_dim: num_input_features,
      num_layers: ff_linear_layers,
//      hidden_dims: [ff_hidden, label_space_dim],
      hidden_dims: [ff_hidden for i in std.range(0, ff_linear_layers - 2)] + [label_space_dim],
//      activations: [ff_activation, 'linear'],
      activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
      dropout: ff_dropout,
    },
    dropout: dropout,
    initializer: {
      regexes: [
        ['.*linear_layers.*weight', { type: 'xavier_uniform' }],
        ['.*linear_layers.*bias', { type: 'zero' }],
        ['.*weight_ih.*', { type: 'xavier_uniform' }],
        ['.*weight_hh.*', { type: 'orthogonal' }],
        ['.*bias_ih.*', { type: 'zero' }],
        ['.*bias_hh.*', { type: 'lstm_hidden_bias' }],
      ],
    },
    constraint_violation: {
      hierarchy_reader: {
        type: 'networkx-edgelist',
        filepath: data_dir + '/' + dataset_metadata.dir_name + '/' + 'hierarchy.edgelist',
      },
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 16,
    },
    //num_workers: 1,
//    pin_memory: false,
  },
  trainer: {
    num_epochs: if test == '1' then 50 else 30,
    grad_norm: 5.0,
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
//    checkpointer: {
//      num_serialized_models_to_keep: 1,
//    },
    epoch_callbacks: ['track_epoch_callback'] + (if use_wandb then ['log_metrics_to_wandb'] else []),
    
  },
}

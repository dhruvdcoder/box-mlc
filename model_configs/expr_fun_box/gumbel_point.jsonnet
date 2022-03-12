// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == "1" then false else true);
// model class specific variable
local dataset_name = 'expr_fun';
local dataset_metadata = (import 'model_configs/components/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;
//local lr_encoder = std.parseJson(std.extVar('lr_encoder'));
local lr_encoder = 0.1;
//local lr_beta =  std.parseJson(std.extVar('lr_beta'));
local lr_beta =  0.1;
//local lr_boxes =  std.parseJson(std.extVar('lr_boxes'));
local lr_boxes =  0.1;
//local ff_hidden= std.parseJson(std.extVar('ff_hidden'));
local ff_hidden= 100;
//local ff_hidden = 300;
local ff_dropout = 0;
//local ff_dropout = 0.1;
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation='sigmoid';
//local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local ff_linear_layers=3;
//local ff_weight_decay =  std.parseJson(std.extVar('ff_weight_decay'));
local ff_weight_decay=0.000004;
local deltabox_delta_temp=20;
local box_space_dim =std.parseInt(std.toString(std.floor(ff_hidden/2))) ;

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
    type: 'box-point-model',
    debug_level: 0,
    feedforward: {
      input_dim: num_input_features,
      num_layers: ff_linear_layers,
      hidden_dims: [ff_hidden for i in std.range(0,ff_linear_layers-2)] + [box_space_dim],
      activations: ([ff_activation for i in std.range(0,ff_linear_layers-2)] + ['linear']),
      dropout: ([ff_dropout for i in std.range(0,ff_linear_layers-2)] + [0]),
  },
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
        //delta_min:0.1,
        //delta_max: 0.11,
        delta_min: 0.5,
        delta_max: 0.7,
        box_type_factory: {
          type: 'box_factory',
          name: 'mindelta_from_vector',
          kwargs_dict: { beta: deltabox_delta_temp },
        },
      },
    },
    constraint_violation: {
          type: 'constraint-violation',
          hierarchy_reader: {
                'type': 'networkx-edgelist',
                'filepath': data_dir + '/' + dataset_metadata.dir_name + '/hierarchy_tc.edgelist'
            }
    },
    initializer: {
      regexes: [
         [@'.*_feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else {type: 'kaiming_uniform', nonlinearity: 'relu'})],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },

  },
 data_loader: {
    shuffle: true,
    batch_size: 64,
    pin_memory: true,
  },
  trainer: {
    num_epochs: if test == '1' then 5 else 300,
    //grad_norm: 5.0,
    patience: 13,
    validation_metric: '+micro_map',
    cuda_device: std.parseInt(cuda_device),
    tensorboard_writer: { should_log_learning_rate: true },
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 3,
      verbose: true,
    },
    optimizer: {
      parameter_groups: [[[@'.*_feedforward._linear_layers.*weight', @'.*linear_layers.*bias'], { weight_decay: ff_weight_decay, lr:  lr_encoder}], [[@'.*gumbel_beta'], { weight_decay: 0, lr:  lr_beta}]  ],
      lr: lr_boxes,
      weight_decay: 0,
      type: 'yogi',
    },
    checkpointer: {
      num_serialized_models_to_keep: 1,
    },
     epoch_callbacks: ['track_epoch_callback'] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}

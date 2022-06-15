// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

// model and data specific variable
// dataset variables
local batch_size = std.parseJson(std.extVar('batch_size'));

local dataset_name = std.parseJson(std.extVar('dataset_name'));
local dataset_reader = std.parseJson(std.extVar('dataset_reader'));
local dataset_metadata = (import '../model_configs/components/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = 1024;

local scorer = std.parseJson(std.extVar('scorer'));
local binary_nll_loss = std.parseJson(std.extVar('binary_nll_loss'));

// model variables
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
local ff_activation = std.parseJson(std.extVar('ff_activation'));
//local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
//local ff_linear_layers=2;
local ff_weight_decay =  std.parseJson(std.extVar('ff_weight_decay'));
//local ff_weight_decay = 0;

local cnn_kernel_size = 4;

local transformer_model = std.parseJson(std.extVar('transformer_model'));

local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
{
  type: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: dataset_reader,
    tokenizer: {
      type: 'spacy',
    },
    token_indexers: {
      text: {
        type: 'single_id',
      },
    },
    test: if test == '1' then true else false,
  },
  validation_dataset_reader: {
    type: dataset_reader,
    tokenizer: {
      type: 'spacy',
    },
    token_indexers: {
      text: {
        type: 'single_id',
      },
    },
    test: if test == '1' then true else false,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),
  model: {
    type: 'toy-vector-encode',
    debug_level: 0,
    binary_nll_loss: binary_nll_loss,
    encoder_stack: {
      debug_level: 0,
      textfield_embedder: {
        token_embedders: {
          text: {
            type: 'embedding',
            embedding_dim: 300,
            pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz',
            trainable: true,
          },
        },
      },
      seq2seq_encoder: {
        type: 'lstm',
        input_size: 300,
        hidden_size: 512,
        num_layers: 3,
        bidirectional: true,
      },
      seq2vec_encoder: {
        type: 'mean',
        embedding_dim: 1024,
      },
    },
    feedforward: {
      input_dim: num_input_features,
      num_layers: ff_linear_layers,
      hidden_dims: [ff_hidden for i in std.range(0, ff_linear_layers - 2)] + [label_space_dim],
      activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
    },
    scorer: {
      type: scorer
    },
    add_new_metrics: false,
    initializer: {
      regexes: [
        [@'.*_feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform'})],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
        type: 'bucket',
        batch_size: if test == '1' then 1 else batch_size, # TODO: might need to set batch size to 2 for BERT
        sorting_keys: ['text'],
    },
    max_instances_in_memory: if test == '1' then 8 else 1000,
    start_method: if cuda_device == '-1' then 'spawn' else 'fork',
    num_workers: 4 # and request proportional number of cpus from gypsum
  },
  trainer: {
    num_epochs: if test == '1' then 20 else 200,
    patience: 8,
    validation_metric: '+MAP',
    cuda_device: std.parseInt(cuda_device),
    optimizer: {
      lr: 0.0001,
      weight_decay: ff_weight_decay,
      type: 'adamw',
    },
    checkpointer: {
      type: 'default',
      keep_most_recent_by_count: 1,
    },
    callbacks: [
      "track_epoch_callback",
      {
        type: 'wandb_allennlp',
        sub_callbacks: [
          {
            type: 'log_best_validation_metrics',
            priority: 100,
          },
        ],
        watch_model: false,
        save_model_archive: false,
        # DEBUG
        should_log_parameter_statistics: false,
//        distribution_interval: 100,
      },
    ]
  },
}
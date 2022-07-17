// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.parseJson(std.extVar('DATA_DIR'));
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);
// model class specific variable
local dataset_name = std.parseJson(std.extVar('dataset_name'));
local dataset_reader = std.parseJson(std.extVar('dataset_reader'));
local dataset_metadata = (import '../model_configs/components/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

local patience = std.parseJson(std.extVar('patience'));
local batch_size = std.parseJson(std.extVar('batch_size'));
local gumbel_beta = std.parseJson(std.extVar('gumbel_beta'));
//local gumbel_beta = 0.00001;
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
//local ff_hidden = 100;
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
//local ff_dropout = 0;
local ff_activation = std.parseJson(std.extVar('ff_activation'));
//local ff_activation = 'sigmoid';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
//local ff_linear_layers = 1;
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
//local ff_weight_decay = 0.1;
//local ff_linear_layers=2;
local lr_encoder = std.parseJson(std.extVar('lr_encoder'));
//local lr_encoder = 0.001;
//local lr_ratio = std.parseJson(std.extVar('lr_ratio'));
//local lr_ratio = 1;
//local lr_boxes = lr_ratio * lr_encoder;
local lr_boxes = std.parseJson(std.extVar('lr_boxes'));
//local ff_weight_decay = 0;
//local volume_temp = 1;
local volume_temp = std.parseJson(std.extVar('volume_temp'));
//local volume_temp = 0.1;
//local step_up = std.parseJson(std.extVar('step_up'));
//local box_space_dim = std.parseInt(std.toString(std.floor(ff_hidden / step_up)));
local box_space_dim = ff_hidden;
//local delta = std.parseJson(std.extVar('delta'));
local delta = 1e-8;
local label_delta_init = std.parseJson(std.extVar('label_delta_init'));
//local label_delta_init = 0.1;
local box_weight_decay = std.parseJson(std.extVar('box_weight_decay'));
local distance_weight = std.parseJson(std.extVar('distance_weight'));
local exponential_scaling = std.parseJson(std.extVar('exponential_scaling'));
//local box_weight_decay = 0;
//local box_weight_decay = 0;
//local final_activation = if std.parseJson(std.extVar('ff_same_final_activation')) then ff_activation else 'linear';
local final_activation = ff_activation;

local transformer_model = std.parseJson(std.extVar('transformer_model'));

local cnn_kernel_size = 64;
local position_emb_size = 64;
local seq2vec_hidden_size = 1024;
local dropout = 0;
local concatenate_mention_rep = false;
local add_position_features = false;
local init_weight = std.parseJson(std.extVar('init_weight'));
local init_bias = std.parseJson(std.extVar('init_bias'));
local init_embed_weight = std.parseJson(std.extVar('init_embed_weight'));
local init_embed_delta = std.parseJson(std.extVar('init_embed_delta'));

local gain = if ff_activation == 'tanh' then 5 / 3 else 1;
{
  type: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: dataset_reader,
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: transformer_model,
      max_length: 512,
    },
    token_indexers: {
      text: {
        type: 'pretrained_transformer',
        model_name: transformer_model,
      },
    },
    test: if test == '1' then true else false,
  },
  validation_dataset_reader: {
    type: dataset_reader,
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: transformer_model,
      max_length: 512,
    },
//    tokenizer: {
//      type: 'spacy',
//    },
    token_indexers: {
      text: {
        type: 'pretrained_transformer',
        model_name: transformer_model,
      },
//        type: 'single_id',
//        token_min_padding_length: cnn_kernel_size,
//      },
//      [if add_position_features then 'positions']: {
//        type: 'single_id',
//        namespace: 'positions',
//        feature_name: 'position',
//        default_value: '0',
//        token_min_padding_length: cnn_kernel_size,
//      },
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
    type: 'multi-instance-typing-box-model',
    debug_level: 0,
    add_new_metrics: false,
    encoder_stack: {
      debug_level: 0,
      textfield_embedder: {
          token_embedders: {
            text: {
              type: "pretrained_transformer",
              model_name: transformer_model,
            }
          }
      },
      seq2vec_encoder: { // Aggregates token embeddings to one embedding
        type: 'bert_pooler',
        pretrained_model: transformer_model,
      },
    },
    feedforward: {
      input_dim: (seq2vec_hidden_size),
      num_layers: ff_linear_layers,
      hidden_dims: ([ff_hidden for i in std.range(0, ff_linear_layers - 2)] + [box_space_dim]),
      activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + ['linear']), // TODO: maybe change to relu
      dropout: ([dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
    },
    vec2box: {
      type: 'vec2box',
      box_factory: {
        type: 'box_factory',
        kwargs_dict: {
          delta: 1e-5,
        },
        name: 'center_fixed_delta_from_vector',
      },
    },
    intersect: { type: 'gumbel', beta: gumbel_beta, approximation_mode: null },
    volume: { type: 'bessel-approx', log_scale: true, beta: volume_temp, gumbel_beta: gumbel_beta },
    label_embeddings: {
      type: 'box-embedding-module',
      embedding_dim: box_space_dim,
    },
    initializer: {
      regexes: [
        [@'.*linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
        ['_label_embeddings.center.weight', { type: 'uniform' }, ],
        [ '_label_embeddings.delta', { type: 'constant', val: 0.01 }],
//        [@'.*linear_layers.*weight', init_weight],
//        [@'.*linear_layers.*bias', init_bias],
//        ['_label_embeddings.center.weight', init_embed_weight ],
//        [ '_label_embeddings.delta', init_embed_delta ],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
        type: 'bucket',
        batch_size: if test == '1' then 2 else batch_size, # TODO: might need to set batch size to 2 for BERT
        sorting_keys: ['text'],
    },
    max_instances_in_memory: if test == '1' then 16 else 1000,
    start_method: if cuda_device == '-1' then 'spawn' else 'fork',
    num_workers: 4 # and request proportional number of cpus from gypsum
  },
  trainer: {
    num_epochs: if test == '1' then 20 else 200,
    grad_norm: 10.0,
    patience: patience,
    validation_metric: '+MAP',
    num_gradient_accumulation_steps: 16,
    // TODO: num_gradient_accumulation_steps might need to set to 16 or something if memory constraints
    cuda_device: std.parseInt(cuda_device),
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 0,
      verbose: true,
    },
    optimizer: {
      type: 'adamw',
      lr: lr_boxes,  //lr_box
      weight_decay: 0,
      parameter_groups: [
        [
          [
            '.*linear_layers.*weight',
            '.*linear_layers.*bias',
          ],
          {
            lr: lr_encoder,
            weight_decay: ff_weight_decay,
          },
        ],
        [
          [
            '.*label_embeddings.delta',
          ],
          {
            lr: lr_boxes,
            weight_decay: box_weight_decay,
          },
        ],
      ],
    },
    callbacks: [
      {
        type: 'wandb_allennlp',
        sub_callbacks: [
          {
            type: 'log_best_validation_metrics',
            priority: 100,
          },
        ],
        watch_model: false,
        should_log_parameter_statistics: false,
        save_model_archive: false,

      },
    ],
    checkpointer: {
      type: 'default',
      keep_most_recent_by_count: 1,
    },
  },
}

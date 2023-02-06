import sys
import os

if len(sys.argv) < 2:
    raise ValueError('Invalid params', sys.argv)

# name = sys.argv[-1]

for name in sys.argv[1:]:
    dims, dist = name.split('_')
    if dist in ['softmax', 'sparsemax']:
        cmd = f"""/work/asempruch_umass_edu/boxem/bin/allennlp \
                train-with-wandb \
                model_configs2/general_mbm.jsonnet \
                -s ../saved_models/{name} \
                -f \
                --include-package \
                box_mlc \
                --include-package box_embeddings \
                --env.dataset_name="expr_fun" \
                --env.box_weight_decay=0 \
                --env.ff_dropout=0.1 \
                --env.ff_hidden={dims} \
                --env.ff_linear_layers=2 \
                --env.ff_weight_decay=0.0001 \
                --env.gumbel_beta=0.001 \
                --env.label_delta_init=0.1 \
                --env.lr_encoder=0.0001 \
                --env.volume_temp=0.5 \
                --env.ff_activation="tanh" \
                --env.label_space_dim=5 \
                --env.DATA_DIR=/gypsum/scratch1/dhruveshpate/multilabel_classification/multilabel-learning/.data \
                --env.ROOT_DIR=/work/asempruch_umass_edu/boxem \
                --env.TEST=0 \
                --env.CUDA_DEVICE=0 \
                --env.distance_weight=0.01 \
                --env.patience=16 \
                --env.lr_boxes=0.001 \
                --env.dimension_dist=true \
                --env.dimension_dist_regularization="{dist}" \
                --env.dimension_dist_init="uniform" \
            """

    elif dist == 'gumbel':
        cmd = f"""/work/asempruch_umass_edu/boxem/bin/allennlp \
                train-with-wandb \
                model_configs2/general_mbm.jsonnet \
                -s ../saved_models/{name} \
                -f \
                --include-package \
                box_mlc \
                --include-package box_embeddings \
                --env.dataset_name="expr_fun" \
                --env.box_weight_decay=0 \
                --env.ff_dropout=0.1 \
                --env.ff_hidden={dims} \
                --env.ff_linear_layers=2 \
                --env.ff_weight_decay=0.0001 \
                --env.gumbel_beta=0.001 \
                --env.label_delta_init=0.1 \
                --env.lr_encoder=0.0001 \
                --env.volume_temp=0.5 \
                --env.ff_activation="tanh" \
                --env.label_space_dim=5 \
                --env.DATA_DIR=/gypsum/scratch1/dhruveshpate/multilabel_classification/multilabel-learning/.data \
                --env.ROOT_DIR=/work/asempruch_umass_edu/boxem \
                --env.TEST=0 \
                --env.CUDA_DEVICE=0 \
                --env.distance_weight=0.01 \
                --env.patience=16 \
                --env.lr_boxes=0.001 \
                --env.dimension_dist=false \
                --env.dimension_dist_regularization="NA" \
                --env.dimension_dist_init="NA" \
            """

    elif dist.startswith('q2b'):
        use_dims = dist[3:]

        cmd = f"""/work/asempruch_umass_edu/boxem/bin/allennlp \
                    train-with-wandb \
                    model_configs2/general_mbm_q2b.jsonnet \
                    -s ../saved_models/{name} \
                    -f \
                    --include-package box_mlc \
                    --include-package box_embeddings \
                    --env.dataset_name="expr_fun" \
                    --env.box_weight_decay=0 \
                    --env.ff_dropout=0.1 \
                    --env.ff_hidden={dims} \
                    --env.ff_linear_layers=2 \
                    --env.ff_weight_decay=0.0001 \
                    --env.gumbel_beta=0.001 \
                    --env.label_delta_init=0.1 \
                    --env.lr_encoder=0.0001 \
                    --env.volume_temp=0.5 \
                    --env.ff_activation="tanh" \
                    --env.label_space_dim=5 \
                    --env.DATA_DIR=/gypsum/scratch1/dhruveshpate/multilabel_classification/multilabel-learning/.data \
                    --env.ROOT_DIR=/work/asempruch_umass_edu/boxem \
                    --env.TEST=0 \
                    --env.CUDA_DEVICE=0 \
                    --env.distance_weight=0.01 \
                    --env.patience=16 \
                    --env.lr_boxes=0.001 \
                    --env.alpha=1.0 \
                    --env.gamma=1.0 \
                    --env.distance_type="l1" \
                    --env.num_distance_dims={use_dims}
                """

    os.system(cmd)
def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    
    config['mode'] = "train"
    config['use_wandb'] = False
    config['use_cuda'] = True
    config['log_dir'] = ""
    config['model_load_dir'] = ""
    config['best_model_dir'] = ""

    # Hyperparameters for dataset. 
    config['view'] = 'all' # all/plax/psax
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'all'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    # number of AS classes
    config['num_classes'] = 4


    #Hyperaparameters for tabular dataset.
    config['use_tab'] = True
    config['scale_feats'] = True
    config['num_ex'] = None
    config['drop_cols'] = []
    config['categorical_cols'] = []
    
    
    # Hyperparameters for Contrastive Learning
    config['feature_dim'] = 1024
    config['temp'] = 0.1

    # Hyperparameters for models.
    config['model'] = "FTC_TAD" # r2plus1d_18/x3d/resnet50/slowfast/tvn/FTC
    config['pretrained'] = False
    config['restore'] = True
    config['loss_type'] = 'cross_entropy' # cross_entropy/evidential/laplace_cdf/SupCon/SimCLR
    config['abstention'] = False
    config["coteaching"] = True
    config['multimodal'] = "fttrans" # clip/mlp/fttrans
    config['latent_dim'] = 1024
    config['tab_input_dim'] = 1
    config['tab_emb_dims'] = [16, 32, 72]
    config['ds_max_length'] = 128
    config['num_hidden_layers'] = 16       # number of transformer layers
    config['intermediate_size'] = 8192     # MLP size inside transformers
    config['attention_heads'] = 16

    # Hyperparameters for training.
    config['batch_size'] = 16
    config['num_epochs'] = 100 #110
    config['lr'] = 1e-4 
    config['sampler'] = 'AS' # imbalanced sampling based on AS/bicuspid/random
 
    return config

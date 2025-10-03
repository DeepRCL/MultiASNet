from FTC.FTC_TAD import get_model_tad
    
def get_model(config):
    if config['model'] == "FTC_TAD":
        latent_dim = config['latent_dim']              # was 1024
        tab_input_dim = config['tab_input_dim']        # was 1
        tab_emb_dims = config['tab_emb_dims']          # was [16, 32, 72]
        ds_max_length = config['ds_max_length']        # was 128
        num_hidden_layers = config['num_hidden_layers']  # was 16
        intermediate_size = config['intermediate_size']  # was 8192
        attention_heads = config['attention_heads']

        rm_branch = None            # select branch to not train: None, 'SD', 'EF'
        use_conv = False            # use convolutions instead of MLP for the regressors - worse results
        model = get_model_tad(latent_dim, 
                            tab_input_dim=tab_input_dim, 
                            tab_emb_dims=tab_emb_dims, 
                            img_per_video=ds_max_length, 
                            num_hidden_layers=num_hidden_layers, 
                            intermediate_size=intermediate_size, 
                            rm_branch=rm_branch, use_conv=use_conv,
                            attention_heads=attention_heads, use_tab=config['use_tab'],
                            multimodal = config["multimodal"])

    return model


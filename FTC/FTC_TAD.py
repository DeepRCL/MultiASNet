from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
import torch
import torch.nn as nn
import torchvision
from FTC.posembedding import build_position_encoding
from FTC.deformable_transformer import DeformableTransformer
from FTC.util.misc import NestedTensor, construct_ASTransformer
from einops import rearrange

class CrossAttention(nn.Module):
    """
    Apply cross attention between x tensor and tab_x tensor.

    Arguments:
        dim: 
        context_dim:
        heads: number of attention heads.
        dim_head: dimension of q, k, v 
    


    """
    def __init__(
        self,
        dim=1024, #dimension of img input
        context_dim=1024, #dimension of tab input
        heads = 12,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.tab_norm = nn.LayerNorm(context_dim)
        self.vid_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)
        self.norm_q = nn.LayerNorm(inner_dim)
        self.norm_k = nn.LayerNorm(inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Storage for top 5 indices across forward calls
        self.most_attended_list = []
        self.least_attended_list = []

    def forward(self, x, tab_x):
        #get shape of video input
        b, f, _ = x.shape
        h = self.heads
        kv_input = self.tab_norm(tab_x)
        q = self.norm_q(self.to_q(self.vid_norm(x)))
        k = self.norm_k(self.to_k(kv_input))
        v = self.to_v(kv_input)
        q, k, v = map(lambda t: rearrange(tensor=t, pattern='b f (h d) -> b h f d', h=h), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = sim.softmax(dim = -1) 
        attn = self.dropout(attn)

        # aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # merge heads
        out = rearrange(out, 'b h f d -> b f (h d)')

        # Get importance of each tab_x token
        attn_mean = attn.mean(dim=(1, 2))  # Average over heads & frames: (b, tokens)
        # Get top 5 most and least attended indices
        top5_most = torch.topk(attn_mean, k=5, dim=-1).indices
        top5_least = torch.topk(attn_mean, k=5, dim=-1, largest=False).indices

        # Save to lists (convert to CPU to avoid memory issues)
        self.most_attended_list.append(top5_most.detach().cpu())
        self.least_attended_list.append(top5_least.detach().cpu())
        
        return self.to_out(out)

class EmbeddingMappingFunction(nn.Module):
    def __init__(self, video_dim, hidden_dim, tabular_dim, num_heads=8, num_transformer_layers=2):
        super(EmbeddingMappingFunction, self).__init__()
        self.fc1 = nn.Linear(video_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.layernorm = nn.LayerNorm(hidden_dim)


    def forward(self, x):
        residual = x  # Preserve the input as the residual

        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layernorm(x)

        # Add the residual connection
        x += residual

        return x
    
class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        
        x = x.mean(dim=1)
        return x
        
# New Multi Branch Auto Encoding Transformer
class FTC(nn.Module):
    def __init__(self, 
                AE, 
                pos_embed, 
                transformer, 
                embedding_dim, 
                use_tab, 
                cross_attention, 
                tab_input_dim, 
                tab_emb_dims,
                loaded_parameters, 
                pretrained = False,
                multimodal = "fttrans",
                bicuspid = True,
                age = True):
        super(FTC, self).__init__()
        
        self.pretrained = pretrained
        self.use_tab = use_tab
        self.tab_input_dim = tab_input_dim
        self.tab_emb_dims = tab_emb_dims
        
        self.AE = AE
        self.pos_embed = pos_embed
        self.transformer = transformer

        # Get the parameters for the tabular model
        loaded_categories = loaded_parameters['categories']
        loaded_num_continuous = loaded_parameters['num_continuous']
        loaded_dim = loaded_parameters['dim']
        loaded_depth = loaded_parameters['depth']
        loaded_heads = loaded_parameters['heads']
        loaded_dim_head = loaded_parameters['dim_head']
        loaded_dim_out = loaded_parameters['dim_out']
        loaded_num_special_tokens = loaded_parameters['num_special_tokens']
        loaded_attn_dropout = loaded_parameters['attn_dropout']
        loaded_ff_dropout = loaded_parameters['ff_dropout']
        loaded_hidden_dim = loaded_parameters['hidden_dim']
        loaded_numerical_features = loaded_parameters['numerical_features']
        loaded_classification = loaded_parameters['classification']
        loaded_emb_type = loaded_parameters['emb_type']
        loaded_numerical_bins = loaded_parameters['numerical_bins']

        self.aorticstenosispred = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=4, bias=True),
            # nn.Softmax(dim=2),
            # Reduce(),
            )
        
        self.attentionweights = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=1, bias=True),
            )

        self.em = embedding_dim

        # load the pretrained weights TODO: Make this a config
        # self.load_state_dict(torch.load("best_model.pth")["model"], strict=False)

        self.cross_attention = cross_attention

        self.multimodal = multimodal
        if multimodal == "fttrans":
            print(loaded_numerical_bins)
            self.tab_embed = construct_ASTransformer(loaded_categories, loaded_num_continuous, loaded_numerical_features,
                                                    loaded_dim, loaded_depth, loaded_heads, loaded_dim_head, loaded_dim_out,
                                                    loaded_num_special_tokens, loaded_attn_dropout, loaded_ff_dropout,
                                                    loaded_hidden_dim, loaded_classification,
                                                    loaded_numerical_bins, loaded_emb_type)
            
            self.tab_embed.load_state_dict(torch.load('../as_transformer.pth'))
            
            # Set requires_grad to False for all parameters
            for param in self.tab_embed.parameters():
                param.requires_grad = False
        elif multimodal == "mlp":
            self.tab_embed = nn.Sequential(
                nn.Linear(36, self.tab_emb_dims[1]*self.tab_emb_dims[2]),
                nn.ReLU(),
                nn.Linear(self.tab_emb_dims[1]*self.tab_emb_dims[2], self.tab_emb_dims[1]*self.tab_emb_dims[2])
            )
        elif multimodal == "clip":
            self.tab_embed = nn.Sequential(
                nn.Linear(36, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 32*embedding_dim)) #TODO

        # Map video embeddings to video+tab embeddings
        self.map_embed = EmbeddingMappingFunction(embedding_dim, embedding_dim, embedding_dim)
        self.vt_proj_head = EmbeddingMappingFunction(embedding_dim, embedding_dim, embedding_dim)

    def forward(self, x, tab_x, bicuspid, age, split):   
        # Video dimension (B x F x C x H x W)
        x = x.permute(0,2,1,3,4)
        
        nB, nF, nC, nH, nW = x.shape

        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        if self.pretrained:
            # with torch.no_grad():
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
        else:
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
            
            
        embeddings_reshaped = embeddings.view(nB, nF, self.em )
        embeddings_reshaped = embeddings_reshaped.permute(0,2,1)
            
        # Creatining porsitional embedding and nested tensor
        nF = 32
        mask = torch.ones((nB, nF), dtype=torch.bool).cuda()
        samples = NestedTensor(embeddings_reshaped, mask)
        pos = self.pos_embed(samples).cuda()
        
        outputs = self.transformer([embeddings_reshaped],[pos],[mask]) #, prompt=output_tensor)

        #integrate tab data
        if split=='Train':
            if self.use_tab:

                if self.multimodal == "fttrans":
                    tab_x = torch.unsqueeze(tab_x, dim=-1) 
                    
                    # B x F x 1
                    #Tranform x feature values to higher dim using a shared mlp layer
                    _, tab_x = self.tab_embed(tab_x)
                    #cross attention between video and tabular embeddings
                    ca_outputs = self.cross_attention(outputs, tab_x)
                elif self.multimodal == "mlp":
                    b, _ = tab_x.shape

                    #Tranform x feature values to higher dim using a shared mlp layer
                    tab_x = self.tab_embed(tab_x).reshape(b, self.tab_emb_dims[1], self.tab_emb_dims[2])
                    #cross attention between video and tabular embeddings
                    ca_outputs = self.cross_attention(outputs, tab_x)
                elif self.multimodal == "clip":
                    b, _ = tab_x.shape
                    ca_outputs = self.tab_embed(tab_x).reshape(b, outputs.shape[1], outputs.shape[2])

                ca_outputs = self.vt_proj_head(ca_outputs)

                # we want to get ca_outputs from outputs using our embedding mapping function
                learned_joint_emb = self.map_embed(outputs)
        else:
            print("Cross-attention module has been skipped.")
            outputs = self.map_embed(outputs)
            learned_joint_emb = outputs
            ca_outputs = outputs

        # B x F x Emb => B x T x 4
        as_prediction = self.aorticstenosispred(learned_joint_emb)

        # cross attention preds
        if split=='Train':
            if self.use_tab:
                #ca_outputs = torch.cat((bv_output_tensor, ca_outputs), dim=1)
                #ca_outputs = torch.cat((age_output_tensor, ca_outputs), dim=1)
                as_ca_predictions = self.aorticstenosispred(ca_outputs)
        else:
            #ca_outputs = torch.cat((bv_output_tensor, ca_outputs), dim=1)
            #ca_outputs = torch.cat((age_output_tensor, ca_outputs), dim=1)
            as_ca_predictions = as_prediction
        
        # attention weights B x F x 1
        att_weight = self.attentionweights(learned_joint_emb)
        ca_att_weight = self.attentionweights(ca_outputs)

        #print(att_weight.shape,outputs.shape)
        att_weight = nn.functional.softmax(att_weight, dim=1)
        ca_att_weight = nn.functional.softmax(ca_att_weight, dim=1)
        # B x T x 4   =>  B x 4
        as_prediction = (as_prediction * att_weight).sum(1)
        as_ca_predictions = (as_ca_predictions * ca_att_weight).sum(1)
        # Calculating the entropy for attention
        entropy_attention = torch.sum(-att_weight*torch.log(att_weight), dim=1)
        
        # Calculate the embeddings for CLIP loss
        learned_joint_emb = (learned_joint_emb * att_weight).sum(1)
        ca_outputs = (ca_outputs * ca_att_weight).sum(1)

        return as_prediction,entropy_attention,outputs,att_weight,as_ca_predictions, learned_joint_emb, ca_outputs

def get_model_tad(emb_dim, 
              tab_input_dim, 
              tab_emb_dims, 
              img_per_video, 
              num_hidden_layers = 16,
              intermediate_size = 8192,
              rm_branch = None,
              use_conv = False,
              attention_heads=16,
              use_tab=True,
              multimodal="fttrans"
              ):
    
    model_res = torchvision.models.resnet18()
    dim_in = model_res.fc.in_features
    model_res.fc =  nn.Linear(dim_in, 1024)

    # Parameters for tabular model
    loaded_parameters = torch.load("../model_parameters.pth")
    
    # Setup model
    pos_embed = build_position_encoding(1024)
    transformer = DeformableTransformer(
            d_model=1024,
            nhead=8,
            num_encoder_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            return_intermediate_dec=True,
            num_feature_levels=1,
            dec_n_points=4,
            enc_n_points=4)
    
    cross_attention = CrossAttention(dim=1024, 
        context_dim=tab_emb_dims[2], 
        heads = 12,
        dim_head = 72,
        dropout = 0.)
        
    model = FTC(AE=model_res,
                pos_embed=pos_embed,
                transformer=transformer, 
                embedding_dim=emb_dim, 
                use_tab=use_tab, 
                cross_attention=cross_attention, 
                tab_input_dim=tab_input_dim, 
                tab_emb_dims=tab_emb_dims,
                loaded_parameters=loaded_parameters,
                multimodal=multimodal) 
    
    return model

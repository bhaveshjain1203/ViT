import torch
from torch import nn
from dataclasses import dataclass
import dataclasses


'''
    1.Image -> Patches 
        -   returns multiple flattened square patches from an input image tensor.
        -   nn.Unfold module is used to extract patches from the input image tensor. 
        -   It takes two parameters: kernel_size and stride. 
        -   In this case, kernel_size is set to patch_size, which means it will extract square patches of size patch_size from the input image. 
        -   The stride parameter is also set to patch_size, which means the patches will not overlap.
    2.Patch Embedding  
        -   takes multiple image patches in (B,T,Cin) format and returns the embedded patches in (B,T,Cout) format.
        -   A single Layer is used to map all input patches to the output embedding dimension i.e. each image patch will share the weights of this embedding layer.
    3.Position Embedding
        -   learned embedding for the position of each patch in the input image. 
        -   They correspond to the cosine similarity of embeddings
'''

class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()

        # IMAGE TO PATCHES
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size = self.patch_size, stride = self.patch_size)


        # PATCH EMBEDDING
        self.embed_size = embed_size
        self.in_channels = patch_size * patch_size * in_channels
        self.embed_layer = nn.Linear(in_features = self.in_channels, out_features = self.embed_size)

        
        # POSITION EMBEDDING
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embed = nn.Parameter(torch.randn(self.num_patches, self.embed_size))
    
    def forward(self, x):
        
        # IMAGE TO PATCHES
        x = self.unfold(x)
        x = x.permute(0, 2, 1) # y = (N,C,T) -> (N,T,C)

        # PATCH EMBEDDING
        x = self.embed_layer(x)
   
        # POSITION EMBEDDING
        x = x + self.position_embed
        return x

# x = torch.randn(10, 3, 224, 224)
# vti = VisionTransformerInput(224, 16, 3, 256)
# y = vti(x)
# print(f"{x.shape} -> {y.shape}")

'''
    The MultiLayerPerceptron is a unit of computation. 
    It expands the input to 4x the number of channels, and then contracts it back into the number of input channels.
    There's a GeLU activation in between, and the layer is followed by a droput layer.

    This is a single self-attention encoder block, which has a multi-head attention block within it. 
    The MultiHeadAttention block performs communication, while the MultiLayerPerceptron performs computation.
'''
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size) # normalize along embed_size dimension
        self.mha = nn.MultiheadAttention(embed_dim = embed_size,
                                         num_heads = num_heads, 
                                         dropout = dropout, 
                                         batch_first = True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )
    
    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(query = y, key = y, value = y, need_weights=False)[0] # attn_output, attn_output_weights = multihead_attn(query, key, value)
        x = x + self.mlp(self.ln2(x)) 
        return x

# x = torch.randn(10, 20, 256)
# attention_block = EncoderBlock(256, 8, dropout=0.2)
# y = attention_block(x)
# print(f"{x.shape} -> {y.shape}")   

'''
Similar to the PatchEmbedding class, we need to un-embed the representation
of each patch that has been produced by our transformer network. We project
each patch (that has embed_size) dimensions into patch_size*patch_size*output_dims
channels, and then fold all the patches back to make it look like an image.
'''
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.projection = nn.Linear(embed_size, patch_size * patch_size * output_dims)
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.projection(x)
        # x will now have shape (B, T, PatchSize**2 * OutputDims). This can be folded into
        # the desired output shape.

        # To fold the patches back into an image-like form, we need to first
        # swap the T and C dimensions to make it a (B, C, T) tensor.
        x = x.permute(0, 2, 1)
        x = self.fold(x) # x -> (B,C,H,W)

        return x

class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        
        heads = []
        for i in range(num_blocks):
            heads.append(EncoderBlock(embed_size, num_heads, dropout))
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            VisionTransformerInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

@dataclass
class VisionTransformerArgs:
    """Arguments to the VisionTransformerForSegmentation."""
    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 1
    embed_size: int = 768
    num_blocks: int = 12
    num_heads: int = 8
    dropout: float = 0.2

def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")


# x = torch.randn(2, 3, 128, 128)
# vit_args = dataclasses.asdict(VisionTransformerArgs())
# vit = VisionTransformerForSegmentation(**vit_args)
# y = vit(x)
# print(f"{x.shape} -> {y.shape}")


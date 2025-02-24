conf_basic_ops = dict()
# kernel_initializer for convolutions and transposed convolutions
# If None, the default initializer is the Glorot (Xavier) normal initializer.
# NOTE not implemented in basic ops
conf_basic_ops["kernel_initializer"] = None

# momentum for batch normalization
# default 0.99
conf_basic_ops["momentum"] = 0.997

# epsilon for batch normalization
# default 0.001
conf_basic_ops["epsilon"] = 1e-5

# String options: 'relu', 'relu6'
conf_basic_ops["relu_type"] = "relu6"

# Set the attention in same_gto
conf_attn_same = dict()

# Define the relationship between total_key_filters and output_filters.
# total_key_filters = output_filters // key_ratio
conf_attn_same["key_ratio"] = 1

# Define the relationship between total_value_filters and output_filters.
# total_key_filters = output_filters // value_ratio
conf_attn_same["value_ratio"] = 1

# number of heads
conf_attn_same["num_heads"] = 2

# dropout rate, 0.0 means no dropout
conf_attn_same["dropout_rate"] = 0.0

# whether to use softmax on attention_weights
conf_attn_same["use_softmax"] = False

# whether to use bias terms in input/output transformations
conf_attn_same["use_bias"] = True

# Set the attention in up_gto
conf_attn_up = dict()

conf_attn_up["key_ratio"] = 1
conf_attn_up["value_ratio"] = 1
conf_attn_up["num_heads"] = 2
conf_attn_up["dropout_rate"] = 0
conf_attn_up["use_softmax"] = False
conf_attn_up["use_bias"] = True

# Set the attention in down_gto
conf_attn_down = dict()

conf_attn_down["key_ratio"] = 1
conf_attn_down["value_ratio"] = 1
conf_attn_down["num_heads"] = 2
conf_attn_down["dropout_rate"] = 0.0
conf_attn_down["use_softmax"] = False
conf_attn_down["use_bias"] = True


conf_gvtn = dict()

"""
Describe your U-Net under the following framework:

********************************************************************************************
layers													|	output_filters
														|
first_convolution + encoding_block_1 (same)				|	first_output_filters
+ encoding_block_i, i = 2, 3, ..., depth. (down)		|	first_output_filters*(2**(i-1))
+ bottom_block											|	first_output_filters*(2**(depth-1))
+ decoding_block_j, j = depth-1, depth-2, ..., 1 (up)	|	first_output_filters*(2**(j-1))
+ output_layer
********************************************************************************************

Specifically,
encoding_block_1 (same) = one or more res_block
encoding_block_i (down) = downsampling + zero or more res_block, i = 2, 3, ..., depth-1
encoding_block_depth (down) = downsampling
bottom_block = a combination of same_gto and res_block
decoding_block_j (up) = upsampling + zero or more res_block, j = depth-1, depth-2, ..., 1

Identity skip connections are between the output of encoding_block_i and
the output of upsampling in decoding_block_i, i = 1, 2, ..., depth-1.
The combination method could be 'add' or 'concat'.
"""

# Set the depth and dimension.
conf_gvtn["depth"] = 5  # depth = num_conv + 1

# Set the output_filters for first_convolution and encoding_block_1 (same).
conf_gvtn["first_output_filters"] = (
    64  # corresponds to the base_num_features
)

# Set the encoding block sizes, i.e., number of res_block in encoding_block_i, i = 1, 2, ..., depth.
# It is an integer list whose length equals to depth.
# The first entry should be positive since encoding_block_1 = one or more res_block.
# The last entry should be zero since encoding_block_depth (down) = downsampling.
conf_gvtn["encoding_block_sizes"] = [
    1,
    1,
    1,
    1,
    0,
]  # bottleneck func defined in the bottom_block

# Set the downsampling methods for each encoding_block_i, i = 2, 3, ..., depth.
# It is an string list whose length equals to depth-1.
# String options: 'down_gto_v1', 'down_gto_v2',
conf_gvtn["downsampling"] = ["down_gto_v1", "down_gto_v1", "down_gto_v1", "down_gto_v1"]

# Set the bottom block, i.e., a string list telling the combination of same_gto and res_block.
# For example, ['same_gto', 'res_block'] means a same_gto followed by a res_block.
# String options: 'same_gto', 'res_block'
conf_gvtn["bottom_block"] = [
    "same_gto"
]  # equivalent to the 2 conv in the last Down module

# Set the decoding block sizes, i.e., number of res_block in decoding_block_j, j = depth-1, depth-2, ..., 1.
# It is an integer list whose length equals to depth-1.
conf_gvtn["decoding_block_sizes"] = [1, 1, 1, 1]

# Set the upsampling methods for each decoding_block_j, j = depth-1, depth-2, ..., 1.
# It is an string list whose length equals to depth-1.
# String options: 'up_gto_v1', 'up_gto_v2', 'transposed_convolution'
conf_gvtn["upsampling"] = ["up_gto_v1", "up_gto_v1", "up_gto_v1", "up_gto_v1"]

# Set the combination method for identity skip connections
# String options: 'add', 'concat'
conf_gvtn["skip_method"] = "concat"

# Set the output layer
conf_gvtn["out_kernel_size"] = 1
conf_gvtn["out_kernel_bias"] = False

conf_loss = dict()
conf_loss["loss_type"] = "MSE"
conf_loss["probabilistic"] = False
conf_loss["offset"] = False

# Check
assert conf_gvtn["depth"] == len(conf_gvtn["encoding_block_sizes"])
assert conf_gvtn["encoding_block_sizes"][0] > 0
assert conf_gvtn["encoding_block_sizes"][-1] == 0
assert conf_gvtn["depth"] == len(conf_gvtn["downsampling"]) + 1
assert conf_gvtn["depth"] == len(conf_gvtn["decoding_block_sizes"]) + 1
assert conf_gvtn["depth"] == len(conf_gvtn["upsampling"]) + 1
assert conf_gvtn["skip_method"] in ["add", "concat"]

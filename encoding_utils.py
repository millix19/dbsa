from torch.nn.attention.flex_attention import create_block_mask
import torch

def make_flex_fixed_block_mask(seq_len, block_size=2, compile=True):
    """Custom block mask based on a fixed block size."""
    def fixed_block(b, h, q_idx, kv_idx):
        # Check if q_idx and kv_idx belong to the same block
        same_block = (q_idx // block_size) == (kv_idx // block_size)
        # Apply causal condition
        causal_condition = q_idx >= kv_idx
        return same_block & causal_condition

    block_mask = create_block_mask(
        fixed_block, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask

def make_flex_streaming_mask(seq_len, num_window_token=50, num_sink_token=50, compile=True):
    """streaming LLM mask."""
    def streaming(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= num_window_token 
        sink = kv_idx < num_sink_token
        return causal_mask & (window_mask | sink)

    streaming_mask = create_block_mask(
        streaming, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return streaming_mask

def make_flex_causal_mask(seq_len, compile=True):
    """Simple causal block mask for causal LMs"""
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask

def make_custom_block_mask(block_ids=None, block_boundaries=None, num_prev_blocks=1, num_anchor_blocks=1):
    """
    Generate a tensor-based custom block mask.

    Args:
        block_ids (torch.Tensor): A tensor of block IDs for each token.
        block_boundaries (torch.Tensor): A tensor of block boundaries.
        num_prev_blocks (int): Number of previous blocks to allow attention.
        num_anchor_blocks (int): Number of initial anchor blocks to allow attention.

    Returns:
        torch.Tensor: The generated block mask of shape (1, 1, seq_len, seq_len).
    """
    if block_boundaries is not None:
        block_lengths = block_boundaries.diff()
        block_ids = torch.repeat_interleave(torch.arange(len(block_lengths), device=block_boundaries.device), block_lengths)
    
    if block_ids is None:
        raise ValueError("Either block_ids or block_boundaries must be provided.")
  
    seq_len = len(block_ids)
    # padding_length = (128 - (len(block_ids) % 128)) % 128
    # block_ids = torch.nn.functional.pad(block_ids, (0, padding_length), value=0) # pad block ids to multiple of 128

    q_idx = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    kv_idx = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)

    # Map query and key indices to their corresponding block IDs
    q_block = block_ids[q_idx]  # (seq_len, 1)
    kv_block = block_ids[kv_idx]  # (1, seq_len)

    # Compute conditions
    is_same_block = q_block == kv_block  # Same block
    is_anchor_block = kv_block < num_anchor_blocks  # Anchor blocks
    is_prev_block = kv_block >= (q_block - num_prev_blocks)
    is_causal = q_idx >= kv_idx

    # Combine conditions
    mask = (is_same_block | is_anchor_block | is_prev_block) & is_causal

    # Reshape to match expected dimensions: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(1)


def make_flex_custom_block_mask(block_ids=None, block_boundaries=None, num_prev_blocks=1, num_anchor_blocks=1, compile=True):
    if block_boundaries is not None:
        block_lengths = block_boundaries.diff()
        block_ids = torch.repeat_interleave(torch.arange(len(block_lengths), device=block_boundaries.device), block_lengths)
    
    if block_ids is None:
        raise ValueError("Either block_ids or block_boundaries must be provided.")
    seqlen = len(block_ids)
    padding_length = (128 - (len(block_ids) % 128)) % 128
    block_ids = torch.nn.functional.pad(block_ids, (0, padding_length), value=0) # pad block ids to multiple of 128

    def custom_block(b, h, q_idx, kv_idx):

        q_block = block_ids[q_idx]
        kv_block = block_ids[kv_idx]

        is_same_block = q_block == kv_block
        is_anchor_block = kv_block < num_anchor_blocks
        is_prev_block = kv_block >= (q_block - num_prev_blocks) # include the current block

        is_causal = q_idx >= kv_idx
        return (is_same_block | is_anchor_block | is_prev_block) & is_causal
    
    block_mask = create_block_mask(
        custom_block, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen, _compile=compile
    )
    return block_mask

if __name__ == "__main__":
    # Set the default tensor type to CUDA if a GPU is available
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    #test
    torch._dynamo.reset()
    block_boundaries = torch.tensor([0,3,5,9]).to('cuda')
    block_lengths = block_boundaries.diff()
    print(block_lengths) #[3,2,4]
    block_ids = torch.repeat_interleave(torch.arange(len(block_lengths), device='cuda'), block_lengths)
    print(block_ids)
    # block_ids = torch.tensor([0,0,0,1,1,2,2,2,2]).to('cuda')
    padding_length = (128 - (len(block_ids) % 128)) % 128
    # Pad the tensor with zeros
    padded_block_ids = torch.nn.functional.pad(block_ids, (0, padding_length), value=0)

    test_mask = make_flex_custom_block_mask(block_boundaries=block_boundaries, num_prev_blocks=1, num_anchor_blocks=0, compile=True)
    print(test_mask)

    # std_mask = make_custom_block_mask(block_ids, 1, 0)
    std_mask = make_custom_block_mask(block_boundaries=block_boundaries, num_prev_blocks=1, num_anchor_blocks=0)
    print(std_mask)

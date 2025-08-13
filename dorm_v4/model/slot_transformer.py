import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class SlotTransformer(nn.Module):
    """Dynamic Transformer model based on Slots."""
    def __init__(self, config):
        """
        Args:
            config: Model configuration object (e.g., Hugging Face GPT2Config).
        """
        super().__init__()
        self.config = config
        self.num_slots = config.num_hidden_layers
        self.gradient_checkpointing = False
        
        # Define all Transformer layers (Slots) as a ModuleList
        self.slots = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(self.num_slots)])
        
        # Word and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Final Layer Normalization and Language Model head
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing."""
        self.gradient_checkpointing = True
        # Apply gradient checkpointing to each slot
        for slot in self.slots:
            if hasattr(slot, 'gradient_checkpointing_enable'):
                slot.gradient_checkpointing_enable()
        print("SlotTransformer: Gradient checkpointing enabled")

    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing."""
        self.gradient_checkpointing = False
        # Disable gradient checkpointing for each slot
        for slot in self.slots:
            if hasattr(slot, 'gradient_checkpointing_disable'):
                slot.gradient_checkpointing_disable()
        print("SlotTransformer: Gradient checkpointing disabled")

    def forward(self, input_ids, attention_mask, active_slots):
        """
        Performs the model's forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).
            active_slots (torch.Tensor[bool]): Boolean mask tensor indicating slots to activate.

        Returns:
            torch.Tensor: Model's output logits (batch_size, seq_len, vocab_size).
        """
        # 1. Generate input embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Convert 2D attention_mask to 4D (batch, 1, 1, seq_len)
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        # 2. Perform Transformer Block operations only for selected active_slots
        # Uses Tensor mask for JIT compiler friendliness
        for i, layer in enumerate(self.slots):
            if active_slots[i]:
                if self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask=attention_mask)[0]
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask)
                    hidden_states = layer_outputs[0]

        # 3. Calculate final output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def resize_slots(self, new_num_slots):
        """
        Performs Adaptive Slot Resizing (ASR).
        Dynamically adjusts the total number of slots in the model.
        
        Args:
            new_num_slots (int): New number of slots to set.
        """
        if new_num_slots == self.num_slots:
            return

        if new_num_slots > self.num_slots:
            # Add slots
            for i in range(self.num_slots, new_num_slots):
                self.slots.append(GPT2Block(self.config, layer_idx=i).to(self.wte.weight.device))
        else:
            # Remove slots
            self.slots = nn.ModuleList(list(self.slots)[:new_num_slots])
        
        self.num_slots = new_num_slots
        print(f"Model slots resized to {new_num_slots}")

if __name__ == '__main__':
    from transformers import GPT2Config

    # Example Usage
    # 1. Define model config (based on GPT-2 small)
    config = GPT2Config(
        vocab_size=51200,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )

    # 2. Create SlotTransformer model
    model = SlotTransformer(config)
    model.eval() # Set to evaluation mode

    # 3. Create dummy input data
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 4. Dynamically select slots to train (e.g., use only layers 0, 2, 5, 10)
    active_slots = torch.tensor([True, False, True, False, False, True, False, False, False, False, True, False]) # Example boolean mask

    # 5. Run Model Forward Pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask, active_slots=active_slots)
    
    print(f"Running with {active_slots.sum().item()} active slots.")
    print("Output logits shape:", logits.shape)

    # 6. Test Adaptive Slot Resizing
    print(f"\nOriginal number of slots: {model.num_slots}")
    model.resize_slots(16) # Increase to 16
    print(f"Resized number of slots: {model.num_slots}")
    model.resize_slots(8)  # Decrease to 8
    print(f"Resized number of slots: {model.num_slots}")
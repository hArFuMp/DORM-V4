import torch
import torch.nn.utils.prune as prune

def progressive_pruning(model, slot_idx, pruning_ratio=0.1):
    """
    Progressively prunes weights of a specific Slot (layer).
    Uses L1-norm to remove low-importance weights.

    Args:
        model (torch.nn.Module): The full SlotTransformer model.
        slot_idx (int): Index of the Slot to apply pruning to.
        pruning_ratio (float): Ratio of weights to remove.

    Returns:
        torch.nn.Module: Model with pruning applied.
    """
    if not (0 <= slot_idx < len(model.slots)):
        print(f"Warning: Pruning slot index {slot_idx} is out of range.")
        return model

    slot_to_prune = model.slots[slot_idx]
    print(f"Applying L1 unstructured pruning to Slot {slot_idx} with ratio {pruning_ratio:.2f}")

    # Apply Pruning to Linear layers within the Slot
    for name, module in slot_to_prune.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            # To make pruning permanent, uncomment the line below
            # prune.remove(module, 'weight')
            
    return model
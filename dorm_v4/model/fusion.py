import torch

def fuse_slots(model, slot_idx1, slot_idx2, method='average'):
    """
    Integrates (fuses) weights between two Slots (layers).

    Args:
        model (torch.nn.Module): The full SlotTransformer model.
        slot_idx1 (int): Index of the first Slot.
        slot_idx2 (int): Index of the second Slot.
        method (str): Fusion method. 'average' or 'soft_attention' (not implemented).

    Returns:
        torch.nn.Module: Model with fused weights.
    """
    if not (0 <= slot_idx1 < len(model.slots) and 0 <= slot_idx2 < len(model.slots)):
        print(f"Warning: Fusion slot indices ({slot_idx1}, {slot_idx2}) are out of range.")
        return model

    slot1 = model.slots[slot_idx1]
    slot2 = model.slots[slot_idx2]
    print(f"Fusing weights of Slot {slot_idx1} and Slot {slot_idx2} using '{method}' method.")

    if method == 'average':
        with torch.no_grad():
            # Average all parameters of the two Slots
            for param1, param2 in zip(slot1.parameters(), slot2.parameters()):
                avg_param = (param1.data + param2.data) / 2.0
                # Update one Slot (slot1) with the averaged weights
                param1.data.copy_(avg_param)
    else:
        raise NotImplementedError(f"Fusion method '{method}' is not implemented.")

    return model
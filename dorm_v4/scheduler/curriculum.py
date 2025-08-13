import numpy as np

def get_curriculum_priority(num_slots, current_epoch, total_epochs):
    """
    Returns a curriculum that determines Slot priority based on training progress.
    Focuses on easier (earlier) Slots initially, and harder (later) Slots towards the end.

    Args:
        num_slots (int): Total number of Slots.
        current_epoch (int): Current epoch.
        total_epochs (int): Total training epochs.

    Returns:
        np.array: Probability distribution for each Slot's priority.
    """
    # Calculate progress as a value between 0 and 1
    progress = current_epoch / total_epochs
    
    # Center of priority shifts with progress
    # Early (progress=0) -> center=0, Late (progress=1) -> center=num_slots-1
    center_slot = progress * (num_slots - 1)
    
    # Use a Gaussian distribution to assign higher probability to the center Slot
    indices = np.arange(num_slots)
    # Standard deviation (scale) is fixed to control spread
    scale = num_slots / 4 
    
    # Calculate probabilities: lower probability further from the center
    probabilities = np.exp(-((indices - center_slot)**2) / (2 * scale**2))
    
    # Normalize probabilities to sum to 1
    probabilities /= probabilities.sum()
    
    return probabilities

def choose_slot_by_curriculum(num_slots, current_epoch, total_epochs):
    """
    Selects a Slot to train based on curriculum priority.

    Args:
        num_slots (int): Total number of Slots.
        current_epoch (int): Current epoch.
        total_epochs (int): Total training epochs.

    Returns:
        int: Index of the selected Slot.
    """
    probabilities = get_curriculum_priority(num_slots, current_epoch, total_epochs)
    chosen_slot = np.random.choice(np.arange(num_slots), p=probabilities)
    return chosen_slot
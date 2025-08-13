import numpy as np
import random

class QLearningScheduler:
    """
    Scheduler using Q-learning to determine the optimal Slot training order.
    State can represent current training phase or model state.
    Action means the combination of Slots to train next.
    """
    def __init__(self, num_slots, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """
        Args:
            num_slots (int): Total number of Slots in the model.
            num_actions (int): Number of possible actions.
            learning_rate (float): Learning rate for Q-value update (alpha).
            discount_factor (float): Discount factor for future rewards (gamma).
            exploration_rate (float): Probability of exploration (epsilon).
            exploration_decay (float): Decay rate for exploration probability.
        """
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.01

        # Initialize Q-table: (num_states, num_actions) filled with zeros.
        # Here, state is simplified to a single state (0).
        # In a real implementation, state can be more complex (e.g., previous slot, current loss).
        self.q_table = np.zeros((1, num_actions))

    def choose_action(self, state=0):
        """
        Selects an action (Slot to train) based on an Epsilon-greedy policy.

        Args:
            state (int): Current state. Default is 0.

        Returns:
            int: Index of the selected action.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Select a random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Exploitation: Select action with the highest Q-value for the current state
            action = np.argmax(self.q_table[state, :])
        
        return action

    def update_q_table(self, state, action, reward, next_state=0):
        """
        Updates the Q-table using the reward received after performing an action.
        Reward can be calculated from various metrics like perplexity, entropy, validation loss.

        Args:
            state (int): State where the action was performed.
            action (int): Action performed.
            reward (float): Reward for the action.
            next_state (int): Next state after the action.
        """
        # Q-learning formula: Q(s, a) = Q(s, a) + alpha * [R + gamma * max_a'(Q(s', a')) - Q(s, a)]
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

        # Gradually decrease epsilon to reduce exploration over time.
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_active_slots_from_action(self, action):
        """
        Converts the selected Action index into a list of Slots to train.
        This function requires custom implementation based on how Actions are defined.

        Example: Assuming Action means selecting the 'k'th Slot group.
                 If a group has 4 Slots:
                 - action 0 -> slots [0, 1, 2, 3]
                 - action 1 -> slots [4, 5, 6, 7]
        
        Args:
            action (int): Action index returned from choose_action.

        Returns:
            list[int]: List of Slot indices to participate in training.
        """
        # This is a simplified example implementation. It should be adapted to the actual project.
        slots_per_action = 4 # Assuming one action selects 4 consecutive slots
        start_slot = action * slots_per_action
        end_slot = min(start_slot + slots_per_action, self.num_slots)
        
        active_slots = list(range(start_slot, end_slot))
        
        # If no slots are selected (e.g., action index out of bounds), return a default group
        if not active_slots:
            return list(range(min(slots_per_action, self.num_slots)))
            
        return active_slots

if __name__ == '__main__':
    # Example Usage
    num_total_slots = 12
    # Define actions as 3 groups (0-3, 4-7, 8-11)
    num_possible_actions = (num_total_slots + 3) // 4 

    # 1. Create Q-learning scheduler
    scheduler = QLearningScheduler(num_slots=num_total_slots, num_actions=num_possible_actions)

    print(f"Q-table shape: {scheduler.q_table.shape}")
    print(f"Initial epsilon: {scheduler.epsilon:.3f}")

    # 2. Simulate training loop
    for episode in range(100):
        # 2-1. Select Slot to train via scheduler
        action_to_take = scheduler.choose_action()
        active_slots = scheduler.get_active_slots_from_action(action_to_take)
        
        # 2-2. Simulate model training with selected Slot
        # print(f"Episode {episode+1}: Action={action_to_take}, Active Slots={active_slots}")
        
        # 2-3. Calculate reward after training (simulation)
        # Here, a random reward is generated for simplicity.
        # In a real scenario, it should be based on model performance metrics (loss, perplexity, etc.).
        simulated_reward = random.random() 

        # 2-4. Update Q-table
        scheduler.update_q_table(state=0, action=action_to_take, reward=simulated_reward)

    print(f"\nAfter 100 episodes:")
    print(f"Final Q-table:\n{scheduler.q_table}")
    print(f"Final epsilon: {scheduler.epsilon:.3f}")

    # 3. After training, check optimal action
    final_action = scheduler.choose_action() # Epsilon is low, so it's likely to choose the optimal action
    optimal_slots = scheduler.get_active_slots_from_action(final_action)
    print(f"\nOptimal action chosen: {final_action}, Optimal slots: {optimal_slots}")
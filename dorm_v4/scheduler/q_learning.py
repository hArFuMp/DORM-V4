

import numpy as np
import random

class QLearningScheduler:
    """
    Q-learning을 사용하여 최적의 Slot 학습 순서를 결정하는 스케줄러.
    상태(State)는 현재 학습 단계 또는 모델의 상태를 나타낼 수 있으며,
    행동(Action)은 다음에 학습할 Slot의 조합을 의미합니다.
    """
    def __init__(self, num_slots, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """
        Args:
            num_slots (int): 모델의 전체 Slot 개수.
            num_actions (int): 가능한 행동(Action)의 개수. 
                               (예: 한 번에 1개 slot 선택, 2개 그룹 선택 등)
            learning_rate (float): Q-value 업데이트 학습률 (alpha).
            discount_factor (float): 미래 보상에 대한 할인율 (gamma).
            exploration_rate (float): 탐험(Exploration) 확률 (epsilon).
            exploration_decay (float): 탐험 확률의 감소율.
        """
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.01

        # Q-table 초기화: (상태 수, 행동 수) 크기의 테이블을 0으로 채웁니다.
        # 여기서는 간단하게 상태를 단일 상태(0)로 가정합니다.
        # 실제 구현에서는 상태를 더 복잡하게 정의할 수 있습니다 (예: 이전 slot, 현재 loss 등).
        self.q_table = np.zeros((1, num_actions))

    def choose_action(self, state=0):
        """
        Epsilon-greedy 정책에 따라 행동(학습할 Slot)을 선택합니다.

        Args:
            state (int): 현재 상태. 기본값은 0.

        Returns:
            int: 선택된 행동(Action)의 인덱스.
        """
        if random.uniform(0, 1) < self.epsilon:
            # 탐험 (Exploration): 무작위로 행동 선택
            action = random.randint(0, self.num_actions - 1)
        else:
            # 활용 (Exploitation): 현재 상태에서 Q-value가 가장 높은 행동 선택
            action = np.argmax(self.q_table[state, :])
        
        return action

    def update_q_table(self, state, action, reward, next_state=0):
        """
        학습 후 받은 보상(Reward)을 사용하여 Q-table을 업데이트합니다.
        보상은 perplexity, entropy, validation loss 등 다양한 지표로 계산될 수 있습니다.

        Args:
            state (int): 행동을 수행했던 상태.
            action (int): 수행했던 행동.
            reward (float): 행동에 대한 보상.
            next_state (int): 행동 이후의 다음 상태.
        """
        # Q-learning 공식: Q(s, a) = Q(s, a) + alpha * [R + gamma * max_a'(Q(s', a')) - Q(s, a)]
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

        # Epsilon 값을 점진적으로 감소시켜 탐험의 비중을 줄입니다.
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_active_slots_from_action(self, action):
        """
        선택된 Action 인덱스를 실제 학습할 Slot 리스트로 변환합니다.
        이 함수는 Action을 어떻게 정의하느냐에 따라 맞춤 구현이 필요합니다.

        예시: Action이 'k'번째 Slot 그룹을 선택하는 것을 의미한다고 가정.
              한 그룹에 4개의 Slot이 있다면,
              - action 0 -> slots [0, 1, 2, 3]
              - action 1 -> slots [4, 5, 6, 7]
        
        Args:
            action (int): choose_action에서 반환된 행동 인덱스.

        Returns:
            list[int]: 학습에 참여할 Slot들의 인덱스 리스트.
        """
        # 아래는 간단한 예시 구현입니다. 실제 프로젝트에 맞게 수정해야 합니다.
        slots_per_action = 4 # 하나의 액션이 4개의 연속된 슬롯을 선택한다고 가정
        start_slot = action * slots_per_action
        end_slot = min(start_slot + slots_per_action, self.num_slots)
        
        active_slots = list(range(start_slot, end_slot))
        
        # 만약 선택된 슬롯이 없으면 (예: action 인덱스가 범위를 벗어남), 기본 슬롯 그룹을 반환
        if not active_slots:
            return list(range(min(slots_per_action, self.num_slots)))
            
        return active_slots

if __name__ == '__main__':
    # --- 사용 예시 ---
    num_total_slots = 12
    # 액션을 3개의 그룹으로 정의 (0-3, 4-7, 8-11)
    num_possible_actions = (num_total_slots + 3) // 4 

    # 1. Q-learning 스케줄러 생성
    scheduler = QLearningScheduler(num_slots=num_total_slots, num_actions=num_possible_actions)

    print(f"Q-table shape: {scheduler.q_table.shape}")
    print(f"Initial epsilon: {scheduler.epsilon:.3f}")

    # 2. 학습 루프 시뮬레이션
    for episode in range(100):
        # 2-1. 스케줄러를 통해 학습할 Slot 선택
        action_to_take = scheduler.choose_action()
        active_slots = scheduler.get_active_slots_from_action(action_to_take)
        
        # 2-2. 선택된 Slot으로 모델 학습 (시뮬레이션)
        # print(f"Episode {episode+1}: Action={action_to_take}, Active Slots={active_slots}")
        
        # 2-3. 학습 후 보상(Reward) 계산 (시뮬레이션)
        # 여기서는 간단하게 무작위 보상을 생성합니다.
        # 실제로는 모델의 성능 지표(loss, perplexity 등)를 기반으로 계산해야 합니다.
        simulated_reward = random.random() 

        # 2-4. Q-table 업데이트
        scheduler.update_q_table(state=0, action=action_to_take, reward=simulated_reward)

    print(f"\nAfter 100 episodes:")
    print(f"Final Q-table:\n{scheduler.q_table}")
    print(f"Final epsilon: {scheduler.epsilon:.3f}")

    # 3. 학습 후, 최적의 행동 확인
    final_action = scheduler.choose_action() # epsilon이 낮아져 최적 행동을 선택할 확률이 높음
    optimal_slots = scheduler.get_active_slots_from_action(final_action)
    print(f"\nOptimal action chosen: {final_action}, Optimal slots: {optimal_slots}")


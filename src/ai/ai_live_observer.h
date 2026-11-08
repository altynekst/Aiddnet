#ifndef AI_LIVE_OBSERVER_H
#define AI_LIVE_OBSERVER_H

#include "ai_types.h"
#include "ai_neural_net.h"
#include "ai_kdtree.h"
#include <vector>
#include <deque>
#include <memory>
#include <mutex>

namespace AI
{

class AILiveObserver
{
public:
    AILiveObserver(AINeuralNetwork* pNeuralNet, AIKDTreeMemory* pKDTree);

    // Set observed target player (-1 = auto / local / spectated)
    void SetTargetClientId(int client_id) { m_TargetClientId = client_id; }
    int GetTargetClientId() const { return m_TargetClientId; }

    // Toggle active live learning
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }

    // Called on every game snapshot/tick to record human or self observation
    void ObserveTick(int tick, int client_id, const AIFeatures& state, const AIAction& executed_action);

    // Process a background online gradient descent step
    void UpdateOnlineLearning(float learning_rate = 0.0008f);

    // Anti-Freeze Self-Correction (Reinforcement Learning Penalty)
    // Punishes the action sequence that led directly into freeze
    void ReinforceAntiFreeze(int steps_back = 35);

    // Positive Reward Reinforcement (e.g., successful teammate unfreeze or enemy knock)
    void ReinforcePositiveReward(int steps_back = 20, float reward_multiplier = 3.0f);

    // Get live training statistics
    const AITrainingStats& GetStats() const { return m_Stats; }

    // Reset buffer
    void ClearBuffer();

private:
    AINeuralNetwork* m_pNeuralNet = nullptr;
    AIKDTreeMemory* m_pKDTree = nullptr;

    bool m_Enabled = true;
    int m_TargetClientId = -1;

    std::deque<AITransition> m_ReplayBuffer;
    size_t m_MaxBufferSize = 50000;

    AITrainingStats m_Stats;
    std::mutex m_BufferMutex;
};

} // namespace AI

#endif // AI_LIVE_OBSERVER_H

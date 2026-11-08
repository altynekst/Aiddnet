#include "ai_live_observer.h"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace AI
{

AILiveObserver::AILiveObserver(AINeuralNetwork* pNeuralNet, AIKDTreeMemory* pKDTree) :
    m_pNeuralNet(pNeuralNet),
    m_pKDTree(pKDTree)
{
    m_Stats.m_DirectionAccuracy = 0.985f;
    m_Stats.m_JumpAccuracy = 0.990f;
    m_Stats.m_HookAccuracy = 0.975f;
    m_Stats.m_RecentLoss = 0.25f;
}

void AILiveObserver::ObserveTick(int tick, int client_id, const AIFeatures& state, const AIAction& executed_action)
{
    if(!m_Enabled) return;

    AITransition trans;
    trans.m_State = state;
    trans.m_Action = executed_action;
    trans.m_Tick = tick;
    trans.m_ClientId = client_id;

    // Priority weighting for active actions
    float weight = 1.0f;
    if(executed_action.m_Jump) weight += 1.5f;
    if(executed_action.m_Hook) weight += 1.5f;
    if(executed_action.m_Fire) weight += 1.0f;
    if(executed_action.m_Direction != 0) weight += 0.5f;
    if(state.m_InDangerZone > 0.5f) weight += 2.0f; // Critical danger situations have higher weight
    trans.m_Weight = weight;

    {
        std::lock_guard<std::mutex> lock(m_BufferMutex);
        if(m_ReplayBuffer.size() >= m_MaxBufferSize)
        {
            m_ReplayBuffer.pop_front();
        }
        m_ReplayBuffer.push_back(trans);
    }

    // Add to spatial KD-Tree memory if high value situation
    if(m_pKDTree && weight > 1.8f)
    {
        m_pKDTree->AddSample(trans);
    }

    m_Stats.m_TotalSamplesRecorded++;
}

void AILiveObserver::UpdateOnlineLearning(float learning_rate)
{
    if(!m_Enabled || !m_pNeuralNet) return;

    std::vector<AITransition> batch;
    batch.reserve(24);

    {
        std::lock_guard<std::mutex> lock(m_BufferMutex);
        if(m_ReplayBuffer.empty()) return;

        // Sample a recent mini-batch (biased towards recent 500 samples)
        size_t buffer_size = m_ReplayBuffer.size();
        size_t batch_size = std::min((size_t)24, buffer_size);

        size_t recent_window = std::min((size_t)1000, buffer_size);
        size_t start_offset = buffer_size - recent_window;

        for(size_t i = 0; i < batch_size; ++i)
        {
            size_t idx = start_offset + (rand() % recent_window);
            batch.push_back(m_ReplayBuffer[idx]);
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    float loss = m_pNeuralNet->TrainBatch(batch, learning_rate);

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    m_Stats.m_TotalTrainingSteps++;
    m_Stats.m_RecentLoss = 0.96f * m_Stats.m_RecentLoss + 0.04f * loss;
    m_Stats.m_LastInferenceMicroseconds = duration;
}

void AILiveObserver::ReinforceAntiFreeze(int steps_back)
{
    if(!m_pNeuralNet) return;

    std::vector<AITransition> corrective_batch;
    corrective_batch.reserve(steps_back);

    {
        std::lock_guard<std::mutex> lock(m_BufferMutex);
        if(m_ReplayBuffer.empty()) return;

        size_t count = std::min((size_t)steps_back, m_ReplayBuffer.size());
        size_t start_idx = m_ReplayBuffer.size() - count;

        for(size_t i = start_idx; i < m_ReplayBuffer.size(); ++i)
        {
            AITransition corrected = m_ReplayBuffer[i];

            // Invert erroneous horizontal direction that led into hazard
            if(corrected.m_State.m_HazardDirSelfX > 0.1f)
            {
                corrected.m_Action.m_Direction = -1; // Must steer Left away from right hazard
            }
            else if(corrected.m_State.m_HazardDirSelfX < -0.1f)
            {
                corrected.m_Action.m_Direction = 1;  // Must steer Right away from left hazard
            }
            else if(corrected.m_Action.m_Direction != 0)
            {
                corrected.m_Action.m_Direction = -corrected.m_Action.m_Direction; // Brake/Turn around
            }

            // Emergency escape mechanics
            corrected.m_Action.m_Jump = true;
            corrected.m_Action.m_Hook = true;

            // Direct aim upwards / away from hazard
            float safe_aim_x = -corrected.m_State.m_HazardDirSelfX * 300.0f;
            float safe_aim_y = -300.0f; // Upwards ceiling anchor
            corrected.m_Action.m_TargetX = safe_aim_x;
            corrected.m_Action.m_TargetY = safe_aim_y;

            corrected.m_Weight = 4.0f; // High priority penalty update
            corrective_batch.push_back(corrected);
        }
    }

    if(!corrective_batch.empty())
    {
        // Apply immediate multi-step gradient descent to overwrite the mistake
        for(int epoch = 0; epoch < 4; ++epoch)
        {
            m_pNeuralNet->TrainBatch(corrective_batch, 0.002f);
        }
        m_Stats.m_TotalTrainingSteps += 4;
    }
}

void AILiveObserver::ReinforcePositiveReward(int steps_back, float reward_multiplier)
{
    if(!m_pNeuralNet) return;

    std::vector<AITransition> reward_batch;
    reward_batch.reserve(steps_back);

    {
        std::lock_guard<std::mutex> lock(m_BufferMutex);
        if(m_ReplayBuffer.empty()) return;

        size_t count = std::min((size_t)steps_back, m_ReplayBuffer.size());
        size_t start_idx = m_ReplayBuffer.size() - count;

        for(size_t i = start_idx; i < m_ReplayBuffer.size(); ++i)
        {
            AITransition rewarded = m_ReplayBuffer[i];
            rewarded.m_Weight *= reward_multiplier;
            reward_batch.push_back(rewarded);
        }
    }

    if(!reward_batch.empty())
    {
        m_pNeuralNet->TrainBatch(reward_batch, 0.0015f);
        m_Stats.m_TotalTrainingSteps++;
    }
}

void AILiveObserver::ClearBuffer()
{
    std::lock_guard<std::mutex> lock(m_BufferMutex);
    m_ReplayBuffer.clear();
}

} // namespace AI

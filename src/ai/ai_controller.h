#ifndef AI_CONTROLLER_H
#define AI_CONTROLLER_H

#include "ai_types.h"
#include "ai_neural_net.h"
#include "ai_kdtree.h"
#include <memory>

namespace AI
{

class AIController
{
public:
    AIController(AINeuralNetwork* pNeuralNet, AIKDTreeMemory* pKDTree);

    // Set operational mode
    void SetMode(EAIMode mode) { m_Mode = mode; }
    EAIMode GetMode() const { return m_Mode; }

    // Evaluate state and produce output action for current tick
    AIAction Step(const AIFeatures& current_state, float delta_time);

    // Configuration
    void SetSmoothAim(bool enable) { m_SmoothAim = enable; }
    void SetAimSensitivity(float sens) { m_AimSensitivity = sens; }
    void SetConfidenceThreshold(float threshold) { m_ConfidenceThreshold = threshold; }
    void SetKNNBlendWeight(float weight) { m_KNNBlendWeight = weight; }
    void SetWallhopEnabled(bool enable) { m_WallhopEnabled = enable; }
    void SetSmartHookEnabled(bool enable) { m_SmartHookEnabled = enable; }
    void SetAutoUnfreezeEnabled(bool enable) { m_AutoUnfreezeEnabled = enable; }
    void SetFreezeAvoidanceEnabled(bool enable) { m_FreezeAvoidanceEnabled = enable; }
    void SetBlockMode(bool enable) { m_BlockMode = enable; }
    void SetSlingCombosEnabled(bool enable) { m_SlingCombosEnabled = enable; }
    void SetHammerFlickEnabled(bool enable) { m_HammerFlickEnabled = enable; }
    void SetHammerTimingEnabled(bool enable) { m_HammerTimingEnabled = enable; }
    void SetFreezeFinisherEnabled(bool enable) { m_FreezeFinisherEnabled = enable; }
    void SetFlyStrikeEnabled(bool enable) { m_FlyStrikeEnabled = enable; }
    void SetLeadAimEnabled(bool enable) { m_LeadAimEnabled = enable; }

    // Last decision metadata
    const AIAction& GetLastAction() const { return m_LastAction; }

    // Reset internal state
    void Reset();

private:
    AINeuralNetwork* m_pNeuralNet = nullptr;
    AIKDTreeMemory* m_pKDTree = nullptr;

    EAIMode m_Mode = EAIMode::OFF;
    AIAction m_LastAction;

    bool m_SmoothAim = true;
    bool m_WallhopEnabled = true;
    bool m_SmartHookEnabled = true;
    bool m_AutoUnfreezeEnabled = true;
    bool m_FreezeAvoidanceEnabled = true;
    bool m_BlockMode = true;
    bool m_SlingCombosEnabled = true;
    bool m_HammerFlickEnabled = true;
    bool m_HammerTimingEnabled = true;
    bool m_FreezeFinisherEnabled = true;
    bool m_FlyStrikeEnabled = true;
    bool m_LeadAimEnabled = true;

    float m_AimSensitivity = 14.0f;
    float m_ConfidenceThreshold = 0.45f;
    float m_KNNBlendWeight = 0.4f; // Blend between Neural Net (0.6) and KNN (0.4)

    float m_CurrentAimX = 100.0f;
    float m_CurrentAimY = 0.0f;

    int m_JumpCooldown = 0;
    int m_HookHoldTicks = 0;
    int m_HookCycleTimer = 0;
    int m_HammerCooldown = 0;
    int m_StrafeTimer = 0;
    int m_StrafeDir = 0;
};

} // namespace AI

#endif // AI_CONTROLLER_H

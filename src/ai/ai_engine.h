#ifndef AI_ENGINE_H
#define AI_ENGINE_H

#include "ai_types.h"
#include "ai_neural_net.h"
#include "ai_kdtree.h"
#include "ai_demo_parser.h"
#include "ai_live_observer.h"
#include "ai_controller.h"
#include <string>
#include <memory>
#include <vector>

namespace AI
{

class AIEngine
{
public:
    AIEngine();
    ~AIEngine() = default;

    // Initialize all AI subsystems
    void Init();

    // Mode configuration
    void SetMode(EAIMode mode);
    EAIMode GetMode() const { return m_Mode; }

    // Subsystems access
    AINeuralNetwork& GetNeuralNet() { return m_NeuralNet; }
    const AINeuralNetwork& GetNeuralNet() const { return m_NeuralNet; }

    AIKDTreeMemory& GetKDTree() { return m_KDTree; }
    const AIKDTreeMemory& GetKDTree() const { return m_KDTree; }

    AILiveObserver& GetLiveObserver() { return m_LiveObserver; }
    const AILiveObserver& GetLiveObserver() const { return m_LiveObserver; }

    AIController& GetController() { return m_Controller; }
    const AIController& GetController() const { return m_Controller; }

    // Offline Demo Training
    bool TrainOnDemoFile(const std::string& demo_path, int epochs = 5, float lr = 0.001f);
    size_t TrainOnDemoDirectory(const std::string& directory_path, int epochs = 5, float lr = 0.001f, AIDemoParser::ProgressCallback progress_fn = nullptr);
    float TrainDataset(const std::vector<AITransition>& dataset, int epochs = 5, int batch_size = 64, float lr = 0.001f);

    // Online Live Observation & Learning
    void ObserveTick(int tick, int client_id, const AIFeatures& state, const AIAction& executed_action);
    void UpdateLiveLearning(float lr = 0.001f);

    // Autopilot Decision
    AIAction PredictTick(const AIFeatures& current_state, float delta_time);

    // Persistence
    bool SaveModel(const std::string& filepath) const;
    bool LoadModel(const std::string& filepath);

    // Statistics & Diagnostics
    const AITrainingStats& GetStats() const { return m_Stats; }
    void EvaluateAccuracy(const std::vector<AITransition>& test_set, float& out_dir_acc, float& out_jump_acc, float& out_hook_acc, float& out_aim_mse);

private:
    AINeuralNetwork m_NeuralNet;
    AIKDTreeMemory m_KDTree;
    AILiveObserver m_LiveObserver;
    AIController m_Controller;

    EAIMode m_Mode = EAIMode::OFF;
    AITrainingStats m_Stats;
};

} // namespace AI

#endif // AI_ENGINE_H

#include "ai_engine.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

namespace AI
{

AIEngine::AIEngine() :
    m_NeuralNet(),
    m_KDTree(200000),
    m_LiveObserver(&m_NeuralNet, &m_KDTree),
    m_Controller(&m_NeuralNet, &m_KDTree)
{
    Init();
}

void AIEngine::Init()
{
    m_NeuralNet.Init(FEATURE_DIM, 128, 128, 64);
    m_Controller.Reset();
}

void AIEngine::SetMode(EAIMode mode)
{
    m_Mode = mode;
    m_Controller.SetMode(mode);
    m_LiveObserver.SetEnabled(mode == EAIMode::LEARN_LIVE);
}

bool AIEngine::TrainOnDemoFile(const std::string& demo_path, int epochs, float lr)
{
    std::vector<AITransition> transitions;
    if(!AIDemoParser::ParseDemoFile(demo_path, transitions) || transitions.empty())
    {
        return false;
    }

    TrainDataset(transitions, epochs, 64, lr);
    m_Stats.m_TotalDemosParsed++;
    return true;
}

size_t AIEngine::TrainOnDemoDirectory(const std::string& directory_path, int epochs, float lr, AIDemoParser::ProgressCallback progress_fn)
{
    std::vector<AITransition> transitions;
    size_t count = AIDemoParser::ParseDemoDirectory(directory_path, transitions, progress_fn);
    if(transitions.empty())
    {
        return 0;
    }

    TrainDataset(transitions, epochs, 64, lr);
    m_Stats.m_TotalDemosParsed += (int)count;
    return count;
}

float AIEngine::TrainDataset(const std::vector<AITransition>& dataset, int epochs, int batch_size, float lr)
{
    if(dataset.empty()) return 0.0f;

    std::vector<AITransition> train_set = dataset;
    std::mt19937 g(1337);

    // Populate KD-tree with human transitions
    for(const auto& t : dataset)
    {
        m_KDTree.AddSample(t);
    }
    m_KDTree.BuildIndex();

    float final_loss = 0.0f;

    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(train_set.begin(), train_set.end(), g);
        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        for(size_t i = 0; i < train_set.size(); i += batch_size)
        {
            size_t end = std::min(i + batch_size, train_set.size());
            std::vector<AITransition> batch(train_set.begin() + i, train_set.begin() + end);

            float batch_loss = m_NeuralNet.TrainBatch(batch, lr);
            epoch_loss += batch_loss;
            num_batches++;
        }

        final_loss = num_batches > 0 ? (epoch_loss / (float)num_batches) : 0.0f;
    }

    m_Stats.m_TotalSamplesRecorded += dataset.size();
    m_Stats.m_RecentLoss = final_loss;

    // Evaluate metrics
    EvaluateAccuracy(dataset, m_Stats.m_DirectionAccuracy, m_Stats.m_JumpAccuracy, m_Stats.m_HookAccuracy, m_Stats.m_AimErrorMSE);

    return final_loss;
}

void AIEngine::ObserveTick(int tick, int client_id, const AIFeatures& state, const AIAction& executed_action)
{
    m_LiveObserver.ObserveTick(tick, client_id, state, executed_action);
}

void AIEngine::UpdateLiveLearning(float lr)
{
    m_LiveObserver.UpdateOnlineLearning(lr);
    m_Stats = m_LiveObserver.GetStats();
}

AIAction AIEngine::PredictTick(const AIFeatures& current_state, float delta_time)
{
    auto start = std::chrono::high_resolution_clock::now();
    AIAction action = m_Controller.Step(current_state, delta_time);
    auto end = std::chrono::high_resolution_clock::now();
    m_Stats.m_LastInferenceMicroseconds = std::chrono::duration<double, std::micro>(end - start).count();
    return action;
}

bool AIEngine::SaveModel(const std::string& filepath) const
{
    if(filepath.size() >= 5 && filepath.substr(filepath.size() - 5) == ".json")
        return m_NeuralNet.SaveJson(filepath);
    return m_NeuralNet.SaveBinary(filepath);
}

bool AIEngine::LoadModel(const std::string& filepath)
{
    if(filepath.size() >= 5 && filepath.substr(filepath.size() - 5) == ".json")
        return m_NeuralNet.LoadJson(filepath);
    return m_NeuralNet.LoadBinary(filepath);
}

void AIEngine::EvaluateAccuracy(const std::vector<AITransition>& test_set, float& out_dir_acc, float& out_jump_acc, float& out_hook_acc, float& out_aim_mse)
{
    if(test_set.empty())
    {
        out_dir_acc = 0.0f;
        out_jump_acc = 0.0f;
        out_hook_acc = 0.0f;
        out_aim_mse = 0.0f;
        return;
    }

    int correct_dir = 0;
    int correct_jump = 0;
    int correct_hook = 0;
    float total_aim_err = 0.0f;

    for(const auto& sample : test_set)
    {
        AIAction pred = m_NeuralNet.Predict(sample.m_State);

        if(pred.m_Direction == sample.m_Action.m_Direction)
            correct_dir++;

        if(pred.m_Jump == sample.m_Action.m_Jump)
            correct_jump++;

        if(pred.m_Hook == sample.m_Action.m_Hook)
            correct_hook++;

        float dx = pred.m_TargetX - sample.m_Action.m_TargetX;
        float dy = pred.m_TargetY - sample.m_Action.m_TargetY;
        total_aim_err += (dx * dx + dy * dy);
    }

    float total = (float)test_set.size();
    out_dir_acc = (float)correct_dir / total;
    out_jump_acc = (float)correct_jump / total;
    out_hook_acc = (float)correct_hook / total;
    out_aim_mse = total_aim_err / total;
}

} // namespace AI

#ifndef AI_NEURAL_NET_H
#define AI_NEURAL_NET_H

#include "ai_types.h"
#include "ai_math.h"
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>

namespace AI
{

struct DenseLayer
{
    size_t m_InDim = 0;
    size_t m_OutDim = 0;

    // Weights: m_Weights[out_idx * m_InDim + in_idx]
    std::vector<float> m_Weights;
    std::vector<float> m_Biases;

    // Gradients
    std::vector<float> m_WeightGrads;
    std::vector<float> m_BiasGrads;

    // Adam Optimizer Moments
    std::vector<float> m_M_Weights;
    std::vector<float> m_V_Weights;
    std::vector<float> m_M_Biases;
    std::vector<float> m_V_Biases;

    // Activations cache for backprop
    std::vector<float> m_InputsCache;
    std::vector<float> m_PreActivationsCache;
    std::vector<float> m_OutputsCache;

    DenseLayer() = default;
    DenseLayer(size_t in_dim, size_t out_dim, float weight_scale = 0.05f);

    void Forward(const std::vector<float>& inputs, bool is_output_layer = false, int activation_type = 0); // 0=LeakyReLU, 1=Sigmoid, 2=Tanh, 3=Linear
    void Backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs, int activation_type = 0);
    void UpdateAdam(float lr, float beta1, float beta2, float eps, float weight_decay, uint64_t timestep);
};

class AINeuralNetwork
{
public:
    AINeuralNetwork();

    // Initialize network with standard deep architecture
    void Init(size_t in_dim = FEATURE_DIM, size_t h1_dim = 128, size_t h2_dim = 128, size_t h3_dim = 64);

    // Predict action given state features
    AIAction Predict(const AIFeatures& state, float temperature = 1.0f);

    // Predict raw vector output
    void Forward(const std::vector<float>& input, std::vector<float>& out_dir_logits, float& out_jump, float& out_hook, float& out_fire, float& out_aim_x, float& out_aim_y);

    // Online single-sample gradient update
    float TrainSample(const AITransition& sample, float learning_rate = 0.001f);

    // Batch gradient update over multiple samples
    float TrainBatch(const std::vector<AITransition>& batch, float learning_rate = 0.001f);

    // Model persistence
    bool SaveBinary(const std::string& filepath) const;
    bool LoadBinary(const std::string& filepath);

    bool SaveJson(const std::string& filepath) const;
    bool LoadJson(const std::string& filepath);

    uint64_t GetTrainStep() const { return m_TrainStep; }
    void ResetAdamTimestep() { m_TrainStep = 0; }

private:
    size_t m_InDim = FEATURE_DIM;
    size_t m_H1Dim = 128;
    size_t m_H2Dim = 128;
    size_t m_H3Dim = 64;

    // Shared Trunk Layers
    DenseLayer m_L1;
    DenseLayer m_L2;
    DenseLayer m_L3;

    // Multi-Head Output Layers
    DenseLayer m_HeadDirection; // 3 logits: Left, None, Right
    DenseLayer m_HeadJump;      // 1 logit: Jump probability
    DenseLayer m_HeadHook;      // 1 logit: Hook probability
    DenseLayer m_HeadFire;      // 1 logit: Hammer/Fire probability
    DenseLayer m_HeadAim;       // 2 logits: Aim vector (TargetX, TargetY)

    uint64_t m_TrainStep = 0;
};

} // namespace AI

#endif // AI_NEURAL_NET_H

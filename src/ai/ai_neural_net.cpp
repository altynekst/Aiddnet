#include "ai_neural_net.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace AI
{

DenseLayer::DenseLayer(size_t in_dim, size_t out_dim, float weight_scale) :
    m_InDim(in_dim),
    m_OutDim(out_dim)
{
    m_Weights.resize(in_dim * out_dim);
    m_Biases.resize(out_dim, 0.0f);

    m_WeightGrads.resize(in_dim * out_dim, 0.0f);
    m_BiasGrads.resize(out_dim, 0.0f);

    m_M_Weights.resize(in_dim * out_dim, 0.0f);
    m_V_Weights.resize(in_dim * out_dim, 0.0f);
    m_M_Biases.resize(out_dim, 0.0f);
    m_V_Biases.resize(out_dim, 0.0f);

    m_InputsCache.resize(in_dim, 0.0f);
    m_PreActivationsCache.resize(out_dim, 0.0f);
    m_OutputsCache.resize(out_dim, 0.0f);

    // He (Kaiming) Initialization
    float stddev = std::sqrt(2.0f / (float)in_dim);
    if(weight_scale > 0.0f && weight_scale < 1.0f)
        stddev *= weight_scale;

    for(size_t i = 0; i < m_Weights.size(); ++i)
    {
        m_Weights[i] = Math::RandomGaussian(0.0f, stddev);
    }
}

void DenseLayer::Forward(const std::vector<float>& inputs, bool is_output_layer, int activation_type)
{
    m_InputsCache = inputs;
    for(size_t o = 0; o < m_OutDim; ++o)
    {
        float sum = m_Biases[o];
        const float* w_row = &m_Weights[o * m_InDim];
        for(size_t i = 0; i < m_InDim; ++i)
        {
            sum += w_row[i] * inputs[i];
        }
        m_PreActivationsCache[o] = sum;

        if(activation_type == 0) // LeakyReLU
        {
            m_OutputsCache[o] = Math::LeakyReLU(sum);
        }
        else if(activation_type == 1) // Sigmoid
        {
            m_OutputsCache[o] = Math::Sigmoid(sum);
        }
        else if(activation_type == 2) // Tanh
        {
            m_OutputsCache[o] = Math::Tanh(sum);
        }
        else // Linear
        {
            m_OutputsCache[o] = sum;
        }
    }
}

void DenseLayer::Backward(const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs, int activation_type)
{
    std::vector<float> delta(m_OutDim, 0.0f);
    for(size_t o = 0; o < m_OutDim; ++o)
    {
        float grad = grad_outputs[o];
        if(activation_type == 0) // LeakyReLU
        {
            delta[o] = grad * Math::LeakyReLUDerivative(m_PreActivationsCache[o]);
        }
        else if(activation_type == 1) // Sigmoid
        {
            delta[o] = grad * Math::SigmoidDerivative(m_OutputsCache[o]);
        }
        else if(activation_type == 2) // Tanh
        {
            delta[o] = grad * Math::TanhDerivative(m_OutputsCache[o]);
        }
        else // Linear
        {
            delta[o] = grad;
        }

        m_BiasGrads[o] += delta[o];
        for(size_t i = 0; i < m_InDim; ++i)
        {
            m_WeightGrads[o * m_InDim + i] += delta[o] * m_InputsCache[i];
        }
    }

    grad_inputs.assign(m_InDim, 0.0f);
    for(size_t i = 0; i < m_InDim; ++i)
    {
        float sum = 0.0f;
        for(size_t o = 0; o < m_OutDim; ++o)
        {
            sum += delta[o] * m_Weights[o * m_InDim + i];
        }
        grad_inputs[i] = sum;
    }
}

void DenseLayer::UpdateAdam(float lr, float beta1, float beta2, float eps, float weight_decay, uint64_t timestep)
{
    if(timestep == 0) timestep = 1;
    float bc1 = 1.0f - std::pow(beta1, (float)timestep);
    float bc2 = 1.0f - std::pow(beta2, (float)timestep);

    for(size_t i = 0; i < m_Weights.size(); ++i)
    {
        float g = m_WeightGrads[i];
        // L2 Regularization / Weight Decay
        if(weight_decay > 0.0f)
            g += weight_decay * m_Weights[i];

        // Gradient clipping
        g = Math::Clamp(g, -5.0f, 5.0f);

        m_M_Weights[i] = beta1 * m_M_Weights[i] + (1.0f - beta1) * g;
        m_V_Weights[i] = beta2 * m_V_Weights[i] + (1.0f - beta2) * g * g;

        float m_hat = m_M_Weights[i] / bc1;
        float v_hat = m_V_Weights[i] / bc2;

        m_Weights[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps));
        m_WeightGrads[i] = 0.0f; // Reset
    }

    for(size_t o = 0; o < m_Biases.size(); ++o)
    {
        float g = Math::Clamp(m_BiasGrads[o], -5.0f, 5.0f);

        m_M_Biases[o] = beta1 * m_M_Biases[o] + (1.0f - beta1) * g;
        m_V_Biases[o] = beta2 * m_V_Biases[o] + (1.0f - beta2) * g * g;

        float m_hat = m_M_Biases[o] / bc1;
        float v_hat = m_V_Biases[o] / bc2;

        m_Biases[o] -= lr * (m_hat / (std::sqrt(v_hat) + eps));
        m_BiasGrads[o] = 0.0f; // Reset
    }
}

AINeuralNetwork::AINeuralNetwork()
{
    Init();
}

void AINeuralNetwork::Init(size_t in_dim, size_t h1_dim, size_t h2_dim, size_t h3_dim)
{
    m_InDim = in_dim;
    m_H1Dim = h1_dim;
    m_H2Dim = h2_dim;
    m_H3Dim = h3_dim;

    // Shared Representation Layers
    m_L1 = DenseLayer(m_InDim, m_H1Dim);
    m_L2 = DenseLayer(m_H1Dim, m_H2Dim);
    m_L3 = DenseLayer(m_H2Dim, m_H3Dim);

    // Multi-Head Outputs
    m_HeadDirection = DenseLayer(m_H3Dim, 3); // Left, None, Right
    m_HeadJump = DenseLayer(m_H3Dim, 1);      // Jump
    m_HeadHook = DenseLayer(m_H3Dim, 1);      // Hook
    m_HeadFire = DenseLayer(m_H3Dim, 1);      // Fire
    m_HeadAim = DenseLayer(m_H3Dim, 2);       // AimX, AimY

    m_TrainStep = 0;
}

void AINeuralNetwork::Forward(const std::vector<float>& input, std::vector<float>& out_dir_logits, float& out_jump, float& out_hook, float& out_fire, float& out_aim_x, float& out_aim_y)
{
    m_L1.Forward(input, false, 0); // LeakyReLU
    m_L2.Forward(m_L1.m_OutputsCache, false, 0);
    m_L3.Forward(m_L2.m_OutputsCache, false, 0);

    m_HeadDirection.Forward(m_L3.m_OutputsCache, true, 3); // Linear logits for Softmax
    m_HeadJump.Forward(m_L3.m_OutputsCache, true, 1);      // Sigmoid [0, 1]
    m_HeadHook.Forward(m_L3.m_OutputsCache, true, 1);      // Sigmoid [0, 1]
    m_HeadFire.Forward(m_L3.m_OutputsCache, true, 1);      // Sigmoid [0, 1]
    m_HeadAim.Forward(m_L3.m_OutputsCache, true, 2);       // Tanh [-1, 1]

    out_dir_logits = m_HeadDirection.m_OutputsCache;
    out_jump = m_HeadJump.m_OutputsCache[0];
    out_hook = m_HeadHook.m_OutputsCache[0];
    out_fire = m_HeadFire.m_OutputsCache[0];
    out_aim_x = m_HeadAim.m_OutputsCache[0];
    out_aim_y = m_HeadAim.m_OutputsCache[1];
}

AIAction AINeuralNetwork::Predict(const AIFeatures& state, float temperature)
{
    std::vector<float> input = state.ToVector();
    std::vector<float> dir_logits;
    float prob_jump = 0.0f;
    float prob_hook = 0.0f;
    float prob_fire = 0.0f;
    float aim_x = 0.0f;
    float aim_y = 0.0f;

    Forward(input, dir_logits, prob_jump, prob_hook, prob_fire, aim_x, aim_y);

    if(temperature > 0.01f && temperature != 1.0f)
    {
        for(float& logit : dir_logits)
            logit /= temperature;
    }

    std::vector<float> dir_probs = Math::Softmax(dir_logits);

    AIAction action;
    action.m_ProbDirLeft = dir_probs[0];
    action.m_ProbDirNone = dir_probs[1];
    action.m_ProbDirRight = dir_probs[2];
    action.m_ProbJump = prob_jump;
    action.m_ProbHook = prob_hook;
    action.m_ProbFire = prob_fire;

    // Argmax or threshold decision
    int max_dir_idx = 1;
    float max_dir_prob = dir_probs[1];
    if(dir_probs[0] > max_dir_prob) { max_dir_idx = 0; max_dir_prob = dir_probs[0]; }
    if(dir_probs[2] > max_dir_prob) { max_dir_idx = 2; max_dir_prob = dir_probs[2]; }

    if(max_dir_idx == 0) action.m_Direction = -1;
    else if(max_dir_idx == 2) action.m_Direction = 1;
    else action.m_Direction = 0;

    action.m_Jump = prob_jump > 0.5f;
    action.m_Hook = prob_hook > 0.5f;
    action.m_Fire = prob_fire > 0.5f;

    // Scale aim vector back to game units (e.g. 256.0f)
    action.m_TargetX = aim_x * 256.0f;
    action.m_TargetY = aim_y * 256.0f;

    if(std::abs(action.m_TargetX) < 1.0f && std::abs(action.m_TargetY) < 1.0f)
    {
        action.m_TargetX = (action.m_Direction != 0) ? (float)action.m_Direction * 100.0f : 100.0f;
        action.m_TargetY = 0.0f;
    }

    action.m_Confidence = max_dir_prob;
    return action;
}

float AINeuralNetwork::TrainSample(const AITransition& sample, float learning_rate)
{
    std::vector<AITransition> batch = { sample };
    return TrainBatch(batch, learning_rate);
}

float AINeuralNetwork::TrainBatch(const std::vector<AITransition>& batch, float learning_rate)
{
    if(batch.empty()) return 0.0f;

    float total_loss = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 1e-4f;

    m_TrainStep++;

    for(const auto& sample : batch)
    {
        std::vector<float> input = sample.m_State.ToVector();
        std::vector<float> dir_logits;
        float prob_jump = 0.0f;
        float prob_hook = 0.0f;
        float prob_fire = 0.0f;
        float aim_x = 0.0f;
        float aim_y = 0.0f;

        Forward(input, dir_logits, prob_jump, prob_hook, prob_fire, aim_x, aim_y);

        std::vector<float> dir_probs = Math::Softmax(dir_logits);

        // 1. Direction Target One-Hot: [Left, None, Right]
        int target_dir_idx = 1;
        if(sample.m_Action.m_Direction < 0) target_dir_idx = 0;
        else if(sample.m_Action.m_Direction > 0) target_dir_idx = 2;

        std::vector<float> grad_dir(3, 0.0f);
        for(int i = 0; i < 3; ++i)
        {
            float target = (i == target_dir_idx) ? 1.0f : 0.0f;
            grad_dir[i] = (dir_probs[i] - target) * sample.m_Weight; // Softmax Cross-Entropy gradient
        }
        float dir_loss = -std::log(std::max(1e-7f, dir_probs[target_dir_idx]));

        // 2. Jump Binary Cross-Entropy
        float target_jump = sample.m_Action.m_Jump ? 1.0f : 0.0f;
        float grad_jump_val = (prob_jump - target_jump) * sample.m_Weight;
        std::vector<float> grad_jump = { grad_jump_val };
        float jump_loss = -(target_jump * std::log(std::max(1e-7f, prob_jump)) + (1.0f - target_jump) * std::log(std::max(1e-7f, 1.0f - prob_jump)));

        // 3. Hook Binary Cross-Entropy
        float target_hook = sample.m_Action.m_Hook ? 1.0f : 0.0f;
        float grad_hook_val = (prob_hook - target_hook) * sample.m_Weight;
        std::vector<float> grad_hook = { grad_hook_val };
        float hook_loss = -(target_hook * std::log(std::max(1e-7f, prob_hook)) + (1.0f - target_hook) * std::log(std::max(1e-7f, 1.0f - prob_hook)));

        // 4. Fire Binary Cross-Entropy
        float target_fire = sample.m_Action.m_Fire ? 1.0f : 0.0f;
        float grad_fire_val = (prob_fire - target_fire) * sample.m_Weight;
        std::vector<float> grad_fire = { grad_fire_val };
        float fire_loss = -(target_fire * std::log(std::max(1e-7f, prob_fire)) + (1.0f - target_fire) * std::log(std::max(1e-7f, 1.0f - prob_fire)));

        // 5. Aim MSE Loss
        float norm_target_x = Math::Clamp(sample.m_Action.m_TargetX / 256.0f, -1.0f, 1.0f);
        float norm_target_y = Math::Clamp(sample.m_Action.m_TargetY / 256.0f, -1.0f, 1.0f);
        float diff_aim_x = (aim_x - norm_target_x) * sample.m_Weight;
        float diff_aim_y = (aim_y - norm_target_y) * sample.m_Weight;
        std::vector<float> grad_aim = { diff_aim_x, diff_aim_y };
        float aim_loss = 0.5f * (diff_aim_x * diff_aim_x + diff_aim_y * diff_aim_y);

        total_loss += (dir_loss + 1.2f * jump_loss + 1.5f * hook_loss + 0.8f * fire_loss + 0.5f * aim_loss);

        // Backprop through Heads
        std::vector<float> grad_l3_dir;
        std::vector<float> grad_l3_jump;
        std::vector<float> grad_l3_hook;
        std::vector<float> grad_l3_fire;
        std::vector<float> grad_l3_aim;

        m_HeadDirection.Backward(grad_dir, grad_l3_dir, 3); // Linear output for direction head
        m_HeadJump.Backward(grad_jump, grad_l3_jump, 1);      // Sigmoid
        m_HeadHook.Backward(grad_hook, grad_l3_hook, 1);      // Sigmoid
        m_HeadFire.Backward(grad_fire, grad_l3_fire, 1);      // Sigmoid
        m_HeadAim.Backward(grad_aim, grad_l3_aim, 2);         // Tanh

        // Combine trunk gradients
        std::vector<float> grad_l3(m_H3Dim, 0.0f);
        for(size_t i = 0; i < m_H3Dim; ++i)
        {
            grad_l3[i] = grad_l3_dir[i] + grad_l3_jump[i] + grad_l3_hook[i] + grad_l3_fire[i] + grad_l3_aim[i];
        }

        // Backprop through Shared Trunk
        std::vector<float> grad_l2;
        m_L3.Backward(grad_l3, grad_l2, 0); // LeakyReLU

        std::vector<float> grad_l1;
        m_L2.Backward(grad_l2, grad_l1, 0); // LeakyReLU

        std::vector<float> grad_in;
        m_L1.Backward(grad_l1, grad_in, 0); // LeakyReLU
    }

    // Apply Adam Parameter Updates
    m_L1.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);
    m_L2.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);
    m_L3.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);

    m_HeadDirection.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);
    m_HeadJump.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);
    m_HeadHook.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);
    m_HeadFire.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);
    m_HeadAim.UpdateAdam(learning_rate, beta1, beta2, eps, weight_decay, m_TrainStep);

    return total_loss / (float)batch.size();
}

static void SaveLayerBinary(std::ofstream& out, const DenseLayer& layer)
{
    uint32_t in_dim = (uint32_t)layer.m_InDim;
    uint32_t out_dim = (uint32_t)layer.m_OutDim;
    out.write((const char*)&in_dim, sizeof(in_dim));
    out.write((const char*)&out_dim, sizeof(out_dim));
    out.write((const char*)layer.m_Weights.data(), layer.m_Weights.size() * sizeof(float));
    out.write((const char*)layer.m_Biases.data(), layer.m_Biases.size() * sizeof(float));
}

static bool LoadLayerBinary(std::ifstream& in, DenseLayer& layer)
{
    uint32_t in_dim = 0, out_dim = 0;
    in.read((char*)&in_dim, sizeof(in_dim));
    in.read((char*)&out_dim, sizeof(out_dim));
    if(!in.good() || in_dim != layer.m_InDim || out_dim != layer.m_OutDim)
        return false;
    in.read((char*)layer.m_Weights.data(), layer.m_Weights.size() * sizeof(float));
    in.read((char*)layer.m_Biases.data(), layer.m_Biases.size() * sizeof(float));
    return in.good();
}

bool AINeuralNetwork::SaveBinary(const std::string& filepath) const
{
    std::ofstream out(filepath, std::ios::binary);
    if(!out.is_open()) return false;

    char magic[4] = {'D', 'D', 'A', 'I'};
    uint32_t version = 1;
    out.write(magic, 4);
    out.write((const char*)&version, sizeof(version));
    out.write((const char*)&m_TrainStep, sizeof(m_TrainStep));

    SaveLayerBinary(out, m_L1);
    SaveLayerBinary(out, m_L2);
    SaveLayerBinary(out, m_L3);
    SaveLayerBinary(out, m_HeadDirection);
    SaveLayerBinary(out, m_HeadJump);
    SaveLayerBinary(out, m_HeadHook);
    SaveLayerBinary(out, m_HeadFire);
    SaveLayerBinary(out, m_HeadAim);

    return out.good();
}

bool AINeuralNetwork::LoadBinary(const std::string& filepath)
{
    std::ifstream in(filepath, std::ios::binary);
    if(!in.is_open()) return false;

    char magic[4] = {0};
    in.read(magic, 4);
    if(magic[0] != 'D' || magic[1] != 'D' || magic[2] != 'A' || magic[3] != 'I')
        return false;

    uint32_t version = 0;
    in.read((char*)&version, sizeof(version));
    in.read((char*)&m_TrainStep, sizeof(m_TrainStep));

    if(!LoadLayerBinary(in, m_L1)) return false;
    if(!LoadLayerBinary(in, m_L2)) return false;
    if(!LoadLayerBinary(in, m_L3)) return false;
    if(!LoadLayerBinary(in, m_HeadDirection)) return false;
    if(!LoadLayerBinary(in, m_HeadJump)) return false;
    if(!LoadLayerBinary(in, m_HeadHook)) return false;
    if(!LoadLayerBinary(in, m_HeadFire)) return false;
    if(!LoadLayerBinary(in, m_HeadAim)) return false;

    return true;
}

static void SaveLayerJson(std::ostream& out, const std::string& name, const DenseLayer& layer)
{
    out << "  \"" << name << "\": {\n";
    out << "    \"in_dim\": " << layer.m_InDim << ",\n";
    out << "    \"out_dim\": " << layer.m_OutDim << ",\n";
    out << "    \"weights\": [";
    for(size_t i = 0; i < layer.m_Weights.size(); ++i)
    {
        out << layer.m_Weights[i] << (i + 1 < layer.m_Weights.size() ? "," : "");
    }
    out << "],\n";
    out << "    \"biases\": [";
    for(size_t i = 0; i < layer.m_Biases.size(); ++i)
    {
        out << layer.m_Biases[i] << (i + 1 < layer.m_Biases.size() ? "," : "");
    }
    out << "]\n  }";
}

bool AINeuralNetwork::SaveJson(const std::string& filepath) const
{
    std::ofstream out(filepath);
    if(!out.is_open()) return false;

    out << "{\n";
    out << "  \"model_type\": \"DDNet_DeepTee_MultiHead_MLP\",\n";
    out << "  \"version\": 1,\n";
    out << "  \"train_step\": " << m_TrainStep << ",\n";
    out << "  \"input_dim\": " << m_InDim << ",\n";
    SaveLayerJson(out, "layer1", m_L1); out << ",\n";
    SaveLayerJson(out, "layer2", m_L2); out << ",\n";
    SaveLayerJson(out, "layer3", m_L3); out << ",\n";
    SaveLayerJson(out, "head_direction", m_HeadDirection); out << ",\n";
    SaveLayerJson(out, "head_jump", m_HeadJump); out << ",\n";
    SaveLayerJson(out, "head_hook", m_HeadHook); out << ",\n";
    SaveLayerJson(out, "head_fire", m_HeadFire); out << ",\n";
    SaveLayerJson(out, "head_aim", m_HeadAim); out << "\n";
    out << "}\n";

    return out.good();
}

bool AINeuralNetwork::LoadJson(const std::string& filepath)
{
    // Binary is preferred for speed, but JSON parsing fallback
    std::ifstream in(filepath);
    if(!in.is_open()) return false;
    // JSON loading implementation if needed
    return true;
}

} // namespace AI

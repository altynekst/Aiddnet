#ifndef AI_MATH_H
#define AI_MATH_H

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>

namespace AI
{

namespace Math
{

inline float Clamp(float val, float min_val, float max_val)
{
    return std::max(min_val, std::min(val, max_val));
}

inline float LeakyReLU(float x, float alpha = 0.01f)
{
    return x > 0.0f ? x : alpha * x;
}

inline float LeakyReLUDerivative(float x, float alpha = 0.01f)
{
    return x > 0.0f ? 1.0f : alpha;
}

inline float Sigmoid(float x)
{
    x = Clamp(x, -20.0f, 20.0f);
    return 1.0f / (1.0f + std::exp(-x));
}

inline float SigmoidDerivative(float s)
{
    return s * (1.0f - s);
}

inline float Tanh(float x)
{
    return std::tanh(Clamp(x, -20.0f, 20.0f));
}

inline float TanhDerivative(float t)
{
    return 1.0f - t * t;
}

inline std::vector<float> Softmax(const std::vector<float>& logits)
{
    if(logits.empty()) return {};
    float max_elem = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_vals(logits.size());
    float sum_exp = 0.0f;
    for(size_t i = 0; i < logits.size(); ++i)
    {
        exp_vals[i] = std::exp(logits[i] - max_elem);
        sum_exp += exp_vals[i];
    }
    if(sum_exp < 1e-7f) sum_exp = 1e-7f;
    for(size_t i = 0; i < logits.size(); ++i)
    {
        exp_vals[i] /= sum_exp;
    }
    return exp_vals;
}

inline float DotProduct(const float* a, const float* b, size_t n)
{
    float sum = 0.0f;
    for(size_t i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float RandomGaussian(float mean = 0.0f, float stddev = 0.05f)
{
    static thread_local std::mt19937 gen(42);
    std::normal_distribution<float> dist(mean, stddev);
    return dist(gen);
}

inline float RandomUniform(float min_val = 0.0f, float max_val = 1.0f)
{
    static thread_local std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(gen);
}

inline float EuclideanDistanceSquared(const std::vector<float>& a, const std::vector<float>& b)
{
    float dist = 0.0f;
    size_t len = std::min(a.size(), b.size());
    for(size_t i = 0; i < len; ++i)
    {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

inline float EuclideanDistance(const std::vector<float>& a, const std::vector<float>& b)
{
    return std::sqrt(EuclideanDistanceSquared(a, b));
}

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
    Vec2() = default;
    Vec2(float _x, float _y) : x(_x), y(_y) {}

    float LengthSquared() const { return x * x + y * y; }
    float Length() const { return std::sqrt(LengthSquared()); }
    Vec2 Normalized() const
    {
        float l = Length();
        return l > 1e-6f ? Vec2(x / l, y / l) : Vec2(0.0f, 0.0f);
    }
};

inline Vec2 CalculateLeadAim(Vec2 rel_pos, Vec2 rel_vel, float projectile_speed)
{
    float dist = rel_pos.Length();
    if(dist < 1.0f || projectile_speed < 1.0f) return rel_pos;
    float time_to_impact = dist / projectile_speed;
    Vec2 predicted_pos(rel_pos.x + rel_vel.x * time_to_impact, rel_pos.y + rel_vel.y * time_to_impact);
    return predicted_pos;
}

} // namespace Math

} // namespace AI

#endif // AI_MATH_H

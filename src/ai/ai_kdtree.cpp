#include "ai_kdtree.h"
#include <queue>
#include <cmath>

namespace AI
{

AIKDTreeMemory::AIKDTreeMemory(size_t max_capacity) :
    m_MaxCapacity(max_capacity),
    m_pRoot(nullptr),
    m_IsDirty(true)
{
    m_Samples.reserve(std::min(max_capacity, (size_t)10000));
}

void AIKDTreeMemory::AddSample(const AITransition& sample)
{
    if(m_Samples.size() >= m_MaxCapacity)
    {
        // Replace random or oldest sample
        size_t idx = rand() % m_Samples.size();
        m_Samples[idx] = sample;
    }
    else
    {
        m_Samples.push_back(sample);
    }
    m_IsDirty = true;
}

void AIKDTreeMemory::BuildIndex()
{
    if(m_Samples.empty())
    {
        m_pRoot.reset();
        m_IsDirty = false;
        return;
    }

    m_pRoot = BuildRecursive(m_Samples, 0, 0, (int)m_Samples.size());
    m_IsDirty = false;
}

std::unique_ptr<KDNode> AIKDTreeMemory::BuildRecursive(std::vector<AITransition>& samples, int depth, int start, int end)
{
    if(start >= end) return nullptr;

    int axis = depth % FEATURE_DIM;
    int mid = start + (end - start) / 2;

    auto comparator = [axis](const AITransition& a, const AITransition& b) {
        auto va = a.m_State.ToVector();
        auto vb = b.m_State.ToVector();
        return va[axis] < vb[axis];
    };

    std::nth_element(samples.begin() + start, samples.begin() + mid, samples.begin() + end, comparator);

    auto node = std::make_unique<KDNode>(samples[mid], axis);
    node->m_pLeft = BuildRecursive(samples, depth + 1, start, mid);
    node->m_pRight = BuildRecursive(samples, depth + 1, mid + 1, end);

    return node;
}

void AIKDTreeMemory::SearchNearestRecursive(const KDNode* pNode, const std::vector<float>& target, size_t k, float max_dist_sq, std::vector<std::pair<float, const AITransition*>>& best_nodes) const
{
    if(!pNode) return;

    auto node_vec = pNode->m_Sample.m_State.ToVector();
    float dist_sq = Math::EuclideanDistanceSquared(node_vec, target);

    if(dist_sq <= max_dist_sq)
    {
        best_nodes.emplace_back(dist_sq, &pNode->m_Sample);
        std::sort(best_nodes.begin(), best_nodes.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if(best_nodes.size() > k)
        {
            best_nodes.pop_back();
            max_dist_sq = best_nodes.back().first;
        }
    }

    int axis = pNode->m_Axis;
    float diff = target[axis] - node_vec[axis];
    float diff_sq = diff * diff;

    const KDNode* first = diff <= 0.0f ? pNode->m_pLeft.get() : pNode->m_pRight.get();
    const KDNode* second = diff <= 0.0f ? pNode->m_pRight.get() : pNode->m_pLeft.get();

    SearchNearestRecursive(first, target, k, max_dist_sq, best_nodes);

    if(diff_sq < max_dist_sq || best_nodes.size() < k)
    {
        SearchNearestRecursive(second, target, k, max_dist_sq, best_nodes);
    }
}

std::vector<AITransition> AIKDTreeMemory::FindNearest(const AIFeatures& state, size_t k, float max_radius) const
{
    std::vector<AITransition> result;
    if(m_Samples.empty()) return result;

    if(m_IsDirty || !m_pRoot)
    {
        // Linear scan fallback if tree is dirty
        std::vector<float> target = state.ToVector();
        std::vector<std::pair<float, const AITransition*>> scored;
        scored.reserve(m_Samples.size());

        for(const auto& s : m_Samples)
        {
            float d = Math::EuclideanDistanceSquared(s.m_State.ToVector(), target);
            if(d <= max_radius * max_radius)
            {
                scored.emplace_back(d, &s);
            }
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        size_t count = std::min(k, scored.size());
        for(size_t i = 0; i < count; ++i)
            result.push_back(*scored[i].second);
        return result;
    }

    std::vector<float> target = state.ToVector();
    std::vector<std::pair<float, const AITransition*>> best_nodes;
    SearchNearestRecursive(m_pRoot.get(), target, k, max_radius * max_radius, best_nodes);

    for(const auto& p : best_nodes)
    {
        result.push_back(*p.second);
    }
    return result;
}

bool AIKDTreeMemory::PredictAction(const AIFeatures& state, AIAction& out_action, size_t k) const
{
    auto nearest = FindNearest(state, k, 1.8f);
    if(nearest.empty()) return false;

    float weight_sum = 0.0f;
    float dir_sum = 0.0f;
    float jump_votes = 0.0f;
    float hook_votes = 0.0f;
    float fire_votes = 0.0f;
    float aim_x_sum = 0.0f;
    float aim_y_sum = 0.0f;

    auto target_vec = state.ToVector();

    for(const auto& sample : nearest)
    {
        float d = Math::EuclideanDistance(sample.m_State.ToVector(), target_vec);
        float w = 1.0f / (d + 0.01f);
        weight_sum += w;

        dir_sum += (float)sample.m_Action.m_Direction * w;
        if(sample.m_Action.m_Jump) jump_votes += w;
        if(sample.m_Action.m_Hook) hook_votes += w;
        if(sample.m_Action.m_Fire) fire_votes += w;

        aim_x_sum += sample.m_Action.m_TargetX * w;
        aim_y_sum += sample.m_Action.m_TargetY * w;
    }

    if(weight_sum > 0.0f)
    {
        float avg_dir = dir_sum / weight_sum;
        if(avg_dir > 0.35f) out_action.m_Direction = 1;
        else if(avg_dir < -0.35f) out_action.m_Direction = -1;
        else out_action.m_Direction = 0;

        out_action.m_Jump = (jump_votes / weight_sum) > 0.45f;
        out_action.m_Hook = (hook_votes / weight_sum) > 0.45f;
        out_action.m_Fire = (fire_votes / weight_sum) > 0.45f;
        out_action.m_TargetX = aim_x_sum / weight_sum;
        out_action.m_TargetY = aim_y_sum / weight_sum;
        out_action.m_Confidence = 0.85f;
        return true;
    }

    return false;
}

} // namespace AI

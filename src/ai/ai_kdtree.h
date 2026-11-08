#ifndef AI_KDTREE_H
#define AI_KDTREE_H

#include "ai_types.h"
#include "ai_math.h"
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>

namespace AI
{

struct KDNode
{
    AITransition m_Sample;
    int m_Axis = 0;
    std::unique_ptr<KDNode> m_pLeft;
    std::unique_ptr<KDNode> m_pRight;

    KDNode(const AITransition& sample, int axis) :
        m_Sample(sample),
        m_Axis(axis)
    {}
};

class AIKDTreeMemory
{
public:
    AIKDTreeMemory(size_t max_capacity = 200000);

    // Add state-action transition into memory buffer
    void AddSample(const AITransition& sample);

    // Rebuild KD-Tree index for fast O(log N) spatial queries
    void BuildIndex();

    // Query k nearest human gameplay situations
    std::vector<AITransition> FindNearest(const AIFeatures& state, size_t k = 5, float max_radius = 2.0f) const;

    // Get blended action recommendation from closest human demonstrations
    bool PredictAction(const AIFeatures& state, AIAction& out_action, size_t k = 5) const;

    // Get buffer size
    size_t Size() const { return m_Samples.size(); }
    void Clear() { m_Samples.clear(); m_pRoot.reset(); }

    const std::vector<AITransition>& GetSamples() const { return m_Samples; }

private:
    size_t m_MaxCapacity = 200000;
    std::vector<AITransition> m_Samples;
    std::unique_ptr<KDNode> m_pRoot;
    bool m_IsDirty = true;

    std::unique_ptr<KDNode> BuildRecursive(std::vector<AITransition>& samples, int depth, int start, int end);
    void SearchNearestRecursive(const KDNode* pNode, const std::vector<float>& target, size_t k, float max_dist_sq, std::vector<std::pair<float, const AITransition*>>& best_nodes) const;
};

} // namespace AI

#endif // AI_KDTREE_H

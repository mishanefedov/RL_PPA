#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <memory>

struct VectorHasher {
    std::size_t operator()(const std::vector<float>& vec) const {
        std::size_t seed = vec.size();
        for (auto& i : vec) {
            seed ^= std::hash<float>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class Obstacle {
public:
    std::vector<float> pos, size;
    float velocity;
    std::vector<float> direction;
    float dt;
    std::vector<std::vector<float>> futurePositions;

    Obstacle(std::vector<float> pos, std::vector<float> size, float velocity, std::vector<float> direction, float dt)
        : pos(pos), size(size), velocity(velocity), direction(direction), dt(dt) {
        calculateFuturePositions();
    }

    void calculateFuturePositions() {
        futurePositions.clear();
        for (int i = 0; i < 20; ++i) {
            std::vector<float> newPos = {
                pos[0] + i * direction[0] * velocity * dt,
                pos[1] + i * direction[1] * velocity * dt,
                pos[2] + i * direction[2] * velocity * dt
            };
            futurePositions.push_back(newPos);
        }
    }

    bool collidesAtTimeStep(const std::vector<float> &gripperPos, int timeStep, float safetyMargin) const {
        if (timeStep >= futurePositions.size()) {
            return false;
        }
        const std::vector<float>& obsPos = futurePositions[timeStep];
        return (gripperPos[0] + safetyMargin > obsPos[0] - size[0]) &&
               (gripperPos[0] - safetyMargin < obsPos[0] + size[0]) &&
               (gripperPos[1] + safetyMargin > obsPos[1] - size[1]) &&
               (gripperPos[1] - safetyMargin < obsPos[1] + size[1]) &&
               (gripperPos[2] + safetyMargin > obsPos[2] - size[2]) &&
               (gripperPos[2] - safetyMargin < obsPos[2] + size[2]);
    }
};

class Obstacles {
public:
    std::vector<Obstacle> obstacles;

    Obstacles(const std::vector<Obstacle>& obstacles) : obstacles(obstacles) {}

    bool collidesAtTimeStep(const std::vector<float>& pos, int timeStep, float safetyMargin) const {
        for (const auto& obstacle : obstacles) {
            if (obstacle.collidesAtTimeStep(pos, timeStep, safetyMargin)) {
                return true;
            }
        }
        return false;
    }
};


class Node {
public:
    std::vector<float> pos;
    float cost, heuristic;
    std::shared_ptr<Node> parent;
    int depth;

    Node(const std::vector<float>& pos, std::shared_ptr<Node> parent = nullptr, int depth = 0)
        : pos(pos), cost(0), heuristic(0), parent(parent), depth(depth) {}

    float totalCost() const {
        return cost + heuristic;
    }
};


struct CompareNode {
    bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) const {
        return a->totalCost() > b->totalCost();
    }
};

class FindTrajectory {
public:
    std::vector<float> start, goal;
    std::vector<Obstacle> obstacles;
    std::vector<std::pair<float, float>> env_dimension;
    float stepSize, safetyMargin;
    bool gripperClosed;

    FindTrajectory(std::vector<float> start, 
                   std::vector<float> goal, 
                   std::vector<Obstacle> obstacles, 
                   std::vector<std::pair<float, float>> env_dimension,
                   float stepSize = 0.01, 
                   float safetyMargin = 0.045, 
                   bool gripperClosed = true)
    : start(start), goal(goal), obstacles(obstacles), env_dimension(env_dimension),
      stepSize(stepSize), safetyMargin(safetyMargin), gripperClosed(gripperClosed) {}

    std::vector<std::vector<float>> aStarSearch() {
        if (!isPositionInBound(goal)) {
            return {};
        }
        
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, CompareNode> openSet;
        std::unordered_map<std::vector<float>, float, VectorHasher> costSoFar;

        auto startNode = std::make_shared<Node>(start);
        startNode->heuristic = heuristic(start);
        openSet.push(startNode);
        costSoFar[start] = 0;

        while (!openSet.empty()) {
            auto current = openSet.top();
            openSet.pop();

            // if (current->pos == goal) {
            if (isNearestNodeToGoal(current.get()) || current->depth > 15) {
                std::vector<std::vector<float>> path;
                for (auto node = current; node != nullptr; node = node->parent) {
                    path.push_back(node->pos);
                }
                std::reverse(path.begin(), path.end());
                return trajectoryToActions(path);
            }

            for (const std::vector<float>& nextPos : getNeighbors(current->pos, current->depth)) {
                float newCost = costSoFar[current->pos] + stepSize;
                
                if (costSoFar.find(nextPos) == costSoFar.end() || newCost < costSoFar[nextPos]) {
                    costSoFar[nextPos] = newCost;
                    auto nextNode = std::make_shared<Node>(nextPos, current, current->depth + 1);
                    nextNode->cost = newCost;
                    nextNode->heuristic = heuristic(nextPos);
                    openSet.push(nextNode);
                }
            }
        }

        return {};
    }


private:
    std::vector<float> roundAction(std::vector<float> action, int n) {
        std::vector<float> roundedAction;
        float multiplier = std::pow(10.0, n);
        
        for (auto &value : action) {
            roundedAction.push_back(std::round(value * multiplier) / multiplier);
        }
        
        return roundedAction;
    }
    std::vector<std::vector<float>> trajectoryToActions(const std::vector<std::vector<float>>& path) {
        std::vector<std::vector<float>> actions;
        if (path.size() < 2) {
            return actions;
        }
        
        for (size_t i = 1; i < path.size(); ++i) {
            const auto& current = path[i - 1];
            const auto& next = path[i];
            std::vector<float> action(4);
            
            action[0] = (next[0] - current[0]) / stepSize;
            action[1] = (next[1] - current[1]) / stepSize;
            action[2] = (next[2] - current[2]) / stepSize;
            action[3] = gripperClosed ? 1.0f : -1.0f;
            actions.push_back(roundAction(action, 1));
        }
        
        return actions;
    }
    bool isNearestNodeToGoal(const Node* node) const {
        return euclideanDistance(node->pos, goal) < stepSize - 0.00;
    }

    float euclideanDistance(const std::vector<float>& pos1, const std::vector<float>& pos2) const {
        float distance = 0.0;
        for (size_t i = 0; i < pos1.size(); ++i) {
            distance += std::pow(pos1[i] - pos2[i], 2);
        }
        return std::sqrt(distance);
    }


    float heuristic(const std::vector<float>& pos) const {
        float h = 0.0;
        for (size_t i = 0; i < pos.size(); ++i) {
            h += std::abs(pos[i] - goal[i]);
        }
        return h;
    }


    std::vector<std::vector<float>> getNeighbors(const std::vector<float>& pos, const int depth = 0) const {
        std::vector<std::vector<float>> neighbors;
        for (float dx = -stepSize; dx <= stepSize; dx += stepSize) {
            for (float dy = -stepSize; dy <= stepSize; dy += stepSize) {
                for (float dz = -stepSize; dz <= stepSize; dz += stepSize) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    std::vector<float> newPos = {pos[0] + dx, pos[1] + dy, pos[2] + dz};
                    
                    if (isPositionValid(newPos, depth)) {
                        neighbors.push_back(newPos);
                    }
                }
            }
        }
        return neighbors;
    }

    bool isPositionValid(const std::vector<float>& pos, const int depth = 0) const {
        if(!isPositionInBound(pos)){
            return false;
        }
        for (const Obstacle& obstacle : obstacles) {
            if (obstacle.collidesAtTimeStep(pos, depth, safetyMargin)) return false;
        }
        return true;
    }
    bool isPositionInBound(const std::vector<float>& position) const {
        for(size_t i = 0; i < position.size(); ++i) {
            if(position[i] < env_dimension[i].first || position[i] > env_dimension[i].second) {
                return false;
            }
        }
        return true;
    }
};

int main() {
    return 0;
}

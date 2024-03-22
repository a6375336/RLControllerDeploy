#pragma once
/*
description: RL������
date: 20240126
author: BUAA-XMH
*/
#ifndef RLCONTROLLER_H
#define RLCONTROLLER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>

#include <FSM_States/ControlFSMData.h>

constexpr int DIM_CONTROLLER_INPUT = 36;
constexpr int DIM_CONTROLLER_OUTPUT = 8;

enum class ACTION_TYPE {
    POSITION = 1,
    VELOCITY,
    TORQUE
};

class RLController
{
public:
    RLController() {}
    RLController(const std::string& path);
    ~RLController();
    
    std::vector<float> GetTorqueCmd(/*36ά����, ������lastActions_(dim 8)*/
        std::vector<float> baseLinVer,       // dim 3
        std::vector<float> baseAngVer,       // dim 3
        std::vector<float> projectedGravity, // dim 3
        std::vector<float> commands,         // dim 3, x��yƽ����zת��
        std::vector<float> q,                // dim 8
        std::vector<float> qdot              // dim 8
    );

    
    void run(ControlFSMData<float>& data);


    std::vector<float> GetTorqueCmd(const std::vector<float>& inputs);
private:
    /*
    * RL ������ǰ�����
    * ���������
    */
    at::Tensor  GetActions(const std::vector<float>& inputs);

    at::Tensor  Forward(const std::vector<torch::jit::IValue>& inputs);

    void InputsPreTreatment(
        std::vector<float>& baseLinVer,       // dim 3 vx, vy, vz
        std::vector<float>& baseAngVer,       // dim 3 wx, wy, wz
        std::vector<float>& projectedGravity, // dim 3 
        std::vector<float>& commands,         // dim 3, x��yƽ����zת��
        std::vector<float>& q,                // dim 8 
        std::vector<float>& qdot              // dim 8
    );

    void QPreTreatmend(std::vector<float>& q) {
        for (int i = 0; i < q.size(); ++i) {
            q[i] -= qOffset[i];
            q[i] *= jointDir[i];
        }
    }
    
    void QdPreTreatmend(std::vector<float>& qd) {
        for (int i = 0; i < qd.size(); ++i) {
            qd[i] *= jointDir[i];
        }
    }

    inline at::Tensor Vec2Tensor(const std::vector<float>& vec) {
        return torch::tensor(vec);
    }
    
    inline std::vector<float> Tensor2Vec(const at::Tensor& tensor) {
        return std::vector<float> (tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    }

    std::vector<float> GetProjectedGravity(const std::vector<float>& quat);

    std::vector<float> Vec3ToStdVec(const Vec3<float>& vec3) {
        return std::vector<float>(vec3.data(), vec3.data() + 3);
    }

    void PrintVec(const std::vector<float>& vec, const std::string& str) {
        std::cout << str;
        for (int i = 0; i < vec.size(); ++i) {
            std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

    std::vector<float> SetupCommand(ControlFSMData<float> & data);

    void UpdateLegCMD(ControlFSMData<float> & data, const std::vector<float> torqueCmd);


private:
    torch::jit::script::Module module_;
    
    ACTION_TYPE actionType_ = ACTION_TYPE::POSITION;

    std::vector<float> lastActions_ = std::vector<float>(DIM_CONTROLLER_OUTPUT, 0.0);
    // ��ȷ�
    float actionScale_ = 0.5;
    at::Tensor pGains_ = torch::tensor({100, 100, 100, 100, 
        200, 200, 40, 40});
    at::Tensor dGains_ = torch::tensor({3.0, 3.0, 3.0, 3.0,
        6.0, 6.0, 1.0, 1.0});

    float linVerScales_ = 2.0;
    float angVerScales_ = 0.25;
    float qScales_ = 1.0;
    float qdotScales_ = 0.05;

    at::Tensor torqueLimits_ = torch::tensor({
        100, 100, 80,80, 100, 100, 30, 30});
    
    std::vector<float> jointDir = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0}; // 左腿，右腿

    std::vector<float> qOffset = {  -0.0027, -0.6726, 1.22377, -0.552595, 0.0027508, -0.672686, 1.22377, -0.552596}; // 左腿，右腿
};



#endif
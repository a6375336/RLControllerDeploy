#include "RLController.h"


RLController::RLController(const std::string& path)
{
    module_ = torch::jit::load(path);  
    
}

RLController::~RLController()
{
}

at::Tensor RLController::Forward(const std::vector<torch::jit::IValue>& inputs)
{
    auto temp = module_.forward(inputs);

    at::Tensor output = temp.toTensor();
    return output;
}

at::Tensor RLController::GetActions(const std::vector<float>& inputs)
{
    
    auto tensor = torch::tensor(inputs);

    std::vector<torch::jit::IValue> temp;
    temp.push_back(tensor);
    return Forward(temp);
}

std::vector<float> RLController::GetTorqueCmd(const std::vector<float>& inputs)
{
    
    auto actions = GetActions(inputs);
    
    lastActions_ = Tensor2Vec(actions);

    actions *= actionScale_;
    at::Tensor torques = torch::zeros(actions.sizes());

    std::vector<float> dofPosVec(8,0.0);
    std::copy(inputs.begin() + 12, inputs.begin() + 20, dofPosVec.begin());
    at::Tensor dofPos = Vec2Tensor(dofPosVec);
    

    std::vector<float> dofVelVec(8, 0.0);
    std::copy(inputs.begin() + 20, inputs.begin() + 28, dofVelVec.begin());
    at::Tensor dofVel = Vec2Tensor(dofVelVec);

    
    switch (actionType_) {
        case ACTION_TYPE::POSITION:
            torques = pGains_ * (actions - dofPos) - dGains_ * dofVel;
            break;
        // case ACTION_TYPE::VELOCITY:
            // torques = pGains_ * (actions - dofVel) - dGains_ * dofVel;
            break;
        case ACTION_TYPE::TORQUE:
            torques = actions;
            break;
        default:
            break;
    };
    torch::clip(torques, -torqueLimits_, torqueLimits_);
    std::vector<float> torqueCmd(torques.data_ptr<float>(), torques.data_ptr<float>() + torques.numel());
    
    return torqueCmd;
}

std::vector<float> RLController::GetTorqueCmd(/*36ά����, ������lastActions_(dim 8)*/
    std::vector<float> baseLinVer,              // dim 3 vx, vy, vz
    std::vector<float> baseAngVer,              // dim 3 wx, wy, wz
    std::vector<float> projectedGravity,        // dim 3 ��������ϵ�µ� ����������ϵ-z��������
    std::vector<float> commands,                // dim 3, x��yƽ����zת��
    std::vector<float> q,                       // dim 8 
    std::vector<float> qdot              // dim 8
)
{
    InputsPreTreatment(baseLinVer, baseAngVer, projectedGravity,
        commands, q, qdot);
    

    std::vector<float> inputs;
    inputs.insert(inputs.end(), baseLinVer.begin(), baseLinVer.end());
    inputs.insert(inputs.end(), baseAngVer.begin(), baseAngVer.end());
    inputs.insert(inputs.end(), projectedGravity.begin(), projectedGravity.end());
    inputs.insert(inputs.end(), commands.begin(), commands.end());
    inputs.insert(inputs.end(), q.begin(), q.end());
    inputs.insert(inputs.end(), qdot.begin(), qdot.end());
    inputs.insert(inputs.end(), lastActions_.begin(), lastActions_.end());

    if (inputs.size() != DIM_CONTROLLER_INPUT) {
        std::cerr << "[RLController][GetTorqueCmd] error when cpt torques in rl controller, inputs dim is not set properly"
            << std::endl;
        std::cout << "[RLController] inputs size: " << inputs.size()<< std::endl;
    }

    return GetTorqueCmd(inputs);
}

void RLController::InputsPreTreatment(
    std::vector<float>& baseLinVer,       // dim 3 vx, vy, vz
    std::vector<float>& baseAngVer,       // dim 3 wx, wy, wz
    std::vector<float>& projectedGravity, // dim 3 
    std::vector<float>& commands,         // dim 3, x��yƽ����zת��
    std::vector<float>& q,                // dim 8 
    std::vector<float>& qdot              // dim 8
)
{
    for (auto& c : baseLinVer) {
        c *= linVerScales_;
    }
    
    for (auto& c : baseAngVer) {
        c *= angVerScales_;
    }

    for (auto& c : q) {
        c *= qScales_;
    }

    for (auto& c : qdot) {
        c *= qdotScales_;
    }
    commands[0] *= linVerScales_;
    commands[1] *= linVerScales_;
    commands[2] *= angVerScales_;
}

std::vector<float> RLController::GetProjectedGravity(const std::vector<float>& quat)
{
    //float qw = quat.back();
    //at::Tensor qVec = torch::tensor({ quat[0],quat[1], quat[2]});
    //at::Tensor v = torch::tensor({ 0.0,0.0,-1.0 });
    //at::Tensor a = v * (2 * qw * 2 - 1.0);
    //a.unsqueeze(-1);
}

void RLController::run(ControlFSMData<float>& data)
{
    // std::cout << "[RLController] run start" << std::endl;
    //1, 状态估计
    auto& seResult = data._stateEstimator->getResult();
    // std::cout << "[RLController] _stateEstimator" << std::endl;
    //2, 获得指令
    auto cmd = SetupCommand(data);
    // std::cout << "[RLController] SetupCommand" << std::endl;
    //3，计算关节力矩

    /*关节角度 8*/
    std::vector<float> q_all;

    std::vector<float> q1(data._legController->datas[0].q.data(), data._legController->datas[0].q.data() + 4);
    std::vector<float> q2(data._legController->datas[1].q.data(), data._legController->datas[1].q.data() + 4); 
    q_all.insert(q_all.end(), q2.begin(), q2.end());
    q_all.insert(q_all.end(), q1.begin(), q1.end());
    QPreTreatmend(q_all);
    // std::cout << "[RLController] q" << std::endl;

    /*关节角速度 8*/
    std::vector<float> qd_all;
    std::vector<float> qd1(data._legController->datas[0].qd.data(), data._legController->datas[0].qd.data() + 4);
    std::vector<float> qd2(data._legController->datas[1].qd.data(), data._legController->datas[1].qd.data() + 4); 
    qd_all.insert(qd_all.end(), qd2.begin(), qd2.end());
    qd_all.insert(qd_all.end(), qd1.begin(), qd1.end());
    QdPreTreatmend(qd_all);

    // std::cout << "[RLController] qd" << std::endl;

    auto torqueCmd = GetTorqueCmd(
        Vec3ToStdVec(seResult.vBody),
        Vec3ToStdVec(seResult.omegaBody),
        Vec3ToStdVec(seResult.rBody * Vec3<float>(0.0, 0.0, -1.0)),
        cmd,
        q_all,/*关节角度*/
        qd_all/*关节角速度*/
    );

    // PrintVec(Vec3ToStdVec(seResult.vBody), "[vBody] ");
    // PrintVec(Vec3ToStdVec(seResult.omegaBody), "[wBody] ");
    // PrintVec(Vec3ToStdVec(seResult.rBody * Vec3<float>(0.0, 0.0, -1.0)), "[GraPro] ");
    // PrintVec(cmd, "[cmd] ");
    // PrintVec(q_all, "[q_all] ");
    // PrintVec(qd_all, "[qd_all] ");

    // std::vector<float> torqueCmd(8, 0.0);
    //  torqueCmd[0] = 4.0;
    // // torqueCmd[6] = 10.0;
    
    //<To do>: 3，输出力矩
    UpdateLegCMD(data, torqueCmd);
}

std::vector<float> RLController::SetupCommand(ControlFSMData<float> & data) { //质心速度指令
    
    static float x_vel_cmd = 0;
    static float y_vel_cmd = 0;
    float filter(0.0005);               //注意这个滤波器,决定了行走的加速度,很重要
  
    float scale_x = 0.300;
    float scale_y = 0.300;
    float scale_z = 0.300;

    float gaitTime = 50 * 0.01 ; // gait->getCurrentGaitTime(dtMPC);

    float padX      =  data._gamepad->get().leftStickAnalog[1]
                   *scale_x / gaitTime;
    float padY      = -data._gamepad->get().leftStickAnalog[0]
                   *scale_y / gaitTime;
    static float yaw_turn_rate = (data._gamepad->get().leftTriggerAnalog - data._gamepad->get().rightTriggerAnalog)
                   *scale_z / gaitTime; //偏航角速度

    //x,y 方向水平速度
    x_vel_cmd = x_vel_cmd * (1.0 - filter) + padX * filter;
    y_vel_cmd = y_vel_cmd * (1.0 - filter) + padY * filter;
  
    // //偏航角
    // _yaw_des = data._stateEstimator->getResult().rpy[2] + dt * _yaw_turn_rate;
    // //机身高度
    // _body_height = data.userParameters->trot_height;

    return std::vector<float>{x_vel_cmd, y_vel_cmd, yaw_turn_rate};
}

void RLController::UpdateLegCMD(ControlFSMData<float> & data, const std::vector<float> torqueCmd) {
  LegControllerCommand<float> * cmd = data._legController->commands;
  
  //关节力矩前馈, 关节期望位置, 关节期望速度, 关节p, 关节d

  // 左腿
  int leg = 1;
  for (int i = 0; i < 4; ++i) {
    cmd[leg].tauFeedForward[i] = torqueCmd[i ] * jointDir[i ];
  }
  // 右腿
  leg = 0;
  for (int i = 0; i < 4; ++i) {
    cmd[leg].tauFeedForward[i] = torqueCmd[4 + i ] * jointDir[4 + i ];
  }
}
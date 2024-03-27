import time

import pybullet
import pybullet_data
import torch
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt # pic
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import sys
sys.path.append(r'G:\Project\RL_code\RL_Deploy\RLControllerDeploy-main\RLControllerDeploy-main\rsl_rl')
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

p_gains = torch.Tensor([100, 100, 200, 100, 100, 100, 200, 100])
d_gains = torch.Tensor([3.0, 3.0, 6.0, 3.0, 3.0, 3.0, 6.0, 3.0])
torques_limit = torch.Tensor([100, 100, 150, 30, 100, 100, 150, 30])
actions_limit = torch.Tensor([100,100,100,100,100,100,100,100])
joint_damping = torch.Tensor([3.0, 3.0, 6.0, 3.0, 3.0, 3.0, 6.0, 3.0])

def Cpt_torques(actions, q, qd):
    torques = ( p_gains * (actions - q) - d_gains * qd)
    # torques = torch.Tensor([0,0,0,0,100, -100,100 ,100])

    torques = torch.clip(torques, -torques_limit, torques_limit)
    return torques
    
def ForceControl(r_ind, numJoints, actions, q, qd):
    torques = Cpt_torques(actions, q, qd)
    # print("torques {}".format(torques))
    
    for jointIdx in range(numJoints):
        pybullet.setJointMotorControl2(
            bodyIndex=r_ind,
            jointIndex=jointIdx,
            controlMode=pybullet.TORQUE_CONTROL,#TORQUE_CONTROL, # POSITION_CONTROL, VELOCITY_CONTROL
            force=torques[jointIdx]
        )
    return torques
    
# def PosControl(r_ind, numJoints, actions, defaultAngle):
#     temp = torch.Tensor([0, 0,0,10,0,0,0,0])
#     for jointIdx in range(numJoints):
#         pybullet.setJointMotorControl2(
#             bodyIndex=r_ind,
#             jointIndex=jointIdx,
#             controlMode=pybullet.VELOCITY_CONTROL,
#             targetPosition= temp[jointIdx],#actions[jointIdx] + defaultAngle[jointIdx], # actions[jointIdx] + defaultAngle[jointIdx],
#             force=torques_limit[jointIdx],
#             maxVelocity = 30
#         )

def GetJointState(r_ind, num_joints, defaultAngle):
    q = []
    qd = []
    # 获取关节角度和角速度
    joint_states = pybullet.getJointStates(r_ind, range(num_joints))
    for i, state in enumerate(joint_states):
        joint_position, joint_velocity, _, _ = state
        q.append(joint_position - defaultAngle[i])
        qd.append(joint_velocity)
    q_torch = torch.tensor(q)
    # print("q_torch {}".format(q_torch))
    qd_torch = torch.tensor(qd)
    # qd_torch *= 0.05
    # print("qd_torch {}".format(qd_torch))
    return q_torch, qd_torch

def GetObservation(r_ind, num_joints, defaultAngle, lastAction, timeStep):
    
    linear_velocity, angular_velocity = pybullet.getBaseVelocity(r_ind)
    ang_vel = torch.Tensor([angular_velocity[0], angular_velocity[1], angular_velocity[2]])
    ang_vel *= 0.25
    # print("ang_vel {}".format(ang_vel))

    position, orientation = pybullet.getBasePositionAndOrientation(r_ind)
    base_rotation_matrix = np.array(pybullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    projected_grativity = np.dot(np.linalg.inv(base_rotation_matrix), np.array([0.0, 0.0, -1.0]))
    projected_grativity_torch = torch.Tensor([projected_grativity[0], projected_grativity[1], projected_grativity[2]])
    # print("projected_grativity_torch {}".format(projected_grativity_torch))
    
    vcmd = torch.Tensor([0., 0.0, 0.0])
    # print("vcmd {}".format(ang_vel))
    
    q_torch, qd_torch = GetJointState(r_ind, num_joints, defaultAngle)
    qd_torch *= 0.05
    
    res = torch.cat((ang_vel,projected_grativity_torch,vcmd,q_torch,qd_torch, lastAction), dim=-1)
    
    return res

def PlotSubFunc(picHandleArray, xData, yData, title, xlabel, ylabel):
    for i in range(len(picHandleArray)):
        plt.sca(picHandleArray[i])
        plt.plot(xData, yData[:, i].detach().numpy())
        plt.title("{} {}".format(title, i))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

def PlotObs(obsTimeAxis, obsRecd):
    obsRecd = torch.reshape(obsRecd, (len(obsTimeAxis), -1))
    plt.figure()
    angPic = [plt.subplot(7,3,i + 1) for i in range(3)]
    graPic = [plt.subplot(7,3,i + 4) for i in range(3)]
    vcmdPic = [plt.subplot(7,3,i + 7) for i in range(3)]
    qPic = [plt.subplot(7,4,i + 13) for i in range(8)]
    qdPic = [plt.subplot(7,4,i + 21) for i in range(8)]
    
    # 角速度
    for i in range(3):
        plt.sca(angPic[i])
        plt.plot(obsTimeAxis, obsRecd[:, i].detach().numpy())
        plt.title("angular vel {}".format(i))
        plt.xlabel("time")
        plt.ylabel("rad/s")
        plt.grid(True)
    
    # 重力投影
    for i in range(3):
        plt.sca(graPic[i])
        plt.plot(obsTimeAxis, obsRecd[:, i + 3].detach().numpy())
        plt.title("projected gravity {}".format(i))
        plt.xlabel("time")
        plt.ylabel("m")
        plt.grid(True)
    
    # vcmd
    for i in range(3):
        plt.sca(vcmdPic[i])
        plt.plot(obsTimeAxis, obsRecd[:, i + 6].detach().numpy())
        plt.title("vcmd {}".format(i))
        plt.xlabel("time")
        plt.ylabel("m/s")
        plt.grid(True)
    
    # q
    for i in range(8):
        plt.sca(qPic[i])
        plt.plot(obsTimeAxis, obsRecd[:, i + 9].detach().numpy())
        plt.title("q {}".format(i))
        plt.xlabel("time")
        plt.ylabel("rad")
        plt.grid(True)
    
    # qd
    for i in range(8):
        plt.sca(qdPic[i])
        plt.plot(obsTimeAxis, obsRecd[:, i + 17].detach().numpy())
        plt.title("qd {}".format(i))
        plt.xlabel("time")
        plt.ylabel("rad/s")
        plt.grid(True)
    
def PlotActions(obsTimeAxis, actionsRecd):
    actionsRecd = torch.reshape(actionsRecd, (len(obsTimeAxis), -1))
    plt.figure()
    subPic = [plt.subplot(2,4,i+1) for i in range(8)]
    
    for i in range(8):
        plt.sca(subPic[i])
        plt.plot(obsTimeAxis, actionsRecd[:, i].detach().numpy())
        plt.title("action {}".format(i+1))
        plt.xlabel("time")
        plt.ylabel("rad")
        plt.grid(True)
    
    
def PlotTorques(torquesTimeAxis, torquesRecd):
    torquesRecd = torch.reshape(torquesRecd, (len(torquesTimeAxis), -1))
    plt.figure()
    subPic = [plt.subplot(2,4,i+1) for i in range(8)]
    
    for i in range(8):
        plt.sca(subPic[i])
        plt.plot(torquesTimeAxis, torquesRecd[:, i].detach().numpy())
        plt.title("torque {}".format(i+1))
        plt.xlabel("time")
        plt.ylabel("N*m")
        plt.grid(True)

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

if __name__ == '__main__':
    client = pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setPhysicsEngineParameter(numSolverIterations=10)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0)
    pybullet.setGravity(0, 0, -9.81)

    # 载入urdf格式是场景
    pybullet.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    # 载入urdf格式的机器人
    startPos = [0, 0, 0.9]
    startOri = pybullet.getQuaternionFromEuler([0,0,0])
    r_ind = pybullet.loadURDF(
        r'G:\Project\RL_code\RL_Deploy\RLControllerDeploy-main\RLControllerDeploy-main\Sim2SimOnBullet\biped_robot_240313b\urdf\biped_robot_240313b.urdf',
        startPos, startOri, useFixedBase=False)
    # dof 初始化  
    # 获取关节数量
    numJoints = pybullet.getNumJoints(r_ind)
    # 初始位置   
    defaultAngle = [-0.09, -0.66, 1.30, -0.6, 0.09, -0.66, 1.30, -0.61]
    for jointNumber in range(numJoints):
        pybullet.resetJointState(r_ind, jointNumber, defaultAngle[jointNumber])
        # pybullet.changeDynamics(r_ind, jointNumber, angularDamping=joint_damping[jointNumber])

# 循环步进仿真
    # 导入智能体模型
    # model = torch.jit.load(r'G:\Project\RL_code\RL_Deploy\RLControllerDeploy-main\RLControllerDeploy-main\Sim2SimOnBullet\policy_0327.pt') #policy_0325 
    
    # runner = OnPolicyRunner(env=?,
    #                         train_cfg=?,
    #                         )
    # runner.load(r"G:\Project\RL_code\RL_Deploy\RLControllerDeploy-main\RLControllerDeploy-main\Sim2SimOnBullet\model_20000.pt")
    actor_critic = ActorCritic(33, 33, 8,
                               actor_hidden_dims=[512, 256, 128],
                                critic_hidden_dims=[512, 256, 128])
    loaded_dict = torch.load(r"G:\Project\RL_code\RL_Deploy\RLControllerDeploy-main\RLControllerDeploy-main\Sim2SimOnBullet\model_20000.pt",
                             map_location=torch.device('cpu'))
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    model = actor_critic.act_inference
    
    actions = torch.zeros(8)

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    timeStep = 0.005/4 # 1. / 240.
    pybullet.setRealTimeSimulation(0)
    pybullet.setTimeStep(timeStep)
    
    # 关闭位控、速度控制模式
    for i in range(numJoints):
        pybullet.setJointMotorControl2(
            bodyIndex=r_ind,
            jointIndex=i,
            controlMode=pybullet.VELOCITY_CONTROL,
            force=0
        )
        pybullet.setJointMotorControl2(
            bodyIndex=r_ind,
            jointIndex=i,
            controlMode=pybullet.POSITION_CONTROL,
            force=0
        )            
    
    last_lin_vel = [0,0,0]
    
    pidTimesPerStep = 4
    
    lastTimeSec = 5.0
    steps = lastTimeSec / (timeStep * pidTimesPerStep)
    
    obsTimeAxis = np.linspace(0.0, lastTimeSec, int(steps))
    torquesTimeAxis = np.linspace(0.0, lastTimeSec, int(pidTimesPerStep * steps))
    obsRecd = torch.Tensor()
    actionsRecd = torch.Tensor()
    torquesRecd = torch.Tensor()
    
    for i in range(int(steps)):
    # while True:
        obs = GetObservation(r_ind, numJoints, defaultAngle, actions, timeStep)
        obs = torch.clip(obs, -100, 100)
        obsRecd = torch.cat((obsRecd, obs), dim=-1)
        
        actions = model(obs)
        actions = torch.clip(actions, -100, 100)
        actionsRecd = torch.cat((actionsRecd, actions), dim=-1)
        
        q_torch  = obs[9:17]
        qd_torch = obs[17:25]
        
        for _ in range(pidTimesPerStep):
            scaled_actions = actions * 0.5
            torques = ForceControl(r_ind, numJoints, scaled_actions, q_torch, qd_torch)
            torquesRecd = torch.cat((torquesRecd, torques), dim=-1)
            
            pybullet.stepSimulation()
            time.sleep(timeStep)
            q_torch, qd_torch = GetJointState(r_ind, 8, defaultAngle)
    
    
    PlotObs(obsTimeAxis, obsRecd)
    PlotActions(obsTimeAxis, actionsRecd)
    PlotTorques(torquesTimeAxis, torquesRecd)
    plt.show()

    pybullet.disconnect()    


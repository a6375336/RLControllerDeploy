import time

import pybullet
import pybullet_data
import torch
import numpy as np
from pprint import pprint

p_gains = torch.Tensor([25, 25, 50, 25, 25, 25, 50, 25])
d_gains = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
torques_limit = torch.Tensor([100, 100, 150, 30, 100, 100, 150, 30])
actions_limit = torch.Tensor([100,100,100,100,100,100,100,100])
joint_damping = torch.Tensor([3.0, 3.0, 6.0, 3.0, 3.0, 3.0, 6.0, 3.0])


def Cpt_torques(actions, q, qd):
    # torques = []
    # for i in range(len(actions)):
    #     torques.append(p_gains[i] * (actions[i] - q[i]) - d_gains[i] * qd[i])
    torques = ( p_gains * (actions - q) - d_gains * qd)
    print("actions: {}".format(actions))
    print("q: {}".format(q))
    # torques = 0.1 * torques
    torques = torch.clip(torques, -torques_limit, torques_limit)
    return torques
    
def ForceControl(r_ind, numJoints, actions, q, qd):
    torques = Cpt_torques(actions, q, qd)
    print("torques {}".format(torques))
    
    for jointIdx in range(numJoints):
        pybullet.setJointMotorControl2(
            bodyIndex=r_ind,
            jointIndex=jointIdx,
            controlMode=pybullet.TORQUE_CONTROL,#TORQUE_CONTROL, # POSITION_CONTROL, VELOCITY_CONTROL
            force=torques[jointIdx]
        )
    
def PosControl(r_ind, numJoints, actions, defaultAngle):
    for jointIdx in range(numJoints):
        pybullet.setJointMotorControl2(
            bodyIndex=r_ind,
            jointIndex=jointIdx,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition= actions[jointIdx] + defaultAngle[jointIdx], # actions[jointIdx] + defaultAngle[jointIdx],
            force=torques_limit[jointIdx],
            maxVelocity = 30
            # positionGain= 1 * p_gains[jointIdx],
            # velocityGain= 1 * d_gains[jointIdx]
        )

def GetObservation(r_ind, num_joints, defaultAngle, lastAction, timeStep, last_lin_vel):
    
    linear_velocity, angular_velocity = pybullet.getBaseVelocity(r_ind)
    ang_vel = torch.Tensor([angular_velocity[0], angular_velocity[1], angular_velocity[2]])
    ang_vel *= 0.25
    # print("ang_vel {}".format(ang_vel))

    cv = torch.Tensor(linear_velocity)
    lv = torch.Tensor(last_lin_vel)
    
    lin_acc = cv - lv
    lin_acc /= timeStep
    lin_acc*= 2
    # print("lin_acc {}".format(lin_acc))
    
    position, orientation = pybullet.getBasePositionAndOrientation(r_ind)
    base_rotation_matrix = np.array(pybullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    projected_grativity = np.dot(np.linalg.inv(base_rotation_matrix), np.array([0.0, 0.0, -1.0]))
    projected_grativity_torch = torch.Tensor([projected_grativity[0], projected_grativity[1], projected_grativity[2]])
    # print("projected_grativity_torch {}".format(projected_grativity_torch))
    
    vcmd = torch.Tensor([0.0, 0.0, 0.0])
    # print("vcmd {}".format(ang_vel))
    
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
    qd_torch *= 0.05
    # print("qd_torch {}".format(qd_torch))
    
    res = torch.cat((lin_acc, ang_vel,projected_grativity_torch,vcmd,q_torch,qd_torch, lastAction), dim=-1), linear_velocity
    
    return res

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
    startPos = [0, 0, 0.82]
    startOri = pybullet.getQuaternionFromEuler([0,0,0])
    r_ind = pybullet.loadURDF('/home/robot/testRLModel/RLControllerDeploy/Sim2SimOnBullet/biped_robot_240313b/urdf/biped_robot_240313b.urdf',
                        startPos, startOri)
    # dof 初始化  
    # 获取关节数量
    numJoints = pybullet.getNumJoints(r_ind)
    # 初始位置   
    defaultAngle = [-0.09, -0.66, 1.30, -0.6, 0.09, -0.66, 1.30, -0.61]
    for jointNumber in range(numJoints):
        pybullet.resetJointState(r_ind, jointNumber, defaultAngle[jointNumber])
        pybullet.changeDynamics(r_ind, jointNumber, angularDamping=joint_damping[jointNumber])

# 循环步进仿真
    
    
    
    model = torch.jit.load('/home/robot/testRLModel/RLControllerDeploy/Sim2SimOnBullet/policy_36_awgc_0322_11.pt')
    model.eval()
    # model = torch.jit.load('/home/robot/testRLModel/policy_1.pt')
    
    actions = torch.zeros(8)

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    timeStep = 0.005 # 1. / 240.
    # pybullet.setTimeStep(timeStep)
    pybullet.setRealTimeSimulation(0)
    
    last_lin_vel = [0,0,0]
    for i in range(500):
        pybullet.stepSimulation()
        
        obs, last_lin_vel = GetObservation(r_ind, numJoints, defaultAngle, actions, timeStep, last_lin_vel)
        # print("obs: {}".format(obs))
        
        actions = model(obs)
        actions = torch.clip(actions, -100, 100)
        scaled_actions = actions * 0.5
        # print(scaled_actions)
        # ForceControl(r_ind, numJoints, actions, obs[12:20], obs[20:28])
        PosControl(r_ind,numJoints, scaled_actions, defaultAngle)
        
        time.sleep(timeStep)

pybullet.disconnect()

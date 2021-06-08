function [logger] = getDigitFddData(logged_data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% addpath(genpath('C:\Users\mungam\Documents\GitHub\Digit_Controller\version\release_2021.02.11\model\FROST\Kinematics_Dynamics_Generation_Eva\gen'));
addpath(genpath('/home/exo/Documents/eva/Digit_Controller/version/release_2021.02.11/model/FROST/Kinematics_Dynamics_Generation_Eva/gen'));
addpath(genpath('/home/exo/Documents/eva/Digit_Controller/version/release_2021.02.11/model/FROST/Kinematics_Dynamics_Generation_Eva/utils_genKin'));

idx_joints = find(cellfun(@(v) any(strcmp(v,'q_{all}')),{logged_data.Name}));
idx_djoints= find(cellfun(@(v) any(strcmp(v,'q_{all}')),{logged_data.Name}));
idx_motorsCurrent = find(cellfun(@(v) any(strcmp(v,'u_{torque_{current}}')),{logged_data.Name}));
idx_motorsDesired = find(cellfun(@(v) any(strcmp(v,'u_{torque_{command}}')),{logged_data.Name}));

q_all=logged_data(idx_joints).Data;
dq_all=logged_data(idx_djoints).Data;
ua_all=logged_data(idx_motorsCurrent).Data;
ud_all=logged_data(idx_motorsDesired).Data;

t=logged_data(end).Data;
%recording when the force is applied and when it's not
force_t=t-5; %positive is with external force, negative is without ext force
[r,c]=find(force_t<0);
task(1:size(q_all,2))=ones(1,size(q_all,2));
task(1:c(end))=zeros(1,c(end)); %force applied for 1 entry and not applied for 0 entry

%joints in frontal plane
%note q is arranged in the following order
q_all_f=q_all([1,2,3,6,7,13,14,18,24,25],:);
dq_all_f=dq_all([1,2,3,6,7,13,14,18,24,25],:);
% q_all_s=q_all([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28],:);
% dq_all_s=dq_all([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28],:);

%motors in frontal plane
%note u is arranged in the following order
u_names={...
    'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftToeA','LeftToeB',...
    'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightToeA','RightToeB',...
    'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
    'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
ua_all_f=ua_all([1,5,6,7,11,12,13,17],:);
ud_all_f=ud_all([1,5,6,7,11,12,13,17],:);
% ua_all_s=ua_all([3,4,5,6,9,10,11,10,14,15,18,20],:);
% ud_all_s=ud_all([3,4,5,6,9,10,11,10,14,15,18,20],:);
%%
display('Starting to Calculate Angular Momentum and Relative Positions ...');
for i=1:size(q_all,2)
    q=q_all(:,i);
    dq=dq_all(:,i);
    p_com(:,i)=p_COM(q);
    Jp_com = Jp_COM(q);
    dJp_com = dJp_COM(q,dq);
    v_com(:,i) = Jp_com*dq;
    
    %right foot
    p_RF(:,i) = p_RightFoot(q);
    Jp_RF= Jp_RightFoot(q);
    dJp_RF= dJp_RightFoot(q,dq);
    v_RF(:,i) = Jp_RF*dq;
    rpy_RF(:,i)=rpy_RightFoot(q);
    rpy_b_RF(:,i)=rpyb_RightFoot(q);
    
    %Right Toe
    p_RT(:,i) = p_RightToe(q);
    Jp_RT= Jp_RightToe(q);
    dJp_RT = dJp_RightToe(q,dq);
    v_RT(:,i) = Jp_RT*dq;
    rpy_RT(:,i)=rpy_RightToe(q);
    rpy_b_RT(:,i)=rpyb_RightToe(q);
    
    %Right Heel
    p_RH(:,i) = p_RightHeel(q);
    Jp_RH= Jp_RightHeel(q);
    dJp_RH = dJp_RightHeel(q,dq);
    v_RH(:,i) = Jp_RH*dq;
    rpy_RH(:,i)=rpy_RightHeel(q);
    rpy_b_RH(:,i)=rpyb_RightHeel(q);
    
    %left foot
    p_LF(:,i) = p_LeftFoot(q);
    Jp_LF = Jp_LeftFoot(q);
    dJp_LF = dJp_LeftFoot(q,dq);
    v_LF(:,i) = Jp_LF*dq;
    rpy_LF(:,i)=rpy_LeftFoot(q);
    rpy_b_LF(:,i)=rpyb_LeftFoot(q);
    
    %left Toe
    p_LT(:,i) = p_LeftToe(q);
    Jp_LT = Jp_LeftToe(q);
    dJp_LT = dJp_LeftToe(q,dq);
    v_LT(:,i) = Jp_LT*dq;
    rpy_LT(:,i)=rpy_LeftToe(q);
    rpy_b_LT(:,i)=rpyb_LeftToe(q);
    
    %left Heel
    p_LH(:,i) = p_LeftHeel(q);
    Jp_LH = Jp_LeftHeel(q);
    dJp_LH = dJp_LeftHeel(q,dq);
    v_LH(:,i) = Jp_LH*dq;
    rpy_LH(:,i)=rpy_LeftHeel(q);
    rpy_b_LH(:,i)=rpyb_LeftHeel(q);
    
    
    %Angular momentum about the CoM
    LG(:,i)=getDigitAngularMomentum(p_com(:,i),[q;dq]);
    %Angular momentum about the feet
    L_LeftFoot(:,i)=getDigitAngularMomentum(p_LF(:,i),[q;dq]);
    L_RightFoot(:,i)=getDigitAngularMomentum(p_RF(:,i),[q;dq]);
    L_AverageFeet(:,i)=getDigitAngularMomentum(0.5*(p_RF(:,i)+p_LF(:,i)),[q;dq]);
    
    %Relative Height of the CoM w.r.t legs
    rp_COMFoot(:,i)=p_com(:,i)-0.5*(p_LF(:,i)-p_RF(:,i));
    
    %Position of the links
    P_torso=p_TorsoLink(q);
    P_leftHipRoll=p_LeftHipRollLink(q);
    P_leftHipYaw = p_leftHipYawLink(q);
    P_leftHipPitch = p_LeftHipPitchLink(q);
    P_leftKnee = p_LeftKneeLink(q);
    P_leftTarsus = p_LeftTarsusLink(q);
    P_leftToePitch = p_LeftToePitchLink(q);
    P_leftToeRoll = p_LeftToeRollLink(q);
    P_leftShoulderRoll = p_LeftShoulderRollLink(q);
    P_leftShoulderPitch = p_LeftShoulderPitchLink(q);
    P_leftShoulderYaw = p_LeftShoulderYawLink(q);
    P_leftElbow = p_LeftElbowLink(q);
    P_rightHipRoll = p_RightHipRollLink(q);
    P_rightHipYaw = p_RightHipYawLink(q);
    P_rightHipPitch = p_RightHipPitchLink(q);
    P_rightKnee = p_RightKneeLink(q);
    P_rightTarsus = p_RightTarsusLink(q);
    P_rightToePitch = p_RightToePitchLink(q);
    P_rightToeRoll = p_RightToeRollLink(q);
    P_rightShoulderRoll = p_RightShoulderRollLink(q);
    P_rightShoulderPitch = p_RightShoulderPitchLink(q);
    P_rightShoulderYaw = p_RightShoulderYawLink(q);
    P_rightElbow = p_RightElbowLink(q);
    
    %Saving the z position of links
    p_links(:,i)=[P_torso(3);...
                  P_leftHipRoll(3);...
                  P_leftHipYaw(3);...
                  P_leftHipPitch(3);...
                  P_leftKnee(3);...
                  P_leftTarsus(3);...
                  P_leftToePitch(3);...
                  P_leftToeRoll(3);...
                  P_leftShoulderRoll(3);...
                  P_leftShoulderPitch(3);...
                  P_leftShoulderYaw(3);...
                  P_leftElbow(3);...
                  P_rightHipRoll(3);...
                  P_rightHipYaw(3);...
                  P_rightHipPitch(3);...
                  P_rightKnee(3);...
                  P_rightTarsus(3);...
                  P_rightToePitch(3);...
                  P_rightToeRoll(3);...
                  P_rightShoulderRoll(3);...
                  P_rightShoulderPitch(3);...
                  P_rightShoulderYaw(3);...
                  P_rightElbow(3)];
end
display('Calculations done ...');
%% putting data for FDD
%q_all_s = all joints in saggital plane
%dq_all_s = derivative of joints in sagittal plane
%(ua_all_s-ud_all_s)= actual minus desired torque of all motors in sagittal
%plane
%LG = angular momentum about CoM
%L_LeftFoot= angular momentum about left foot
%L_RightFoot = angular momentum about right foot
%rp_COMFoot= relative position of the CoM wrt feet


q_names={'BaseX','BaseY','BaseZ','BaseYaw','BasePitch','BaseRoll',...
    'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftTarsus','LeftToePitch','LeftToeRoll',...
    'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
    'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightTarsus','RightToePitch','RightToeRoll',...
    'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
dq_names={'d_BaseX','d_BaseY','d_BaseZ','d_BaseYaw','d_BasePitch','d_BaseRoll',...
    'LeftHipRoll','d_LeftHipYaw','d_LeftHipPitch','d_LeftKnee','d_LeftTarsus','d_LeftToePitch','d_LeftToeRoll',...
    'LeftShoulderRoll','d_LeftShoulderPitch','d_LeftShoulderYaw','d_LeftElbow',...
    'RightHipRoll','d_RightHipYaw','d_RightHipPitch','d_RightKnee','d_RightTarsus','d_RightToePitch','d_RightToeRoll',...
    'RightShoulderRoll','d_RightShoulderPitch','d_RightShoulderYaw','d_RightElbow'};
u_names={...
    'u_LeftHipRoll','u_LeftHipYaw','u_LeftHipPitch','u_LeftKnee','u_LeftToeA','u_LeftToeB',...
    'RightHipRoll','u_RightHipYaw','u_RightHipPitch','u_RightKnee','u_RightToeA','u_RightToeB',...
    'LeftShoulderRoll','u_LeftShoulderPitch','u_LeftShoulderYaw','u_LeftElbow',...
    'RightShoulderRoll','u_RightShoulderPitch','u_RightShoulderYaw','u_RightElbow'};
LG_names={'LG_x', 'LG_y', 'LG_z'};
L_LeftFoot_names={'L_LeftFoot_x', 'L_LeftFoot_y', 'L_LeftFoot_z'};
L_RightFoot_names={'L_RightFoot_x', 'L_RightFoot_y', 'L_RightFoot_z'};
rp_COMFoot_names={'rp_COMFoot_x', 'rp_COMFoot_y', 'rp_COMFoot_z'};

feat={q_names,dq_names,u_names,LG_names,L_LeftFoot_names,L_RightFoot_names,rp_COMFoot_names,'task','time'};
display('Saving data into logger...')
logger.q_all=q_all;
logger.dq_all=dq_all;
logger.ua_all=ua_all;
logger.ud_all=ud_all;
logger.LG=LG;
logger.L_LeftFoot=L_LeftFoot;
logger.L_RightFoot=L_RightFoot;
logger.L_AverageFeet=L_AverageFeet;
logger.rp_COMFoot=rp_COMFoot;
logger.task=task;
logger.time=t;
logger.feat_names=feat;
logger.p_LeftFoot=p_LF;
logger.p_RightFoot=p_RF;
logger.p_links=p_links;
logger.p_com=p_com;
logger.rpy_LeftFoot=rpy_LF;
logger.rpy_RightFoot=rpy_RF;
logger.rpy_b_RightFoot=rpy_b_RF;
logger.rpy_b_LeftFoot=rpy_b_LF;
display('Saving data done...')
end


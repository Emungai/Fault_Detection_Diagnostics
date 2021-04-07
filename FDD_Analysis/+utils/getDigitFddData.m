function [FDD_info,FDD_info_analyze] = getDigitFddData(logged_data)
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

t=logged_data(1).Data;
%recording when the force is applied and when it's not
force_t=t-5; %positive is with external force, negative is without ext force
[r,c]=find(force_t<0);
task(1:size(q_all,2))=ones(1,size(q_all,2));
task(1:c(end))=zeros(1,c(end)); %force applied for 1 entry and not applied for 0 entry

%joints in frontal plane
%note q is arranged in the following order
% q_names={'BaseX','BaseY','BaseZ','BaseYaw','BasePitch','BaseRoll',...
%     'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftTarsus','LeftToePitch','LeftToeRoll',...
%     'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
%     'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightTarsus','RightToePitch','RightToeRoll',...
%     'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
q_all_f=q_all([1,2,3,6,7,13,14,18,24,25],:);
dq_all_f=dq_all([1,2,3,6,7,13,14,18,24,25],:);
% q_all_s=q_all([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28],:);
% dq_all_s=dq_all([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28],:);

%motors in frontal plane
%note u is arranged in the following order
%  u_names={...
%     'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftToeA','LeftToeB',...
%     'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightToeA','RightToeB',...
%     'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
%     'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
ua_all_f=ua_all([1,5,6,7,11,12,13,17],:);
ud_all_f=ud_all([1,5,6,7,11,12,13,17],:);
% ua_all_s=ua_all([3,4,5,6,9,10,11,10,14,15,18,20],:);
% ud_all_s=ud_all([3,4,5,6,9,10,11,10,14,15,18,20],:);
%% 
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
            
            p_RT(:,i) = p_RightToe(q);
            Jp_RT= Jp_RightToe(q);
            dJp_RT = dJp_RightToe(q,dq);
            v_RT(:,i) = Jp_RT*dq;
            
            p_RH(:,i) = p_RightHeel(q);
            Jp_RH= Jp_RightHeel(q);
            dJp_RH = dJp_RightHeel(q,dq);
            v_RH(:,i) = Jp_RH*dq;
         %left foot   
              p_LF(:,i) = p_LeftFoot(q);
            Jp_LF = Jp_LeftFoot(q);
            dJp_LF = dJp_LeftFoot(q,dq);
            v_LF(:,i) = Jp_LF*dq;
            
             p_LT(:,i) = p_LeftToe(q);
            Jp_LT = Jp_LeftToe(q);
            dJp_LT = dJp_LeftToe(q,dq);
            v_LT(:,i) = Jp_LT*dq;
            
            p_LH(:,i) = p_LeftHeel(q);
            Jp_LH = Jp_LeftHeel(q);
            dJp_LH = dJp_LeftHeel(q,dq);
            v_LH(:,i) = Jp_LH*dq;

%Angular momentum about the CoM
LG(:,i)=getDigitAngularMomentum(p_com(:,i),[q;dq]);
%Angular momentum about the feet
L_LeftFoot(:,i)=getDigitAngularMomentum(p_LF(:,i),[q;dq]);
L_RightFoot(:,i)=getDigitAngularMomentum(p_RF(:,i),[q;dq]);

%Relative Height of the CoM w.r.t legs
rp_COMFoot(:,i)=p_com(:,i)-0.5*(p_LF(:,i)-p_RF(:,i));
end
%% putting data for FDD
%q_all_s = all joints in saggital plane
%dq_all_s = derivative of joints in sagittal plane
%(ua_all_s-ud_all_s)= actual minus desired torque of all motors in sagittal
%plane
%LG = angular momentum about CoM
%L_LeftFoot= angular momentum about left foot
%L_RightFoot = angular momentum about right foot
%rp_COMFoot= relative position of the CoM wrt feet
FDD_info=[q_all_f',dq_all_f',(ua_all_f-ud_all_f)',LG',L_LeftFoot',L_RightFoot',rp_COMFoot',task',t'];
FDD_info_analyze=FDD_info(:,1:end-2);


end


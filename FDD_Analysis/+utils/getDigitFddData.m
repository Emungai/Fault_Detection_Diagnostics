function [outputArg1,outputArg2] = getDigitFddData(inputArg1,inputArg2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
addpath(genpath('C:\Users\mungam\Documents\GitHub\Digit_Controller\version\release_2021.02.11\model\FROST\Kinematics_Dynamics_Generation_Eva\gen'));
addpath(genpath('C:\Users\mungam\Documents\GitHub\Digit_Controller\version\release_2021.02.11\model\FROST\Kinematics_Dynamics_Generation_Eva\utils_genKin'));

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

%joints in sagittal plane
%note q is arranged in the following order
% q_names={'BaseX','BaseY','BaseZ','BaseYaw','BasePitch','BaseRoll',...
%     'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftTarsus','LeftToePitch','LeftToeRoll',...
%     'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
%     'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightTarsus','RightToePitch','RightToeRoll',...
%     'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};

q_all_s=q_all([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28],:);
dq_all_s=dq_all([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28],:);

%motors in sagittal plane
%note u is arranged in the following order
%  u_names={...
%     'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftToeA','LeftToeB',...
%     'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightToeA','RightToeB',...
%     'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
%     'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
ua_all_s=ua_all([3,4,5,6,9,10,11,10,14,15,18,20],:);
ud_all_s=ud_all([3,4,5,6,9,10,11,10,14,15,18,20],:);
%% 
for i=1:size(q_all,2)
    q=q_all(:,i);
    dq=dq_all(:,i);
p_com(:,i)=p_COM(q);
Jp_com(:,i) = Jp_COM(q);
dJp_com(:,i) = dJp_COM(q,dq);
v_com(:,i) = Jp_com*dq;
%right foot
            p_RF(:,i) = p_RightFoot(q);
            Jp_RF(:,i) = Jp_RightFoot(q);
            dJp_RF(:,i) = dJp_RightFoot(q,dq);
            v_RF(:,i) = Jp_RF*dq;
            
            p_RT(:,i) = p_RightToe(q);
            Jp_RT(:,i) = Jp_RightToe(q);
            dJp_RT(:,i) = dJp_RightToe(q,dq);
            v_RT(:,i) = Jp_RT*dq;
            
            p_RH(:,i) = p_RightHeel(q);
            Jp_RH(:,i) = Jp_RightHeel(q);
            dJp_RH(:,i) = dJp_RightHeel(q,dq);
            v_RH(:,i) = Jp_RH*dq;
         %left foot   
              p_LF(:,i) = p_LeftFoot(q);
            Jp_LF(:,i) = Jp_LeftFoot(q);
            dJp_LF(:,i) = dJp_LeftFoot(q,dq);
            v_LF(:,i) = Jp_LF*dq;
            
             p_LT(:,i) = p_LeftToe(q);
            Jp_LT(:,i) = Jp_LeftToe(q);
            dJp_LT(:,i) = dJp_LeftToe(q,dq);
            v_LT(:,i) = Jp_LT*dq;
            
            p_LH(:,i) = p_LeftHeel(q);
            Jp_LH(:,i) = Jp_LeftHeel(q);
            dJp_LH(:,i) = dJp_LeftHeel(q,dq);
            v_LH(:,i) = Jp_LH*dq;

%%	Angular momentum about the CoM
LG(:,i)=getDigitAngularMomentum(p_com,[q;dq]);
%% 	Angular momentum about the feet
L_LeftFoot(:,i)=getDigitAngularMomentum(p_LF,[q;dq]);
L_RightFoot(:,i)=getDigitAngularMomentum(p_RF,[q;dq]);

%% 	Relative Height of the CoM w.r.t legs
rp_COMFoot(:,i)=p_com-0.5*(p_LF-p_RF);
end
%% 	Actual torque minus desired torque (indirectly including y and dy)
u=u_a-u_d;
%% 	Global position of the feet (should be constant if it�s not then the feet are slipping- a way to estimate friction indirectly)

%% 	Feet rotation (I need to look more into this)

%% 	Global position of the CoM
%% 	Global position of the torso
%% 	Time (duration and initial time) external force is applied 

end

function [logged_data,types] = getDigitData(file)
%getDigitData Loads digit data from log file (courtesy of grantgib)
%   Detailed explanation goes here

%% Read in data from text file
disp("Reading in text file, converting to matrix...");

raw_data = readmatrix(file,...
    'Delimiter',',',...
    'Range',1,...
    'ConsecutiveDelimitersRule','join');
total_data = size(raw_data,1);

types = {...
    'time_{system}',...
    's_{time}',...
    'right_{stance}',...
    'q_{joints}', 'qdot_{joints}',...
    'q_{motors}', 'qdot_{motors}',...
    'q_{all}', 'qdot_{all}',...
    'q_{virts_{des}}', 'qdot_{virts_{des}}',...
    'q_{motors_{des}}', 'qdot_{motors_{des}}',...
    'u_{torque_{current}}', 'u_{torque_{command}}',...
    'torque_{limits}', 'damping_{limits}',...
    'time_{simulator}'};

total_types = length(types);

%% Organize into structured array
disp("Organizing data into structured array...");
logged_data = struct();
for i = 1:total_types
    idx_i = i:total_types:total_data;
    data_i = raw_data(idx_i,:)';
    data_i(~any(~isnan(data_i),2),:) = [];  % rem
    
    %     data_i(:,end) = []; % remove last column in case connection was disrupted
    logged_data(i).Name = types{i};
    logged_data(i).Data = data_i;
end

% equal num columns
disp("Resizing columns to make all timeseries same length...");
maxsize = Inf;
dat = {logged_data.Data};
for i = 1:total_types
    sz = size(dat{1,i},2);
    if sz < maxsize
        maxsize = sz;
    end
end
for i = 1:total_types
    logged_data(i).Data = logged_data(i).Data(:,1:maxsize);
end

%% Headers
% joints
idx_joints = find(cellfun(@(v) any(strcmp(v,'q_{joints}')),{logged_data.Name}));
logged_data(idx_joints).Headers = {...
    'LeftShin','LeftTarsus','LeftToePitch','LeftToeRoll','LeftHeelSpring',...
    'RightShin','RightTarsus','RightToePitch','RightToeRoll','RightHeelSpring'};
logged_data(idx_joints+1).Headers = logged_data(idx_joints).Headers;

% motors
idx_motors = find(cellfun(@(v) any(strcmp(v,'q_{motors}')),{logged_data.Name}));
logged_data(idx_motors).Headers = {...
    'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftToeA','LeftToeB',...
    'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightToeA','RightToeB',...
    'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
    'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
logged_data(idx_motors+1).Headers = logged_data(idx_motors).Headers;

% q_all
idx_qall = find(cellfun(@(v) any(strcmp(v,'q_{all}')),{logged_data.Name}));
logged_data(idx_qall).Headers = {...
    'BaseX','BaseY','BaseZ','BaseYaw','BasePitch','BaseRoll',...
    'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftTarsus','LeftToePitch','LeftToeRoll',...
    'LeftShoulderRoll','LeftShoulderPitch','LeftShoulderYaw','LeftElbow',...
    'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightTarsus','RightToePitch','RightToeRoll',...
    'RightShoulderRoll','RightShoulderPitch','RightShoulderYaw','RightElbow'};
logged_data(idx_qall+1).Headers = logged_data(idx_qall).Headers;

% q_virts
idx_virts = find(cellfun(@(v) any(strcmp(v,'q_{virts_{des}}')),{logged_data.Name}));
logged_data(idx_virts).Headers = {...
    'LeftHipRoll','LeftHipYaw','LeftHipPitch','LeftKnee','LeftToePitch','LeftToeRoll',...
    'RightHipRoll','RightHipYaw','RightHipPitch','RightKnee','RightToePitch','RightToeRoll'};
logged_data(idx_virts+1).Headers = logged_data(idx_virts).Headers;

% motors des
idx_motors_des = find(cellfun(@(v) any(strcmp(v,'q_{motors_{des}}')),{logged_data.Name}));
logged_data(idx_motors_des).Headers = logged_data(idx_motors).Headers;
logged_data(idx_motors_des+1).Headers = logged_data(idx_motors).Headers;

% motor torque
idx_u_current = find(cellfun(@(v) any(strcmp(v,'u_{torque_{current}}')),{logged_data.Name}));
logged_data(idx_u_current).Headers = logged_data(idx_motors).Headers;
logged_data(idx_u_current+1).Headers = logged_data(idx_motors).Headers;

% torque limit
idx_torque_limit = find(cellfun(@(v) any(strcmp(v,'torque_{limits}')),{logged_data.Name}));
logged_data(idx_torque_limit).Headers = logged_data(idx_motors).Headers;
logged_data(idx_torque_limit+1).Headers = logged_data(idx_motors).Headers;

%% Compute custom data
disp("Calculating custom timeseries...");
custom_data = struct();
custom_data(1).Name = 'q_{virts_{act}}';
custom_data(1).Data = [...
    logged_data(idx_motors).Data(1,:); logged_data(idx_motors).Data(2,:); logged_data(idx_motors).Data(3,:); logged_data(idx_motors).Data(4,:); logged_data(idx_joints).Data(3,:); logged_data(idx_joints).Data(4,:);
    logged_data(idx_motors).Data(7,:); logged_data(idx_motors).Data(8,:); logged_data(idx_motors).Data(9,:); logged_data(idx_motors).Data(10,:); logged_data(idx_joints).Data(8,:); logged_data(idx_joints).Data(9,:)];

custom_data(2).Name = 'qdot_{virts_{act}}';
custom_data(2).Data = [...
    logged_data(idx_motors+1).Data(1,:); logged_data(idx_motors+1).Data(2,:); logged_data(idx_motors+1).Data(3,:); logged_data(idx_motors+1).Data(4,:); logged_data(idx_joints+1).Data(3,:); logged_data(idx_joints+1).Data(4,:);
    logged_data(idx_motors+1).Data(7,:); logged_data(idx_motors+1).Data(8,:); logged_data(idx_motors+1).Data(9,:); logged_data(idx_motors+1).Data(10,:); logged_data(idx_joints+1).Data(8,:); logged_data(idx_joints+1).Data(9,:)];


end


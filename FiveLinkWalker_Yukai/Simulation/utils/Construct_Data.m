function Data = Construt_Data()

Data.stanceLeg = 0;
Data.lG = 0; 
Data.l_LeftToe = 0;
Data.l_RightToe = 0;
Data.l_LeftToe_vg = 0;
Data.l_RightToe_vg = 0;
Data.x0 = 0;

Data.dx0_next = 0;
Data.x0_next = 0;
Data.dxf_next_goal = 0;

Data.p_com = zeros(3,1);
Data.v_com = zeros(3,1);
Data.vx_com = 0;
Data.vy_com = 0;
Data.vz_com = 0;
Data.px_com = 0;
Data.py_com = 0;
Data.pz_com = 0;
Data.pseudo_com_vx = 0;
Data.q = zeros(7,1);
Data.dq = zeros(7,1);
Data.u = zeros(4,1);

Data.hr = zeros(4,1);
Data.dhr = zeros(4,1);
Data.h0 = zeros(4,1);
Data.dh0 = zeros(4,1);

Data.lCoM=0;%angular momentum about the center of mass
Data.lstance=0; %angular momentum about the stance foot
Data.v_sw=zeros(2,1); %swing leg velocity
Data.p_sw=zeros(2,1); %swing leg position 
Data.p_dsw=0; %swing leg desired position
Data.torso_angle=0; %torso angle
Data.CoM_height=0; %relative height of center of mass with respect to stance foot (p_com-p_st)
Data.p_st=zeros(2,1);%stance leg position
Data.f_ext=0;%external force
end
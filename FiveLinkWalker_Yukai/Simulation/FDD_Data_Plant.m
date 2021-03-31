classdef FDD_Data_Plant < matlab.System & matlab.system.mixin.Propagates & matlab.system.mixin.SampleTime
    % Untitled3 Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.
    
    % Public, tunable properties
    properties
        
    end
    
    
    % Pre-computed constants
    properties(Access = private)
        total_mass = 32;
    end
    
    methods(Access = protected)
        
        
        function Data = stepImpl(obj,x,ctrl_info,GRF)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            Data = Construct_Data();
            q = x(1:7);
            dq = x(8:14);
            
            y=ctrl_info.y;
            dy=ctrl_info.y;
            stanceLeg=ctrl_info.stanceLeg;
            step= ctrl_info.step;
            task=ctrl_info.task;
            
            if stanceLeg == -1
                GRF_sw_z = GRF(6);
                GRF_st_z = GRF(3);
                
                GRF_sw_x = GRF(4);
                GRF_st_x = GRF(1);
                
            else
                GRF_sw_z = GRF(3);
                GRF_st_z = GRF(6);
                
                GRF_sw_x = GRF(1);
                GRF_st_x = GRF(4);
            end
            
            
            
            p_com = p_COM(q);
            Jp_com = Jp_COM(q);
            dJp_com = dJp_COM(q,dq);
            v_com = Jp_com*dq;
            
            p_LT = p_LeftToe(q);
            Jp_LT = Jp_LeftToe(q);
            dJp_LT = dJp_LeftToe(q,dq);
            v_LT = Jp_LT*dq;
            
            p_RT = p_RightToe(q);
            Jp_RT = Jp_RightToe(q);
            dJp_RT = dJp_RightToe(q,dq);
            v_RT = Jp_RT*dq;
            
            %knee
            p_LK = p_LeftKnee(q);
            Jp_LK = Jp_LeftKnee(q);
            
            p_RK = p_RightKnee(q);
            Jp_RK = Jp_RightKnee(q);
            
            %hip
            p_LH = p_LeftHip(q);
            Jp_LH = Jp_LeftHip(q);
            
            p_RH = p_RightHip(q);
            Jp_RH = Jp_RightHip(q);
            
            % com position RELATIVE to toes
            
            rp_LT = p_com - p_LT;%relative position of the left foot to the CoM
            Jrp_LT = Jp_com - Jp_LT;
            dJrp_LT = dJp_com - dJp_LT;
            rv_LT = v_com - v_LT;
            
            rp_RT = p_com - p_RT;
            Jrp_RT = Jp_com - Jp_RT;
            dJrp_RT = dJp_com - dJp_RT;
            rv_RT = v_com - v_RT;
            
            LG = getFLWAngularMomentum(p_com,x);
            L_LeftToe = getFLWAngularMomentum(p_LT,x);
            L_RightToe = getFLWAngularMomentum(p_RT,x);
            %for debugging: angular momentum contribution about feet from
            %linear momentum about CoM
            L_LeftToe_vg = obj.total_mass*cross(rp_LT,v_com);
            L_RightToe_vg = obj.total_mass*cross(rp_RT,v_com);
            
            
            
            
            if stanceLeg == -1
                
                p_stT = p_LT;
                Jp_stT = Jp_LT;
                dJp_stT = dJp_LT;
                v_stT = v_LT;
                
                p_swT = p_RT;
                Jp_swT = Jp_RT;
                dJp_swT = dJp_RT;
                v_swT = v_RT;
                
                rp_stT = rp_LT;
                Jrp_stT = Jrp_LT;
                dJrp_stT = dJrp_LT;
                rv_stT = rv_LT;
                
                rp_swT = rp_RT;
                Jrp_swT = Jrp_RT;
                dJrp_swT = dJrp_RT;
                rv_swT = rv_RT;
                
                L_stToe = L_LeftToe;
                L_swToe = L_RightToe;
                
                p_swK=p_LK;
                Jp_swK=Jp_LK;
            else
                p_stT = p_RT;
                Jp_stT = Jp_RT;
                dJp_stT = dJp_RT;
                v_stT = v_RT;
                
                p_swT = p_LT;
                Jp_swT = Jp_LT;
                dJp_swT = dJp_LT;
                v_swT = v_LT;
                
                rp_stT = rp_RT;
                Jrp_stT = Jrp_RT;
                dJrp_stT = dJrp_RT;
                rv_stT = rv_RT;
                
                rp_swT = rp_LT;
                Jrp_swT = Jrp_LT;
                dJrp_swT = dJrp_LT;
                rv_swT = rv_LT;
                
                L_stToe = L_RightToe;
                L_swToe = L_RightToe;
                
                p_swK=p_RK;
                Jp_swK=Jp_RK;
            end
            
            %% Data assignment
            Data.stanceLeg = stanceLeg;
            Data.lG = LG(2);
            Data.l_LeftToe = L_LeftToe(2);
            Data.l_RightToe = L_RightToe(2);
            Data.l_LeftToe_vg = L_LeftToe_vg(2);
            Data.l_RightToe_vg = L_RightToe_vg(2);
            
            Data.p_com = p_com;
            Data.v_com = v_com;
            Data.vx_com = v_com(1);
            Data.vy_com = v_com(2);
            Data.vz_com = v_com(3);
            Data.px_com = p_com(1);
            Data.py_com = p_com(2);
            Data.pz_com = p_com(3);
            
            Data.q = q;
            Data.dq = dq;
            
            
            Data.lCoM=LG(2);  %angular momentum about center of mass
            Data.lstance=L_stToe(2); %angular momentum about the contact point
            Data.v_sw=v_swT(1:2:3); %swing leg velocity
            Data.p_sw=p_swT(1:2:3); %swing leg position
            
            Data.torso_angle=q(3); %torso angle
            Data.CoM_height=rp_stT(3); %relative height of center of mass with respect to stance foot (p_com-p_st)
            Data.p_st=p_stT(1:2:3);%stance leg position
            Data.f_ext=ctrl_info.f_ext;
            Data.p_relCoMLegs=p_com(1)-0.5*(p_LT(1)+p_RT(1));
            Data.stepDuration=ctrl_info.stepDuration;
            %grabbing the updated GRF information
            
            Data.st_GRF=[GRF_st_x;GRF_st_z];
            Data.sw_GRF=[GRF_sw_x;GRF_sw_z];
            Data.step=step;
            Data.y=y; %virtual constraint (VC)
            Data.dy=dy; %VC derivative
            Data.task=task;
        end
        %% Default functions
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
        end
        
        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
        
        function [name_1, name_2, name_3]  = getInputNamesImpl(~)
            %GETINPUTNAMESIMPL Return input port names for System block
            name_1 = 'x';
            name_2 = 'ctrl_info';
            name_3 = 'GRF';
        end % getInputNamesImpl
        
        function [name_1] = getOutputNamesImpl(~)
            %GETOUTPUTNAMESIMPL Return output port names for System block
            
            name_1 = 'Data';
            
            
        end % getOutputNamesImpl
        
        % PROPAGATES CLASS METHODS ============================================
        function [ Data] = getOutputSizeImpl(~)
            %GETOUTPUTSIZEIMPL Get sizes of output ports.
            
            Data = [1, 1];
            
        end % getOutputSizeImpl
        
        function [Data] = getOutputDataTypeImpl(~)
            %GETOUTPUTDATATYPEIMPL Get data types of output ports.
            
            Data = 'cassieDataBus';
            
        end % getOutputDataTypeImpl
        
        function [Data] = isOutputComplexImpl(~)
            %ISOUTPUTCOMPLEXIMPL Complexity of output ports.
            
            Data = false;
            
        end % isOutputComplexImpl
        
        function [ Data] = isOutputFixedSizeImpl(~)
            %ISOUTPUTFIXEDSIZEIMPL Fixed-size or variable-size output ports.
            
            Data = true;
            
        end % isOutputFixedSizeImpl
    end
end

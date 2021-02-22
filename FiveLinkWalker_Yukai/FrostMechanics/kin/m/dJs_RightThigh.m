function [output1] = dJs_RightThigh(var1,var2)
    if coder.target('MATLAB')
        [output1] = dJs_RightThigh_mex(var1,var2);
    else
        coder.cinclude('dJs_RightThigh_src.h');
        
        output1 = zeros(6, 7);

        
        coder.ceval('dJs_RightThigh_src' ...
            ,coder.wref(output1) ...
            ,coder.rref(var1) ,coder.rref(var2) );
    end
end

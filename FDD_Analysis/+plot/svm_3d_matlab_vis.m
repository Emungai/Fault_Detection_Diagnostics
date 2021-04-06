 function [] = svm_3d_matlab_vis(mdl,X,group,X_test,group_test,plotInfo)
 %from https://www.mathworks.com/matlabcentral/answers/407736-plot-3d-hyperplane-from-fitcsvm-results
 %also see: https://stackoverflow.com/questions/16146212/how-to-plot-a-hyper-plane-in-3d-for-the-svm-results/19969412#19969412
 %Gather support vectors from ClassificationSVM struct
 sv =  mdl.SupportVectors;
 %set step size for finer sampling
 d =0.05;
 %generate grid for predictions at finer sample rate
 [x, y, z] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x(:),y(:),z(:)];
 %get scores, f
 [ ~ , f] = predict(mdl,xGrid);
 %reshape to same grid size as the input
 f = reshape(f(:,2), size(x));
 % Assume class labels are 1 and 0 and convert to logical
 t = logical(group);
  t_test=logical(group_test);
 %plot data points, color by class label
 figure
 plot3(X(t, 1), X(t, 2), X(t, 3), 'b.','LineWidth',2);
 hold on
 plot3(X(~t, 1), X(~t, 2), X(~t, 3), 'r.','LineWidth',2);
 hold on
 plot3(X_test(t_test, 1), X_test(t_test, 2), X_test(t_test, 3), 'bx','LineWidth',2);
 hold on
 plot3(X_test(~t_test, 1), X_test(~t_test, 2), X_test(~t_test, 3), 'rx','LineWidth',2);
 hold on
 % load unscaled support vectors for plotting
 plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'go','LineWidth',2);
 %plot decision surface
 [faces,verts,~] = isosurface(x, y, z, f, 0, x);
 patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','edgecolor','none', 'FaceAlpha', 0.2);
 grid on
 box on
 view(3)
xlabel(plotInfo.xlabel)
ylabel(plotInfo.ylabel)
zlabel(plotInfo.zlabel)
title(plotInfo.title)
 hold off
 end
 %%
%  function [] = svm_3d_matlab_vis(svmStruct,Xdata,group)
% sv =  svmStruct.SupportVectors;
% alphaHat = svmStruct.Alpha;
% bias = svmStruct.Bias;
% kfun = svmStruct.KernelFunction;
% kfunargs = svmStruct.KernelFunctionArgs;
% sh = svmStruct.ScaleData.shift; % shift vector
% scalef = svmStruct.ScaleData.scaleFactor; % scale vector
% 
% group = group(~any(isnan(Xdata),2));
% Xdata =Xdata(~any(isnan(Xdata),2),:); % remove rows with NaN 
% 
% % scale and shift data
% Xdata1 = repmat(scalef,size(Xdata,1),1).*(Xdata+repmat(sh,size(Xdata,1),1));
% k = 50; 
% cubeXMin = min(Xdata1(:,1));
% cubeYMin = min(Xdata1(:,2));
% cubeZMin = min(Xdata1(:,3));
% 
% cubeXMax = max(Xdata1(:,1));
% cubeYMax = max(Xdata1(:,2));
% cubeZMax = max(Xdata1(:,3));
% stepx = (cubeXMax-cubeXMin)/(k-1);
% stepy = (cubeYMax-cubeYMin)/(k-1);
% stepz = (cubeZMax-cubeZMin)/(k-1);
% [x, y, z] = meshgrid(cubeXMin:stepx:cubeXMax,cubeYMin:stepy:cubeYMax,cubeZMin:stepz:cubeZMax);
% mm = size(x);
% x = x(:);
% y = y(:);
% z = z(:);
% f = (feval(kfun,sv,[x y z],kfunargs{:})'*alphaHat(:)) + bias;
% t = strcmp(group, group{1});
% 
% % unscale and unshift data 
% Xdata1 =(Xdata1./repmat(scalef,size(Xdata,1),1)) - repmat(sh,size(Xdata,1),1);
% x =(x./repmat(scalef(1),size(x,1),1)) - repmat(sh(1),size(x,1),1);
% y =(y./repmat(scalef(2),size(y,1),1)) - repmat(sh(2),size(y,1),1);
% z =(z./repmat(scalef(3),size(z,1),1)) - repmat(sh(3),size(z,1),1);
% figure
% plot3(Xdata1(t, 1), Xdata1(t, 2), Xdata1(t, 3), 'b.');
% hold on
% plot3(Xdata1(~t, 1), Xdata1(~t, 2), Xdata1(~t, 3), 'r.');
% hold on
% % load unscaled support vectors for plotting
% sv = svmStruct.SupportVectorIndices;
% sv = [Xdata1(sv, :)];
% plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'go');
% legend(group{1},group{end},'support vectors')
% 
% x0 = reshape(x, mm);
% y0 = reshape(y, mm);
% z0 = reshape(z, mm);
% v0 = reshape(f, mm);
% 
% [faces,verts,colors] = isosurface(x0, y0, z0, v0, 0, x0);
% patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','edgecolor', 'none', 'FaceAlpha', 0.5);
% grid on
% box on
% view(3)
% hold off
% end

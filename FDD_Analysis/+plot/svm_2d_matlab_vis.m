function [] = svm_2d_matlab_vis(mdl,X,group,X_test,group_test,plotInfo)
%SVM_2D_MATLAB_VIS Summary of this function goes here
%   Detailed explanation goes here



sv =  mdl.SupportVectors;
 %set step size for finer sampling
 d =0.05;
 %generate grid for predictions at finer sample rate
 [x, y] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
     min(X(:,2)):d:max(X(:,2)));
 xGrid = [x(:),y(:)];
 
 %get scores, f
 [ ~ , f] = predict(mdl,xGrid);
 %reshape to same grid size as the input
 f = reshape(f(:,2), size(x));
 % Assume class labels are 1 and 0 and convert to logical
 t = logical(group);
 t_test=logical(group_test);
 
 figure
 plot(X(t, 1), X(t, 2), 'b.','LineWidth',2);
 hold on
 plot(X(~t, 1), X(~t, 2), 'r.','LineWidth',2);
 hold on
  plot(X_test(t_test, 1), X_test(t_test, 2), 'bx','LineWidth',2);
 hold on
 plot(X_test(~t_test, 1), X_test(~t_test, 2), 'rx','LineWidth',2);
 % load unscaled support vectors for plotting
 plot(sv(:, 1), sv(:, 2), 'go','LineWidth',2);
 %plot decision boundary
 contour(x,y,f,[0 0],'k');
 xlabel(plotInfo.xlabel)
ylabel(plotInfo.ylabel)

title(plotInfo.title)
 hold off



end

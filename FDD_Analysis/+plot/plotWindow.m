%% plotting window results
step=[1:2000:size(PCA_info_full,1)];
if step(end) < size(PCA_info_full,1)
    step(end+1)=size(PCA_info_full,1);
end
ramp=1;
for j=1:length(step)-1
    ramp=1;
    X=normc(PCA_info_full(step(j):step(j+1),1:end-2));
    t=PCA_info_full(step(j):step(j+1),end-1);
    force=PCA_info_full(step(j):step(j+1),end);
    [L_O,S_O]=RPCA(X);
    %% plotting on Principal Axes
%     
%     figure, hold on
%     k=1;
%     beg=step(j);
%     
%     obs=X;
%     % V=V_svd;
%     % for j=1: length(PCA_info_str)
%     %     obs=PCA_info_str{j}';
%     
%     if ramp
%         prev=round(PCA_info_full(beg,end-1));
%         jetcustom = jet(round(t(end))-round(t(1))+1);
%     else
%         prev=PCA_info_full(1,end);
%         jetcustom = jet(7);
%     end
%     
%     for i=1:size(obs,1)
%         x = V(:,1)'*obs(i,:)';
%         y = V(:,2)'*obs(i,:)';
%         z = V(:,3)'*obs(i,:)';
%         if goodData
%             plot3(x,y,z,'kx','LineWidth',2);
%         else
%             if ramp
%                 if i<size(obs,1)
%                 comp=round(PCA_info_full(i+beg,end-1));
%                 end
%             else
%                 if i<size(obs,1)
%                 comp=PCA_info_full(i+beg,end);
%                 end
%             end
%             if  comp~= prev
%                 prev=comp;
%                 k=k+1;
%             end
%             
%             %         plot3(x,y,z,'kx','LineWidth',2);
%             [LIA_per,LOCB]=ismember(-3000,force);
%             [LIA]=ismember(0,force);
%             if LIA && LIA_per
%                 if i< LOCB
%                     plot3(x,y,z,'x','Color', jetcustom(1,:),'LineWidth',2);
%                 else
%                     plot3(x,y,z,'x','Color', jetcustom(2,:),'LineWidth',2);
%                 end
%             else
%                 plot3(x,y,z,'x','Color', jetcustom(k,:),'LineWidth',2);
%             end
%         end
% %                plot3(x,y,z,'rx','LineWidth',2);
%         % plot(x,y,'x','Color', jetcustom(k,:),'LineWidth',2)
%         
%         
%     end
%     % end
%     colormap(jetcustom);
%     if ramp && k>1
%         
%         cb = colorbar;
%         caxis([round(t(1)) round(t(end))])
%         ylabel(cb,'time (rounded)')
%         
%         
%         
%     elseif ~ramp
%         cb = colorbar;
%         caxis([size(PCA_info_str)])
%         ylabel(cb,'force')
%     end
%     % caxis([0 k-1])
%     % caxis([1 k])
%     %  caxis([0 7])
%     view(85,25), grid on, set(gca,'FontSize',13)
%     xlabel('V1')
%     ylabel('V2')
%     zlabel('V3')
%     
%     title(['RPCA-PCA(L)-X(window', string(t(1)),'-',string(t(end)),')'])
%     % title('PCA(X)-X')
    %%
    feat={'lcom_y','lstance_y','v_sw_x','v_sw_z','p_sw_z','p_dsw_x','torso_angle','com_height','p_st_z','CoM_rel_p_legs','step_duration'};
    ramp=0;
    window=1;
    prev=PCA_info_full(1,end);
    jetcustom = jet(7);
    k=1;
    figure
    for i=1:length(V)
        subplot(2,6,i)
        if ramp
            t=Data.p_com.Time;
            plot(t,L_O(:,i)-S_O(:,i));
            hold on
            plot(t,zeros(size(L_O(:,i))),'k','LineWidth',2);
            
        else
            
            %         fst=length(PCA_info_str{1});
            %         snd=length(PCA_info_str{2});
            %         thd=length(PCA_info_str{3});
            fst=0;
            if window
                [LIA_per,LOCB]=ismember(-3000,force);
                [LIA]=ismember(0,force);
                jetcustom = jet(2);
                if LIA_per && LIA
                    t_act=LOCB-1;
                    
                    plot([t(1:t_act)],L_O(1:t_act,i)-S_O(1:t_act,i),'Color',jetcustom(1,:));
                    hold on
                    plot([t(t_act+1:end)],L_O(t_act+1:end,i)-S_O(t_act+1:end,i),'Color',jetcustom(2,:));
                    if i==10
                        legend(['nominal(',num2str(t(1)),'-',num2str(t(end)),')'],'-3000 knee')
                    end
                elseif LIA && ~LIA_per
                    plot(t,L_O(:,i)-S_O(:,i),'Color',jetcustom(1,:));
                    if i==10
                        legend(['nominal(',num2str(t(1)),'-',num2str(t(end)),')'])
                    end
                elseif ~LIA && LIA_per
                    plot(t,L_O(:,i)-S_O(:,i),'Color',jetcustom(2,:));
                    if i==10
                        legend(['-3000 knee (',num2str(t(1)),'-',num2str(t(end)),')'])
                    end
                else
                    'stop'
                end
                
                
                
                hold on
                plot([t],zeros(size(L_O(:,i))),'k','LineWidth',2);
                
            else
                for j=1:length(PCA_info_str)
                    
                    snd=length(PCA_info_str{j});
                    plot([fst+1:fst+snd],L_O(fst+1:snd+fst,i)-S_O(fst+1:snd+fst,i),'Color',jetcustom(j,:));
                    hold on
                    fst=fst+snd;
                    plot(zeros(size(L_O(:,i))),'k','LineWidth',2);
                end
            end
            %         hold on
            %         plot([fst+1:fst+snd],L_O(fst+1:snd+fst,i)-S_O(fst+1:snd+fst,i),'Color',jetcustom(2,:));
            %         hold on
            %         plot([1+snd+fst:fst+snd+thd],L_O(fst+snd+1:end,i)-S_O(fst+snd+1:end,1),'Color',jetcustom(3,:));
            %         hold on
            
        end
        
        title(feat{i})
    end
    % sgtitle('L-S')
    % legend(['nominal(',num2str(t(1)),'-',num2str(t(end)),')'],'-3000 knee')
end
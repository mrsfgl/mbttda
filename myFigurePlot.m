function myFigurePlot(tau_list, classsize_list, repeats, noise, Dataset, K_list, K,tensor_shape)
%% Path Setup
% addpath 'tensorlab'     % Tens2mat
% addpath 'FOptM-share'   % Convex under unitary constraint
% addpath 'mdmp'          % Tensor Trace
% addpath 'TT_TensNet'    % Merge the proposed networks
% addpath 'TT_Approximate'% Approximation Algorithm
% addpath 'TNPE'          % TNPE algorithm package
% addpath 'Self_Tool'     % manifold dimensional and KNN clasifier

%% Algo(App, TN, TNPE, KNN) * tau * classsize * iter;
tot_iter = p.;
ResErr = nan(5, length(tau_list),length(classsize_list), tot_iter); 
ResStor= nan(5, length(tau_list),length(classsize_list), tot_iter);
ResTime_subspace   = nan(5, length(tau_list),length(classsize_list), tot_iter);
ResTime_embedding  = nan(5, length(tau_list),length(classsize_list), tot_iter);
ResTime_classify   = nan(5, length(tau_list),length(classsize_list), tot_iter);
% KNNS.Storage=1576960;
TTLDA=[];
for ifile = 1
    disp(['ifile ' num2str(ifile) ]);
    path = ['result' Dataset];
    K_idx = find(K_list == K);  % K neighbors for classification
    cd(path);
    for iTau = 1: length(tau_list)
        %disp(['iTau ' num2str(iTau) ]);
        for iClass = 1: length(classsize_list)
            %disp(['iClass ' num2str(iClass) ]);
            for iIter = 1:repeats
                filename= ['Result'... % num2str(noise), ...
                    '_repeat' num2str(iIter),...
                    '_tau' num2str(tau_list(iTau)),...
                    '_lambda',num2str(classsize_list(iClass))...
                    '.mat'];
                load(filename);
                ResErr( 1,iTau,iClass,iIter+(ifile-1) * repeats)            = ThW.PreErr(1);
                ResStor(1,iTau,iClass,iIter+(ifile-1) * repeats)            = (ThW.Storage(1)/KNNS.Storage);
                ResTime_subspace(1,iTau,iClass,iIter+(ifile-1) * repeats)   = sum(ThW.time_subspace);
                ResTime_mat(1,iTau,iClass,iIter+(ifile-1) * repeats)        = sum(ThW.time_mat);
                ResTime_search(1,iTau,iClass,iIter+(ifile-1) * repeats)     = sum(ThW.time_search);
                ResTime_embedding(1,iTau,iClass,iIter+(ifile-1) * repeats)  = ThW.time_embedding(1);
                ResTime_classify(1,iTau,iClass,iIter+(ifile-1) * repeats)   = ThW.time_classify(1);
                
                
%                 ResErr(2,iTau,iClass,iIter+(ifile-1) * repeats)             = KNNS.PreErr(K_idx);
%                 ResStor(2,iTau,iClass,iIter+(ifile-1) * repeats)            = log10(1);
%                 ResTime_classify(2,iTau,iClass,iIter+(ifile-1) * repeats)   = KNNS.time_classify;
                
%                 ResErr(3,iTau,iClass,iIter+(ifile-1) * repeats)             = TTDA.PreErr(K_idx);
%                 ResStor(3,iTau,iClass,iIter+(ifile-1) * repeats)            = log10(TTDA.Storage/KNNS.Storage);
%                 ResTime_subspace(3,iTau,iClass,iIter+(ifile-1) * repeats)   = TTDA.time_subspace;
%                 ResTime_embedding(3,iTau,iClass,iIter+(ifile-1) * repeats)  = TTDA.time_embedding;
%                 ResTime_classify(3,iTau,iClass,iIter+(ifile-1) * repeats)   = TTDA.time_classify;
%                 
if ~isempty(TTLDA)
                ResErr(4,iTau,iClass,iIter+(ifile-1) * repeats)             = TTLDA.PreErr(1);
                ResStor(4,iTau,iClass,iIter+(ifile-1) * repeats)            = (TTLDA.Storage(1)/KNNS.Storage);
                ResTime_subspace(4,iTau,iClass,iIter+(ifile-1) * repeats)   = sum(TTLDA.time_subspace);
                ResTime_mat(4,iTau,iClass,iIter+(ifile-1) * repeats)        = sum(TTLDA.time_mat);
                ResTime_search(4,iTau,iClass,iIter+(ifile-1) * repeats)     = sum(TTLDA.time_search);
                ResTime_embedding(4,iTau,iClass,iIter+(ifile-1) * repeats)  = TTLDA.time_embedding(1);
                ResTime_classify(4,iTau,iClass,iIter+(ifile-1) * repeats)   = TTLDA.time_classify(1);
end
TTLDA=[];
           
if ~isempty(LDA)
                ResErr( 5,iTau,iClass,iIter+(ifile-1) * repeats)            = LDA.LDAPreErr(K_idx);
                ResStor(5,iTau,iClass,iIter+(ifile-1) * repeats)            = (LDA.LDAStorage/KNNS.Storage);
                ResTime_subspace(5,iTau,iClass,iIter+(ifile-1) * repeats)   = LDA.LDAtime_subspace;
                ResTime_embedding(5,iTau,iClass,iIter+(ifile-1) * repeats)  = LDA.LDAtime_embedding;
                ResTime_classify(5,iTau,iClass,iIter+(ifile-1) * repeats)   = LDA.LDAtime_classify;
end
LDA = [];
            end
        end
    end
    cd ..;
end

%%
% ResErr = permute(mean(permute(ResErr, [4,1,2,3]),1), [2,3,4,1]);
ResStor= permute(mean(permute(ResStor,[4,1,2,3]),1), [2,3,4,1]);
% ResTime_subspace= permute(mean(permute(ResTime_subspace,[4,1,2,3]),1), [2,3,4,1]);
ResTime_embedding= permute(mean(permute(ResTime_embedding,[4,1,2,3]),1), [2,3,4,1]);
ResTime_classify= permute(mean(permute(ResTime_classify,[4,1,2,3]),1), [2,3,4,1]);
%
name_shape='';
for i=1:length(tensor_shape)
    name_shape=strcat(name_shape, int2str(tensor_shape(i)));
end

name_path=strcat('Res_',Dataset);
if ~exist(name_path)
    mkdir(name_path);
end

%% classify Time
for iClass = 1
    %close all;
    figure()
    hold on;
    ResStor(ResStor>1)=1;
    legTxt=[];
    for lId=1:length(classsize_list)
        j=4;plot(mean(ResTime_classify(j,:,lId,:),4), ResStor(j,:,lId), '-*','linewidth', 4); 
        legTxt{lId}=['2WTTDA']; % (\tau=',num2str(classsize_list(lId)),')'];
    end
    for lId=1:length(classsize_list)
        j=1;plot(mean(ResTime_classify(j,:,lId,:),4), ResStor(j,:,lId), '-*','linewidth', 4); 
        legTxt{lId+length(classsize_list)}=['3WTTDA']; %(\tau=',num2str(classsize_list(lId)),')'];
    end
%     j=1;plot(ResTime_classify(j,:,iClass), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     j=2;plot(ResTime_classify(j,:,iClass), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     for lId=1:length(classsize_list)
%         j=3;plot(ResTime_classify(j,:,lId,iClass), ResStor(j,:,lId), '-*','linewidth', 4);
%         legTxt{lId+3*length(classsize_list)}=['1WTTDA (\tau=',num2str(classsize_list(lId)),')'];
%     end
%     scatter(mean(ResTime_classify(2,:, iClass)), 0, 'k*');
%     for lId=1:length(classsize_list)
%         j=5;plot(ResTime_classify(j,:,lId,iClass), ResStor(j,:,lId), '-*','linewidth', 4);
%         legTxt{lId+2*length(classsize_list)}=['LDA']; % (\tau=',num2str(classsize_list(lId)),')'];
%     end
%     j=5;plot(mean(ResTime_classify(j,:,iClass,:),4), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     legTxt=[legTxt,'LDA'];
%     scatter(1, mean(ResTime_classify(4,:, iClass)), 'k*');
    legend(legTxt(:),'Location','southeast');
    fig=gcf;
    set(findall(fig,'-property','FontSize'),'FontSize',28)
    set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
    xlabel('Classification Time(s)');
    ylabel('Log of the Normalized Storage Cost');
    grid
    hold off;
    saveas(gcf, [name_path, '/TimeClassify_' Dataset, ...
        '_noise' num2str(noise), ...
        '_K', num2str(K),...
        '_lambda', num2str(classsize_list(iClass)),... 
        '_shape',name_shape,...
        '.pdf']);
end

%% subspace Time
for iClass = 1
    %close all;
    figure('WindowState','maximized')
    hold on;
    ResStor(ResStor>1)=1;
%     for lId=1:length(classsize_list)
%         j=3;[~, idx]=sort(ResStor(j,:,iClass));plot(ResStor(j,idx,lId),ResTime_subspace(j,idx,lId,iClass), '-*','linewidth', 4);
%         legTxt{lId}=['1WTTDA (\tau=',num2str(classsize_list(lId)),')'];
%     end
    for lId=1:length(classsize_list)
%         j=4;[~, idx]=sort(ResStor(j,:,iClass));errorbar(ResStor(j,idx,lId), mean(ResTime_subspace(j,idx,lId,:),4),std(ResTime_subspace(j,idx,lId,:),[],4),  '-*','linewidth', 4); 
        j=4;[~, idx]=sort(ResStor(j,:,iClass));loglog(ResStor(j,idx,lId), mean(ResTime_subspace(j,idx,lId,:),4),  '-*','linewidth', 4,'DisplayName','2WTTDA'); 
        % (\lambda=',num2str(classsize_list(lId)),')'];
    end
    for lId=1:length(classsize_list)
%         j=1;[~, idx]=sort(ResStor(j,:,iClass));errorbar(ResStor(j,idx,lId), mean(ResTime_subspace(j,idx,lId,:),4),std(ResTime_subspace(j,idx,lId,:),[],4),  '-*','linewidth', 4); 
        j=1;[~, idx]=sort(ResStor(j,:,iClass));loglog(ResStor(j,idx,lId), mean(ResTime_subspace(j,idx,lId,:),4),  '-*','linewidth', 4,'DisplayName','3WTTDA'); 
        %(\lambda=',num2str(classsize_list(lId)),')'];
    end
%     j=1;plot(ResStor(j,:,iClass), ResTime_subspace(j,:,iClass), '-*','linewidth', 4); 
%     j=2;plot(ResStor(j,:,iClass), ResTime_subspace(j,:,iClass),  '-*','linewidth', 4); 
%     j=3;plot(ResTime_subspace(j,:,iClass), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     j=5;plot(ResStor(j,:,iClass), mean(ResTime_subspace(j,:,iClass,:),4), '-*','linewidth', 4); 
    for lId=1:length(classsize_list)
%         j=5;[~, idx]=sort(ResStor(j,:,iClass));errorbar(ResStor(j,idx,lId), mean(ResTime_subspace(j,idx,lId,:),4),std(ResTime_subspace(j,idx,lId,:),[],4),  '-*','linewidth', 4); 
        j=5;[~, idx]=sort(ResStor(j,:,iClass));plot(ResStor(j,idx,lId),ResTime_subspace(j,idx,lId,iClass), '-*','linewidth', 4,'DisplayName','LDA'); %(\tau=',num2str(classsize_list(lId)),')'];
    end
    legend('Location','northwest');
    fig=gcf;
    set(findall(fig,'-property','FontSize'),'FontSize',28)
    set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
    %axis([0 1 0 2000]);
    ylabel('Subspace Time(s)');
    xlabel('Log of the Normalized Storage Cost');
    grid
    hold off;
    saveas(gcf, [name_path, '/TimeSubspace_' Dataset,...
        '_noise' num2str(noise), ...
        '_K', num2str(K),...
        '_Tr', num2str(classsize_list(iClass)),...
        '_shape',name_shape,...
        '.pdf']);
end

%% embedding Time
for iClass = 1
    %close all;
    figure()
    hold on;
    for lId=1:length(classsize_list)
        j=4;plot(mean(ResTime_embedding(j,:,lId,:),4), ResStor(j,:,lId), '-*','linewidth', 4); 
        legTxt{lId}=['2WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
    end
    for lId=1:length(classsize_list)
        j=1;plot(mean(ResTime_embedding(j,:,lId,:),4), ResStor(j,:,lId), '-*','linewidth', 4); 
        legTxt{lId+length(classsize_list)}=['3WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
    end
%     j=1;plot(ResTime_embedding(j,:,iClass), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     j=2;plot(ResTime_embedding(j,:,iClass), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     j=3;plot(ResTime_embedding(j,:,iClass), ResStor(j,:,iClass), '-*','linewidth', 4); 
%     j=5;plot(mean(ResTime_embedding(j,:,iClass,:),4), ResStor(j,:,iClass), '-*','linewidth', 4); 
    legend(legTxt,'Location','southeast');
    fig=gcf;
    set(findall(fig,'-property','FontSize'),'FontSize',28)
    set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
    xlabel('Embedding Time(s)');
    ylabel('Log of the Normalized Storage Cost');
    grid
    hold off;
    saveas(gcf, [name_path,'/TimeEmbedding_' Dataset,...
        '_noise' num2str(noise),...
        '_K', num2str(K),...
        '_Tr', num2str(classsize_list(iClass)),...
        '_shape',name_shape,...
        '.pdf']);
end

%% tau Effect
% iClass is a list that measures the number of data from each class  
for iClass = 1
    %close all;
    figure('WindowState','maximized')
    hold on;
%     for lId=1:length(classsize_list)
%         j=3;[~, idx]=sort(ResStor(j,:,iClass));plot(ResStor(j,idx,lId), smooth(1-mean(ResErr(j,idx,lId,:),4)),  '-*', 'linewidth', 4);
%         legTxt{lId}=['1WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
%     end
    for lId=1:length(classsize_list)
% j=4;[~, idx]=sort(ResStor(j,:,iClass));errorbar(ResStor(j,idx,1), 1-mean(ResErr(j,idx,1,:),4),std(ResErr(j,idx,1,:),[],4),  '-*', 'linewidth', 4);
        j=4;[~, idx]=sort(ResStor(j,:,iClass));semilogx(ResStor(j,idx,lId), smooth(1-mean(ResErr(j,idx,lId,:),4)),  '-*', 'linewidth', 4,'DisplayName','2WTTDA'); 
        % (\lambda=',num2str(classsize_list(lId)),')'];
    end
    for lId=1:length(classsize_list)
% j=1;[~, idx]=sort(ResStor(j,:,iClass));errorbar(ResStor(j,idx,1), 1-mean(ResErr(j,idx,1,:),4),std(ResErr(j,idx,1,:),[],4),  '-*', 'linewidth', 4);
        j=1;[~, idx]=sort(ResStor(j,:,iClass));semilogx(ResStor(j,idx,lId), smooth(1-mean(ResErr(j,idx,lId,:),4)),  '-*', 'linewidth', 4,'DisplayName','3WTTDA');  
        % (\lambda=',num2str(classsize_list(lId)),')'];
    end
%     j=2;[~, idx]=sort(ResStor(j,:,iClass));plot(ResStor(j,idx,iClass), (1-ResErr(j,idx,iClass)),  '-*', 'linewidth', 4); 
%     j=3;[~, idx]=sort(ResStor(j,:,iClass));plot(ResStor(j,idx,iClass), smooth(1-(ResErr(j,idx,iClass)) ), '-*', 'linewidth', 4); 
%     scatter(0,1-ResErr(2,end, iClass), 'k*')
    for lId=1:length(classsize_list)
% j=5;[~, idx]=sort(ResStor(j,:,iClass));errorbar(ResStor(j,idx,1), 1-mean(ResErr(j,idx,1,:),4),std(ResErr(j,idx,1,:),[],4),  '-*', 'linewidth', 4);
        j=5;[~, idx]=sort(ResStor(j,:,iClass));plot(ResStor(j,idx,lId), smooth(1-mean(ResErr(j,idx,lId,:),4) ),'-*', 'linewidth', 4,'DisplayName','LDA'); 
        % (\tau=',num2str(classsize_list(lId)),')'];
    end
    legend('Location','southeast');
    fig=gcf;
    set(findall(fig,'-property','FontSize'),'FontSize',28)
    set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
    xlabel('Log of the Normalized Storage Cost');
    ylabel('Classification Accuracy');
    grid
    hold off;
    saveas(gcf, [name_path, '/', Dataset, '_noise' num2str(noise),...
        '_K', num2str(K),...
        '_Tr', num2str(classsize_list(iClass)),...
        '_shape',name_shape,...
        '.pdf']);
end

%%
% for iClass = 1
%     %close all;
%     figure()
%     hold on;
%     ResStor(ResStor>1)=1;
%     for lId=1:length(classsize_list)
%         j=4;plot(ResStor(j,:,lId), ResTime_mat(j,:,lId),  '-*','linewidth', 4); 
%         legTxt{lId}=['2WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
%     end
%     for lId=1:length(classsize_list)
%         j=1;plot(ResStor(j,:,lId), ResTime_mat(j,:,lId),  '-*','linewidth', 4); 
%         legTxt{lId+length(classsize_list)}=['3WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
%     end
%     legend(legTxt,'Location','southeast');
%     fig=gcf;
%     set(findall(fig,'-property','FontSize'),'FontSize',28)
%     set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
%     %axis([0 1 0 2000]);
%     ylabel('Matrix Product Time(s)');
%     xlabel('Log of the Normalized Storage Cost');
%     grid
%     hold off;
% end

%%
% for iClass = 1
%     %close all;
%     figure()
%     hold on;
%     ResStor(ResStor>1)=1;
%     for lId=1:length(classsize_list)
%         j=4;plot(ResStor(j,:,lId), ResTime_search(j,:,lId),  '-*','linewidth', 4); 
%         legTxt=[legTxt,'2WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
%     end
%     for lId=1:length(classsize_list)
%         j=1;plot(ResStor(j,:,lId), ResTime_search(j,:,lId),  '-*','linewidth', 4); 
%         legTxt=[legTxt,'2WTTDA (\lambda=',num2str(classsize_list(lId)),')'];
%     end
%     legend(legTxt,'Location','southeast');
%     fig=gcf;
%     set(findall(fig,'-property','FontSize'),'FontSize',28)
%     set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
%     %axis([0 1 0 2000]);
%     ylabel('Matrix Product Time(s)');
%     xlabel('Log of the Normalized Storage Cost');
%     grid
%     hold off;
% end

%%
% close all;
% tpname = [name_path, '/', Dataset, '_noise' num2str(noise),...
%         '_K', num2str(K),...
%         '_Tr', num2str(classsize_list(iClass)),...
%         '_shape',name_shape,...
%         '.mat'];
% save(tpname);

%% DataSize Effect
%{
for itau = 1: length(tau_list)
    close all;
    figure(2);
    hold on;
    for j=[1,3,4]
        plot(classsize_list, T2V(ResErr(j,itau,:)), '-', 'linewidth', 3);
    end
    legend('ATN-NPE','TNPE','KNN');
    fig=gcf;
    set(findall(fig,'-property','FontSize'),'FontSize',28)
    set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
    axis([min(classsize_list) max(classsize_list) 0 0.6]);
    xlabel('Training Sample Size');
    ylabel('Classification Error');
    hold off;
    saveas(gcf, ['Weizmann', '_K', num2str(K_idx), '_tau', num2str(tau_list(itau)),  '.pdf']);
end
%}
end





    
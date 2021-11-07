function mdaFigurePlot(rList, repeats, Dataset, K_list, K, Storage, tensor_shape)
%% Path Setup
% addpath 'tensorlab'     % Tens2mat
% addpath 'FOptM-share'   % Convex under unitary constraint
% addpath 'mdmp'          % Tensor Trace
% addpath 'TT_TensNet'    % Merge the proposed networks
% addpath 'TT_Approximate'% Approximation Algorithm
% addpath 'TNPE'          % TNPE algorithm package
% addpath 'Self_Tool'     % manifold dimensional and KNN clasifier

%% Algo(App, TN, TNPE, KNN) * tau * classsize * iter;
path = ['mdaResult_' Dataset];
cd(path);
ResErr=nan(2,length(rList),repeats);
ResStor=ResErr;
ResTime_subspace=ResErr;
ResTime_embedding=ResErr;
ResTime_classify=ResErr;
for rId=1:size(rList,2)
    for iIter=1:repeats
        filename = ['Result', ...
            '_repeat' num2str(iIter),...
            '_ranks' num2str(prod(rList(:,rId))/prod(tensor_shape)),...
            '_hout' num2str(1/72),...
            '.mat'];
        load(filename);
        ResErr( 1,rId,iIter )             = DGTDA.PreErr(K_list(K_list == K));
        ResStor( 1,rId,iIter )            = (DGTDA.Storage/Storage);
        ResTime_subspace( 1,rId,iIter )   = DGTDA.time_subspace;
        ResTime_embedding( 1,rId,iIter )  = DGTDA.time_embedding;
        ResTime_classify( 1,rId,iIter )   = DGTDA.time_classify;
        
        
        ResErr( 2,rId,iIter )             = CMDA.PreErr(K_list(K_list == K));
        ResStor( 2,rId,iIter )            = (CMDA.Storage/Storage);
        ResTime_subspace( 2,rId,iIter )   = CMDA.time_subspace;
        ResTime_embedding( 2,rId,iIter )  = CMDA.time_embedding;
        ResTime_classify( 2,rId,iIter )   = CMDA.time_classify;
    end
end
cd ..;

% ResErr=mean(ResErr,3);
ResStor=mean(ResStor,3);
ResTime_subspace=mean(ResTime_subspace,3);
ResTime_embedding=mean(ResTime_embedding,3);
ResTime_classify=mean(ResTime_classify,3);

save('mdaRes.mat','ResErr','ResStor','ResTime_subspace','ResTime_embedding','ResTime_classify');
figure(1)
hold on
plot(mean(ResTime_classify(1,:,:),3), ResStor(1,:), '-*','linewidth', 4,'DisplayName','DGTDA');
plot(mean(ResTime_classify(2,:,:),3), ResStor(2,:), '-*','linewidth', 4,'DisplayName','CMDA');
xlabel('Classification Time(s)');
ylabel('Log of the Normalized Storage Cost');
% grid
%     legend('TTLDA (\lambda=10)','TTLDA (\lambda=100)','TTLDA (\lambda=1000)',['TTLDA-ATN'],['TTNPE-ATN'],'LDA','DGTDA','CMDA','Location','southeast');
%legend('DGTDA','CMDA','Location','southeast');

figure(2)
hold on
plot( ResStor(1,:), mean(ResTime_subspace(1,:,:),3), '-*','linewidth', 4,'DisplayName','DGTDA');
plot( ResStor(2,:), mean(ResTime_subspace(2,:,:),3), '-*','linewidth', 4,'DisplayName','CMDA');
% errorbar( ResStor(1,:), mean(ResTime_subspace(1,:,:),3),std(ResTime_subspace(1,:,:),[],3), '-*','linewidth', 4);
% errorbar( ResStor(2,:), mean(ResTime_subspace(2,:,:),3),std(ResTime_subspace(2,:,:),[],3), '-*','linewidth', 4);
xlabel('Log of the Normalized Storage Cost');
ylabel('Subspace Time(s)');
% grid
%     legend('TTLDA (\lambda=10)','TTLDA (\lambda=100)','TTLDA (\lambda=1000)',['TTLDA-ATN'],['TTNPE-ATN'],'LDA','DGTDA','CMDA','Location','southeast');
%legend('DGTDA','CMDA','Location','southeast');

figure(3)
hold on
plot(mean(ResTime_embedding(1,:,:),3), ResStor(1,:), '-*','linewidth', 4,'DisplayName','DGTDA');
plot(mean(ResTime_embedding(2,:,:),3), ResStor(2,:), '-*','linewidth', 4,'DisplayName','CMDA');
xlabel('Projection Time(s)');
ylabel('Log of the Normalized Storage Cost');
% grid
%     legend('TTLDA (\lambda=10)','TTLDA (\lambda=100)','TTLDA (\lambda=1000)',['TTLDA-ATN'],['TTNPE-ATN'],'LDA','DGTDA','CMDA','Location','southeast');
%legend('DGTDA','CMDA','Location','southeast');

figure(4)
hold on

[~, idx]=sort(ResStor(1,:));plot(ResStor(1,idx), smooth(1-mean(ResErr(1,idx,:),3)), '-*','linewidth', 4,'DisplayName','DGTDA');
[~, idx]=sort(ResStor(2,:));plot(ResStor(2,idx), smooth(1-mean(ResErr(2,idx,:),3)), '-*','linewidth', 4,'DisplayName','CMDA');
% [~, idx]=sort(ResStor(1,:));errorbar(ResStor(1,idx), 1-mean(ResErr(1,idx,:),3), std(ResErr(1,idx,:),[],3),'-*','linewidth', 4);
% [~, idx]=sort(ResStor(2,:));errorbar(ResStor(2,idx), 1-mean(ResErr(2,idx,:),3), std(ResErr(2,idx,:),[],3), '-*','linewidth', 4);
xlabel('Log of the Normalized Storage Cost');
ylabel('Classification Accuracy');
% grid
%     legend('TTLDA (\lambda=10)','TTLDA (\lambda=100)','TTLDA (\lambda=1000)',['TTLDA-ATN'],['TTNPE-ATN'],'LDA','DGTDA','CMDA','Location','southeast');
%legend('DGTDA','CMDA','Location','southeast');
end






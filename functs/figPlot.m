function figPlot(p)
% figPlot(p)
% Function that plots the final results for all methods.

ResErr             = nan(8, length(p.tau_list),length(p.classsize_list), p.repeats, size(p.nAbl,1)); 
ResStor            = ResErr;
ResTime_subspace   = ResErr;
path = p.pathname;
K_idx = find(p.K_list == 1);  % p.K neighbors for classification
cd(path);
for iTau = 1: length(p.tau_list)
    %disp(['iTau ' num2str(iTau) ]);
    for iClass = 1: length(p.classsize_list)
        %disp(['iClass ' num2str(iClass) ]);
        for iIter = 1:p.repeats
            for aId = 1:size(p.nAbl,1)
                filename= ['Result'... % num2str(p.noise), ...
                    '_repeat' num2str(iIter),...
                    '_tau' num2str(p.tau_list(iTau)),... 
                    '_holdout' num2str(p.classsize_list(iClass)/p.nSamp),......
                    '_ablation' num2str(aId),...
                    '_shape',num2str(p.tensor_shape) ...
                    '.mat'];
                load(filename);
                ResErr(1, iTau, iClass,iIter,aId)           = ThW.PreErr(end);
                ResStor(1, iTau,iClass,iIter,aId)           = ThW.Storage(1);
                ResTime_subspace(1,iTau,iClass,iIter,aId)   = sum(ThW.time_subspace);


                ResErr(2,iTau,iClass,iIter,aId)             = TWTTDA.PreErr(end);
                ResStor(2,iTau,iClass,iIter,aId)            = TWTTDA.Storage(1);
                ResTime_subspace(2,iTau,iClass,iIter,aId)   = sum(TWTTDA.time_subspace);

%                 ResErr(3, iTau,iClass,iIter,aId)            = LDA.PreErr;
%                 ResStor(3,iTau,iClass,iIter,aId)            = LDA.Storage;
%                 ResTime_subspace(3,iTau,iClass,iIter,aId)   = LDA.time_subspace;
%                 
                ResErr(4, iTau,iClass,iIter,aId)            = BTT.PreErr;
                ResStor(4,iTau,iClass,iIter,aId)            = BTT.Storage;
                ResTime_subspace(4,iTau,iClass,iIter,aId)   = BTT.time_subspace;
                
                ResErr(5, iTau,iClass,iIter,aId)            = TTDA.PreErr;
                ResStor(5,iTau,iClass,iIter,aId)            = TTDA.Storage;
                ResTime_subspace(5,iTau,iClass,iIter,aId)   = TTDA.time_subspace;

                ResErr(6, iTau,iClass,iIter,aId)            = MPS.PreErr;
                ResStor(6,iTau,iClass,iIter,aId)            = MPS.Storage;
                ResTime_subspace(6,iTau,iClass,iIter,aId)   = MPS.time_subspace;
                
                ResErr(7, iTau,iClass,iIter,aId)            = TWBTT.PreErr;
                ResStor(7,iTau,iClass,iIter,aId)            = TWBTT.Storage;
                ResTime_subspace(7,iTau,iClass,iIter,aId)   = TWBTT.time_subspace;
                
                ResErr(8, iTau,iClass,iIter,aId)            = ThWBTT.PreErr;
                ResStor(8,iTau,iClass,iIter,aId)            = ThWBTT.Storage;
                ResTime_subspace(8,iTau,iClass,iIter,aId)   = ThWBTT.time_subspace;
            end
        end
    end
end
cd ..;

%%
ResStor= permute(mean(permute(ResStor,[4,1,2,3,5]),1), [2,3,4,5,1]);
ResTime_subspace = permute(mean(permute(ResTime_subspace,[4,1,2,3,5]),1), [2,3,4,5,1]);


%% subspace Time
figure
hold on;
for iClass = 1: length(p.classsize_list)
    for aId = 1:size(p.nAbl,1)
        %close all;
        ResStor(ResStor>1)=1;
        j=1;[~, idx]=sort(ResStor(j,:,iClass,aId));
        loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
            '--*','linewidth', 4,'DisplayName','3WTTDA','MarkerSize',15);
        j=8;[~, idx]=sort(ResStor(j,:,iClass,aId));
        loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
            '--*','linewidth', 4,'DisplayName','3WBTT','MarkerSize',15);
        j=2;[~, idx]=sort(ResStor(j,:,iClass,aId));
        loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
            '--*','linewidth', 4,'DisplayName','2WTTDA','MarkerSize',15);
        j=7;[~, idx]=sort(ResStor(j,:,iClass,aId));
        loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
            '--*','linewidth', 4,'DisplayName','2WBTT','MarkerSize',15);
%         j=3;[~, idx]=sort(ResStor(j,:,iClass,aId));
%         loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
%             '-*','linewidth', 4,'DisplayName','LDA');
        j=5;[~, idx]=sort(ResStor(j,:,iClass,aId));
        loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
            '--*','linewidth', 4,'DisplayName','TTDA','MarkerSize',15);
        j=4;[~, idx]=sort(ResStor(j,:,iClass,aId));
        loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
            '--*','linewidth', 4,'DisplayName','BTT','MarkerSize',15);
%         j=6;[~, idx]=sort(ResStor(j,:,iClass,aId));
%         loglog(ResStor(j,idx,iClass,aId), ResTime_subspace(j,idx,iClass,aId),...
%             '-*','linewidth', 4,'DisplayName','MPS');
        legend('Location','northwest');
        fig=gcf;
        fig.CurrentAxes.XScale = 'log';
        fig.CurrentAxes.YScale = 'log';
        set(findall(fig,'-property','FontSize'),'FontSize',28)
        set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
        ylabel('Subspace Time(s)');
        xlabel('Normalized Storage Cost');
        grid
        hold off;
    end
end

%% tau Effect
figure,
hold on;
% iClass is a list that measures the number of data from each class  
for iClass = 1: length(p.classsize_list)
    for aId = 1:size(p.nAbl,1)
        %close all;
        j=1;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '--*', 'linewidth', 4,'DisplayName','3WTTDA','MarkerSize',15);
        j=8;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '--*', 'linewidth', 4, 'DisplayName', '3WBTT','MarkerSize',15);
        j=2;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '--*', 'linewidth', 4,'DisplayName','2WTTDA','MarkerSize',15);
        j=7;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '--*', 'linewidth', 4, 'DisplayName', '2WBTT','MarkerSize',15);
%         j=3;[~, idx]=sort(ResStor(j,:,iClass,aId));
%         semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
%             '-*', 'linewidth', 4, 'DisplayName', 'LDA');
        j=5;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '--*', 'linewidth', 4, 'DisplayName', 'TTDA','MarkerSize',15);
        j=4;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '--*', 'linewidth', 4, 'DisplayName', 'BTT','MarkerSize',15);
        j=6;[~, idx]=sort(ResStor(j,:,iClass,aId));
        semilogx(ResStor(j,idx,iClass,aId), (1-mean(ResErr(j,idx,iClass,:,aId),4)),...
            '-*', 'linewidth', 4, 'DisplayName', 'MPS');
        legend('Location','southeast');
        fig=gcf;
        fig.CurrentAxes.XScale = 'log';
        set(findall(fig,'-property','FontSize'),'FontSize',28)
        set(findall(fig,'-property','FontName'),'FontName','Times New Roman')
        xlabel('Normalized Storage Cost');
        ylabel('Classification Accuracy');
        grid
        hold off;
    end
end


end  
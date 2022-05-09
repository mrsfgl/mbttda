%% Utility script creating a \Latex table.
Dataset         = 'YALE'; % dataset to be tested
[images, labels,K_list,tau_list,tau_list2,tensor_shape,classsize_list,class,lambda_list,lambda_list2] = load_data(Dataset);
rS=[5,2,2,5;4,2,2,4;4,2,2,3;3,3,1,3];
repeats=10;
pathname='C:\Users\SeyyidEmre\Documents\MATLAB\TTNPE-master\mytauResult_YALE';
for i=1:repeats
    for j=1:length(classsize_list)
        for k=1:3
            filename= [pathname,'\Result_', ...
                'repeat' num2str(i),... %                 '_tau' num2str(tau_list(tauId)),...
                '_holdout' num2str(classsize_list(j)/72),...
                '_lambda' num2str(lambda_list2(k)) ...
                '.mat'];
            load(filename);
            stTTDA(i,j,k,:)=TTLDA.Storage(:);
            
            resTTDA(i,j,k,:)=TTLDA.PreErr(:);
        end
        stTTDA_ATN(i,j)=App.Storage;
        stLDA(i,j)=LDA.LDAStorage;
        stTTNPE(i,j)=App2.Storage;
        stKNN(i,j)=size(App.Vrn,1)*(classsize_list(j)*class);
        
        resTTDA_ATN(i,j)=App.PreErr;
        resLDA(i,j)=App.LDAPreErr;
        resTTNPE(i,j)=App2.PreErr;
    end
end

strg=[mean(stKNN,1)./mean(stKNN,1);
    mean(stLDA,1)./mean(stKNN,1);
    mean(stTTNPE,1)./mean(stKNN,1);
    mean(stTTDA_ATN,1)./mean(stKNN,1);
    reshape(mean(stTTDA(:,:,:),1),4,3)'./repmat(mean(stKNN,1),3,1)];

acc=1-[mean(resLDA,1);mean(resTTNPE,1);mean(resTTDA_ATN,1);reshape(mean(resTTDA(:,:,:,1),1),4,3)'];
stdacc=[std(resLDA,1);std(resTTNPE,1);std(resTTDA_ATN,1);reshape(std(resTTDA(:,:,:,1),1),4,3)'];

pathname='C:\Users\SeyyidEmre\Documents\MATLAB\TTNPE-master\mdarankResult_YALE';
for i=1:repeats
    for j=1:length(rS(:,1))
        filename = [pathname '/Result', ...
            '_repeat' num2str(i),...
            '_ranks' num2str(j),...
            '.mat'];
        load(filename);
        stCMDA(i,j)=CMDA.Storage;
        stDGTDA(i,j)=DGTDA.Storage;
        resCMDA(i,j)=CMDA.PreErr;
        resDGTDA(i,j)=DGTDA.PreErr;
    end
end
acc=[acc;1-mean(resCMDA,1);1-mean(resDGTDA,1)]
stdacc=[stdacc;std(resCMDA,1);std(resDGTDA,1)]
strg=[strg;
    mean(stCMDA,1)./mean(stKNN,1);
    mean(stDGTDA,1)./mean(stKNN,1);]

for i=1:size(acc,1)
    tabrow{i}=[];
    for j=1:size(acc,2)-1
        tabrow{i}=[tabrow{i},'$',num2str(100*acc(i,j),3),' \pm ',num2str(100*stdacc(i,j),3),'$',' & '];
    end
    j=size(acc,2);
    tabrow{i}=[tabrow{i},'$',num2str(100*acc(i,j),3),' \pm ',num2str(100*stdacc(i,j),3),'$',' \\'];
end
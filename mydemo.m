clear
clc

%% parameter setup
Dataset               = 'COIL'; % dataset to be tested
pathname              = ['result_full_',Dataset];% the folder to save the results

%% data and path processing
[images, labels, pList] = load_data(Dataset);
if ~exist(pathname)
    mkdir(pathname);% build the result folder for the testing
end
% t_list2=[0.053,0.065,0.08,0.09375];
% ldalist=[0.2,0.2,0.2,0.15];
nVal = 5;

pList.pathname = pathname; % dataset to be tested
pList.noise = 0; % noised to the input data
pList.repeats = 10;% the amount of iteration to get the average results
pList.nSamp = sum(labels==labels(1));
lambda = cell(1,pList.repeats);
acc = lambda;

%% Running results
for repeatId = 1: pList.repeats % each repeat
    for cId = 1:length(pList.classsize_list) % each training size
        data = get_data(images, labels, pList, cId);% get train/test data
        [lambda{repeatId}, acc{repeatId}, data] =  optLambda(data, pList.tensor_shape, pList.classsize_list(cId), nVal);
        for aId = 1:size(pList.nAbl,1)
            % LDA parameter
            para_LDA = [];
            para_LDA.lambda = lambda{repeatId}(1);
            
            % TTDA parameter
            para_TTDA = [];
            para_TTDA.maxiter    = 200;
            para_TTDA.maxiterOut = 1;
            para_TTDA.error_tot  = 10 ^ -1;
            para_TTDA.display    = 0;
            para_TTDA.display2   = 1;
            para_TTDA.Graph.K    = pList.class*pList.classsize_list(cId)-1;
            para_TTDA.Graph.epsilon = 1;
            para_TTDA.lambda     = lambda{repeatId}(1);
            para_TTDA.I          = pList.tensor_shape;
            para_TTDA.nA         = pList.nAbl(aId,:);
            para_TTDA.rndI.flag  = false;
            % 2TTDA parameter
            para_2TTDA = [];
            para_2TTDA.maxiter    = 200;
            para_2TTDA.maxiterOut = 1;
            para_2TTDA.error_tot  = 10 ^ -1;
            para_2TTDA.display    = 0;
            para_2TTDA.display2   = 1;
            para_2TTDA.lambda     = lambda{repeatId}(2);
            para_2TTDA.I          = pList.tensor_shape;
            para_2TTDA.nA         = pList.nAbl(aId,:);
            para_2TTDA.rndI.flag  = false;
            % 3TTDA parameter
            para_3TTDA = [];
            para_3TTDA.maxiter    = 200;
            para_3TTDA.maxiterOut = 1;
            para_3TTDA.error_tot  = 10 ^ -1;
            para_3TTDA.display    = 0;
            para_3TTDA.display2   = 1;
            para_3TTDA.lambda     = lambda{repeatId}(3);
            para_3TTDA.I          = pList.tensor_shape;
            para_3TTDA.nA         = pList.nAbl(aId,:);
            para_3TTDA.rndI.flag  = false;
            for tauId = 1: length(pList.tau_list) % each thresholding tau
                para_LDA.tau        = pList.tau_list(tauId);
                para_TTDA.tau       = pList.tau_list2(tauId);
                para_2TTDA.tau      = pList.tau_list3(tauId);
                para_3TTDA.tau      = pList.tau_list4(tauId);
                
                showinfo = ['At ' num2str(repeatId) ' iterations, '...
                    'tau ' num2str(tauId) ' ablation ' num2str(aId) ...
                    ', class size ' num2str(cId)];
                disp(['********** ' showinfo ' *************' ]);
                % save file
                filename = [pathname '/Result' ...
                    '_repeat' num2str(repeatId),...
                    '_tau' num2str(pList.tau_list(tauId)),...
                    '_holdout' num2str(pList.classsize_list(cId)/pList.nSamp),...
                    '_ablation' num2str(aId),...
                    '_shape' num2str(pList.tensor_shape), ...
                    '.mat'];
                m = matfile(filename,'writable',true);
                
                m.TTNPE  = main_App(data, para_TTDA);
                m.LDA    = myLDA(data, para_LDA);
                m.MPS    = mps(data, para_2TTDA);
                m.BTT    = btt(data, para_TTDA);
                m.TWBTT  = tw_btt(data, para_2TTDA);
                m.ThWBTT = thw_btt(data, para_2TTDA);
                m.TTDA   = ttda(data, para_TTDA);
                m.TWTTDA = twttda(data, para_2TTDA);
                m.ThW    = thwttda(data, para_3TTDA);
            end
        end
    end
end
%% plot results
figPlot(pList)
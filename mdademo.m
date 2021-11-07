clear
clc

%% parameter setup
Dataset         = 'Weizmann'; % dataset to be tested
noise           = 0; % noised to the input data
repeats         = 10;% the amount of iteration to get the average results
pathname        = ['mdaResult_', Dataset];% the folder to save the results
patches         = false;
withRank        = false;

%% data and path processing
[images, labels,rS,tensor_shape,image_shape,class,classSize] = load_dataMDA(Dataset, withRank);
if ~exist(pathname)
    mkdir(pathname);% build the result folder for the testing
end



% rS=[5,2,2,5;4,2,2,4;4,2,2,3;3,3,1,3]';
for repeatId = 1: repeats % each repeat
    for rId = 1:size(rS,2) % each thresholding tau
        for cId=1:length(classSize)
            %display information
            showinfo = ['At ' num2str(repeatId) ' iterations, '...
                ', rank, ' num2str(rId) ', holdout', num2str(classSize(cId)/(size(labels,2)/class))];
            disp(['********** ' showinfo ' *************' ]);
            
            % tunable parameter setup
            ranks         = rS(:,rId)';
            [train_data, train_label, test_data, test_label] =get_data(images, labels, class,classSize(cId),noise);% get train/test data
            
            
            if patches
                if strcmp(Dataset,'GAIT')
                    train_data = ext3Dpatches( train_data, image_shape, classSize(cId), class );
                    test_data = ext3Dpatches( test_data, image_shape, sum(test_label==1), class );
                    tensor_shape(4) = size(train_data,4);
                else
                    Data=train_data;
                    clear train_data
                    for i=1:class
                        for j=1:classSize(cId)
                            tempPatch = im2col(reshape(Data(:,j+(i-1)*(classSize(cId))),image_shape),tensor_shape(1:2),'distinct');
                            train_data(:,:,:,j,i) = reshape(tempPatch,tensor_shape);
                        end
                    end
                    Data=test_data;
                    clear test_data
                    cS=length(test_label)/class;
                    for i=1:class
                        for j=1:cS
                            tempPatch = im2col(reshape(Data(:,j+(i-1)*cS),image_shape),tensor_shape(1:2),'distinct');
                            test_data(:,:,:,j,i) = reshape(tempPatch,tensor_shape);
                        end
                    end
                end
            end
            
            
            KNN.K         = 1;
            Storage=numel(train_data);
            % Algo1:    Approximate TTNPE embedding solver
            DGTDA = main_DGTDA(train_data, train_label, test_data,...
                test_label, tensor_shape, ranks, KNN, patches);
            CMDA  = main_CMDA(train_data, train_label, test_data,...
                test_label, tensor_shape, ranks, KNN, patches);
            % save file
            filename = [pathname '/Result', ...
                '_repeat' num2str(repeatId),...
                '_ranks' num2str(prod(rS(:,rId))/prod(tensor_shape)),...
                '_hout' num2str(cId/72),...
                '.mat'];
            save(filename, 'DGTDA','CMDA');
        end
    end
end

% mdaFigurePlot(1:size(rS,2), repeatId, Dataset, KNN.K, 1, Storage)
mdaFigurePlot(rS, repeats, Dataset, KNN.K, 1, Storage, tensor_shape)
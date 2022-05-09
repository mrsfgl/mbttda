clear 
close 
clc

%% parameter setup
Dataset         = 'COIL'; % dataset to be tested
noise           = 0; % noised to the input data           
repeats         = 10;% the amount of iteration to get the average results
pathname        = ['mdaResult_', Dataset];% the folder to save the results

%% data and path processing
[images, labels,K_list,tau_list,tau_list2,tensor_shape,classsize_list,class,lambda_list,lambda_list2] = load_data(Dataset);
if ~exist(pathname)
    mkdir(pathname);% build the result folder for the testing
end

for repeatId = 1: repeats % each repeat
    for rId = 1: size(rS,2) % each thresholding tau
              %display information
            showinfo = ['At ' num2str(repeatId) ' iterations, '...
                    'tau ' num2str(rId)];
            display(['********** ' showinfo ' *************' ]);
            
            % tunable parameter setup
            ranks         = rS(:,rId)';
            class_size  = classsize_list(1);
            [train_data, train_label, test_data, test_label] =get_data(Dataset, images, labels, class,class_size,noise);% get train/test data
            
            KNN.K               = K_list;
            
            % Algo1:    Approximate TTNPE embedding solver
            MDA = main_MDA(train_data, train_label, test_data,...
                test_label, tensor_shape, ranks,KNN);
            % save file
            filename = [pathname '/Result', ...
                '_repeat' num2str(repeatId),...
                '_ranks' num2str(rId),...
                '.mat'];
            save(filename, 'MDA');
        end
    end
end
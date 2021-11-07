function data = get_data(images, labels, p, id)
    train_idx   = [];
    test_idx    = [];
    
    for i = 1: p.class
        idlist    = find(labels==i);
        tplist    = randsample(idlist, p.classsize_list(id));
        train_idx = [train_idx, tplist];
        test_idx  = [test_idx, setdiff(idlist, tplist)];
    end
    data.train  = images(:, train_idx);
    data.trLbl  = labels(train_idx);
    data.test   = images(:, test_idx);
    data.tsLbl  = labels(test_idx);            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % add noise to data
    Ntr        = randn(size(data.train));
    data.train = double(data.train)/255 + p.noise * Ntr/norm(T2V(Ntr));
    Ntst       = randn(size(data.test));
    data.test  = double(data.test)/255  + p.noise * Ntst/norm(T2V(Ntst));
end
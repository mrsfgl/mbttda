function [opL, acc, data] = optLambda(data, s, c, n)
% [opL, acc, data] = optLambda(data, s, c, n)
% Validation function that finds optimal lambda
% for each method.

lList        = 10.^(-2:1:4);
p.maxiter    = 50;
p.maxiterOut = 1;
p.error_tot  = 10 ^ -1;
p.display    = 0;
p.display2   = 0;
p.tau        = 0.2;
p.nA         = zeros(1, length(s));
p.rndI.flag  = false;
nTests       = 5;

if n>c
    error('Val size should be lesser than Train size!')
end

s(end+1)=c;
s(end+1)=size(data.train,2)/c;
l = length(s);
p.I = s(1:l-2);

data.test = reshape(data.test,prod(s(1:l-2)),[],s(l));
data.tsLbl = reshape(data.tsLbl,[],s(l));

data.train = cat(2,reshape(data.train,[prod(s(1:l-2)),c,s(l)]),data.test(:,1:n,:));
data.trLbl = cat(1,reshape(data.trLbl,c,s(l)),data.tsLbl(1:n,:));

data.test = reshape(data.test(:,n+1:end,:),prod(s(1:l-2)),[]);
data.tsLbl = reshape(data.tsLbl(n+1:end,:),1,[]);

acc  = zeros(nTests, length(lList), 2);
for i=1:nTests
    ind  = randperm(c+n,c);
    vl.train = data.train(:,ind,:); vl.train = reshape(vl.train,[],c*s(l));
    vl.test  = data.train(:,setdiff(1:c+n,ind),:); vl.test = reshape(vl.test,[],n*s(l));
    vl.trLbl = reshape(data.trLbl(1:c,:),1,[]);
    vl.tsLbl = reshape(data.trLbl(c+1:c+n,:),1,[]);
    for lId=1:length(lList)
        p.lambda=lList(lId);
        
        t  = ttda(vl, p);
        t2 = twttda(vl, p);
        t3 = thwttda(vl, p);
        acc(i, lId, 1) = 1-t.PreErr;
        acc(i, lId, 2) = 1-t2.PreErr;
        acc(i, lId, 3) = 1-t3.PreErr;
    end
end
[~, lambdaIn] = sort(mean(acc(:,:,1)));
opL = lList(lambdaIn(end));
[~, lambdaIn] = sort(mean(acc(:,:,2)));
opL(2) = lList(lambdaIn(end));
[~, lambdaIn] = sort(mean(acc(:,:,3)));
opL(3) = lList(lambdaIn(end));

data.train = reshape(data.train, prod(p.I),[]);
data.trLbl = data.trLbl(:)';
end
function [Ui, Plot] = TensNet_Solver(Ui, Z, param, varargin)
%% 
%

% initialize the parameters of the line search algorithm.
if ~isempty(varargin)
    opts = varargin{1};
else
    opts.record = 0;
    opts.mxitr  = 1000;
    opts.gtol   = 1e-5;
    opts.xtol   = 1e-5;
    opts.ftol   = 1e-8;
    opts.tau    = 1e-3;
end
%  Solver
n           = length(Ui);
iter        = 0;
Last        = zeros(size(Ui{n}));
Plot        = ObjVal(Z, Ui, n);
s=size(Z);
sU=zeros(3,n);
for i=1:n
    sU(:,i)=[size(Ui{i})'; ones(3-ndims(Ui{i}),1)];
end

while true
    for i =1 : n
        % solve Ui
        if i==1 && i~=n
            if (prod([sU(1,i+1)^2,s(i+1:end/2).^2,sU(3,n)])+prod(s)*sU(1,i+1))>(prod([s,sU(1,i+1),sU(3,n)])+prod([s.^(1/2),sU(1,i+1)^2,sU(3,n)^2,s(1:i)]))
                Tn = merge_tensor(Ui(2:end),false);
                A  = tens2mat(diagsum(wwq_tensordot(wwq_tensordot(Tn, Z, 2:n, n+2:2*n), Tn, 2+(2:n),2:n), 2, 6), [2,1], [3,4]);
            else
                Tn = merge_tensor(Ui(2:end),false);
                Tn = permute(wwq_tensordot(Tn,Tn,n+1,n+1),[1:n,2*n:-1:n+1]);
                A  = tens2mat(wwq_tensordot(Z,Tn,[2:n,n+2:2*n],2:(2*n-1)),[1,3],[2,4]);
            end
            [Ui{i}, out] = Ui_Solver(A, size(Ui{i}), opts);
            if param.display
                val = ObjVal(Z, Ui, n);
                disp([out.msg,'    For U' num2str(i), ' the objval is ' num2str(val)]);
                Plot = [Plot,val];
            end
        elseif i==n && i~=1
            T1 = merge_tensor(Ui(1:end-1));
            A  = tens2mat(wwq_tensordot(wwq_tensordot(Z, T1, n+(1:n-1), 1:n-1, false), T1, 1:n-1, 1:n-1, false), [2,3], [1,4]);
            [Ui{i}, out] = Un_Solver(A, size(Ui{i}), opts);
            if param.display
                val = ObjVal(Z, Ui, n);
                disp([out.msg,'    For U' num2str(i), ' the objval is ' num2str(val)]);
                Plot = [Plot,val];
            end
        elseif i==n && i==1
            A=Z;
            [Ui{i}, out] = Un_Solver(A, size(Ui{i}), opts);
            if param.display
                val = ObjVal(Z, Ui, n);
                disp([out.msg,'    For U' num2str(i), ' the objval is ' num2str(val)]);
                Plot = [Plot,val];
            end
        else
            if (prod([sU(1,i+1)^2,s(i+1:end/2).^2,sU(3,n)])+prod(s)*sU(1,i+1))>(prod([s,sU(3,i-1)])+prod([s,sU(1,i+1)^2,1./s(1:i-1)]))
                T1 = merge_tensor(Ui(1:i-1));
                Tn = merge_tensor(Ui(i+1:end),false);
                T1_M   = permute(wwq_tensordot(T1, wwq_tensordot(T1, Z, 1:i-1, 1:i-1,false), 1:i-1, n-i+2+(1:i-1),false),[2:n-i+3,1,n-i+4:2*(n-i+2)]);
                A      = tens2mat(diagsum(wwq_tensordot(Tn, wwq_tensordot(Tn, T1_M, 2:n-i+1, n-i+2+(3:n-i+2)), 2:n-i+1, 2+(3:n-i+2)), 2,4), [3,4,2], [5,6,1]);
            else
                Tn = merge_tensor(Ui(i+1:end),false);
                Tn = permute(wwq_tensordot(Tn,Tn,3,3),[1:n-i+1,2*(n-i+1):-1:n-i+2]);
                T1 = merge_tensor(Ui(1:i-1));
                A  = permute(wwq_tensordot(Z,Tn,[i+1:n,n+i+1:2*n],2:(2*(n-i)+1)),[1:i,2*i+1,i+1:2*i,2*i+2]);
                A  = tens2mat(wwq_tensordot(wwq_tensordot(A, T1, i+1+(1:i-1), 1:i-1, false), T1, 1:i-1, 1:i-1, false),[5,1,2],[3,4,6]);
            end
            
            [Ui{i}, out] = Ui_Solver(A, size(Ui{i}), opts);
            if param.display
                val = ObjVal(Z, Ui, n);
                disp([out.msg,'    For U' num2str(i), ' the objval is ' num2str(val)]);
                Plot = [Plot,val];
            end
        end
    end

    error   = norm(T2V(Ui{n} -Last))/norm(T2V(Last));
    Last    = Ui{n};
    iter    = iter +1;
    
    if param.display
        disp(['At iteration ', num2str(iter), ' the change is ', num2str(error),...
            ' the objval is' num2str(ObjVal(Z, Ui, n))]);
        disp(' ');
    end
        
    if iter >= param.maxiter
        if param.display2
        disp('Hit Max Iteration !');
        end
        break;
    end

    if error <= param.error_tot
        if param.display2
        disp('Converged !');
        end
        break;
    end
end
end
function Ui = U2Ui_tau(U, tau, randinit, f)
% U: I1 * ...* In
% tau: scalar

if nargin<4
    f=1;
end
if nargin<3
    randinit.flag = 0;
end
n   = size(U);
n2  = n;
dim = ndims(U);

if randinit.flag
    r = randinit.rank;
    Ui  = cell(1, dim);
    Ui{1} = reshape(orth(rand(n(1), r(1))), 1, n(1), []);
    for i=2:dim-1
        Ui{i} = reshape(orth(rand(n(i)*r(i-1), r(i))), r(i-1), n(i), r(i));
    end
    Un = L(merge_tensor(Ui(1:dim-1)), dim);
    Ui{dim} = Un'*L(U, dim); 
else
    if f==0
        U = reshape(U,[n(1)*n(2),n(3:end)]);
        t = n(1);
        n   = size(U);
        dim = ndims(U);
    end
    
    Ui = cell(dim, 1);
    r = nan(dim-1, 1);
    
    % Initialize B
    if f==1
        B=ndim_unfold(U,1);
    elseif f==0
        B=ndim_unfold(U,dim)';
    end
    
    % svd for the initialized B
    [u, s, v] = svd(B,'econ');
    
    % Process svd result
    r(1) = max(size(find(diag(s) >= tau * max(diag(s)))));
%     r(1) = min(max(size(find(diag(s) >= tau * max(diag(s))))),n(1));
    %     if r(1)==1
    %         r(1)=2;
    %     end
    
    u = u(:, 1:r(1));
    s = s(1: r(1), 1:r(1));
    v = v(:, 1:r(1));
    if f==1
        v = s * v';
    elseif f==0
        u = u * s;
    end
    
    % store output result
    % Ui{1} = u;
    if f==1
        Ui{1} = reshape(u, [1, size(u)]);
    elseif f==0
        Ui{1} = permute(v, [2,1,3]);
    end
    % main loop
    for i = 2 : dim-1
        if f==1
            B = reshape(v, [r(i-1)*n(i), prod(n(i+1:end))]);
            [u,s,v] = svd(B, 'econ');
            
            % Process svd result
            r(i) = length(find(diag(s) >= tau *max(diag(s))));
%             r(i) = min(max(size(find(diag(s) >= tau * max(diag(s))))), n(i));
            %             if r(i)==n(i)
            %                 disp('Hit!!!')
            %             end
            %             if (r(i)==1)
            %                 r(i)=2;
            %             end
            u = u(:, 1:r(i));
            
            Ui{i} = reshape(u, [r(i-1), n(i), r(i)]);
            s = s(1: r(i), 1:r(i));
            v = v(:, 1:r(i));
            v = s * v';
        elseif f==0
            B = reshape(u, [prod(n(1:i-1)), r(i-1)*n(i)]);
            [u,s,v] = svd(B, 'econ');
            r(i) = length(find(diag(s) >= tau *max(diag(s))));
%             r(i) = min(max(size(find(diag(s) >= tau * max(diag(s))))),n2(i));
            %             if r(i)==n(i)
            %                 disp('Hit!!!')
            %             end
            %             r(i) = max(size(find(diag(s) >= tau * max(diag(s)))));
            %             if (r(i)==1)
            %                 r(i)=2;
            %             end
            u = u(:, 1:r(i));
            s = s(1: r(i), 1:r(i));
            v = v(:, 1:r(i))';
            
            Ui{i} = reshape(v, [r(i), n(i), r(i-1)]);
            u = u * s;
        end
    end
    if isempty(i) i=1; end
    if f==1
        Ui{i+1} = v;
    elseif f==0
        Ui{i+1} = reshape(u,[t,n(1)/2,size(u,2)]);
    end
end
end
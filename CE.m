function [W, obj] = CE(X, v, n, dd, c, m, anchor_view, Z, Z_g, alpha, beta)

%% ===================== initialize =====================
MaxIter = 15;
A = anchor_view;
W = cell(1,v);

Lambda_X = cell(1,v);
Lambda_A = cell(1,v);
M = cell(1,v);
I_d = cell(1,v);
Q = cell(1,v);

for i = 1:v
    W{i} = ones(dd(i),c);
    M{i} = ones(dd(i));
    I_d{i} = eye(dd(i));
    Q{i} = eye(dd(i));
end

%% ===================== updating =====================
for iter = 1:MaxIter
    
    tic
    for iv = 1:v
        ed{iv} = L2_distance_1(W{iv}'*X{iv}, W{iv}'*A{iv});
    end

    % calculate Lambda
    for iv = 1:v
        Lambda_X{iv} = diag(Z{iv}*ones(m,1));
    end
    for iv = 1:v
        Lambda_A{iv} = diag(Z{iv}'*ones(n,1));
    end

    % update W
    for iterv = 1:v
        M{iterv} = X{iterv}*Lambda_X{iterv}*X{iterv}' - X{iterv}*Z{iterv}*A{iterv}' - A{iterv}*Z{iterv}'*X{iterv}' + A{iterv}*Lambda_A{iterv}*A{iterv}' + alpha*Q{iterv};
        [W{iterv},~] = Find_K_Max_Gen_Eigen(-M{iterv}+eps, I_d{iterv}+eps, c);
    end

    % update L2,1-norm
    for iterv = 1:v
        qj{iterv} = sqrt(sum(W{iterv}.*W{iterv},2)+eps);
        q{iterv} = 0.5./qj{iterv};
        Q{iterv} = diag(q{iterv});
    end

    % update A
    for iterv = 1:v
        A{iterv} = X{iterv}*Z{iterv}/Lambda_A{iterv};
    end

    % update Z
    for iterv = 1:v
        temp_Z = zeros(m);
        for j = 1:n
            row_vector = (2*beta*Z_g(j,:) - ed{iterv}(j,:))/(2*beta);
            temp_Z(j,:) = EProjSimplex_new(row_vector);
        end
        Z{iterv} = temp_Z;
    end
    
    toc
    %% ===================== calculate obj =====================
    Term1 = 0;
    Term2 = 0;
    Term3 = 0;

    for objIndex = 1:v
        temp_Term1 = 0;
        for i = 1:n
            for j = 1:m
                temp_Term1 = temp_Term1 + ed{objIndex}(i,j)*Z{objIndex}(i,j);
            end
        end
        Term1 = Term1 + temp_Term1;
        Term2 = Term2 + alpha*sum(sqrt(sum(W{objIndex}.*W{objIndex},2)));
        Term3 = Term3 + beta*norm(Z_g-Z{objIndex}, 'fro').^2;
    end
    tempobj = Term1 + Term2 + Term3;

    obj(iter) = tempobj;
    if iter == 1
        err = 0;
    else
        err = obj(iter)-obj(iter-1);
    end

    fprintf('iteration =  %d: , 1+3: %.4f, obj: %.4f; err: %.4f  \n', ...
        iter, Term1+Term3, obj(iter), err);
end


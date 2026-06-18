clc;
clear;
warning off;
addpath('./datasets/');
addpath(genpath('./Tools/'));

load('MSRC_V1_5views.mat'); dataset_name = 'MSRC_V1_5views'; v = 5; n = 210; dd = [1302,48,512,256,210]; c = 7; for i=1:v X{i}=X{i}'; end

%% Initalize param

param_alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
param_beta = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
m = [1 2 3 4 5 6 8 10]*c;  

iter = 0;
wcj = 1;

% ============================================================================================================
% Normalize data ===== need X{i}: dxn
rng('default');
for i=1:v
    X{i} = NormalizeFea(X{i});
end

% 拼接 X
Xnor = cell(1,v);
for iv = 1:v
    Xnor{iv} = X{iv}';
end
ConsenX = DataConcatenate(Xnor);
% ===========================================================================================================
for mm = m
    % =============================================================================================
    disp('----------init Anchor graph and each view anchor----------');

    tic
    opt.style = 1;
    opt.iterMax = 50;
    opt.toy = 0;
    isGraph = 0;
    [anchor_view, Zv, Z_g] = FastmultiCLR(X, v, mm, opt, isGraph);
    for iv = 1:v
        anchor_view{iv} = anchor_view{iv}';
    end
    toc
    % ===================================================================================================

    for alpha = param_alpha
        for beta = param_beta

            iter = iter + 1;

            tic
            [U, obj] = CE(X, v, n, dd, c, mm, anchor_view, Zv, Z_g, alpha, beta);
            toc

            % 
            for iv = 1:v
                U{iv} = U{iv}';
            end
            W = DataConcatenate(U);
            W = W';
            XX = ConsenX';
            d = size(XX,1);  % 

            %% selection num
            select = 0.15;
            selectedFeas = floor(select*d);

            %% clustering
            w = [];
            for iv = 1:d
                w = [w norm(W(iv,:),2)];
            end
            [~,index] = sort(w,'descend');
            Xw = XX(index(1:selectedFeas),:);
            for i = 1:40
                label=litekmeans(Xw',c,'MaxIter',100,'Replicates',20);
                result1 = ClusteringMeasure(Y,label);
                result(i,:) = result1;
            end
            for j=1:2
                a=result(:,j);
                ll=length(a);
                temp=[];
                for i=1:ll
                    if i<ll-18
                        b=sum(a(i:i+19));
                        temp=[temp;b];
                    end
                end
                [e,f]=max(temp);
                e=e./20;
                MEAN(j,:)=[e,f];
                STD(j,:)=std(result(f:f+19,j));
            end

            fprintf('selectedFeas(per) = %d , ', select*100);
            fprintf('\n');
            disp(['mean. ACC: ', num2str(MEAN(1,1))]);
            disp(['mean. STD(ACC): ', num2str(STD(1,1))]);
            disp(['mean. NMI: ', num2str(MEAN(2,1))]);
            disp(['mean. STD(NMI): ', num2str(STD(2,1))]);
            disp(dataset_name);
            fprintf("alpha: %.4f, beta: %.4f, m: %.1f\n",alpha, beta, mm);
            fprintf("\n");

            temp_result = [MEAN(1,1), STD(1,1), MEAN(2,1), STD(2,1), alpha, beta, mm];
            Fin(wcj, (1:7)) = temp_result;
            wcj = wcj + 1;
        end
    end
end





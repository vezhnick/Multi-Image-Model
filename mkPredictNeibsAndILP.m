function mkPredictNeibsAndILP(MIM, Dataset, k, neibsOutputFile, ilpOutputFile)

% function Graph = mkPredictNeibsAndILP(MIM, Dataset, k, neibsOutputFile, ilpOutputFile)
% 
% Predicts ILP for test images and finds their nearest neighbours in the
% training set, using prelearnt mkl metric
%
% Input:
%     MIM = MIM structure
%     Dataset =  Dataset structure
%     k = number of neighbours
%     neibsOutputFile = file to which neighbours will be saved
%     ilpOutputFile = file to which ILP will be saved
%     Verbose = display or not different stats
% Alexander Vezhnevets, 2012

load (Dataset.ImageIndexFile);
load (MIM.KernelImageWeights);

ImagesDB_train = ImagesDB(Dataset.TrainImageIdx);
ImagesDB_test = ImagesDB(Dataset.TestImageIdx);

%w(1) = 0;
avMeanAP = 0;

TestPrediction = zeros(length(ImagesDB_test), 21);
TestLabels = zeros(length(ImagesDB_test), 21);

label_weights = zeros(1,21);
total_c = zeros(1,21);
% calculating label weights
for im = 1 : length(ImagesDB_train)
    
    cur_im = ImagesDB_train{im};
    for c = cur_im.labels
        if(c ~= 0)
            total_c(c) = total_c(c) + 1;
        end
    end    
end

label_weights = 1 ./ total_c;
label_weights = label_weights / min(label_weights);

total_c_tst = zeros(1,21);
% calculating label weights
for im = 1 : length(ImagesDB_test)
    
    cur_im = ImagesDB_test{im};
    for c = cur_im.labels
        if(c ~= 0)
            total_c_tst(c) = total_c_tst(c) + 1;
        end
    end    
end

accuracy = zeros(21,1);
recall = zeros(21,1);
for tst_im = 1 : length(ImagesDB_test)
    
    kNN = zeros(1,k) + 100;
    dist = zeros(1,k) + 100;
    cur_test_im = ImagesDB_test{tst_im};
    
    for train_im = 1 : length(ImagesDB_train)
        cur_train_im = ImagesDB_train{train_im};
        total_dist = zeros(1,length(cur_train_im.Features));
        
        for f = 1 : length(cur_train_im.Features)
            loc_dist = cur_train_im.Features{f} - cur_test_im.Features{f};
            total_dist(f) = norm(loc_dist);
            if(f == 1)
                total_dist(f) = norm(loc_dist);
            elseif(f > 2 && f < 9) % chi-square for everything, but GIST
                total_dist(f) = 0.5 * sum(((loc_dist).^2) ./ (cur_train_im.Features{f} + cur_test_im.Features{f} + eps));
            else
                total_dist(f) = norm(loc_dist); %use L2 for GIST
            end
        end
        
        total_dist = total_dist * w(:,train_im);
        
        if(total_dist < max(dist))
            ins_idx = find(dist == max(dist));
            ins_idx = ins_idx(1);
            dist(ins_idx) = total_dist;
            kNN(ins_idx) = train_im;
        end        
    end
    
    pred_labels = zeros(1,21);
    w_loc = 1;
    dist = exp(-dist*w_loc) / sum(exp(-dist*w_loc));

    ImagesDB_test{tst_im}.kNN = kNN;

    for i = 1 : k
        cur_neib = ImagesDB_train{kNN(i)};
        for c = 1 : length(cur_neib.labels)
            if(cur_neib.labels(c) ~= 0)
                pred_labels(cur_neib.labels(c)) = pred_labels(cur_neib.labels(c)) + dist(i);
            end
        end
    end
    
    TestPrediction(tst_im,:) = pred_labels';
    TestLabels(tst_im, setdiff(cur_test_im.labels, 0)) = 1;

end

%%

load (Dataset.SpIndexFile);

Neibs = zeros(length(ImagesDB_test) + length(ImagesDB_train), k);
ILP_full =  zeros(TotalSP, 22);
for tst_im = 1 : length(ImagesDB_test) 
    Neibs(tst_im + length(ImagesDB_train), :) = ImagesDB_test{tst_im}.kNN;
    cur_offset = Images_spDB{ length(ImagesDB_train) + tst_im}.offset;
    cur_sp = Images_spDB{ length(ImagesDB_train) + tst_im}.SpNum;
    
    ILP_full(cur_offset+1 : cur_offset + cur_sp, 2:end) = repmat(TestPrediction(tst_im,:), cur_sp, 1);
end
ILP_full = ILP_full';
save( ilpOutputFile, 'ILP_full');
save( neibsOutputFile, 'Neibs');
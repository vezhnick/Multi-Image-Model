function Graph_Neibs = BuildSpatialNeibGraph(Dataset, outputFile, IsTrainingGraph, Verbose)

% function Graph = BuildSpatialNeibGraph(Dataset, K, L, outputFile, Verbose);
% 
% Adds edges for adjecent superpixels
%
% Input:
%     Dataset =  Dataset structure
%     K = maximum number of connections per image
%     L = total maximum number of connections
%     outputFile = file to which the graph will be written
%     Verbose = display or not different stats
% 
% Output: 
%    Graph = matrix of connections between superpixels
% Alexander Vezhnevets, 2012

load(Dataset.SpIndexFile);
load(Dataset.featuresFile);

Features = Features';

Graph_Neibs = sparse(TotalSP, TotalSP);

if IsTrainingGraph
    ImagesIdx = Dataset.TrainImageIdx;
else
    ImagesIdx = Dataset.TestImageIdx;
end


for image_i = ImagesIdx
            
    if Verbose
            disp(['image_i = ' num2str(image_i) ';']);       
    end
        
    features_i = full(Features(:, Images_spDB{image_i}.offset + 1 : Images_spDB{image_i}.offset + Images_spDB{image_i}.SpNum)); 
    norm = sum(features_i);
    features_i = features_i ./ repmat(norm, size(features_i,1), 1);
    
    im_j_neib = zeros(size(features_i,2), TotalSP) + 10000;
    
    
    [row col] = find(Images_spDB{image_i}.SpImage == 0);
    
    max_r = max(row);
    max_c = max(col);
    MMM = zeros(Images_spDB{image_i}.SpNum);
    
    for sp = 1: length(row)
        neibs = unique(Images_spDB{image_i}.SpImage ( max(1,row(sp)-1):min(max_r,row(sp)+1), max(1,col(sp)-1):min(max_c,col(sp)+1)));
        neibs = setdiff(neibs, 0);
        for p = 1 : length(neibs)
            for q = p+1 : length(neibs)
                MMM(neibs(p),neibs(q)) = 1;
            end
        end
    end
    
    [row col] = find(MMM);
    
    for sp = 1: length(row)
        loc_dist = features_i(:,row(sp)) - features_i(:,col(sp));
        MMM(row(sp),col(sp))  = 1- 0.5 * sum(((loc_dist).^2) ./ (features_i(:,row(sp)) + features_i(:,col(sp)) + eps));
    end
        
    MMM = max(MMM',MMM);
    
    Graph_Neibs(Images_spDB{image_i}.offset + 1 : Images_spDB{image_i}.offset + Images_spDB{image_i}.SpNum, ...
        Images_spDB{image_i}.offset + 1 : Images_spDB{image_i}.offset + Images_spDB{image_i}.SpNum) = MMM;
        
end

%%
if IsTrainingGraph
    train_idx = false(1, TotalSP);
    
    for i = ImagesIdx
        train_idx( Images_spDB{i}.offset + 1 : Images_spDB{i}.offset + Images_spDB{i}.SpNum) = true;
    end
    
    Graph_Neibs = Graph_Neibs(train_idx,train_idx);
end

save(outputFile, 'Graph_Neibs');


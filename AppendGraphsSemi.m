function Graph = AppendGraphsSemi(MIM, Dataset, K, L, outputFile, Verbose)

load(Dataset.SpIndexFile);
%load(PathsAndNames.ImageIndexFile);
load(MIM.ImageNeibsFile);
load(Dataset.featuresFile);
Features = Features';


Graph = sparse(TotalSP, TotalSP);

for image_i = Dataset.TestImageIdx
        
    features_i = full(Features(:, Images_spDB{image_i}.offset + 1 : Images_spDB{image_i}.offset + Images_spDB{image_i}.SpNum)); 
    norm = sum(features_i);
    features_i = features_i ./ repmat(norm, size(features_i,1), 1);
    
    im_j_neib = zeros(size(features_i,2), TotalSP) + 10000;
    
    for image_j = [Neibs(image_i,:) image_j]
        if Verbose
            disp(['image_i = ' num2str(image_i) '; image_j = ' num2str(image_j)]);
        end
%         if(image_i == image_j)
%             continue;
%         end
                
        features_j = full(Features(:, Images_spDB{image_j}.offset + 1 : Images_spDB{image_j}.offset + Images_spDB{image_j}.SpNum));
                
        norm = sum(features_j);
        features_j = features_j ./ repmat(norm, size(features_j,1), 1);
    
        for i = 1 : size(features_i,2)
            neibs = zeros(1, size(features_j,2));
            for j = 1  : size(features_j,2)
                loc_dist = features_i(:,i) - features_j(:,j);
                neibs(j) = 0.5 * sum(((loc_dist).^2) ./ (features_i(:,i) + features_j(:,j) + eps));
            end

            [val idx] = sort(neibs);
            im_j_neib(i, idx(1:K) + Images_spDB{image_j}.offset) = val(1:K);
            
        end
        
    end
    for i = 1 : size(features_i,2)
        [val idx] = sort(im_j_neib(i,:));
        Graph(i + Images_spDB{image_i}.offset, idx(1:L)) = 1 - val(1:L);
    end
    
end

Graph = max(Graph, Graph');
%%
train_idx = false(1, TotalSP);

for i = Dataset.TestImageIdx  
    train_idx( Images_spDB{i}.offset + 1 : Images_spDB{i}.offset + Images_spDB{i}.SpNum) = true;
end

save(outputFile, 'Graph', 'K', 'L');


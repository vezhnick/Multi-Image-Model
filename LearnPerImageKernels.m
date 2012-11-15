function LearnPerImageKernels(Dataset, outputFile)
% function LearnPerImageKernels(Dataset, outputFile);
% 
% Learns kernel combinations for images
%
% Input:
%     Dataset =  Dataset structure
%     outputFile = file to which the krenel weights are saved
% Alexander Vezhnevets, 2012

load(Dataset.ImageIndexFile);
ImagesDB_train = ImagesDB(Dataset.TrainImageIdx);

n = length(ImagesDB_train);
feat_len = length(ImagesDB_train{1}.Features);
c_w = zeros(n,21);
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

n_plus = 1 ./ total_c;
n_minus = 1 ./ (n - total_c);

dist = zeros(feat_len,n,n);
y = zeros(n,n);

bin_labels = zeros(n,21);

for im_idx = 1 : length(ImagesDB_train)
    
    im_e = ImagesDB_train{im_idx};
    
    for im_j_idx = 1 : length(ImagesDB_train)
        if(im_j_idx == im_idx)
            continue;
        end
        
        im_j = ImagesDB_train{im_j_idx};
        total_dist = zeros(1,length(im_j.Features));
        for f = 1 : length(im_j.Features)
            loc_dist = im_e.Features{f} - im_j.Features{f};
            total_dist(f) = norm(loc_dist);
            if(f == 1)
                total_dist(f) = norm(loc_dist);
            elseif(f > 2 && f < 9)
                total_dist(f) = 0.5 * sum(((loc_dist).^2) ./ (im_e.Features{f} + im_j.Features{f} + eps));
                %total_dist(f) = norm(loc_dist);
            else
                total_dist(f) = norm(loc_dist);
                %total_dist(f) = 0.5 * sum(((loc_dist).^2) ./ (im_e.Features{f} + im_j.Features{f} + eps));
            end
        end
        
        dist(:, im_idx, im_j_idx) = total_dist;
        
        common = intersect(im_e.labels, im_j.labels);
        common = setdiff(common, 0);
        
        y(im_idx, im_j_idx) = length(common) / length(setdiff(im_e.labels, 0));     
        
        for c = 1 : size(c_w,2)
            
            if(ismember(c, im_e.labels))
                c_w(im_idx, c) = n_plus(c);
            else
                c_w(im_idx, c) = n_minus(c);
            end
        end
        
        bin_labels(im_idx, setdiff(im_e.labels, 0)) = 1;

    end
end

%%

bin_labels( bin_labels == 0) = eps;
bin_labels( bin_labels == 1) = 1 - eps;

w = 5*(ones(feat_len,n) + rand(feat_len,n) / 5);

W = sum(c_w');

pi = zeros(size(dist,2), size(dist,3));
ro = zeros(size(dist,2), size(dist,3));
    
for iter = 1 : 100
   
    pi_old = pi;
  
    prediction = zeros(n, 21);
    
    for i = 1 : size(pi,1)
        %norm = 0;
        
        for j = 1 : size(pi,2)
            if i ~= j
                pi(i,j) = exp(-dist(:,i,j)' * w(:,j));
                %ro(i,j) = c_w(i,:) / W(i) * bin_labels(i,:)';
            end
        end      

        nor =  sum(pi(i,:));
        pi(i,:) = pi(i,:) / nor;
        for c = 1 : 21
            prediction(i,c) = pi(i,:) * bin_labels(:,c);
        end
        
        for j = 1 : size(pi,2)
            if i ~= j
                p_jw = bin_labels(j,:) ./ prediction(i,:) / n;
                ro(i,j) = c_w(i,:) / W(i) * p_jw';
            end
        end

    end
    grad = 0 * w;
    
    L = 0;
    
    for i = 1 : size(prediction,1)
        for l = 1 : 21
            if(bin_labels(i,l) > 0.5)
                L = L + c_w(i,l) * log( prediction(i,l));
            else
                L = L + c_w(i,l) * log( 1 - prediction(i,l));
            end
        end
    end
    Grad = pi - ro;
    %mean(mean(pi - pi_old))
    
    for l = 1 : feat_len
        Grad_l = Grad .* reshape(dist(l,:,:), n, n);
        for i = 1 : n            
            grad(l, i) = W(i) * sum(Grad_l(:,i));
            %grad(l, i) = sum(W' .* Grad_l(:,i));
        end
    end
    
    grad;
    w;
    
    for i = 1 : n
        w(:,i) = w(:,i) + 0.01 * grad(:,i) / mean(abs(grad(:,i)));    
        
    end
%    w(w < 0) = 0;
%    w = w / sum(w);
end

save (outputFile, 'w'); 
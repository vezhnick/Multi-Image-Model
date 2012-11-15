function LearnAndInfer(MIM, Dataset, IterNum, outputFile, Verbose)
% function LearnAndInfer(MIM, Dataset, IterNum, outputFile, Verbose);
%
% Infers latent superpixel labels and learns appearance models parameters
%
% Input:
%     MIM = MIM structure
%     Dataset =  Dataset structure
%     IterNum = number of iterations
%     outputFile = file to which the superpixel labels and appearance
%     models parameters are saved
%     Verbose = display or not different stats
% Alexander Vezhnevets, 2012

%% loading data
load(Dataset.labelsTrainFile); %used to measure error
load(Dataset.featuresTrainFile);
load(Dataset.objectnessTrainFile);
load(MIM.TrainGraphFile);
load(MIM.TrainNeibsGraph);
load(Dataset.SpIndexFile);

Graph = Graph;

ILP = zeros(size(Features,1),length(unique(Labels)));

for i = Dataset.TrainImageIdx
    curr_sp_idx = 1 + Images_spDB{i}.offset : Images_spDB{i}.offset + Images_spDB{i}.SpNum;
    ILP(curr_sp_idx, Images_spDB{i}.ImLabels+1) = 1;
end

%%
% initializing variables;
LeafPot = zeros(size(Features,2), size(ILP,2));
freq = zeros(1,size(ILP,2));
latent_y = zeros(size(Features,1),1);


ILP(1,:) = 0; % discard background from ILP

p_per_sp = full(sum(Features,2)) / 5; % pixels per superpixel

%sample random labels for superpixels from image labels (ILP)
for i = 1 : length(latent_y)
    cur_lbs = find(ILP(i,:) > 0) - 1; % image labels for a superpixel
    if(~isempty(cur_lbs))
        cur_lbs = randintrlv(cur_lbs, i); % take a random one
        latent_y(i) = cur_lbs(1);
        freq(cur_lbs(1)+1) = freq(cur_lbs(1)+1) + p_per_sp(i); % add a count to prior
    else
        latent_y(i) = ceil(max(Labels) * rand());
    end
end

NClasses = size(ILP, 2);
labelCost = single(ones(NClasses,NClasses));
labelCost = labelCost - eye(length(labelCost));
%%
for k = 1 : IterNum
    
    %UpdateUnariesA;
    %learning thetta
    WorkOutNaiveBunaries;
    NewPot = NewPot .* OP_mtx;
    log_Pot = -log( eps + NewPot);
    
    [junk init] = max(-log_Pot(:,2:end)');
    Gr_stat = full(median(sum(Graph)));
    alpha_opt = -median(junk - max(junk)) / Gr_stat;
    
    if(k == 1)
        latent_y = init';
        save([outputFile '_iter' num2str(0)], 'LeafPot', 'freq', 'latent_y', 'alpha_opt');        
    end
    
    %inferring y
    [latent_y E Eafter] = GCMex(init, single(full(log_Pot / alpha_opt)'), Graph, labelCost(1:end,1:end),1);
    
    if(Verbose)
        % accuracy measurements for debugging and control
        
        mass_norm = p_per_sp / sum(p_per_sp(Labels~=0));
        per_pix_acc = sum((latent_y(Labels ~= 0) == Labels(Labels ~= 0)) .* mass_norm(Labels ~= 0));
        per_node_acc = mean(latent_y(Labels ~= 0) == Labels(Labels ~= 0));
        
        cm = zeros(NClasses,NClasses);
        
        for i = 1 : length(Labels)
            cm(latent_y(i)+1, Labels(i)+1) = cm(latent_y(i)+1, Labels(i)+1) + mass_norm(i);
        end
        
        for i = 1 : size(cm,1)
            if( sum(cm(:,i)) > 0)
                cm(:,i) = cm(:,i) / sum(cm(:,i));
            end
        end
        cm = cm(2:end,2:end); %discard BG class from the confusion matrix
        fprintf('Step %d, total accuracy = %f, average = %f, per node acc = %f \n', k, per_pix_acc, mean(diag(cm)), per_node_acc);
        save([outputFile '_iter' num2str(k)], 'LeafPot', 'freq', 'latent_y', 'alpha_opt');
        
    end
end

save(outputFile, 'LeafPot', 'freq', 'latent_y', 'alpha_opt');
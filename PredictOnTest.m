function PredictOnTest(MIM, Dataset, outputFile, Verbose)
% function PredictOnTest(MIM, Dataset, outputFile, Verbose);
% 
% Predicts labels for the test images superpixels and evaluates accuracy
%
% Input:
%     MIM = MIM structure
%     Dataset =  Dataset structure
%     outputFile = file to which the predictions and accuracy will be saved
%     Verbose = display or not different stats
% Alexander Vezhnevets, 2012

load(Dataset.labelsFile);
load(Dataset.featuresFile);
load(Dataset.objectnessFile);
load(MIM.TestGraph);
load(MIM.ilpFile);
load(MIM.TestNeibsGraph);
load(MIM.Parameters);

tr_end = length(latent_y);
Gr_tot = max(Graph', Graph);

p_per_sp = full(sum(Features,2)) / 5;

%%

for l = 1 : size(LeafPot,2)
    LeafPot(:,l) = Features(1:tr_end,:)' * (latent_y == l-1) + 10; %P(w|y)
end

tot_leaf = sum(LeafPot,1).* freq;
LeafPot = LeafPot ./ (repmat(tot_leaf, size(LeafPot,1),1));



%%

FullPot = Features * (LeafPot);
FullPot = FullPot .* ILP_full';
FullPot = FullPot .* OP_mtx;

FullPot(:,1) = 0;

NClasses = size(FullPot, 2);
labelCost = single(ones(NClasses,NClasses));
labelCost = labelCost - eye(length(labelCost));

FullPot(:,1) = 0;
log_Pot = -log( eps + FullPot);

[junk init] = max(-log_Pot(:,2:end)');

log_Pot = log_Pot / alpha_opt;

log_Pot(1:tr_end,:) = 100;
for n = 1:tr_end
        log_Pot(n, latent_y(n)+1) = 0;
end

[latent_y E Eafter] = GCMex(init, single(full(log_Pot)'), Gr_tot, labelCost(1:end,1:end),1);

%% Measure and print accuracy 
idx = Labels ~= 0;
idx(1:tr_end) = 0;
per_node = mean(latent_y(idx) == Labels(idx));

mass_norm = p_per_sp / sum(p_per_sp(idx));
per_pix_acc = sum((latent_y(idx) == Labels(idx)) .* mass_norm(idx));
per_node_acc = mean(latent_y(idx) == Labels(idx));

cm = zeros(NClasses,NClasses);

for i = tr_end : length(Labels)
    cm(latent_y(i)+1, Labels(i)+1) = cm(latent_y(i)+1, Labels(i)+1) + mass_norm(i);
end

for i = 1 : size(cm,1)
    if( sum(cm(:,i)) > 0)
        cm(:,i) = cm(:,i) / sum(cm(:,i));
    end
end
%%
cm = cm(2:end,2:end);
per_class_acc = mean(diag(cm));

if Verbose
    fprintf('total accuracy = %f, average = %f, per node acc = % f \n', per_pix_acc, per_class_acc, per_node_acc);
end

save(outputFile, 'Labels', 'idx', 'cm', 'per_pix_acc', 'per_class_acc', 'per_node_acc');
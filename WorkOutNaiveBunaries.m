%% subscript for unary potential calculation
% Alexander Vezhnevets, 2012
for l = 1 : size(LeafPot,2)
    LeafPot(:,l) = Features' * (latent_y == l-1) + 10; %P(w|y)
end

tot_leaf = sum(LeafPot,1).* freq;
LeafPot = LeafPot ./ (repmat(tot_leaf, size(LeafPot,1),1));
NewPot = Features * (LeafPot);
NewPot = NewPot .* ILP;
NewPot(:,1) = 0;
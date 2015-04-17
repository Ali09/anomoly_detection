M = csvread('ahidden_train.txt');
Images = csvread('train_images.txt');
M_t = M';

D = squareform(pdist(M_t));

N = 30;
[row_max, col_max] = size(M_t);
dist_vect = zeros(row_max,1);

for row=1:row_max
    x = D(row,:);
    [sortedX,~] = sort(x,'ascend');
    maxValues = sortedX(1:N);
    dist_vect(row) = sum(maxValues)/N;
end

[sortedX,sortingIndices] = sort(dist_vect,'descend');

top_20 = sortingIndices(1:20);
%display_network(Images(:,top_20));

sub_fig = figure;
for i = 1:20
   subplot(4,5,i) 
   imshow(reshape(Images(:,top_20(i)),28,28,1));
   title({['Image ', num2str(a(i))];['Dist=' num2str(sprintf('%.2f',sortedX(i)))]});
end    
saveas(sub_fig, 'top_dist.png')

%Iforest
a = csvread('depth_val_ind.txt');
depth = csvread('depth_val.txt');
a=a'+1;

sub_fig = figure;
for i = 1:20
   subplot(4,5,i) 
   imshow(reshape(Images(:,a(i)),28,28,1));
   title({['Image ', num2str(a(i))];['Depth=' num2str(sprintf('%.1f',depth(a(i))))]});
end    
saveas(sub_fig, 'iforest_error.png')